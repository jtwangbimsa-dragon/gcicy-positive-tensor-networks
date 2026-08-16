#!/usr/bin/env python3
"""Measure whether a two-factor product metric collapses to one mean H."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigvalsh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.adapter import positive_hermitian_projection  # noqa: E402
from gcicy_metric.pipeline.audit import standard_errors  # noqa: E402
from gcicy_metric.pipeline.factorized_h import factorized_h_metrics  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from gcicy_metric.pipeline.registry import get_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--exact-model", action="store_true")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--factorized", type=Path, required=True)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=72003)
    parser.add_argument("--sampling-workers", type=int, default=1)
    parser.add_argument("--sampling-cluster-size", type=int, default=4)
    parser.add_argument(
        "--sampling-backend", choices=("process", "thread"), default="process"
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def trace_normalize(matrix: np.ndarray) -> np.ndarray:
    candidate = positive_hermitian_projection(matrix)
    return candidate * (len(candidate) / float(np.trace(candidate).real))


def h_diagnostics(matrix: np.ndarray) -> dict[str, float]:
    eigenvalues = np.linalg.eigvalsh(trace_normalize(matrix))
    return {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "log_eigenvalue_span": float(np.log(eigenvalues[-1] / eigenvalues[0])),
    }


def relative_spd_diagnostics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    eigenvalues = eigvalsh(trace_normalize(candidate), trace_normalize(reference))
    if float(eigenvalues[0]) <= 0:
        raise FloatingPointError("relative H spectrum is not positive")
    logs = np.log(eigenvalues)
    centered = logs - np.mean(logs)
    return {
        "centered_log_eigenvalue_span": float(np.ptp(centered)),
        "maximum_abs_centered_log_eigenvalue": float(np.max(np.abs(centered))),
        "centered_log_eigenvalue_rms": float(np.sqrt(np.mean(centered**2))),
    }


def metric_stats(
    adapter,
    points,
    metrics: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    residuals = adapter.residual_values(points, metrics)
    stats = standard_errors(residuals, weights)
    stats["min_metric_eigenvalue"] = float(
        np.min(np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128)))
    )
    return stats


def relative_metric_errors(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    errors = np.linalg.norm(candidate - reference, axis=(1, 2)) / np.maximum(
        np.finfo(float).tiny,
        np.linalg.norm(reference, axis=(1, 2)),
    )
    return {
        "mean": float(np.mean(errors)),
        "rms": float(np.sqrt(np.mean(errors**2))),
        "q99": float(np.quantile(errors, 0.99)),
        "q999": float(np.quantile(errors, 0.999)),
        "maximum": float(np.max(errors)),
    }


def main() -> None:
    args = parse_args()
    if (
        min(
            args.points,
            args.sampling_workers,
            args.sampling_cluster_size,
        )
        <= 0
    ):
        raise SystemExit("point, worker, and cluster counts must be positive")
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    source = adapter.load_h_artifact(args.source.expanduser().resolve(), model)
    exported = adapter.load_h_artifact(args.factorized.expanduser().resolve(), model)
    with np.load(args.factorized.expanduser().resolve(), allow_pickle=False) as payload:
        required = {
            "factorized_source_degree",
            "factorized_source_exponents",
            "factorized_source_normalization",
            "factorized_h_matrices",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"factorized artifact is missing {sorted(missing)}")
        factor_degree = tuple(
            int(value) for value in payload["factorized_source_degree"]
        )
        factor_exponents = np.asarray(
            payload["factorized_source_exponents"], dtype=np.int64
        )
        factor_normalization = float(payload["factorized_source_normalization"])
        factors = np.asarray(payload["factorized_h_matrices"], dtype=np.complex128)
    if factor_degree != source.degree:
        raise ValueError("factorized source degree does not match source artifact")
    if not np.array_equal(factor_exponents, source.section_exponents):
        raise ValueError("factorized source basis does not match source artifact")
    if factors.shape != (2, source.section_count, source.section_count):
        raise ValueError("expected exactly two source-level H factors")

    if args.sampling_workers > 1:
        points, shards = sample_points_parallel(
            adapter,
            model_seed=args.model_seed,
            exact_model=args.exact_model,
            count=args.points,
            seed=args.seed,
            workers=args.sampling_workers,
            cluster_size=args.sampling_cluster_size,
            backend=args.sampling_backend,
        )
    else:
        points = adapter.sample_points(model, args.points, seed=args.seed)
        shards = []
    weights = adapter.importance_weights(points)
    factors = np.asarray([trace_normalize(matrix) for matrix in factors])
    mean_h = trace_normalize(np.mean(factors, axis=0))
    factorized_metrics = factorized_h_metrics(
        adapter,
        points,
        source_exponents=factor_exponents,
        h_matrices=factors,
        source_normalization=factor_normalization,
    )
    mean_h_metrics = factorized_h_metrics(
        adapter,
        points,
        source_exponents=factor_exponents,
        h_matrices=np.stack([mean_h, mean_h]),
        source_normalization=factor_normalization,
    )
    source_metrics = adapter.h_metrics(points, source)
    exported_metrics = adapter.h_metrics(points, exported)

    factor_frobenius_distance = float(
        np.linalg.norm(factors[0] - factors[1])
        / max(np.finfo(float).tiny, np.linalg.norm(mean_h))
    )
    result = {
        "schema_version": 1,
        "adapter": adapter.key,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "source": str(args.source.expanduser().resolve()),
        "factorized": str(args.factorized.expanduser().resolve()),
        "sample": {
            "seed": args.seed,
            "points": args.points,
            "sampling_workers": args.sampling_workers,
            "sampling_cluster_size": args.sampling_cluster_size,
            "sampling_backend": args.sampling_backend,
            "shards": shards,
        },
        "h_matrices": {
            "source": h_diagnostics(source.h_matrix),
            "factor_a": h_diagnostics(factors[0]),
            "factor_b": h_diagnostics(factors[1]),
            "mean_factor": h_diagnostics(mean_h),
            "exported_target": h_diagnostics(exported.h_matrix),
            "factor_relative_frobenius_distance": factor_frobenius_distance,
            "factor_a_relative_to_source": relative_spd_diagnostics(
                factors[0], source.h_matrix
            ),
            "factor_b_relative_to_source": relative_spd_diagnostics(
                factors[1], source.h_matrix
            ),
            "factor_a_relative_to_factor_b": relative_spd_diagnostics(
                factors[0], factors[1]
            ),
        },
        "metric_statistics": {
            "source": metric_stats(adapter, points, source_metrics, weights),
            "single_mean_h": metric_stats(adapter, points, mean_h_metrics, weights),
            "two_factor": metric_stats(adapter, points, factorized_metrics, weights),
            "exported_target": metric_stats(adapter, points, exported_metrics, weights),
        },
        "metric_relative_errors": {
            "two_factor_vs_single_mean_h": relative_metric_errors(
                factorized_metrics, mean_h_metrics
            ),
            "two_factor_vs_source": relative_metric_errors(
                factorized_metrics, source_metrics
            ),
            "exported_target_vs_two_factor": relative_metric_errors(
                exported_metrics, factorized_metrics
            ),
        },
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["h_matrices"], indent=2, sort_keys=True))
    print(json.dumps(result["metric_relative_errors"], indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
