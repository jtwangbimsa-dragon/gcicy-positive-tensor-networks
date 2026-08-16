#!/usr/bin/env python3
"""Audit dense-H versus factorized metric evaluation across numeric precisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, standard_errors  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from scripts.train_gcicy_pipeline import load_specification, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=72501)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def factorized_metrics_numpy(
    values: np.ndarray,
    derivatives: np.ndarray,
    factor: np.ndarray,
    normalization: float,
) -> np.ndarray:
    transformed_values = values @ np.conjugate(factor)
    transformed_derivatives = np.einsum(
        "mc,nmj->ncj",
        np.conjugate(factor),
        derivatives,
        optimize=True,
    )
    denominator = np.real(
        np.einsum(
            "nc,nc->n",
            np.conjugate(transformed_values),
            transformed_values,
            optimize=True,
        )
    )
    first = np.einsum(
        "nci,ncj->nij",
        np.conjugate(transformed_derivatives),
        transformed_derivatives,
        optimize=True,
    )
    gradient = np.einsum(
        "nc,ncj->nj",
        np.conjugate(transformed_values),
        transformed_derivatives,
        optimize=True,
    )
    metrics = first / denominator[:, None, None]
    metrics -= (
        np.conjugate(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metrics *= normalization
    return 0.5 * (metrics + np.conjugate(np.swapaxes(metrics, 1, 2)))


def dense_metrics_numpy(
    values: np.ndarray,
    derivatives: np.ndarray,
    h_matrix: np.ndarray,
    normalization: float,
) -> np.ndarray:
    h_values = np.einsum("ab,nb->na", h_matrix, values, optimize=True)
    denominator = np.real(
        np.einsum("na,na->n", np.conjugate(values), h_values, optimize=True)
    )
    h_derivatives = np.einsum(
        "ab,nbj->naj",
        h_matrix,
        derivatives,
        optimize=True,
    )
    first = np.einsum(
        "nmi,nmj->nij",
        np.conjugate(derivatives),
        h_derivatives,
        optimize=True,
    )
    gradient = np.einsum(
        "nm,nmj->nj",
        np.conjugate(values),
        h_derivatives,
        optimize=True,
    )
    metrics = first / denominator[:, None, None]
    metrics -= (
        np.conjugate(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metrics *= normalization
    return 0.5 * (metrics + np.conjugate(np.swapaxes(metrics, 1, 2)))


def torch_metrics_and_raw(
    values: np.ndarray,
    derivatives: np.ndarray,
    log_omega: np.ndarray,
    factor: np.ndarray,
    normalization: float,
    *,
    device: str,
    factorized: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    values_t = torch.tensor(values, dtype=torch.complex64, device=device)
    derivatives_t = torch.tensor(derivatives, dtype=torch.complex64, device=device)
    log_omega_t = torch.tensor(log_omega, dtype=torch.float32, device=device)
    factor_t = torch.tensor(factor, dtype=torch.complex64, device=device)
    if factorized:
        transformed_values = torch.einsum("mc,nm->nc", torch.conj(factor_t), values_t)
        transformed_derivatives = torch.einsum(
            "mc,nmj->ncj", torch.conj(factor_t), derivatives_t
        )
        denominator = torch.real(
            torch.einsum("nc,nc->n", torch.conj(transformed_values), transformed_values)
        )
        first = torch.einsum(
            "nci,ncj->nij",
            torch.conj(transformed_derivatives),
            transformed_derivatives,
        )
        gradient = torch.einsum(
            "nc,ncj->nj",
            torch.conj(transformed_values),
            transformed_derivatives,
        )
    else:
        h_matrix = factor_t @ torch.conj(factor_t.T)
        h_values = torch.einsum("ab,nb->na", h_matrix, values_t)
        denominator = torch.real(
            torch.einsum("na,na->n", torch.conj(values_t), h_values)
        )
        h_derivatives = torch.einsum("ab,nbj->naj", h_matrix, derivatives_t)
        first = torch.einsum("nmi,nmj->nij", torch.conj(derivatives_t), h_derivatives)
        gradient = torch.einsum("nm,nmj->nj", torch.conj(values_t), h_derivatives)
    metrics = first / denominator[:, None, None]
    metrics -= (
        torch.conj(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metrics = (
        normalization * 0.5 * (metrics + torch.conj(torch.transpose(metrics, 1, 2)))
    )
    eigenvalues = torch.linalg.eigvalsh(metrics)
    raw = torch.log(torch.clamp(eigenvalues, min=1.0e-12)).sum(dim=1) - log_omega_t
    result = (
        np.asarray(metrics.detach().cpu().numpy(), dtype=np.complex128),
        np.asarray(eigenvalues.detach().cpu().numpy(), dtype=np.float64),
        np.asarray(raw.detach().cpu().numpy(), dtype=np.float64),
    )
    del values_t, derivatives_t, factor_t, metrics, eigenvalues, raw
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def raw_from_metrics(
    metrics: np.ndarray, log_omega: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues = np.linalg.eigvalsh(metrics)
    raw = np.log(np.maximum(eigenvalues, 1.0e-12)).sum(axis=1) - log_omega
    return eigenvalues, raw


def quantiles(values: np.ndarray) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    return {
        "q50": float(np.quantile(samples, 0.5)),
        "q90": float(np.quantile(samples, 0.9)),
        "q99": float(np.quantile(samples, 0.99)),
        "q999": float(np.quantile(samples, 0.999)),
        "maximum": float(np.max(samples)),
    }


def method_summary(
    raw: np.ndarray,
    eigenvalues: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    minimum_eigenvalues = np.min(eigenvalues, axis=1)
    return {
        "finite_raw_count": int(np.sum(np.isfinite(raw))),
        "nonpositive_metric_count": int(np.sum(minimum_eigenvalues <= 0)),
        "minimum_metric_eigenvalue": float(np.min(minimum_eigenvalues)),
        "minimum_metric_eigenvalue_quantiles": quantiles(minimum_eigenvalues),
        "raw_quantiles": quantiles(raw),
        "standard_errors": standard_errors(raw, weights),
    }


def comparison_summary(
    reference_metrics: np.ndarray,
    reference_raw: np.ndarray,
    candidate_metrics: np.ndarray,
    candidate_raw: np.ndarray,
) -> dict[str, object]:
    metric_difference = np.linalg.norm(
        candidate_metrics - reference_metrics,
        axis=(1, 2),
    ) / np.maximum(np.linalg.norm(reference_metrics, axis=(1, 2)), 1.0e-300)
    return {
        "relative_metric_frobenius_error": quantiles(metric_difference),
        "absolute_raw_error": quantiles(np.abs(candidate_raw - reference_raw)),
    }


def main() -> None:
    args = parse_args()
    if args.points <= 0 or args.seed < 0:
        raise ValueError("points must be positive and seed cannot be negative")
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    spec_path = args.spec.expanduser().resolve()
    data = load_specification(spec_path)
    adapter = get_adapter(str(data["adapter"]))
    model_data = data["model"]
    model_seed = int(model_data["seed"])
    exact_model = bool(model_data.get("exact", True))
    model = adapter.make_model(model_seed, exact=exact_model)
    training = data["training"]
    artifact_path = (
        args.artifact.expanduser().resolve()
        if args.artifact is not None
        else resolve_path(training["initial_artifact"], spec_path.parent)
    )
    artifact = adapter.load_h_artifact(artifact_path, model)
    workers = int(training.get("sampling_workers", 1))
    cluster_size = int(training.get("sampling_cluster_size", 1))
    backend = str(training.get("sampling_backend", "process"))

    started = time.perf_counter()
    if workers > 1:
        points, shards = sample_points_parallel(
            adapter,
            model_seed=model_seed,
            exact_model=exact_model,
            count=args.points,
            seed=args.seed,
            workers=workers,
            cluster_size=cluster_size,
            backend=backend,
        )
    else:
        points = adapter.sample_points(model, args.points, seed=args.seed)
        shards = []
    evaluated = [
        adapter.section_values_and_jacobian(point, artifact.section_exponents)
        for point in points
    ]
    values = np.asarray([item[0] for item in evaluated], dtype=np.complex128)
    derivatives = np.asarray([item[1] for item in evaluated], dtype=np.complex128)
    log_omega = np.asarray(
        [adapter.holomorphic_volume_log_density(point) for point in points],
        dtype=np.float64,
    )
    weights = np.asarray(adapter.importance_weights(points), dtype=np.float64)

    h_matrix = np.asarray(artifact.h_matrix, dtype=np.complex128)
    h_matrix = 0.5 * (h_matrix + np.conjugate(h_matrix.T))
    factor = np.linalg.cholesky(h_matrix)
    dense128_metrics = dense_metrics_numpy(
        values, derivatives, h_matrix, artifact.normalization
    )
    factor128_metrics = factorized_metrics_numpy(
        values, derivatives, factor, artifact.normalization
    )
    dense128_eigenvalues, dense128_raw = raw_from_metrics(dense128_metrics, log_omega)
    factor128_eigenvalues, factor128_raw = raw_from_metrics(
        factor128_metrics, log_omega
    )
    dense64_metrics, dense64_eigenvalues, dense64_raw = torch_metrics_and_raw(
        values,
        derivatives,
        log_omega,
        factor,
        artifact.normalization,
        device=args.device,
        factorized=False,
    )
    factor64_metrics, factor64_eigenvalues, factor64_raw = torch_metrics_and_raw(
        values,
        derivatives,
        log_omega,
        factor,
        artifact.normalization,
        device=args.device,
        factorized=True,
    )

    factor64 = factor.astype(np.complex64)
    reconstructed64 = np.asarray(
        factor64 @ np.conjugate(factor64.T), dtype=np.complex128
    )
    reconstructed64 = 0.5 * (reconstructed64 + np.conjugate(reconstructed64.T))
    h_eigenvalues = np.linalg.eigvalsh(h_matrix)
    reconstructed_eigenvalues = np.linalg.eigvalsh(reconstructed64)
    raw_disagreement = np.abs(dense64_raw - factor64_raw)
    top_indices = np.argsort(raw_disagreement, kind="stable")[-20:][::-1]
    top_rows = [
        {
            "local_index": int(index),
            "projective_chart": list(adapter.point_projective_chart(points[index])),
            "jacobian_minimum_singular_value": float(
                adapter.point_jacobian_min_singular_value(points[index])
            ),
            "dense64_raw": float(dense64_raw[index]),
            "factor64_raw": float(factor64_raw[index]),
            "factor128_raw": float(factor128_raw[index]),
            "dense64_factor64_absolute_difference": float(raw_disagreement[index]),
        }
        for index in top_indices
    ]
    summary = {
        "schema_version": 1,
        "source_specification": str(spec_path),
        "source_artifact": str(artifact_path),
        "adapter": adapter.key,
        "model_seed": model_seed,
        "exact_model": exact_model,
        "sample": {
            "seed": args.seed,
            "point_count": len(points),
            "sampling_workers": workers,
            "sampling_cluster_size": cluster_size,
            "sampling_backend": backend,
            "sampling_shards": shards,
        },
        "device": args.device,
        "h_matrix_precision": {
            "complex128_minimum_eigenvalue": float(h_eigenvalues[0]),
            "complex128_maximum_eigenvalue": float(h_eigenvalues[-1]),
            "complex128_condition_number": float(h_eigenvalues[-1] / h_eigenvalues[0]),
            "complex64_factor_reconstruction_minimum_eigenvalue": float(
                reconstructed_eigenvalues[0]
            ),
            "complex64_factor_reconstruction_nonpositive_eigenvalue_count": int(
                np.sum(reconstructed_eigenvalues <= 0)
            ),
            "complex64_factor_reconstruction_relative_frobenius_error": float(
                np.linalg.norm(reconstructed64 - h_matrix) / np.linalg.norm(h_matrix)
            ),
        },
        "methods": {
            "dense_complex128": method_summary(
                dense128_raw, dense128_eigenvalues, weights
            ),
            "factorized_complex128": method_summary(
                factor128_raw, factor128_eigenvalues, weights
            ),
            "dense_torch_complex64": method_summary(
                dense64_raw, dense64_eigenvalues, weights
            ),
            "factorized_torch_complex64": method_summary(
                factor64_raw, factor64_eigenvalues, weights
            ),
        },
        "comparisons_against_factorized_complex128": {
            "dense_complex128": comparison_summary(
                factor128_metrics,
                factor128_raw,
                dense128_metrics,
                dense128_raw,
            ),
            "dense_torch_complex64": comparison_summary(
                factor128_metrics,
                factor128_raw,
                dense64_metrics,
                dense64_raw,
            ),
            "factorized_torch_complex64": comparison_summary(
                factor128_metrics,
                factor128_raw,
                factor64_metrics,
                factor64_raw,
            ),
        },
        "dense64_vs_factorized64": comparison_summary(
            factor64_metrics,
            factor64_raw,
            dense64_metrics,
            dense64_raw,
        ),
        "largest_dense64_factorized64_raw_disagreements": top_rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
