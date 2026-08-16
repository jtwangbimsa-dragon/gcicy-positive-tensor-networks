#!/usr/bin/env python3
"""Test whether type-(1,1) H-metric tails track eigensection switching."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.audit import (  # noqa: E402
    normalized_volume_ratios,
    standard_errors,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=72003)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--basis-seed", type=int, default=70499)
    parser.add_argument("--basis-points", type=int, default=4096)
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrays-out", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    observed = np.quantile(np.asarray(values, dtype=np.float64), probabilities)
    return {
        label: float(value)
        for label, value in zip(
            ("minimum", "q10", "median", "q90", "q99", "maximum"),
            observed,
            strict=True,
        )
    }


def summarize_group(
    mask: np.ndarray,
    *,
    weights: np.ndarray,
    ratio: np.ndarray,
    top_two_gap: np.ndarray,
    effective_sections: np.ndarray,
    leading_mass: np.ndarray,
    metric_eigenvalues: np.ndarray,
) -> dict[str, Any] | None:
    if not np.any(mask):
        return None
    selected_metric = metric_eigenvalues[mask]
    return {
        "point_count": int(np.sum(mask)),
        "target_weight_mass": float(np.sum(weights[mask]) / np.sum(weights)),
        "normalized_volume_ratio": quantiles(ratio[mask]),
        "top_two_logit_gap": quantiles(top_two_gap[mask]),
        "effective_eigensection_count": quantiles(effective_sections[mask]),
        "leading_eigensection_mass": quantiles(leading_mass[mask]),
        "metric_relative_eigenvalue_1": quantiles(selected_metric[:, 0]),
        "metric_relative_eigenvalue_2": quantiles(selected_metric[:, 1]),
        "metric_relative_eigenvalue_3": quantiles(selected_metric[:, 2]),
        "metric_relative_condition": quantiles(
            selected_metric[:, 2] / selected_metric[:, 0]
        ),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right)
    return float(result.statistic)


def evaluate_metrics(
    values: np.ndarray,
    derivatives: np.ndarray,
    h_matrix: np.ndarray,
    normalization: float,
) -> np.ndarray:
    h_values = np.einsum("ab,nb->na", h_matrix, values, optimize=True)
    denominator = np.real(
        np.einsum("na,na->n", np.conjugate(values), h_values, optimize=True)
    )
    if np.any(denominator <= 0):
        raise FloatingPointError("H-metric denominator is not positive")
    h_derivatives = np.einsum(
        "ab,nbj->naj", h_matrix, derivatives, optimize=True
    )
    first = np.einsum(
        "nmi,nmj->nij",
        np.conjugate(derivatives),
        h_derivatives,
        optimize=True,
    )
    gradient = np.einsum(
        "nm,nmj->nj", np.conjugate(values), h_derivatives, optimize=True
    )
    metric = first / denominator[:, None, None]
    metric -= (
        np.conjugate(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metric *= normalization
    return 0.5 * (metric + np.conjugate(np.swapaxes(metric, 1, 2)))


def main() -> None:
    args = parse_args()
    if args.points <= 0 or args.points % 4:
        raise SystemExit("--points must be a positive multiple of four")
    if args.basis_points <= 0 or args.basis_points % 4 or args.workers <= 0:
        raise SystemExit("basis points must be a positive multiple of four")

    started = time.perf_counter()
    artifact_path = args.artifact.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    arrays_path = (
        args.arrays_out.expanduser().resolve()
        if args.arrays_out is not None
        else out_path.with_suffix(".npz")
    )
    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    model = adapter.make_model(args.model_seed, exact=True)
    artifact = adapter.load_h_artifact(artifact_path, model)

    with np.load(artifact_path, allow_pickle=False) as payload:
        if "basis_selected_indices" not in payload.files:
            raise ValueError("artifact does not record basis_selected_indices")
        selected_indices = np.asarray(payload["basis_selected_indices"], dtype=np.int64)

    print("reconstructing the saved section basis and FS reference H", flush=True)
    basis_points, basis_shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.basis_points,
        seed=args.basis_seed,
        workers=args.workers,
        cluster_size=4,
        backend="process",
    )
    recomputed_basis = adapter.restricted_section_basis(basis_points, artifact.degree)
    if (
        selected_indices.shape != (artifact.section_count,)
        or len(np.unique(selected_indices)) != artifact.section_count
        or np.min(selected_indices) < 0
        or np.max(selected_indices) >= len(recomputed_basis.ambient_exponents)
    ):
        raise ValueError("saved basis indices are invalid")
    if not np.array_equal(
        recomputed_basis.ambient_exponents[selected_indices], artifact.section_exponents
    ):
        raise ValueError("saved indices do not reproduce the artifact section basis")
    basis = replace(
        recomputed_basis,
        selected_indices=selected_indices,
        selected_exponents=artifact.section_exponents,
        numerical_rank=artifact.section_count,
    )
    reference_h, relation_error = adapter.restricted_fubini_study_h_matrix(
        basis_points, basis
    )
    reference_h = np.asarray(reference_h, dtype=np.complex128)
    reference_h = 0.5 * (reference_h + reference_h.conjugate().T)
    reference_h *= len(reference_h) / float(np.trace(reference_h).real)

    print(f"sampling {args.points} blind points with seed {args.seed}", flush=True)
    points, point_shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.points,
        seed=args.seed,
        workers=args.workers,
        cluster_size=4,
        backend="process",
    )
    evaluated = [
        adapter.section_values_and_jacobian(point, artifact.section_exponents)
        for point in points
    ]
    values = np.asarray([row[0] for row in evaluated], dtype=np.complex128)
    derivatives = np.asarray([row[1] for row in evaluated], dtype=np.complex128)
    weights = np.asarray(adapter.importance_weights(points), dtype=np.float64)

    h_matrix = np.asarray(artifact.h_matrix, dtype=np.complex128)
    h_matrix = 0.5 * (h_matrix + h_matrix.conjugate().T)
    reference_cholesky = np.linalg.cholesky(reference_h)
    left_solved = np.linalg.solve(reference_cholesky, h_matrix)
    relative_h = np.linalg.solve(reference_cholesky.conjugate(), left_solved.T).T
    relative_h = 0.5 * (relative_h + relative_h.conjugate().T)
    relative_eigenvalues, relative_eigenvectors = np.linalg.eigh(relative_h)
    if relative_eigenvalues[0] <= 0:
        raise FloatingPointError("reference-relative H is not positive definite")
    centered_relative_eigenvalues = relative_eigenvalues / np.exp(
        np.mean(np.log(relative_eigenvalues))
    )

    whitened_sections = values @ reference_cholesky.conjugate()
    eigensection_values = whitened_sections @ relative_eigenvectors.conjugate()
    spectral_terms = relative_eigenvalues[None, :] * np.abs(eigensection_values) ** 2
    spectral_denominator = np.sum(spectral_terms, axis=1)
    direct_denominator = np.real(
        np.einsum(
            "na,ab,nb->n",
            np.conjugate(values),
            h_matrix,
            values,
            optimize=True,
        )
    )
    denominator_relative_error = np.max(
        np.abs(spectral_denominator - direct_denominator)
        / np.maximum(np.abs(direct_denominator), 1.0e-300)
    )
    spectral_probabilities = spectral_terms / spectral_denominator[:, None]
    sorted_probabilities = np.sort(spectral_probabilities, axis=1)
    leading_mass = sorted_probabilities[:, -1]
    second_mass = sorted_probabilities[:, -2]
    effective_sections = 1.0 / np.sum(spectral_probabilities**2, axis=1)
    logits = np.log(relative_eigenvalues)[None, :] + np.log(
        np.maximum(np.abs(eigensection_values) ** 2, 1.0e-300)
    )
    top_two_logits = np.partition(logits, -2, axis=1)[:, -2:]
    top_two_gap = np.max(top_two_logits, axis=1) - np.min(top_two_logits, axis=1)
    dominant_index = np.argmax(spectral_probabilities, axis=1)

    print("evaluating MA ratios and metric generalized eigenvalues", flush=True)
    metrics = evaluate_metrics(
        values, derivatives, h_matrix, float(artifact.normalization)
    )
    raw = np.asarray(adapter.residual_values(points, metrics), dtype=np.float64)
    valid = np.isfinite(raw)
    if not np.all(valid):
        raise FloatingPointError("non-finite residual encountered in audit")
    ratio, log_ratio = normalized_volume_ratios(raw, weights)
    error_statistics = standard_errors(raw, weights)
    cluster_ids = np.arange(args.points, dtype=np.int64) // 4
    error_statistics["normalized_ratio_above_3_cluster_count"] = int(
        len(np.unique(cluster_ids[ratio > 3.0]))
    )
    baseline_metrics = adapter.baseline_metrics(points)
    metric_relative_eigenvalues = np.asarray(
        [
            eigvalsh(candidate, reference, check_finite=False)
            for candidate, reference in zip(metrics, baseline_metrics, strict=True)
        ],
        dtype=np.float64,
    )
    if np.any(metric_relative_eigenvalues <= 0):
        raise FloatingPointError("metric generalized eigenvalues are not positive")
    metric_relative_condition = (
        metric_relative_eigenvalues[:, 2] / metric_relative_eigenvalues[:, 0]
    )

    order = np.argsort(ratio, kind="stable")
    top_one_percent = np.zeros(args.points, dtype=bool)
    top_one_tenth_percent = np.zeros(args.points, dtype=bool)
    top_one_percent[order[-max(1, args.points // 100) :]] = True
    top_one_tenth_percent[order[-max(1, args.points // 1000) :]] = True
    groups = {
        "all": np.ones(args.points, dtype=bool),
        "bulk_bottom_90_percent_by_count": ratio <= np.quantile(ratio, 0.9),
        "top_1_percent_by_count": top_one_percent,
        "top_0_1_percent_by_count": top_one_tenth_percent,
        "ratio_above_3": ratio > 3.0,
        "ratio_above_10": ratio > 10.0,
    }
    group_rows = {
        name: summarize_group(
            mask,
            weights=weights,
            ratio=ratio,
            top_two_gap=top_two_gap,
            effective_sections=effective_sections,
            leading_mass=leading_mass,
            metric_eigenvalues=metric_relative_eigenvalues,
        )
        for name, mask in groups.items()
    }

    worst = int(np.argmax(ratio))
    correlations = {
        "log_ratio_vs_top_two_logit_gap": correlation(log_ratio, top_two_gap),
        "log_ratio_vs_effective_eigensection_count": correlation(
            log_ratio, effective_sections
        ),
        "log_ratio_vs_leading_eigensection_mass": correlation(log_ratio, leading_mass),
        "log_ratio_vs_metric_relative_eigenvalue_1": correlation(
            log_ratio, metric_relative_eigenvalues[:, 0]
        ),
        "log_ratio_vs_metric_relative_eigenvalue_2": correlation(
            log_ratio, metric_relative_eigenvalues[:, 1]
        ),
        "log_ratio_vs_metric_relative_eigenvalue_3": correlation(
            log_ratio, metric_relative_eigenvalues[:, 2]
        ),
        "log_ratio_vs_metric_relative_condition": correlation(
            log_ratio, metric_relative_condition
        ),
    }
    report = {
        "schema": "type11-eigensection-switching-audit-v1",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "model_seed": args.model_seed,
        "basis_seed": args.basis_seed,
        "basis_points": args.basis_points,
        "basis_shards": basis_shards,
        "basis_relation_error": float(relation_error),
        "seed": args.seed,
        "point_count": args.points,
        "point_shards": point_shards,
        "section_count": artifact.section_count,
        "reference_relative_h_spectrum": {
            "minimum": float(centered_relative_eigenvalues[0]),
            "maximum": float(centered_relative_eigenvalues[-1]),
            "condition_number": float(
                centered_relative_eigenvalues[-1] / centered_relative_eigenvalues[0]
            ),
            "log_eigenvalue_span": float(
                np.log(centered_relative_eigenvalues[-1])
                - np.log(centered_relative_eigenvalues[0])
            ),
        },
        "spectral_denominator_maximum_relative_error": float(
            denominator_relative_error
        ),
        "standard_errors": error_statistics,
        "correlations": correlations,
        "groups": group_rows,
        "worst_point": {
            "index": worst,
            "cluster_start": 4 * (worst // 4),
            "normalized_volume_ratio": float(ratio[worst]),
            "log_normalized_volume_ratio": float(log_ratio[worst]),
            "importance_weight": float(weights[worst]),
            "top_two_logit_gap": float(top_two_gap[worst]),
            "effective_eigensection_count": float(effective_sections[worst]),
            "leading_eigensection_mass": float(leading_mass[worst]),
            "second_eigensection_mass": float(second_mass[worst]),
            "dominant_eigensection_index": int(dominant_index[worst]),
            "metric_relative_eigenvalues": [
                float(value) for value in metric_relative_eigenvalues[worst]
            ],
            "metric_relative_condition": float(metric_relative_condition[worst]),
        },
        "arrays": str(arrays_path),
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": {
            "switching_prediction": (
                "If dominance switching causes the positive-ratio ridge, high-ratio "
                "groups should have materially smaller top-two gaps than the bulk."
            ),
            "anisotropy_prediction": (
                "If one metric direction creates the determinant spike, high-ratio "
                "groups should mainly increase relative eigenvalue 3."
            ),
            "scope_limit": (
                "This is a basis-covariant finite blind-sample association test, not "
                "a proof of causation or a global supremum certificate."
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        importance_weights=weights,
        sampling_cluster_ids=cluster_ids,
        normalized_volume_ratio=ratio,
        log_normalized_volume_ratio=log_ratio,
        top_two_logit_gap=top_two_gap,
        effective_eigensection_count=effective_sections,
        leading_eigensection_mass=leading_mass,
        second_eigensection_mass=second_mass,
        dominant_eigensection_index=dominant_index,
        metric_relative_eigenvalues=metric_relative_eigenvalues,
        metric_relative_condition=metric_relative_condition,
    )
    report["arrays_sha256"] = sha256_file(arrays_path)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "reference_relative_h_spectrum": report[
                    "reference_relative_h_spectrum"
                ],
                "correlations": correlations,
                "top_1_percent": group_rows["top_1_percent_by_count"],
                "worst_point": report["worst_point"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
