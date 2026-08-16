#!/usr/bin/env python3
"""Stress-test an accepted gCICY H metric on geometry-conditioned hard regions."""

from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    generic_global_h_metric,
    generic_global_h_metrics,
    generic_importance_log_weight,
    generic_importance_weights,
    generic_monge_ampere_log_error,
    generic_point_in_chart,
    generic_residual_values,
    make_exact_generic_model,
    sample_generic_gcicy_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "outputs" / "gcicy_generic_global_h_metric_k3_rank64_whitened_gpu.npz",
    )
    parser.add_argument("--seeds", default="10101,10102,10103,10104,10105,10106,10107,10108")
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--projector-points-per-seed", type=int, default=2)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "gcicy_k3_hard_region_audit.json",
    )
    parser.add_argument(
        "--samples-out",
        type=Path,
        default=ROOT / "outputs" / "gcicy_k3_hard_region_samples.csv",
    )
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(array**2))),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    result = spearmanr(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    return {"spearman_rho": float(result.statistic), "two_sided_p_value": float(result.pvalue)}


def chart_label(chart: tuple[int, int, int]) -> str:
    return f"x{chart[0]}_y{chart[1]}_z{chart[2]}"


def orthogonal_projector(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame, _ = np.linalg.qr(np.asarray(tangent, dtype=np.complex128))
    frame = frame[:, :3]
    return frame @ frame.conjugate().T, frame


def subset_summary(mask: np.ndarray, absolute_residual: np.ndarray) -> dict:
    values = absolute_residual[np.asarray(mask, dtype=bool)]
    return {"count": int(len(values)), "absolute_centered_residual": distribution(values)}


def main() -> None:
    args = parse_args()
    if args.points <= 0 or args.projector_points_per_seed < 0:
        raise SystemExit("point counts must be non-negative and --points must be positive")
    if not args.artifact.exists():
        raise SystemExit(f"missing artifact: {args.artifact}")

    artifact = np.load(args.artifact)
    model_seed = int(artifact["generic_model_seed"])
    model = make_exact_generic_model(model_seed)
    if not np.allclose(artifact["p1_coefficients"], model.p1_coefficients) or not np.allclose(
        artifact["p2_tensor"], model.p2_tensor
    ):
        raise SystemExit("artifact coefficients do not match its exact model")
    exponents = artifact["global_section_exponents"]
    h_matrix = artifact["global_h_matrix"]
    normalization = float(artifact["global_section_normalization"])

    sample_rows: list[dict] = []
    charts_seen: set[tuple[int, int, int]] = set()
    implicit_choices: set[tuple[int, int, int]] = set()
    max_projector_error = 0.0
    max_principal_angle = 0.0
    max_implicit_ma_error = 0.0
    max_implicit_importance_error = 0.0
    projector_comparisons = 0

    for seed in parse_seeds(args.seeds):
        points = sample_generic_gcicy_points(model, args.points, seed=seed)
        metrics = generic_global_h_metrics(points, exponents, h_matrix, normalization=normalization)
        residuals = generic_residual_values(points, metrics)
        importance = generic_importance_weights(points)
        unweighted_mean = float(np.mean(residuals))
        weighted_mean = float(np.sum(importance * residuals) / np.sum(importance))

        for index, (point, metric, residual, weight) in enumerate(
            zip(points, metrics, residuals, importance, strict=True)
        ):
            eigenvalues = np.linalg.eigvalsh(metric)
            chart = point.projective_chart
            charts_seen.add(chart)
            sample_rows.append(
                {
                    "point_id": f"{seed}:{index}",
                    "seed": seed,
                    "index": index,
                    "chart": chart_label(chart),
                    "raw_ma_residual": float(residual),
                    "absolute_centered_residual": float(abs(residual - unweighted_mean)),
                    "absolute_weighted_centered_residual": float(abs(residual - weighted_mean)),
                    "importance_weight": float(weight),
                    "importance_log_weight": float(generic_importance_log_weight(point)),
                    "jacobian_min_singular_value": float(point.jacobian_min_singular_value),
                    "residue_denominator_abs": float(abs(point.residue_denominator)),
                    "rational_denominator_margin": float(
                        min(np.min(np.abs(point.x)), np.min(np.abs(point.y)))
                    ),
                    "minimum_homogeneous_coordinate_abs": float(
                        min(np.min(np.abs(point.x)), np.min(np.abs(point.y)), np.min(np.abs(point.z)))
                    ),
                    "selected_homogeneous_coordinate_abs": float(
                        min(abs(point.x[chart[0]]), abs(point.y[chart[1]]), abs(point.z[chart[2]]))
                    ),
                    "tangent_operator_norm": float(np.linalg.norm(point.tangent_basis, ord=2)),
                    "metric_min_eigenvalue": float(eigenvalues[0]),
                    "metric_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                }
            )

        for point in points[: args.projector_points_per_seed]:
            reference_projector, reference_frame = orthogonal_projector(point.tangent_basis)
            reference_metric = generic_global_h_metric(
                point, exponents, h_matrix, normalization=normalization
            )
            reference_ma = generic_monge_ampere_log_error(point, reference_metric)
            reference_importance = generic_importance_log_weight(point)
            for independent in combinations(range(7), 3):
                try:
                    candidate = generic_point_in_chart(
                        model,
                        point.x,
                        point.y,
                        point.z,
                        point.projective_chart,
                        independent=independent,
                    )
                except FloatingPointError:
                    continue
                candidate_projector, candidate_frame = orthogonal_projector(candidate.tangent_basis)
                projector_error = float(np.linalg.norm(candidate_projector - reference_projector))
                overlap = np.linalg.svd(reference_frame.conjugate().T @ candidate_frame, compute_uv=False)
                principal_angle = float(np.arccos(np.clip(np.min(overlap), 0.0, 1.0)))
                candidate_metric = generic_global_h_metric(
                    candidate, exponents, h_matrix, normalization=normalization
                )
                candidate_ma = generic_monge_ampere_log_error(candidate, candidate_metric)
                max_projector_error = max(max_projector_error, projector_error)
                max_principal_angle = max(max_principal_angle, principal_angle)
                max_implicit_ma_error = max(max_implicit_ma_error, abs(candidate_ma - reference_ma))
                max_implicit_importance_error = max(
                    max_implicit_importance_error,
                    abs(generic_importance_log_weight(candidate) - reference_importance),
                )
                implicit_choices.add(independent)
                projector_comparisons += 1
        print(f"seed {seed}: completed {len(points)} stress points", flush=True)

    columns = list(sample_rows[0])
    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    with args.samples_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sample_rows)

    arrays = {
        key: np.asarray([row[key] for row in sample_rows], dtype=float)
        for key in columns
        if key not in {"point_id", "seed", "index", "chart"}
    }
    absolute_residual = arrays["absolute_centered_residual"]
    stress_scores = {
        "small_jacobian_singular_value": -np.log10(arrays["jacobian_min_singular_value"]),
        "small_residue_denominator": -np.log10(arrays["residue_denominator_abs"]),
        "large_importance_weight": np.log10(arrays["importance_weight"]),
        "small_metric_eigenvalue": -np.log10(arrays["metric_min_eigenvalue"]),
        "large_metric_condition_number": np.log10(arrays["metric_condition_number"]),
        "small_rational_denominator_margin": -np.log10(arrays["rational_denominator_margin"]),
        "small_homogeneous_coordinate": -np.log10(arrays["minimum_homogeneous_coordinate_abs"]),
        "large_tangent_operator_norm": np.log10(arrays["tangent_operator_norm"]),
    }
    correlations = {
        label: correlation(absolute_residual, score) for label, score in stress_scores.items()
    }

    bottom_one_percent = {
        "jacobian_min_singular_value": arrays["jacobian_min_singular_value"]
        <= np.quantile(arrays["jacobian_min_singular_value"], 0.01),
        "residue_denominator_abs": arrays["residue_denominator_abs"]
        <= np.quantile(arrays["residue_denominator_abs"], 0.01),
        "metric_min_eigenvalue": arrays["metric_min_eigenvalue"]
        <= np.quantile(arrays["metric_min_eigenvalue"], 0.01),
        "rational_denominator_margin": arrays["rational_denominator_margin"]
        <= np.quantile(arrays["rational_denominator_margin"], 0.01),
    }
    hard_subsets = {
        f"bottom_1_percent_{label}": subset_summary(mask, absolute_residual)
        for label, mask in bottom_one_percent.items()
    }
    hard_subsets["top_1_percent_importance_weight"] = subset_summary(
        arrays["importance_weight"] >= np.quantile(arrays["importance_weight"], 0.99),
        absolute_residual,
    )

    chart_rows = {}
    for chart in sorted({row["chart"] for row in sample_rows}):
        values = np.asarray(
            [row["absolute_centered_residual"] for row in sample_rows if row["chart"] == chart],
            dtype=float,
        )
        chart_rows[chart] = {"count": int(len(values)), "residual": distribution(values)}

    finite = all(np.all(np.isfinite(values)) for values in arrays.values())
    gates = {
        "all_values_finite": bool(finite),
        "metric_positive_on_all_points": bool(np.min(arrays["metric_min_eigenvalue"]) > 0),
        "complete_projective_chart_coverage": len(charts_seen) == 24,
        "complete_implicit_choice_coverage": len(implicit_choices) == 35,
        "tangent_projector_consistency": max_projector_error <= 1e-9,
        "implicit_ma_consistency": max_implicit_ma_error <= 1e-8,
        "implicit_importance_consistency": max_implicit_importance_error <= 1e-8,
    }
    summary = {
        "description": "Fixed-sample hard-region and implicit-tangent stress audit of the accepted gCICY k=3 H metric.",
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": sha256(args.artifact),
        "model_seed": model_seed,
        "sample_seeds": parse_seeds(args.seeds),
        "points_per_seed": args.points,
        "total_points": len(sample_rows),
        "projective_charts_seen": len(charts_seen),
        "implicit_coordinate_choices_seen": len(implicit_choices),
        "projector_comparisons": projector_comparisons,
        "max_tangent_projector_frobenius_error": max_projector_error,
        "max_tangent_principal_angle_radians": max_principal_angle,
        "max_implicit_ma_error": max_implicit_ma_error,
        "max_implicit_importance_log_weight_error": max_implicit_importance_error,
        "distributions": {key: distribution(values) for key, values in arrays.items()},
        "stress_correlations_with_absolute_centered_residual": correlations,
        "hard_subsets": hard_subsets,
        "per_chart": chart_rows,
        "samples_csv": str(args.samples_out.resolve()),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "residual |centered|: "
        f"median={summary['distributions']['absolute_centered_residual']['median']:.6e}, "
        f"p95={summary['distributions']['absolute_centered_residual']['p95']:.6e}, "
        f"p99={summary['distributions']['absolute_centered_residual']['p99']:.6e}, "
        f"max={summary['distributions']['absolute_centered_residual']['max']:.6e}"
    )
    print(
        f"atlas={len(charts_seen)}/24, implicit={len(implicit_choices)}/35, "
        f"projector={max_projector_error:.3e}, implicit_MA={max_implicit_ma_error:.3e}"
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.samples_out}")
    if not summary["all_gates_passed"]:
        raise SystemExit("hard-region audit failed a geometry or positivity gate")


if __name__ == "__main__":
    main()
