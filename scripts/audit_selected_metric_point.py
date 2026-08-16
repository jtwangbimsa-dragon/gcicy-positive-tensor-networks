#!/usr/bin/env python3
"""Audit one disclosed metric-tail point across every available local chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--exact-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--point-index", type=int, required=True)
    parser.add_argument("--cluster-size", type=int, default=4)
    parser.add_argument("--minimum-selected-coordinate", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def complex_value(value: complex) -> dict[str, float]:
    number = complex(value)
    return {
        "real": float(number.real),
        "imag": float(number.imag),
        "magnitude": float(abs(number)),
    }


def metric_summary(metric: np.ndarray) -> dict[str, Any]:
    value = np.asarray(metric, dtype=np.complex128)
    eigenvalues = np.linalg.eigvalsh(value)
    return {
        "eigenvalues": [float(item) for item in eigenvalues],
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "log_determinant": float(np.sum(np.log(eigenvalues))),
        "hermitian_error": float(np.max(np.abs(value - value.conjugate().T))),
    }


def projective_separation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.complex128)
    b = np.asarray(right, dtype=np.complex128)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    overlap = min(1.0, float(abs(np.vdot(a, b))))
    return float(np.sqrt(max(0.0, 1.0 - overlap**2)))


def point_summary(
    adapter: Any, model: Any, point: Any, artifact: Any
) -> dict[str, Any]:
    metric = adapter.h_metric(point, artifact)
    baseline = adapter.baseline_metric(point)
    values, derivatives = adapter.section_values_and_jacobian(
        point,
        artifact.section_exponents,
    )
    denominator = float(
        np.real(
            np.vdot(
                values,
                np.asarray(artifact.h_matrix, dtype=np.complex128) @ values,
            )
        )
    )
    output = {
        "projective_chart": [int(value) for value in point.projective_chart],
        "independent_indices": [int(value) for value in point.independent_indices],
        "dependent_indices": [int(value) for value in point.dependent_indices],
        "jacobian_min_singular_value": float(point.jacobian_min_singular_value),
        "residue_denominator": complex_value(point.residue_denominator),
        "tangent_basis_condition_number": float(
            np.linalg.cond(np.asarray(point.tangent_basis, dtype=np.complex128))
        ),
        "homogeneous_coordinates": [
            [complex_value(value) for value in factor] for factor in point.coordinates
        ],
        "affine_coordinates": [
            complex_value(value) for value in point.affine_coordinates
        ],
        "importance_log_weight": float(adapter.importance_log_weight(point)),
        "holomorphic_volume_log_density": float(
            adapter.holomorphic_volume_log_density(point)
        ),
        "candidate_log_monge_ampere_ratio": float(
            adapter.monge_ampere_log_error(point, metric)
        ),
        "baseline_log_monge_ampere_ratio": float(
            adapter.monge_ampere_log_error(point, baseline)
        ),
        "candidate_metric": metric_summary(metric),
        "baseline_metric": metric_summary(baseline),
        "section_euclidean_norm": float(np.linalg.norm(values)),
        "section_h_norm_squared": denominator,
        "section_h_rayleigh_quotient": float(
            denominator / np.vdot(values, values).real
        ),
        "section_derivative_frobenius_norm": float(np.linalg.norm(derivatives)),
    }
    if hasattr(model, "p_tensor") and hasattr(point, "y"):
        y = np.asarray(point.y, dtype=np.complex128)
        degree = int(np.asarray(model.p_tensor).shape[0] - 1)
        monomials = np.asarray(
            [y[0] ** (degree - power) * y[1] ** power for power in range(degree + 1)],
            dtype=np.complex128,
        )
        positive_row = monomials @ np.asarray(model.p_tensor, dtype=np.complex128)
        output["positive_equation_row_norm"] = float(np.linalg.norm(positive_row))
    return output


def atlas_consistency(
    adapter: Any,
    model: Any,
    point: Any,
    artifact: Any,
    *,
    minimum_selected_coordinate: float,
) -> dict[str, Any]:
    reference_metric = adapter.h_metric(point, artifact)
    reference_baseline = adapter.baseline_metric(point)
    reference_ma = adapter.monge_ampere_log_error(point, reference_metric)
    reference_baseline_ma = adapter.monge_ampere_log_error(point, reference_baseline)
    reference_importance = adapter.importance_log_weight(point)

    projective_rows = []
    for chart in adapter.projective_charts():
        selected = tuple(int(value) for value in chart)
        if not adapter.chart_is_available(
            point,
            selected,
            minimum=minimum_selected_coordinate,
        ):
            continue
        candidate = adapter.rechart_point(model, point, selected)
        candidate_metric = adapter.h_metric(candidate, artifact)
        candidate_baseline = adapter.baseline_metric(candidate)
        projective_rows.append(
            {
                "chart": list(selected),
                "candidate_ma_absolute_error": float(
                    abs(
                        adapter.monge_ampere_log_error(candidate, candidate_metric)
                        - reference_ma
                    )
                ),
                "baseline_ma_absolute_error": float(
                    abs(
                        adapter.monge_ampere_log_error(candidate, candidate_baseline)
                        - reference_baseline_ma
                    )
                ),
                "importance_log_weight_absolute_error": float(
                    abs(adapter.importance_log_weight(candidate) - reference_importance)
                ),
                "minimum_metric_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(candidate_metric))
                ),
            }
        )

    implicit_rows = []
    reference_chart = adapter.point_projective_chart(point)
    for independent in adapter.implicit_coordinate_choices():
        selected = tuple(int(value) for value in independent)
        try:
            candidate = adapter.rechart_point(
                model,
                point,
                reference_chart,
                independent=selected,
            )
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        candidate_metric = adapter.h_metric(candidate, artifact)
        candidate_baseline = adapter.baseline_metric(candidate)
        implicit_rows.append(
            {
                "independent_indices": list(selected),
                "candidate_ma_absolute_error": float(
                    abs(
                        adapter.monge_ampere_log_error(candidate, candidate_metric)
                        - reference_ma
                    )
                ),
                "baseline_ma_absolute_error": float(
                    abs(
                        adapter.monge_ampere_log_error(candidate, candidate_baseline)
                        - reference_baseline_ma
                    )
                ),
                "importance_log_weight_absolute_error": float(
                    abs(adapter.importance_log_weight(candidate) - reference_importance)
                ),
                "minimum_metric_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(candidate_metric))
                ),
            }
        )

    def maximum(rows: Sequence[dict[str, Any]], key: str) -> float:
        return float(max((row[key] for row in rows), default=0.0))

    return {
        "projective_chart_count": len(projective_rows),
        "implicit_coordinate_choice_count": len(implicit_rows),
        "maximum_projective_candidate_ma_error": maximum(
            projective_rows,
            "candidate_ma_absolute_error",
        ),
        "maximum_implicit_candidate_ma_error": maximum(
            implicit_rows,
            "candidate_ma_absolute_error",
        ),
        "maximum_projective_baseline_ma_error": maximum(
            projective_rows,
            "baseline_ma_absolute_error",
        ),
        "maximum_implicit_baseline_ma_error": maximum(
            implicit_rows,
            "baseline_ma_absolute_error",
        ),
        "maximum_projective_importance_error": maximum(
            projective_rows,
            "importance_log_weight_absolute_error",
        ),
        "maximum_implicit_importance_error": maximum(
            implicit_rows,
            "importance_log_weight_absolute_error",
        ),
        "minimum_recharted_metric_eigenvalue": float(
            min(
                [
                    row["minimum_metric_eigenvalue"]
                    for row in projective_rows + implicit_rows
                ],
                default=float("inf"),
            )
        ),
        "projective_rows": projective_rows,
        "implicit_rows": implicit_rows,
    }


def main() -> None:
    args = parse_args()
    if args.point_index < 0 or args.cluster_size <= 0:
        raise SystemExit("point-index must be non-negative and cluster-size positive")
    if args.minimum_selected_coordinate <= 0:
        raise SystemExit("minimum-selected-coordinate must be positive")

    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    artifact_path = args.artifact.expanduser().resolve()
    artifact = adapter.load_h_artifact(artifact_path, model)
    cluster_id = args.point_index // args.cluster_size
    cluster_start = cluster_id * args.cluster_size
    sample_count = cluster_start + args.cluster_size
    points, sampling = adapter.sample_points_with_diagnostics(
        model,
        sample_count,
        seed=args.seed,
    )
    selected = points[args.point_index]
    cluster = points[cluster_start:sample_count]
    separations = [
        projective_separation(cluster[left].x, cluster[right].x)
        for left in range(len(cluster))
        for right in range(left + 1, len(cluster))
    ]

    output = {
        "schema_version": 1,
        "description": (
            "Disclosed tail-point audit across every available projective chart "
            "and nonsingular implicit-coordinate choice."
        ),
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "artifact": str(artifact_path),
        "seed": args.seed,
        "point_index": args.point_index,
        "sample_count_for_replay": sample_count,
        "sampling_diagnostics": sampling,
        "cluster": {
            "id": cluster_id,
            "point_indices": list(range(cluster_start, sample_count)),
            "minimum_projective_x_root_separation": float(min(separations)),
            "maximum_projective_x_root_separation": float(max(separations)),
            "points": [
                point_summary(adapter, model, point, artifact) for point in cluster
            ],
        },
        "selected_point": point_summary(adapter, model, selected, artifact),
        "atlas_consistency": atlas_consistency(
            adapter,
            model,
            selected,
            artifact,
            minimum_selected_coordinate=args.minimum_selected_coordinate,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
