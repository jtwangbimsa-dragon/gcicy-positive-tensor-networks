#!/usr/bin/env python3
"""Compare the two independent random-line implementations for type-(1,1)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, kstest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, standard_errors  # noqa: E402
from gcicy_metric.simple_patch import fubini_study_metric  # noqa: E402
from gcicy_metric.type21_hirzebruch_x3 import (  # noqa: E402
    hirzebruch_type21_holomorphic_volume_log_density,
    hirzebruch_type21_importance_log_weight,
    hirzebruch_type21_proposal_log_density,
    sample_hirzebruch_type21_points_with_diagnostics,
)


EXPECTED_X3_OVER_PROPOSAL = 11.0 / 12.0
EXPECTED_TOTAL_OVER_PROPOSAL = 23.0 / 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--projection-seed", type=int, default=73801)
    parser.add_argument("--null-space-seed", type=int, default=73802)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def mean_standard_error(values: np.ndarray, expected: float) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    standard_error = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    mean = float(np.mean(samples))
    return {
        "sample_count": int(len(samples)),
        "mean": mean,
        "standard_error": standard_error,
        "expected": expected,
        "difference": mean - expected,
        "z_score": (mean - expected) / standard_error,
    }


def direct_product_fs_metric(point: Any) -> np.ndarray:
    ambient = np.zeros((6, 6), dtype=np.complex128)
    for start, dimension in ((0, 4), (4, 1)):
        block = slice(start, start + dimension)
        ambient[block, block] = fubini_study_metric(
            point.affine_coordinates[block]
        )
    metric = point.tangent_basis.conjugate().T @ ambient @ point.tangent_basis
    return 0.5 * (metric + metric.conjugate().T)


def summarize(model: Any, points: list[Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    weights_log = np.asarray(
        [hirzebruch_type21_importance_log_weight(point) for point in points]
    )
    weights = np.exp(weights_log - np.max(weights_log))
    proposal_log = np.asarray(
        [hirzebruch_type21_proposal_log_density(point) for point in points]
    )
    omega_log = np.asarray(
        [hirzebruch_type21_holomorphic_volume_log_density(point) for point in points]
    )
    baseline_metrics = np.asarray([direct_product_fs_metric(point) for point in points])
    baseline_eigenvalues = np.linalg.eigvalsh(baseline_metrics)
    if np.min(baseline_eigenvalues) <= 0:
        raise FloatingPointError("direct product FS metric is not positive")
    baseline_logdet = np.sum(np.log(baseline_eigenvalues), axis=1)
    fs_stats = standard_errors(baseline_logdet - omega_log, weights)

    x3_over_proposal = np.empty(len(points), dtype=np.float64)
    total_over_proposal = np.exp(baseline_logdet - proposal_log)
    for index, point in enumerate(points):
        x_metric = fubini_study_metric(point.affine_coordinates[:4])
        x_tangent = point.tangent_basis[:4, :]
        pulled_x = x_tangent.conjugate().T @ x_metric @ x_tangent
        pulled_x = 0.5 * (pulled_x + pulled_x.conjugate().T)
        x_eigenvalues = np.linalg.eigvalsh(pulled_x)
        if x_eigenvalues[0] <= 0:
            raise FloatingPointError("pulled P4 metric is not positive")
        x3_over_proposal[index] = math.exp(
            float(np.sum(np.log(x_eigenvalues))) - proposal_log[index]
        )

    y = np.asarray([point.y for point in points[::4]])
    y_coordinate = np.abs(y[:, 1]) ** 2 / np.sum(np.abs(y) ** 2, axis=1)
    y_uniform = kstest(y_coordinate, "uniform")
    return {
        "diagnostics": diagnostics,
        "y_coordinate": y_coordinate,
        "proposal_log_density": proposal_log,
        "centered_log_importance_weight": weights_log - np.mean(weights_log),
        "sampler_statistics": {
            "base_y_uniform_ks_statistic": float(y_uniform.statistic),
            "base_y_uniform_ks_pvalue": float(y_uniform.pvalue),
            "importance_effective_sample_size": float(
                np.sum(weights) ** 2 / np.sum(weights**2)
            ),
            "fubini_study_residual": fs_stats,
            "topological_moments": {
                "jx3_over_three_jx2jy": mean_standard_error(
                    x3_over_proposal, EXPECTED_X3_OVER_PROPOSAL
                ),
                "jx_plus_jy_cubed_over_three_jx2jy": mean_standard_error(
                    total_over_proposal, EXPECTED_TOTAL_OVER_PROPOSAL
                ),
                "pointwise_identity_total_minus_x_maximum_absolute_error": float(
                    np.max(np.abs((total_over_proposal - x3_over_proposal) - 1.0))
                ),
            },
        },
    }


def main() -> None:
    args = parse_args()
    if args.points <= 0 or args.points % 4:
        raise SystemExit("--points must be a positive multiple of four")
    if args.projection_seed == args.null_space_seed:
        raise SystemExit("the two sampler seeds must be distinct")
    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    model = adapter.make_model(args.model_seed, exact=True)
    started = time.perf_counter()
    rows = {}
    for label, safe_projection, seed in (
        ("analytic_projection", True, args.projection_seed),
        ("scipy_null_space", False, args.null_space_seed),
    ):
        points, diagnostics = sample_hirzebruch_type21_points_with_diagnostics(
            model,
            args.points,
            seed=seed,
            safe_projection=safe_projection,
        )
        rows[label] = summarize(model, points, diagnostics)

    comparisons = {}
    for key in (
        "y_coordinate",
        "proposal_log_density",
        "centered_log_importance_weight",
    ):
        result = ks_2samp(rows["analytic_projection"][key], rows["scipy_null_space"][key])
        comparisons[f"{key}_two_sample_ks"] = {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
        }
    gates = {
        "both_return_complete_four_root_fibres": all(
            row["diagnostics"]["all_returned_clusters_complete"]
            and row["diagnostics"]["accepted_root_count_histogram"] == {
                "4": args.points // 4
            }
            for row in rows.values()
        ),
        "both_base_y_uniform_p_above_1e_3": all(
            row["sampler_statistics"]["base_y_uniform_ks_pvalue"] > 1.0e-3
            for row in rows.values()
        ),
        "implementation_distributions_ks_p_above_1e_4": all(
            row["pvalue"] > 1.0e-4 for row in comparisons.values()
        ),
        "both_topological_moments_within_four_standard_errors": all(
            abs(moment["z_score"]) < 4.0
            for row in rows.values()
            for moment in row["sampler_statistics"]["topological_moments"].values()
            if isinstance(moment, dict)
        ),
        "both_pointwise_topological_identities_below_1e_6": all(
            row["sampler_statistics"]["topological_moments"][
                "pointwise_identity_total_minus_x_maximum_absolute_error"
            ]
            < 1.0e-6
            for row in rows.values()
        ),
    }
    output = {
        "schema": "type11-sampler-implementation-parity-v1",
        "model_seed": args.model_seed,
        "points_per_implementation": args.points,
        "independent_fibres_per_implementation": args.points // 4,
        "implementations": {
            label: {
                "seed": seed,
                "safe_projection": safe_projection,
                **rows[label]["sampler_statistics"],
                "sampling_diagnostics": rows[label]["diagnostics"],
            }
            for label, safe_projection, seed in (
                ("analytic_projection", True, args.projection_seed),
                ("scipy_null_space", False, args.null_space_seed),
            )
        },
        "two_sample_comparisons": comparisons,
        "gates": gates,
        "success": bool(all(gates.values())),
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_limit": (
            "Agreement rejects implementation-specific projection bias but does not "
            "constitute a proof that the common random-line measure is exact."
        ),
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_value(output), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success": output["success"], "gates": gates}, indent=2))
    print(f"wrote {out}")
    if not output["success"]:
        raise SystemExit("sampler implementation parity audit failed")


if __name__ == "__main__":
    main()
