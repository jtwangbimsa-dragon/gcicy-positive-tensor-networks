#!/usr/bin/env python3
"""Audit numerical fibre-degree completeness for any registered gCICY adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, sampling_cluster_statistics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--points-per-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--inexact-model",
        action="store_true",
        help="Use floating rather than exact integer model coefficients.",
    )
    return parser.parse_args()


def merge_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in rows:
        for label, count in row.get(key, {}).items():
            merged[str(label)] = merged.get(str(label), 0) + int(count)
    return dict(sorted(merged.items(), key=lambda item: item[0]))


def main() -> None:
    args = parse_args()
    if args.points_per_seed <= 0:
        raise SystemExit("points-per-seed must be positive")
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=not args.inexact_model)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_started = time.perf_counter()
        points, diagnostics = adapter.sample_points_with_diagnostics(
            model,
            args.points_per_seed,
            seed=int(seed),
        )
        log_weights = np.asarray(
            [adapter.importance_log_weight(point) for point in points],
            dtype=float,
        )
        shifted = np.exp(log_weights - float(np.max(log_weights)))
        weights = shifted / float(np.sum(shifted))
        cluster_ids = adapter.sampling_cluster_ids(points)
        cluster_statistics = sampling_cluster_statistics(weights, cluster_ids)
        sorted_weights = np.sort(weights)[::-1]
        top_count = max(1, int(np.ceil(0.01 * len(weights))))
        rows.append(
            {
                "seed": int(seed),
                **diagnostics,
                **cluster_statistics,
                "maximum_mean_normalized_importance_weight": float(
                    len(weights) * np.max(weights)
                ),
                "top_one_percent_importance_weight_fraction": float(
                    np.sum(sorted_weights[:top_count])
                ),
                "minimum_jacobian_singular_value": float(
                    min(adapter.point_jacobian_min_singular_value(point) for point in points)
                ),
                "elapsed_seconds": float(time.perf_counter() - seed_started),
            }
        )

    aggregate = {
        "attempted_clusters": int(sum(row["attempted_clusters"] for row in rows)),
        "accepted_clusters": int(sum(row["accepted_clusters"] for row in rows)),
        "rejected_clusters": int(sum(row["rejected_clusters"] for row in rows)),
        "returned_complete_clusters": int(
            sum(row["returned_complete_clusters"] for row in rows)
        ),
        "root_count_histogram": merge_counts(rows, "root_count_histogram"),
        "valid_root_count_histogram": merge_counts(
            rows,
            "valid_root_count_histogram",
        ),
        "accepted_root_count_histogram": merge_counts(
            rows,
            "accepted_root_count_histogram",
        ),
        "rejection_reasons": merge_counts(rows, "rejection_reasons"),
        "minimum_projective_root_separation": float(
            min(row["minimum_projective_root_separation"] for row in rows)
        ),
        "maximum_accepted_relative_residual": float(
            max(row["maximum_accepted_relative_residual"] for row in rows)
        ),
        "minimum_point_effective_sample_size": float(
            min(row["point_effective_sample_size"] for row in rows)
        ),
        "minimum_cluster_weight_effective_sample_size": float(
            min(row["cluster_weight_effective_sample_size"] for row in rows)
        ),
        "maximum_mean_normalized_importance_weight": float(
            max(row["maximum_mean_normalized_importance_weight"] for row in rows)
        ),
        "maximum_top_one_percent_importance_weight_fraction": float(
            max(row["top_one_percent_importance_weight_fraction"] for row in rows)
        ),
        "minimum_jacobian_singular_value": float(
            min(row["minimum_jacobian_singular_value"] for row in rows)
        ),
    }
    gates = {
        "all_requested_points_returned": all(
            row["returned_points"] == args.points_per_seed for row in rows
        ),
        "all_returned_clusters_complete": all(
            row["all_returned_clusters_complete"] for row in rows
        ),
        "accepted_fibres_have_expected_root_count": all(
            sum(row["accepted_root_count_histogram"].values())
            == row["accepted_clusters"]
            for row in rows
        ),
        "finite_positive_root_separation": bool(
            np.isfinite(aggregate["minimum_projective_root_separation"])
            and aggregate["minimum_projective_root_separation"] > 0
        ),
        "accepted_residuals_within_tolerance": all(
            row["maximum_accepted_relative_residual"] <= row["residual_tolerance"]
            for row in rows
        ),
        "importance_weights_not_used_for_acceptance": all(
            not row["importance_weight_used_for_acceptance"] for row in rows
        ),
    }
    output = {
        "schema_version": 1,
        "description": (
            "Numerical fibre-degree-completeness audit with root, rejection, conditioning, "
            "point-weight, and cluster-weight diagnostics."
        ),
        "adapter": adapter.key,
        "configuration": adapter.configuration.to_dict(),
        "model_seed": int(args.model_seed),
        "exact_model": bool(not args.inexact_model),
        "model": adapter.model_metadata(model),
        "seeds": [int(seed) for seed in args.seeds],
        "points_per_seed": int(args.points_per_seed),
        "selection_policy": (
            "A fibre is accepted only when the expected number of distinct projective "
            "roots is returned, every root has a finite residual within tolerance, "
            "and every root admits a nonsingular implicit chart. Projective root "
            "separation is audited explicitly. Importance weights are evaluated only "
            "after acceptance."
        ),
        "per_seed": rows,
        "aggregate": aggregate,
        "gates": gates,
        "success": bool(all(gates.values())),
        "timing_seconds": float(time.perf_counter() - started),
    }
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success": output["success"], **aggregate}, indent=2))
    print(f"wrote {args.out}")
    if not output["success"]:
        raise SystemExit("sampler audit failed")


if __name__ == "__main__":
    main()
