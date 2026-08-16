#!/usr/bin/env python3
"""Replay and diagnose paired type-(1,1) H-metric tail witnesses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.audit import (  # noqa: E402
    normalized_volume_ratios,
    standard_errors,
)
from gcicy_metric.pipeline.parallel_sampling import (  # noqa: E402
    sample_points_parallel,
)
from gcicy_metric.simple_patch import fubini_study_metric  # noqa: E402
from gcicy_metric.type21_hirzebruch_x3 import (  # noqa: E402
    local_equations_jacobian_and_scales,
)
from scripts.audit_selected_metric_point import (  # noqa: E402
    atlas_consistency,
    point_summary,
    projective_separation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--target-artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73103)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def direct_geometry(model: Any, point: Any, cluster: list[Any]) -> dict[str, Any]:
    equations, jacobian, scales = local_equations_jacobian_and_scales(
        model,
        point.affine_coordinates,
        point.projective_chart,
    )
    reduced_jacobian = jacobian[[0, 2], :5]
    ambient_metric = np.zeros((5, 5), dtype=np.complex128)
    ambient_metric[:4, :4] = fubini_study_metric(point.affine_coordinates[:4])
    ambient_metric[4, 4] = fubini_study_metric(
        point.affine_coordinates[4:5]
    )[0, 0]
    inverse_metric = np.linalg.inv(ambient_metric)
    gram = reduced_jacobian @ inverse_metric @ reduced_jacobian.conjugate().T
    norms = np.sqrt(np.real(np.diag(gram)))
    normalized_gram = gram / np.outer(norms, norms)
    normalized_gram = 0.5 * (normalized_gram + normalized_gram.conjugate().T)
    separations = [
        projective_separation(cluster[left].x, cluster[right].x)
        for left in range(len(cluster))
        for right in range(left + 1, len(cluster))
    ]
    return {
        "relative_equation_residual": float(
            np.max(np.abs(equations) / np.maximum(1.0, scales))
        ),
        "normalized_reduced_jacobian_min_singular_value": math.sqrt(
            max(0.0, float(np.linalg.eigvalsh(normalized_gram)[0]))
        ),
        "minimum_projective_x_root_separation": float(min(separations)),
        "maximum_projective_x_root_separation": float(max(separations)),
    }


def main() -> None:
    args = parse_args()
    if args.points <= 0 or args.points % 4 or args.workers <= 0:
        raise SystemExit("points must be a positive multiple of four; workers must be positive")

    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    model = adapter.make_model(20260731, exact=True)
    points, shards = sample_points_parallel(
        adapter,
        model_seed=20260731,
        exact_model=True,
        count=args.points,
        seed=args.seed,
        workers=args.workers,
        cluster_size=4,
        backend="process",
    )
    source = adapter.load_h_artifact(args.source_artifact.expanduser().resolve(), model)
    target = adapter.load_h_artifact(args.target_artifact.expanduser().resolve(), model)
    weights = np.asarray(adapter.importance_weights(points), dtype=np.float64)

    ratios: dict[str, np.ndarray] = {}
    errors: dict[str, dict[str, float]] = {}
    artifacts = {"source": source, "target": target}
    for name, artifact in artifacts.items():
        metrics = adapter.h_metrics(points, artifact)
        raw = np.asarray(adapter.residual_values(points, metrics), dtype=np.float64)
        ratios[name], _ = normalized_volume_ratios(raw, weights)
        errors[name] = standard_errors(raw, weights)

    def witness(name: str, index: int) -> dict[str, Any]:
        cluster_start = 4 * (index // 4)
        cluster = points[cluster_start : cluster_start + 4]
        selected = points[index]
        return {
            "index": index,
            "cluster_start": cluster_start,
            "cluster_point_indices": list(range(cluster_start, cluster_start + 4)),
            "source_ratio": float(ratios["source"][index]),
            "target_ratio": float(ratios["target"][index]),
            "source_cluster_ratios": [
                float(value) for value in ratios["source"][cluster_start : cluster_start + 4]
            ],
            "target_cluster_ratios": [
                float(value) for value in ratios["target"][cluster_start : cluster_start + 4]
            ],
            "importance_weight": float(weights[index]),
            "geometry": direct_geometry(model, selected, cluster),
            "source_point": point_summary(adapter, model, selected, source),
            "target_point": point_summary(adapter, model, selected, target),
            "source_atlas": atlas_consistency(
                adapter,
                model,
                selected,
                source,
                minimum_selected_coordinate=1.0e-8,
            ),
            "target_atlas": atlas_consistency(
                adapter,
                model,
                selected,
                target,
                minimum_selected_coordinate=1.0e-8,
            ),
            "selected_by": name,
        }

    source_worst = int(np.argmax(ratios["source"]))
    target_worst = int(np.argmax(ratios["target"]))
    output = {
        "schema": "type11-paired-tail-witness-v1",
        "seed": args.seed,
        "point_count": args.points,
        "sampling_shards": shards,
        "source_artifact": str(args.source_artifact.expanduser().resolve()),
        "target_artifact": str(args.target_artifact.expanduser().resolve()),
        "source_errors": errors["source"],
        "target_errors": errors["target"],
        "ratio_spearman": float(spearmanr(ratios["source"], ratios["target"]).statistic),
        "source_worst": witness("source", source_worst),
        "target_worst": witness("target", target_worst),
        "interpretation_limit": (
            "These are witnesses on one finite blind sample, not global extrema."
        ),
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source_worst": {
                    "index": source_worst,
                    "source_ratio": output["source_worst"]["source_ratio"],
                    "target_ratio": output["source_worst"]["target_ratio"],
                    "geometry": output["source_worst"]["geometry"],
                },
                "target_worst": {
                    "index": target_worst,
                    "source_ratio": output["target_worst"]["source_ratio"],
                    "target_ratio": output["target_worst"]["target_ratio"],
                    "geometry": output["target_worst"]["geometry"],
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
