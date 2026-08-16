#!/usr/bin/env python3
"""Discover and serialize audited local neighborhoods of metric tail points."""

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

from gcicy_metric.pipeline import (  # noqa: E402
    file_sha256,
    get_adapter,
    save_active_point_pool,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def require_exact_keys(row: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return row


def weighted_log_mean_exp(raw: np.ndarray, weights: np.ndarray) -> float:
    positive = np.asarray(weights, dtype=float)
    values = np.asarray(raw, dtype=float)
    if (
        values.shape != positive.shape
        or values.size == 0
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(positive))
        or np.min(positive) <= 0
    ):
        raise ValueError("weighted log-mean-exp inputs must be finite and positive")
    return float(
        np.logaddexp.reduce(np.log(positive) + values) - np.log(np.sum(positive))
    )


def audit_center(adapter, model, artifact, point: Any) -> dict[str, Any]:
    reference_metric = adapter.h_metric(point, artifact)
    reference_raw = adapter.monge_ampere_log_error(point, reference_metric)
    projective_errors = []
    projective_seen = 0
    for chart in adapter.projective_charts():
        if not adapter.chart_is_available(point, chart, minimum=1e-8):
            continue
        candidate = adapter.rechart_point(model, point, chart)
        metric = adapter.h_metric(candidate, artifact)
        raw = adapter.monge_ampere_log_error(candidate, metric)
        projective_errors.append(abs(raw - reference_raw))
        projective_seen += 1
    implicit_errors = []
    implicit_seen = 0
    current_chart = adapter.point_projective_chart(point)
    for independent in adapter.implicit_coordinate_choices():
        try:
            candidate = adapter.rechart_point(
                model,
                point,
                current_chart,
                independent=independent,
            )
            metric = adapter.h_metric(candidate, artifact)
            raw = adapter.monge_ampere_log_error(candidate, metric)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            continue
        implicit_errors.append(abs(raw - reference_raw))
        implicit_seen += 1
    metric_eigenvalues = np.linalg.eigvalsh(reference_metric)
    residue_denominator = getattr(point, "residue_denominator", None)
    return {
        "projective_charts_seen": int(projective_seen),
        "projective_charts_expected": int(adapter.expected_projective_chart_count()),
        "implicit_choices_seen": int(implicit_seen),
        "implicit_choices_expected": int(
            adapter.expected_implicit_coordinate_choice_count()
        ),
        "maximum_projective_log_ratio_error": float(
            max(projective_errors, default=float("inf"))
        ),
        "maximum_implicit_log_ratio_error": float(
            max(implicit_errors, default=float("inf"))
        ),
        "minimum_metric_eigenvalue": float(metric_eigenvalues[0]),
        "jacobian_minimum_singular_value": float(
            adapter.point_jacobian_min_singular_value(point)
        ),
        "residue_denominator_magnitude": (
            float(abs(residue_denominator))
            if residue_denominator is not None
            else None
        ),
    }


def center_audit_passed(row: dict[str, Any], discovery: dict[str, Any]) -> bool:
    return bool(
        row["projective_charts_seen"] == row["projective_charts_expected"]
        and row["implicit_choices_seen"] == row["implicit_choices_expected"]
        and row["maximum_projective_log_ratio_error"]
        <= float(discovery["maximum_cross_chart_log_ratio_error"])
        and row["maximum_implicit_log_ratio_error"]
        <= float(discovery["maximum_cross_chart_log_ratio_error"])
        and row["minimum_metric_eigenvalue"] > 0
        and row["jacobian_minimum_singular_value"]
        >= float(discovery["minimum_jacobian_singular_value"])
        and (
            row["residue_denominator_magnitude"] is None
            or row["residue_denominator_magnitude"] > 1e-12
        )
    )


def main() -> None:
    args = parse_args()
    spec_path = args.spec.expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if int(spec.get("schema_version", -1)) != 1:
        raise SystemExit("active-set specification must use schema_version 1")
    adapter = get_adapter(str(spec["adapter"]))
    model_spec = spec["model"]
    model_seed = int(model_spec["seed"])
    exact_model = bool(model_spec.get("exact", True))
    model = adapter.make_model(model_seed, exact=exact_model)
    artifact_path = resolve_path(spec["artifact"], spec_path.parent)
    artifact = adapter.load_h_artifact(artifact_path, model)
    discovery = spec["discovery"]
    output_spec = spec["output"]
    active_pool_path = resolve_path(output_spec["active_pool"], spec_path.parent)
    summary_path = resolve_path(output_spec["summary"], spec_path.parent)
    if active_pool_path.exists() or summary_path.exists():
        raise FileExistsError("active-set outputs already exist; refusing to overwrite")
    batches = [
        require_exact_keys(row, {"seed", "points"}, "discovery batch")
        for row in discovery["global_batches"]
    ]
    if not batches:
        raise ValueError("at least one discovery batch is required")
    seeds = [int(row["seed"]) for row in batches]
    if len(set(seeds)) != len(seeds):
        raise ValueError("discovery seeds must be distinct")
    preselected_count = int(discovery["preselected_candidate_count"])
    if preselected_count <= 0:
        raise ValueError("preselected candidate count must be positive")
    sampling_workers = int(discovery.get("sampling_workers", 1))
    sampling_cluster_size = int(discovery.get("sampling_cluster_size", 1))
    sampling_backend = str(discovery.get("sampling_backend", "process"))
    if any(int(row["points"]) % sampling_cluster_size for row in batches):
        raise ValueError("discovery point counts must contain complete sampling clusters")
    started = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    sampling_rows = []
    global_log_numerator = float("-inf")
    global_weight_sum = 0.0
    global_point_count = 0
    for row in batches:
        seed = int(row["seed"])
        count = int(row["points"])
        batch_started = time.perf_counter()
        if sampling_workers > 1:
            points, shards = sample_points_parallel(
                adapter,
                model_seed=model_seed,
                exact_model=exact_model,
                count=count,
                seed=seed,
                workers=sampling_workers,
                cluster_size=sampling_cluster_size,
                backend=sampling_backend,
            )
        else:
            points = adapter.sample_points(model, count, seed=seed)
            shards = []
        metrics = adapter.h_metrics(points, artifact)
        raw = adapter.residual_values(points, metrics)
        weights = adapter.importance_weights(points)
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError(f"discovery seed {seed} has non-finite residuals")
        batch_log_numerator = float(
            np.logaddexp.reduce(np.log(weights) + raw)
        )
        global_log_numerator = float(
            np.logaddexp(global_log_numerator, batch_log_numerator)
        )
        global_weight_sum += float(np.sum(weights))
        global_point_count += len(points)
        retain = min(preselected_count, len(points))
        indices = np.argpartition(raw, -retain)[-retain:]
        for index in indices:
            candidates.append(
                {
                    "point": points[int(index)],
                    "raw": float(raw[int(index)]),
                    "seed": seed,
                    "point_index": int(index),
                    "jacobian_minimum_singular_value": float(
                        adapter.point_jacobian_min_singular_value(points[int(index)])
                    ),
                }
            )
        sampling_rows.append(
            {
                "seed": seed,
                "points": int(len(points)),
                "batch_log_normalization": weighted_log_mean_exp(raw, weights),
                "maximum_raw_log_ratio": float(np.max(raw)),
                "sampling_shards": shards,
                "runtime_seconds": float(time.perf_counter() - batch_started),
            }
        )
        print(
            f"seed={seed}, points={len(points)}, max_raw={np.max(raw):.6e}, "
            f"seconds={time.perf_counter() - batch_started:.1f}",
            flush=True,
        )
    global_log_normalization = float(
        global_log_numerator - np.log(global_weight_sum)
    )
    for row in candidates:
        row["positive_log_ratio"] = float(row["raw"] - global_log_normalization)
    candidates.sort(key=lambda row: row["positive_log_ratio"], reverse=True)
    candidates = candidates[:preselected_count]
    threshold = float(discovery["positive_log_ratio_threshold"])
    violating_candidates = [
        row for row in candidates if row["positive_log_ratio"] > threshold
    ]
    centers: list[dict[str, Any]] = []
    rejected_center_audits = []
    maximum_centers = int(discovery["maximum_distinct_centers"])
    minimum_center_distance = float(
        discovery["minimum_product_fubini_study_center_distance"]
    )
    for candidate in violating_candidates:
        if any(
            adapter.point_distance(candidate["point"], center["point"])
            < minimum_center_distance
            for center in centers
        ):
            continue
        audit = audit_center(adapter, model, artifact, candidate["point"])
        if not center_audit_passed(audit, discovery):
            rejected_center_audits.append(
                {
                    "seed": candidate["seed"],
                    "point_index": candidate["point_index"],
                    "positive_log_ratio": candidate["positive_log_ratio"],
                    "audit": audit,
                }
            )
            continue
        candidate["audit"] = audit
        centers.append(candidate)
        if len(centers) >= maximum_centers:
            break
    active_points = []
    center_ids = []
    radii = []
    source_scores = []
    neighborhood_rows = []
    radii_spec = [float(value) for value in discovery["neighborhood_radii"]]
    points_per_radius = int(discovery["points_per_center_per_radius"])
    seed_sequence = np.random.SeedSequence(int(discovery["neighborhood_seed"]))
    child_sequences = seed_sequence.spawn(len(centers) * len(radii_spec))
    child_index = 0
    for center_id, center in enumerate(centers):
        active_points.append(center["point"])
        center_ids.append(center_id)
        radii.append(0.0)
        source_scores.append(center["positive_log_ratio"])
        for radius in radii_spec:
            local_seed = int(child_sequences[child_index].generate_state(1)[0])
            child_index += 1
            points, diagnostics = adapter.sample_local_neighborhood(
                model,
                center["point"],
                points_per_radius,
                seed=local_seed,
                radius=radius,
            )
            active_points.extend(points)
            center_ids.extend([center_id] * len(points))
            radii.extend([radius] * len(points))
            source_scores.extend([center["positive_log_ratio"]] * len(points))
            neighborhood_rows.append(
                {
                    "center_id": center_id,
                    "seed": local_seed,
                    **diagnostics,
                }
            )
    active_pool_sha256 = None
    active_summary = None
    if active_points:
        active_metrics = adapter.h_metrics(active_points, artifact)
        active_raw = adapter.residual_values(active_points, active_metrics)
        active_u = active_raw - global_log_normalization
        active_pool_sha256 = save_active_point_pool(
            active_pool_path,
            adapter,
            active_points,
            model_seed=model_seed,
            exact_model=exact_model,
            center_ids=center_ids,
            radii=radii,
            source_center_log_ratios=source_scores,
            metadata={
                "specification": str(spec_path),
                "specification_sha256": file_sha256(spec_path),
                "source_artifact": str(artifact_path),
                "source_artifact_sha256": file_sha256(artifact_path),
                "global_log_normalization": global_log_normalization,
                "reported_monte_carlo_sample": False,
            },
        )
        active_summary = {
            "point_count": int(len(active_points)),
            "minimum_positive_log_ratio_under_source": float(np.min(active_u)),
            "mean_positive_log_ratio_under_source": float(np.mean(active_u)),
            "maximum_positive_log_ratio_under_source": float(np.max(active_u)),
            "maximum_normalized_ratio_under_source": float(np.exp(np.max(active_u))),
        }
    summary = {
        "schema_version": 1,
        "adapter": adapter.key,
        "model_seed": model_seed,
        "exact_model": exact_model,
        "specification": str(spec_path),
        "specification_sha256": file_sha256(spec_path),
        "source_artifact": str(artifact_path),
        "source_artifact_sha256": file_sha256(artifact_path),
        "global_point_count": int(global_point_count),
        "global_independent_fibre_count": int(
            global_point_count // sampling_cluster_size
        ),
        "global_log_normalization": global_log_normalization,
        "sampling": sampling_rows,
        "positive_log_ratio_threshold": threshold,
        "equivalent_normalized_ratio_threshold": float(np.exp(threshold)),
        "preselected_candidate_count": int(len(candidates)),
        "violating_candidate_count_within_preselection": int(
            len(violating_candidates)
        ),
        "audited_distinct_center_count": int(len(centers)),
        "rejected_center_audits": rejected_center_audits,
        "centers": [
            {
                key: value
                for key, value in center.items()
                if key != "point"
            }
            for center in centers
        ],
        "neighborhoods": neighborhood_rows,
        "active_pool": str(active_pool_path) if active_points else None,
        "active_pool_sha256": active_pool_sha256,
        "active_pool_summary": active_summary,
        "reported_monte_carlo_estimate": False,
        "success": bool(not violating_candidates or centers),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if violating_candidates and not centers:
        raise SystemExit("violations were found but none passed the center audit")


if __name__ == "__main__":
    main()
