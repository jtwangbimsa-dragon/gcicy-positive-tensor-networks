#!/usr/bin/env python3
"""Cluster delete-group jackknife for saved model/teacher MA point arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.audit import standard_errors  # noqa: E402


METRICS = (
    "sigma",
    "inverse_sigma",
    "sqrt_squared_energy",
    "weighted_centered_log_ma_rms",
    "normalized_ratio_max",
    "positive_log_ratio_q999",
    "positive_log_ratio_cvar_1pct",
    "normalized_ratio_above_3_weighted_mass",
    "metric_volume_maximum_point_mass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def jackknife_summary(point: float, leave_one_group: np.ndarray) -> dict:
    group_count = len(leave_one_group)
    pseudo = group_count * point - (group_count - 1) * leave_one_group
    estimate = float(np.mean(pseudo))
    standard_error = float(np.std(pseudo, ddof=1) / np.sqrt(group_count))
    return {
        "point_estimate": float(point),
        "bias_corrected_estimate": estimate,
        "standard_error": standard_error,
        "confidence_interval_95": [
            estimate - 1.96 * standard_error,
            estimate + 1.96 * standard_error,
        ],
    }


def selected_metrics(log_eta: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    row = standard_errors(log_eta, weights)
    return {name: float(row[name]) for name in METRICS}


def main() -> None:
    args = parse_args()
    arrays_path = args.arrays.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    if args.groups < 2:
        raise SystemExit("groups must be at least two")
    with np.load(arrays_path, allow_pickle=False) as payload:
        required = {
            "schema",
            "model_log_eta",
            "teacher_log_eta",
            "importance_weights",
            "sampling_cluster_ids",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"point array artifact lacks fields: {missing}")
        model = np.asarray(payload["model_log_eta"], dtype=np.float64)
        teacher = np.asarray(payload["teacher_log_eta"], dtype=np.float64)
        weights = np.asarray(payload["importance_weights"], dtype=np.float64)
        cluster_ids = np.asarray(payload["sampling_cluster_ids"], dtype=np.int64)
        schema = str(payload["schema"])
    if not (
        model.ndim == teacher.ndim == weights.ndim == cluster_ids.ndim == 1
        and model.shape == teacher.shape == weights.shape == cluster_ids.shape
        and len(model) > 0
    ):
        raise ValueError("saved point arrays are not aligned vectors")
    if (
        not np.all(np.isfinite(model))
        or not np.all(np.isfinite(teacher))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("saved point arrays contain invalid values")
    unique_clusters = np.unique(cluster_ids)
    if len(unique_clusters) < 2 * args.groups:
        raise ValueError("jackknife needs at least two fibre clusters per group")
    cluster_to_group = {
        int(cluster): index % args.groups
        for index, cluster in enumerate(unique_clusters.tolist())
    }
    point_groups = np.asarray(
        [cluster_to_group[int(cluster)] for cluster in cluster_ids],
        dtype=np.int64,
    )

    full_model = selected_metrics(model, weights)
    full_teacher = selected_metrics(teacher, weights)
    leave_model = {name: [] for name in METRICS}
    leave_teacher = {name: [] for name in METRICS}
    deleted_weight_mass = []
    deleted_cluster_count = []
    normalized_weights = weights / np.sum(weights)
    for group in range(args.groups):
        deleted = point_groups == group
        retained = ~deleted
        deleted_weight_mass.append(float(np.sum(normalized_weights[deleted])))
        deleted_cluster_count.append(int(np.unique(cluster_ids[deleted]).size))
        model_row = selected_metrics(model[retained], weights[retained])
        teacher_row = selected_metrics(teacher[retained], weights[retained])
        for name in METRICS:
            leave_model[name].append(model_row[name])
            leave_teacher[name].append(teacher_row[name])

    model_summary = {}
    teacher_summary = {}
    paired_difference = {}
    for name in METRICS:
        model_values = np.asarray(leave_model[name], dtype=np.float64)
        teacher_values = np.asarray(leave_teacher[name], dtype=np.float64)
        model_summary[name] = jackknife_summary(full_model[name], model_values)
        teacher_summary[name] = jackknife_summary(full_teacher[name], teacher_values)
        paired_difference[name] = jackknife_summary(
            full_model[name] - full_teacher[name],
            model_values - teacher_values,
        )

    report = {
        "schema": "ma-cluster-delete-group-jackknife-v1",
        "point_arrays": str(arrays_path),
        "point_arrays_sha256": sha256_file(arrays_path),
        "point_arrays_schema": schema,
        "points": len(model),
        "fibre_clusters": len(unique_clusters),
        "group_count": args.groups,
        "group_assignment": "sorted fibre cluster index modulo group count",
        "deleted_weight_mass_range": [
            float(np.min(deleted_weight_mass)),
            float(np.max(deleted_weight_mass)),
        ],
        "deleted_cluster_count_range": [
            int(np.min(deleted_cluster_count)),
            int(np.max(deleted_cluster_count)),
        ],
        "model": model_summary,
        "teacher": teacher_summary,
        "paired_model_minus_teacher": paired_difference,
        "caveat": (
            "Jackknife intervals for sample maxima and thresholded tail mass are "
            "diagnostic; q999 and CVaR are more stable tail summaries."
        ),
    }
    atomic_json(out_path, report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
