#!/usr/bin/env python3
"""Compare scalar Ritz observables for a positive tensor metric and controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    cluster_delete_group_jackknife_scalar_ritz,
    confidence_interval,
    estimate_scalar_laplacian_ritz,
    get_adapter,
    metric_volume_weights,
    paired_metric_distortion,
    positive_tensor_network_from_artifact_payload,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    dense_h_potential_and_metric,
)


METRIC_KEYS = ("low_degree_reference", "tensor_network", "full_h_teacher")


def parse_integer_list(value: str) -> tuple[int, ...]:
    entries = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not entries or len(set(entries)) != len(entries):
        raise argparse.ArgumentTypeError("expected distinct comma-separated integers")
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", help="registered adapter key; defaults to model metadata")
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--teacher-artifact", type=Path)
    parser.add_argument("--point-arrays", type=Path)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument(
        "--sampling-cluster-size",
        type=int,
        help="complete fibre-root cluster size; defaults to model metadata",
    )
    parser.add_argument("--seeds", type=parse_integer_list, default=(72601,))
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--metric-chunk-size", type=int, default=512)
    parser.add_argument("--trial-levels", type=parse_integer_list, default=(1, 2, 3))
    parser.add_argument("--eigenvalue-count", type=int, default=3)
    parser.add_argument("--mass-relative-threshold", type=float, default=1.0e-9)
    parser.add_argument("--cubic-feature-count", type=int, default=256)
    parser.add_argument("--cubic-feature-seed", type=int, default=314159)
    parser.add_argument("--cluster-jackknife-groups", type=int, default=0)
    parser.add_argument("--cluster-jackknife-level", type=int)
    parser.add_argument("--registered-relative-shift", type=float, default=0.02)
    parser.add_argument("--minimum-point-ess-per-rank", type=float, default=1.0)
    parser.add_argument("--minimum-fibre-ess-per-rank", type=float, default=1.0)
    parser.add_argument("--maximum-jackknife-relative-standard-error", type=float)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_delete_group_relative_shift(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Jackknife candidate/reference Ritz shifts using matched deleted fibres."""

    candidate_full = np.asarray(candidate["point_estimate"], dtype=float)
    reference_full = np.asarray(reference["point_estimate"], dtype=float)
    candidate_leave = np.asarray(candidate["leave_one_group_eigenvalues"], dtype=float)
    reference_leave = np.asarray(reference["leave_one_group_eigenvalues"], dtype=float)
    if (
        candidate_full.shape != reference_full.shape
        or candidate_leave.shape != reference_leave.shape
        or candidate_leave.ndim != 2
        or candidate_leave.shape[1] != candidate_full.size
    ):
        raise ValueError("paired Ritz jackknife arrays are not aligned")
    group_count = candidate_leave.shape[0]
    if group_count < 2 or int(candidate["group_count"]) != group_count:
        raise ValueError("candidate Ritz jackknife group count is invalid")
    if int(reference["group_count"]) != group_count:
        raise ValueError("paired Ritz jackknife group counts differ")
    point_shift = candidate_full / reference_full - 1.0
    leave_shift = candidate_leave / reference_leave - 1.0
    pseudo = group_count * point_shift[None, :] - (group_count - 1) * leave_shift
    bias_corrected = np.mean(pseudo, axis=0)
    standard_errors = np.std(pseudo, axis=0, ddof=1) / np.sqrt(group_count)
    intervals = np.column_stack(
        (
            bias_corrected - 1.96 * standard_errors,
            bias_corrected + 1.96 * standard_errors,
        )
    )
    return {
        "group_count": int(group_count),
        "point_relative_shifts": point_shift.tolist(),
        "bias_corrected_relative_shifts": bias_corrected.tolist(),
        "standard_errors": standard_errors.tolist(),
        "normal_95_percent_intervals": intervals.tolist(),
        "leave_one_group_relative_shifts": leave_shift.tolist(),
    }


def summarize_pairwise_shifts(
    dataset_rows: Sequence[dict[str, Any]],
    *,
    candidate_key: str,
    reference_key: str,
    level: int,
    registered_relative_shift: float,
) -> dict[str, Any]:
    per_dataset = []
    shifts = []
    for row in dataset_rows:
        candidate = row["metrics"][candidate_key]["trial_levels"][str(level)]
        reference = row["metrics"][reference_key]["trial_levels"][str(level)]
        candidate_values = np.asarray(candidate["eigenvalues"], dtype=float)
        reference_values = np.asarray(reference["eigenvalues"], dtype=float)
        if candidate_values.shape != reference_values.shape:
            raise ValueError("paired Ritz eigenvalue arrays differ in shape")
        relative = candidate_values / reference_values - 1.0
        shifts.append(relative)
        entry: dict[str, Any] = {
            "dataset": row["dataset"],
            "candidate_eigenvalues": candidate_values.tolist(),
            "reference_eigenvalues": reference_values.tolist(),
            "relative_shifts": relative.tolist(),
            "maximum_absolute_relative_shift": float(np.max(np.abs(relative))),
        }
        candidate_jackknife = candidate.get("cluster_delete_group_jackknife")
        reference_jackknife = reference.get("cluster_delete_group_jackknife")
        if candidate_jackknife is not None or reference_jackknife is not None:
            if candidate_jackknife is None or reference_jackknife is None:
                raise ValueError("paired Ritz jackknife is missing for one metric")
            entry["paired_cluster_delete_group_jackknife"] = (
                paired_delete_group_relative_shift(
                    candidate_jackknife,
                    reference_jackknife,
                )
            )
        per_dataset.append(entry)

    shift_array = np.asarray(shifts, dtype=float)
    means = np.mean(shift_array, axis=0)
    intervals = [
        confidence_interval(shift_array[:, index].tolist())
        for index in range(shift_array.shape[1])
    ]
    maximum_mean = float(np.max(np.abs(means)))
    maximum_dataset = float(np.max(np.abs(shift_array)))
    return {
        "candidate": candidate_key,
        "reference": reference_key,
        "trial_level": int(level),
        "per_dataset": per_dataset,
        "mean_relative_shifts": means.tolist(),
        "across_dataset_95_percent_intervals": intervals,
        "maximum_absolute_mean_relative_shift": maximum_mean,
        "maximum_absolute_per_dataset_relative_shift": maximum_dataset,
        "registered_relative_shift_threshold": float(registered_relative_shift),
        "registered_mean_shift_gate": bool(maximum_mean <= registered_relative_shift),
    }


def _load_replay_points(adapter, geometry, path: Path):
    arrays_path = path.expanduser().resolve()
    with np.load(arrays_path, allow_pickle=False) as payload:
        schema = str(np.asarray(payload["schema"]).item())
        if schema != "type11-positive-tensor-network-blind-arrays-v1":
            raise ValueError("unrecognized positive tensor-network point-array schema")
        saved_adapter = str(np.asarray(payload["adapter"]).item())
        if saved_adapter != adapter.key:
            raise ValueError("point-array adapter does not match the model adapter")
        metadata_keys = {
            "schema",
            "adapter",
            "model_log_eta",
            "importance_weights",
            "sampling_cluster_ids",
            "teacher_log_eta",
        }
        storage = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key not in metadata_keys
        }
        if not storage:
            raise ValueError("point-array artifact does not contain replayable points")
        saved_weights = np.asarray(payload["importance_weights"], dtype=float)
        saved_clusters = np.asarray(payload["sampling_cluster_ids"], dtype=np.int64)
    points = adapter.points_from_storage_payload(geometry, storage)
    recomputed_weights = np.asarray(adapter.importance_weights(points), dtype=float)
    recomputed_weights /= np.sum(recomputed_weights)
    if not np.allclose(recomputed_weights, saved_weights, rtol=5.0e-12, atol=5.0e-14):
        raise ValueError("replayed point importance weights do not match saved weights")
    cluster_ids = np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
    if not np.array_equal(cluster_ids, saved_clusters):
        raise ValueError("replayed fibre cluster ids do not match saved ids")
    return points, cluster_ids, {
        "kind": "replayed_blind_point_arrays",
        "path": str(arrays_path),
        "sha256": sha256_file(arrays_path),
        "points": int(len(points)),
    }


def _sample_fresh_points(
    adapter,
    geometry,
    *,
    count: int,
    seed: int,
    workers: int,
    cluster_size: int,
):
    backend = "thread" if workers == 1 else "process"
    points, shards = sample_points_parallel(
        adapter,
        model_seed=int(geometry.seed),
        exact_model=True,
        count=count,
        seed=seed,
        workers=workers,
        cluster_size=cluster_size,
        backend=backend,
    )
    cluster_ids = np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
    return points, cluster_ids, {
        "kind": "fresh_parallel_sample",
        "seed": int(seed),
        "points": int(len(points)),
        "shards": shards,
    }


def _section_arrays(adapter, points, exponents):
    rows = [adapter.section_values_and_jacobian(point, exponents) for point in points]
    return (
        np.asarray([row[0] for row in rows], dtype=np.complex128),
        np.asarray([row[1] for row in rows], dtype=np.complex128),
    )


def _evaluate_metric_triplet(
    tensor_model,
    source,
    teacher,
    adapter,
    points,
    *,
    device,
    complex_dtype,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    import torch

    source_values, source_derivatives = _section_arrays(
        adapter, points, source.section_exponents
    )
    teacher_values, teacher_derivatives = _section_arrays(
        adapter, points, teacher.section_exponents
    )
    source_h = torch.tensor(source.h_matrix, dtype=complex_dtype, device=device)
    teacher_h = torch.tensor(teacher.h_matrix, dtype=complex_dtype, device=device)
    rows: dict[str, list[np.ndarray]] = {key: [] for key in METRIC_KEYS}
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            source_v = torch.tensor(
                source_values[start:stop], dtype=complex_dtype, device=device
            )
            source_d = torch.tensor(
                source_derivatives[start:stop], dtype=complex_dtype, device=device
            )
            teacher_v = torch.tensor(
                teacher_values[start:stop], dtype=complex_dtype, device=device
            )
            teacher_d = torch.tensor(
                teacher_derivatives[start:stop], dtype=complex_dtype, device=device
            )
            _, source_metric = dense_h_potential_and_metric(
                source_v,
                source_d,
                source_h,
                float(source.normalization),
            )
            _, tensor_metric = tensor_model.potential_and_metric(source_v, source_d)
            _, teacher_metric = dense_h_potential_and_metric(
                teacher_v,
                teacher_d,
                teacher_h,
                float(teacher.normalization),
            )
            for key, metric in zip(
                METRIC_KEYS,
                (source_metric, tensor_metric, teacher_metric),
                strict=True,
            ):
                rows[key].append(metric.detach().cpu().numpy())
    return {
        key: np.concatenate(metric_rows, axis=0).astype(np.complex128, copy=False)
        for key, metric_rows in rows.items()
    }


def _trial_arrays(adapter, points, *, level: int, cubic_count: int, cubic_seed: int):
    rows = [
        adapter.scalar_laplacian_trial_data(
            point,
            level=level,
            cubic_feature_count=cubic_count,
            cubic_feature_seed=cubic_seed,
        )
        for point in points
    ]
    values = np.asarray([row.values for row in rows], dtype=float)
    gradients = np.asarray(
        [row.holomorphic_gradients for row in rows], dtype=np.complex128
    )
    linear_count = int(rows[0].linear_feature_count)
    return values, gradients, {
        1: linear_count,
        2: linear_count + linear_count * (linear_count + 1) // 2,
        3: int(values.shape[1]),
    }


def _aggregate_metric_level_rows(dataset_rows, metric_key: str, level: int):
    estimates = [row["metrics"][metric_key]["trial_levels"][str(level)] for row in dataset_rows]
    eigenvalues = np.asarray([row["eigenvalues"] for row in estimates], dtype=float)
    retained_ranks = sorted(
        {int(row["retained_trial_rank"]) for row in estimates}
    )
    minimum_point_ess = float(
        min(row["integration_effective_sample_size"] for row in estimates)
    )
    minimum_fibre_ess = float(
        min(row["integration_cluster_effective_sample_size"] for row in estimates)
    )
    output = {
        "mean_eigenvalues": np.mean(eigenvalues, axis=0).tolist(),
        "across_dataset_95_percent_intervals": [
            confidence_interval(eigenvalues[:, index].tolist())
            for index in range(eigenvalues.shape[1])
        ],
        "retained_trial_ranks": retained_ranks,
        "minimum_metric_eigenvalue": float(
            min(row["minimum_metric_eigenvalue"] for row in estimates)
        ),
        "minimum_point_effective_sample_size": minimum_point_ess,
        "minimum_fibre_effective_sample_size": minimum_fibre_ess,
        "minimum_point_ess_per_retained_rank": float(
            minimum_point_ess / max(retained_ranks)
        ),
        "minimum_fibre_ess_per_retained_rank": float(
            minimum_fibre_ess / max(retained_ranks)
        ),
    }
    jackknives = [row.get("cluster_delete_group_jackknife") for row in estimates]
    if any(value is not None for value in jackknives):
        if not all(value is not None for value in jackknives):
            raise ValueError("cluster jackknife is missing for one dataset")
        relative_errors = []
        for value in jackknives:
            estimates_array = np.asarray(
                value["bias_corrected_eigenvalues"], dtype=float
            )
            errors_array = np.asarray(value["standard_errors"], dtype=float)
            relative_errors.extend(
                (errors_array / np.maximum(np.abs(estimates_array), 1.0e-15)).tolist()
            )
        output["maximum_cluster_jackknife_relative_standard_error"] = float(
            max(relative_errors)
        )
    return output


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for tensor-network Ritz audit") from exc

    args = parse_args()
    levels = tuple(int(level) for level in args.trial_levels)
    if tuple(sorted(levels)) != levels or any(level not in (1, 2, 3) for level in levels):
        raise SystemExit("trial levels must be an increasing subset of 1,2,3")
    if (
        args.points <= 0
        or args.workers <= 0
        or args.metric_chunk_size <= 0
        or args.eigenvalue_count <= 0
        or args.registered_relative_shift <= 0
        or args.minimum_point_ess_per_rank <= 0
        or args.minimum_fibre_ess_per_rank <= 0
    ):
        raise SystemExit("invalid positive audit argument")
    if (
        args.maximum_jackknife_relative_standard_error is not None
        and args.maximum_jackknife_relative_standard_error <= 0
    ):
        raise SystemExit("jackknife relative standard-error gate must be positive")
    if args.point_arrays is not None and len(args.seeds) != 1:
        raise SystemExit("point-array replay represents exactly one dataset")
    selected_jackknife_level = (
        max(levels)
        if args.cluster_jackknife_level is None
        else int(args.cluster_jackknife_level)
    )
    if args.cluster_jackknife_groups and selected_jackknife_level not in levels:
        raise SystemExit("jackknife level must be one of the enabled trial levels")

    started = time.perf_counter()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    model_path = args.model.expanduser().resolve()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "type11-positive-tensor-network-v1":
        raise ValueError("unrecognized positive tensor-network artifact")
    adapter_key = args.adapter or payload.get(
        "adapter", "p4p1_type11_hirzebruch_x3"
    )
    model_seed = (
        args.model_seed
        if args.model_seed is not None
        else int(payload.get("model_seed", 20260731))
    )
    sampling_cluster_size = (
        args.sampling_cluster_size
        if args.sampling_cluster_size is not None
        else int(payload.get("sampling_cluster_size", 4))
    )
    if sampling_cluster_size <= 0 or args.points % sampling_cluster_size:
        raise SystemExit(
            "points must be divisible by the positive sampling-cluster-size"
        )
    source_path = (
        args.source_artifact.expanduser().resolve()
        if args.source_artifact is not None
        else Path(payload["source_artifact"]).expanduser().resolve()
    )
    teacher_value = (
        args.teacher_artifact
        if args.teacher_artifact is not None
        else payload.get("teacher_artifact")
    )
    if teacher_value is None:
        raise ValueError(
            "Ritz comparison requires --teacher-artifact when the model is teacher-free"
        )
    teacher_path = Path(teacher_value).expanduser().resolve()
    if sha256_file(source_path) != payload["source_artifact_sha256"]:
        raise ValueError("source artifact SHA256 does not match the saved model")
    saved_teacher_sha256 = payload.get("teacher_artifact_sha256")
    if (
        args.teacher_artifact is None
        and saved_teacher_sha256 is not None
        and sha256_file(teacher_path) != saved_teacher_sha256
    ):
        raise ValueError("teacher artifact SHA256 does not match the saved model")

    precision = str(payload["precision"])
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    adapter = get_adapter(adapter_key)
    geometry = adapter.make_model(model_seed, exact=True)
    source = adapter.load_h_artifact(source_path, geometry)
    teacher = adapter.load_h_artifact(teacher_path, geometry)
    tensor_model = positive_tensor_network_from_artifact_payload(
        source.h_matrix,
        payload,
        device=device,
    )
    tensor_model.eval()

    dataset_rows = []
    for dataset_index, seed in enumerate(args.seeds):
        dataset_started = time.perf_counter()
        if args.point_arrays is not None:
            points, cluster_ids, dataset = _load_replay_points(
                adapter, geometry, args.point_arrays
            )
            dataset["seed_label"] = int(seed)
        else:
            points, cluster_ids, dataset = _sample_fresh_points(
                adapter,
                geometry,
                count=args.points,
                seed=int(seed),
                workers=args.workers,
                cluster_size=sampling_cluster_size,
            )
        print(
            f"dataset={dataset_index} points={len(points)}: evaluating metrics",
            flush=True,
        )
        metrics = _evaluate_metric_triplet(
            tensor_model,
            source,
            teacher,
            adapter,
            points,
            device=device,
            complex_dtype=complex_dtype,
            chunk_size=args.metric_chunk_size,
        )
        print(
            f"dataset={dataset_index}: building level-{max(levels)} trial space",
            flush=True,
        )
        values, gradients, feature_counts = _trial_arrays(
            adapter,
            points,
            level=max(levels),
            cubic_count=args.cubic_feature_count,
            cubic_seed=args.cubic_feature_seed,
        )
        omega_weights = np.asarray(adapter.importance_weights(points), dtype=float)
        omega_weights /= np.sum(omega_weights)
        metric_reports = {}
        for key in METRIC_KEYS:
            print(f"dataset={dataset_index}: Ritz metric={key}", flush=True)
            metric_weights, _ = metric_volume_weights(adapter, points, metrics[key])
            trial_reports = {}
            for level in levels:
                count = feature_counts[level]
                estimate = estimate_scalar_laplacian_ritz(
                    values[:, :count],
                    gradients[:, :count],
                    metrics[key],
                    metric_weights,
                    eigenvalue_count=args.eigenvalue_count,
                    mass_relative_threshold=args.mass_relative_threshold,
                    cluster_ids=cluster_ids,
                )
                if (
                    args.cluster_jackknife_groups
                    and level == selected_jackknife_level
                ):
                    estimate["cluster_delete_group_jackknife"] = (
                        cluster_delete_group_jackknife_scalar_ritz(
                            values[:, :count],
                            gradients[:, :count],
                            metrics[key],
                            metric_weights,
                            cluster_ids,
                            eigenvalue_count=args.eigenvalue_count,
                            mass_relative_threshold=args.mass_relative_threshold,
                            group_count=args.cluster_jackknife_groups,
                        )
                    )
                trial_reports[str(level)] = estimate
            metric_reports[key] = {
                "metric_volume_effective_sample_size": float(
                    1.0 / np.sum(metric_weights**2)
                ),
                "trial_levels": trial_reports,
            }

        metric_distortions = {
            "low_degree_reference_to_tensor_network": paired_metric_distortion(
                metrics["low_degree_reference"],
                metrics["tensor_network"],
                omega_weights,
            ),
            "tensor_network_to_full_h_teacher": paired_metric_distortion(
                metrics["tensor_network"],
                metrics["full_h_teacher"],
                omega_weights,
            ),
            "low_degree_reference_to_full_h_teacher": paired_metric_distortion(
                metrics["low_degree_reference"],
                metrics["full_h_teacher"],
                omega_weights,
            ),
        }
        dataset_rows.append(
            {
                "dataset": dataset,
                "fibre_cluster_count": int(len(np.unique(cluster_ids))),
                "metrics": metric_reports,
                "metric_distortions": metric_distortions,
                "runtime_seconds": float(time.perf_counter() - dataset_started),
            }
        )

    summaries = {
        key: {
            "trial_levels": {
                str(level): _aggregate_metric_level_rows(dataset_rows, key, level)
                for level in levels
            }
        }
        for key in METRIC_KEYS
    }
    comparisons = []
    for candidate in ("low_degree_reference", "tensor_network"):
        for level in levels:
            comparisons.append(
                summarize_pairwise_shifts(
                    dataset_rows,
                    candidate_key=candidate,
                    reference_key="full_h_teacher",
                    level=level,
                    registered_relative_shift=args.registered_relative_shift,
                )
            )
    registered_comparison = next(
        row
        for row in comparisons
        if row["candidate"] == "tensor_network" and row["trial_level"] == max(levels)
    )
    finite_positive = all(
        np.all(np.asarray(estimate["eigenvalues"], dtype=float) > 0)
        for row in dataset_rows
        for key in METRIC_KEYS
        for estimate in row["metrics"][key]["trial_levels"].values()
    )
    stable_ranks = all(
        len(summaries[key]["trial_levels"][str(level)]["retained_trial_ranks"]) == 1
        for key in METRIC_KEYS
        for level in levels
    )
    complete_clusters = all(
        row["dataset"]["points"]
        == sampling_cluster_size * row["fibre_cluster_count"]
        for row in dataset_rows
    )
    nested_checks = []
    for row in dataset_rows:
        for key in METRIC_KEYS:
            for source_level, target_level in zip(levels[:-1], levels[1:]):
                source_values = np.asarray(
                    row["metrics"][key]["trial_levels"][str(source_level)][
                        "eigenvalues"
                    ],
                    dtype=float,
                )
                target_values = np.asarray(
                    row["metrics"][key]["trial_levels"][str(target_level)][
                        "eigenvalues"
                    ],
                    dtype=float,
                )
                tolerance = 1.0e-7 * np.maximum(1.0, np.abs(source_values))
                nested_checks.append(bool(np.all(target_values <= source_values + tolerance)))
    point_ess_gate = all(
        summaries[key]["trial_levels"][str(level)][
            "minimum_point_ess_per_retained_rank"
        ]
        >= args.minimum_point_ess_per_rank
        for key in METRIC_KEYS
        for level in levels
    )
    fibre_ess_gate = all(
        summaries[key]["trial_levels"][str(level)][
            "minimum_fibre_ess_per_retained_rank"
        ]
        >= args.minimum_fibre_ess_per_rank
        for key in METRIC_KEYS
        for level in levels
    )
    jackknife_gate = True
    if args.maximum_jackknife_relative_standard_error is not None:
        jackknife_gate = all(
            summaries[key]["trial_levels"][str(selected_jackknife_level)].get(
                "maximum_cluster_jackknife_relative_standard_error",
                float("inf"),
            )
            <= args.maximum_jackknife_relative_standard_error
            for key in METRIC_KEYS
        )
    gates = {
        "finite_positive_spectra": bool(finite_positive),
        "positive_metrics": bool(
            all(
                summaries[key]["trial_levels"][str(level)][
                    "minimum_metric_eigenvalue"
                ]
                > 0
                for key in METRIC_KEYS
                for level in levels
            )
        ),
        "stable_trial_rank_across_datasets": bool(stable_ranks),
        "nested_ritz_monotonicity": bool(all(nested_checks)),
        "complete_sampling_clusters": bool(complete_clusters),
        "point_ess_per_retained_rank": bool(point_ess_gate),
        "fibre_ess_per_retained_rank": bool(fibre_ess_gate),
        "cluster_jackknife_relative_standard_error": bool(jackknife_gate),
        "tensor_teacher_registered_ritz_shift": bool(
            registered_comparison["registered_mean_shift_gate"]
        ),
    }
    report = {
        "schema": "type11-positive-tensor-network-scalar-ritz-v1",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "source_artifact": str(source_path),
        "source_artifact_sha256": sha256_file(source_path),
        "teacher_artifact": str(teacher_path),
        "teacher_artifact_sha256": sha256_file(teacher_path),
        "adapter": adapter.key,
        "model_seed": int(model_seed),
        "sampling_cluster_size": int(sampling_cluster_size),
        "device": str(device),
        "precision": precision,
        "site_count": int(tensor_model.site_count),
        "bond_dimension": int(tensor_model.bond_dimension),
        "architecture": tensor_model.architecture,
        "physical_dictionary_rank": tensor_model.physical_dictionary_rank,
        "trainable_real_parameter_count": int(
            tensor_model.trainable_real_parameter_count
        ),
        "trial_levels": list(levels),
        "eigenvalue_count": int(args.eigenvalue_count),
        "mass_relative_threshold": float(args.mass_relative_threshold),
        "integration_measure": "metric_volume",
        "cubic_feature_count": int(args.cubic_feature_count),
        "cubic_feature_seed": int(args.cubic_feature_seed),
        "cluster_jackknife_groups": int(args.cluster_jackknife_groups),
        "cluster_jackknife_level": (
            int(selected_jackknife_level) if args.cluster_jackknife_groups else None
        ),
        "minimum_point_ess_per_retained_rank": float(
            args.minimum_point_ess_per_rank
        ),
        "minimum_fibre_ess_per_retained_rank": float(
            args.minimum_fibre_ess_per_rank
        ),
        "maximum_jackknife_relative_standard_error": (
            float(args.maximum_jackknife_relative_standard_error)
            if args.maximum_jackknife_relative_standard_error is not None
            else None
        ),
        "datasets": dataset_rows,
        "metric_summaries": summaries,
        "paired_ritz_comparisons": comparisons,
        "gates": gates,
        "success": bool(all(gates.values())),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(out_path)
    print(f"success={report['success']}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
