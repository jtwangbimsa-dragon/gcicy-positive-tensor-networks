#!/usr/bin/env python3
"""Train and tail-audit a cymetric-style PhiFS model on a registered gCICY.

The official cymetric point generator accepts polynomial CICY configurations,
not negative-degree generalized columns.  This script therefore keeps the
PhiFS ansatz and network scale, while using the repository's audited gCICY
samplers, tangent pullbacks, and holomorphic-volume density.  The neural
potential is made exactly projective invariant with normalized density-matrix
features on every ambient projective factor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.common_point_pool import (  # noqa: E402
    load_common_point_pool,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from gcicy_metric.pipeline.risk import (  # noqa: E402
    smooth_upper_log_ratio_excess_torch,
    weighted_cvar_numpy,
    weighted_cvar_torch,
)


RATIO_THRESHOLDS = (1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0)
LOW_RATIO_THRESHOLDS = (0.5, 0.2, 0.1)
ABS_LOG_THRESHOLDS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="registered gCICY adapter key")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-cache-dir",
        type=Path,
        help="shared deterministic dataset cache; defaults to output-dir",
    )
    parser.add_argument("--h-artifact", type=Path, required=True)
    parser.add_argument(
        "--base-metric",
        choices=("fubini_study", "h_artifact"),
        default="fubini_study",
        help="metric to which the learned ddbar(phi) correction is added",
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--train-points", type=int, default=90_000)
    parser.add_argument("--validation-points", type=int, default=10_000)
    parser.add_argument("--blind-points", type=int, default=200_000)
    parser.add_argument("--train-seed", type=int, default=72001)
    parser.add_argument("--train-common-pool", type=Path)
    parser.add_argument("--validation-seed", type=int, default=72002)
    parser.add_argument("--validation-common-pool", type=Path)
    parser.add_argument("--blind-seed", type=int, default=72003)
    parser.add_argument("--blind-common-pool", type=Path)
    parser.add_argument(
        "--blind-common-pool-split",
        choices=("confirmation", "blind"),
        default="blind",
    )
    parser.add_argument("--torch-seed", type=int, default=72004)
    parser.add_argument("--sampling-workers", type=int, default=6)
    parser.add_argument(
        "--sampling-cluster-size",
        type=int,
        required=True,
        help="number of complete fibre roots returned by one sampling cluster",
    )
    parser.add_argument("--sampling-backend", choices=("process", "thread"), default="process")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--activation", choices=("gelu", "tanh", "silu"), default="gelu")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience-evaluations", type=int, default=12)
    parser.add_argument(
        "--early-stopping-min-relative-improvement",
        type=float,
        default=0.0,
        help="minimum relative validation-score improvement that resets patience",
    )
    parser.add_argument("--potential-scale", type=float, default=0.1)
    parser.add_argument("--sigma-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--ma-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-fraction", type=float, default=0.02)
    parser.add_argument("--tail-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--tail-smooth-temperature", type=float, default=0.05)
    parser.add_argument("--positivity-weight", type=float, default=20.0)
    parser.add_argument("--relative-eigenvalue-floor", type=float, default=0.01)
    parser.add_argument("--correction-weight", type=float, default=1.0e-5)
    parser.add_argument("--top-points", type=int, default=64)
    parser.add_argument("--atlas-points", type=int, default=8)
    parser.add_argument("--h-eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--load-model", type=Path)
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="continue optimization after loading --load-model",
    )
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_status(output_dir: Path, stage: str, **extra: Any) -> None:
    payload = {
        "stage": stage,
        "updated_unix": time.time(),
        **extra,
    }
    (output_dir / "status.json").write_text(
        json.dumps(json_value(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def weighted_log_mean_exp(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        raise ValueError("weighted log mean requires finite positive-weight values")
    values = values[valid]
    weights = weights[valid]
    maximum = float(np.max(values))
    return float(
        maximum
        + np.log(np.sum(weights * np.exp(values - maximum)) / np.sum(weights))
    )


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: Sequence[float] = QUANTILES,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return {f"q{probability:.4f}": float("nan") for probability in probabilities}
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return {
        f"q{probability:.4f}": float(
            np.interp(probability, cumulative, values, left=values[0], right=values[-1])
        )
        for probability in probabilities
    }


def weighted_cvar(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)[::-1]
    values = values[order]
    weights = weights[order]
    target = (1.0 - probability) * float(np.sum(weights))
    if target <= 0:
        return float(values[0])
    remaining = target
    numerator = 0.0
    for value, weight in zip(values, weights, strict=True):
        used = min(float(weight), remaining)
        numerator += used * float(value)
        remaining -= used
        if remaining <= 0:
            break
    return numerator / target


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(weights) ** 2 / np.sum(weights**2))


def threshold_summary(
    mask: np.ndarray,
    weights: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, float | int]:
    mask = np.asarray(mask, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    return {
        "point_count": int(np.sum(mask)),
        "point_fraction": float(np.mean(mask)),
        "weighted_mass": float(np.sum(weights[mask]) / np.sum(weights)),
        "cluster_count": int(len(np.unique(cluster_ids[mask]))) if np.any(mask) else 0,
        "cluster_fraction": float(
            len(np.unique(cluster_ids[mask])) / len(np.unique(cluster_ids))
        )
        if len(cluster_ids)
        else float("nan"),
    }


def tail_statistics(
    raw_log_ratio: np.ndarray,
    weights: np.ndarray,
    min_eigenvalue: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    fixed_log_normalization: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    raw = np.asarray(raw_log_ratio, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    minimum = np.asarray(min_eigenvalue, dtype=np.float64)
    clusters = np.asarray(cluster_ids, dtype=np.int64)
    valid = np.isfinite(raw) & np.isfinite(minimum) & (minimum > 0)
    invalid = ~valid
    normalization = (
        weighted_log_mean_exp(raw[valid], weights[valid])
        if fixed_log_normalization is None
        else float(fixed_log_normalization)
    )
    ratio = np.full(len(raw), np.nan, dtype=np.float64)
    ratio[valid] = np.exp(np.clip(raw[valid] - normalization, -745.0, 709.0))
    valid_weights = weights[valid]
    valid_ratio = ratio[valid]
    residual = 1.0 - valid_ratio
    abs_log = np.abs(raw[valid] - normalization)
    valid_mass = float(np.sum(valid_weights) / np.sum(weights))
    stats: dict[str, Any] = {
        "normalization_log_kappa": normalization,
        "normalization_kappa": float(np.exp(normalization)),
        "normalization_mode": "fixed" if fixed_log_normalization is not None else "self",
        "sigma": float(np.sum(valid_weights * np.abs(residual)) / np.sum(valid_weights)),
        "chi_l2": float(
            np.sqrt(np.sum(valid_weights * residual**2) / np.sum(valid_weights))
        ),
        "centered_log_rms": float(
            np.sqrt(np.sum(valid_weights * (raw[valid] - normalization) ** 2) / np.sum(valid_weights))
        ),
        "positive_metric_weighted_mass": valid_mass,
        "nonpositive_or_nonfinite": threshold_summary(invalid, weights, clusters),
        "importance_effective_sample_size": effective_sample_size(weights),
        "ratio_weighted_quantiles": weighted_quantiles(valid_ratio, valid_weights),
        "ratio_unweighted_quantiles": {
            f"q{probability:.4f}": float(np.quantile(valid_ratio, probability))
            for probability in QUANTILES
        },
        "abs_log_ratio_weighted_quantiles": weighted_quantiles(abs_log, valid_weights),
        "abs_log_ratio_weighted_cvar": {
            f"cvar_{probability:.4f}": weighted_cvar(abs_log, valid_weights, probability)
            for probability in (0.99, 0.999, 0.9999)
        },
        "min_eigenvalue_weighted_quantiles": weighted_quantiles(minimum, weights),
        "ratio_upper_tails": {
            str(threshold): threshold_summary(ratio > threshold, weights, clusters)
            for threshold in RATIO_THRESHOLDS
        },
        "ratio_lower_tails": {
            str(threshold): threshold_summary(ratio < threshold, weights, clusters)
            for threshold in LOW_RATIO_THRESHOLDS
        },
        "abs_log_ratio_tails": {
            str(threshold): threshold_summary(
                valid & (np.abs(raw - normalization) > threshold), weights, clusters
            )
            for threshold in ABS_LOG_THRESHOLDS
        },
        "max_ratio": float(np.nanmax(ratio)),
        "min_ratio": float(np.nanmin(ratio)),
        "min_metric_eigenvalue": float(np.nanmin(minimum)),
    }
    return stats, ratio


def chart_identifier(chart: Sequence[int], homogeneous_sizes: Sequence[int]) -> int:
    identifier = 0
    for index, size in zip(chart, homogeneous_sizes, strict=True):
        if index < 0 or index >= size:
            raise ValueError("projective chart index is out of range")
        identifier = identifier * size + int(index)
    return identifier


def point_arrays(points: Sequence[Any], adapter: Any) -> dict[str, np.ndarray]:
    affine = np.asarray([point.affine_coordinates for point in points], dtype=np.complex128)
    tangent = np.asarray([point.tangent_basis for point in points], dtype=np.complex128)
    homogeneous_sizes = tuple(
        dimension + 1 for dimension in adapter.configuration.ambient_dimensions
    )
    charts = np.asarray(
        [
            chart_identifier(adapter.point_projective_chart(point), homogeneous_sizes)
            for point in points
        ],
        dtype=np.int64,
    )
    base_metrics = np.asarray(
        [adapter.baseline_metric(point) for point in points],
        dtype=np.complex128,
    )
    base_eigenvalues = np.linalg.eigvalsh(base_metrics)
    if np.any(base_eigenvalues <= 0) or not np.all(np.isfinite(base_eigenvalues)):
        raise FloatingPointError("baseline metric is not positive definite")
    log_determinants = np.sum(np.log(base_eigenvalues), axis=1)
    ma_errors = np.asarray(
        [
            adapter.monge_ampere_log_error(point, metric)
            for point, metric in zip(points, base_metrics, strict=True)
        ],
        dtype=np.float64,
    )
    log_omega = np.asarray(log_determinants - ma_errors, dtype=np.float64)
    storage_payload = {
        key: np.asarray(values)
        for key, values in adapter.point_storage_payload(points).items()
    }
    return {
        "coords": np.concatenate((affine.real, affine.imag), axis=1),
        "charts": charts,
        "projective_charts": np.asarray(
            [adapter.point_projective_chart(point) for point in points],
            dtype=np.int64,
        ),
        "independent_indices": np.asarray(
            [point.independent_indices for point in points],
            dtype=np.int64,
        ),
        "tangent": tangent,
        "base_metrics": base_metrics,
        "log_omega": log_omega,
        "point_storage_keys": np.asarray(tuple(storage_payload)),
        **storage_payload,
    }


def h_evaluation(
    adapter: Any,
    points: Sequence[Any],
    artifact: Any,
    log_omega: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_rows = []
    minimum_rows = []
    metric_rows = []
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        metrics = adapter.h_metrics(points[start:stop], artifact)
        metric_rows.append(np.asarray(metrics, dtype=np.complex128))
        eigenvalues = np.linalg.eigvalsh(metrics)
        minimum_rows.append(eigenvalues[:, 0])
        valid = eigenvalues[:, 0] > 0
        raw = np.full(stop - start, np.nan, dtype=np.float64)
        raw[valid] = np.sum(np.log(eigenvalues[valid]), axis=1) - log_omega[start:stop][valid]
        raw_rows.append(raw)
        if start == 0 or stop == len(points) or (start // batch_size) % 50 == 0:
            print(f"H audit {stop}/{len(points)}", flush=True)
    return (
        np.concatenate(raw_rows),
        np.concatenate(minimum_rows),
        np.concatenate(metric_rows),
    )


def dataset_cache_path(cache_dir: Path, label: str, seed: int, count: int) -> Path:
    return cache_dir / f"{label}_seed{seed}_n{count}.npz"


def prepare_dataset(
    *,
    output_dir: Path,
    data_cache_dir: Path,
    label: str,
    count: int,
    seed: int,
    adapter: Any,
    model: Any,
    model_seed: int,
    sampling_workers: int,
    sampling_backend: str,
    sampling_cluster_size: int,
    include_h: bool,
    h_artifact: Any,
    h_artifact_hash: str,
    h_eval_batch_size: int,
    force: bool,
    common_pool_path: Path | None = None,
    common_pool_split: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    del output_dir
    if common_pool_path is not None:
        resolved_pool = common_pool_path.expanduser().resolve()
        common_pool = load_common_point_pool(
            resolved_pool,
            adapter,
            model,
            expected_model_seed=model_seed,
            expected_exact_model=True,
            expected_split=common_pool_split,
        )
        if int(common_pool.metadata["point_count"]) != count:
            raise ValueError("common point pool count does not match request")
        if int(common_pool.metadata["sampling_seed"]) != seed:
            raise ValueError("common point pool seed does not match request")
        common_pool_hash = sha256_file(resolved_pool)
        cache_suffix = h_artifact_hash[:12] if include_h else "no_h"
        derived_path = (
            data_cache_dir
            / f"{label}_common_seed{seed}_n{count}_{cache_suffix}.npz"
        )
        expected_metadata = {
            "schema": "gcicy-phifs-common-pool-v1",
            "adapter": adapter.key,
            "model_seed": model_seed,
            "seed": seed,
            "count": count,
            "sampling_cluster_size": sampling_cluster_size,
            "include_h": include_h,
            "h_artifact_sha256": h_artifact_hash if include_h else None,
            "common_pool": str(resolved_pool),
            "common_pool_sha256": common_pool_hash,
            "common_pool_split": common_pool_split,
        }
        if derived_path.exists() and not force:
            with np.load(derived_path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
                if any(
                    metadata.get(key) != value
                    for key, value in expected_metadata.items()
                ):
                    raise ValueError(
                        f"cached common-pool dataset metadata mismatch: {derived_path}"
                    )
                arrays = {
                    key: np.asarray(payload[key])
                    for key in payload.files
                    if key != "metadata_json"
                }
            print(
                f"loaded cached frozen {common_pool_split} dataset from {derived_path}",
                flush=True,
            )
            return arrays, metadata, derived_path
        points = common_pool.points
        arrays = point_arrays(points, adapter)
        arrays["importance_weights"] = np.asarray(
            common_pool.importance_weights,
            dtype=np.float64,
        )
        arrays["cluster_ids"] = np.asarray(
            common_pool.sampling_cluster_ids,
            dtype=np.int64,
        )
        arrays["log_omega"] = np.asarray(
            common_pool.holomorphic_volume_log_density,
            dtype=np.float64,
        )
        if include_h:
            h_raw, h_minimum, h_metrics = h_evaluation(
                adapter,
                points,
                h_artifact,
                arrays["log_omega"],
                h_eval_batch_size,
            )
            arrays["h_raw_log_ratio"] = h_raw
            arrays["h_min_eigenvalue"] = h_minimum
            arrays["h_metrics"] = h_metrics
        metadata = {
            **expected_metadata,
            "importance_effective_sample_size": effective_sample_size(
                arrays["importance_weights"]
            ),
        }
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            derived_path,
            metadata_json=json.dumps(json_value(metadata)),
            **arrays,
        )
        print(
            f"cached frozen {common_pool_split} dataset at {derived_path}",
            flush=True,
        )
        return arrays, metadata, derived_path

    path = dataset_cache_path(data_cache_dir, label, seed, count)
    expected_metadata = {
        "schema": "gcicy-phifs-dataset-v1",
        "adapter": adapter.key,
        "adapter_version": adapter.version,
        "model_seed": model_seed,
        "seed": seed,
        "count": count,
        "sampling_cluster_size": sampling_cluster_size,
        "include_h": include_h,
        "h_artifact_sha256": h_artifact_hash if include_h else None,
    }
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                raise ValueError(f"cached dataset metadata mismatch: {path}")
            arrays = {key: np.asarray(payload[key]) for key in payload.files if key != "metadata_json"}
        print(f"loaded cached {label} dataset from {path}", flush=True)
        return arrays, metadata, path

    print(f"sampling {label}: seed={seed}, points={count}", flush=True)
    sample_start = time.perf_counter()
    if sampling_workers > 1:
        points, shards = sample_points_parallel(
            adapter,
            model_seed=model_seed,
            exact_model=True,
            count=count,
            seed=seed,
            workers=sampling_workers,
            cluster_size=sampling_cluster_size,
            backend=sampling_backend,
        )
    else:
        points = adapter.sample_points(model, count, seed=seed)
        shards = [{"worker": 0, "points": count, "seed": seed}]
    arrays = point_arrays(points, adapter)
    log_weights = np.asarray([adapter.importance_log_weight(point) for point in points], dtype=np.float64)
    weights = np.exp(log_weights - float(np.max(log_weights)))
    weights /= float(np.mean(weights))
    arrays["importance_weights"] = weights
    arrays["cluster_ids"] = adapter.sampling_cluster_ids(points)
    if include_h:
        h_raw, h_minimum, h_metrics = h_evaluation(
            adapter,
            points,
            h_artifact,
            arrays["log_omega"],
            h_eval_batch_size,
        )
        arrays["h_raw_log_ratio"] = h_raw
        arrays["h_min_eigenvalue"] = h_minimum
        arrays["h_metrics"] = h_metrics
    metadata = {
        **expected_metadata,
        "sampling_workers": sampling_workers,
        "sampling_backend": sampling_backend,
        "sampling_shards": shards,
        "sampling_seconds": time.perf_counter() - sample_start,
        "importance_effective_sample_size": effective_sample_size(weights),
        "complete_sampling_clusters": count // sampling_cluster_size,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata_json=json.dumps(json_value(metadata)), **arrays)
    print(f"wrote {path} ({path.stat().st_size / (1 << 20):.1f} MiB)", flush=True)
    return arrays, metadata, path


def model_parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def main() -> None:
    try:
        import torch
        from torch.func import hessian, vmap
    except ImportError as exc:
        raise SystemExit("PyTorch with torch.func is required") from exc

    args = parse_args()
    adapter = get_adapter(args.adapter)
    for name in ("train_points", "validation_points", "blind_points"):
        value = int(getattr(args, name))
        if value <= 0 or value % args.sampling_cluster_size:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be a positive multiple of "
                "sampling-cluster-size"
            )
    if min(args.batch_size, args.eval_batch_size, args.hidden_layers, args.hidden_width) <= 0:
        raise SystemExit("batch sizes and network dimensions must be positive")
    if args.epochs < 0 or args.eval_every <= 0:
        raise SystemExit("epochs must be non-negative and eval-every positive")
    if args.resume_training and args.load_model is None:
        raise SystemExit("--resume-training requires --load-model")
    if args.early_stopping_min_relative_improvement < 0:
        raise SystemExit("early-stopping minimum relative improvement cannot be negative")
    if min(
        args.sigma_loss_weight,
        args.log_energy_loss_weight,
        args.ma_loss_weight,
        args.tail_loss_weight,
    ) < 0:
        raise SystemExit("geometric loss weights cannot be negative")
    if (
        args.sigma_loss_weight
        + args.log_energy_loss_weight
        + args.ma_loss_weight
        + args.tail_loss_weight
        <= 0
    ):
        raise SystemExit("at least one geometric loss weight must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_cache_dir = (
        output_dir
        if args.data_cache_dir is None
        else args.data_cache_dir.expanduser().resolve()
    )
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    h_artifact_path = args.h_artifact.expanduser().resolve()
    if not h_artifact_path.exists():
        raise SystemExit(f"missing H artifact: {h_artifact_path}")
    h_artifact_hash = sha256_file(h_artifact_path)
    protocol_path = args.protocol.expanduser().resolve() if args.protocol is not None else None
    if protocol_path is not None and not protocol_path.exists():
        raise SystemExit(f"missing preregistered protocol: {protocol_path}")
    protocol_hash = sha256_file(protocol_path) if protocol_path is not None else None
    registered_arm: str | None = None
    if protocol_path is not None:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("schema") == "gcicy-residual-phi-parameter-matched-protocol-v1":
            geometry = protocol["geometry"]
            data = protocol["data"]
            network = protocol["network"]
            objective = protocol["objective"]
            optimization = protocol["optimization"]
            matching_arms = [
                name
                for name, arm in network["arms"].items()
                if int(arm["hidden_width"]) == args.hidden_width
            ]
            if len(matching_arms) != 1:
                raise SystemExit(
                    "network width does not identify exactly one preregistered arm: "
                    f"{matching_arms}"
                )
            registered_arm = matching_arms[0]
            matching_stages = [
                stage
                for stage in optimization["stages"]
                if math.isclose(
                    float(stage["learning_rate"]),
                    args.learning_rate,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
            ]
            if len(matching_stages) != 1:
                raise SystemExit(
                    "learning rate does not identify exactly one preregistered stage"
                )
            stage = matching_stages[0]
            declared = {
                "adapter": geometry["adapter"],
                "model_seed": geometry["model_seed"],
                "sampling_cluster_size": geometry["sampling_cluster_size"],
                "train_seed": data["train"]["seed"],
                "train_points": data["train"]["points"],
                "validation_seed": data["selection"]["seed"],
                "validation_points": data["selection"]["points"],
                "blind_seed": data["development_confirmation"]["seed"],
                "blind_points": data["development_confirmation"]["points"],
                "base_metric": "h_artifact",
                "hidden_layers": network["hidden_layers"],
                "hidden_width": network["arms"][registered_arm]["hidden_width"],
                "activation": network["activation"],
                "potential_scale": network["potential_scale"],
                "batch_size": optimization["batch_size"],
                "eval_batch_size": optimization["evaluation_batch_size"],
                "eval_every": optimization["evaluation_frequency_epochs"],
                "epochs": stage["maximum_epochs_per_invocation"],
                "patience_evaluations": stage["patience_evaluations"],
                "early_stopping_min_relative_improvement": stage[
                    "minimum_relative_selection_improvement"
                ],
                "dtype": optimization["training_precision"],
                "sigma_loss_weight": objective["sigma_weight"],
                "log_energy_loss_weight": objective["log_energy_weight"],
                "ma_loss_weight": objective["squared_ma_weight"],
                "tail_loss_weight": objective["upper_tail_weight"],
                "tail_fraction": objective["tail_fraction"],
                "tail_ratio_threshold": objective["tail_ratio_threshold"],
                "tail_smooth_temperature": objective[
                    "tail_smooth_temperature"
                ],
                "positivity_weight": objective["positivity_weight"],
                "relative_eigenvalue_floor": objective[
                    "relative_eigenvalue_floor"
                ],
                "correction_weight": objective["correction_weight"],
            }
            feature_count = sum(
                (dimension + 1) ** 2
                for dimension in adapter.configuration.ambient_dimensions
            )
            parameter_count = (
                (feature_count + 1) * args.hidden_width
                + (args.hidden_layers - 1)
                * (args.hidden_width + 1)
                * args.hidden_width
                + args.hidden_width
            )
            declared_count = network["arms"][registered_arm][
                "trainable_real_parameter_count"
            ]
            if parameter_count != declared_count:
                raise SystemExit(
                    f"preregistered parameter count mismatch for {registered_arm}: "
                    f"declared={declared_count}, computed={parameter_count}"
                )
        elif "common_data" in protocol and "common_training" in protocol:
            geometry = protocol["geometry"]
            data = protocol["common_data"]
            training = protocol["common_training"]
            declared = {
                "adapter": geometry["adapter"],
                "model_seed": geometry["model_seed"],
                "sampling_cluster_size": geometry["sampling_cluster_size"],
                "train_seed": data["train_seed"],
                "train_points": data["train_points"],
                "validation_seed": data["validation_seed"],
                "validation_points": data["validation_points"],
                "blind_seed": data["blind_seed"],
                "blind_points": data["blind_points"],
                "sampling_workers": data["sampling_workers"],
                "epochs": training["epochs"],
                "activation": training["activation"],
                "batch_size": training["batch_size"],
                "learning_rate": training["learning_rate"],
                "potential_scale": training["potential_scale"],
                "positivity_weight": training["positivity_weight"],
                "relative_eigenvalue_floor": training["relative_eigenvalue_floor"],
                "correction_weight": training["correction_weight"],
                "dtype": training["precision"],
            }
            matching_arms = [
                name
                for name, arm in protocol["arms"].items()
                if arm["hidden_layers"] == args.hidden_layers
                and arm["hidden_width"] == args.hidden_width
                and arm["torch_seed"] == args.torch_seed
            ]
            if len(matching_arms) != 1:
                raise SystemExit(
                    "network architecture and seed do not identify exactly one "
                    f"preregistered arm: {matching_arms}"
                )
            registered_arm = matching_arms[0]
            feature_count = sum(
                (dimension + 1) ** 2 for dimension in geometry["ambient_dimensions"]
            )
            parameter_count = (
                (feature_count + 1) * args.hidden_width
                + (args.hidden_layers - 1)
                * (args.hidden_width + 1)
                * args.hidden_width
                + args.hidden_width
            )
            declared_count = protocol["arms"][registered_arm][
                "trainable_real_parameter_count"
            ]
            if parameter_count != declared_count:
                raise SystemExit(
                    f"preregistered parameter count mismatch for {registered_arm}: "
                    f"declared={declared_count}, computed={parameter_count}"
                )
        else:
            declared = {
                "model_seed": protocol["geometry"]["model_seed"],
                "train_seed": protocol["data"]["train"]["seed"],
                "train_points": protocol["data"]["train"]["points"],
                "validation_seed": protocol["data"]["validation"]["seed"],
                "validation_points": protocol["data"]["validation"]["points"],
                "blind_seed": protocol["data"]["formal_blind"]["seed"],
                "blind_points": protocol["data"]["formal_blind"]["points"],
                "sampling_workers": protocol["data"]["sampling_workers"],
                "epochs": protocol["method"]["epochs"],
                "batch_size": protocol["method"]["batch_size"],
                "learning_rate": protocol["method"]["learning_rate"],
                "potential_scale": protocol["method"]["potential_scale"],
            }
        mismatches = {
            key: {"declared": value, "actual": getattr(args, key)}
            for key, value in declared.items()
            if getattr(args, key) != value
        }
        if mismatches:
            raise SystemExit(f"arguments do not match preregistered protocol: {mismatches}")
    write_status(output_dir, "initializing")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    real_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    complex_dtype = torch.complex128 if args.dtype == "float64" else torch.complex64
    torch.manual_seed(args.torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.torch_seed)

    model = adapter.make_model(args.model_seed, exact=True)
    h_artifact = adapter.load_h_artifact(h_artifact_path, model)
    h_eigenvalues = np.linalg.eigvalsh(h_artifact.h_matrix)
    h_condition = float(h_eigenvalues[-1] / h_eigenvalues[0])

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "hostname": platform.node(),
        "pid": os.getpid(),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(json_value(environment), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "configuration.json").write_text(
        json.dumps(json_value(vars(args)), indent=2) + "\n", encoding="utf-8"
    )

    total_start = time.perf_counter()
    write_status(output_dir, "sampling_train")
    train_np, train_metadata, train_path = prepare_dataset(
        output_dir=output_dir,
        data_cache_dir=data_cache_dir,
        label="train",
        count=args.train_points,
        seed=args.train_seed,
        adapter=adapter,
        model=model,
        model_seed=args.model_seed,
        sampling_workers=args.sampling_workers,
        sampling_backend=args.sampling_backend,
        sampling_cluster_size=args.sampling_cluster_size,
        include_h=args.base_metric == "h_artifact",
        h_artifact=h_artifact,
        h_artifact_hash=h_artifact_hash,
        h_eval_batch_size=args.h_eval_batch_size,
        force=args.force_data,
        common_pool_path=args.train_common_pool,
        common_pool_split="train",
    )
    write_status(output_dir, "sampling_validation")
    validation_np, validation_metadata, validation_path = prepare_dataset(
        output_dir=output_dir,
        data_cache_dir=data_cache_dir,
        label="validation",
        count=args.validation_points,
        seed=args.validation_seed,
        adapter=adapter,
        model=model,
        model_seed=args.model_seed,
        sampling_workers=args.sampling_workers,
        sampling_backend=args.sampling_backend,
        sampling_cluster_size=args.sampling_cluster_size,
        include_h=args.base_metric == "h_artifact",
        h_artifact=h_artifact,
        h_artifact_hash=h_artifact_hash,
        h_eval_batch_size=args.h_eval_batch_size,
        force=args.force_data,
        common_pool_path=args.validation_common_pool,
        common_pool_split="selection",
    )

    base_metric_array_key = (
        "h_metrics" if args.base_metric == "h_artifact" else "base_metrics"
    )

    def to_torch(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            "coords": torch.as_tensor(dataset["coords"], dtype=real_dtype, device=device),
            "charts": torch.as_tensor(dataset["charts"], dtype=torch.int64, device=device),
            "tangent": torch.as_tensor(dataset["tangent"], dtype=complex_dtype, device=device),
            "base_metrics": torch.as_tensor(
                dataset[base_metric_array_key], dtype=complex_dtype, device=device
            ),
            "log_omega": torch.as_tensor(dataset["log_omega"], dtype=real_dtype, device=device),
            "weights": torch.as_tensor(
                dataset["importance_weights"], dtype=real_dtype, device=device
            ),
        }

    train = to_torch(train_np)
    validation = to_torch(validation_np)

    activation_type: type[torch.nn.Module]
    if args.activation == "gelu":
        activation_type = torch.nn.GELU
    elif args.activation == "silu":
        activation_type = torch.nn.SiLU
    else:
        activation_type = torch.nn.Tanh

    affine_dimensions = tuple(adapter.configuration.ambient_dimensions)
    homogeneous_sizes = tuple(dimension + 1 for dimension in affine_dimensions)
    ambient_dimension = int(sum(affine_dimensions))
    feature_dimension = int(sum(size * size for size in homogeneous_sizes))
    projective_charts = tuple(
        sorted(
            adapter.projective_charts(),
            key=lambda chart: chart_identifier(chart, homogeneous_sizes),
        )
    )
    expected_identifiers = tuple(range(int(np.prod(homogeneous_sizes))))
    observed_identifiers = tuple(
        chart_identifier(chart, homogeneous_sizes) for chart in projective_charts
    )
    if observed_identifiers != expected_identifiers:
        raise ValueError("adapter projective charts do not form the full ambient atlas")

    class ProjectiveInvariantPhi(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = []
            width = feature_dimension
            for _ in range(args.hidden_layers):
                layers.append(torch.nn.Linear(width, args.hidden_width))
                layers.append(activation_type())
                width = args.hidden_width
            layers.append(torch.nn.Linear(width, 1, bias=False))
            self.net = torch.nn.Sequential(*layers)
            torch.nn.init.zeros_(self.net[-1].weight)

        def forward(self, features: Any) -> Any:
            return self.net(features).squeeze(-1)

    potential = ProjectiveInvariantPhi().to(device=device, dtype=real_dtype)

    def homogeneous_block(
        active_real: Any,
        active_imag: Any,
        size: int,
        fixed_index: int,
    ) -> tuple[Any, Any]:
        real_parts = []
        imag_parts = []
        cursor = 0
        for index in range(size):
            if index == fixed_index:
                real_parts.append(torch.ones_like(active_real[0]))
                imag_parts.append(torch.zeros_like(active_imag[0]))
            else:
                real_parts.append(active_real[cursor])
                imag_parts.append(active_imag[cursor])
                cursor += 1
        return torch.stack(real_parts), torch.stack(imag_parts)

    def density_features(real: Any, imag: Any) -> Any:
        denominator = torch.clamp(torch.sum(real**2 + imag**2), min=1.0e-12)
        features = [(real[index] ** 2 + imag[index] ** 2) / denominator for index in range(len(real))]
        for left in range(len(real)):
            for right in range(left + 1, len(real)):
                features.append(
                    (real[left] * real[right] + imag[left] * imag[right]) / denominator
                )
                features.append(
                    (imag[left] * real[right] - real[left] * imag[right]) / denominator
                )
        return torch.stack(features)

    def projective_features(coords: Any, identifier: int) -> Any:
        chart = projective_charts[identifier]
        real = coords[:ambient_dimension]
        imag = coords[ambient_dimension:]
        blocks = []
        cursor = 0
        for dimension, size, fixed_index in zip(
            affine_dimensions,
            homogeneous_sizes,
            chart,
            strict=True,
        ):
            stop = cursor + dimension
            homogeneous_real, homogeneous_imag = homogeneous_block(
                real[cursor:stop],
                imag[cursor:stop],
                size,
                fixed_index,
            )
            blocks.append(density_features(homogeneous_real, homogeneous_imag))
            cursor = stop
        return torch.cat(blocks)

    def make_single_potential(identifier: int):
        def single(coords: Any) -> Any:
            return args.potential_scale * potential(projective_features(coords, identifier))

        return single

    hessian_functions = [
        vmap(hessian(make_single_potential(identifier)))
        for identifier in range(len(projective_charts))
    ]

    def correction_batch(coords: Any, charts: Any, tangent: Any) -> Any:
        ambient = torch.zeros(
            (len(coords), ambient_dimension, ambient_dimension),
            dtype=complex_dtype,
            device=device,
        )
        for identifier, batched_hessian in enumerate(hessian_functions):
            indices = torch.nonzero(charts == identifier, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            real_hessian = batched_hessian(coords[indices])
            h_xx = real_hessian[:, :ambient_dimension, :ambient_dimension]
            h_xy = real_hessian[:, :ambient_dimension, ambient_dimension:]
            h_yx = real_hessian[:, ambient_dimension:, :ambient_dimension]
            h_yy = real_hessian[:, ambient_dimension:, ambient_dimension:]
            # Repository metrics use rows anti-holomorphic and columns holomorphic.
            complex_hessian = 0.25 * (h_xx + h_yy).to(complex_dtype)
            complex_hessian -= 0.25j * (h_xy - h_yx).to(complex_dtype)
            ambient[indices] = complex_hessian
        pulled_back = torch.einsum(
            "nai,nab,nbj->nij", torch.conj(tangent), ambient, tangent
        )
        return 0.5 * (pulled_back + torch.conj(torch.transpose(pulled_back, 1, 2)))

    def batch_metrics(dataset: dict[str, Any], indices: Any) -> tuple[Any, Any, Any, Any]:
        correction = correction_batch(
            dataset["coords"][indices], dataset["charts"][indices], dataset["tangent"][indices]
        )
        metric = dataset["base_metrics"][indices] + correction
        metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        eigenvalues = torch.linalg.eigvalsh(metric)
        safe_eigenvalues = torch.clamp(eigenvalues, min=1.0e-12)
        raw = torch.sum(torch.log(safe_eigenvalues), dim=1) - dataset["log_omega"][indices]
        return metric, correction, eigenvalues, raw

    def evaluate_phi(dataset: dict[str, Any], count: int) -> tuple[np.ndarray, np.ndarray]:
        raw_rows = []
        minimum_rows = []
        potential.eval()
        for start in range(0, count, args.eval_batch_size):
            indices = torch.arange(start, min(start + args.eval_batch_size, count), device=device)
            with torch.enable_grad():
                _, _, eigenvalues, raw = batch_metrics(dataset, indices)
            raw_array = raw.detach().cpu().numpy().astype(np.float64)
            minimum_array = eigenvalues[:, 0].detach().cpu().numpy().astype(np.float64)
            raw_array[minimum_array <= 0] = np.nan
            raw_rows.append(raw_array)
            minimum_rows.append(minimum_array)
        return np.concatenate(raw_rows), np.concatenate(minimum_rows)

    def metric_array_evaluation(
        dataset_np: dict[str, np.ndarray],
        array_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues = np.linalg.eigvalsh(dataset_np[array_key])
        raw = np.sum(np.log(eigenvalues), axis=1) - dataset_np["log_omega"]
        return raw, eigenvalues[:, 0]

    def baseline_evaluation(dataset_np: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return metric_array_evaluation(dataset_np, base_metric_array_key)

    train_base_raw, train_base_minimum = baseline_evaluation(train_np)
    validation_base_raw, validation_base_minimum = baseline_evaluation(validation_np)
    fixed_log_kappa = weighted_log_mean_exp(
        train_base_raw,
        train_np["importance_weights"],
    )
    validation_clusters = validation_np["cluster_ids"]
    baseline_validation_fixed, _ = tail_statistics(
        validation_base_raw,
        validation_np["importance_weights"],
        validation_base_minimum,
        validation_clusters,
        fixed_log_normalization=fixed_log_kappa,
    )

    def geometric_selection_score(raw: np.ndarray, weights: np.ndarray) -> float:
        log_ratio = np.asarray(raw, dtype=np.float64) - fixed_log_kappa
        probabilities = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(log_ratio) & np.isfinite(probabilities) & (probabilities > 0)
        if not np.all(valid):
            return float("inf")
        probabilities = probabilities / np.sum(probabilities)
        ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))
        sigma = float(np.sum(probabilities * np.abs(1.0 - ratio)))
        log_energy = float(np.sum(probabilities * log_ratio**2))
        ma = float(np.sum(probabilities * (ratio - 1.0) ** 2))
        log_threshold = float(np.log(args.tail_ratio_threshold))
        upper_excess = (
            np.logaddexp(
                0.0,
                (log_ratio - log_threshold) / args.tail_smooth_temperature,
            )
            * args.tail_smooth_temperature
        )
        tail = weighted_cvar_numpy(
            upper_excess**2,
            probabilities,
            tail_fraction=args.tail_fraction,
        ).value
        return float(
            args.sigma_loss_weight * sigma
            + args.log_energy_loss_weight * log_energy
            + args.ma_loss_weight * ma
            + args.tail_loss_weight * tail
        )

    loaded_model = None
    if args.load_model is not None:
        loaded_path = args.load_model.expanduser().resolve()
        loaded_model = torch.load(loaded_path, map_location="cpu")
        potential.load_state_dict(loaded_model["state_dict"])
    else:
        loaded_path = None

    initial_indices = torch.arange(min(args.batch_size, args.train_points), device=device)
    with torch.enable_grad():
        initial_metric, initial_correction, _, _ = batch_metrics(train, initial_indices)
    initial_correction_max = float(torch.max(torch.abs(initial_correction)).detach().cpu())
    initial_base_metric_error = float(
        torch.max(
            torch.abs(initial_metric - train["base_metrics"][initial_indices])
        )
        .detach()
        .cpu()
    )
    if args.load_model is None and max(
        initial_correction_max,
        initial_base_metric_error,
    ) > 1.0e-7:
        raise RuntimeError(
            "zero-output initialization did not reproduce the selected base metric"
        )

    optimizer = torch.optim.Adam(potential.parameters(), lr=args.learning_rate)
    if loaded_model is None:
        best_state = copy.deepcopy(potential.state_dict())
        best_epoch = 0
        best_validation = baseline_validation_fixed
        best_score = geometric_selection_score(
            validation_base_raw,
            validation_np["importance_weights"],
        )
    else:
        loaded_validation_raw, loaded_validation_minimum = evaluate_phi(
            validation,
            args.validation_points,
        )
        best_validation, _ = tail_statistics(
            loaded_validation_raw,
            validation_np["importance_weights"],
            loaded_validation_minimum,
            validation_clusters,
            fixed_log_normalization=fixed_log_kappa,
        )
        best_score = geometric_selection_score(
            loaded_validation_raw,
            validation_np["importance_weights"],
        )
        best_state = copy.deepcopy(potential.state_dict())
        best_epoch = int(loaded_model.get("best_epoch", 0))
    history: list[dict[str, Any]] = []
    stale = 0
    patience_reference_score = best_score
    checkpoint_path = output_dir / "phi_model.pt"

    def save_best_checkpoint() -> None:
        temporary_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "fixed_log_kappa": fixed_log_kappa,
                "network": {
                    "input_dimension": feature_dimension,
                    "hidden_layers": args.hidden_layers,
                    "hidden_width": args.hidden_width,
                    "activation": args.activation,
                    "potential_scale": args.potential_scale,
                    "parameter_count": model_parameter_count(potential),
                },
                "adapter": adapter.key,
                "model_seed": args.model_seed,
                "h_artifact_sha256": h_artifact_hash,
            },
            temporary_path,
        )
        temporary_path.replace(checkpoint_path)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    optimization_start = time.perf_counter()
    write_status(
        output_dir,
        "training",
        fixed_log_kappa=fixed_log_kappa,
        baseline_validation_sigma=baseline_validation_fixed["sigma"],
        baseline_validation_selection_score=best_score,
    )
    print(
        f"device={device}, dtype={args.dtype}, parameters={model_parameter_count(potential)}, "
        f"train={args.train_points}, validation={args.validation_points}, "
        f"{args.base_metric} validation sigma={baseline_validation_fixed['sigma']:.6e}, "
        f"selection_score={best_score:.6e}",
        flush=True,
    )

    if args.load_model is None or args.resume_training:
        for epoch in range(1, args.epochs + 1):
            potential.train()
            permutation = torch.randperm(args.train_points, device=device)
            epoch_sums = {
                "loss": 0.0,
                "sigma": 0.0,
                "log_energy": 0.0,
                "ma": 0.0,
                "tail": 0.0,
                "barrier": 0.0,
                "correction": 0.0,
            }
            batch_count = 0
            for start in range(0, args.train_points, args.batch_size):
                indices = permutation[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                metric, correction, eigenvalues, raw = batch_metrics(train, indices)
                del metric
                weights = train["weights"][indices]
                weight_sum = torch.sum(weights)
                ratio = torch.exp(torch.clamp(raw - fixed_log_kappa, min=-15.0, max=10.0))
                sigma_loss = torch.sum(weights * torch.abs(1.0 - ratio)) / weight_sum
                log_ratio = raw - fixed_log_kappa
                log_energy_loss = torch.sum(weights * log_ratio**2) / weight_sum
                ma_loss = torch.sum(weights * (ratio - 1.0) ** 2) / weight_sum
                upper_excess = smooth_upper_log_ratio_excess_torch(
                    log_ratio,
                    ratio_threshold=args.tail_ratio_threshold,
                    smooth_temperature=args.tail_smooth_temperature,
                )
                tail_loss = weighted_cvar_torch(
                    upper_excess**2,
                    weights,
                    tail_fraction=args.tail_fraction,
                )
                base_eigenvalues = torch.linalg.eigvalsh(train["base_metrics"][indices])
                eigen_scale = torch.clamp(torch.mean(base_eigenvalues, dim=1), min=1.0e-12)
                relative = eigenvalues / eigen_scale[:, None]
                barrier = torch.nn.functional.softplus(
                    (args.relative_eigenvalue_floor - relative) * 40.0
                ).mean() / 40.0
                correction_size = torch.mean(torch.abs(correction) ** 2) / torch.clamp(
                    torch.mean(torch.abs(train["base_metrics"][indices]) ** 2), min=1.0e-12
                )
                loss = (
                    args.sigma_loss_weight * sigma_loss
                    + args.log_energy_loss_weight * log_energy_loss
                    + args.ma_loss_weight * ma_loss
                    + args.tail_loss_weight * tail_loss
                    + args.positivity_weight * barrier
                    + args.correction_weight * correction_size
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(potential.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_sums["loss"] += float(loss.detach().cpu())
                epoch_sums["sigma"] += float(sigma_loss.detach().cpu())
                epoch_sums["log_energy"] += float(log_energy_loss.detach().cpu())
                epoch_sums["ma"] += float(ma_loss.detach().cpu())
                epoch_sums["tail"] += float(tail_loss.detach().cpu())
                epoch_sums["barrier"] += float(barrier.detach().cpu())
                epoch_sums["correction"] += float(correction_size.detach().cpu())
                batch_count += 1

            if epoch % args.eval_every:
                continue
            validation_raw, validation_minimum = evaluate_phi(validation, args.validation_points)
            validation_fixed, _ = tail_statistics(
                validation_raw,
                validation_np["importance_weights"],
                validation_minimum,
                validation_clusters,
                fixed_log_normalization=fixed_log_kappa,
            )
            score = geometric_selection_score(
                validation_raw,
                validation_np["importance_weights"],
            )
            positive = validation_fixed["nonpositive_or_nonfinite"]["point_count"] == 0
            accepted = bool(np.isfinite(score) and positive and score < best_score)
            if accepted:
                best_score = score
                best_epoch = epoch
                best_validation = validation_fixed
                best_state = copy.deepcopy(potential.state_dict())
                save_best_checkpoint()
            material_improvement = bool(
                np.isfinite(score)
                and positive
                and score
                < patience_reference_score
                * (1.0 - args.early_stopping_min_relative_improvement)
            )
            if material_improvement:
                patience_reference_score = score
                stale = 0
            else:
                stale += 1
            row = {
                "epoch": epoch,
                "accepted": accepted,
                "train_loss": epoch_sums["loss"] / batch_count,
                "train_sigma_loss": epoch_sums["sigma"] / batch_count,
                "train_log_energy_loss": epoch_sums["log_energy"] / batch_count,
                "train_ma_loss": epoch_sums["ma"] / batch_count,
                "train_tail_loss": epoch_sums["tail"] / batch_count,
                "train_barrier": epoch_sums["barrier"] / batch_count,
                "train_correction_size": epoch_sums["correction"] / batch_count,
                "validation_selection_score": score,
                "validation_sigma_fixed_kappa": validation_fixed["sigma"],
                "validation_chi_fixed_kappa": validation_fixed["chi_l2"],
                "validation_max_ratio_fixed_kappa": validation_fixed["max_ratio"],
                "validation_min_eigenvalue": validation_fixed["min_metric_eigenvalue"],
                "material_improvement": material_improvement,
                "patience_reference_score": patience_reference_score,
                "stale_evaluations": stale,
            }
            history.append(row)
            print(
                f"epoch {epoch}: train_sigma={row['train_sigma_loss']:.6e}, "
                f"val_sigma={row['validation_sigma_fixed_kappa']:.6e}, "
                f"selection_score={score:.6e}, "
                f"val_chi={row['validation_chi_fixed_kappa']:.6e}, "
                f"max_r={row['validation_max_ratio_fixed_kappa']:.3e}, "
                f"min_eig={row['validation_min_eigenvalue']:.3e}, "
                f"accepted={accepted}, material={material_improvement}",
                flush=True,
            )
            write_status(
                output_dir,
                "training",
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_sigma=best_validation["sigma"],
                best_validation_selection_score=best_score,
                stale_evaluations=stale,
            )
            if args.patience_evaluations > 0 and stale >= args.patience_evaluations:
                print(f"early stopping after {stale} stale evaluations", flush=True)
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimization_seconds = time.perf_counter() - optimization_start
    training_device_memory = (
        {
            "maximum_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "maximum_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        }
        if device.type == "cuda"
        else None
    )
    potential.load_state_dict(best_state)
    save_best_checkpoint()

    write_status(output_dir, "sampling_and_auditing_blind", best_epoch=best_epoch)
    blind_np, blind_metadata, blind_path = prepare_dataset(
        output_dir=output_dir,
        data_cache_dir=data_cache_dir,
        label="blind",
        count=args.blind_points,
        seed=args.blind_seed,
        adapter=adapter,
        model=model,
        model_seed=args.model_seed,
        sampling_workers=args.sampling_workers,
        sampling_backend=args.sampling_backend,
        sampling_cluster_size=args.sampling_cluster_size,
        include_h=True,
        h_artifact=h_artifact,
        h_artifact_hash=h_artifact_hash,
        h_eval_batch_size=args.h_eval_batch_size,
        force=args.force_data,
        common_pool_path=args.blind_common_pool,
        common_pool_split=(
            args.blind_common_pool_split
            if args.blind_common_pool is not None
            else None
        ),
    )
    blind = to_torch(blind_np)
    blind_eval_start = time.perf_counter()
    blind_fs_raw, blind_fs_minimum = metric_array_evaluation(
        blind_np,
        "base_metrics",
    )
    blind_phi_raw, blind_phi_minimum = evaluate_phi(blind, args.blind_points)
    weights = blind_np["importance_weights"]
    clusters = blind_np["cluster_ids"]
    fs_self, fs_ratio = tail_statistics(
        blind_fs_raw, weights, blind_fs_minimum, clusters
    )
    fs_fixed, _ = tail_statistics(
        blind_fs_raw,
        weights,
        blind_fs_minimum,
        clusters,
        fixed_log_normalization=fixed_log_kappa,
    )
    phi_self, phi_ratio = tail_statistics(
        blind_phi_raw, weights, blind_phi_minimum, clusters
    )
    phi_fixed, _ = tail_statistics(
        blind_phi_raw,
        weights,
        blind_phi_minimum,
        clusters,
        fixed_log_normalization=fixed_log_kappa,
    )
    h_self, h_ratio = tail_statistics(
        blind_np["h_raw_log_ratio"],
        weights,
        blind_np["h_min_eigenvalue"],
        clusters,
    )
    blind_eval_seconds = time.perf_counter() - blind_eval_start

    valid_phi = np.isfinite(phi_ratio)
    ranking = np.argsort(np.where(valid_phi, phi_ratio, -np.inf))[::-1]
    top_indices = ranking[: min(args.top_points, len(ranking))]
    point_storage_keys = tuple(
        str(value) for value in np.asarray(blind_np["point_storage_keys"]).tolist()
    )

    def serializable_point_value(value: np.ndarray) -> Any:
        array = np.asarray(value)
        if np.iscomplexobj(array):
            return np.stack((array.real, array.imag), axis=-1).tolist()
        return array.tolist()

    top_rows = []
    for index in top_indices:
        row = {
            "index": int(index),
            "cluster_id": int(clusters[index]),
            "chart_id": int(blind_np["charts"][index]),
            "importance_weight": float(weights[index]),
            "fs_ratio": float(fs_ratio[index]),
            "phi_ratio": float(phi_ratio[index]),
            "h_ratio": float(h_ratio[index]),
            "phi_min_eigenvalue": float(blind_phi_minimum[index]),
            "h_min_eigenvalue": float(blind_np["h_min_eigenvalue"][index]),
            "point_storage": {
                key: serializable_point_value(blind_np[key][index])
                for key in point_storage_keys
            },
        }
        top_rows.append(row)

    atlas_start = time.perf_counter()
    atlas_witnesses = []
    max_atlas_spread = 0.0
    for index in top_indices[: args.atlas_points]:
        payload = {
            key: blind_np[key][[index]]
            for key in point_storage_keys
        }
        original = adapter.points_from_storage_payload(model, payload)[0]
        values = []
        charts_seen = []
        for chart in adapter.projective_charts():
            if not adapter.chart_is_available(original, chart, minimum=1.0e-5):
                continue
            candidate = adapter.rechart_point(model, original, chart)
            candidate_np = point_arrays([candidate], adapter)
            candidate_np["importance_weights"] = np.ones(1, dtype=np.float64)
            if args.base_metric == "h_artifact":
                candidate_np["h_metrics"] = adapter.h_metrics(
                    [candidate],
                    h_artifact,
                )
            candidate_torch = to_torch(candidate_np)
            candidate_raw, candidate_minimum = evaluate_phi(candidate_torch, 1)
            if candidate_minimum[0] > 0 and np.isfinite(candidate_raw[0]):
                values.append(float(candidate_raw[0]))
                charts_seen.append(list(chart))
        spread = float(np.max(values) - np.min(values)) if values else float("nan")
        if np.isfinite(spread):
            max_atlas_spread = max(max_atlas_spread, spread)
        atlas_witnesses.append(
            {
                "blind_index": int(index),
                "charts_seen": charts_seen,
                "raw_log_ratio_values": values,
                "max_minus_min": spread,
            }
        )
    atlas_seconds = time.perf_counter() - atlas_start

    tail_arrays_path = output_dir / "blind_tail_arrays.npz"
    tail_payload = {
        "schema": np.asarray("gcicy-common-residual-phi-point-arrays-v1"),
        "adapter": np.asarray(adapter.key),
        "common_pool_sha256": np.asarray(
            blind_metadata.get("common_pool_sha256") or ""
        ),
        "model_log_eta": blind_phi_raw,
        "importance_weights": weights,
        "sampling_cluster_ids": clusters,
        "metric_minimum_eigenvalues": blind_phi_minimum,
        "cluster_ids": clusters,
        "chart_ids": blind_np["charts"],
        "fs_raw_log_ratio": blind_fs_raw,
        "phi_raw_log_ratio": blind_phi_raw,
        "h_raw_log_ratio": blind_np["h_raw_log_ratio"],
        "fs_normalized_ratio": fs_ratio,
        "phi_normalized_ratio": phi_ratio,
        "h_normalized_ratio": h_ratio,
        "fs_min_eigenvalue": blind_fs_minimum,
        "phi_min_eigenvalue": blind_phi_minimum,
        "h_min_eigenvalue": blind_np["h_min_eigenvalue"],
        "top_phi_indices": top_indices,
    }
    tail_payload.update(
        {key: blind_np[key] for key in point_storage_keys}
    )
    np.savez_compressed(tail_arrays_path, **tail_payload)

    phi_severe = bool(phi_self["max_ratio"] > 10.0)
    h_severe = bool(h_self["max_ratio"] > 10.0)
    comparison = {
        "paired_same_blind_points": True,
        "h_has_ratio_above_10": h_severe,
        "phi_has_ratio_above_10": phi_severe,
        "observational_interpretation": (
            "The same gCICY blind sample has a severe H tail but no severe PhiFS tail; "
            "this is evidence against the spike being forced by the gCICY structure."
            if h_severe and not phi_severe
            else "Both ansatzes show a ratio above 10 on this blind sample; the control is inconclusive and the shared geometry/sampler must be investigated."
            if h_severe and phi_severe
            else "PhiFS has a ratio above 10 while the H metric does not on this blind sample; "
            "the PhiFS fit is not yet a successful structural control."
            if phi_severe
            else "No ratio above 10 was observed for either ansatz on this blind sample."
        ),
        "claim_limit": (
            "Finite blind sampling detects observed tails but does not prove a global sup-norm bound."
        ),
    }

    report = {
        "schema": "gcicy-cymetric-style-phifs-tail-audit-v1",
        "scientific_scope": {
            "geometry": adapter.configuration.name,
            "ambient_dimensions": list(adapter.configuration.ambient_dimensions),
            "positive_columns": [
                list(column) for column in adapter.configuration.positive_columns
            ],
            "generalized_columns": [
                list(column) for column in adapter.configuration.generalized_columns
            ],
            "adapter": adapter.key,
            "model_seed": args.model_seed,
            "method": (
                "g = g_H + partial partialbar phi_theta"
                if args.base_metric == "h_artifact"
                else "g = g_FS + partial partialbar phi_theta"
            ),
            "implementation_boundary": (
                "Official cymetric does not natively generate points for negative-degree generalized columns. "
                "This run adapts the PhiFS ansatz and specified dense network to the audited gCICY geometry backend."
            ),
            "globality": (
                f"phi_theta uses {feature_dimension} real entries of normalized rank-one density matrices "
                "on all ambient projective factors, "
                "so projective invariance is exact rather than learned by a transition loss."
            ),
        },
        "preregistered_protocol": {
            "path": str(protocol_path) if protocol_path is not None else None,
            "sha256": protocol_hash,
            "registered_arm": registered_arm,
        },
        "configuration": json_value(vars(args)),
        "environment": environment,
        "network": {
            "input_dimension": feature_dimension,
            "hidden_layers": args.hidden_layers,
            "hidden_width": args.hidden_width,
            "activation": args.activation,
            "output_bias": False,
            "potential_scale": args.potential_scale,
            "parameter_count": model_parameter_count(potential),
            "initial_correction_max_abs": initial_correction_max,
            "initial_base_metric_max_abs_error": initial_base_metric_error,
        },
        "training": {
            "base_metric": args.base_metric,
            "fixed_log_kappa_from_training_base": fixed_log_kappa,
            "fixed_kappa_from_training_base": float(np.exp(fixed_log_kappa)),
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "baseline_validation_fixed_kappa": baseline_validation_fixed,
            "best_validation_fixed_kappa": best_validation,
            "early_stopping": {
                "patience_evaluations": args.patience_evaluations,
                "minimum_relative_improvement": (
                    args.early_stopping_min_relative_improvement
                ),
                "final_stale_evaluations": stale,
            },
            "history": history,
            "loaded_model": str(loaded_path) if loaded_path is not None else None,
            "device_memory": training_device_memory,
        },
        "data": {
            "train": train_metadata,
            "validation": validation_metadata,
            "blind": blind_metadata,
            "independent_blind_sampling_clusters": (
                args.blind_points // args.sampling_cluster_size
            ),
            "cache_files": {
                "train": {
                    "path": str(train_path),
                    "sha256": sha256_file(train_path),
                },
                "validation": {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                },
                "blind": {
                    "path": str(blind_path),
                    "sha256": sha256_file(blind_path),
                },
            },
        },
        "h_artifact": {
            "path": str(h_artifact_path),
            "sha256": h_artifact_hash,
            "section_count": h_artifact.section_count,
            "h_min_eigenvalue": float(h_eigenvalues[0]),
            "h_max_eigenvalue": float(h_eigenvalues[-1]),
            "h_condition_number": h_condition,
        },
        "blind_test": {
            "fubini_study_self_normalized": fs_self,
            "fubini_study_fixed_training_kappa": fs_fixed,
            "trained_phi_self_normalized": phi_self,
            "trained_phi_fixed_training_kappa": phi_fixed,
            "reference_h_self_normalized": h_self,
            "top_phi_points": top_rows,
        },
        "atlas": {
            "points_checked": len(atlas_witnesses),
            "max_raw_log_ratio_spread": max_atlas_spread,
            "witnesses": atlas_witnesses,
        },
        "comparison": comparison,
        "files": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "tail_arrays": str(tail_arrays_path),
            "tail_arrays_sha256": sha256_file(tail_arrays_path),
        },
        "timing_seconds": {
            "optimization": optimization_seconds,
            "blind_phi_evaluation": blind_eval_seconds,
            "atlas": atlas_seconds,
            "wall_total": time.perf_counter() - total_start,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(json_value(report), indent=2) + "\n", encoding="utf-8")
    write_status(
        output_dir,
        "complete",
        report=str(report_path),
        best_epoch=best_epoch,
        phi_sigma=phi_self["sigma"],
        phi_max_ratio=phi_self["max_ratio"],
        h_sigma=h_self["sigma"],
        h_max_ratio=h_self["max_ratio"],
    )
    print(
        f"blind self-normalized: FS sigma={fs_self['sigma']:.6e}, max={fs_self['max_ratio']:.3e}; "
        f"PhiFS sigma={phi_self['sigma']:.6e}, max={phi_self['max_ratio']:.3e}; "
        f"H sigma={h_self['sigma']:.6e}, max={h_self['max_ratio']:.3e}",
        flush=True,
    )
    print(comparison["observational_interpretation"], flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
