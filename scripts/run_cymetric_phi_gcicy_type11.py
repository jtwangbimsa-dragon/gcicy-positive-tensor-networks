#!/usr/bin/env python3
"""Train and tail-audit a cymetric-style PhiFS model on the type-(1,1) gCICY.

The official cymetric point generator accepts polynomial CICY configurations,
not the negative-degree generalized column used here.  This script therefore
keeps the PhiFS ansatz and network scale, while using the repository's audited
gCICY sampler, tangent pullbacks, and holomorphic-volume density.  The neural
potential is made exactly projective invariant with normalized density-matrix
features on P4 x P1.
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
from gcicy_metric.type21_hirzebruch_x3 import (  # noqa: E402
    hirzebruch_type21_holomorphic_volume_log_density,
)


ADAPTER_KEY = "p4p1_type11_hirzebruch_x3"
DEFAULT_H_ARTIFACT = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_k4_l2_tail_refined_v4_gpu.npz"
)
RATIO_THRESHOLDS = (1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0)
LOW_RATIO_THRESHOLDS = (0.5, 0.2, 0.1)
ABS_LOG_THRESHOLDS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--h-artifact", type=Path, default=DEFAULT_H_ARTIFACT)
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
    parser.add_argument("--sampling-backend", choices=("process", "thread"), default="process")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument(
        "--feature-map",
        choices=("ambient_density", "source_density"),
        default="ambient_density",
        help=(
            "projectively invariant input: ambient-coordinate density matrices "
            "or the normalized density matrix of the H-artifact section vector"
        ),
    )
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
        help="continue optimization after loading a saved model",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="run a synchronized forward/update timing audit and exit",
    )
    parser.add_argument("--benchmark-warmup-updates", type=int, default=10)
    parser.add_argument("--benchmark-repetitions", type=int, default=50)
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


def direct_product_fs_metrics(
    affine: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    """Pull back the P4 x P1 product FS metric, excluding the auxiliary P1."""

    affine = np.asarray(affine, dtype=np.complex128)
    tangent = np.asarray(tangent, dtype=np.complex128)
    ambient = np.zeros((len(affine), 6, 6), dtype=np.complex128)
    for start, dimension in ((0, 4), (4, 1)):
        block = slice(start, start + dimension)
        values = affine[:, block]
        rho = 1.0 + np.sum(np.abs(values) ** 2, axis=1)
        identity = np.eye(dimension, dtype=np.complex128)[None, :, :]
        numerator = rho[:, None, None] * identity
        numerator -= values[:, :, None] * np.conjugate(values[:, None, :])
        ambient[:, block, block] = numerator / rho[:, None, None] ** 2
    metrics = np.einsum(
        "nai,nab,nbj->nij",
        np.conjugate(tangent),
        ambient,
        tangent,
        optimize=True,
    )
    return 0.5 * (metrics + np.conjugate(np.swapaxes(metrics, 1, 2)))


def point_arrays(points: Sequence[Any]) -> dict[str, np.ndarray]:
    affine = np.asarray([point.affine_coordinates for point in points], dtype=np.complex128)
    tangent = np.asarray([point.tangent_basis for point in points], dtype=np.complex128)
    charts = np.asarray(
        [int(point.projective_chart[0]) * 2 + int(point.projective_chart[1]) for point in points],
        dtype=np.int64,
    )
    log_omega = np.asarray(
        [hirzebruch_type21_holomorphic_volume_log_density(point) for point in points],
        dtype=np.float64,
    )
    return {
        "coords": np.concatenate((affine.real, affine.imag), axis=1),
        "charts": charts,
        "tangent": tangent,
        "base_metrics": direct_product_fs_metrics(affine, tangent),
        "log_omega": log_omega,
        "coordinates_x": np.asarray([point.x for point in points], dtype=np.complex128),
        "coordinates_y": np.asarray([point.y for point in points], dtype=np.complex128),
        "coordinates_z": np.asarray([point.z for point in points], dtype=np.complex128),
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


def dataset_cache_path(output_dir: Path, label: str, seed: int, count: int) -> Path:
    return output_dir / f"{label}_seed{seed}_n{count}.npz"


def prepare_dataset(
    *,
    output_dir: Path,
    label: str,
    count: int,
    seed: int,
    adapter: Any,
    model: Any,
    model_seed: int,
    sampling_workers: int,
    sampling_backend: str,
    include_h: bool,
    h_artifact: Any,
    h_artifact_hash: str,
    h_eval_batch_size: int,
    force: bool,
    common_pool_path: Path | None = None,
    common_pool_split: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
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
        points = common_pool.points
        arrays = point_arrays(points)
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
            "schema": "gcicy-type11-phifs-common-pool-v1",
            "adapter": ADAPTER_KEY,
            "model_seed": model_seed,
            "seed": seed,
            "count": count,
            "include_h": include_h,
            "h_artifact_sha256": h_artifact_hash if include_h else None,
            "common_pool": str(resolved_pool),
            "common_pool_sha256": sha256_file(resolved_pool),
            "common_pool_split": common_pool_split,
            "importance_effective_sample_size": effective_sample_size(
                arrays["importance_weights"]
            ),
        }
        print(f"loaded frozen {common_pool_split} dataset from {resolved_pool}", flush=True)
        return arrays, metadata, resolved_pool

    path = dataset_cache_path(output_dir, label, seed, count)
    expected_metadata = {
        "schema": "gcicy-type11-phifs-dataset-v2",
        "adapter": ADAPTER_KEY,
        "model_seed": model_seed,
        "seed": seed,
        "count": count,
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
            cluster_size=4,
            backend=sampling_backend,
        )
    else:
        points = adapter.sample_points(model, count, seed=seed)
        shards = [{"worker": 0, "points": count, "seed": seed}]
    arrays = point_arrays(points)
    log_weights = np.asarray([adapter.importance_log_weight(point) for point in points], dtype=np.float64)
    weights = np.exp(log_weights - float(np.max(log_weights)))
    weights /= float(np.mean(weights))
    arrays["importance_weights"] = weights
    arrays["cluster_ids"] = np.arange(count, dtype=np.int64) // 4
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
        "complete_four_root_clusters": count // 4,
    }
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
    for name in ("train_points", "validation_points", "blind_points"):
        value = int(getattr(args, name))
        if value <= 0 or value % 4:
            raise SystemExit(f"--{name.replace('_', '-')} must be a positive multiple of four")
    if min(args.batch_size, args.eval_batch_size, args.hidden_layers, args.hidden_width) <= 0:
        raise SystemExit("batch sizes and network dimensions must be positive")
    if args.epochs < 0 or args.eval_every <= 0:
        raise SystemExit("epochs must be non-negative and eval-every positive")
    geometric_loss_weights = (
        args.sigma_loss_weight,
        args.log_energy_loss_weight,
        args.ma_loss_weight,
        args.tail_loss_weight,
    )
    if min(geometric_loss_weights) < 0 or not any(geometric_loss_weights):
        raise SystemExit(
            "geometric loss weights must be non-negative with at least one enabled"
        )
    if (
        not 0 < args.tail_fraction <= 1
        or args.tail_ratio_threshold <= 1
        or args.tail_smooth_temperature <= 0
    ):
        raise SystemExit("invalid upper-tail loss configuration")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h_artifact_path = args.h_artifact.expanduser().resolve()
    if not h_artifact_path.exists():
        raise SystemExit(f"missing H artifact: {h_artifact_path}")
    h_artifact_hash = sha256_file(h_artifact_path)
    protocol_path = args.protocol.expanduser().resolve() if args.protocol is not None else None
    if protocol_path is not None and not protocol_path.exists():
        raise SystemExit(f"missing preregistered protocol: {protocol_path}")
    protocol_hash = sha256_file(protocol_path) if protocol_path is not None else None
    if protocol_path is not None:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if "frozen_data" in protocol:
            frozen_data = protocol["frozen_data"]
            phi_model = protocol["models"]["phi"]
            objective = protocol["objective"]
            schedule = protocol["plateau_schedule"]
            declared = {
                "model_seed": protocol["geometry"]["model_seed"],
                "train_seed": frozen_data["train"]["seed"],
                "train_points": frozen_data["train"]["points"],
                "validation_seed": frozen_data["selection"]["seed"],
                "validation_points": frozen_data["selection"]["points"],
                "hidden_width": phi_model["hidden_width"],
                "hidden_layers": phi_model["hidden_layers"],
                "activation": phi_model["activation"],
                "feature_map": phi_model.get("feature_map", "ambient_density"),
                "batch_size": schedule["batch_size"],
                "sigma_loss_weight": objective["sigma_weight"],
                "log_energy_loss_weight": objective["log_energy_weight"],
                "ma_loss_weight": objective["ma_E2_weight"],
                "tail_loss_weight": objective["upper_tail_cvar_weight"],
                "tail_fraction": objective["tail_fraction"],
                "tail_ratio_threshold": objective["tail_ratio_threshold"],
                "tail_smooth_temperature": objective["tail_smooth_temperature"],
                "positivity_weight": objective["phi_positivity_barrier_weight"],
                "correction_weight": objective["phi_correction_weight"],
            }
            allowed_blind_protocols = [
                frozen_data["development_confirmation"],
                protocol["new_final_blind"],
            ]
            if not any(
                args.blind_seed == entry["seed"]
                and args.blind_points == entry["points"]
                for entry in allowed_blind_protocols
            ):
                raise SystemExit(
                    "blind arguments do not match either the preregistered "
                    "development confirmation or new final blind"
                )
            allowed_learning_rates = {
                schedule["primary_stage"]["phi_learning_rate"],
                schedule["precision_stage"]["phi_learning_rate"],
            }
            if args.learning_rate not in allowed_learning_rates:
                raise SystemExit(
                    "learning rate does not match a preregistered phi stage: "
                    f"{args.learning_rate} not in {sorted(allowed_learning_rates)}"
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
                "hidden_width": 64,
                "hidden_layers": 3,
                "activation": "gelu",
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

    adapter = get_adapter(ADAPTER_KEY)
    model = adapter.make_model(args.model_seed, exact=True)
    h_artifact = adapter.load_h_artifact(h_artifact_path, model)
    h_eigenvalues = np.linalg.eigvalsh(h_artifact.h_matrix)
    h_condition = float(h_eigenvalues[-1] / h_eigenvalues[0])
    source_exponents_numpy = np.asarray(
        h_artifact.section_exponents,
        dtype=np.int64,
    )
    if source_exponents_numpy.ndim != 2 or source_exponents_numpy.shape[1] != 9:
        raise SystemExit("the type-(1,1) H artifact has unexpected section exponents")
    if np.any(source_exponents_numpy[:, 7:] != 0):
        raise SystemExit(
            "source-density features currently require sections independent of "
            "the auxiliary P1 coordinates"
        )

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
        label="train",
        count=args.train_points,
        seed=args.train_seed,
        adapter=adapter,
        model=model,
        model_seed=args.model_seed,
        sampling_workers=args.sampling_workers,
        sampling_backend=args.sampling_backend,
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
        label="validation",
        count=args.validation_points,
        seed=args.validation_seed,
        adapter=adapter,
        model=model,
        model_seed=args.model_seed,
        sampling_workers=args.sampling_workers,
        sampling_backend=args.sampling_backend,
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
    phi_label = (
        "H-base residual Phi"
        if args.base_metric == "h_artifact"
        else "PhiFS"
    )

    def to_torch(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            "coords": torch.as_tensor(dataset["coords"], dtype=real_dtype, device=device),
            "charts": torch.as_tensor(dataset["charts"], dtype=torch.int64, device=device),
            "tangent": torch.as_tensor(dataset["tangent"], dtype=complex_dtype, device=device),
            "base_metrics": torch.as_tensor(
                dataset[base_metric_array_key],
                dtype=complex_dtype,
                device=device,
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

    feature_dimension = (
        29
        if args.feature_map == "ambient_density"
        else int(len(source_exponents_numpy) ** 2)
    )

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
    source_exponents = torch.as_tensor(
        source_exponents_numpy[:, :7],
        dtype=torch.int64,
        device=device,
    )
    source_pair_indices = torch.triu_indices(
        len(source_exponents_numpy),
        len(source_exponents_numpy),
        offset=1,
        device=device,
    )

    def insertion_maps(size: int) -> tuple[Any, Any]:
        maps = torch.zeros(
            (size, size, size - 1),
            dtype=real_dtype,
            device=device,
        )
        for fixed_index in range(size):
            cursor = 0
            for index in range(size):
                if index == fixed_index:
                    continue
                maps[fixed_index, index, cursor] = 1
                cursor += 1
        return maps, torch.eye(size, dtype=real_dtype, device=device)

    x_insertion_maps, x_fixed_vectors = insertion_maps(5)
    y_insertion_maps, y_fixed_vectors = insertion_maps(2)

    def homogeneous_block(
        active_real: Any,
        active_imag: Any,
        maps: Any,
        fixed_vectors: Any,
        selector: Any,
    ) -> tuple[Any, Any]:
        insertion = torch.einsum("c,cij->ij", selector, maps)
        fixed = torch.einsum("c,ci->i", selector, fixed_vectors)
        return insertion @ active_real + fixed, insertion @ active_imag

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

    def homogeneous_xy(coords: Any, chart_selector: Any) -> tuple[Any, Any]:
        product_selector = chart_selector.reshape(5, 2)
        x_selector = torch.sum(product_selector, dim=1)
        y_selector = torch.sum(product_selector, dim=0)
        real = coords[:6]
        imag = coords[6:]
        x_real, x_imag = homogeneous_block(
            real[:4],
            imag[:4],
            x_insertion_maps,
            x_fixed_vectors,
            x_selector,
        )
        y_real, y_imag = homogeneous_block(
            real[4:5],
            imag[4:5],
            y_insertion_maps,
            y_fixed_vectors,
            y_selector,
        )
        return torch.cat((x_real, y_real)), torch.cat((x_imag, y_imag))

    def ambient_density_features(coords: Any, chart_selector: Any) -> Any:
        xy_real, xy_imag = homogeneous_xy(coords, chart_selector)
        x_real, y_real = xy_real[:5], xy_real[5:]
        x_imag, y_imag = xy_imag[:5], xy_imag[5:]
        return torch.cat((density_features(x_real, x_imag), density_features(y_real, y_imag)))

    def source_density_features(coords: Any, chart_selector: Any) -> Any:
        xy_real, xy_imag = homogeneous_xy(coords, chart_selector)
        homogeneous = torch.complex(xy_real, xy_imag)
        section_values = torch.prod(
            homogeneous[None, :] ** source_exponents,
            dim=1,
        )
        real = section_values.real
        imag = section_values.imag
        denominator = torch.clamp(
            torch.sum(real**2 + imag**2),
            min=1.0e-12,
        )
        diagonal = (real**2 + imag**2) / denominator
        left, right = source_pair_indices
        pair_real = (real[left] * real[right] + imag[left] * imag[right]) / denominator
        pair_imag = (imag[left] * real[right] - real[left] * imag[right]) / denominator
        interleaved_pairs = torch.stack((pair_real, pair_imag), dim=1).reshape(-1)
        return torch.cat((diagonal, interleaved_pairs))

    def invariant_features(coords: Any, chart_selector: Any) -> Any:
        if args.feature_map == "source_density":
            return source_density_features(coords, chart_selector)
        return ambient_density_features(coords, chart_selector)

    def single_potential(coords: Any, chart_selector: Any) -> Any:
        return args.potential_scale * potential(
            invariant_features(coords, chart_selector)
        )

    batched_hessian = vmap(
        hessian(single_potential, argnums=0),
        in_dims=(0, 0),
    )

    def correction_batch(coords: Any, charts: Any, tangent: Any) -> Any:
        chart_selectors = torch.nn.functional.one_hot(
            charts,
            num_classes=10,
        ).to(dtype=real_dtype)
        real_hessian = batched_hessian(coords, chart_selectors)
        h_xx = real_hessian[:, :6, :6]
        h_xy = real_hessian[:, :6, 6:]
        h_yx = real_hessian[:, 6:, :6]
        h_yy = real_hessian[:, 6:, 6:]
        # Repository metrics use rows anti-holomorphic and columns holomorphic.
        ambient = 0.25 * (h_xx + h_yy).to(complex_dtype)
        ambient -= 0.25j * (h_xy - h_yx).to(complex_dtype)
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

    def baseline_evaluation(
        dataset_np: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues = np.linalg.eigvalsh(dataset_np[base_metric_array_key])
        raw = np.sum(np.log(eigenvalues), axis=1) - dataset_np["log_omega"]
        return raw, eigenvalues[:, 0]

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
        initial_metric, initial_correction, _, _ = batch_metrics(
            train,
            initial_indices,
        )
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

    def benchmark_training_objective(indices: Any) -> Any:
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
            torch.mean(torch.abs(train["base_metrics"][indices]) ** 2),
            min=1.0e-12,
        )
        return (
            args.sigma_loss_weight * sigma_loss
            + args.log_energy_loss_weight * log_energy_loss
            + args.ma_loss_weight * ma_loss
            + args.tail_loss_weight * tail_loss
            + args.positivity_weight * barrier
            + args.correction_weight * correction_size
        )

    if args.benchmark_only:
        if args.benchmark_warmup_updates < 0 or args.benchmark_repetitions < 1:
            raise SystemExit("benchmark counts must be nonnegative and positive")
        count = min(args.batch_size, args.train_points)
        indices = torch.arange(count, device=device)

        def synchronize() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for _ in range(args.benchmark_warmup_updates):
            optimizer.zero_grad(set_to_none=True)
            loss = benchmark_training_objective(indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(potential.parameters(), max_norm=5.0)
            optimizer.step()
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        metric_forward_seconds: list[float] = []
        forward_loss_seconds: list[float] = []
        update_seconds: list[float] = []
        for _ in range(args.benchmark_repetitions):
            synchronize()
            started = time.perf_counter()
            metric_outputs = batch_metrics(train, indices)
            synchronize()
            metric_forward_seconds.append(time.perf_counter() - started)
            del metric_outputs

            synchronize()
            started = time.perf_counter()
            loss = benchmark_training_objective(indices)
            synchronize()
            forward_loss_seconds.append(time.perf_counter() - started)
            del loss

            synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = benchmark_training_objective(indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(potential.parameters(), max_norm=5.0)
            optimizer.step()
            synchronize()
            update_seconds.append(time.perf_counter() - started)
            del loss

        def timing_summary(values: list[float]) -> dict[str, float]:
            milliseconds = 1000.0 * np.asarray(values, dtype=np.float64)
            return {
                "median_ms": float(np.median(milliseconds)),
                "p10_ms": float(np.quantile(milliseconds, 0.10)),
                "p90_ms": float(np.quantile(milliseconds, 0.90)),
                "median_microseconds_per_point": float(
                    1000.0 * np.median(milliseconds) / count
                ),
            }

        payload = {
            "schema": "gcicy-type11-residual-phi-synchronized-timing-v1",
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "dtype": args.dtype,
            "feature_map": args.feature_map,
            "hidden_layers": args.hidden_layers,
            "hidden_width": args.hidden_width,
            "parameter_count": model_parameter_count(potential),
            "batch_size": count,
            "warmup_updates": args.benchmark_warmup_updates,
            "repetitions": args.benchmark_repetitions,
            "metric_forward": timing_summary(metric_forward_seconds),
            "forward_and_training_loss": timing_summary(forward_loss_seconds),
            "complete_adam_update": timing_summary(update_seconds),
            "peak_allocated_gib": (
                float(torch.cuda.max_memory_allocated(device) / 2**30)
                if device.type == "cuda"
                else None
            ),
        }
        timing_path = output_dir / "synchronized_timing.json"
        timing_path.write_text(
            json.dumps(json_value(payload), indent=2) + "\n",
            encoding="utf-8",
        )
        write_status(output_dir, "complete", timing=str(timing_path))
        print(json.dumps(json_value(payload), indent=2), flush=True)
        return

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
                "schema": "gcicy-type11-residual-phi-v1",
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "fixed_log_kappa": fixed_log_kappa,
                "network": {
                    "input_dimension": feature_dimension,
                    "feature_map": args.feature_map,
                    "hidden_layers": args.hidden_layers,
                    "hidden_width": args.hidden_width,
                    "activation": args.activation,
                    "potential_scale": args.potential_scale,
                    "parameter_count": model_parameter_count(potential),
                },
                "adapter": ADAPTER_KEY,
                "model_seed": args.model_seed,
                "dtype": args.dtype,
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

    termination_reason = (
        "evaluation_only"
        if args.load_model is not None and not args.resume_training
        else "epoch_cap"
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
                termination_reason = "validation_plateau"
                break
    else:
        validation_raw, validation_minimum = evaluate_phi(validation, args.validation_points)
        best_validation, _ = tail_statistics(
            validation_raw,
            validation_np["importance_weights"],
            validation_minimum,
            validation_clusters,
            fixed_log_normalization=fixed_log_kappa,
        )
        best_score = geometric_selection_score(
            validation_raw,
            validation_np["importance_weights"],
        )
        best_state = copy.deepcopy(potential.state_dict())
        best_epoch = int(loaded_model.get("best_epoch", -1))

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
    torch.save(
        {
            "schema": "gcicy-type11-residual-phi-v1",
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "fixed_log_kappa": fixed_log_kappa,
            "network": {
                "input_dimension": feature_dimension,
                "feature_map": args.feature_map,
                "hidden_layers": args.hidden_layers,
                "hidden_width": args.hidden_width,
                "activation": args.activation,
                "potential_scale": args.potential_scale,
                "parameter_count": model_parameter_count(potential),
            },
            "adapter": ADAPTER_KEY,
            "model_seed": args.model_seed,
            "dtype": args.dtype,
            "h_artifact": str(h_artifact_path),
            "h_artifact_sha256": h_artifact_hash,
            "base_metric": args.base_metric,
            "train_common_pool_sha256": train_metadata.get(
                "common_pool_sha256"
            ),
            "validation_common_pool_sha256": validation_metadata.get(
                "common_pool_sha256"
            ),
            "objective": {
                "sigma": args.sigma_loss_weight,
                "log_energy": args.log_energy_loss_weight,
                "ma": args.ma_loss_weight,
                "tail": args.tail_loss_weight,
                "tail_fraction": args.tail_fraction,
                "tail_ratio_threshold": args.tail_ratio_threshold,
                "tail_smooth_temperature": args.tail_smooth_temperature,
                "positivity": args.positivity_weight,
                "correction": args.correction_weight,
            },
        },
        checkpoint_path,
    )

    write_status(output_dir, "sampling_and_auditing_blind", best_epoch=best_epoch)
    blind_np, blind_metadata, blind_path = prepare_dataset(
        output_dir=output_dir,
        label="blind",
        count=args.blind_points,
        seed=args.blind_seed,
        adapter=adapter,
        model=model,
        model_seed=args.model_seed,
        sampling_workers=args.sampling_workers,
        sampling_backend=args.sampling_backend,
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
    blind_fs_eigenvalues = np.linalg.eigvalsh(blind_np["base_metrics"])
    blind_fs_raw = (
        np.sum(np.log(blind_fs_eigenvalues), axis=1) - blind_np["log_omega"]
    )
    blind_fs_minimum = blind_fs_eigenvalues[:, 0]
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
    top_rows = []
    for index in top_indices:
        top_rows.append(
            {
                "index": int(index),
                "cluster_id": int(clusters[index]),
                "chart_id": int(blind_np["charts"][index]),
                "importance_weight": float(weights[index]),
                "fs_ratio": float(fs_ratio[index]),
                "phi_ratio": float(phi_ratio[index]),
                "h_ratio": float(h_ratio[index]),
                "phi_min_eigenvalue": float(blind_phi_minimum[index]),
                "h_min_eigenvalue": float(blind_np["h_min_eigenvalue"][index]),
                "x": [[float(value.real), float(value.imag)] for value in blind_np["coordinates_x"][index]],
                "y": [[float(value.real), float(value.imag)] for value in blind_np["coordinates_y"][index]],
                "z": [[float(value.real), float(value.imag)] for value in blind_np["coordinates_z"][index]],
            }
        )

    atlas_start = time.perf_counter()
    atlas_witnesses = []
    max_atlas_spread = 0.0
    for index in top_indices[: args.atlas_points]:
        payload = {
            "coordinates_x": blind_np["coordinates_x"][[index]],
            "coordinates_y": blind_np["coordinates_y"][[index]],
            "coordinates_z": blind_np["coordinates_z"][[index]],
        }
        original = adapter.points_from_storage_payload(model, payload)[0]
        values = []
        charts_seen = []
        for chart in adapter.projective_charts():
            if not adapter.chart_is_available(original, chart, minimum=1.0e-5):
                continue
            candidate = adapter.rechart_point(model, original, chart)
            candidate_np = point_arrays([candidate])
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
    np.savez_compressed(
        tail_arrays_path,
        schema=np.asarray("gcicy-common-residual-phi-point-arrays-v1"),
        adapter=np.asarray(adapter.key),
        common_pool_sha256=np.asarray(
            blind_metadata.get("common_pool_sha256") or ""
        ),
        model_log_eta=blind_phi_raw,
        importance_weights=weights,
        sampling_cluster_ids=clusters,
        metric_minimum_eigenvalues=blind_phi_minimum,
        cluster_ids=clusters,
        chart_ids=blind_np["charts"],
        fs_raw_log_ratio=blind_fs_raw,
        phi_raw_log_ratio=blind_phi_raw,
        h_raw_log_ratio=blind_np["h_raw_log_ratio"],
        fs_normalized_ratio=fs_ratio,
        phi_normalized_ratio=phi_ratio,
        h_normalized_ratio=h_ratio,
        fs_min_eigenvalue=blind_fs_minimum,
        phi_min_eigenvalue=blind_phi_minimum,
        h_min_eigenvalue=blind_np["h_min_eigenvalue"],
        top_phi_indices=top_indices,
    )

    phi_severe = bool(phi_self["max_ratio"] > 10.0)
    h_severe = bool(h_self["max_ratio"] > 10.0)
    comparison = {
        "paired_same_blind_points": True,
        "h_has_ratio_above_10": h_severe,
        "phi_has_ratio_above_10": phi_severe,
        "observational_interpretation": (
            f"The same gCICY blind sample has a severe H tail but no severe {phi_label} tail; "
            "this is evidence against the spike being forced by the gCICY structure."
            if h_severe and not phi_severe
            else "Both ansatzes show a ratio above 10 on this blind sample; the control is inconclusive and the shared geometry/sampler must be investigated."
            if h_severe and phi_severe
            else f"{phi_label} has a ratio above 10 while the H metric does not on this blind sample; "
            f"the {phi_label} fit is not yet a successful structural control."
            if phi_severe
            else "No ratio above 10 was observed for either ansatz on this blind sample."
        ),
        "claim_limit": (
            "Finite blind sampling detects observed tails but does not prove a global sup-norm bound."
        ),
    }

    report = {
        "schema": "gcicy-type11-cymetric-style-phifs-tail-audit-v1",
        "scientific_scope": {
            "geometry": "smooth direct type-(1,1) gCICY in P4 x P1",
            "configuration": [[1, 4], [3, -1]],
            "adapter": ADAPTER_KEY,
            "model_seed": args.model_seed,
            "method": (
                "g = g_Hbase + partial partialbar phi_theta"
                if args.base_metric == "h_artifact"
                else "g = g_FS + partial partialbar phi_theta"
            ),
            "implementation_boundary": (
                "Official cymetric does not natively generate points for a negative-degree generalized column. "
                "This run adapts the projective-invariant scalar-potential ansatz "
                "and 3x64 network scale to the audited gCICY geometry backend."
            ),
            "globality": (
                "phi_theta uses the 29 real entries of normalized rank-one density matrices on P4 and P1, "
                "so projective invariance is exact rather than learned by a transition loss."
            ),
        },
        "preregistered_protocol": {
            "path": str(protocol_path) if protocol_path is not None else None,
            "sha256": protocol_hash,
        },
        "configuration": json_value(vars(args)),
        "environment": environment,
        "network": {
            "input_dimension": feature_dimension,
            "feature_map": args.feature_map,
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
            "termination_reason": termination_reason,
            "early_stopping": {
                "patience_evaluations": args.patience_evaluations,
                "minimum_relative_improvement": (
                    args.early_stopping_min_relative_improvement
                ),
                "evaluations_since_material_improvement": stale,
                "patience_reference_score": patience_reference_score,
            },
            "baseline_validation_fixed_kappa": baseline_validation_fixed,
            "best_validation_fixed_kappa": best_validation,
            "history": history,
            "loaded_model": str(loaded_path) if loaded_path is not None else None,
        },
        "data": {
            "train": train_metadata,
            "validation": validation_metadata,
            "blind": blind_metadata,
            "independent_blind_fibres": args.blind_points // 4,
            "cache_files": {
                "train": str(train_path),
                "validation": str(validation_path),
                "blind": str(blind_path),
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
            "current_k4_h_self_normalized": h_self,
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
        "training_device_memory": training_device_memory,
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
        f"{phi_label} sigma={phi_self['sigma']:.6e}, "
        f"max={phi_self['max_ratio']:.3e}; "
        f"H sigma={h_self['sigma']:.6e}, max={h_self['max_ratio']:.3e}",
        flush=True,
    )
    print(comparison["observational_interpretation"], flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
