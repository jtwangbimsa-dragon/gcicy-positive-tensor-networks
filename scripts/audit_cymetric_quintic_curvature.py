#!/usr/bin/env python3
"""Audit cymetric/ModNet Ricci curvature on fixed Fermat blind fibres."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from cymetric.models.tfhelper import prepare_tf_basis  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cymetric_quintic_tail_experiment import build_model  # noqa: E402


PROBABILITIES = np.array([0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument("--data-source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=5000)
    parser.add_argument("--fibre-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=202607193)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if tf.is_tensor(value):
        return json_value(value.numpy())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_whole_fibres(
    point_count: int, sample_points: int, fibre_size: int, seed: int
) -> np.ndarray:
    if fibre_size <= 0 or point_count % fibre_size:
        raise ValueError("the blind point count must be divisible by fibre-size")
    if sample_points <= 0 or sample_points % fibre_size:
        raise ValueError("sample-points must be a positive multiple of fibre-size")
    if sample_points > point_count:
        raise ValueError("sample-points exceeds the blind pool")
    generator = np.random.default_rng(seed)
    fibre_count = point_count // fibre_size
    selected_fibres = np.sort(
        generator.choice(fibre_count, sample_points // fibre_size, replace=False)
    )
    offsets = np.arange(fibre_size, dtype=np.int64)
    return (selected_fibres[:, None] * fibre_size + offsets[None, :]).reshape(-1)


def cluster_standard_error(values: np.ndarray, fibre_size: int) -> float:
    clusters = values.reshape(-1, fibre_size).mean(axis=1)
    if len(clusters) < 2:
        return float("nan")
    return float(np.std(clusters, ddof=1) / math.sqrt(len(clusters)))


def weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        raise ValueError("weighted quantiles require positive finite weights")
    selected_values = values[finite]
    selected_weights = weights[finite]
    order = np.argsort(selected_values, kind="mergesort")
    selected_values = selected_values[order]
    cumulative = np.cumsum(selected_weights[order])
    cumulative /= cumulative[-1]
    quantiles = np.interp(
        PROBABILITIES,
        cumulative,
        selected_values,
        left=selected_values[0],
        right=selected_values[-1],
    )
    return {
        f"q{probability:.3f}": float(value)
        for probability, value in zip(PROBABILITIES, quantiles, strict=True)
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    model_run_dir = args.model_run_dir.expanduser().resolve()
    data_source_run_dir = args.data_source_run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})

    try:
        paths = {
            "model_report": model_run_dir / "report.json",
            "model_weights": model_run_dir / "phi_network.weights.h5",
            "model_tail_arrays": model_run_dir / "blind_test_tail_arrays.npz",
            "basis": data_source_run_dir / "training_data" / "basis.pickle",
            "blind_points": data_source_run_dir / "blind_points.npz",
        }
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")

        report = json.loads(paths["model_report"].read_text(encoding="utf-8"))
        configuration = report["configuration"]
        raw_basis = np.load(paths["basis"], allow_pickle=True)
        basis = prepare_tf_basis(raw_basis)
        blind = np.load(paths["blind_points"], allow_pickle=False)
        points = np.asarray(blind["X"], dtype=np.float32)
        weights = np.asarray(blind["weights"], dtype=np.float64)
        omega_squared = np.asarray(blind["omega_squared"], dtype=np.float64)
        tail = np.load(paths["model_tail_arrays"], allow_pickle=False)
        saved_weights = np.asarray(tail["weights"], dtype=np.float64)
        saved_omega = np.asarray(tail["omega_squared"], dtype=np.float64)
        saved_determinants = np.asarray(tail["determinant"], dtype=np.float64)
        saved_minimum_eigenvalues = np.asarray(
            tail["min_eigenvalue"], dtype=np.float64
        )
        if not (
            len(points)
            == len(weights)
            == len(omega_squared)
            == len(saved_weights)
            == len(saved_omega)
            == len(saved_determinants)
        ):
            raise RuntimeError("blind points and model tail arrays are not aligned")
        if not (
            np.array_equal(weights, saved_weights)
            and np.array_equal(omega_squared, saved_omega)
        ):
            raise RuntimeError("blind integration labels differ from the saved audit")

        network, model = build_model(
            points.shape[1],
            int(configuration["hidden_layers"]),
            int(configuration["hidden_width"]),
            str(configuration["activation"]),
            basis,
            network_family=str(configuration["network_family"]),
            modnet_order=int(configuration["modnet_order"]),
            modnet_weight_variance=float(configuration["modnet_weight_variance"]),
            modnet_skip=not bool(configuration["modnet_no_skip"]),
        )
        network.load_weights(paths["model_weights"])
        expected_weight_hash = report["network"]["weights_sha256"]
        actual_weight_hash = sha256_file(paths["model_weights"])
        if actual_weight_hash != expected_weight_hash:
            raise RuntimeError("model weight hash differs from the training report")

        indices = select_whole_fibres(
            len(points), args.sample_points, args.fibre_size, args.seed
        )
        sample_points = points[indices]
        determinants = np.empty(len(indices), dtype=np.float64)
        minimum_eigenvalues = np.empty(len(indices), dtype=np.float64)
        maximum_eigenvalues = np.empty(len(indices), dtype=np.float64)
        ricci_scalars = np.empty(len(indices), dtype=np.float64)

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, points.shape[1]), dtype=tf.float32)
            ],
            reduce_retracing=True,
        )
        def evaluate_batch(
            batch: tf.Tensor,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            metric = model(batch, training=False)
            eigenvalues = tf.math.real(tf.linalg.eigvalsh(metric))
            return (
                tf.math.real(tf.linalg.det(metric)),
                tf.reduce_min(eigenvalues, axis=-1),
                tf.reduce_max(eigenvalues, axis=-1),
                model.compute_ricci_scalar(batch),
            )

        write_json(
            status_path,
            {
                "state": "running",
                "phase": "curvature",
                "sample_points": len(indices),
                "completed": 0,
            },
        )
        curvature_started = time.perf_counter()
        for start in range(0, len(indices), args.batch_size):
            stop = min(start + args.batch_size, len(indices))
            batch_results = evaluate_batch(
                tf.convert_to_tensor(sample_points[start:stop], dtype=tf.float32)
            )
            arrays = [np.asarray(result.numpy(), dtype=np.float64) for result in batch_results]
            determinants[start:stop] = arrays[0]
            minimum_eigenvalues[start:stop] = arrays[1]
            maximum_eigenvalues[start:stop] = arrays[2]
            ricci_scalars[start:stop] = arrays[3]
            print(f"curvature={stop}/{len(indices)}", flush=True)
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "curvature",
                    "sample_points": len(indices),
                    "completed": stop,
                },
            )

        recomputed_difference = determinants - saved_determinants[indices]
        recomputed_minimum_difference = (
            minimum_eigenvalues - saved_minimum_eigenvalues[indices]
        )
        full_volume_omega = float(np.mean(weights))
        full_det_over_omega = saved_determinants / omega_squared
        full_volume_k = float(np.mean(full_det_over_omega * weights))
        sample_det_over_omega = determinants / omega_squared[indices]
        metric_measure_weights = sample_det_over_omega * weights[indices]
        absolute_ricci = np.abs(ricci_scalars)
        numerator_terms = metric_measure_weights * absolute_ricci
        numerator = float(np.mean(numerator_terms))
        ricci_measure = full_volume_k ** (1.0 / 3.0) * numerator / full_volume_omega
        numerator_standard_error = cluster_standard_error(
            numerator_terms, args.fibre_size
        )
        ricci_measure_standard_error = (
            full_volume_k ** (1.0 / 3.0)
            * numerator_standard_error
            / full_volume_omega
        )
        metric_weight_sum = float(np.sum(metric_measure_weights))

        arrays_path = output_dir / "curvature_arrays.npz"
        np.savez_compressed(
            arrays_path,
            indices=indices,
            determinant=determinants,
            min_eigenvalue=minimum_eigenvalues,
            max_eigenvalue=maximum_eigenvalues,
            ricci_scalar=ricci_scalars,
            det_over_omega=sample_det_over_omega,
            integration_weight=weights[indices],
            metric_measure_weight=metric_measure_weights,
        )

        result = {
            "schema": "cymetric-quintic-curvature-audit-v1",
            "scientific_scope": {
                "geometry": "Fermat quintic hypersurface X_5 in P^4",
                "metric": report["scientific_scope"]["model"],
                "curvature_method": "official cymetric PhiFSModel.compute_ricci_scalar",
                "ricci_measure_definition": (
                    "Vol_K^(1/3)/Vol_Omega * integral_X dVol_K |R|"
                ),
                "claim_limit": (
                    "Monte Carlo curvature estimate on independently selected whole "
                    "fibres; not a global sup-norm certificate."
                ),
            },
            "configuration": vars(args),
            "model": {
                "network_family": configuration["network_family"],
                "hidden_width": configuration["hidden_width"],
                "trainable_parameter_count": int(network.count_params()),
                "training_report_schema": report["schema"],
            },
            "sampling": {
                "blind_pool_points": len(points),
                "sample_points": len(indices),
                "sample_fibres": len(indices) // args.fibre_size,
                "fibre_size": args.fibre_size,
                "seed": args.seed,
                "whole_fibre_cluster_standard_error": True,
            },
            "replay_validation": {
                "maximum_abs_determinant_difference": float(
                    np.max(np.abs(recomputed_difference))
                ),
                "maximum_abs_minimum_eigenvalue_difference": float(
                    np.max(np.abs(recomputed_minimum_difference))
                ),
            },
            "volume_estimates": {
                "full_blind_volume_omega": full_volume_omega,
                "full_blind_volume_k": full_volume_k,
                "curvature_sample_volume_k": float(
                    np.mean(metric_measure_weights)
                ),
                "sample_relative_volume_k_difference": float(
                    np.mean(metric_measure_weights) / full_volume_k - 1.0
                ),
            },
            "curvature": {
                "ricci_measure": ricci_measure,
                "ricci_measure_cluster_standard_error": ricci_measure_standard_error,
                "metric_volume_weighted_mean_abs_ricci_scalar": float(
                    np.sum(metric_measure_weights * absolute_ricci) / metric_weight_sum
                ),
                "metric_volume_weighted_rms_ricci_scalar": float(
                    np.sqrt(
                        np.sum(metric_measure_weights * ricci_scalars**2)
                        / metric_weight_sum
                    )
                ),
                "abs_ricci_scalar_metric_volume_quantiles": weighted_quantiles(
                    absolute_ricci, metric_measure_weights
                ),
                "metric_minimum_eigenvalue_quantiles": weighted_quantiles(
                    minimum_eigenvalues, metric_measure_weights
                ),
                "metric_condition_number_quantiles": weighted_quantiles(
                    maximum_eigenvalues / minimum_eigenvalues,
                    metric_measure_weights,
                ),
            },
            "provenance": {
                **{name + "_sha256": sha256_file(path) for name, path in paths.items()},
                "curvature_arrays_sha256": sha256_file(arrays_path),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "tensorflow": tf.__version__,
                "physical_devices": [device.name for device in tf.config.list_physical_devices()],
            },
            "timing_seconds": {
                "curvature": time.perf_counter() - curvature_started,
                "wall_total": time.perf_counter() - started,
            },
        }
        report_path = output_dir / "report.json"
        write_json(report_path, result)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps(json_value(result), indent=2, sort_keys=True), flush=True)
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "exception",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
