#!/usr/bin/env python3
"""Train the official cymetric PhiFSModel on the Fermat quintic and audit tails.

The training and point-generation paths intentionally use cymetric's public APIs.
The additional code only performs an independent, pointwise audit of the trained
metric-volume ratio.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

from cymetric.models.tfhelper import prepare_tf_basis, train_model  # noqa: E402
from cymetric.models.tfmodels import PhiFSModel  # noqa: E402
from cymetric.pointgen.pointgen import PointGenerator  # noqa: E402


PROBABILITIES = np.array(
    [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 1.0],
    dtype=np.float64,
)
RATIO_THRESHOLDS = (1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0)
LOW_RATIO_THRESHOLDS = (0.5, 0.2, 0.1, 0.0)
ABS_RESIDUAL_THRESHOLDS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-source-run-dir",
        type=Path,
        help=(
            "read training_data and blind_points.npz from an existing run; "
            "this provides an exact common-point comparison"
        ),
    )
    parser.add_argument(
        "--blind-source-run-dir",
        type=Path,
        help=(
            "optionally read only blind_points.npz from another run; this "
            "allows a new training dataset while preserving the frozen blind pool"
        ),
    )
    parser.add_argument("--train-points", type=int, default=100_000)
    parser.add_argument("--test-points", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument(
        "--network-family",
        choices=("dense", "modnet"),
        default="dense",
        help=(
            "dense reproduces the standard cymetric baseline; modnet uses the "
            "Fermat symmetric-power features s_k^(j/5)"
        ),
    )
    parser.add_argument(
        "--modnet-order",
        type=int,
        default=5,
        help="number of power sums and roots in each ModNet feature direction",
    )
    parser.add_argument(
        "--modnet-weight-variance",
        type=float,
        default=1.0e-4,
        help="variance of the normal initializer used by the ModNet-style network",
    )
    parser.add_argument(
        "--modnet-no-skip",
        action="store_true",
        help="disable the residual connection between equal-width hidden layers",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--volk-batch-size", type=int, default=50_000)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--train-seed", type=int, default=202607151)
    parser.add_argument("--test-seed", type=int, default=202607152)
    parser.add_argument("--model-seed", type=int, default=202607153)
    parser.add_argument("--top-points", type=int, default=64)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="build the model and validate ModNet features without training",
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help=(
            "generate and certify training_data without training or blind-test "
            "evaluation"
        ),
    )
    parser.add_argument("--skip-fs-baseline", action="store_true")
    parser.add_argument("--force-data", action="store_true")
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
    if tf.is_tensor(value):
        array = value.numpy()
        return json_value(array.item() if array.ndim == 0 else array.tolist())
    if isinstance(value, Path):
        return str(value)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def provenance() -> dict[str, Any]:
    cymetric_module = Path(sys.modules["cymetric"].__file__).resolve()
    cymetric_root = cymetric_module.parents[1]
    gpu_query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "tf_keras_module": str(Path(tf.keras.__file__).resolve()),
        "keras": package_version("keras"),
        "cymetric": package_version("cymetric"),
        "cymetric_module": str(cymetric_module),
        "cymetric_git_commit": command_output(
            ["git", "-C", str(cymetric_root), "rev-parse", "HEAD"]
        ),
        "physical_devices": [str(device) for device in tf.config.list_physical_devices()],
        "gpu": gpu_query,
    }


def make_point_generator(seed: int | None = None) -> PointGenerator:
    monomials = 5 * np.eye(5, dtype=np.int64)
    coefficients = np.ones(5, dtype=np.complex128)
    kmoduli = np.ones(1, dtype=np.float64)
    ambient = np.array([4], dtype=np.int64)
    generator = PointGenerator(monomials, coefficients, kmoduli, ambient)
    if seed is not None:
        # PointGenerator.__init__ resets NumPy's global RNG to 2021.
        # Apply the requested seed only after construction.
        generator._set_seed(seed)
        random.seed(seed)
    return generator


def prepare_training_data(
    output_dir: Path,
    n_points: int,
    seed: int,
    force: bool,
) -> tuple[np.lib.npyio.NpzFile, dict[str, Any], float, dict[str, str]]:
    data_dir = output_dir / "training_data"
    dataset_path = data_dir / "dataset.npz"
    basis_path = data_dir / "basis.pickle"
    if force or not (dataset_path.exists() and basis_path.exists()):
        data_dir.mkdir(parents=True, exist_ok=True)
        generator = make_point_generator(seed)
        kappa = float(generator.prepare_dataset(n_points, str(data_dir), val_split=0.1))
        generator.prepare_basis(str(data_dir), kappa=kappa)
    raw_basis = np.load(basis_path, allow_pickle=True)
    kappa = float(np.real(raw_basis["KAPPA"]))
    basis = prepare_tf_basis(raw_basis)
    data = np.load(dataset_path, allow_pickle=False)
    hashes = {
        "dataset_sha256": sha256_file(dataset_path),
        "basis_sha256": sha256_file(basis_path),
    }
    return data, basis, kappa, hashes


@tf.keras.utils.register_keras_serializable(package="gcicy_metric")
class SymmetricPowerFeatures(tf.keras.layers.Layer):
    """Projectively invariant Fermat features ``s_k**(j/order)``."""

    def __init__(self, order: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if order <= 0:
            raise ValueError("feature order must be positive")
        self.order = int(order)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        n_complex = tf.shape(inputs)[-1] // 2
        real = inputs[..., :n_complex]
        imag = inputs[..., n_complex:]
        absolute_squared = tf.square(real) + tf.square(imag)
        norm_squared = tf.maximum(
            tf.reduce_sum(absolute_squared, axis=-1, keepdims=True),
            tf.cast(1.0e-12, inputs.dtype),
        )
        normalized = absolute_squared / norm_squared
        power_sums = tf.stack(
            [
                tf.reduce_sum(tf.pow(normalized, k), axis=-1)
                for k in range(1, self.order + 1)
            ],
            axis=-1,
        )
        exponents = tf.cast(
            tf.range(1, self.order + 1), inputs.dtype
        ) / tf.cast(self.order, inputs.dtype)
        features = tf.pow(
            tf.maximum(power_sums[..., :, None], tf.cast(1.0e-12, inputs.dtype)),
            exponents,
        )
        return tf.reshape(features, (-1, self.order * self.order))

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "order": self.order}


def build_model(
    n_input: int,
    hidden_layers: int,
    hidden_width: int,
    activation: str,
    basis: dict[str, Any],
    *,
    network_family: str = "dense",
    modnet_order: int = 5,
    modnet_weight_variance: float = 1.0e-4,
    modnet_skip: bool = True,
) -> tuple[tf.keras.Sequential, PhiFSModel]:
    if network_family == "modnet":
        if hidden_layers != 2:
            raise ValueError("the controlled ModNet-style baseline uses two hidden layers")
        if modnet_order <= 0 or modnet_weight_variance <= 0:
            raise ValueError("invalid ModNet feature order or initializer variance")
        inputs = tf.keras.Input(shape=(n_input,), dtype=tf.float32)
        features = SymmetricPowerFeatures(modnet_order, name="symmetric_power_features")(
            inputs
        )
        def initializer() -> tf.keras.initializers.Initializer:
            return tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=math.sqrt(modnet_weight_variance),
            )

        first = tf.keras.layers.Dense(
            hidden_width,
            activation=activation,
            kernel_initializer=initializer(),
            bias_initializer="zeros",
            name="hidden_1",
        )(features)
        second = tf.keras.layers.Dense(
            hidden_width,
            activation=activation,
            kernel_initializer=initializer(),
            bias_initializer="zeros",
            name="hidden_2",
        )(first)
        hidden = tf.keras.layers.Add(name="hidden_skip")([first, second]) if modnet_skip else second
        output = tf.keras.layers.Dense(
            1,
            use_bias=False,
            kernel_initializer=initializer(),
            name="potential",
        )(hidden)
        network = tf.keras.Model(inputs=inputs, outputs=output, name="quintic_modnet")
        model = PhiFSModel(network, basis, alpha=[1.0] * 5)
        return network, model

    network = tf.keras.Sequential(name="quintic_phi_network")
    network.add(tf.keras.Input(shape=(n_input,), dtype=tf.float32))
    for _ in range(hidden_layers):
        network.add(tf.keras.layers.Dense(hidden_width, activation=activation))
    network.add(tf.keras.layers.Dense(1, use_bias=False))
    model = PhiFSModel(network, basis, alpha=[1.0] * 5)
    return network, model


def modnet_feature_preflight(
    network: tf.keras.Model,
    points: np.ndarray,
    order: int,
) -> dict[str, float | int]:
    sample = np.asarray(points[: min(128, len(points))], dtype=np.float32)
    if sample.ndim != 2 or sample.shape[1] % 2:
        raise ValueError("ModNet preflight requires concatenated real/imaginary points")
    layer = network.get_layer("symmetric_power_features")
    actual = np.asarray(layer(tf.convert_to_tensor(sample)).numpy(), dtype=np.float64)

    n_complex = sample.shape[1] // 2
    complex_points = sample[:, :n_complex] + 1j * sample[:, n_complex:]
    scale = 1.7 * np.exp(0.37j)
    scaled_points = complex_points * scale
    scaled_input = np.concatenate(
        (scaled_points.real, scaled_points.imag), axis=-1
    ).astype(np.float32)
    scaled_features = np.asarray(
        layer(tf.convert_to_tensor(scaled_input)).numpy(), dtype=np.float64
    )

    absolute_squared = np.abs(complex_points) ** 2
    normalized = absolute_squared / np.sum(absolute_squared, axis=-1, keepdims=True)
    power_sums = np.stack(
        [np.sum(normalized**k, axis=-1) for k in range(1, order + 1)],
        axis=-1,
    )
    expected = np.power(
        power_sums[:, :, None],
        np.arange(1, order + 1, dtype=np.float64)[None, None, :] / order,
    ).reshape(len(sample), order * order)
    direct_error = float(np.max(np.abs(actual - expected)))
    scaling_error = float(np.max(np.abs(actual - scaled_features)))
    s1_error = float(np.max(np.abs(actual[:, :order] - 1.0)))
    if direct_error > 2.0e-6 or scaling_error > 2.0e-6 or s1_error > 2.0e-6:
        raise RuntimeError(
            "ModNet feature normalization preflight failed: "
            f"direct={direct_error}, scaling={scaling_error}, s1={s1_error}"
        )
    return {
        "sample_points": len(sample),
        "feature_dimension": order * order,
        "maximum_direct_formula_error": direct_error,
        "maximum_projective_scaling_error": scaling_error,
        "maximum_s1_identity_error": s1_error,
    }


def generate_blind_test_points(
    n_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = make_point_generator(seed)
    blocks = []
    remaining = n_points
    while remaining > 0:
        block = generator.generate_point_weights(
            max(remaining, 128), omega=True, normalize_to_vol_j=True
        )
        if len(block) == 0:
            raise RuntimeError("cymetric generated no valid test points")
        blocks.append(block[:remaining])
        remaining -= min(len(block), remaining)
    point_weights = np.concatenate(blocks, axis=0)[:n_points]
    complex_points = point_weights["point"]
    points = np.concatenate((complex_points.real, complex_points.imag), axis=-1).astype(
        np.float32
    )
    weights = np.asarray(point_weights["weight"], dtype=np.float64)
    omega = np.asarray(
        np.real(point_weights["omega"] * np.conj(point_weights["omega"])),
        dtype=np.float64,
    )
    return points, weights, omega, complex_points


def load_blind_test_points(
    source_run_dir: Path,
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    path = source_run_dir / "blind_points.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    stored = np.load(path, allow_pickle=False)
    points = np.asarray(stored["X"], dtype=np.float32)
    weights = np.asarray(stored["weights"], dtype=np.float64)
    omega = np.asarray(stored["omega_squared"], dtype=np.float64)
    complex_points = np.asarray(stored["complex_points"], dtype=np.complex128)
    if not (
        len(points)
        == len(weights)
        == len(omega)
        == len(complex_points)
        == expected_count
    ):
        raise RuntimeError("stored blind-point arrays have an unexpected length")
    reconstructed = np.concatenate(
        (complex_points.real, complex_points.imag), axis=-1
    ).astype(np.float32)
    if not np.array_equal(points, reconstructed):
        raise RuntimeError("stored real and complex blind-point arrays disagree")
    return points, weights, omega, complex_points, sha256_file(path)


def evaluate_determinants(
    model: PhiFSModel,
    points: np.ndarray,
    batch_size: int,
    fs_baseline: bool,
) -> tuple[np.ndarray, np.ndarray]:
    determinants = np.empty(len(points), dtype=np.float64)
    min_eigenvalues = np.empty(len(points), dtype=np.float64)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, points.shape[1]), dtype=tf.float32)],
        reduce_retracing=True,
    )
    def evaluate_batch(batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if fs_baseline:
            metric = model.fubini_study_pb(batch)
        else:
            metric = model(batch, training=False)
        determinant = tf.math.real(tf.linalg.det(metric))
        eigenvalues = tf.math.real(tf.linalg.eigvalsh(metric))
        return determinant, tf.reduce_min(eigenvalues, axis=-1)

    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        batch = tf.convert_to_tensor(points[start:stop], dtype=tf.float32)
        batch_determinants, batch_min_eigenvalues = evaluate_batch(batch)
        determinants[start:stop] = batch_determinants.numpy()
        min_eigenvalues[start:stop] = batch_min_eigenvalues.numpy()
    return determinants, min_eigenvalues


class NetworkCheckpoint(tf.keras.callbacks.Callback):
    """Persist the underlying potential network after every Volk phase."""

    def __init__(self, weights_path: Path, progress_path: Path) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.progress_path = progress_path
        self.completed_phases = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.completed_phases += 1
        self.model.model.save_weights(self.weights_path)
        write_json(
            self.progress_path,
            {
                "completed_volk_phases": self.completed_phases,
                "inner_epoch_index": int(epoch),
                "logs": logs or {},
            },
        )


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray = PROBABILITIES,
) -> np.ndarray:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        return np.full_like(probabilities, np.nan)
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, values, left=values[0], right=values[-1])


def named_quantiles(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    result = weighted_quantiles(values, weights)
    return {f"q{probability:.4f}": float(value) for probability, value in zip(PROBABILITIES, result)}


def cvar(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    threshold = weighted_quantiles(values, weights, np.array([probability]))[0]
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (values >= threshold)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def tail_entry(mask: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    valid_weights = np.isfinite(weights) & (weights > 0)
    denominator = np.sum(weights[valid_weights])
    count = int(np.count_nonzero(mask))
    return {
        "count": count,
        "point_fraction": float(count / len(mask)),
        "weighted_mass": float(np.sum(weights[mask & valid_weights]) / denominator),
    }


def ratio_statistics(
    determinants: np.ndarray,
    weights: np.ndarray,
    omega: np.ndarray,
    min_eigenvalues: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    finite_inputs = (
        np.isfinite(determinants)
        & np.isfinite(weights)
        & np.isfinite(omega)
        & (weights > 0)
        & (omega > 0)
    )
    if not np.any(finite_inputs):
        raise RuntimeError("no finite positive-weight points remain for the tail audit")
    quotient = np.full_like(determinants, np.nan, dtype=np.float64)
    quotient[finite_inputs] = determinants[finite_inputs] / omega[finite_inputs]
    volume_cy = float(np.mean(weights[finite_inputs]))
    volume_k = float(np.mean(quotient[finite_inputs] * weights[finite_inputs]))
    normalization = volume_cy / volume_k
    ratio = quotient * normalization
    residual = np.abs(1.0 - ratio)
    weight_sum = np.sum(weights[finite_inputs])
    sigma = float(np.sum(residual[finite_inputs] * weights[finite_inputs]) / weight_sum)
    rms = float(
        np.sqrt(np.sum(np.square(residual[finite_inputs]) * weights[finite_inputs]) / weight_sum)
    )
    positive = finite_inputs & (ratio > 0)
    log_abs = np.full_like(ratio, np.nan)
    log_abs[positive] = np.abs(np.log(ratio[positive]))
    statistics = {
        "n_points": int(len(ratio)),
        "n_finite_inputs": int(np.count_nonzero(finite_inputs)),
        "volume_cy_mc": volume_cy,
        "volume_k_mc": volume_k,
        "normalization_volume_cy_over_volume_k": normalization,
        "sigma_official_formula": sigma,
        "weighted_rms_abs_residual": rms,
        "ratio_weighted_quantiles": named_quantiles(ratio, weights),
        "ratio_unweighted_quantiles": {
            f"q{probability:.4f}": float(value)
            for probability, value in zip(PROBABILITIES, np.nanquantile(ratio, PROBABILITIES))
        },
        "abs_residual_weighted_quantiles": named_quantiles(residual, weights),
        "abs_log_ratio_weighted_quantiles_positive_only": named_quantiles(
            log_abs[positive], weights[positive]
        ),
        "abs_residual_weighted_cvar": {
            "cvar_0.9900": cvar(residual, weights, 0.99),
            "cvar_0.9990": cvar(residual, weights, 0.999),
            "cvar_0.9999": cvar(residual, weights, 0.9999),
        },
        "ratio_upper_tails": {
            str(threshold): tail_entry(ratio > threshold, weights)
            for threshold in RATIO_THRESHOLDS
        },
        "ratio_lower_tails": {
            str(threshold): tail_entry(ratio < threshold, weights)
            for threshold in LOW_RATIO_THRESHOLDS
        },
        "abs_residual_tails": {
            str(threshold): tail_entry(residual > threshold, weights)
            for threshold in ABS_RESIDUAL_THRESHOLDS
        },
        "nonpositive_determinant": tail_entry(determinants <= 0, weights),
        "nonpositive_min_eigenvalue": tail_entry(min_eigenvalues <= 0, weights),
        "nonfinite_determinant_count": int(np.count_nonzero(~np.isfinite(determinants))),
        "determinant_weighted_quantiles": named_quantiles(determinants, weights),
        "min_eigenvalue_weighted_quantiles": named_quantiles(min_eigenvalues, weights),
    }
    return statistics, ratio, residual


def worst_point_records(
    complex_points: np.ndarray,
    weights: np.ndarray,
    omega: np.ndarray,
    determinants: np.ndarray,
    min_eigenvalues: np.ndarray,
    ratio: np.ndarray,
    residual: np.ndarray,
    n_top: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    score = np.nan_to_num(residual, nan=-np.inf, posinf=np.inf, neginf=np.inf)
    indices = np.argsort(score)[-min(n_top, len(score)) :][::-1]
    records = []
    for rank, index in enumerate(indices, start=1):
        point = complex_points[index]
        abs_coordinates = np.abs(point)
        records.append(
            {
                "rank": rank,
                "index": int(index),
                "ratio": float(ratio[index]),
                "abs_residual": float(residual[index]),
                "weight": float(weights[index]),
                "omega_squared": float(omega[index]),
                "determinant": float(determinants[index]),
                "min_eigenvalue": float(min_eigenvalues[index]),
                "largest_coordinate_index": int(np.argmax(abs_coordinates)),
                "largest_coordinate_abs": float(np.max(abs_coordinates)),
                "smallest_coordinate_abs": float(np.min(abs_coordinates)),
                "quintic_residual_abs": float(np.abs(np.sum(np.power(point, 5)))),
            }
        )
    return records, indices


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    started = time.time()
    configuration = vars(args).copy()
    configuration["output_dir"] = str(output_dir)
    write_json(
        status_path,
        {"state": "running", "phase": "initializing", "configuration": configuration},
    )

    try:
        if args.data_source_run_dir is not None and args.force_data:
            raise ValueError("--force-data cannot be used with --data-source-run-dir")
        if args.preflight_only and args.dataset_only:
            raise ValueError("--preflight-only and --dataset-only are mutually exclusive")
        tf.keras.utils.set_random_seed(args.model_seed)
        environment = provenance()
        write_json(output_dir / "environment.json", environment)

        write_json(status_path, {"state": "running", "phase": "preparing_training_data"})
        data_source_run_dir = (
            output_dir
            if args.data_source_run_dir is None
            else args.data_source_run_dir.expanduser().resolve()
        )
        blind_source_run_dir = (
            args.data_source_run_dir
            if args.blind_source_run_dir is None
            else args.blind_source_run_dir
        )
        if blind_source_run_dir is not None:
            blind_source_run_dir = blind_source_run_dir.expanduser().resolve()
        data, basis, kappa, data_hashes = prepare_training_data(
            data_source_run_dir, args.train_points, args.train_seed, args.force_data
        )
        train_count = int(len(data["X_train"]))
        val_count = int(len(data["X_val"]))
        network, model = build_model(
            int(data["X_train"].shape[1]),
            args.hidden_layers,
            args.hidden_width,
            args.activation,
            basis,
            network_family=args.network_family,
            modnet_order=args.modnet_order,
            modnet_weight_variance=args.modnet_weight_variance,
            modnet_skip=not args.modnet_no_skip,
        )
        initial_parameter_count = int(network.count_params())
        feature_preflight = None
        if args.network_family == "modnet":
            feature_preflight = modnet_feature_preflight(
                network,
                data["X_train"],
                args.modnet_order,
            )
            expected_parameter_count = (
                args.modnet_order**2 * args.hidden_width
                + args.hidden_width
                + args.hidden_width**2
                + args.hidden_width
                + args.hidden_width
            )
            if initial_parameter_count != expected_parameter_count:
                raise RuntimeError(
                    "unexpected ModNet parameter count: "
                    f"{initial_parameter_count} != {expected_parameter_count}"
                )
        if args.dataset_only:
            report = {
                "schema": "quintic-training-dataset-v1",
                "state": "complete",
                "geometry": "Fermat quintic hypersurface X_5 in P^4",
                "data": {
                    "requested_point_count": int(args.train_points),
                    "training_count": train_count,
                    "validation_count": val_count,
                    "train_seed": int(args.train_seed),
                    **data_hashes,
                },
                "network": {
                    "family": args.network_family,
                    "parameter_count": initial_parameter_count,
                    "trained": False,
                },
                "blind_test": None,
                "environment": environment,
            }
            report_path = output_dir / "report.json"
            write_json(report_path, report)
            write_json(
                status_path,
                {
                    "state": "complete",
                    "phase": "dataset_only",
                    "training_count": train_count,
                    "validation_count": val_count,
                    "report_sha256": sha256_file(report_path),
                },
            )
            print(json.dumps(json_value(report), indent=2, sort_keys=True))
            return
        if args.preflight_only:
            preflight = {
                "state": "complete",
                "network_family": args.network_family,
                "parameter_count": initial_parameter_count,
                "feature_preflight": feature_preflight,
                "environment": environment,
            }
            write_json(output_dir / "preflight.json", preflight)
            write_json(status_path, preflight)
            print(json.dumps(json_value(preflight), indent=2, sort_keys=True))
            return

        write_json(
            status_path,
            {
                "state": "running",
                "phase": "training",
                "train_count": train_count,
                "validation_count": val_count,
            },
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        checkpoint = NetworkCheckpoint(
            output_dir / "checkpoint.weights.h5",
            output_dir / "training_progress.json",
        )
        train_started = time.time()
        model, history = train_model(
            model,
            data,
            optimizer=optimizer,
            epochs=args.epochs,
            batch_sizes=[args.batch_size, args.volk_batch_size],
            verbose=1,
            custom_metrics=[],
            callbacks=[checkpoint],
        )
        train_seconds = time.time() - train_started
        weights_path = output_dir / "phi_network.weights.h5"
        network.save_weights(weights_path)
        write_json(output_dir / "training_history.json", history)

        write_json(status_path, {"state": "running", "phase": "generating_blind_test"})
        test_started = time.time()
        blind_points_sha256 = None
        if blind_source_run_dir is None:
            points, weights, omega, complex_points = generate_blind_test_points(
                args.test_points, args.test_seed
            )
            blind_points_path = output_dir / "blind_points.npz"
            np.savez_compressed(
                blind_points_path,
                X=points,
                weights=weights,
                omega_squared=omega,
                complex_points=complex_points,
            )
            blind_points_sha256 = sha256_file(blind_points_path)
        else:
            (
                points,
                weights,
                omega,
                complex_points,
                blind_points_sha256,
            ) = load_blind_test_points(blind_source_run_dir, args.test_points)
        generation_seconds = time.time() - test_started

        write_json(status_path, {"state": "running", "phase": "evaluating_trained_model"})
        evaluation_started = time.time()
        determinants, min_eigenvalues = evaluate_determinants(
            model, points, args.eval_batch_size, fs_baseline=False
        )
        trained_statistics, ratio, residual = ratio_statistics(
            determinants, weights, omega, min_eigenvalues
        )
        records, worst_indices = worst_point_records(
            complex_points,
            weights,
            omega,
            determinants,
            min_eigenvalues,
            ratio,
            residual,
            args.top_points,
        )

        baseline_statistics = None
        baseline_determinants = np.empty(0, dtype=np.float64)
        baseline_min_eigenvalues = np.empty(0, dtype=np.float64)
        baseline_ratio = np.empty(0, dtype=np.float64)
        if not args.skip_fs_baseline:
            write_json(status_path, {"state": "running", "phase": "evaluating_fs_baseline"})
            baseline_determinants, baseline_min_eigenvalues = evaluate_determinants(
                model, points, args.eval_batch_size, fs_baseline=True
            )
            baseline_statistics, baseline_ratio, _ = ratio_statistics(
                baseline_determinants, weights, omega, baseline_min_eigenvalues
            )
        evaluation_seconds = time.time() - evaluation_started

        np.savez_compressed(
            output_dir / "blind_test_tail_arrays.npz",
            weights=weights,
            omega_squared=omega,
            determinant=determinants,
            min_eigenvalue=min_eigenvalues,
            normalized_ratio=ratio,
            abs_residual=residual,
            fs_determinant=baseline_determinants,
            fs_min_eigenvalue=baseline_min_eigenvalues,
            fs_normalized_ratio=baseline_ratio,
        )
        np.savez_compressed(
            output_dir / "worst_points.npz",
            indices=worst_indices,
            points=complex_points[worst_indices],
            weights=weights[worst_indices],
            omega_squared=omega[worst_indices],
            determinant=determinants[worst_indices],
            min_eigenvalue=min_eigenvalues[worst_indices],
            normalized_ratio=ratio[worst_indices],
            abs_residual=residual[worst_indices],
        )

        report = {
            "schema": "cymetric-quintic-tail-audit-v1",
            "scientific_scope": {
                "geometry": "Fermat quintic hypersurface X_5 in P^4",
                "equation": "z0^5 + z1^5 + z2^5 + z3^5 + z4^5 = 0",
                "model": (
                    "official cymetric PhiFSModel with a ModNet-style potential network"
                    if args.network_family == "modnet"
                    else "official cymetric PhiFSModel"
                ),
                "compatibility_note": (
                    "The checked-out cymetric release declares TensorFlow >=2.7,<2.15; "
                    "this run uses the recorded host TensorFlow version and is accepted only "
                    "after a complete forward/backward smoke test."
                ),
                "claim_limit": (
                    "Finite independent sampling can detect observed tails but cannot prove "
                    "the global absence of arbitrarily narrow spikes."
                ),
            },
            "configuration": configuration,
            "environment": environment,
            "data": {
                "source_run_dir": str(data_source_run_dir),
                "training_source_run_dir": str(data_source_run_dir),
                "blind_source_run_dir": (
                    str(blind_source_run_dir)
                    if blind_source_run_dir is not None
                    else str(output_dir)
                ),
                "train_count": train_count,
                "validation_count": val_count,
                "blind_test_count": int(len(points)),
                "kappa": kappa,
                "blind_points_sha256": blind_points_sha256,
                **data_hashes,
            },
            "network": {
                "input_dimension": int(points.shape[1]),
                "network_family": args.network_family,
                "feature_definition": (
                    "s_k^(j/order), s_k=sum_i |Z_i|^(2k)/(sum_i |Z_i|^2)^k"
                    if args.network_family == "modnet"
                    else None
                ),
                "feature_dimension": (
                    args.modnet_order**2 if args.network_family == "modnet" else int(points.shape[1])
                ),
                "modnet_order": args.modnet_order if args.network_family == "modnet" else None,
                "modnet_skip_connection": (
                    not args.modnet_no_skip if args.network_family == "modnet" else None
                ),
                "modnet_weight_variance": (
                    args.modnet_weight_variance if args.network_family == "modnet" else None
                ),
                "hidden_layers": args.hidden_layers,
                "hidden_width": args.hidden_width,
                "activation": args.activation,
                "output_dimension": 1,
                "use_output_bias": False,
                "parameter_count": initial_parameter_count,
                "feature_preflight": feature_preflight,
                "weights_sha256": sha256_file(weights_path),
            },
            "timing_seconds": {
                "training": train_seconds,
                "blind_test_generation": generation_seconds,
                "blind_test_evaluation_total": evaluation_seconds,
                "wall_total": time.time() - started,
            },
            "trained_phi_model": trained_statistics,
            "fubini_study_baseline": baseline_statistics,
            "worst_points_by_abs_residual": records,
        }
        report_path = output_dir / "report.json"
        write_json(report_path, report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "wall_seconds": time.time() - started,
            },
        )
        print(json.dumps(json_value(report), indent=2, sort_keys=True, allow_nan=False))
    except Exception as error:
        failure = {
            "state": "failed",
            "phase": "exception",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.time() - started,
        }
        write_json(status_path, failure)
        raise


if __name__ == "__main__":
    main()
