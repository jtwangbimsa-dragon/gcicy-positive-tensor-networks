#!/usr/bin/env python3
"""Generate disjoint cymetric point pools for the generic-quintic kill test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
from typing import Any

import numpy as np


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf  # noqa: E402
from cymetric.models.fubinistudy import FSModel  # noqa: E402
from cymetric.models.tfhelper import prepare_tf_basis  # noqa: E402
from cymetric.pointgen.pointgen import PointGenerator  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.generic_quintic import (  # noqa: E402
    GENERIC_QUINTIC_COEFFICIENTS,
    GENERIC_QUINTIC_EXPONENTS,
    experiment_manifest,
    quintic_value,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-points", type=int, default=100_000)
    parser.add_argument("--teacher-validation-fraction", type=float, default=0.1)
    parser.add_argument("--native-points", type=int, default=100_000)
    parser.add_argument("--selection-points", type=int, default=20_000)
    parser.add_argument("--confirmation-points", type=int, default=50_000)
    parser.add_argument("--blind-points", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=202607262)
    parser.add_argument("--batch-size", type=int, default=2048)
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
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_generator(seed: int) -> PointGenerator:
    coefficients = GENERIC_QUINTIC_COEFFICIENTS.astype(np.complex128)
    coefficients = coefficients / np.max(np.abs(coefficients))
    generator = PointGenerator(
        GENERIC_QUINTIC_EXPONENTS,
        coefficients,
        np.ones(1, dtype=np.float64),
        np.asarray([4], dtype=np.int64),
    )
    generator._set_seed(seed)
    random.seed(seed)
    return generator


def generate_points(
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = make_generator(seed)
    blocks = []
    remaining = count
    while remaining:
        block = generator.generate_point_weights(
            max(remaining, 128),
            omega=True,
            normalize_to_vol_j=True,
        )
        if len(block) == 0:
            raise RuntimeError("cymetric generated no valid generic-quintic points")
        take = min(remaining, len(block))
        blocks.append(block[:take])
        remaining -= take
    rows = np.concatenate(blocks, axis=0)
    complex_points = np.asarray(rows["point"], dtype=np.complex128)
    x_values = np.concatenate(
        (complex_points.real, complex_points.imag),
        axis=1,
    ).astype(np.float32)
    weights = np.asarray(rows["weight"], dtype=np.float64)
    omega_squared = np.asarray(
        np.real(rows["omega"] * np.conj(rows["omega"])),
        dtype=np.float64,
    )
    return x_values, weights, omega_squared, complex_points


def relative_polynomial_residual(complex_points: np.ndarray) -> np.ndarray:
    monomial_scales = np.prod(
        np.power(
            np.abs(complex_points)[:, None, :],
            GENERIC_QUINTIC_EXPONENTS[None, :, :],
        ),
        axis=-1,
    )
    scale = monomial_scales @ np.abs(GENERIC_QUINTIC_COEFFICIENTS)
    return np.abs(quintic_value(complex_points)) / np.maximum(
        scale,
        np.finfo(np.float64).tiny,
    )


def write_pullbacks_and_fs(
    model: FSModel,
    x_values: np.ndarray,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    pullbacks_path = output_dir / "pullbacks.npy"
    fs_path = output_dir / "fs_reference.npz"
    pullbacks = np.lib.format.open_memmap(
        pullbacks_path,
        mode="w+",
        dtype=np.complex64,
        shape=(
            len(x_values),
            int(model.nfold),
            x_values.shape[1] // 2,
        ),
    )
    determinants = np.empty(len(x_values), dtype=np.float64)
    minimum_eigenvalues = np.empty(len(x_values), dtype=np.float64)
    for start in range(0, len(x_values), batch_size):
        stop = min(start + batch_size, len(x_values))
        tensor = tf.convert_to_tensor(x_values[start:stop], dtype=tf.float32)
        pullbacks[start:stop] = (
            model.pullbacks(tensor).numpy().astype(np.complex64)
        )
        metric = model.fubini_study_pb(tensor)
        determinants[start:stop] = tf.math.real(tf.linalg.det(metric)).numpy()
        minimum_eigenvalues[start:stop] = tf.reduce_min(
            tf.math.real(tf.linalg.eigvalsh(metric)),
            axis=1,
        ).numpy()
        if stop == len(x_values) or stop % (10 * batch_size) == 0:
            print(
                f"{output_dir.name}: pullbacks {stop}/{len(x_values)}",
                flush=True,
            )
    pullbacks.flush()
    np.savez_compressed(
        fs_path,
        determinant=determinants,
        minimum_eigenvalue=minimum_eigenvalues,
    )
    if not (
        np.all(np.isfinite(determinants))
        and np.all(determinants > 0)
        and np.all(np.isfinite(minimum_eigenvalues))
        and np.all(minimum_eigenvalues > 0)
    ):
        raise FloatingPointError("official FS reference is not finite and positive")
    return {
        "pullbacks": str(pullbacks_path),
        "pullbacks_sha256": sha256_file(pullbacks_path),
        "fs_reference": str(fs_path),
        "fs_reference_sha256": sha256_file(fs_path),
        "minimum_fs_determinant": float(np.min(determinants)),
        "minimum_fs_metric_eigenvalue": float(np.min(minimum_eigenvalues)),
    }


def write_pool(
    model: FSModel,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    x_values, weights, omega_squared, complex_points = generate_points(count, seed)
    points_path = output_dir / "points.npz"
    np.savez_compressed(
        points_path,
        X=x_values,
        weights=weights,
        omega_squared=omega_squared,
        complex_points=complex_points,
    )
    residual = relative_polynomial_residual(complex_points)
    if not (
        np.all(np.isfinite(weights))
        and np.all(weights > 0)
        and np.all(np.isfinite(omega_squared))
        and np.all(omega_squared > 0)
        and np.all(np.isfinite(residual))
    ):
        raise FloatingPointError("generated point data are invalid")
    geometry = write_pullbacks_and_fs(
        model,
        x_values,
        output_dir,
        batch_size,
    )
    report = {
        "count": count,
        "seed": seed,
        "points": str(points_path),
        "points_sha256": sha256_file(points_path),
        "maximum_relative_polynomial_residual": float(np.max(residual)),
        "mean_relative_polynomial_residual": float(np.mean(residual)),
        "minimum_weight": float(np.min(weights)),
        "minimum_omega_squared": float(np.min(omega_squared)),
        **geometry,
    }
    write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    args = parse_args()
    counts = {
        "teacher": args.teacher_points,
        "native": args.native_points,
        "selection": args.selection_points,
        "confirmation": args.confirmation_points,
        "blind": args.blind_points,
    }
    if (
        any(value <= 0 for value in counts.values())
        or args.batch_size <= 0
        or not 0 < args.teacher_validation_fraction < 1
    ):
        raise ValueError("counts, batch size, and validation fraction are invalid")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite the frozen data protocol")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    started = time.perf_counter()
    write_json(status_path, {"state": "running", "phase": "teacher_dataset"})
    try:
        training_dir = output_dir / "teacher_training_data"
        training_dir.mkdir()
        generator = make_generator(args.seed)
        kappa = float(
            generator.prepare_dataset(
                args.teacher_points,
                str(training_dir),
                val_split=args.teacher_validation_fraction,
            )
        )
        generator.prepare_basis(str(training_dir), kappa=kappa)
        dataset_path = training_dir / "dataset.npz"
        basis_path = training_dir / "basis.pickle"
        raw_basis = np.load(basis_path, allow_pickle=True)
        model = FSModel(prepare_tf_basis(raw_basis))
        dataset = np.load(dataset_path, allow_pickle=False)

        teacher_geometry = {}
        for split_name, array_name in (
            ("train", "X_train"),
            ("validation", "X_val"),
        ):
            split_dir = output_dir / f"teacher_{split_name}"
            split_dir.mkdir()
            x_values = np.asarray(dataset[array_name], dtype=np.float32)
            complex_points = (
                x_values[:, :5].astype(np.float64)
                + 1j * x_values[:, 5:].astype(np.float64)
            )
            residual = relative_polynomial_residual(complex_points)
            teacher_geometry[split_name] = {
                "count": len(x_values),
                "maximum_relative_polynomial_residual": float(np.max(residual)),
                **write_pullbacks_and_fs(
                    model,
                    x_values,
                    split_dir,
                    args.batch_size,
                ),
            }

        pool_reports = {}
        for offset, (name, count) in enumerate(
            (
                ("native", args.native_points),
                ("selection", args.selection_points),
                ("confirmation", args.confirmation_points),
                ("blind", args.blind_points),
            ),
            start=1,
        ):
            write_json(
                status_path,
                {"state": "running", "phase": f"{name}_pool"},
            )
            pool_reports[name] = write_pool(
                model,
                output_dir / name,
                count=count,
                seed=args.seed + offset,
                batch_size=args.batch_size,
            )

        manifest = {
            "schema": "generic-quintic-kill-test-data-v1",
            "geometry": experiment_manifest(),
            "configuration": {
                **counts,
                "teacher_validation_fraction": args.teacher_validation_fraction,
                "base_seed": args.seed,
                "batch_size": args.batch_size,
            },
            "teacher": {
                "kappa": kappa,
                "dataset": str(dataset_path),
                "dataset_sha256": sha256_file(dataset_path),
                "basis": str(basis_path),
                "basis_sha256": sha256_file(basis_path),
                "splits": teacher_geometry,
            },
            "pools": pool_reports,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "tensorflow": tf.__version__,
                "visible_gpus": [
                    device.name for device in tf.config.get_visible_devices("GPU")
                ],
            },
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "manifest.json", manifest)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "wall_seconds": time.perf_counter() - started,
                "manifest_sha256": sha256_file(output_dir / "manifest.json"),
            },
        )
        print(json.dumps(json_value(manifest), indent=2, sort_keys=True))
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()

