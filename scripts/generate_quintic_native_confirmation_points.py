#!/usr/bin/env python3
"""Generate and freeze fresh Fermat-quintic points with official pullbacks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cymetric_quintic_tail_experiment import (  # noqa: E402
    generate_blind_test_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=25_000)
    parser.add_argument("--seed", type=int, required=True)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.batch_size <= 0:
        raise ValueError("point count and batch size must be positive")
    source_dir = args.source_run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a generated point set")
    basis_path = source_dir / "training_data" / "basis.pickle"
    if not basis_path.exists():
        raise FileNotFoundError(basis_path)
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    points_path = output_dir / "points.npz"
    pullbacks_path = output_dir / "pullbacks.npy"
    fs_path = output_dir / "fs_reference.npz"
    report_path = output_dir / "report.json"
    started = time.perf_counter()
    write_json(status_path, {"state": "running", "phase": "generating_points"})

    try:
        points, weights, omega, complex_points = generate_blind_test_points(
            args.count,
            args.seed,
        )
        if not (
            len(points)
            == len(weights)
            == len(omega)
            == len(complex_points)
            == args.count
        ):
            raise RuntimeError("generated point arrays have inconsistent lengths")
        if not (
            np.all(np.isfinite(points))
            and np.all(np.isfinite(weights))
            and np.all(weights > 0)
            and np.all(np.isfinite(omega))
            and np.all(omega > 0)
            and np.all(np.isfinite(complex_points))
        ):
            raise FloatingPointError("generated point arrays are not finite and positive")
        np.savez_compressed(
            points_path,
            X=np.asarray(points, dtype=np.float32),
            weights=np.asarray(weights, dtype=np.float64),
            omega_squared=np.asarray(omega, dtype=np.float64),
            complex_points=np.asarray(complex_points, dtype=np.complex128),
        )

        raw_basis = np.load(basis_path, allow_pickle=True)
        model = FSModel(prepare_tf_basis(raw_basis))
        nfold = int(model.nfold)
        coordinate_count = points.shape[1] // 2
        pullbacks = np.lib.format.open_memmap(
            pullbacks_path,
            mode="w+",
            dtype=np.complex64,
            shape=(args.count, nfold, coordinate_count),
        )
        determinants = np.empty(args.count, dtype=np.float64)
        minimum_eigenvalues = np.empty(args.count, dtype=np.float64)
        for start in range(0, args.count, args.batch_size):
            stop = min(start + args.batch_size, args.count)
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "official_pullbacks",
                    "completed": stop,
                    "count": args.count,
                },
            )
            tensor = tf.convert_to_tensor(points[start:stop], dtype=tf.float32)
            pullbacks[start:stop] = (
                model.pullbacks(tensor).numpy().astype(np.complex64)
            )
            metrics = model.fubini_study_pb(tensor)
            determinants[start:stop] = tf.math.real(
                tf.linalg.det(metrics)
            ).numpy()
            minimum_eigenvalues[start:stop] = tf.reduce_min(
                tf.math.real(tf.linalg.eigvalsh(metrics)),
                axis=1,
            ).numpy()
            print(f"official_pullbacks={stop}/{args.count}", flush=True)
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

        polynomial = np.sum(np.power(complex_points, 5), axis=1)
        polynomial_scale = np.sum(np.power(np.abs(complex_points), 5), axis=1)
        relative_polynomial_residual = np.abs(polynomial) / np.maximum(
            polynomial_scale,
            np.finfo(np.float64).tiny,
        )
        report = {
            "schema": "quintic-native-confirmation-points-v1",
            "purpose": (
                "A preregistered fresh point set split into 5000 confirmation "
                "points followed by 20000 tail-gate points."
            ),
            "configuration": {
                "count": args.count,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "tensorflow_visible_gpus": [
                    item.name for item in tf.config.get_visible_devices("GPU")
                ],
            },
            "quality": {
                "maximum_relative_quintic_residual": float(
                    np.max(relative_polynomial_residual)
                ),
                "minimum_weight": float(np.min(weights)),
                "minimum_omega_squared": float(np.min(omega)),
                "minimum_fs_determinant": float(np.min(determinants)),
                "minimum_fs_metric_eigenvalue": float(
                    np.min(minimum_eigenvalues)
                ),
            },
            "source": {
                "source_run_dir": source_dir,
                "basis": basis_path,
                "basis_sha256": sha256_file(basis_path),
            },
            "artifacts": {
                "points": points_path,
                "points_sha256": sha256_file(points_path),
                "pullbacks": pullbacks_path,
                "pullbacks_sha256": sha256_file(pullbacks_path),
                "fs_reference": fs_path,
                "fs_reference_sha256": sha256_file(fs_path),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "tensorflow": tf.__version__,
            },
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(report_path, json_value(report))
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report_sha256": sha256_file(report_path),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps(json_value(report), indent=2, sort_keys=True), flush=True)
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
