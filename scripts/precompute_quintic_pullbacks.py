#!/usr/bin/env python3
"""Precompute cymetric pullbacks for the common Fermat-quintic point sets."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from cymetric.models.fubinistudy import FSModel  # noqa: E402
from cymetric.models.tfhelper import prepare_tf_basis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument(
        "--blind-reference-run-dir",
        type=Path,
        help=(
            "optional run containing blind_points.npz and an FS-populated "
            "blind_test_tail_arrays.npz; training data still come from "
            "--source-run-dir"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--validation-fs-count", type=int, default=512)
    parser.add_argument("--determinant-absolute-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--determinant-relative-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--eigenvalue-absolute-tolerance", type=float, default=5.0e-7)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse shape- and dtype-checked pullback arrays already in output-dir",
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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pullbacks(
    model: FSModel,
    x_values: np.ndarray,
    path: Path,
    batch_size: int,
) -> None:
    nfold = int(model.nfold)
    n_coordinates = x_values.shape[1] // 2
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.complex64,
        shape=(len(x_values), nfold, n_coordinates),
    )
    for start in range(0, len(x_values), batch_size):
        stop = min(start + batch_size, len(x_values))
        tensor = tf.convert_to_tensor(x_values[start:stop], dtype=tf.float32)
        output[start:stop] = model.pullbacks(tensor).numpy().astype(np.complex64)
        if stop == len(x_values) or stop % (10 * batch_size) == 0:
            print(f"{path.name}: {stop}/{len(x_values)}", flush=True)
    output.flush()


def evaluate_fs(
    model: FSModel,
    x_values: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    determinants = np.empty(len(x_values), dtype=np.float64)
    minimum_eigenvalues = np.empty(len(x_values), dtype=np.float64)
    for start in range(0, len(x_values), batch_size):
        stop = min(start + batch_size, len(x_values))
        tensor = tf.convert_to_tensor(x_values[start:stop], dtype=tf.float32)
        metrics = model.fubini_study_pb(tensor)
        determinants[start:stop] = tf.math.real(tf.linalg.det(metrics)).numpy()
        minimum_eigenvalues[start:stop] = tf.reduce_min(
            tf.math.real(tf.linalg.eigvalsh(metrics)), axis=1
        ).numpy()
    return determinants, minimum_eigenvalues


def main() -> None:
    args = parse_args()
    if min(
        args.determinant_absolute_tolerance,
        args.determinant_relative_tolerance,
        args.eigenvalue_absolute_tolerance,
    ) <= 0:
        raise ValueError("FS reproduction tolerances must be positive")
    started = time.perf_counter()
    source_dir = args.source_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})

    try:
        dataset_path = source_dir / "training_data" / "dataset.npz"
        basis_path = source_dir / "training_data" / "basis.pickle"
        blind_source_dir = (
            args.blind_reference_run_dir.resolve()
            if args.blind_reference_run_dir is not None
            else source_dir
        )
        blind_path = blind_source_dir / "blind_points.npz"
        source_tail_path = blind_source_dir / "blind_test_tail_arrays.npz"
        for path in (dataset_path, basis_path, blind_path, source_tail_path):
            if not path.exists():
                raise FileNotFoundError(path)

        dataset = np.load(dataset_path, allow_pickle=False)
        train_x = np.asarray(dataset["X_train"], dtype=np.float32)
        validation_x = np.asarray(dataset["X_val"], dtype=np.float32)
        blind = np.load(blind_path, allow_pickle=False)
        blind_x = np.asarray(blind["X"], dtype=np.float32)

        raw_basis = np.load(basis_path, allow_pickle=True)
        model = FSModel(prepare_tf_basis(raw_basis))
        splits = {
            "train": train_x,
            "validation": validation_x,
            "blind": blind_x,
        }
        output_paths: dict[str, Path] = {}
        for name, values in splits.items():
            write_json(
                status_path,
                {"state": "running", "phase": "pullbacks", "split": name},
            )
            path = output_dir / f"{name}_pullbacks.npy"
            expected_shape = (len(values), int(model.nfold), values.shape[1] // 2)
            reusable = False
            if args.reuse_existing and path.exists():
                existing = np.load(path, mmap_mode="r")
                reusable = existing.shape == expected_shape and existing.dtype == np.complex64
                del existing
            if reusable:
                print(f"{path.name}: reusing verified {expected_shape}", flush=True)
            else:
                write_pullbacks(model, values, path, args.batch_size)
            output_paths[name] = path

        fs_count = min(args.validation_fs_count, len(validation_x))
        fs_tensor = tf.convert_to_tensor(validation_x[:fs_count], dtype=tf.float32)
        fs_metrics = model.fubini_study_pb(fs_tensor).numpy().astype(np.complex64)
        fs_path = output_dir / "validation_official_fs_metrics.npy"
        np.save(fs_path, fs_metrics, allow_pickle=False)

        rechecked_determinants, rechecked_minimum_eigenvalues = evaluate_fs(
            model, blind_x, args.batch_size
        )
        source_tail = np.load(source_tail_path, allow_pickle=False)
        source_determinants = np.asarray(source_tail["fs_determinant"], dtype=np.float64)
        source_minimum_eigenvalues = np.asarray(
            source_tail["fs_min_eigenvalue"], dtype=np.float64
        )
        determinant_difference = np.abs(rechecked_determinants - source_determinants)
        determinant_relative_difference = determinant_difference / np.maximum(
            np.abs(source_determinants), np.finfo(np.float64).tiny
        )
        eigenvalue_difference = np.abs(
            rechecked_minimum_eigenvalues - source_minimum_eigenvalues
        )
        source_reproduction = {
            "determinant_maximum_absolute_difference": float(
                np.max(determinant_difference)
            ),
            "determinant_maximum_relative_difference": float(
                np.max(determinant_relative_difference)
            ),
            "minimum_eigenvalue_maximum_absolute_difference": float(
                np.max(eigenvalue_difference)
            ),
            "tolerances": {
                "determinant_absolute": args.determinant_absolute_tolerance,
                "determinant_relative": args.determinant_relative_tolerance,
                "minimum_eigenvalue_absolute": args.eigenvalue_absolute_tolerance,
            },
        }
        determinant_failed = (
            source_reproduction["determinant_maximum_absolute_difference"]
            > args.determinant_absolute_tolerance
            and source_reproduction["determinant_maximum_relative_difference"]
            > args.determinant_relative_tolerance
        )
        eigenvalue_failed = (
            source_reproduction["minimum_eigenvalue_maximum_absolute_difference"]
            > args.eigenvalue_absolute_tolerance
        )
        if determinant_failed or eigenvalue_failed:
            raise RuntimeError(
                "regenerated blind points do not reproduce the source FS determinants: "
                f"{source_reproduction}"
            )

        report = {
            "schema": "quintic-common-pullbacks-v1",
            "source_run_dir": source_dir,
            "blind_reference_run_dir": blind_source_dir,
            "counts": {name: len(values) for name, values in splits.items()},
            "shapes": {
                name: list(np.load(path, mmap_mode="r").shape)
                for name, path in output_paths.items()
            },
            "validation_official_fs_metric_count": fs_count,
            "source_fs_blind_reproduction": source_reproduction,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "tensorflow": tf.__version__,
            },
            "source_sha256": {
                "dataset": sha256_file(dataset_path),
                "basis": sha256_file(basis_path),
                "blind_points": sha256_file(blind_path),
                "cymetric_tail": sha256_file(source_tail_path),
            },
            "output_sha256": {
                name: sha256_file(path) for name, path in output_paths.items()
            }
            | {"validation_official_fs_metrics": sha256_file(fs_path)},
            "wall_seconds": time.perf_counter() - started,
        }
        report_path = output_dir / "report.json"
        write_json(report_path, report)
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
