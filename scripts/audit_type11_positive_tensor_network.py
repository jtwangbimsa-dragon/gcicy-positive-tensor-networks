#!/usr/bin/env python3
"""Blind-audit a saved positive tensor-network metric."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    get_adapter,
    positive_tensor_network_from_artifact_payload,
)
from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    evaluate_log_eta_and_minimum_eigenvalue_arrays,
    evaluate_model,
    prepare_dataset,
    weighted_log_mean_exp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", help="registered adapter key; defaults to model metadata")
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--teacher-artifact", type=Path)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--seed", type=int, default=72203)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument(
        "--common-pool",
        type=Path,
        help="immutable common confirmation or blind point pool",
    )
    parser.add_argument(
        "--common-pool-split",
        choices=("train", "selection", "confirmation", "blind"),
        default="confirmation",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--sampling-cluster-size",
        type=int,
        help="complete fibre-root cluster size; defaults to model metadata",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--arrays-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for tensor-network audit") from exc

    args = parse_args()
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
    if (
        args.points <= 0
        or sampling_cluster_size <= 0
        or args.points % sampling_cluster_size
        or args.workers <= 0
    ):
        raise SystemExit(
            "points must be positive and divisible by sampling-cluster-size"
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
    teacher_path = (
        None
        if teacher_value is None
        else Path(teacher_value).expanduser().resolve()
    )
    if sha256_file(source_path) != payload["source_artifact_sha256"]:
        raise ValueError("source artifact SHA256 does not match the saved model")
    saved_teacher_sha256 = payload.get("teacher_artifact_sha256")
    if (
        teacher_path is not None
        and args.teacher_artifact is None
        and saved_teacher_sha256 is not None
        and sha256_file(teacher_path) != saved_teacher_sha256
    ):
        raise ValueError("teacher artifact SHA256 does not match the saved model")

    precision = str(payload["precision"])
    real_dtype = torch.float32 if precision == "complex64" else torch.float64
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    adapter = get_adapter(adapter_key)
    geometry = adapter.make_model(model_seed, exact=True)
    source = adapter.load_h_artifact(source_path, geometry)
    teacher = (
        None
        if teacher_path is None
        else adapter.load_h_artifact(teacher_path, geometry)
    )
    model = positive_tensor_network_from_artifact_payload(
        source.h_matrix,
        payload,
        device=device,
    )
    model.eval()

    dataset = prepare_dataset(
        adapter,
        geometry,
        source,
        teacher,
        count=args.points,
        seed=args.seed,
        workers=args.workers,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        sampling_cluster_size=sampling_cluster_size,
        common_pool_path=args.common_pool,
        common_pool_split=(
            args.common_pool_split if args.common_pool is not None else None
        ),
    )
    if payload.get("fixed_log_kappa") is not None:
        fixed_log_kappa = torch.tensor(
            float(payload["fixed_log_kappa"]),
            dtype=real_dtype,
            device=device,
        )
        fixed_log_kappa_source = payload.get(
            "fixed_log_kappa_source", "saved_model"
        )
    elif dataset["teacher_log_eta"] is not None:
        fixed_log_kappa = weighted_log_mean_exp(
            dataset["teacher_log_eta"],
            dataset["weights"],
        ).detach()
        fixed_log_kappa_source = "audit_teacher_pool"
    else:
        raise ValueError("model has neither saved kappa nor a teacher to estimate it")
    metrics = evaluate_model(
        model,
        dataset,
        fixed_log_kappa=fixed_log_kappa,
    )
    arrays_metadata = None
    if args.arrays_out is not None:
        arrays_path = args.arrays_out.expanduser().resolve()
        model_log_eta, model_minimum_eigenvalues = (
            evaluate_log_eta_and_minimum_eigenvalue_arrays(model, dataset)
        )
        point_payload = {
            key: np.asarray(value)
            for key, value in dataset["point_storage_payload"].items()
        }
        arrays_payload = {
            "schema": np.asarray("type11-positive-tensor-network-blind-arrays-v1"),
            "adapter": np.asarray(adapter.key),
            "model_log_eta": np.asarray(model_log_eta),
            "metric_minimum_eigenvalues": np.asarray(
                model_minimum_eigenvalues
            ),
            "importance_weights": np.asarray(dataset["weights_numpy"]),
            "sampling_cluster_ids": np.asarray(dataset["sampling_cluster_ids"]),
            **point_payload,
        }
        if dataset["teacher_log_eta"] is not None:
            arrays_payload["teacher_log_eta"] = np.asarray(
                dataset["teacher_log_eta"].detach().cpu().numpy()
            )
        save_npz_atomic(arrays_path, **arrays_payload)
        arrays_metadata = {
            "path": str(arrays_path),
            "sha256": sha256_file(arrays_path),
        }
    report = {
        "schema": "type11-positive-tensor-network-blind-audit-v1",
        "adapter": adapter.key,
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "source_artifact": str(source_path),
        "teacher_artifact": None if teacher_path is None else str(teacher_path),
        "model_seed": model_seed,
        "seed": args.seed,
        "points": args.points,
        "sampling_cluster_size": sampling_cluster_size,
        "point_shards": dataset["shards"],
        "common_pool": dataset["common_pool"],
        "common_pool_sha256": dataset["common_pool_sha256"],
        "importance_effective_sample_size": dataset[
            "importance_effective_sample_size"
        ],
        "site_count": model.site_count,
        "bond_dimension": model.bond_dimension,
        "architecture": model.architecture,
        "physical_dictionary_rank": model.physical_dictionary_rank,
        "trainable_real_parameter_count": model.trainable_real_parameter_count,
        "device": str(device),
        "precision": precision,
        "fixed_teacher_log_kappa": float(fixed_log_kappa.cpu()),
        "fixed_log_kappa_source": fixed_log_kappa_source,
        "metrics": metrics,
        "point_arrays": arrays_metadata,
        "runtime_seconds": time.perf_counter() - started,
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
