#!/usr/bin/env python3
"""Compress dense TN cores into one shared local section-pair dictionary."""

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

from gcicy_metric.pipeline import (
    PositiveTensorNetworkMetric,
    compress_cores_to_shared_local_dictionary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--train-physical-dictionary", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for tensor-network compression") from exc

    args = parse_args()
    if args.rank <= 0:
        raise SystemExit("dictionary rank must be positive")
    started = time.perf_counter()
    model_path = args.model.expanduser().resolve()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "type11-positive-tensor-network-v1":
        raise ValueError("unrecognized positive tensor-network artifact")
    if payload.get("architecture", "dense_local_cores") != "dense_local_cores":
        raise ValueError("the compression source must use dense local cores")
    state = payload["state_dict"]
    dense_keys = sorted(
        (key for key in state if key.startswith("cores.")),
        key=lambda key: int(key.split(".")[1]),
    )
    if len(dense_keys) != int(payload["site_count"]):
        raise ValueError("dense core count does not match the saved site count")
    dense_cores = tuple(state[key].detach().cpu().numpy() for key in dense_keys)
    compression = compress_cores_to_shared_local_dictionary(
        dense_cores,
        rank=args.rank,
    )

    precision = str(payload["precision"])
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    reference_h = state["reference_h"].detach().cpu().numpy()
    model = PositiveTensorNetworkMetric(
        reference_h,
        site_count=int(payload["site_count"]),
        bond_dimension=int(payload["bond_dimension"]),
        target_normalization=float(payload["target_normalization"]),
        positive_floor=float(payload["positive_floor"]),
        initialization_noise=0.0,
        physical_dictionary=compression.physical_dictionary,
        trainable_physical_dictionary=args.train_physical_dictionary,
        dtype=complex_dtype,
        device="cpu",
    )
    compressed_state = model.state_dict()
    for index, coefficient in enumerate(compression.coefficient_cores):
        compressed_state[f"coefficient_cores.{index}"] = torch.tensor(
            coefficient,
            dtype=complex_dtype,
        )
    model.load_state_dict(compressed_state)

    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = dict(payload)
    output_payload.update(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "architecture": "shared_local_dictionary",
            "physical_dictionary_rank": int(args.rank),
            "trainable_physical_dictionary": bool(
                args.train_physical_dictionary
            ),
            "compression_source_model": str(model_path),
            "compression_source_model_sha256": sha256_file(model_path),
            "compression_initialization": "joint_complex_svd_of_dense_local_cores",
        }
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output_path)

    singular_values = np.asarray(compression.singular_values, dtype=float)
    summary = {
        "schema": "positive-tensor-network-local-dictionary-compression-v1",
        "source_model": str(model_path),
        "source_model_sha256": sha256_file(model_path),
        "model": str(output_path),
        "model_sha256": sha256_file(output_path),
        "adapter": payload.get("adapter"),
        "model_seed": payload.get("model_seed"),
        "site_count": int(payload["site_count"]),
        "bond_dimension": int(payload["bond_dimension"]),
        "section_count": int(reference_h.shape[0]),
        "physical_dictionary_rank": int(args.rank),
        "trainable_physical_dictionary": bool(args.train_physical_dictionary),
        "trainable_real_parameter_count": int(
            model.trainable_real_parameter_count
        ),
        "source_trainable_real_parameter_count": int(
            2 * sum(np.asarray(core).size for core in dense_cores)
        ),
        "retained_squared_frobenius_energy_fraction": float(
            compression.retained_energy_fraction
        ),
        "relative_dense_core_frobenius_error": float(
            compression.relative_frobenius_error
        ),
        "singular_values": singular_values.tolist(),
        "relative_singular_values": (singular_values / singular_values[0]).tolist(),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    write_text_atomic(args.summary.expanduser().resolve(), json.dumps(summary, indent=2) + "\n")
    print(
        f"rank={args.rank} parameters={model.trainable_real_parameter_count} "
        f"retained={compression.retained_energy_fraction:.8f}",
        flush=True,
    )
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {args.summary.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
