#!/usr/bin/env python3
"""Losslessly express a shared TN dictionary in the full matrix-unit basis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--maximum-relative-core-error", type=float, default=1.0e-12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for dictionary conversion") from exc

    args = parse_args()
    source_path = args.model.expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("schema") not in {
        "type11-positive-tensor-network-v1",
        "quintic-positive-tensor-network-v1",
    }:
        raise ValueError("unrecognized positive tensor-network artifact")
    if payload.get("architecture") != "shared_local_dictionary":
        raise ValueError("matrix-unit conversion requires a shared-dictionary model")

    state = payload["state_dict"]
    dictionary_tensor = state.get("physical_dictionary")
    if dictionary_tensor is None:
        raise ValueError("saved model has no physical dictionary")
    dictionary = dictionary_tensor.detach().cpu().numpy()
    if dictionary.ndim != 3 or dictionary.shape[1] != dictionary.shape[2]:
        raise ValueError("physical dictionary must have shape (q,d,d)")
    source_rank, local_dimension, _ = dictionary.shape
    target_rank = local_dimension * local_dimension

    coefficient_keys = sorted(
        (key for key in state if key.startswith("coefficient_cores.")),
        key=lambda key: int(key.split(".")[1]),
    )
    if len(coefficient_keys) != int(payload["site_count"]):
        raise ValueError("saved coefficient-core count does not match site count")

    matrix_units = np.zeros(
        (target_rank, local_dimension, local_dimension),
        dtype=dictionary.dtype,
    )
    flat = np.arange(target_rank)
    matrix_units[flat, flat // local_dimension, flat % local_dimension] = 1

    converted = []
    maximum_relative_error = 0.0
    for key in coefficient_keys:
        old = state[key].detach().cpu().numpy()
        if old.shape[-1] != source_rank:
            raise ValueError(f"coefficient core {key} does not match dictionary rank")
        effective = np.einsum("...a,aij->...ij", old, dictionary, optimize=True)
        new = effective.reshape(*effective.shape[:-2], target_rank)
        reconstructed = np.einsum("...a,aij->...ij", new, matrix_units, optimize=True)
        denominator = max(float(np.linalg.norm(effective)), np.finfo(float).tiny)
        relative_error = float(np.linalg.norm(reconstructed - effective) / denominator)
        maximum_relative_error = max(maximum_relative_error, relative_error)
        converted.append(new)

    if maximum_relative_error > args.maximum_relative_core_error:
        raise FloatingPointError(
            "matrix-unit conversion exceeded the registered core-error tolerance"
        )

    output_state = {key: value.detach().cpu().clone() for key, value in state.items()}
    dtype = dictionary_tensor.dtype
    output_state["physical_dictionary"] = torch.tensor(matrix_units, dtype=dtype)
    for key, coefficient in zip(coefficient_keys, converted, strict=True):
        output_state[key] = torch.tensor(coefficient, dtype=dtype)

    output_payload = dict(payload)
    output_payload.update(
        {
            "state_dict": output_state,
            "physical_dictionary_rank": target_rank,
            "trainable_physical_dictionary": False,
            "physical_dictionary_gauge": "canonical_matrix_units_row_major",
            "training_mode": "teacher_free_fixed_complete_dictionary_initialization",
            "matrix_unit_conversion_source_model": str(source_path),
            "matrix_unit_conversion_source_sha256": sha256_file(source_path),
            "matrix_unit_conversion_source_rank": source_rank,
        }
    )

    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output_path)

    coefficient_parameter_count = 2 * sum(value.size for value in converted)
    summary = {
        "schema": "positive-tensor-network-matrix-unit-conversion-v1",
        "source_model": str(source_path),
        "source_model_sha256": sha256_file(source_path),
        "model": str(output_path),
        "model_sha256": sha256_file(output_path),
        "source_rank": int(source_rank),
        "target_rank": int(target_rank),
        "local_dimension": int(local_dimension),
        "site_count": int(payload["site_count"]),
        "bond_dimension": int(payload["bond_dimension"]),
        "fixed_dictionary": True,
        "trainable_real_parameter_count": int(coefficient_parameter_count),
        "maximum_relative_core_error": maximum_relative_error,
    }
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
