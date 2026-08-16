#!/usr/bin/env python3
"""Complete a TN local dictionary with fixed canonical matrix units."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    PositiveTensorNetworkMetric,
    canonical_matrix_unit_dictionary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--maximum-relative-core-error",
        type=float,
        default=1.0e-12,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for dictionary completion") from exc

    args = parse_args()
    source_path = args.model.expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "type11-positive-tensor-network-v1":
        raise ValueError("unrecognized positive tensor-network artifact")
    architecture = payload.get("architecture", "dense_local_cores")
    state = payload["state_dict"]
    site_count = int(payload["site_count"])

    if architecture == "shared_local_dictionary":
        dictionary = state.get("physical_dictionary")
        if dictionary is None:
            raise ValueError("shared-dictionary model has no physical dictionary")
        dictionary_numpy = dictionary.detach().cpu().numpy()
        coefficient_keys = sorted(
            (key for key in state if key.startswith("coefficient_cores.")),
            key=lambda key: int(key.split(".")[1]),
        )
        if len(coefficient_keys) != site_count:
            raise ValueError("coefficient-core count does not match site count")
        dense_cores = tuple(
            np.einsum(
                "lrq,qpi->lrpi",
                state[key].detach().cpu().numpy(),
                dictionary_numpy,
            )
            for key in coefficient_keys
        )
        source_rank = int(dictionary_numpy.shape[0])
    elif architecture == "dense_local_cores":
        dense_keys = sorted(
            (key for key in state if key.startswith("cores.")),
            key=lambda key: int(key.split(".")[1]),
        )
        if len(dense_keys) != site_count:
            raise ValueError("dense-core count does not match site count")
        dense_cores = tuple(
            state[key].detach().cpu().numpy() for key in dense_keys
        )
        source_rank = None
    else:
        raise ValueError(f"unsupported architecture: {architecture}")

    output_dimension = int(dense_cores[0].shape[2])
    section_count = int(dense_cores[0].shape[3])
    if any(
        core.ndim != 4
        or core.shape[2:] != (output_dimension, section_count)
        for core in dense_cores
    ):
        raise ValueError("materialized cores have inconsistent physical shapes")
    canonical_dictionary = canonical_matrix_unit_dictionary(
        output_dimension,
        section_count,
    )
    precision = str(payload["precision"])
    complex_dtype = (
        torch.complex64 if precision == "complex64" else torch.complex128
    )
    reference_h = state["reference_h"].detach().cpu().numpy()
    model = PositiveTensorNetworkMetric(
        reference_h,
        site_count=site_count,
        bond_dimension=int(payload["bond_dimension"]),
        target_normalization=float(payload["target_normalization"]),
        positive_floor=float(payload["positive_floor"]),
        initialization_noise=0.0,
        physical_dictionary=canonical_dictionary,
        trainable_physical_dictionary=False,
        transfer_implementation=str(
            payload.get("transfer_implementation", "vectorized")
        ),
        dtype=complex_dtype,
        device="cpu",
    )
    completed_state = model.state_dict()
    relative_errors: list[float] = []
    for index, dense_core in enumerate(dense_cores):
        coefficient = dense_core.reshape(
            dense_core.shape[0],
            dense_core.shape[1],
            output_dimension * section_count,
        )
        completed_state[f"coefficient_cores.{index}"] = torch.tensor(
            coefficient,
            dtype=complex_dtype,
        )
        reconstructed = np.einsum(
            "lrq,qpi->lrpi",
            coefficient,
            canonical_dictionary,
        )
        relative_errors.append(
            float(
                np.linalg.norm(reconstructed - dense_core)
                / max(np.linalg.norm(dense_core), np.finfo(float).tiny)
            )
        )
    maximum_relative_error = max(relative_errors)
    if maximum_relative_error > args.maximum_relative_core_error:
        raise FloatingPointError(
            "canonical completion exceeded the registered core-error tolerance"
        )
    model.load_state_dict(completed_state)

    output_payload = dict(payload)
    output_payload.update(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": "shared_local_dictionary",
            "physical_dictionary_rank": int(
                output_dimension * section_count
            ),
            "trainable_physical_dictionary": False,
            "physical_dictionary_gauge": "fixed_canonical_matrix_units",
            "dictionary_completion_source_model": str(source_path),
            "dictionary_completion_source_sha256": sha256_file(source_path),
            "dictionary_completion_source_rank": source_rank,
            "dictionary_completion_rule": (
                "exact materialization followed by fixed canonical matrix-unit "
                "coordinates"
            ),
        }
    )
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output_path)

    summary = {
        "schema": "positive-tensor-network-complete-dictionary-v1",
        "source_model": str(source_path),
        "source_model_sha256": sha256_file(source_path),
        "model": str(output_path),
        "model_sha256": sha256_file(output_path),
        "source_architecture": architecture,
        "source_dictionary_rank": source_rank,
        "target_dictionary_rank": int(output_dimension * section_count),
        "site_count": site_count,
        "bond_dimension": int(payload["bond_dimension"]),
        "trainable_physical_dictionary": False,
        "trainable_real_parameter_count": int(
            model.trainable_real_parameter_count
        ),
        "maximum_relative_core_error": maximum_relative_error,
        "maximum_relative_core_error_gate": float(
            args.maximum_relative_core_error
        ),
    }
    write_json_atomic(args.summary.expanduser().resolve(), summary)
    print(
        f"rank={source_rank}->{summary['target_dictionary_rank']} "
        f"parameters={summary['trainable_real_parameter_count']} "
        f"relative_core_error={maximum_relative_error:.3e}",
        flush=True,
    )
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {args.summary.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
