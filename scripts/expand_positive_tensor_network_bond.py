#!/usr/bin/env python3
"""Embed a saved positive tensor network into a larger bond dimension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import PositiveTensorNetworkMetric  # noqa: E402
from gcicy_metric.pipeline.exact_lift_tree import (  # noqa: E402
    EXACT_LIFT_TREE_SCHEMA,
    NATIVE_POWER_LIFT_TREE_SCHEMA,
)
from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    activate_one_sided_expanded_bond_channels,
    expand_tensor_network_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bond-dimension", type=int, required=True)
    parser.add_argument("--initialization-noise", type=float, default=0.0)
    parser.add_argument(
        "--one-sided-activation-scale",
        type=float,
        default=0.0,
        help=(
            "function-preserving activation of new outgoing bond columns; "
            "requires zero initialization-noise"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260746)
    parser.add_argument(
        "--precision",
        choices=("preserve", "complex64", "complex128"),
        default="preserve",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    import torch

    args = parse_args()
    if args.one_sided_activation_scale < 0.0:
        raise ValueError("one-sided activation scale cannot be negative")
    if args.initialization_noise and args.one_sided_activation_scale:
        raise ValueError(
            "one-sided activation requires initialization-noise to be zero"
        )
    input_path = args.model.expanduser().resolve()
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if payload.get("schema") not in {
        "type11-positive-tensor-network-v1",
        "quintic-positive-tensor-network-v1",
        EXACT_LIFT_TREE_SCHEMA,
        NATIVE_POWER_LIFT_TREE_SCHEMA,
    }:
        raise ValueError("unrecognized positive tensor-network artifact")
    source_bond_dimension = int(payload["bond_dimension"])
    if args.bond_dimension <= source_bond_dimension:
        raise ValueError("target bond dimension must be larger than the source")
    precision = (
        payload["precision"]
        if args.precision == "preserve"
        else args.precision
    )
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    architecture = payload.get("architecture", "dense_local_cores")
    if architecture == "shared_local_dictionary":
        dictionary_value = payload["state_dict"].get("physical_dictionary")
        if dictionary_value is None:
            raise ValueError("shared-dictionary model has no saved dictionary")
        physical_dictionary = np.asarray(
            dictionary_value.detach().cpu(), dtype=np.complex128
        )
        trainable_physical_dictionary = bool(
            payload.get("trainable_physical_dictionary", False)
        )
    elif architecture == "dense_local_cores":
        physical_dictionary = None
        trainable_physical_dictionary = False
    else:
        raise ValueError(f"unsupported tensor-network architecture: {architecture}")
    model = PositiveTensorNetworkMetric(
        reference_h,
        site_count=int(payload["site_count"]),
        bond_dimension=args.bond_dimension,
        target_normalization=float(payload["target_normalization"]),
        positive_floor=float(payload["positive_floor"]),
        initialization_noise=args.initialization_noise,
        physical_dictionary=physical_dictionary,
        trainable_physical_dictionary=trainable_physical_dictionary,
        transfer_implementation=str(
            payload.get("transfer_implementation", "vectorized")
        ),
        seed=args.seed,
        dtype=complex_dtype,
        device="cpu",
    )
    expanded_state = expand_tensor_network_state_dict(
        model.state_dict(), payload["state_dict"]
    )
    expanded_state = activate_one_sided_expanded_bond_channels(
        expanded_state,
        payload["state_dict"],
        scale=args.one_sided_activation_scale,
        seed=args.seed,
    )
    model.load_state_dict(expanded_state)
    output_payload = dict(payload)
    output_payload["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    output_payload["bond_dimension"] = args.bond_dimension
    output_payload["bond_dimensions"] = (
        args.bond_dimension,
    ) * max(int(payload["site_count"]) - 1, 0)
    output_payload["architecture"] = architecture
    output_payload["precision"] = precision
    if payload.get("schema") in {
        EXACT_LIFT_TREE_SCHEMA,
        NATIVE_POWER_LIFT_TREE_SCHEMA,
    }:
        output_payload["schema"] = NATIVE_POWER_LIFT_TREE_SCHEMA
        output_payload["geometry_role"] = (
            "teacher_free_native_rank_activation"
        )
        output_payload["teacher_runtime_dependency"] = False
        output_payload["initial_exact_identity"] = payload.get(
            "initial_exact_identity",
            payload.get("exact_identity"),
        )
        output_payload.pop("exact_identity", None)
    output_payload["bond_expansion"] = {
        "source_model": str(input_path),
        "source_model_sha256": sha256_file(input_path),
        "source_bond_dimension": source_bond_dimension,
        "target_bond_dimension": args.bond_dimension,
        "initialization_noise": args.initialization_noise,
        "one_sided_activation_scale": args.one_sided_activation_scale,
        "precision": precision,
        "seed": args.seed,
    }

    output_path = args.out.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output_path)
    summary = {
        "schema": "positive-tensor-network-bond-expansion-v1",
        "source_model": str(input_path),
        "source_model_sha256": sha256_file(input_path),
        "expanded_model": str(output_path),
        "expanded_model_sha256": sha256_file(output_path),
        "source_bond_dimension": source_bond_dimension,
        "target_bond_dimension": args.bond_dimension,
        "initialization_noise": args.initialization_noise,
        "one_sided_activation_scale": args.one_sided_activation_scale,
        "precision": precision,
        "function_preserving_by_construction": args.initialization_noise == 0.0,
        "new_channels_have_first_order_activation": (
            args.one_sided_activation_scale > 0.0
        ),
        "trainable_real_parameter_count": model.trainable_real_parameter_count,
    }
    write_text_atomic(summary_path, json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
