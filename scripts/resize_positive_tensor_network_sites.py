#!/usr/bin/env python3
"""Initialize a new degree by resizing shared-dictionary TN sites."""

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
    activate_one_sided_repeated_shared_dictionary_bridges,
    activate_repeated_shared_dictionary_bridges,
    repeat_shared_dictionary_coefficient_cores,
    resize_shared_dictionary_coefficient_cores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-site-count", type=int, required=True)
    parser.add_argument(
        "--transfer-rule",
        choices=("interpolate", "repeat"),
        default="interpolate",
        help="repeat is lossless for the learned norm and requires an integer multiple",
    )
    parser.add_argument(
        "--bridge-relative-noise",
        type=float,
        default=0.0,
        help="relative RMS noise added only to dormant repeated-block channels",
    )
    parser.add_argument(
        "--bridge-one-sided-activation-scale",
        type=float,
        default=0.0,
        help=(
            "function-preserving activation of dormant repeated-block channels; "
            "mutually exclusive with --bridge-relative-noise"
        ),
    )
    parser.add_argument("--bridge-noise-seed", type=int, default=20260732)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
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
        raise SystemExit("PyTorch is required for tensor-network site transfer") from exc

    args = parse_args()
    if args.bridge_relative_noise and args.bridge_one_sided_activation_scale:
        raise ValueError(
            "matched bridge noise and one-sided bridge activation are mutually exclusive"
        )
    if args.target_site_count < 3:
        raise SystemExit("target site count must be at least three")
    source_path = args.model.expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    artifact_schema = payload.get("schema")
    if artifact_schema not in {
        "type11-positive-tensor-network-v1",
        "quintic-positive-tensor-network-v1",
    }:
        raise ValueError("unrecognized positive tensor-network artifact")
    if payload.get("architecture") != "shared_local_dictionary":
        raise ValueError("site transfer requires a shared-local-dictionary model")
    source_site_count = int(payload["site_count"])
    if source_site_count < 3 or source_site_count == args.target_site_count:
        raise ValueError("source and target site counts must be distinct and at least three")

    state = payload["state_dict"]
    coefficient_keys = sorted(
        (key for key in state if key.startswith("coefficient_cores.")),
        key=lambda key: int(key.split(".")[1]),
    )
    if len(coefficient_keys) != source_site_count:
        raise ValueError("saved coefficient-core count does not match source sites")
    source_coefficients = tuple(
        state[key].detach().cpu().numpy() for key in coefficient_keys
    )
    if args.transfer_rule == "repeat":
        repeat_count, remainder = divmod(args.target_site_count, source_site_count)
        if remainder or repeat_count < 2:
            raise ValueError(
                "repeat transfer requires target sites to be an integer multiple "
                "of source sites"
            )
        target_coefficients = repeat_shared_dictionary_coefficient_cores(
            source_coefficients,
            repeat_count,
        )
        if args.bridge_relative_noise > 0:
            target_coefficients = activate_repeated_shared_dictionary_bridges(
                target_coefficients,
                source_site_count=source_site_count,
                relative_noise=args.bridge_relative_noise,
                seed=args.bridge_noise_seed,
            )
        elif args.bridge_one_sided_activation_scale > 0:
            target_coefficients = (
                activate_one_sided_repeated_shared_dictionary_bridges(
                    target_coefficients,
                    source_site_count=source_site_count,
                    relative_scale=args.bridge_one_sided_activation_scale,
                    seed=args.bridge_noise_seed,
                )
            )
        transfer_rule = (
            "exact learned-norm MPS repetition with zeroth-channel scalar bridges; "
            f"repeat_count={repeat_count}"
        )
        if args.bridge_relative_noise > 0:
            transfer_rule += (
                "; dormant bridge activation relative_noise="
                f"{args.bridge_relative_noise:.8g}, seed={args.bridge_noise_seed}"
            )
        elif args.bridge_one_sided_activation_scale > 0:
            transfer_rule += (
                "; function-preserving one-sided bridge activation relative_scale="
                f"{args.bridge_one_sided_activation_scale:.8g}, "
                f"seed={args.bridge_noise_seed}"
            )
    else:
        if args.bridge_relative_noise or args.bridge_one_sided_activation_scale:
            raise ValueError("bridge activation is available only for repeat transfer")
        target_coefficients = resize_shared_dictionary_coefficient_cores(
            source_coefficients,
            args.target_site_count,
        )
        transfer_rule = "fixed boundaries and linear interior-core interpolation"
    dictionary = state["physical_dictionary"].detach().cpu().numpy()
    reference_h = state["reference_h"].detach().cpu().numpy()
    source_normalization = float(payload["target_normalization"]) * source_site_count
    target_normalization = source_normalization / args.target_site_count
    precision = str(payload["precision"])
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    model = PositiveTensorNetworkMetric(
        reference_h,
        site_count=args.target_site_count,
        bond_dimension=int(payload["bond_dimension"]),
        target_normalization=target_normalization,
        positive_floor=float(payload["positive_floor"]),
        initialization_noise=0.0,
        physical_dictionary=dictionary,
        trainable_physical_dictionary=bool(
            payload.get("trainable_physical_dictionary", False)
        ),
        dtype=complex_dtype,
        device="cpu",
    )
    target_state = model.state_dict()
    for index, coefficient in enumerate(target_coefficients):
        target_state[f"coefficient_cores.{index}"] = torch.tensor(
            coefficient,
            dtype=complex_dtype,
        )
    model.load_state_dict(target_state)

    if artifact_schema == "type11-positive-tensor-network-v1":
        source_degree = tuple(int(value) for value in payload["source_degree"])
        target_degree = [value * args.target_site_count for value in source_degree]
    else:
        source_degree = (1,)
        target_degree = [int(args.target_site_count)]
    output_payload = dict(payload)
    output_payload.update(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "site_count": int(args.target_site_count),
            "target_degree": target_degree,
            "degree_k": int(args.target_site_count),
            "target_normalization": float(target_normalization),
            "teacher_artifact": None,
            "teacher_artifact_sha256": None,
            "training_mode": "teacher_free_geometric_site_transfer_initialization",
            "fixed_log_kappa": None,
            "fixed_log_kappa_source": "requires_target_degree_estimation",
            "site_transfer_source_model": str(source_path),
            "site_transfer_source_model_sha256": sha256_file(source_path),
            "site_transfer_rule": transfer_rule,
        }
    )
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(out_path)

    summary = {
        "schema": "positive-tensor-network-site-transfer-v1",
        "source_model": str(source_path),
        "source_model_sha256": sha256_file(source_path),
        "model": str(out_path),
        "model_sha256": sha256_file(out_path),
        "source_site_count": source_site_count,
        "target_site_count": int(args.target_site_count),
        "source_degree": list(source_degree),
        "target_degree": target_degree,
        "bond_dimension": int(model.bond_dimension),
        "physical_dictionary_rank": int(model.physical_dictionary_rank),
        "trainable_real_parameter_count": int(model.trainable_real_parameter_count),
        "target_normalization": float(target_normalization),
        "fixed_log_kappa": None,
        "site_transfer_rule": output_payload["site_transfer_rule"],
        "bridge_relative_noise": float(args.bridge_relative_noise),
        "bridge_one_sided_activation_scale": float(
            args.bridge_one_sided_activation_scale
        ),
        "bridge_noise_seed": int(args.bridge_noise_seed),
    }
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temporary.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_temporary.replace(summary_path)
    print(
        f"sites={source_site_count}->{args.target_site_count} "
        f"parameters={model.trainable_real_parameter_count}",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
