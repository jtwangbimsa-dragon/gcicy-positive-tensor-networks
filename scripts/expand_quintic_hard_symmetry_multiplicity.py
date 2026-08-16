#!/usr/bin/env python3
"""Embed a compact hard-Fermat TN into a larger sector multiplicity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    PositiveTensorNetworkMetric,
    phase_s5_two_site_orbit_labels,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--multiplicity", type=int, required=True)
    parser.add_argument(
        "--activation-noise",
        type=float,
        default=0.0,
        help="relative complex noise applied only to newly introduced channels",
    )
    parser.add_argument("--seed", type=int, default=202607233)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def expand_compact_block_parameters(
    source: torch.Tensor,
    *,
    source_multiplicity: int,
    target_multiplicity: int,
    left_boundary: bool,
    right_boundary: bool,
    activation_noise: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed orbit coefficients and return the inherited-entry mask."""

    if source_multiplicity <= 0 or target_multiplicity <= source_multiplicity:
        raise ValueError("target multiplicity must exceed a positive source")
    if activation_noise < 0:
        raise ValueError("activation noise must be non-negative")
    copy_dimensions = int(not left_boundary) + int(not right_boundary)
    source_stride = source_multiplicity**copy_dimensions
    target_stride = target_multiplicity**copy_dimensions
    if source.numel() % source_stride:
        raise ValueError("source orbit count is incompatible with its multiplicity")
    base_orbit_count = source.numel() // source_stride
    target = torch.zeros(
        base_orbit_count * target_stride,
        dtype=source.dtype,
        device=source.device,
    )
    inherited = torch.zeros_like(target, dtype=torch.bool)

    if copy_dimensions == 0:
        target.copy_(source)
        inherited.fill_(True)
    elif copy_dimensions == 1:
        source_matrix = source.reshape(base_orbit_count, source_multiplicity)
        target_matrix = target.reshape(base_orbit_count, target_multiplicity)
        inherited_matrix = inherited.reshape(base_orbit_count, target_multiplicity)
        target_matrix[:, :source_multiplicity] = source_matrix
        inherited_matrix[:, :source_multiplicity] = True
    else:
        source_tensor = source.reshape(
            base_orbit_count,
            source_multiplicity,
            source_multiplicity,
        )
        target_tensor = target.reshape(
            base_orbit_count,
            target_multiplicity,
            target_multiplicity,
        )
        inherited_tensor = inherited.reshape(
            base_orbit_count,
            target_multiplicity,
            target_multiplicity,
        )
        target_tensor[
            :, :source_multiplicity, :source_multiplicity
        ] = source_tensor
        inherited_tensor[
            :, :source_multiplicity, :source_multiplicity
        ] = True

    if activation_noise:
        if generator is None:
            generator = torch.Generator(device=source.device)
        source_rms = torch.sqrt(torch.mean(torch.abs(source) ** 2)).real
        scale = activation_noise * torch.clamp(source_rms, min=1.0e-12)
        real = torch.randn(
            target.shape,
            generator=generator,
            dtype=target.real.dtype,
            device=target.device,
        )
        if target.is_complex():
            imaginary = torch.randn(
                target.shape,
                generator=generator,
                dtype=target.real.dtype,
                device=target.device,
            )
            noise = (real + 1j * imaginary) / math.sqrt(2.0)
        else:
            noise = real
        target = target + (~inherited) * scale * noise
    return target, inherited


def main() -> None:
    args = parse_args()
    if args.multiplicity <= 0:
        raise ValueError("target multiplicity must be positive")
    if args.activation_noise < 0:
        raise ValueError("activation noise must be non-negative")
    input_path = args.model.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "quintic-positive-tensor-network-v1":
        raise ValueError("unrecognized quintic tensor-network artifact")
    if not payload.get("fermat_s5_orbit_tying", False):
        raise ValueError("source model does not impose exact S5 symmetry")
    if not payload.get("fermat_two_site_blocking", False):
        raise ValueError("source model does not use permanent two-site blocks")
    if not payload.get("fermat_block_compact_storage", False):
        raise ValueError("source model does not use compact block storage")
    source_multiplicity = int(payload.get("fermat_phase_charge_multiplicity", 0))
    if args.multiplicity <= source_multiplicity:
        raise ValueError("target multiplicity must exceed the source multiplicity")

    precision = str(payload["precision"])
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    dictionary = np.asarray(
        payload["state_dict"]["physical_dictionary"].detach().cpu(),
        dtype=np.complex128,
    )
    site_count = int(payload["site_count"])
    starts = tuple(int(value) for value in payload["fermat_two_site_block_starts"])
    expected_starts = tuple(range(0, site_count, 2))
    if starts != expected_starts:
        raise ValueError("multiplicity expansion requires the registered full blocking")

    target_bond_dimension = 21 * args.multiplicity
    model = PositiveTensorNetworkMetric(
        reference_h,
        site_count=site_count,
        bond_dimension=target_bond_dimension,
        target_normalization=float(payload["target_normalization"]),
        output_dimension=int(payload["output_dimension"]),
        positive_floor=float(payload["positive_floor"]),
        initialization_noise=0.0,
        physical_dictionary=dictionary,
        trainable_physical_dictionary=False,
        transfer_implementation=str(payload.get("transfer_implementation", "vectorized")),
        seed=args.seed,
        dtype=complex_dtype,
        device="cpu",
    )
    labels = tuple(
        phase_s5_two_site_orbit_labels(
            left_boundary=start == 0,
            right_boundary=start + 1 == site_count - 1,
            multiplicity=args.multiplicity,
        )
        for start in starts
    )
    model.block_two_site_coefficient_orbits_(
        starts,
        labels,
        drop_covered_coefficient_cores=True,
    )

    source_parameters = tuple(
        payload["state_dict"][f"blocked_two_site_orbit_parameters.{index}"]
        for index in range(len(starts))
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    inherited_counts = []
    with torch.no_grad():
        for block_index, (start, source_parameter, target_parameter) in enumerate(
            zip(
                starts,
                source_parameters,
                model.blocked_two_site_orbit_parameters,
                strict=True,
            )
        ):
            expanded, inherited = expand_compact_block_parameters(
                source_parameter.to(dtype=complex_dtype),
                source_multiplicity=source_multiplicity,
                target_multiplicity=args.multiplicity,
                left_boundary=start == 0,
                right_boundary=start + 1 == site_count - 1,
                activation_noise=args.activation_noise,
                generator=generator,
            )
            if expanded.shape != target_parameter.shape:
                raise RuntimeError(
                    f"block {block_index} expansion shape mismatch: "
                    f"{expanded.shape} != {target_parameter.shape}"
                )
            target_parameter.copy_(expanded)
            inherited_counts.append(int(torch.count_nonzero(inherited).item()))

    output_payload = dict(payload)
    output_payload["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    output_payload["bond_dimension"] = target_bond_dimension
    output_payload["fermat_phase_charge_multiplicity"] = args.multiplicity
    output_payload["phase_multiplicity_expansion"] = {
        "source_model": str(input_path),
        "source_model_sha256": sha256_file(input_path),
        "source_multiplicity": source_multiplicity,
        "target_multiplicity": args.multiplicity,
        "activation_noise": args.activation_noise,
        "seed": args.seed,
        "function_preserving_by_construction": args.activation_noise == 0.0,
        "inherited_complex_parameter_counts": inherited_counts,
    }

    output_path = args.out.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite an existing expansion artifact")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output_path)
    summary = {
        "schema": "quintic-hard-symmetry-multiplicity-expansion-v1",
        "source_model": str(input_path),
        "source_model_sha256": sha256_file(input_path),
        "expanded_model": str(output_path),
        "expanded_model_sha256": sha256_file(output_path),
        "source_multiplicity": source_multiplicity,
        "target_multiplicity": args.multiplicity,
        "source_bond_dimension": int(payload["bond_dimension"]),
        "target_bond_dimension": target_bond_dimension,
        "activation_noise": args.activation_noise,
        "seed": args.seed,
        "function_preserving_by_construction": args.activation_noise == 0.0,
        "trainable_real_parameter_count": model.trainable_real_parameter_count,
        "inherited_complex_parameter_count": int(sum(inherited_counts)),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
