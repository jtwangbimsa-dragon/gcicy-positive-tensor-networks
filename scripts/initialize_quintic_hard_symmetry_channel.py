#!/usr/bin/env python3
"""Activate one exact-nested hard-symmetry bond channel without changing F."""

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

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bond-index", type=int, required=True)
    parser.add_argument("--relative-seed-scale", type=float, default=1.0e-3)
    parser.add_argument(
        "--seed-reference-scale",
        choices=("inherited_rms", "unit_rms"),
        default="inherited_rms",
        help="normalize a seed to the adjacent inherited RMS or to unit RMS",
    )
    parser.add_argument("--seed", type=int, default=202607243)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def one_sided_channel_masks(
    model: torch.nn.Module,
    *,
    bond_index: int,
    source_multiplicity: int,
    target_multiplicity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left train and right seed masks for one newly added bond copy."""

    starts = tuple(int(value) for value in model.blocked_two_site_starts)
    if not 0 <= bond_index < len(starts) - 1:
        raise ValueError("bond index must lie between two permanent blocks")
    if not 0 < source_multiplicity < target_multiplicity:
        raise ValueError("target multiplicity must exceed a positive source")
    new_copy = source_multiplicity
    left_start = starts[bond_index]
    right_start = starts[bond_index + 1]
    if right_start != left_start + 2:
        raise ValueError("channel activation requires adjacent degree-two blocks")

    left_parameter = model.blocked_two_site_orbit_parameters[bond_index]
    right_parameter = model.blocked_two_site_orbit_parameters[bond_index + 1]
    left_boundary = left_start == 0
    right_boundary = right_start + 1 == model.site_count - 1

    if left_boundary:
        if left_parameter.numel() % target_multiplicity:
            raise ValueError("left boundary block has an incompatible compact shape")
        left_view = left_parameter.reshape(-1, target_multiplicity)
        left_mask_view = torch.zeros_like(left_view, dtype=torch.bool)
        left_mask_view[:, new_copy] = True
    else:
        stride = target_multiplicity**2
        if left_parameter.numel() % stride:
            raise ValueError("left interior block has an incompatible compact shape")
        left_view = left_parameter.reshape(
            -1,
            target_multiplicity,
            target_multiplicity,
        )
        left_mask_view = torch.zeros_like(left_view, dtype=torch.bool)
        left_mask_view[:, :source_multiplicity, new_copy] = True

    if right_boundary:
        if right_parameter.numel() % target_multiplicity:
            raise ValueError("right boundary block has an incompatible compact shape")
        right_view = right_parameter.reshape(-1, target_multiplicity)
        right_mask_view = torch.zeros_like(right_view, dtype=torch.bool)
        right_mask_view[:, new_copy] = True
    else:
        stride = target_multiplicity**2
        if right_parameter.numel() % stride:
            raise ValueError("right interior block has an incompatible compact shape")
        right_view = right_parameter.reshape(
            -1,
            target_multiplicity,
            target_multiplicity,
        )
        right_mask_view = torch.zeros_like(right_view, dtype=torch.bool)
        right_mask_view[:, new_copy, :source_multiplicity] = True

    return left_mask_view.reshape(-1), right_mask_view.reshape(-1)


def seed_one_sided_channel_(
    model: torch.nn.Module,
    *,
    bond_index: int,
    source_multiplicity: int,
    target_multiplicity: int,
    relative_scale: float,
    generator: torch.Generator,
    reference_scale: str = "inherited_rms",
) -> dict[str, object]:
    """Seed the right factor while the left factor remains exactly zero."""

    if not np.isfinite(relative_scale) or relative_scale <= 0:
        raise ValueError("relative channel seed scale must be finite and positive")
    if reference_scale not in {"inherited_rms", "unit_rms"}:
        raise ValueError("unknown channel seed reference scale")
    left_mask, right_mask = one_sided_channel_masks(
        model,
        bond_index=bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
    )
    left = model.blocked_two_site_orbit_parameters[bond_index]
    right = model.blocked_two_site_orbit_parameters[bond_index + 1]
    if bool(torch.any(torch.abs(left[left_mask]) > 0)):
        raise ValueError("left activation entries must be zero in the exact embedding")
    if bool(torch.any(torch.abs(right[right_mask]) > 0)):
        raise ValueError("right seed entries must be zero in the exact embedding")

    inherited = right[~right_mask]
    inherited_rms = torch.sqrt(torch.mean(torch.abs(inherited) ** 2)).real
    inherited_rms = torch.clamp(
        inherited_rms,
        min=torch.finfo(inherited_rms.dtype).tiny,
    )
    real = torch.randn(
        int(torch.count_nonzero(right_mask)),
        generator=generator,
        dtype=right.real.dtype,
        device=right.device,
    )
    imaginary = torch.randn(
        real.shape,
        generator=generator,
        dtype=right.real.dtype,
        device=right.device,
    )
    noise = (real + 1j * imaginary) / math.sqrt(2.0)
    noise_rms = torch.sqrt(torch.mean(torch.abs(noise) ** 2)).real
    scale = (
        inherited_rms
        if reference_scale == "inherited_rms"
        else torch.ones_like(inherited_rms)
    )
    seeded = relative_scale * scale * noise / noise_rms
    with torch.no_grad():
        right[right_mask] = seeded
    return {
        "bond_index": int(bond_index),
        "left_block_index": int(bond_index),
        "right_block_index": int(bond_index + 1),
        "new_copy": int(source_multiplicity),
        "left_train_complex_parameters": int(torch.count_nonzero(left_mask)),
        "right_seed_complex_parameters": int(torch.count_nonzero(right_mask)),
        "relative_seed_scale": float(relative_scale),
        "seed_reference_scale": reference_scale,
        "adjacent_inherited_rms": float(inherited_rms),
        "absolute_seed_rms": float(torch.sqrt(torch.mean(torch.abs(seeded) ** 2))),
    }


def random_fidelity_probe(
    model: torch.nn.Module,
    *,
    seed: int,
    point_count: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real_dtype = model.reference_h.real.dtype
    values = torch.randn(
        point_count,
        model.section_count,
        generator=generator,
        dtype=real_dtype,
    )
    values = values + 1j * torch.randn(
        values.shape,
        generator=generator,
        dtype=real_dtype,
    )
    derivatives = torch.randn(
        point_count,
        model.section_count,
        3,
        generator=generator,
        dtype=real_dtype,
    )
    derivatives = derivatives + 1j * torch.randn(
        derivatives.shape,
        generator=generator,
        dtype=real_dtype,
    )
    return values, derivatives


def main() -> None:
    args = parse_args()
    input_path = args.model.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite a channel initializer artifact")
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    expansion = payload.get("phase_multiplicity_expansion")
    if not isinstance(expansion, dict):
        raise ValueError("input model is not a registered exact multiplicity expansion")
    if not bool(expansion.get("function_preserving_by_construction", False)):
        raise ValueError("input expansion must have zero activation noise")
    source_multiplicity = int(expansion["source_multiplicity"])
    target_multiplicity = int(expansion["target_multiplicity"])
    if target_multiplicity != int(payload["fermat_phase_charge_multiplicity"]):
        raise ValueError("expansion metadata and model multiplicity disagree")

    precision = str(payload["precision"])
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device="cpu",
        trainable_physical_dictionary=False,
    ).to(dtype=complex_dtype)
    values, derivatives = random_fidelity_probe(model, seed=args.seed + 101)
    with torch.no_grad():
        baseline_potential, baseline_metric = model.potential_and_metric(
            values,
            derivatives,
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    activation = seed_one_sided_channel_(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        relative_scale=args.relative_seed_scale,
        generator=generator,
        reference_scale=args.seed_reference_scale,
    )
    with torch.no_grad():
        seeded_potential, seeded_metric = model.potential_and_metric(values, derivatives)
    potential_error = float(torch.max(torch.abs(seeded_potential - baseline_potential)))
    metric_error = float(torch.max(torch.abs(seeded_metric - baseline_metric)))
    tolerance = 2.0e-6 if precision == "complex64" else 2.0e-12
    if potential_error > tolerance or metric_error > tolerance:
        raise RuntimeError(
            "one-sided seed failed the exact function-preservation gate: "
            f"potential={potential_error}, metric={metric_error}"
        )

    output_payload = dict(payload)
    output_payload["state_dict"] = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    output_payload["adaptive_channel_initialization"] = {
        "schema": "quintic-hard-symmetry-one-sided-channel-v1",
        "source_model": str(input_path),
        "source_model_sha256": sha256_file(input_path),
        "source_multiplicity": source_multiplicity,
        "target_multiplicity": target_multiplicity,
        "seed": args.seed,
        **activation,
        "epoch_zero_potential_max_abs": potential_error,
        "epoch_zero_metric_max_abs": metric_error,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output_path)
    summary = {
        **output_payload["adaptive_channel_initialization"],
        "schema": "quintic-hard-symmetry-one-sided-channel-summary-v1",
        "output_model": str(output_path),
        "output_model_sha256": sha256_file(output_path),
        "trainable_real_parameter_count": int(model.trainable_real_parameter_count),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
