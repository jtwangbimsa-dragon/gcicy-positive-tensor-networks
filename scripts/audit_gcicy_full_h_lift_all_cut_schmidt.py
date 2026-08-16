#!/usr/bin/env python3
"""Audit all ordered four-site Schmidt cuts for two equivalent full-H lifts."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient-artifact", type=Path, required=True)
    parser.add_argument("--coefficient-seed", type=int, required=True)
    parser.add_argument("--full-h-artifact", type=Path, required=True)
    parser.add_argument(
        "--rank-tolerances",
        type=float,
        nargs="+",
        default=[1e-8, 1e-10, 1e-12],
    )
    parser.add_argument("--center-sketch-dimension", type=int, default=64)
    parser.add_argument("--center-sketch-seed", type=int, default=86611)
    parser.add_argument("--alternative-null-scale", type=float, default=0.25)
    parser.add_argument("--alternative-seed", type=int, default=86612)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("complex64", "complex128"),
        default="complex128",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_error(left: Any, right: Any) -> float:
    import torch

    return float(
        torch.linalg.vector_norm(left - right)
        / torch.clamp(
            torch.linalg.vector_norm(right),
            min=torch.finfo(right.real.dtype).tiny,
        )
    )


def rank_sweep(
    singular_values: np.ndarray,
    tolerances: list[float],
) -> dict[str, int]:
    leading = float(singular_values[0]) if len(singular_values) else 0.0
    return {
        f"{tolerance:.1e}": (
            int(np.count_nonzero(singular_values > tolerance * leading))
            if leading > 0
            else 0
        )
        for tolerance in tolerances
    }


def spectrum_summary(
    singular_values: np.ndarray,
    tolerances: list[float],
    *,
    complete: bool,
) -> dict[str, Any]:
    values = np.asarray(singular_values, dtype=np.float64)
    squared = values**2
    total = float(np.sum(squared))
    probabilities = squared / total if total > 0 else squared
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    cumulative = np.cumsum(probabilities)
    return {
        "complete_spectrum": bool(complete),
        "singular_value_count": int(len(values)),
        "rank_by_relative_tolerance": rank_sweep(values, tolerances),
        "leading_singular_values": [float(value) for value in values[:64]],
        "normalized_squared_singular_values": [
            float(value) for value in probabilities[:64]
        ],
        "entropy_of_reported_spectrum": entropy,
        "ranks_for_reported_energy": {
            str(target): (
                int(np.searchsorted(cumulative, target, side="left") + 1)
                if total > 0
                else 0
            )
            for target in (0.9, 0.95, 0.99, 0.999)
        },
    }


def exact_small_side_spectrum(amplitudes: Any) -> np.ndarray:
    """Return exact Schmidt values when the selected side has small dimension."""

    import torch

    channel_count, side_dimension, environment_dimension = amplitudes.shape
    flattened = amplitudes.reshape(
        channel_count * side_dimension,
        environment_dimension,
    )
    channel_overlaps = flattened @ flattened.conj().T
    channel_overlaps = channel_overlaps.reshape(
        channel_count,
        side_dimension,
        channel_count,
        side_dimension,
    )
    overlap_features = (
        channel_overlaps.permute(1, 3, 0, 2)
        .contiguous()
        .reshape(side_dimension**2, channel_count**2)
    )
    reshuffled_gram = overlap_features @ overlap_features.conj().T
    small_gram = (
        reshuffled_gram.reshape(
            side_dimension,
            side_dimension,
            side_dimension,
            side_dimension,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(side_dimension**2, side_dimension**2)
    )
    small_gram = 0.5 * (small_gram + small_gram.conj().T)
    eigenvalues = torch.linalg.eigvalsh(small_gram).real
    scale = torch.max(torch.abs(eigenvalues))
    floor = (
        128
        * torch.finfo(eigenvalues.dtype).eps
        * torch.clamp(scale, min=torch.finfo(eigenvalues.dtype).tiny)
    )
    eigenvalues = torch.where(
        eigenvalues > floor,
        eigenvalues,
        torch.zeros_like(eigenvalues),
    )
    return (
        torch.sqrt(torch.clamp(eigenvalues, min=0))
        .flip(0)
        .detach()
        .cpu()
        .numpy()
    )


def random_rank_one_basis(
    dimension: int,
    count: int,
    *,
    generator: Any,
    dtype: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    import torch

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    left = torch.randn(
        count,
        dimension,
        generator=generator,
        dtype=real_dtype,
        device=device,
    ) + 1j * torch.randn(
        count,
        dimension,
        generator=generator,
        dtype=real_dtype,
        device=device,
    )
    right = torch.randn(
        count,
        dimension,
        generator=generator,
        dtype=real_dtype,
        device=device,
    ) + 1j * torch.randn(
        count,
        dimension,
        generator=generator,
        dtype=real_dtype,
        device=device,
    )
    left = left.to(dtype)
    right = right.to(dtype)
    left /= torch.linalg.vector_norm(left, dim=1, keepdim=True)
    right /= torch.linalg.vector_norm(right, dim=1, keepdim=True)
    raw = torch.einsum("pi,pj->ijp", left, right).reshape(
        dimension**2,
        count,
    )
    _, triangular = torch.linalg.qr(raw, mode="reduced")
    return left, right, triangular


def projected_center_spectrum(
    amplitudes: Any,
    *,
    sketch_dimension: int,
    seed: int,
) -> np.ndarray:
    """Return singular values of an orthogonal two-sided random projection.

    The projected rank is a rigorous lower bound on the exact center-cut rank
    up to numerical rank decisions. It is not the complete center spectrum.
    """

    import torch

    channel_count, left_dimension, right_dimension = amplitudes.shape
    del channel_count
    if sketch_dimension > min(left_dimension**2, right_dimension**2):
        raise ValueError("center sketch exceeds an operator-space dimension")
    generator = torch.Generator(device=amplitudes.device)
    generator.manual_seed(seed)
    left_a, left_b, left_triangular = random_rank_one_basis(
        left_dimension,
        sketch_dimension,
        generator=generator,
        dtype=amplitudes.dtype,
        device=amplitudes.device,
    )
    right_a, right_b, right_triangular = random_rank_one_basis(
        right_dimension,
        sketch_dimension,
        generator=generator,
        dtype=amplitudes.dtype,
        device=amplitudes.device,
    )
    first = torch.einsum(
        "pi,air,qr->apq",
        left_a.conj(),
        amplitudes.conj(),
        right_a,
    )
    second = torch.einsum(
        "pj,ajs,qs->apq",
        left_b.conj(),
        amplitudes,
        right_b,
    )
    raw_projection = torch.sum(first * second, dim=0)
    orthogonal_left = torch.linalg.solve(
        left_triangular.conj().T,
        raw_projection,
    )
    orthogonal_projection = torch.linalg.solve(
        right_triangular.T,
        orthogonal_left.T,
    ).T
    return (
        torch.linalg.svdvals(orthogonal_projection)
        .detach()
        .cpu()
        .numpy()
    )


def lift_report(
    purification: Any,
    *,
    local_dimension: int,
    tolerances: list[float],
    center_sketch_dimension: int,
    center_sketch_seed: int,
) -> dict[str, Any]:
    amplitudes = purification.reshape(
        purification.shape[0],
        local_dimension,
        local_dimension,
        local_dimension,
        local_dimension,
    )
    first = exact_small_side_spectrum(
        amplitudes.reshape(
            purification.shape[0],
            local_dimension,
            local_dimension**3,
        )
    )
    third = exact_small_side_spectrum(
        amplitudes.permute(0, 4, 1, 2, 3).reshape(
            purification.shape[0],
            local_dimension,
            local_dimension**3,
        )
    )
    center = projected_center_spectrum(
        amplitudes.reshape(
            purification.shape[0],
            local_dimension**2,
            local_dimension**2,
        ),
        sketch_dimension=center_sketch_dimension,
        seed=center_sketch_seed,
    )
    center_summary = spectrum_summary(
        center,
        tolerances,
        complete=False,
    )
    center_summary.update(
        {
            "projection": (
                "two-sided orthonormalized random rank-one operator subspaces"
            ),
            "sketch_dimension": int(center_sketch_dimension),
            "sketch_seed": int(center_sketch_seed),
            "rank_interpretation": (
                "The numerical rank of this projected matrix is a lower bound "
                "on the exact center-cut operator-Schmidt rank. The displayed "
                "values are not the leading singular values of the full matrix."
            ),
        }
    )
    return {
        "cut_1_of_4": spectrum_summary(first, tolerances, complete=True),
        "cut_2_of_4": center_summary,
        "cut_3_of_4": spectrum_summary(third, tolerances, complete=True),
    }


def main() -> None:
    import torch

    args = parse_args()
    coefficient_path = args.coefficient_artifact.expanduser().resolve()
    full_h_path = args.full_h_artifact.expanduser().resolve()
    tolerances = sorted(
        {float(value) for value in args.rank_tolerances},
        reverse=True,
    )
    if any(value <= 0 for value in tolerances):
        raise SystemExit("rank tolerances must be positive")
    if args.center_sketch_dimension <= 0:
        raise SystemExit("center sketch dimension must be positive")
    if args.alternative_null_scale <= 0:
        raise SystemExit("alternative null scale must be positive")
    dtype = torch.complex64 if args.precision == "complex64" else torch.complex128
    device = torch.device(args.device)

    with np.load(coefficient_path, allow_pickle=False) as payload:
        key = f"mu1111_coefficients_seed_{args.coefficient_seed}"
        coefficients = np.asarray(payload[key], dtype=np.complex128)
        symmetric_indices = np.asarray(
            payload["degree_one_fourfold_indices"],
            dtype=np.int64,
        )
        coefficient_schema = str(payload["schema"].item())
    with np.load(full_h_path, allow_pickle=False) as payload:
        full_h = np.asarray(payload["global_h_matrix"], dtype=np.complex128)
    local_dimension = int(np.max(symmetric_indices)) + 1
    symmetric_lookup = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(symmetric_indices)
    }
    ordered_indices = np.asarray(
        list(product(range(local_dimension), repeat=4)),
        dtype=np.int64,
    )
    ordered_to_symmetric = np.asarray(
        [
            symmetric_lookup[tuple(sorted(int(value) for value in row))]
            for row in ordered_indices
        ],
        dtype=np.int64,
    )
    ordered_coefficients = coefficients[:, ordered_to_symmetric]
    coefficient_tensor = torch.as_tensor(
        ordered_coefficients,
        dtype=dtype,
        device=device,
    )
    full_h_tensor = torch.as_tensor(full_h, dtype=dtype, device=device)
    identity = torch.eye(len(full_h), dtype=dtype, device=device)
    product_gram = coefficient_tensor @ coefficient_tensor.conj().T
    minimum_right_inverse = coefficient_tensor.conj().T @ torch.linalg.solve(
        product_gram,
        identity,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.alternative_seed)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    random_lift = torch.randn(
        minimum_right_inverse.shape,
        generator=generator,
        dtype=real_dtype,
        device=device,
    ) + 1j * torch.randn(
        minimum_right_inverse.shape,
        generator=generator,
        dtype=real_dtype,
        device=device,
    )
    random_lift = random_lift.to(dtype)
    null_component = random_lift - minimum_right_inverse @ (
        coefficient_tensor @ random_lift
    )
    null_component *= (
        args.alternative_null_scale
        * torch.linalg.vector_norm(minimum_right_inverse)
        / torch.clamp(
            torch.linalg.vector_norm(null_component),
            min=torch.finfo(real_dtype).tiny,
        )
    )
    alternative_right_inverse = minimum_right_inverse + null_component

    hermitian_h = 0.5 * (full_h_tensor + full_h_tensor.conj().T)
    cholesky = torch.linalg.cholesky(hermitian_h)
    purifications = {
        "minimum_norm": cholesky.conj().T @ minimum_right_inverse.T,
        "nonminimum_null_perturbation": (
            cholesky.conj().T @ alternative_right_inverse.T
        ),
    }
    reports = {}
    right_inverse_errors = {}
    for index, (label, right_inverse) in enumerate(
        (
            ("minimum_norm", minimum_right_inverse),
            ("nonminimum_null_perturbation", alternative_right_inverse),
        )
    ):
        right_inverse_errors[label] = relative_error(
            coefficient_tensor @ right_inverse,
            identity,
        )
        reports[label] = lift_report(
            purifications[label],
            local_dimension=local_dimension,
            tolerances=tolerances,
            center_sketch_dimension=args.center_sketch_dimension,
            center_sketch_seed=args.center_sketch_seed + index,
        )

    rng = np.random.default_rng(86613)
    random_section = torch.as_tensor(
        rng.normal(size=len(full_h)) + 1j * rng.normal(size=len(full_h)),
        dtype=dtype,
        device=device,
    )
    ordered_products = coefficient_tensor.T @ random_section
    source_value = torch.real(random_section.conj() @ hermitian_h @ random_section)
    value_errors = {}
    for label, purification in purifications.items():
        lifted_value = torch.real(
            torch.sum(torch.abs(purification @ ordered_products) ** 2)
        )
        value_errors[label] = float(
            torch.abs(lifted_value - source_value)
            / torch.clamp(
                torch.abs(source_value),
                min=torch.finfo(source_value.dtype).tiny,
            )
        )

    output = {
        "schema": "gcicy-full-h-equivalent-lifts-all-cut-schmidt-v1",
        "coefficient_artifact": str(coefficient_path),
        "coefficient_artifact_sha256": sha256(coefficient_path),
        "coefficient_schema": coefficient_schema,
        "coefficient_seed": args.coefficient_seed,
        "full_h_artifact": str(full_h_path),
        "full_h_artifact_sha256": sha256(full_h_path),
        "section_count_degree_one": local_dimension,
        "section_count_degree_four": int(len(full_h)),
        "ordered_product_count": int(local_dimension**4),
        "rank_tolerances": tolerances,
        "minimum_norm_right_inverse_error": right_inverse_errors["minimum_norm"],
        "alternative_right_inverse_error": right_inverse_errors[
            "nonminimum_null_perturbation"
        ],
        "alternative_lift": {
            "kind": "deterministic random null-space perturbation",
            "seed": args.alternative_seed,
            "relative_null_component_frobenius_norm": args.alternative_null_scale,
        },
        "random_full_h_value_relative_errors": value_errors,
        "lifts": reports,
        "interpretation_boundary": (
            "Edge-cut spectra are complete for the two explicit coefficient "
            "lifts. Center-cut results are projected spectra and rank lower "
            "bounds only. Comparing two equivalent lifts tests, but does not "
            "optimize over or classify, the full affine family of lifts."
        ),
        "device": str(device),
        "precision": args.precision,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(output, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
