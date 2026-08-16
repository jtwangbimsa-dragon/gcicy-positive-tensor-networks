#!/usr/bin/env python3
"""Audit a canonical full-H lift's first-cut operator-Schmidt spectrum."""

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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
            if leading > 0.0
            else 0
        )
        for tolerance in tolerances
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
    if any(value <= 0.0 for value in tolerances):
        raise SystemExit("rank tolerances must be positive")
    dtype = (
        torch.complex64
        if args.precision == "complex64"
        else torch.complex128
    )
    device = torch.device(args.device)

    with np.load(coefficient_path, allow_pickle=False) as payload:
        key = f"mu1111_coefficients_seed_{args.coefficient_seed}"
        if key not in payload.files:
            raise KeyError(f"{coefficient_path} has no {key}")
        coefficients = np.asarray(payload[key], dtype=np.complex128)
        symmetric_indices = np.asarray(
            payload["degree_one_fourfold_indices"],
            dtype=np.int64,
        )
        coefficient_schema = str(payload["schema"].item())
    with np.load(full_h_path, allow_pickle=False) as payload:
        full_h = np.asarray(payload["global_h_matrix"], dtype=np.complex128)
        source_exponents = np.asarray(
            payload["global_section_exponents"],
            dtype=np.int64,
        )
    if coefficients.shape[0] != full_h.shape[0]:
        raise ValueError("multiplication coefficients and full H disagree")
    if full_h.shape[0] != full_h.shape[1]:
        raise ValueError("full H must be square")
    if len(symmetric_indices) != coefficients.shape[1]:
        raise ValueError("symmetric product index count is inconsistent")

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
    identity = torch.eye(
        len(full_h),
        dtype=dtype,
        device=device,
    )
    product_gram = coefficient_tensor @ coefficient_tensor.conj().T
    right_inverse = coefficient_tensor.conj().T @ torch.linalg.solve(
        product_gram,
        identity,
    )
    right_inverse_error = relative_error(
        coefficient_tensor @ right_inverse,
        identity,
    )

    hermitian_h = 0.5 * (full_h_tensor + full_h_tensor.conj().T)
    cholesky = torch.linalg.cholesky(hermitian_h)
    purification = cholesky.conj().T @ right_inverse.T
    amplitudes = purification.reshape(
        len(full_h),
        local_dimension,
        local_dimension**3,
    )

    # For the first cut, each purification channel is an
    # (n_local x n_local^3) matrix. The left operator-Schmidt Gram matrix
    # follows from all pairwise channel overlaps without materializing the
    # 121 x 121^3 coefficient matrix.
    flattened = amplitudes.reshape(
        len(full_h) * local_dimension,
        local_dimension**3,
    )
    channel_overlaps = flattened @ flattened.conj().T
    channel_overlaps = channel_overlaps.reshape(
        len(full_h),
        local_dimension,
        len(full_h),
        local_dimension,
    )
    overlap_features = (
        channel_overlaps.permute(1, 3, 0, 2)
        .contiguous()
        .reshape(local_dimension**2, len(full_h) ** 2)
    )
    reshuffled_gram = overlap_features @ overlap_features.conj().T
    left_gram = (
        reshuffled_gram.reshape(
            local_dimension,
            local_dimension,
            local_dimension,
            local_dimension,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(local_dimension**2, local_dimension**2)
    )
    left_gram = 0.5 * (left_gram + left_gram.conj().T)
    eigenvalues = torch.linalg.eigvalsh(left_gram).real
    scale = torch.max(torch.abs(eigenvalues))
    numerical_floor = (
        128.0
        * torch.finfo(eigenvalues.dtype).eps
        * torch.clamp(scale, min=torch.finfo(eigenvalues.dtype).tiny)
    )
    eigenvalues = torch.where(
        eigenvalues > numerical_floor,
        eigenvalues,
        torch.zeros_like(eigenvalues),
    )
    singular_values = (
        torch.sqrt(torch.clamp(eigenvalues, min=0.0))
        .flip(0)
        .detach()
        .cpu()
        .numpy()
    )

    rng = np.random.default_rng(72901)
    random_section = rng.normal(size=len(full_h)) + 1j * rng.normal(
        size=len(full_h)
    )
    random_section_tensor = torch.as_tensor(
        random_section,
        dtype=dtype,
        device=device,
    )
    ordered_products = coefficient_tensor.T @ random_section_tensor
    reconstructed_section = right_inverse.T @ ordered_products
    source_value = torch.real(
        random_section_tensor.conj()
        @ hermitian_h
        @ random_section_tensor
    )
    lifted_value = torch.real(torch.sum(torch.abs(purification @ ordered_products) ** 2))

    output = {
        "schema": "gcicy-full-h-canonical-lift-schmidt-audit-v1",
        "coefficient_artifact": str(coefficient_path),
        "coefficient_artifact_sha256": sha256(coefficient_path),
        "coefficient_schema": coefficient_schema,
        "coefficient_seed": int(args.coefficient_seed),
        "full_h_artifact": str(full_h_path),
        "full_h_artifact_sha256": sha256(full_h_path),
        "section_count_degree_one": local_dimension,
        "section_count_degree_four": int(len(full_h)),
        "ordered_product_count": int(local_dimension**4),
        "source_exponent_count": int(len(source_exponents)),
        "lift": "minimum-norm ordered right inverse of the fitted symmetric multiplication map",
        "right_inverse_relative_error": right_inverse_error,
        "random_section_reconstruction_relative_error": relative_error(
            reconstructed_section,
            random_section_tensor,
        ),
        "random_full_h_value_relative_error": float(
            torch.abs(lifted_value - source_value)
            / torch.clamp(
                torch.abs(source_value),
                min=torch.finfo(source_value.dtype).tiny,
            )
        ),
        "first_cut": {
            "left_operator_dimension": int(local_dimension**2),
            "right_operator_dimension": int(local_dimension**6),
            "rank_by_relative_tolerance": rank_sweep(
                singular_values,
                tolerances,
            ),
            "leading_singular_values": [
                float(value) for value in singular_values[:32]
            ],
            "singular_value_ratio_9_to_1": (
                float(singular_values[8] / singular_values[0])
                if len(singular_values) > 8 and singular_values[0] > 0
                else 0.0
            ),
        },
        "interpretation_boundary": (
            "This is a rank audit of one canonical minimum-norm coefficient "
            "lift. Rank above D rules out exact representation of this lift "
            "by a bond-D chain, but is not a theorem about every "
            "diagonal-equivalent lift."
        ),
        "device": str(device),
        "precision": args.precision,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(output, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
