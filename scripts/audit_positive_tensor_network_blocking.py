#!/usr/bin/env python3
"""Audit whether pairwise Veronese blocking preserves a trained positive TN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def normalized_symmetric_square_embedding(section_count: int) -> np.ndarray:
    """Return the isometry Sym^2(C^d) -> C^d tensor C^d."""

    if section_count <= 0:
        raise ValueError("section_count must be positive")
    symmetric_count = section_count * (section_count + 1) // 2
    embedding = np.zeros(
        (section_count, section_count, symmetric_count),
        dtype=np.complex128,
    )
    column = 0
    for left in range(section_count):
        for right in range(left, section_count):
            if left == right:
                embedding[left, right, column] = 1.0
            else:
                embedding[left, right, column] = 1.0 / np.sqrt(2.0)
                embedding[right, left, column] = 1.0 / np.sqrt(2.0)
            column += 1
    return embedding


def block_core_pair(
    first: np.ndarray,
    second: np.ndarray,
    embedding: np.ndarray,
) -> np.ndarray:
    """Contract an internal bond and restrict the two inputs to Sym^2."""

    first_array = np.asarray(first, dtype=np.complex128)
    second_array = np.asarray(second, dtype=np.complex128)
    source_embedding = np.asarray(embedding, dtype=np.complex128)
    if first_array.ndim != 4 or second_array.ndim != 4:
        raise ValueError("dense cores must have shape (left, right, output, input)")
    if first_array.shape[1] != second_array.shape[0]:
        raise ValueError("paired cores have incompatible internal bonds")
    if first_array.shape[2:] != second_array.shape[2:]:
        raise ValueError("paired cores have different physical dimensions")
    section_count = first_array.shape[3]
    expected_embedding_shape = (
        section_count,
        section_count,
        section_count * (section_count + 1) // 2,
    )
    if source_embedding.shape != expected_embedding_shape:
        raise ValueError("symmetric-square embedding has the wrong shape")
    return np.einsum(
        "lbpi,brqj,ija->lrpqa",
        first_array,
        second_array,
        source_embedding,
        optimize=True,
    )


def project_block_output_to_symmetric(
    rectangular_block: np.ndarray,
    embedding: np.ndarray,
) -> np.ndarray:
    """Project the unrestricted output pair onto its symmetric subspace."""

    block = np.asarray(rectangular_block, dtype=np.complex128)
    source_embedding = np.asarray(embedding, dtype=np.complex128)
    if block.ndim != 5 or block.shape[2:4] != source_embedding.shape[:2]:
        raise ValueError("blocked core and output embedding are incompatible")
    return np.einsum(
        "pqa,lrpqb->lrab",
        np.conj(source_embedding),
        block,
        optimize=True,
    )


def materialized_dense_cores(payload: dict[str, Any]) -> tuple[np.ndarray, ...]:
    state = payload["state_dict"]
    site_count = int(payload["site_count"])
    architecture = payload.get("architecture", "dense_local_cores")
    if architecture == "dense_local_cores":
        return tuple(
            np.asarray(state[f"cores.{site}"].detach().cpu(), dtype=np.complex128)
            for site in range(site_count)
        )
    if architecture != "shared_local_dictionary":
        raise ValueError(f"unsupported architecture: {architecture}")
    dictionary = np.asarray(
        state["physical_dictionary"].detach().cpu(),
        dtype=np.complex128,
    )
    return tuple(
        np.einsum(
            "lrq,qpi->lrpi",
            np.asarray(
                state[f"coefficient_cores.{site}"].detach().cpu(),
                dtype=np.complex128,
            ),
            dictionary,
            optimize=True,
        )
        for site in range(site_count)
    )


def shared_dictionary_spectrum(
    blocks: tuple[np.ndarray, ...],
    *,
    reported_ranks: tuple[int, ...] = (),
) -> dict[str, Any]:
    rows = np.concatenate(
        [block.reshape(-1, int(np.prod(block.shape[2:]))) for block in blocks],
        axis=0,
    )
    singular_values = np.linalg.svd(rows, compute_uv=False)
    squared = np.square(singular_values)
    total = float(np.sum(squared))
    tail_squared = np.concatenate((np.cumsum(squared[::-1])[::-1], [0.0]))
    residuals = np.sqrt(np.maximum(tail_squared, 0.0) / total)
    machine_tolerance = (
        singular_values[0]
        * max(rows.shape)
        * np.finfo(np.float64).eps
    )
    numerical_rank = int(np.count_nonzero(singular_values > machine_tolerance))

    ranks_for_residual: dict[str, int | None] = {}
    for tolerance in (1.0e-1, 5.0e-2, 1.0e-2, 5.0e-3, 1.0e-3, 1.0e-4):
        admissible = np.flatnonzero(residuals <= tolerance)
        ranks_for_residual[f"{tolerance:.0e}"] = (
            int(admissible[0]) if len(admissible) else None
        )
    return {
        "matrix_shape": list(rows.shape),
        "numerical_rank": numerical_rank,
        "machine_rank_tolerance": float(machine_tolerance),
        "leading_singular_values": [
            float(value) for value in singular_values[: min(20, len(singular_values))]
        ],
        "rank_for_relative_frobenius_residual": ranks_for_residual,
        "relative_frobenius_residual_at_rank": {
            str(rank): float(residuals[min(rank, len(singular_values))])
            for rank in sorted(
                set((25, 64, 74, 79, 95, 114, 225) + reported_ranks)
            )
            if rank <= len(singular_values)
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-budget", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    cores = materialized_dense_cores(payload)
    if len(cores) % 2:
        raise ValueError("pairwise blocking requires an even site count")
    input_count = int(cores[0].shape[3])
    if any(core.shape[2:] != (input_count, input_count) for core in cores):
        raise ValueError("the source model must have square local physical maps")
    embedding = normalized_symmetric_square_embedding(input_count)
    gram = embedding.reshape(input_count**2, -1).conj().T @ embedding.reshape(
        input_count**2, -1
    )
    if not np.allclose(gram, np.eye(len(gram)), rtol=1.0e-13, atol=1.0e-13):
        raise FloatingPointError("symmetric-square embedding is not isometric")

    rectangular_blocks = tuple(
        block_core_pair(cores[index], cores[index + 1], embedding)
        for index in range(0, len(cores), 2)
    )
    projected_blocks = tuple(
        project_block_output_to_symmetric(block, embedding)
        for block in rectangular_blocks
    )
    pair_rows = []
    total_rectangular_norm_squared = 0.0
    total_projected_norm_squared = 0.0
    for pair, (rectangular, projected) in enumerate(
        zip(rectangular_blocks, projected_blocks, strict=True)
    ):
        rectangular_norm_squared = float(np.sum(np.abs(rectangular) ** 2))
        projected_norm_squared = float(np.sum(np.abs(projected) ** 2))
        total_rectangular_norm_squared += rectangular_norm_squared
        total_projected_norm_squared += projected_norm_squared
        pair_rows.append(
            {
                "pair": pair,
                "external_bonds": list(rectangular.shape[:2]),
                "rectangular_output_input_shape": list(rectangular.shape[2:]),
                "symmetric_output_input_shape": list(projected.shape[2:]),
                "antisymmetric_output_frobenius_fraction": float(
                    max(
                        0.0,
                        1.0
                        - projected_norm_squared / rectangular_norm_squared,
                    )
                ),
            }
        )

    blocked_site_count = len(rectangular_blocks)
    bond_dimension = int(payload["bond_dimension"])
    coefficient_factor = (
        2 * bond_dimension
        + max(0, blocked_site_count - 2) * bond_dimension**2
    )
    symmetric_count = embedding.shape[2]
    rectangular_local_dimension = input_count**2 * symmetric_count
    square_local_dimension = symmetric_count**2
    maximum_rectangular_rank = args.parameter_budget // (
        2 * (rectangular_local_dimension + coefficient_factor)
    )
    maximum_square_rank = args.parameter_budget // (
        2 * (square_local_dimension + coefficient_factor)
    )
    maximum_fixed_rectangular_rank = min(
        rectangular_local_dimension,
        args.parameter_budget // (2 * coefficient_factor),
    )
    maximum_fixed_square_rank = min(
        square_local_dimension,
        args.parameter_budget // (2 * coefficient_factor),
    )
    report = {
        "schema": "positive-tensor-network-pair-blocking-audit-v1",
        "source": {
            "model": str(model_path),
            "sha256": sha256_file(model_path),
            "site_count": len(cores),
            "input_and_output_dimension": input_count,
            "bond_dimension": bond_dimension,
        },
        "blocking": {
            "blocked_site_count": blocked_site_count,
            "symmetric_input_dimension": symmetric_count,
            "unrestricted_blocked_output_dimension": input_count**2,
            "forced_symmetric_output_dimension": symmetric_count,
            "aggregate_antisymmetric_output_frobenius_fraction": float(
                max(
                    0.0,
                    1.0
                    - total_projected_norm_squared
                    / total_rectangular_norm_squared,
                )
            ),
            "pairs": pair_rows,
        },
        "shared_dictionary": {
            "exact_rectangular_block": shared_dictionary_spectrum(
                rectangular_blocks,
                reported_ranks=(
                    maximum_rectangular_rank,
                    maximum_fixed_rectangular_rank,
                ),
            ),
            "forced_symmetric_output_block": shared_dictionary_spectrum(
                projected_blocks,
                reported_ranks=(
                    maximum_square_rank,
                    maximum_fixed_square_rank,
                ),
            ),
        },
        "parameter_budget": {
            "real_parameters": args.parameter_budget,
            "coefficient_factor_per_dictionary_element": coefficient_factor,
            "rectangular_local_operator_dimension": rectangular_local_dimension,
            "square_local_operator_dimension": square_local_dimension,
            "maximum_rectangular_dictionary_rank": maximum_rectangular_rank,
            "maximum_square_dictionary_rank": maximum_square_rank,
            "maximum_fixed_rectangular_dictionary_rank": (
                maximum_fixed_rectangular_rank
            ),
            "maximum_fixed_square_dictionary_rank": maximum_fixed_square_rank,
            "exact_rectangular_core_real_parameters_with_fixed_dictionary": (
                2 * rectangular_local_dimension * coefficient_factor
            ),
            "exact_rectangular_real_parameters_with_trainable_dictionary": (
                2
                * rectangular_local_dimension
                * (rectangular_local_dimension + coefficient_factor)
            ),
        },
        "interpretation": {
            "exact_pair_blocking_requires_unrestricted_output": True,
            "forced_symmetric_output_is_not_a_nested_reparameterization": True,
            "frobenius_spectrum_is_a_core_space_diagnostic_not_a_metric_error_bound": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
