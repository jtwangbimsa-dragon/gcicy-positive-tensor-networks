#!/usr/bin/env python3
"""Audit the effective operator rank used by a positive tensor network."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path)
    parser.add_argument("--comparison-rank", type=int, default=22)
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_arrays(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy()
        for key, value in payload["state_dict"].items()
    }


def indexed_arrays(state: dict[str, np.ndarray], prefix: str) -> list[np.ndarray]:
    rows = []
    index = 0
    while f"{prefix}.{index}" in state:
        rows.append(np.asarray(state[f"{prefix}.{index}"], dtype=np.complex128))
        index += 1
    if not rows:
        raise ValueError(f"artifact contains no {prefix} tensors")
    return rows


def materialized_core_matrix(payload: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if payload.get("schema") != "type11-positive-tensor-network-v1":
        raise ValueError("unrecognized positive tensor-network artifact")
    state = state_arrays(payload)
    architecture = payload["architecture"]
    dictionary_diagnostics: dict[str, Any] = {}
    if architecture == "shared_local_dictionary":
        dictionary = np.asarray(state["physical_dictionary"], dtype=np.complex128)
        flattened_dictionary = dictionary.reshape(len(dictionary), -1)
        gram = flattened_dictionary @ flattened_dictionary.conj().T
        dictionary_diagnostics = {
            "physical_dictionary_rank": int(len(dictionary)),
            "row_orthonormality_frobenius_error": float(
                np.linalg.norm(gram - np.eye(len(dictionary)), ord="fro")
            ),
        }
        cores = [
            np.einsum("lrq,qpi->lrpi", coefficient, dictionary)
            for coefficient in indexed_arrays(state, "coefficient_cores")
        ]
    elif architecture == "dense_local_cores":
        cores = indexed_arrays(state, "cores")
    else:
        raise ValueError(f"unsupported tensor-network architecture: {architecture}")
    section_count = int(cores[0].shape[-1])
    if any(core.shape[-2:] != (section_count, section_count) for core in cores):
        raise ValueError("local cores do not share a square physical shape")
    bond_norms = [
        np.linalg.norm(core.reshape(*core.shape[:2], -1), axis=2)
        for core in cores
    ]
    flattened_bond_norms = np.concatenate([value.reshape(-1) for value in bond_norms])
    maximum_bond_norm = float(np.max(flattened_bond_norms))
    relative_bond_norms = flattened_bond_norms / maximum_bond_norm
    matrix = np.concatenate(
        [core.reshape(-1, section_count**2) for core in cores],
        axis=0,
    )
    metadata = {
        "architecture": architecture,
        "site_count": int(payload["site_count"]),
        "bond_dimension": int(payload["bond_dimension"]),
        "section_count": section_count,
        "local_core_matrix_count": int(len(matrix)),
        "bond_block_norms": [value.tolist() for value in bond_norms],
        "bond_block_count": int(len(flattened_bond_norms)),
        "exactly_zero_bond_block_count": int(np.sum(flattened_bond_norms == 0)),
        "bond_block_counts_above_relative_threshold": {
            str(threshold): int(np.sum(relative_bond_norms > threshold))
            for threshold in (1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3)
        },
        "minimum_positive_relative_bond_block_norm": float(
            np.min(relative_bond_norms[relative_bond_norms > 0])
        )
        if np.any(relative_bond_norms > 0)
        else 0.0,
        **dictionary_diagnostics,
    }
    return matrix, metadata


def spectrum_summary(
    matrix: np.ndarray,
    *,
    comparison_rank: int,
    relative_rank_tolerance: float,
) -> tuple[dict[str, Any], np.ndarray]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    squared = singular_values**2
    total = float(np.sum(squared))
    if total <= 0 or not np.isfinite(total):
        raise FloatingPointError("materialized local cores have zero or nonfinite norm")
    probabilities = squared / total
    positive = probabilities > 0
    entropy_rank = float(
        np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    )
    participation_rank = float(1.0 / np.sum(probabilities**2))
    cutoff = min(comparison_rank, len(singular_values))
    tail_fraction = float(np.sum(squared[cutoff:]) / total)
    numerical_rank = int(
        np.sum(singular_values > relative_rank_tolerance * singular_values[0])
    )
    summary = {
        "singular_values": singular_values.tolist(),
        "relative_singular_values": (singular_values / singular_values[0]).tolist(),
        "relative_rank_tolerance": relative_rank_tolerance,
        "numerical_rank": numerical_rank,
        "entropy_effective_rank": entropy_rank,
        "participation_effective_rank": participation_rank,
        "comparison_rank": comparison_rank,
        "frobenius_energy_fraction_above_comparison_rank": tail_fraction,
        "best_comparison_rank_relative_frobenius_error": float(np.sqrt(tail_fraction)),
    }
    return summary, singular_values


def reference_comparison(
    matrix: np.ndarray,
    reference: np.ndarray,
    comparison_rank: int,
) -> dict[str, float]:
    if matrix.shape[1] != reference.shape[1]:
        raise ValueError("model and reference physical dimensions differ")
    _, _, candidate_vh = np.linalg.svd(matrix, full_matrices=False)
    _, _, reference_vh = np.linalg.svd(reference, full_matrices=False)
    rank = min(comparison_rank, len(candidate_vh), len(reference_vh))
    candidate_basis = candidate_vh[:rank]
    reference_basis = reference_vh[:rank]
    cosines = np.linalg.svd(
        candidate_basis @ reference_basis.conj().T,
        compute_uv=False,
    )
    cosines = np.clip(cosines, 0.0, 1.0)
    projection = matrix @ reference_basis.conj().T @ reference_basis
    return {
        "rank_compared": rank,
        "minimum_principal_angle_cosine": float(np.min(cosines)),
        "maximum_principal_angle_radians": float(np.max(np.arccos(cosines))),
        "candidate_relative_residual_outside_reference_rank_subspace": float(
            np.linalg.norm(matrix - projection, ord="fro")
            / np.linalg.norm(matrix, ord="fro")
        ),
    }


def main() -> None:
    import torch

    args = parse_args()
    if args.comparison_rank <= 0:
        raise SystemExit("--comparison-rank must be positive")
    if not 0 < args.relative_rank_tolerance < 1:
        raise SystemExit("--relative-rank-tolerance must lie in (0,1)")
    model_path = args.model.expanduser().resolve()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    matrix, metadata = materialized_core_matrix(payload)
    spectrum, _ = spectrum_summary(
        matrix,
        comparison_rank=args.comparison_rank,
        relative_rank_tolerance=args.relative_rank_tolerance,
    )
    report: dict[str, Any] = {
        "schema": "positive-tensor-network-core-spectrum-audit-v1",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "metadata": metadata,
        "spectrum": spectrum,
    }
    if args.reference_model is not None:
        reference_path = args.reference_model.expanduser().resolve()
        reference_payload = torch.load(
            reference_path,
            map_location="cpu",
            weights_only=False,
        )
        reference_matrix, reference_metadata = materialized_core_matrix(reference_payload)
        report["reference"] = {
            "model": str(reference_path),
            "model_sha256": sha256_file(reference_path),
            "metadata": reference_metadata,
            "comparison": reference_comparison(
                matrix,
                reference_matrix,
                args.comparison_rank,
            ),
        }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
