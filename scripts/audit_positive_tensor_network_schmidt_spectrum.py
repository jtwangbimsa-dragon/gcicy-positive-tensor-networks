#!/usr/bin/env python3
"""Audit global Schmidt-channel usage in a positive tensor network.

The shared-dictionary coefficient cores form an MPS representation of the
purification operator B.  Across any spatial cut, B has amplitude Schmidt rank
at most D.  Consequently H = B^dagger B has operator-Schmidt rank at most D^2.
This script measures how much of that available rank is actually used without
materializing the exponentially large degree-k operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SUPPORTED_SCHEMAS = {
    "type11-positive-tensor-network-v1",
    "quintic-positive-tensor-network-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indexed_state_arrays(
    state: dict[str, Any],
    prefix: str,
) -> tuple[np.ndarray, ...]:
    arrays: list[np.ndarray] = []
    index = 0
    while f"{prefix}.{index}" in state:
        value = state[f"{prefix}.{index}"]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays.append(np.asarray(value, dtype=np.complex128))
        index += 1
    if not arrays:
        raise ValueError(f"artifact contains no {prefix} tensors")
    return tuple(arrays)


def materialize_physical_cores(
    payload: dict[str, Any],
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Return MPS cores in an orthonormal physical matrix-element basis."""

    if payload.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError("unrecognized positive tensor-network artifact")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("artifact contains no state_dict")
    architecture = payload.get("architecture")
    diagnostics: dict[str, Any] = {"architecture": architecture}
    if architecture == "shared_local_dictionary":
        coefficients = _indexed_state_arrays(state, "coefficient_cores")
        dictionary_value = state.get("physical_dictionary")
        if dictionary_value is None:
            raise ValueError("shared-dictionary artifact has no physical_dictionary")
        if hasattr(dictionary_value, "detach"):
            dictionary_value = dictionary_value.detach().cpu().numpy()
        dictionary = np.asarray(dictionary_value, dtype=np.complex128)
        flattened_dictionary = dictionary.reshape(len(dictionary), -1)
        if any(core.shape[2] != len(dictionary) for core in coefficients):
            raise ValueError("coefficient-core and dictionary ranks do not agree")
        cores = tuple(
            np.einsum(
                "lrq,qp->lrp",
                coefficient,
                flattened_dictionary,
                optimize=True,
            )
            for coefficient in coefficients
        )
        dictionary_gram = (
            flattened_dictionary @ flattened_dictionary.conj().T
        )
        identity = np.eye(len(dictionary), dtype=np.complex128)
        dictionary_singular_values = np.linalg.svd(
            flattened_dictionary,
            compute_uv=False,
        )
        diagnostics.update(
            {
                "dictionary_rank": int(len(dictionary)),
                "physical_matrix_element_dimension": int(
                    flattened_dictionary.shape[1]
                ),
                "dictionary_numerical_rank": int(
                    np.linalg.matrix_rank(flattened_dictionary)
                ),
                "dictionary_row_orthonormality_frobenius_error": float(
                    np.linalg.norm(dictionary_gram - identity, ord="fro")
                ),
                "dictionary_condition_number": float(
                    dictionary_singular_values[0] / dictionary_singular_values[-1]
                )
                if dictionary_singular_values[-1] > 0
                else float("inf"),
            }
        )
    elif architecture == "dense_local_cores":
        dense_cores = _indexed_state_arrays(state, "cores")
        cores = tuple(core.reshape(*core.shape[:2], -1) for core in dense_cores)
        diagnostics.update(
            {
                "dictionary_rank": None,
                "physical_matrix_element_dimension": int(cores[0].shape[2]),
            }
        )
    else:
        raise ValueError(f"unsupported architecture: {architecture}")

    if len(cores) < 2:
        raise ValueError("Schmidt audit requires at least two sites")
    if cores[0].shape[0] != 1 or cores[-1].shape[1] != 1:
        raise ValueError("tensor network must have scalar open boundaries")
    physical_dimension = int(cores[0].shape[2])
    for left, right in zip(cores, cores[1:]):
        if left.shape[1] != right.shape[0]:
            raise ValueError("adjacent tensor-network bond dimensions do not agree")
        if left.shape[2] != physical_dimension:
            raise ValueError("all sites must share one physical dimension")
    diagnostics["site_count"] = int(len(cores))
    diagnostics["stored_bond_dimension"] = int(payload["bond_dimension"])
    return cores, diagnostics


def _normalize_hermitian_gram(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.conj().T)
    trace = float(np.real(np.trace(matrix)))
    if not np.isfinite(trace) or trace <= 0:
        raise FloatingPointError("Schmidt environment has nonpositive trace")
    return matrix / trace


def left_gram_environments(
    cores: Iterable[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Return normalized L^dagger L after each site, including the boundary."""

    environments = [np.ones((1, 1), dtype=np.complex128)]
    gram = environments[0]
    for core in cores:
        gram = np.einsum(
            "lrp,lm,msp->rs",
            core.conj(),
            gram,
            core,
            optimize=True,
        )
        gram = _normalize_hermitian_gram(gram)
        environments.append(gram)
    return tuple(environments)


def right_gram_environments(
    cores: Iterable[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Return normalized R R^dagger before each site, including the boundary."""

    core_tuple = tuple(cores)
    environments: list[np.ndarray | None] = [None] * (len(core_tuple) + 1)
    gram = np.ones((1, 1), dtype=np.complex128)
    environments[-1] = gram
    for site in range(len(core_tuple) - 1, -1, -1):
        core = core_tuple[site]
        gram = np.einsum(
            "lrp,rs,msp->lm",
            core,
            gram,
            core.conj(),
            optimize=True,
        )
        gram = _normalize_hermitian_gram(gram)
        environments[site] = gram
    return tuple(value for value in environments if value is not None)


def _rank_for_cumulative_energy(probabilities: np.ndarray, target: float) -> int:
    return int(np.searchsorted(np.cumsum(probabilities), target, side="left") + 1)


def spectrum_from_environments(
    left_gram: np.ndarray,
    right_gram: np.ndarray,
    *,
    relative_rank_tolerance: float,
) -> dict[str, Any]:
    """Compute normalized amplitude Schmidt data from two bond Gram matrices."""

    if left_gram.shape != right_gram.shape:
        raise ValueError("left and right Schmidt environments have different shapes")
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (left_gram + left_gram.conj().T)
    )
    largest_left = max(float(np.max(eigenvalues)), 0.0)
    if float(np.min(eigenvalues)) < -1.0e-10 * max(largest_left, 1.0):
        raise FloatingPointError("left Schmidt environment is not positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    overlap = square_root @ right_gram @ square_root
    overlap = 0.5 * (overlap + overlap.conj().T)
    squared_singular_values = np.linalg.eigvalsh(overlap)
    largest_overlap = max(float(np.max(squared_singular_values)), 0.0)
    if float(np.min(squared_singular_values)) < -1.0e-9 * max(
        largest_overlap, 1.0
    ):
        raise FloatingPointError("combined Schmidt environment is not positive")
    squared_singular_values = np.clip(squared_singular_values, 0.0, None)[::-1]
    total = float(np.sum(squared_singular_values))
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Schmidt spectrum has zero or nonfinite weight")
    probabilities = squared_singular_values / total
    singular_values = np.sqrt(probabilities)
    relative_singular_values = singular_values / singular_values[0]
    numerical_rank = int(
        np.sum(relative_singular_values > relative_rank_tolerance)
    )
    positive = probabilities > 0
    entropy_rank = float(
        np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    )
    participation_rank = float(1.0 / np.sum(probabilities**2))
    return {
        "bond_dimension": int(left_gram.shape[0]),
        "normalized_squared_singular_values": probabilities.tolist(),
        "normalized_singular_values": singular_values.tolist(),
        "relative_singular_values": relative_singular_values.tolist(),
        "relative_rank_tolerance": float(relative_rank_tolerance),
        "amplitude_numerical_rank": numerical_rank,
        "amplitude_entropy_effective_rank": entropy_rank,
        "amplitude_participation_effective_rank": participation_rank,
        "rank_for_cumulative_squared_weight": {
            "0.99": _rank_for_cumulative_energy(probabilities, 0.99),
            "0.999": _rank_for_cumulative_energy(probabilities, 0.999),
            "0.9999": _rank_for_cumulative_energy(probabilities, 0.9999),
            "0.999999": _rank_for_cumulative_energy(probabilities, 0.999999),
        },
        "positive_operator_schmidt_rank_upper_bound": numerical_rank**2,
        "stored_positive_operator_rank_upper_bound": int(left_gram.shape[0] ** 2),
    }


def schmidt_spectrum_across_cuts(
    cores: Iterable[np.ndarray],
    *,
    relative_rank_tolerance: float = 1.0e-10,
) -> list[dict[str, Any]]:
    core_tuple = tuple(np.asarray(core, dtype=np.complex128) for core in cores)
    if not 0 < relative_rank_tolerance < 1:
        raise ValueError("relative_rank_tolerance must lie in (0,1)")
    left = left_gram_environments(core_tuple)
    right = right_gram_environments(core_tuple)
    rows = []
    for cut in range(1, len(core_tuple)):
        row = spectrum_from_environments(
            left[cut],
            right[cut],
            relative_rank_tolerance=relative_rank_tolerance,
        )
        row["cut_after_site"] = cut
        row["left_site_count"] = cut
        row["right_site_count"] = len(core_tuple) - cut
        rows.append(row)
    return rows


def summarize_cuts(cuts: list[dict[str, Any]]) -> dict[str, Any]:
    central = min(
        cuts,
        key=lambda row: abs(row["left_site_count"] - row["right_site_count"]),
    )
    return {
        "central_cut_after_site": int(central["cut_after_site"]),
        "central_cut": central,
        "maximum_amplitude_numerical_rank": int(
            max(row["amplitude_numerical_rank"] for row in cuts)
        ),
        "maximum_amplitude_entropy_effective_rank": float(
            max(row["amplitude_entropy_effective_rank"] for row in cuts)
        ),
        "maximum_amplitude_participation_effective_rank": float(
            max(row["amplitude_participation_effective_rank"] for row in cuts)
        ),
        "minimum_rank_for_99_9_percent_weight": int(
            min(row["rank_for_cumulative_squared_weight"]["0.999"] for row in cuts)
        ),
        "maximum_rank_for_99_9_percent_weight": int(
            max(row["rank_for_cumulative_squared_weight"]["0.999"] for row in cuts)
        ),
    }


def main() -> None:
    import torch

    args = parse_args()
    if not 0 < args.relative_rank_tolerance < 1:
        raise SystemExit("--relative-rank-tolerance must lie in (0,1)")
    model_path = args.model.expanduser().resolve()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    cores, metadata = materialize_physical_cores(payload)
    cuts = schmidt_spectrum_across_cuts(
        cores,
        relative_rank_tolerance=args.relative_rank_tolerance,
    )
    report = {
        "schema": "positive-tensor-network-schmidt-spectrum-audit-v1",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "metadata": metadata,
        "summary": summarize_cuts(cuts),
        "cuts": cuts,
        "interpretation": {
            "audited_object": "purification operator B in Hilbert-Schmidt basis",
            "positive_operator": "H = B^dagger B",
            "rank_statement": (
                "If B has amplitude Schmidt rank r across a cut, H has "
                "operator-Schmidt rank at most r^2 across the same cut."
            ),
        },
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
