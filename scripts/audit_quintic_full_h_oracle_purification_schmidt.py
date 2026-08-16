#!/usr/bin/env python3
"""Audit a symmetry-natural purification of a Fermat full-H oracle.

For a positive degree-k section metric ``H``, choose the principal square root
``B = H^(1/2)``.  Quotient-basis monomials are embedded into k ordered
degree-one sites by uniform symmetrization.  This produces an exact local
purification whose squared norm is ``s^dagger H s``.

The full local tensor has ``25^k`` entries for a quintic and is never
materialized.  Across a cut, rows and columns with the same left/right
occupation counts are identical.  We therefore build the exactly equivalent
weighted sparse matrix on occupation-count classes and compute its leading
Schmidt spectrum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-h", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-singular-values", type=int, default=256)
    parser.add_argument("--exact-row-dimension", type=int, default=1500)
    parser.add_argument("--relative-entry-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--cuts",
        type=int,
        nargs="*",
        help="Cuts to audit. By default, audit 1,...,floor(k/2).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weak_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if total < 0 or parts <= 0:
        raise ValueError("composition arguments are invalid")
    if parts == 1:
        return ((total,),)
    rows = []
    for first in range(total + 1):
        for tail in weak_compositions(total - first, parts - 1):
            rows.append((first, *tail))
    return tuple(rows)


def multinomial_count(exponent: tuple[int, ...]) -> int:
    total = sum(exponent)
    result = math.factorial(total)
    for value in exponent:
        result //= math.factorial(value)
    return result


def principal_positive_square_root(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    value = 0.5 * (value + value.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues[0] <= 0:
        raise ValueError("full-H oracle must be positive definite")
    return (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conj().T


def cholesky_purification(matrix: np.ndarray) -> np.ndarray:
    """Return ``B`` with ``B^dagger B = matrix`` from a Cholesky factor."""

    value = np.asarray(matrix, dtype=np.complex128)
    value = 0.5 * (value + value.conj().T)
    lower = np.linalg.cholesky(value)
    return lower.conj().T


def block_spectral_purification(
    matrix: np.ndarray,
    *,
    relative_block_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Diagonalize each exact sparsity block without mixing distinct blocks."""

    if not 0 <= relative_block_tolerance < 1:
        raise ValueError("relative block tolerance must lie in [0,1)")
    value = np.asarray(matrix, dtype=np.complex128)
    value = 0.5 * (value + value.conj().T)
    largest = float(np.max(np.abs(value)))
    adjacency = np.abs(value) > relative_block_tolerance * largest
    unseen = set(range(len(value)))
    blocks: list[tuple[int, ...]] = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbors = set(np.flatnonzero(adjacency[current]))
            new = neighbors & unseen
            unseen.difference_update(new)
            component.update(new)
            frontier.extend(new)
        blocks.append(tuple(sorted(component)))
    blocks.sort(key=lambda row: row[0])

    factor = np.zeros_like(value)
    for block in blocks:
        indices = np.asarray(block, dtype=np.int64)
        submatrix = value[np.ix_(indices, indices)]
        eigenvalues, eigenvectors = np.linalg.eigh(submatrix)
        if eigenvalues[0] <= 0:
            raise ValueError("full-H block must be positive definite")
        subfactor = np.sqrt(eigenvalues)[:, None] * eigenvectors.conj().T
        factor[np.ix_(indices, indices)] = subfactor
    return factor, tuple(blocks)


def _validate_basis(exponents: np.ndarray) -> tuple[int, int]:
    values = np.asarray(exponents, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] <= 0:
        raise ValueError("section exponents must be a nonempty matrix")
    if np.any(values < 0):
        raise ValueError("section exponents must be nonnegative")
    degrees = np.sum(values, axis=1)
    if not np.all(degrees == degrees[0]):
        raise ValueError("section exponents do not have a common degree")
    if len({tuple(row) for row in values}) != len(values):
        raise ValueError("section exponent basis contains duplicates")
    return int(degrees[0]), int(values.shape[1])


def _compatible_splits(
    exponent: tuple[int, ...],
    *,
    left_degree: int,
    left_index: dict[tuple[int, ...], int],
    right_index: dict[tuple[int, ...], int],
) -> tuple[tuple[int, int, int, int], ...]:
    rows = []
    for left in left_index:
        if any(left_component > total for left_component, total in zip(left, exponent)):
            continue
        right = tuple(total - left_component for left_component, total in zip(left, exponent))
        if right not in right_index:
            continue
        if sum(left) != left_degree:
            raise RuntimeError("left composition has the wrong degree")
        rows.append(
            (
                left_index[left],
                right_index[right],
                multinomial_count(left),
                multinomial_count(right),
            )
        )
    return tuple(rows)


def compressed_coefficient_cut(
    exponents: np.ndarray,
    coefficient_matrix: np.ndarray,
    *,
    cut: int,
    row_embedding: str,
    column_embedding: str,
    relative_entry_tolerance: float = 1.0e-12,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Compress a lifted section-pair tensor to occupation-count classes."""

    degree, coordinate_count = _validate_basis(exponents)
    if not 0 < cut < degree:
        raise ValueError("cut must lie strictly inside the algebraic degree")
    if not 0 <= relative_entry_tolerance < 1:
        raise ValueError("relative entry tolerance must lie in [0,1)")
    valid_embeddings = {"normalized", "uniform"}
    if row_embedding not in valid_embeddings:
        raise ValueError("unknown row embedding")
    if column_embedding not in valid_embeddings:
        raise ValueError("unknown column embedding")
    matrix = np.asarray(coefficient_matrix, dtype=np.complex128)
    if matrix.shape != (len(exponents), len(exponents)):
        raise ValueError("coefficient matrix and section basis dimensions differ")
    largest = float(np.max(np.abs(matrix)))
    threshold = relative_entry_tolerance * largest
    nonzero_pairs = np.argwhere(np.abs(matrix) > threshold)

    left_compositions = weak_compositions(cut, coordinate_count)
    right_compositions = weak_compositions(degree - cut, coordinate_count)
    left_index = {value: index for index, value in enumerate(left_compositions)}
    right_index = {value: index for index, value in enumerate(right_compositions)}
    exponent_rows = tuple(tuple(int(value) for value in row) for row in exponents)
    split_rows = tuple(
        _compatible_splits(
            exponent,
            left_degree=cut,
            left_index=left_index,
            right_index=right_index,
        )
        for exponent in exponent_rows
    )
    full_multiplicities = np.asarray(
        [multinomial_count(exponent) for exponent in exponent_rows],
        dtype=np.float64,
    )

    left_count = len(left_compositions)
    right_count = len(right_compositions)
    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[complex] = []
    for output_index, input_index in nonzero_pairs:
        value = matrix[output_index, input_index]
        row_denominator = (
            math.sqrt(full_multiplicities[output_index])
            if row_embedding == "normalized"
            else full_multiplicities[output_index]
        )
        column_denominator = (
            math.sqrt(full_multiplicities[input_index])
            if column_embedding == "normalized"
            else full_multiplicities[input_index]
        )
        denominator = row_denominator * column_denominator
        for (
            output_left,
            output_right,
            output_left_multiplicity,
            output_right_multiplicity,
        ) in split_rows[output_index]:
            for (
                input_left,
                input_right,
                input_left_multiplicity,
                input_right_multiplicity,
            ) in split_rows[input_index]:
                row_indices.append(output_left * left_count + input_left)
                column_indices.append(output_right * right_count + input_right)
                repetition_weight = math.sqrt(
                    output_left_multiplicity
                    * input_left_multiplicity
                    * output_right_multiplicity
                    * input_right_multiplicity
                )
                data.append(value * repetition_weight / denominator)

    result = sparse.coo_matrix(
        (np.asarray(data, dtype=np.complex128), (row_indices, column_indices)),
        shape=(left_count**2, right_count**2),
    ).tocsr()
    result.sum_duplicates()
    diagnostics = {
        "degree": degree,
        "coordinate_count": coordinate_count,
        "cut": int(cut),
        "shape": [int(result.shape[0]), int(result.shape[1])],
        "nonzero_entries": int(result.nnz),
        "density": float(result.nnz / (result.shape[0] * result.shape[1])),
        "coefficient_nonzero_entries": int(len(nonzero_pairs)),
        "row_embedding": row_embedding,
        "column_embedding": column_embedding,
        "relative_entry_tolerance": float(relative_entry_tolerance),
        "absolute_entry_threshold": threshold,
        "compressed_frobenius_squared": float(np.sum(np.abs(result.data) ** 2)),
    }
    return result, diagnostics


def compressed_purification_factor_cut(
    exponents: np.ndarray,
    factor: np.ndarray,
    *,
    cut: int,
    relative_entry_tolerance: float = 1.0e-12,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build a purification Schmidt matrix from a supplied factor ``B``."""

    return compressed_coefficient_cut(
        exponents,
        factor,
        cut=cut,
        row_embedding="normalized",
        column_embedding="uniform",
        relative_entry_tolerance=relative_entry_tolerance,
    )


def compressed_purification_cut(
    exponents: np.ndarray,
    h_matrix: np.ndarray,
    *,
    cut: int,
    relative_entry_tolerance: float = 1.0e-12,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build the principal-root Schmidt matrix on occupation-count classes."""

    matrix = np.asarray(h_matrix, dtype=np.complex128)
    factor = principal_positive_square_root(matrix)
    result, diagnostics = compressed_purification_factor_cut(
        exponents,
        factor,
        cut=cut,
        relative_entry_tolerance=relative_entry_tolerance,
    )
    diagnostics["principal_square_root_nonzero_entries"] = diagnostics[
        "coefficient_nonzero_entries"
    ]
    diagnostics["factorization_relative_frobenius_error"] = float(
        np.linalg.norm(factor.conj().T @ factor - matrix)
        / np.linalg.norm(matrix)
    )
    return result, diagnostics


def compressed_operator_cut(
    exponents: np.ndarray,
    h_matrix: np.ndarray,
    *,
    cut: int,
    relative_entry_tolerance: float = 1.0e-12,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build the exact local operator-Schmidt matrix representing ``s^dagger Hs``."""

    matrix = np.asarray(h_matrix, dtype=np.complex128)
    hermitian = 0.5 * (matrix + matrix.conj().T)
    result, diagnostics = compressed_coefficient_cut(
        exponents,
        hermitian,
        cut=cut,
        row_embedding="uniform",
        column_embedding="uniform",
        relative_entry_tolerance=relative_entry_tolerance,
    )
    diagnostics["hermiticity_relative_frobenius_error"] = float(
        np.linalg.norm(matrix - matrix.conj().T) / np.linalg.norm(matrix)
    )
    return result, diagnostics


def _rank_for_weight(
    cumulative: np.ndarray,
    target: float,
    *,
    computed_count: int,
) -> int | str:
    reached = np.flatnonzero(cumulative >= target)
    if len(reached):
        return int(reached[0] + 1)
    return f">{computed_count}"


def leading_schmidt_spectrum(
    matrix: sparse.csr_matrix,
    *,
    maximum_singular_values: int,
    exact_row_dimension: int,
) -> dict[str, Any]:
    if maximum_singular_values <= 0 or exact_row_dimension <= 0:
        raise ValueError("spectrum dimensions must be positive")
    total = float(np.sum(np.abs(matrix.data) ** 2))
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("compressed purification has zero or invalid norm")
    smaller_dimension = min(matrix.shape)
    exact = smaller_dimension <= exact_row_dimension
    if exact:
        if matrix.shape[0] <= matrix.shape[1]:
            gram = (matrix @ matrix.conj().T).toarray()
        else:
            gram = (matrix.conj().T @ matrix).toarray()
        gram = 0.5 * (gram + gram.conj().T)
        squared = np.linalg.eigvalsh(gram)
        squared = np.clip(squared, 0.0, None)[::-1]
    else:
        count = min(maximum_singular_values, smaller_dimension - 1)
        try:
            singular_values = svds(
                matrix,
                k=count,
                which="LM",
                return_singular_vectors=False,
                solver="propack",
                tol=1.0e-10,
            )
        except (TypeError, ValueError):
            singular_values = svds(
                matrix,
                k=count,
                which="LM",
                return_singular_vectors=False,
                tol=1.0e-10,
            )
        squared = np.sort(np.abs(singular_values) ** 2)[::-1]
    probabilities = squared / total
    cumulative = np.cumsum(probabilities)
    retained = {
        str(dimension): float(cumulative[min(dimension, len(cumulative)) - 1])
        for dimension in (1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 256)
        if dimension <= len(cumulative)
    }
    return {
        "method": "exact-gram" if exact else "leading-sparse-svd",
        "computed_singular_value_count": int(len(squared)),
        "total_squared_weight": total,
        "captured_squared_weight": float(cumulative[-1]),
        "leading_normalized_squared_singular_values": probabilities.tolist(),
        "leading_relative_singular_values": (
            np.sqrt(squared / squared[0])
        ).tolist(),
        "rank_for_cumulative_squared_weight": {
            str(target): _rank_for_weight(
                cumulative,
                target,
                computed_count=len(cumulative),
            )
            for target in (0.99, 0.999, 0.9999, 0.999999)
        },
        "squared_weight_captured_by_bond_dimension": retained,
    }


def lifted_degree_cuts(
    source_cuts: list[dict[str, Any]],
    *,
    source_degree: int,
    power: int,
) -> list[dict[str, Any]]:
    """Lift source-cut summaries to the exact tensor-product degree hierarchy."""

    by_cut = {int(row["cut"]): row for row in source_cuts}
    rows = []
    target_degree = source_degree * power
    for cut in range(1, target_degree):
        segment_cut = cut % source_degree
        if segment_cut == 0:
            rows.append(
                {
                    "cut_after_site": cut,
                    "source_cut": 0,
                    "exact_seam": True,
                    "rank_for_cumulative_squared_weight": {
                        "0.99": 1,
                        "0.999": 1,
                        "0.9999": 1,
                        "0.999999": 1,
                    },
                    "squared_weight_captured_by_bond_dimension": {
                        str(dimension): 1.0
                        for dimension in (1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 256)
                    },
                }
            )
            continue
        canonical_cut = min(segment_cut, source_degree - segment_cut)
        source = by_cut[canonical_cut]
        rows.append(
            {
                "cut_after_site": cut,
                "source_cut": canonical_cut,
                "exact_seam": False,
                "rank_for_cumulative_squared_weight": source[
                    "spectrum"
                ]["rank_for_cumulative_squared_weight"],
                "squared_weight_captured_by_bond_dimension": source[
                    "spectrum"
                ]["squared_weight_captured_by_bond_dimension"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    source_path = args.full_h.expanduser().resolve()
    payload = np.load(source_path)
    degree = int(payload["degree"])
    exponents = np.asarray(payload["exponents"], dtype=np.int64)
    h_matrix = np.asarray(payload["global_h_matrix"], dtype=np.complex128)
    basis_degree, coordinate_count = _validate_basis(exponents)
    if basis_degree != degree:
        raise SystemExit("artifact degree and exponent degree disagree")
    cuts = (
        tuple(args.cuts)
        if args.cuts
        else tuple(range(1, degree // 2 + 1))
    )
    if any(not 0 < cut <= degree // 2 for cut in cuts):
        raise SystemExit("cuts must lie in 1,...,floor(k/2)")

    cut_rows = []
    for cut in cuts:
        matrix, diagnostics = compressed_purification_cut(
            exponents,
            h_matrix,
            cut=cut,
            relative_entry_tolerance=args.relative_entry_tolerance,
        )
        spectrum = leading_schmidt_spectrum(
            matrix,
            maximum_singular_values=args.maximum_singular_values,
            exact_row_dimension=args.exact_row_dimension,
        )
        cut_rows.append(
            {
                "cut": int(cut),
                "matrix": diagnostics,
                "spectrum": spectrum,
            }
        )
        print(
            f"cut={cut} shape={matrix.shape} nnz={matrix.nnz} "
            f"captured={spectrum['captured_squared_weight']:.9f} "
            f"r999={spectrum['rank_for_cumulative_squared_weight']['0.999']}",
            flush=True,
        )

    power = 30 // degree if 30 % degree == 0 else 1
    target_degree = degree * power
    lifted = (
        lifted_degree_cuts(cut_rows, source_degree=degree, power=power)
        if set(cuts) == set(range(1, degree // 2 + 1))
        else []
    )
    report = {
        "schema": "quintic-full-h-oracle-purification-schmidt-v1",
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "degree": degree,
            "section_count": int(len(exponents)),
            "coordinate_count": coordinate_count,
        },
        "purification": {
            "choice": "principal-positive-square-root",
            "output_embedding": "normalized-symmetric-monomial-state",
            "input_embedding": "uniform-monomial-synthesis",
            "scope": (
                "This is an exact symmetry-natural lift and gives a constructive "
                "upper bound for this purification, not a proof of minimal rank "
                "over all function-equivalent lifts or purification gauges."
            ),
        },
        "configuration": {
            "maximum_singular_values": int(args.maximum_singular_values),
            "exact_row_dimension": int(args.exact_row_dimension),
            "relative_entry_tolerance": float(args.relative_entry_tolerance),
            "cuts": list(cuts),
        },
        "source_degree_cuts": cut_rows,
        "power_lift": {
            "power": power,
            "target_degree": target_degree,
            "exact_identity": f"F_{target_degree}=F_{degree}^{power}",
            "seam_rank": 1,
            "target_degree_cuts": lifted,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
