#!/usr/bin/env python3
"""Audit exact quotient-multiplication lifts of a Fermat full-H oracle.

For a split ``k = a + b``, the quotient multiplication map is

``M_ab: V_a tensor V_b -> V_k``.

We construct exact right inverses ``R`` with ``M_ab R = I`` and lift the
teacher operator to

``H_ab = R H_k R^dagger``.

The reshaped operator-Schmidt spectrum of ``H_ab`` measures the root rank of
a multiplication-tree representation.  Different right inverses are exact
function-equivalent lifts, so their comparison directly audits lift/gauge
dependence rather than changing the target metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from gcicy_metric.fermat_quintic import (
    fermat_quintic_quotient_basis,
    fermat_reduce_exponent,
)
from scripts.audit_quintic_full_h_oracle_purification_schmidt import (
    leading_schmidt_spectrum,
    multinomial_count,
)


LIFT_MODES = ("tensor-minimal", "uniform-exact", "canonical-exact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-h", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=("5+5",),
        help="Degree splits such as 5+5 or 4+6.",
    )
    parser.add_argument(
        "--lift-modes",
        choices=LIFT_MODES,
        nargs="*",
        default=list(LIFT_MODES),
    )
    parser.add_argument("--maximum-singular-values", type=int, default=512)
    parser.add_argument("--exact-row-dimension", type=int, default=1500)
    parser.add_argument("--relative-entry-tolerance", type=float, default=1.0e-12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_split(value: str) -> tuple[int, int]:
    fields = value.split("+")
    if len(fields) != 2:
        raise ValueError(f"invalid degree split: {value}")
    left, right = (int(field) for field in fields)
    if left <= 0 or right <= 0:
        raise ValueError("split degrees must be positive")
    return left, right


def quotient_multiplication_map(
    left_degree: int,
    right_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Return normalized child bases and the exact quotient product map."""

    _, left_exponents, _ = fermat_quintic_quotient_basis(left_degree)
    _, right_exponents, _ = fermat_quintic_quotient_basis(right_degree)
    _, output_exponents, _ = fermat_quintic_quotient_basis(
        left_degree + right_degree
    )
    output_index = {
        tuple(int(value) for value in exponent): index
        for index, exponent in enumerate(output_exponents)
    }
    left_weights = np.asarray(
        [multinomial_count(tuple(map(int, exponent))) for exponent in left_exponents],
        dtype=np.float64,
    )
    right_weights = np.asarray(
        [multinomial_count(tuple(map(int, exponent))) for exponent in right_exponents],
        dtype=np.float64,
    )

    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    right_count = len(right_exponents)
    for left_index, left in enumerate(left_exponents):
        for right_index, right in enumerate(right_exponents):
            exponent = tuple(int(value) for value in left + right)
            scale = math.sqrt(left_weights[left_index] * right_weights[right_index])
            for reduced, coefficient in fermat_reduce_exponent(exponent):
                rows.append(output_index[reduced])
                columns.append(left_index * right_count + right_index)
                data.append(scale * coefficient)
    multiplication = sparse.coo_matrix(
        (np.asarray(data, dtype=np.complex128), (rows, columns)),
        shape=(len(output_exponents), len(left_exponents) * len(right_exponents)),
    ).tocsr()
    multiplication.sum_duplicates()
    return left_exponents, right_exponents, output_exponents, multiplication


def exact_product_right_inverse(
    left_exponents: np.ndarray,
    right_exponents: np.ndarray,
    output_exponents: np.ndarray,
    *,
    mode: str,
) -> sparse.csr_matrix:
    """Construct an exact right inverse using products needing no reduction."""

    if mode not in LIFT_MODES:
        raise ValueError(f"unknown lift mode: {mode}")
    left_lookup = {
        tuple(map(int, exponent)): index
        for index, exponent in enumerate(left_exponents)
    }
    right_lookup = {
        tuple(map(int, exponent)): index
        for index, exponent in enumerate(right_exponents)
    }
    left_weights = np.asarray(
        [multinomial_count(tuple(map(int, exponent))) for exponent in left_exponents],
        dtype=np.float64,
    )
    right_weights = np.asarray(
        [multinomial_count(tuple(map(int, exponent))) for exponent in right_exponents],
        dtype=np.float64,
    )
    left_degree = int(np.sum(left_exponents[0]))
    pair_count = len(left_exponents) * len(right_exponents)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for output_index, output in enumerate(output_exponents):
        candidates: list[tuple[int, int]] = []
        output_tuple = tuple(map(int, output))
        for left_tuple, left_index in left_lookup.items():
            if sum(left_tuple) != left_degree:
                raise RuntimeError("left quotient basis has an inconsistent degree")
            if any(left_value > total for left_value, total in zip(left_tuple, output_tuple)):
                continue
            right_tuple = tuple(
                total - left_value
                for left_value, total in zip(left_tuple, output_tuple)
            )
            right_index = right_lookup.get(right_tuple)
            if right_index is not None:
                candidates.append((left_index, right_index))
        if not candidates:
            raise RuntimeError("standard quotient monomial has no exact product split")
        if mode == "canonical-exact":
            candidates = [min(candidates)]
        output_multinomial = float(multinomial_count(output_tuple))
        for left_index, right_index in candidates:
            pair_scale = math.sqrt(
                left_weights[left_index] * right_weights[right_index]
            )
            if mode == "tensor-minimal":
                coefficient = pair_scale / output_multinomial
            elif mode == "uniform-exact":
                coefficient = 1.0 / (len(candidates) * pair_scale)
            else:
                coefficient = 1.0 / pair_scale
            rows.append(left_index * len(right_exponents) + right_index)
            columns.append(output_index)
            data.append(coefficient)
    result = sparse.coo_matrix(
        (np.asarray(data, dtype=np.complex128), (rows, columns)),
        shape=(pair_count, len(output_exponents)),
    ).tocsr()
    result.sum_duplicates()
    return result


def multiplication_right_inverse_error(
    multiplication: sparse.csr_matrix,
    right_inverse: sparse.csr_matrix,
) -> float:
    difference = multiplication @ right_inverse - sparse.eye(
        multiplication.shape[0],
        dtype=np.complex128,
        format="csr",
    )
    return float(sparse.linalg.norm(difference) / math.sqrt(multiplication.shape[0]))


def lifted_operator_schmidt_matrix(
    h_matrix: np.ndarray,
    right_inverse: sparse.csr_matrix,
    *,
    left_count: int,
    right_count: int,
    relative_entry_tolerance: float,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Reshape ``R H R^dagger`` without materializing the pair-space operator."""

    if not 0 <= relative_entry_tolerance < 1:
        raise ValueError("relative entry tolerance must lie in [0,1)")
    matrix = np.asarray(h_matrix, dtype=np.complex128)
    largest = float(np.max(np.abs(matrix)))
    nonzero_h = np.argwhere(np.abs(matrix) > relative_entry_tolerance * largest)
    inverse = right_inverse.tocsc()
    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    for output_index, input_index in nonzero_h:
        output_start, output_end = (
            inverse.indptr[output_index],
            inverse.indptr[output_index + 1],
        )
        input_start, input_end = (
            inverse.indptr[input_index],
            inverse.indptr[input_index + 1],
        )
        output_pairs = inverse.indices[output_start:output_end]
        output_values = inverse.data[output_start:output_end]
        input_pairs = inverse.indices[input_start:input_end]
        input_values = inverse.data[input_start:input_end]
        h_value = matrix[output_index, input_index]
        for output_pair, output_value in zip(output_pairs, output_values):
            output_left, output_right = divmod(int(output_pair), right_count)
            for input_pair, input_value in zip(input_pairs, input_values):
                input_left, input_right = divmod(int(input_pair), right_count)
                rows.append(output_left * left_count + input_left)
                columns.append(output_right * right_count + input_right)
                data.append(output_value * h_value * np.conj(input_value))
    result = sparse.coo_matrix(
        (np.asarray(data, dtype=np.complex128), (rows, columns)),
        shape=(left_count**2, right_count**2),
    ).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    diagnostics = {
        "shape": [int(value) for value in result.shape],
        "nonzero_entries": int(result.nnz),
        "density": float(result.nnz / (result.shape[0] * result.shape[1])),
        "source_h_entries": int(len(nonzero_h)),
        "compressed_frobenius_squared": float(np.sum(np.abs(result.data) ** 2)),
    }
    return result, diagnostics


def main() -> None:
    args = parse_args()
    source_path = args.full_h.expanduser().resolve()
    payload = np.load(source_path)
    degree = int(payload["degree"])
    teacher_exponents = np.asarray(payload["exponents"], dtype=np.int64)
    h_matrix = np.asarray(payload["global_h_matrix"], dtype=np.complex128)
    h_matrix = 0.5 * (h_matrix + h_matrix.conj().T)
    requested_splits = tuple(parse_split(value) for value in args.splits)
    if any(left + right != degree for left, right in requested_splits):
        raise SystemExit("every multiplication split must sum to the teacher degree")

    split_rows = []
    for left_degree, right_degree in requested_splits:
        (
            left_exponents,
            right_exponents,
            output_exponents,
            multiplication,
        ) = quotient_multiplication_map(left_degree, right_degree)
        if not np.array_equal(output_exponents, teacher_exponents):
            raise RuntimeError("multiplication output basis differs from teacher basis")
        mode_rows = []
        for mode in dict.fromkeys(args.lift_modes):
            right_inverse = exact_product_right_inverse(
                left_exponents,
                right_exponents,
                output_exponents,
                mode=mode,
            )
            right_inverse_error = multiplication_right_inverse_error(
                multiplication,
                right_inverse,
            )
            if right_inverse_error > 2.0e-13:
                raise RuntimeError(
                    f"{mode} right inverse failed: {right_inverse_error:.3e}"
                )
            started = time.perf_counter()
            schmidt, matrix_diagnostics = lifted_operator_schmidt_matrix(
                h_matrix,
                right_inverse,
                left_count=len(left_exponents),
                right_count=len(right_exponents),
                relative_entry_tolerance=args.relative_entry_tolerance,
            )
            build_seconds = time.perf_counter() - started
            started = time.perf_counter()
            spectrum = leading_schmidt_spectrum(
                schmidt,
                maximum_singular_values=args.maximum_singular_values,
                exact_row_dimension=args.exact_row_dimension,
            )
            spectrum_seconds = time.perf_counter() - started
            row = {
                "mode": mode,
                "right_inverse": {
                    "shape": [int(value) for value in right_inverse.shape],
                    "nonzero_entries": int(right_inverse.nnz),
                    "right_inverse_relative_frobenius_error": right_inverse_error,
                },
                "matrix": matrix_diagnostics,
                "spectrum": spectrum,
                "timing": {
                    "matrix_wall_seconds": float(build_seconds),
                    "spectrum_wall_seconds": float(spectrum_seconds),
                },
            }
            mode_rows.append(row)
            print(
                f"split={left_degree}+{right_degree} mode={mode} "
                f"shape={schmidt.shape} nnz={schmidt.nnz} "
                f"r99={spectrum['rank_for_cumulative_squared_weight']['0.99']} "
                f"r999={spectrum['rank_for_cumulative_squared_weight']['0.999']} "
                f"captured={spectrum['captured_squared_weight']:.9f}",
                flush=True,
            )
        split_rows.append(
            {
                "left_degree": left_degree,
                "right_degree": right_degree,
                "left_section_count": int(len(left_exponents)),
                "right_section_count": int(len(right_exponents)),
                "output_section_count": int(len(output_exponents)),
                "multiplication_map": {
                    "shape": [int(value) for value in multiplication.shape],
                    "nonzero_entries": int(multiplication.nnz),
                },
                "lifts": mode_rows,
            }
        )

    report = {
        "schema": "quintic-full-h-oracle-multiplication-tree-v1",
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "degree": degree,
            "section_count": int(len(teacher_exponents)),
        },
        "configuration": {
            "splits": [list(value) for value in requested_splits],
            "lift_modes": list(dict.fromkeys(args.lift_modes)),
            "maximum_singular_values": int(args.maximum_singular_values),
            "exact_row_dimension": int(args.exact_row_dimension),
            "relative_entry_tolerance": float(args.relative_entry_tolerance),
            "child_basis": "sqrt-multinomial-normalized quotient monomials",
        },
        "splits": split_rows,
        "interpretation_scope": (
            "Every registered R is an exact right inverse and therefore gives "
            "the same scalar teacher metric. The spectra quantify exact lift "
            "dependence. They are not gauge-independent lower bounds."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
