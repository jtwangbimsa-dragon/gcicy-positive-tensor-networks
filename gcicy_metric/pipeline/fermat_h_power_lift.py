"""Exact matrix-level power lifts in the Fermat quintic section ring."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from gcicy_metric.fermat_quintic import (
    fermat_quintic_quotient_basis,
    fermat_reduce_exponent,
)


def fermat_quotient_product_matrix(
    left_degree: int,
    right_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Return ``M`` with ``s_left tensor s_right = M.T s_output``."""

    if left_degree < 0 or right_degree < 0:
        raise ValueError("section degrees must be non-negative")
    _, left_exponents, _ = fermat_quintic_quotient_basis(left_degree)
    _, right_exponents, _ = fermat_quintic_quotient_basis(right_degree)
    _, output_exponents, _ = fermat_quintic_quotient_basis(
        left_degree + right_degree
    )
    output_index = {
        tuple(map(int, exponent)): index
        for index, exponent in enumerate(output_exponents)
    }
    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    right_count = len(right_exponents)
    for left_index, left in enumerate(left_exponents):
        for right_index, right in enumerate(right_exponents):
            exponent = tuple(map(int, left + right))
            for reduced, coefficient in fermat_reduce_exponent(exponent):
                rows.append(output_index[reduced])
                columns.append(left_index * right_count + right_index)
                data.append(coefficient)
    multiplication = sparse.coo_matrix(
        (np.asarray(data, dtype=np.complex128), (rows, columns)),
        shape=(len(output_exponents), len(left_exponents) * len(right_exponents)),
    ).tocsr()
    multiplication.sum_duplicates()
    return left_exponents, right_exponents, output_exponents, multiplication


def fermat_full_h_product(
    left_h: np.ndarray,
    right_h: np.ndarray,
    *,
    left_degree: int,
    right_degree: int,
    normalize_trace: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact quotient-basis H for ``F_left * F_right``."""

    left_exponents, right_exponents, output_exponents, multiplication = (
        fermat_quotient_product_matrix(left_degree, right_degree)
    )
    left = np.asarray(left_h, dtype=np.complex128)
    right = np.asarray(right_h, dtype=np.complex128)
    if left.shape != (len(left_exponents), len(left_exponents)):
        raise ValueError("left H dimension does not match its quotient basis")
    if right.shape != (len(right_exponents), len(right_exponents)):
        raise ValueError("right H dimension does not match its quotient basis")
    left = 0.5 * (left + left.conj().T)
    right = 0.5 * (right + right.conj().T)
    product = sparse.kron(
        sparse.csr_matrix(left),
        sparse.csr_matrix(right),
        format="csr",
    )
    output = multiplication.conjugate() @ product @ multiplication.T
    output = np.asarray(output.toarray(), dtype=np.complex128)
    output = 0.5 * (output + output.conj().T)
    if normalize_trace:
        output *= len(output) / np.trace(output).real
    if np.linalg.eigvalsh(output)[0] <= 0:
        raise FloatingPointError("exact full-H product is not positive definite")
    return output_exponents, output


def fermat_full_h_power_lift(
    source_h: np.ndarray,
    *,
    source_degree: int,
    power: int,
    normalize_trace: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact quotient-basis H representing ``F_source**power``."""

    if source_degree <= 0 or power <= 0:
        raise ValueError("source degree and power must be positive")
    _, source_exponents, _ = fermat_quintic_quotient_basis(source_degree)
    source = np.asarray(source_h, dtype=np.complex128)
    if source.shape != (len(source_exponents), len(source_exponents)):
        raise ValueError("source H dimension does not match its quotient basis")
    if power == 1:
        result = 0.5 * (source + source.conj().T)
        if normalize_trace:
            result = result * (len(result) / np.trace(result).real)
        return source_exponents, result

    current_degree = source_degree
    current_h = source
    current_exponents = source_exponents
    for _ in range(1, power):
        current_exponents, current_h = fermat_full_h_product(
            current_h,
            source,
            left_degree=current_degree,
            right_degree=source_degree,
            normalize_trace=normalize_trace,
        )
        current_degree += source_degree
    return current_exponents, current_h

