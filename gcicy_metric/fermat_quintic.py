"""Exact section-ring utilities for the Fermat quintic."""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable

import numpy as np


FERMAT_COORDINATE_COUNT = 5
FERMAT_DEGREE = 5


def exponent_compositions(total: int, variables: int) -> np.ndarray:
    if total < 0 or variables <= 0:
        raise ValueError("composition dimensions must be valid")
    rows: list[tuple[int, ...]] = []

    def visit(remaining: int, slots: int, prefix: tuple[int, ...]) -> None:
        if slots == 1:
            rows.append(prefix + (remaining,))
            return
        for value in range(remaining + 1):
            visit(remaining - value, slots - 1, prefix + (value,))

    visit(total, variables, ())
    return np.asarray(rows, dtype=np.int64)


@functools.lru_cache(maxsize=None)
def fermat_reduce_exponent(
    exponent: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Reduce one monomial modulo ``z0^5 + ... + z4^5`` exactly."""

    if len(exponent) != FERMAT_COORDINATE_COUNT or any(value < 0 for value in exponent):
        raise ValueError("a Fermat exponent must contain five non-negative entries")
    if exponent[0] < FERMAT_DEGREE:
        return ((exponent, 1),)
    base = list(exponent)
    base[0] -= FERMAT_DEGREE
    coefficients: dict[tuple[int, ...], int] = {}
    for coordinate in range(1, FERMAT_COORDINATE_COUNT):
        image = base.copy()
        image[coordinate] += FERMAT_DEGREE
        for reduced, coefficient in fermat_reduce_exponent(tuple(image)):
            coefficients[reduced] = coefficients.get(reduced, 0) - coefficient
    return tuple(
        (reduced, coefficient)
        for reduced, coefficient in sorted(coefficients.items())
        if coefficient
    )


def fermat_quintic_quotient_basis(
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ambient monomials, standard quotient monomials, and their lift."""

    if degree < 0:
        raise ValueError("the Fermat quotient degree must be non-negative")
    ambient = exponent_compositions(degree, FERMAT_COORDINATE_COUNT)
    quotient = ambient[ambient[:, 0] < FERMAT_DEGREE]
    index = {tuple(exponent): position for position, exponent in enumerate(quotient)}
    lift = np.zeros((len(ambient), len(quotient)), dtype=np.complex128)
    for row, exponent in enumerate(ambient):
        for reduced, coefficient in fermat_reduce_exponent(tuple(exponent)):
            lift[row, index[reduced]] = coefficient
    return ambient, quotient, lift


def fubini_study_h(exponents: np.ndarray) -> np.ndarray:
    values = np.asarray(exponents, dtype=np.int64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("section exponents must form a non-empty matrix")
    degrees = np.sum(values, axis=1)
    if np.any(degrees != degrees[0]):
        raise ValueError("all sections must have the same degree")
    degree = int(degrees[0])
    diagonal = []
    for exponent in values:
        coefficient = math.factorial(degree)
        for value in exponent:
            coefficient /= math.factorial(int(value))
        diagonal.append(coefficient)
    matrix = np.diag(np.asarray(diagonal, dtype=np.float64)).astype(np.complex128)
    return matrix * (len(matrix) / np.trace(matrix).real)


def fermat_quotient_fubini_study_h(
    ambient_exponents: np.ndarray, quotient_lift: np.ndarray
) -> np.ndarray:
    """Induce the ambient tensor-power FS form on the quotient basis."""

    ambient_h = fubini_study_h(ambient_exponents)
    quotient_h = quotient_lift.conjugate().T @ ambient_h @ quotient_lift
    quotient_h = 0.5 * (quotient_h + quotient_h.conjugate().T)
    return quotient_h * (len(quotient_h) / np.trace(quotient_h).real)


def fermat_phase_sector_indices(
    exponents: np.ndarray,
) -> dict[tuple[int, ...], np.ndarray]:
    """Group quotient sections by their exact diagonal phase character."""

    values = np.asarray(exponents, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != FERMAT_COORDINATE_COUNT:
        raise ValueError("Fermat section exponents must have shape (N, 5)")
    sectors: dict[tuple[int, ...], list[int]] = {}
    for index, exponent in enumerate(values):
        charge = tuple(int(value % FERMAT_DEGREE) for value in exponent)
        sectors.setdefault(charge, []).append(index)
    return {
        charge: np.asarray(indices, dtype=np.int64)
        for charge, indices in sorted(sectors.items())
    }


def fermat_permutation_sector_block(
    exponents: np.ndarray,
    source_indices: np.ndarray,
    permutation: Iterable[int],
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Represent one coordinate permutation from a fixed phase sector."""

    values = np.asarray(exponents, dtype=np.int64)
    source = np.asarray(source_indices, dtype=np.int64)
    order = np.asarray(tuple(permutation), dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != FERMAT_COORDINATE_COUNT:
        raise ValueError("Fermat section exponents must have shape (N, 5)")
    if sorted(order.tolist()) != list(range(FERMAT_COORDINATE_COUNT)):
        raise ValueError("coordinate permutation is invalid")
    sectors = fermat_phase_sector_indices(values)
    lookup = {tuple(exponent): index for index, exponent in enumerate(values)}
    source_charge = tuple(int(value % FERMAT_DEGREE) for value in values[source[0]])
    if any(
        tuple(int(value % FERMAT_DEGREE) for value in values[index])
        != source_charge
        for index in source
    ):
        raise ValueError("source indices must belong to one phase sector")
    target_charge = tuple(source_charge[index] for index in order)
    target = sectors[target_charge]
    target_lookup = {int(index): position for position, index in enumerate(target)}
    block = np.zeros((len(source), len(target)), dtype=np.complex128)
    for row, source_index in enumerate(source):
        transformed = tuple(int(value) for value in values[source_index, order])
        for reduced, coefficient in fermat_reduce_exponent(transformed):
            global_column = lookup[reduced]
            try:
                column = target_lookup[global_column]
            except KeyError as error:
                raise RuntimeError("permutation escaped its predicted phase sector") from error
            block[row, column] += coefficient
    if block.shape[0] != block.shape[1] or np.linalg.matrix_rank(block) != len(block):
        raise RuntimeError("coordinate permutation did not induce an invertible sector map")
    return target_charge, target, block


def fermat_permutation_representation(
    exponents: np.ndarray,
    permutation: Iterable[int],
) -> np.ndarray:
    """Return ``R`` such that ``s(z[p^{-1}]) = R s(z)`` in the quotient basis."""

    values = np.asarray(exponents, dtype=np.int64)
    sectors = fermat_phase_sector_indices(values)
    representation = np.zeros((len(values), len(values)), dtype=np.complex128)
    for indices in sectors.values():
        _, target, block = fermat_permutation_sector_block(
            values, indices, permutation
        )
        representation[np.ix_(indices, target)] = block
    if np.linalg.matrix_rank(representation) != len(values):
        raise RuntimeError("coordinate permutation representation is singular")
    return representation
