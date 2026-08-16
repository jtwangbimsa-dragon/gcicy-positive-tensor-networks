"""A fixed low-symmetry smooth-quintic candidate for solver kill tests."""

from __future__ import annotations

from itertools import permutations, product
from math import comb

import numpy as np


GENERIC_QUINTIC_SCHEMA = "generic-quintic-kill-test-v1"
GENERIC_QUINTIC_SEED = 202607261


# The Fermat terms make the polynomial numerically well scaled. The mixed
# support removes the large permutation and phase symmetry used by the Fermat
# benchmark. Exact smoothness is certified separately with Singular.
GENERIC_QUINTIC_EXPONENTS = np.asarray(
    [
        [5, 0, 0, 0, 0],
        [0, 5, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 0, 5],
        [4, 1, 0, 0, 0],
        [0, 4, 1, 0, 0],
        [0, 0, 4, 1, 0],
        [0, 0, 0, 4, 1],
    ],
    dtype=np.int64,
)

GENERIC_QUINTIC_COEFFICIENTS = np.asarray(
    [
        13,
        17,
        19,
        23,
        29,
        1,
        -2,
        3,
        -5,
    ],
    dtype=np.int64,
)


def quintic_value(points: np.ndarray) -> np.ndarray:
    """Evaluate the fixed homogeneous quintic."""

    values = np.asarray(points, dtype=np.complex128)
    if values.shape[-1] != 5:
        raise ValueError("quintic points must have five homogeneous coordinates")
    monomials = np.prod(
        np.power(values[..., None, :], GENERIC_QUINTIC_EXPONENTS),
        axis=-1,
    )
    return np.einsum(
        "...m,m->...",
        monomials,
        GENERIC_QUINTIC_COEFFICIENTS.astype(np.complex128),
    )


def quintic_gradient(points: np.ndarray) -> np.ndarray:
    """Evaluate the five homogeneous first derivatives."""

    values = np.asarray(points, dtype=np.complex128)
    if values.shape[-1] != 5:
        raise ValueError("quintic points must have five homogeneous coordinates")
    result = np.zeros(values.shape, dtype=np.complex128)
    for coordinate in range(5):
        positive = GENERIC_QUINTIC_EXPONENTS[:, coordinate] > 0
        exponents = GENERIC_QUINTIC_EXPONENTS[positive].copy()
        factors = (
            GENERIC_QUINTIC_COEFFICIENTS[positive]
            * exponents[:, coordinate]
        )
        exponents[:, coordinate] -= 1
        monomials = np.prod(
            np.power(values[..., None, :], exponents),
            axis=-1,
        )
        result[..., coordinate] = np.einsum(
            "...m,m->...",
            monomials,
            factors.astype(np.complex128),
        )
    return result


def support_preserving_coordinate_permutations() -> list[tuple[int, ...]]:
    """Return coordinate permutations preserving the monomial support."""

    support = {tuple(row) for row in GENERIC_QUINTIC_EXPONENTS.tolist()}
    preserving = []
    for permutation in permutations(range(5)):
        transformed = {
            tuple(row[list(permutation)])
            for row in GENERIC_QUINTIC_EXPONENTS
        }
        if transformed == support:
            preserving.append(tuple(int(value) for value in permutation))
    return preserving


def projective_fifth_root_phase_symmetries() -> list[tuple[int, ...]]:
    """Return diagonal Z5 phases modulo the common projective phase."""

    symmetries = []
    for tail in product(range(5), repeat=4):
        charge = np.asarray((0, *tail), dtype=np.int64)
        phases = GENERIC_QUINTIC_EXPONENTS @ charge
        if np.all(np.mod(phases - phases[0], 5) == 0):
            symmetries.append(tuple(int(value) for value in charge))
    return symmetries


def section_count(degree: int) -> int:
    """Dimension of H^0(X,O_X(degree)) for a smooth quintic threefold."""

    if degree < 0:
        return 0
    ambient = comb(degree + 4, 4)
    relation = comb(degree - 1, 4) if degree >= 5 else 0
    return ambient - relation


def experiment_manifest(
    *,
    teacher_degree: int = 2,
    target_degree: int = 20,
) -> dict[str, object]:
    """Return the immutable geometry and scale contract for the kill test."""

    if teacher_degree <= 0 or target_degree <= 0:
        raise ValueError("degrees must be positive")
    if target_degree % teacher_degree:
        raise ValueError("target degree must be a multiple of teacher degree")
    teacher_sections = section_count(teacher_degree)
    target_sections = section_count(target_degree)
    return {
        "schema": GENERIC_QUINTIC_SCHEMA,
        "seed": GENERIC_QUINTIC_SEED,
        "ambient": [4],
        "exponents": GENERIC_QUINTIC_EXPONENTS.tolist(),
        "coefficients": GENERIC_QUINTIC_COEFFICIENTS.tolist(),
        "teacher_degree": teacher_degree,
        "target_degree": target_degree,
        "power": target_degree // teacher_degree,
        "teacher_section_count": teacher_sections,
        "teacher_dense_h_real_parameters": teacher_sections**2,
        "target_section_count": target_sections,
        "target_dense_h_real_parameters": target_sections**2,
        "support_preserving_coordinate_permutations": [
            list(value)
            for value in support_preserving_coordinate_permutations()
        ],
        "projective_fifth_root_phase_symmetries": [
            list(value)
            for value in projective_fifth_root_phase_symmetries()
        ],
    }
