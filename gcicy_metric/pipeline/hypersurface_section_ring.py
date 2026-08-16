"""Section-ring arithmetic for one homogeneous projective hypersurface.

The implementation fixes a pure-power leading monomial ``z_p**d`` and uses
the standard monomials with exponent of ``z_p`` smaller than ``d`` as a basis
of

``C[z_0, ..., z_{n-1}] / (f)``.

Multiplication matrices use the convention

``products = M.T @ output_sections``.

Thus ``M`` has shape ``(dim V_{a+b}, dim V_a * dim V_b)`` and a right inverse
``R`` satisfying ``M @ R = I`` reconstructs section values as

``output_sections = R.T @ products``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable

import numpy as np
from scipy import sparse


Exponent = tuple[int, ...]


def homogeneous_monomial_exponents(total: int, variables: int) -> np.ndarray:
    """Return all non-negative exponent vectors of a fixed total degree."""

    if total < 0 or variables <= 0:
        raise ValueError("composition dimensions must be valid")
    rows: list[Exponent] = []

    def visit(remaining: int, slots: int, prefix: Exponent) -> None:
        if slots == 1:
            rows.append(prefix + (remaining,))
            return
        for value in range(remaining + 1):
            visit(remaining - value, slots - 1, prefix + (value,))

    visit(total, variables, ())
    return np.asarray(rows, dtype=np.int64)


def hypersurface_section_count(
    degree: int,
    *,
    variables: int,
    hypersurface_degree: int,
) -> int:
    """Dimension of the degree-``degree`` component of a principal quotient."""

    if degree < 0:
        return 0
    ambient = comb(degree + variables - 1, variables - 1)
    relation = (
        comb(
            degree - hypersurface_degree + variables - 1,
            variables - 1,
        )
        if degree >= hypersurface_degree
        else 0
    )
    return ambient - relation


def evaluate_monomials(
    points: np.ndarray,
    exponents: np.ndarray,
) -> np.ndarray:
    """Evaluate homogeneous monomials at one point or a batch of points."""

    values = np.asarray(points, dtype=np.complex128)
    powers = np.asarray(exponents, dtype=np.int64)
    if powers.ndim != 2 or values.shape[-1] != powers.shape[1]:
        raise ValueError("point and exponent dimensions do not agree")
    return np.prod(
        np.power(values[..., None, :], powers),
        axis=-1,
    )


@dataclass(frozen=True)
class QuotientMultiplicationMap:
    """One exact multiplication map in standard quotient bases."""

    left_degree: int
    right_degree: int
    left_exponents: np.ndarray
    right_exponents: np.ndarray
    output_exponents: np.ndarray
    matrix: sparse.csr_matrix

    @property
    def left_count(self) -> int:
        return int(len(self.left_exponents))

    @property
    def right_count(self) -> int:
        return int(len(self.right_exponents))

    @property
    def output_count(self) -> int:
        return int(len(self.output_exponents))


class HypersurfaceSectionRing:
    """Numerical exact arithmetic in a principal homogeneous quotient ring."""

    def __init__(
        self,
        exponents: np.ndarray,
        coefficients: np.ndarray,
        *,
        pivot_coordinate: int = 0,
    ) -> None:
        powers = np.asarray(exponents, dtype=np.int64)
        values = np.asarray(coefficients, dtype=np.complex128)
        if powers.ndim != 2 or len(powers) == 0:
            raise ValueError("polynomial exponents must form a non-empty matrix")
        if values.shape != (len(powers),):
            raise ValueError("one polynomial coefficient is required per exponent")
        if np.any(powers < 0):
            raise ValueError("polynomial exponents must be non-negative")
        degrees = np.sum(powers, axis=1)
        if np.any(degrees != degrees[0]) or int(degrees[0]) <= 0:
            raise ValueError("the defining polynomial must be homogeneous")
        if not np.all(np.isfinite(values)) or np.any(values == 0):
            raise ValueError("polynomial coefficients must be finite and non-zero")
        if not 0 <= pivot_coordinate < powers.shape[1]:
            raise ValueError("pivot coordinate is outside the ambient space")

        combined: dict[Exponent, complex] = {}
        for exponent, coefficient in zip(powers, values):
            key = tuple(int(value) for value in exponent)
            combined[key] = combined.get(key, 0.0j) + complex(coefficient)
        combined = {
            exponent: coefficient
            for exponent, coefficient in combined.items()
            if coefficient != 0
        }
        if not combined:
            raise ValueError("the defining polynomial is zero")

        degree = int(degrees[0])
        pivot_exponent = tuple(
            degree if coordinate == pivot_coordinate else 0
            for coordinate in range(powers.shape[1])
        )
        try:
            pivot_coefficient = combined[pivot_exponent]
        except KeyError as error:
            raise ValueError(
                "the chosen pivot coordinate needs a non-zero pure-power term"
            ) from error

        self.variable_count = int(powers.shape[1])
        self.hypersurface_degree = degree
        self.pivot_coordinate = int(pivot_coordinate)
        self.pivot_exponent = pivot_exponent
        self.pivot_coefficient = pivot_coefficient
        self.polynomial_terms = tuple(sorted(combined.items()))
        self._nonpivot_terms = tuple(
            (exponent, -coefficient / pivot_coefficient)
            for exponent, coefficient in self.polynomial_terms
            if exponent != pivot_exponent
        )
        self._reduction_cache: dict[
            Exponent,
            tuple[tuple[Exponent, complex], ...],
        ] = {}
        self._basis_cache: dict[int, np.ndarray] = {}
        self._multiplication_cache: dict[
            tuple[int, int],
            QuotientMultiplicationMap,
        ] = {}

    def ambient_exponents(self, degree: int) -> np.ndarray:
        return homogeneous_monomial_exponents(degree, self.variable_count)

    def quotient_exponents(self, degree: int) -> np.ndarray:
        """Return the standard quotient monomials at one degree."""

        if degree < 0:
            raise ValueError("section degree must be non-negative")
        cached = self._basis_cache.get(degree)
        if cached is None:
            ambient = self.ambient_exponents(degree)
            cached = ambient[
                ambient[:, self.pivot_coordinate] < self.hypersurface_degree
            ].copy()
            cached.setflags(write=False)
            expected = hypersurface_section_count(
                degree,
                variables=self.variable_count,
                hypersurface_degree=self.hypersurface_degree,
            )
            if len(cached) != expected:
                raise RuntimeError("standard quotient basis has the wrong dimension")
            self._basis_cache[degree] = cached
        return cached

    def reduce_exponent(
        self,
        exponent: Iterable[int],
    ) -> tuple[tuple[Exponent, complex], ...]:
        """Reduce one monomial to the standard quotient basis."""

        key = tuple(int(value) for value in exponent)
        if (
            len(key) != self.variable_count
            or any(value < 0 for value in key)
        ):
            raise ValueError("monomial exponent has invalid dimensions")
        cached = self._reduction_cache.get(key)
        if cached is not None:
            return cached
        if key[self.pivot_coordinate] < self.hypersurface_degree:
            result = ((key, 1.0 + 0.0j),)
            self._reduction_cache[key] = result
            return result

        base = list(key)
        base[self.pivot_coordinate] -= self.hypersurface_degree
        coefficients: dict[Exponent, complex] = {}
        for relation_exponent, relation_coefficient in self._nonpivot_terms:
            image = tuple(
                base[coordinate] + relation_exponent[coordinate]
                for coordinate in range(self.variable_count)
            )
            for reduced, coefficient in self.reduce_exponent(image):
                coefficients[reduced] = (
                    coefficients.get(reduced, 0.0j)
                    + relation_coefficient * coefficient
                )
        result = tuple(
            (reduced, coefficient)
            for reduced, coefficient in sorted(coefficients.items())
            if coefficient != 0
        )
        self._reduction_cache[key] = result
        return result

    def reduction_matrix(self, degree: int) -> sparse.csr_matrix:
        """Map ambient monomial coefficients to the quotient basis."""

        ambient = self.ambient_exponents(degree)
        quotient = self.quotient_exponents(degree)
        index = {
            tuple(int(value) for value in exponent): position
            for position, exponent in enumerate(quotient)
        }
        rows: list[int] = []
        columns: list[int] = []
        data: list[complex] = []
        for row, exponent in enumerate(ambient):
            for reduced, coefficient in self.reduce_exponent(exponent):
                rows.append(row)
                columns.append(index[reduced])
                data.append(coefficient)
        result = sparse.coo_matrix(
            (np.asarray(data, dtype=np.complex128), (rows, columns)),
            shape=(len(ambient), len(quotient)),
        ).tocsr()
        result.sum_duplicates()
        result.eliminate_zeros()
        return result

    def multiplication_map(
        self,
        left_degree: int,
        right_degree: int,
    ) -> QuotientMultiplicationMap:
        """Return the exact raw-monomial quotient multiplication map."""

        if left_degree < 0 or right_degree < 0:
            raise ValueError("multiplication degrees must be non-negative")
        key = (int(left_degree), int(right_degree))
        cached = self._multiplication_cache.get(key)
        if cached is not None:
            return cached

        left = self.quotient_exponents(left_degree)
        right = self.quotient_exponents(right_degree)
        output = self.quotient_exponents(left_degree + right_degree)
        output_index = {
            tuple(int(value) for value in exponent): position
            for position, exponent in enumerate(output)
        }
        rows: list[int] = []
        columns: list[int] = []
        data: list[complex] = []
        right_count = len(right)
        for left_index, left_exponent in enumerate(left):
            for right_index, right_exponent in enumerate(right):
                product = tuple(
                    int(value)
                    for value in left_exponent + right_exponent
                )
                for reduced, coefficient in self.reduce_exponent(product):
                    rows.append(output_index[reduced])
                    columns.append(left_index * right_count + right_index)
                    data.append(coefficient)
        matrix = sparse.coo_matrix(
            (np.asarray(data, dtype=np.complex128), (rows, columns)),
            shape=(len(output), len(left) * len(right)),
        ).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        cached = QuotientMultiplicationMap(
            left_degree=left_degree,
            right_degree=right_degree,
            left_exponents=left,
            right_exponents=right,
            output_exponents=output,
            matrix=matrix,
        )
        self._multiplication_cache[key] = cached
        return cached


def multiplication_associativity_error(
    ring: HypersurfaceSectionRing,
    degrees: tuple[int, int, int],
) -> float:
    """Return the relative Frobenius associativity defect for three factors."""

    left_degree, middle_degree, right_degree = degrees
    left_middle = ring.multiplication_map(left_degree, middle_degree)
    middle_right = ring.multiplication_map(middle_degree, right_degree)
    left_total = ring.multiplication_map(
        left_degree + middle_degree,
        right_degree,
    )
    right_total = ring.multiplication_map(
        left_degree,
        middle_degree + right_degree,
    )
    left_path = left_total.matrix @ sparse.kron(
        left_middle.matrix,
        sparse.eye(
            middle_right.right_count,
            dtype=np.complex128,
            format="csr",
        ),
        format="csr",
    )
    right_path = right_total.matrix @ sparse.kron(
        sparse.eye(
            left_middle.left_count,
            dtype=np.complex128,
            format="csr",
        ),
        middle_right.matrix,
        format="csr",
    )
    difference = left_path - right_path
    denominator = max(float(sparse.linalg.norm(left_path)), np.finfo(float).tiny)
    return float(sparse.linalg.norm(difference) / denominator)
