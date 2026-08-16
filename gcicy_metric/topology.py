"""Exact intersection data for the target generalized CICY configuration."""

from __future__ import annotations

from fractions import Fraction
from itertools import product


AMBIENT_DIMENSIONS = (1, 1, 5)
CONFIGURATION_COLUMNS = (
    (1, 1, 3),
    (1, 1, 1),
    (-1, 1, 1),
    (1, -1, 1),
)


def _configuration_class() -> dict[tuple[int, int, int], int]:
    polynomial = {(0, 0, 0): 1}
    for column in CONFIGURATION_COLUMNS:
        updated: dict[tuple[int, int, int], int] = {}
        for exponent, coefficient in polynomial.items():
            for factor, degree in enumerate(column):
                target = list(exponent)
                target[factor] += 1
                target_tuple = tuple(target)
                updated[target_tuple] = updated.get(target_tuple, 0) + coefficient * degree
        polynomial = updated
    return polynomial


CONFIGURATION_CLASS = _configuration_class()


def triple_intersection(first: int, second: int, third: int) -> int:
    """Return the exact triple intersection of ambient hyperplane classes on X."""

    insertions = [0, 0, 0]
    for factor in (first, second, third):
        if factor not in (0, 1, 2):
            raise ValueError("factor indices must be 0, 1, or 2")
        insertions[factor] += 1
    required = tuple(dimension - insertion for dimension, insertion in zip(AMBIENT_DIMENSIONS, insertions))
    if min(required) < 0:
        return 0
    return int(CONFIGURATION_CLASS.get(required, 0))


TRIPLE_INTERSECTIONS = {
    (first, second, third): triple_intersection(first, second, third)
    for first in range(3)
    for second in range(first, 3)
    for third in range(second, 3)
}


def kahler_volume(kahler: tuple[int, int, int] = (1, 1, 1)) -> Fraction:
    """Return integral J^3/3! in the selected ambient normalization."""

    total = sum(
        kahler[first] * kahler[second] * kahler[third] * triple_intersection(first, second, third)
        for first, second, third in product(range(3), repeat=3)
    )
    return Fraction(total, 6)


def line_bundle_slope(
    line_bundle: tuple[int, int, int],
    kahler: tuple[int, int, int] = (1, 1, 1),
) -> Fraction:
    """Return integral c1(L) J^2/2! for a line bundle degree vector."""

    total = sum(
        line_bundle[first] * kahler[second] * kahler[third] * triple_intersection(first, second, third)
        for first, second, third in product(range(3), repeat=3)
    )
    return Fraction(total, 2)


PROPOSAL_INTERSECTION = triple_intersection(0, 1, 2)


def normalized_volume_ratio(kahler: tuple[int, int, int] = (1, 1, 1)) -> Fraction:
    """Return (integral J^3/3!)/(integral Jx Jy Jz)."""

    return kahler_volume(kahler) / PROPOSAL_INTERSECTION


def normalized_slope_ratio(
    line_bundle: tuple[int, int, int],
    kahler: tuple[int, int, int] = (1, 1, 1),
) -> Fraction:
    """Return (integral c1(L) J^2/2!)/(integral Jx Jy Jz)."""

    return line_bundle_slope(line_bundle, kahler) / PROPOSAL_INTERSECTION
