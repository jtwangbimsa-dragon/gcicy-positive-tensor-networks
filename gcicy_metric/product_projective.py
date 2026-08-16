"""Numerical utilities for arbitrary products of projective spaces."""

from __future__ import annotations

from itertools import product
from math import factorial
from typing import Sequence

import numpy as np

from .simple_patch import fubini_study_metric


Array = np.ndarray


def factor_sizes(dimensions: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in dimensions)
    if not values or min(values) < 1:
        raise ValueError("projective dimensions must be positive")
    return tuple(value + 1 for value in values)


def factor_offsets(dimensions: Sequence[int]) -> tuple[int, ...]:
    sizes = factor_sizes(dimensions)
    offsets = []
    cursor = 0
    for size in sizes:
        offsets.append(cursor)
        cursor += size
    return tuple(offsets)


def _compositions(total: int, length: int) -> list[tuple[int, ...]]:
    if total < 0 or length <= 0:
        raise ValueError("total must be non-negative and length must be positive")
    if length == 1:
        return [(total,)]
    output = []
    for first in range(total + 1):
        for rest in _compositions(total - first, length - 1):
            output.append((first, *rest))
    return output


def product_monomial_exponents(
    degree: Sequence[int],
    dimensions: Sequence[int],
) -> Array:
    degrees = tuple(int(value) for value in degree)
    sizes = factor_sizes(dimensions)
    if len(degrees) != len(sizes) or min(degrees) < 0:
        raise ValueError("degree must be non-negative and match the ambient factors")
    blocks = [
        _compositions(value, size)
        for value, size in zip(degrees, sizes, strict=True)
    ]
    rows = [
        tuple(entry for block in choice for entry in block)
        for choice in product(*blocks)
    ]
    return np.asarray(rows, dtype=np.int64)


def normalize_product_coordinates(
    coordinates: Sequence[Array],
    chart: Sequence[int],
    dimensions: Sequence[int],
) -> tuple[Array, ...]:
    sizes = factor_sizes(dimensions)
    selected_chart = tuple(int(value) for value in chart)
    if len(coordinates) != len(sizes) or len(selected_chart) != len(sizes):
        raise ValueError("coordinates and chart must match the ambient factors")
    output = []
    for values, selected, size in zip(
        coordinates,
        selected_chart,
        sizes,
        strict=True,
    ):
        array = np.asarray(values, dtype=np.complex128).reshape(-1)
        if array.shape != (size,) or not 0 <= selected < size:
            raise ValueError("coordinate or chart shape does not match the ambient product")
        denominator = array[selected]
        if abs(denominator) < 1e-14:
            raise FloatingPointError("requested projective chart is not available")
        output.append(array / denominator)
    return tuple(output)


def affine_coordinates_from_product(
    coordinates: Sequence[Array],
    chart: Sequence[int],
    dimensions: Sequence[int],
) -> Array:
    normalized = normalize_product_coordinates(coordinates, chart, dimensions)
    return np.concatenate(
        [
            np.delete(values, selected)
            for values, selected in zip(normalized, chart, strict=True)
        ]
    )


def homogeneous_product_from_affine(
    chart_coordinates: Array,
    chart: Sequence[int],
    dimensions: Sequence[int],
) -> tuple[Array, ...]:
    sizes = factor_sizes(dimensions)
    selected_chart = tuple(int(value) for value in chart)
    affine_dimension = sum(int(value) for value in dimensions)
    affine = np.asarray(chart_coordinates, dtype=np.complex128).reshape(-1)
    if affine.shape != (affine_dimension,) or len(selected_chart) != len(sizes):
        raise ValueError("unexpected affine-coordinate or chart shape")
    output = []
    cursor = 0
    for size, selected in zip(sizes, selected_chart, strict=True):
        values = np.empty(size, dtype=np.complex128)
        values[selected] = 1.0
        mask = np.arange(size) != selected
        values[mask] = affine[cursor : cursor + size - 1]
        cursor += size - 1
        output.append(values)
    return tuple(output)


def choose_product_chart(coordinates: Sequence[Array]) -> tuple[int, ...]:
    return tuple(int(np.argmax(np.abs(values))) for values in coordinates)


def active_homogeneous_indices(
    chart: Sequence[int],
    dimensions: Sequence[int],
) -> list[int]:
    offsets = factor_offsets(dimensions)
    selected = {
        offset + int(chart_index)
        for offset, chart_index in zip(offsets, chart, strict=True)
    }
    return [
        index
        for index in range(sum(factor_sizes(dimensions)))
        if index not in selected
    ]


def affine_monomial_values_and_jacobian(
    chart_coordinates: Array,
    chart: Sequence[int],
    exponents: Array,
    dimensions: Sequence[int],
) -> tuple[Array, Array]:
    coordinates = homogeneous_product_from_affine(
        chart_coordinates,
        chart,
        dimensions,
    )
    homogeneous = np.concatenate(coordinates)
    powers = np.asarray(exponents, dtype=np.int64)
    affine_dimension = sum(int(value) for value in dimensions)
    if powers.ndim != 2 or powers.shape[1] != len(homogeneous):
        raise ValueError("unexpected exponent shape")
    active = active_homogeneous_indices(chart, dimensions)
    values = np.prod(homogeneous[None, :] ** powers, axis=1)
    jacobian = np.empty((len(powers), affine_dimension), dtype=np.complex128)
    for column, homogeneous_index in enumerate(active):
        exponent = powers[:, homogeneous_index]
        reduced = powers.copy()
        reduced[:, homogeneous_index] = np.maximum(
            reduced[:, homogeneous_index] - 1,
            0,
        )
        jacobian[:, column] = exponent * np.prod(
            homogeneous[None, :] ** reduced,
            axis=1,
        )
    return values, jacobian


def evaluate_product_sections_in_chart(
    coordinates: Sequence[Array],
    chart: Sequence[int],
    exponents: Array,
    dimensions: Sequence[int],
) -> Array:
    normalized = normalize_product_coordinates(coordinates, chart, dimensions)
    homogeneous = np.concatenate(normalized)
    powers = np.asarray(exponents, dtype=np.int64)
    if powers.ndim != 2 or powers.shape[1] != len(homogeneous):
        raise ValueError("unexpected exponent shape")
    return np.prod(homogeneous[None, :] ** powers, axis=1)


def product_fubini_study_metric(
    chart_coordinates: Array,
    dimensions: Sequence[int],
    *,
    weights: Sequence[float] | None = None,
) -> Array:
    dims = tuple(int(value) for value in dimensions)
    selected_weights = (
        tuple(1.0 for _ in dims)
        if weights is None
        else tuple(float(value) for value in weights)
    )
    if len(selected_weights) != len(dims):
        raise ValueError("weights must match the ambient factors")
    coordinates = np.asarray(chart_coordinates, dtype=np.complex128).reshape(-1)
    if coordinates.shape != (sum(dims),):
        raise ValueError("unexpected affine-coordinate shape")
    metric = np.zeros((sum(dims), sum(dims)), dtype=np.complex128)
    cursor = 0
    for dimension, weight in zip(dims, selected_weights, strict=True):
        block = slice(cursor, cursor + dimension)
        metric[block, block] = weight * fubini_study_metric(coordinates[block])
        cursor += dimension
    return metric


def product_multinomial_section_weights(
    exponents: Array,
    dimensions: Sequence[int],
) -> Array:
    powers = np.asarray(exponents, dtype=np.int64)
    sizes = factor_sizes(dimensions)
    offsets = factor_offsets(dimensions)
    if powers.ndim != 2 or powers.shape[1] != sum(sizes):
        raise ValueError("unexpected exponent shape")
    weights = np.ones(len(powers), dtype=float)
    for offset, size in zip(offsets, sizes, strict=True):
        block = powers[:, offset : offset + size]
        for row, degree in enumerate(np.sum(block, axis=1)):
            numerator = factorial(int(degree))
            denominator = int(
                np.prod([factorial(int(value)) for value in block[row]])
            )
            weights[row] *= numerator / denominator
    return weights
