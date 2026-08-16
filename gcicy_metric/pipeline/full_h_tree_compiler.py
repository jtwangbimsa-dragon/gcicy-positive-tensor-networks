"""Compile dense algebraic metrics into multiplication-tree coordinates.

This module contains the coordinate-sensitive part of the compiler:

1. whiten the section spaces using their Hermitian covariance matrices;
2. compute the minimum-norm right inverse of a quotient multiplication map;
3. lift a positive Hermitian form or one of its purification factors;
4. audit how much operator-Schmidt data depend on the chosen exact lift.

For the multiplication convention used by
``hypersurface_section_ring.QuotientMultiplicationMap``, section values obey

``p = M.T @ s``.

Consequently, a right inverse ``M @ R = I`` gives ``s = R.T @ p``.  The
transpose here is algebraic, not a Hermitian transpose.  This distinction is
essential when the defining polynomial has complex coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .hypersurface_section_ring import QuotientMultiplicationMap


def section_covariance(
    section_values: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``E[s s^dagger]`` for row-batched section values."""

    values = np.asarray(section_values, dtype=np.complex128)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("section values must have shape (points, sections)")
    if weights is None:
        normalized = np.full(len(values), 1.0 / len(values), dtype=np.float64)
    else:
        normalized = np.asarray(weights, dtype=np.float64)
        if normalized.shape != (len(values),):
            raise ValueError("one covariance weight is required per point")
        if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
            raise ValueError("covariance weights must be finite and non-negative")
        total = float(np.sum(normalized))
        if total <= 0:
            raise ValueError("covariance weights must have positive total mass")
        normalized = normalized / total
    covariance = np.einsum(
        "bi,bj,b->ij",
        values,
        values.conj(),
        normalized,
        optimize=True,
    )
    return 0.5 * (covariance + covariance.conj().T)


@dataclass(frozen=True)
class SectionWhitening:
    """An invertible change from raw to covariance-orthonormal sections."""

    gram: np.ndarray
    whitening: np.ndarray
    unwhitening: np.ndarray
    raw_eigenvalues: np.ndarray
    used_eigenvalues: np.ndarray

    @property
    def condition_number(self) -> float:
        return float(np.max(self.used_eigenvalues) / np.min(self.used_eigenvalues))

    @property
    def clipping_count(self) -> int:
        return int(np.count_nonzero(self.raw_eigenvalues != self.used_eigenvalues))

    def identity_error(self) -> float:
        identity = self.whitening @ self.gram @ self.whitening.conj().T
        return float(
            np.linalg.norm(identity - np.eye(len(identity)))
            / np.sqrt(len(identity))
        )


def hermitian_whitening(
    gram: np.ndarray,
    *,
    relative_eigenvalue_floor: float = 1.0e-12,
) -> SectionWhitening:
    """Build a stable Hermitian square-root whitening transform."""

    matrix = np.asarray(gram, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) == 0:
        raise ValueError("section Gram matrix must be non-empty and square")
    if not 0 <= relative_eigenvalue_floor < 1:
        raise ValueError("relative eigenvalue floor must lie in [0, 1)")
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    largest = float(np.max(eigenvalues))
    if not np.isfinite(largest) or largest <= 0:
        raise ValueError("section Gram matrix is not positive definite")
    floor = max(
        largest * relative_eigenvalue_floor,
        np.finfo(np.float64).tiny,
    )
    if float(np.min(eigenvalues)) <= 0 and relative_eigenvalue_floor == 0:
        raise ValueError("section Gram matrix is singular")
    used = np.maximum(eigenvalues, floor)
    whitening = (eigenvectors * np.power(used, -0.5)) @ eigenvectors.conj().T
    unwhitening = (eigenvectors * np.sqrt(used)) @ eigenvectors.conj().T
    return SectionWhitening(
        gram=matrix,
        whitening=whitening,
        unwhitening=unwhitening,
        raw_eigenvalues=eigenvalues,
        used_eigenvalues=used,
    )


@dataclass(frozen=True)
class MinimumNormMultiplicationLift:
    """A weighted minimum-norm exact right inverse of multiplication."""

    multiplication: QuotientMultiplicationMap
    left_whitening: SectionWhitening
    right_whitening: SectionWhitening
    output_whitening: SectionWhitening
    whitened_multiplication: np.ndarray
    whitened_right_inverse: np.ndarray
    raw_right_inverse: np.ndarray
    singular_values: np.ndarray
    numerical_rank: int

    def right_inverse_error(self) -> float:
        multiplication = self.multiplication.matrix.toarray()
        difference = (
            multiplication @ self.raw_right_inverse
            - np.eye(multiplication.shape[0])
        )
        return float(np.linalg.norm(difference) / np.sqrt(multiplication.shape[0]))

    def whitened_right_inverse_error(self) -> float:
        difference = (
            self.whitened_multiplication @ self.whitened_right_inverse
            - np.eye(self.whitened_multiplication.shape[0])
        )
        return float(
            np.linalg.norm(difference)
            / np.sqrt(self.whitened_multiplication.shape[0])
        )


def minimum_norm_multiplication_lift(
    multiplication: QuotientMultiplicationMap,
    *,
    left_gram: np.ndarray,
    right_gram: np.ndarray,
    output_gram: np.ndarray,
    relative_eigenvalue_floor: float = 1.0e-12,
    relative_singular_value_tolerance: float = 1.0e-12,
) -> MinimumNormMultiplicationLift:
    """Compute the covariance-weighted Moore-Penrose multiplication lift."""

    if not 0 <= relative_singular_value_tolerance < 1:
        raise ValueError("singular-value tolerance must lie in [0, 1)")
    left = hermitian_whitening(
        left_gram,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
    )
    right = hermitian_whitening(
        right_gram,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
    )
    output = hermitian_whitening(
        output_gram,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
    )
    if len(left.gram) != multiplication.left_count:
        raise ValueError("left Gram dimension disagrees with multiplication")
    if len(right.gram) != multiplication.right_count:
        raise ValueError("right Gram dimension disagrees with multiplication")
    if len(output.gram) != multiplication.output_count:
        raise ValueError("output Gram dimension disagrees with multiplication")

    pair_whitening = np.kron(left.whitening, right.whitening)
    raw = multiplication.matrix.toarray()
    whitened = (
        output.unwhitening.T
        @ raw
        @ pair_whitening.T
    )
    left_vectors, singular_values, right_vectors_h = np.linalg.svd(
        whitened,
        full_matrices=False,
    )
    threshold = (
        relative_singular_value_tolerance * float(singular_values[0])
        if len(singular_values)
        else np.inf
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank != whitened.shape[0]:
        raise np.linalg.LinAlgError(
            "quotient multiplication is not numerically surjective after whitening"
        )
    whitened_right_inverse = (
        right_vectors_h.conj().T[:, :rank]
        * np.reciprocal(singular_values[:rank])
    ) @ left_vectors.conj().T[:rank]
    raw_right_inverse = (
        pair_whitening.T
        @ whitened_right_inverse
        @ output.unwhitening.T
    )
    result = MinimumNormMultiplicationLift(
        multiplication=multiplication,
        left_whitening=left,
        right_whitening=right,
        output_whitening=output,
        whitened_multiplication=whitened,
        whitened_right_inverse=whitened_right_inverse,
        raw_right_inverse=raw_right_inverse,
        singular_values=singular_values,
        numerical_rank=rank,
    )
    if result.right_inverse_error() > 2.0e-9:
        raise FloatingPointError("weighted multiplication right inverse is inaccurate")
    return result


def exact_monomial_right_inverse(
    multiplication: QuotientMultiplicationMap,
    *,
    mode: str = "canonical",
) -> np.ndarray:
    """Construct a sparse-in-spirit exact lift from unreduced monomial splits."""

    if mode not in {"canonical", "uniform"}:
        raise ValueError("exact monomial lift mode must be canonical or uniform")
    left = multiplication.left_exponents
    right = multiplication.right_exponents
    output = multiplication.output_exponents
    right_lookup = {
        tuple(int(value) for value in exponent): index
        for index, exponent in enumerate(right)
    }
    result = np.zeros(
        (multiplication.left_count * multiplication.right_count, len(output)),
        dtype=np.complex128,
    )
    for output_index, output_exponent in enumerate(output):
        candidates: list[int] = []
        for left_index, left_exponent in enumerate(left):
            remainder = output_exponent - left_exponent
            if np.any(remainder < 0):
                continue
            right_index = right_lookup.get(
                tuple(int(value) for value in remainder)
            )
            if right_index is not None:
                candidates.append(
                    left_index * multiplication.right_count + right_index
                )
        if not candidates:
            raise RuntimeError("quotient monomial has no unreduced product split")
        if mode == "canonical":
            candidates = [min(candidates)]
        result[candidates, output_index] = 1.0 / len(candidates)
    raw = multiplication.matrix.toarray()
    error = np.linalg.norm(raw @ result - np.eye(len(output))) / np.sqrt(len(output))
    if error > 2.0e-13:
        raise RuntimeError("exact monomial lift is not a right inverse")
    return result


def lift_purification_factor(
    factor: np.ndarray,
    right_inverse: np.ndarray,
) -> np.ndarray:
    """Lift ``F = ||B s||^2`` to product coordinates."""

    values = np.asarray(factor, dtype=np.complex128)
    inverse = np.asarray(right_inverse, dtype=np.complex128)
    if values.ndim != 2 or inverse.ndim != 2:
        raise ValueError("purification factor and right inverse must be matrices")
    if values.shape[1] != inverse.shape[1]:
        raise ValueError("purification and right-inverse dimensions disagree")
    return values @ inverse.T


@dataclass(frozen=True)
class PurificationSchmidtChannels:
    """Teacher-derived local channels across one multiplication cut."""

    orientation: str
    singular_values: np.ndarray
    numerical_rank: int
    retained_rank: int
    retained_squared_weight: float
    selected_singular_indices: np.ndarray
    leaf_tensor: np.ndarray
    leaf_combiner: np.ndarray
    reconstructed_factor: np.ndarray
    lifted_factor_relative_error: float
    factor_relative_error: float
    hermitian_relative_error: float


def purification_schmidt_channels(
    factor: np.ndarray,
    right_inverse: np.ndarray,
    multiplication_matrix: np.ndarray,
    *,
    left_count: int,
    right_count: int,
    section_gram: np.ndarray | None = None,
    squared_weight_retention: float = 1.0,
    maximum_rank: int | None = None,
    orientation: str = "auto",
    relative_singular_value_tolerance: float = 1.0e-12,
    relative_component_tolerance: float = 1.0e-13,
) -> PurificationSchmidtChannels:
    """Compile a purification factor into normalized Schmidt leaf channels.

    The factor is first lifted from the output section space to the product
    section space.  Its left or right Schmidt components are then pushed back
    through the multiplication map separately.  Their sum reconstructs the
    original factor, while each component becomes one trainable tree channel.

    Channel tensors are normalized in the section covariance metric.  The
    compensating norms are returned as ``leaf_combiner`` coefficients, fixing
    the otherwise arbitrary bilinear gauge at round zero.
    """

    values = np.asarray(factor, dtype=np.complex128)
    inverse = np.asarray(right_inverse, dtype=np.complex128)
    multiplication = np.asarray(multiplication_matrix, dtype=np.complex128)
    if values.ndim != 2 or inverse.ndim != 2 or multiplication.ndim != 2:
        raise ValueError("factor, right inverse, and multiplication must be matrices")
    if left_count <= 0 or right_count <= 0:
        raise ValueError("multiplication cut dimensions must be positive")
    pair_count = int(left_count) * int(right_count)
    output_count = values.shape[1]
    if (
        inverse.shape != (pair_count, output_count)
        or multiplication.shape != (output_count, pair_count)
    ):
        raise ValueError("multiplication cut dimensions are inconsistent")
    if not 0 < squared_weight_retention <= 1:
        raise ValueError("squared-weight retention must lie in (0, 1]")
    if maximum_rank is not None and maximum_rank <= 0:
        raise ValueError("maximum rank must be positive")
    if orientation not in {"auto", "left", "right"}:
        raise ValueError("Schmidt orientation must be auto, left, or right")
    if not 0 <= relative_singular_value_tolerance < 1:
        raise ValueError("relative singular-value tolerance must lie in [0, 1)")
    if not 0 <= relative_component_tolerance < 1:
        raise ValueError("relative component tolerance must lie in [0, 1)")

    gram = (
        np.eye(output_count, dtype=np.complex128)
        if section_gram is None
        else np.asarray(section_gram, dtype=np.complex128)
    )
    if gram.shape != (output_count, output_count):
        raise ValueError("section Gram matrix is not aligned with the factor")
    gram = 0.5 * (gram + gram.conj().T)
    if float(np.min(np.linalg.eigvalsh(gram))) <= 0:
        raise ValueError("section Gram matrix must be positive definite")

    lifted = lift_purification_factor(values, inverse)
    tensor = lifted.reshape(values.shape[0], left_count, right_count)

    def decomposition(mode: str):
        if mode == "left":
            matrix = tensor.transpose(1, 0, 2).reshape(
                left_count,
                values.shape[0] * right_count,
            )
        else:
            matrix = tensor.transpose(2, 0, 1).reshape(
                right_count,
                values.shape[0] * left_count,
            )
        left_vectors, singular_values, right_vectors_h = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        threshold = (
            relative_singular_value_tolerance * float(singular_values[0])
            if len(singular_values)
            else np.inf
        )
        numerical_rank = int(np.count_nonzero(singular_values > threshold))
        if numerical_rank == 0:
            raise ValueError("purification factor has zero numerical Schmidt rank")
        active_values = singular_values[:numerical_rank]
        cumulative = np.cumsum(np.square(active_values))
        target = squared_weight_retention * float(cumulative[-1])
        retained = min(
            int(np.searchsorted(cumulative, target, side="left") + 1),
            numerical_rank,
        )
        if maximum_rank is not None:
            retained = min(retained, int(maximum_rank))
        return (
            matrix,
            left_vectors,
            singular_values,
            right_vectors_h,
            numerical_rank,
            retained,
        )

    candidates = {
        mode: decomposition(mode)
        for mode in (("left", "right") if orientation == "auto" else (orientation,))
    }
    chosen_orientation = min(
        candidates,
        key=lambda mode: (
            candidates[mode][5],
            candidates[mode][4],
            0 if mode == "left" else 1,
        ),
    )
    (
        _,
        left_vectors,
        singular_values,
        right_vectors_h,
        numerical_rank,
        retained_rank,
    ) = candidates[chosen_orientation]

    pair_components = []
    for component in range(retained_rank):
        if chosen_orientation == "left":
            right_piece = right_vectors_h[component].reshape(
                values.shape[0],
                right_count,
            )
            pair_component = np.einsum(
                "i,pj->pij",
                singular_values[component] * left_vectors[:, component],
                right_piece,
                optimize=True,
            )
        else:
            left_piece = right_vectors_h[component].reshape(
                values.shape[0],
                left_count,
            )
            pair_component = np.einsum(
                "pi,j->pij",
                singular_values[component] * left_piece,
                left_vectors[:, component],
                optimize=True,
            )
        pair_components.append(pair_component)
    pair_components_array = np.asarray(pair_components, dtype=np.complex128)
    output_components = np.einsum(
        "apq,oq->apo",
        pair_components_array.reshape(
            retained_rank,
            values.shape[0],
            pair_count,
        ),
        multiplication,
        optimize=True,
    )
    component_norms_squared = np.real(
        np.einsum(
            "api,ij,apj->a",
            output_components,
            gram,
            output_components.conj(),
            optimize=True,
        )
    )
    largest_component_norm = float(
        np.sqrt(max(float(np.max(component_norms_squared)), 0.0))
    )
    component_threshold = relative_component_tolerance * max(
        largest_component_norm,
        np.finfo(np.float64).tiny,
    )
    component_norms = np.sqrt(np.maximum(component_norms_squared, 0.0))
    selected = np.flatnonzero(component_norms > component_threshold)
    if len(selected) == 0:
        raise ValueError("all Schmidt components vanish after multiplication")
    output_components = output_components[selected]
    pair_components_array = pair_components_array[selected]
    component_norms = component_norms[selected]
    leaf_tensor = output_components / component_norms[:, None, None]
    reconstructed_factor = np.einsum(
        "a,api->pi",
        component_norms,
        leaf_tensor,
        optimize=True,
    )
    reconstructed_lifted = np.sum(pair_components_array, axis=0).reshape(
        values.shape[0],
        pair_count,
    )
    lifted_scale = max(np.linalg.norm(lifted), np.finfo(np.float64).tiny)
    factor_scale = max(np.linalg.norm(values), np.finfo(np.float64).tiny)
    target_h = values.conj().T @ values
    reconstructed_h = reconstructed_factor.conj().T @ reconstructed_factor
    hermitian_scale = max(np.linalg.norm(target_h), np.finfo(np.float64).tiny)
    active_values = singular_values[:numerical_rank]
    retained_squared_weight = float(
        np.sum(np.square(singular_values[:retained_rank]))
        / np.sum(np.square(active_values))
    )
    return PurificationSchmidtChannels(
        orientation=chosen_orientation,
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        retained_rank=len(selected),
        retained_squared_weight=retained_squared_weight,
        selected_singular_indices=selected,
        leaf_tensor=leaf_tensor,
        leaf_combiner=component_norms.astype(np.complex128),
        reconstructed_factor=reconstructed_factor,
        lifted_factor_relative_error=float(
            np.linalg.norm(reconstructed_lifted - lifted) / lifted_scale
        ),
        factor_relative_error=float(
            np.linalg.norm(reconstructed_factor - values) / factor_scale
        ),
        hermitian_relative_error=float(
            np.linalg.norm(reconstructed_h - target_h) / hermitian_scale
        ),
    )


def lift_hermitian_form(
    hermitian: np.ndarray,
    right_inverse: np.ndarray,
) -> np.ndarray:
    """Lift ``s^dagger H s`` to quotient-product coordinates."""

    matrix = np.asarray(hermitian, dtype=np.complex128)
    inverse = np.asarray(right_inverse, dtype=np.complex128)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] != inverse.shape[1]
    ):
        raise ValueError("Hermitian and right-inverse dimensions disagree")
    lifted = inverse.conj() @ matrix @ inverse.T
    return 0.5 * (lifted + lifted.conj().T)


def pull_back_lifted_hermitian(
    lifted_hermitian: np.ndarray,
    multiplication: QuotientMultiplicationMap,
) -> np.ndarray:
    """Recover the output-space Hermitian form from an exact product lift."""

    matrix = multiplication.matrix.toarray()
    lifted = np.asarray(lifted_hermitian, dtype=np.complex128)
    recovered = matrix.conj() @ lifted @ matrix.T
    return 0.5 * (recovered + recovered.conj().T)


def operator_schmidt_singular_values(
    lifted_hermitian: np.ndarray,
    *,
    left_count: int,
    right_count: int,
) -> np.ndarray:
    """Return the operator-Schmidt spectrum across one multiplication cut."""

    matrix = np.asarray(lifted_hermitian, dtype=np.complex128)
    expected = left_count * right_count
    if matrix.shape != (expected, expected):
        raise ValueError("lifted operator dimension disagrees with the cut")
    schmidt = (
        matrix.reshape(left_count, right_count, left_count, right_count)
        .transpose(0, 2, 1, 3)
        .reshape(left_count**2, right_count**2)
    )
    return np.linalg.svd(schmidt, compute_uv=False)


def rank_for_squared_weight(
    singular_values: np.ndarray,
    threshold: float,
) -> int:
    """Smallest rank retaining a requested fraction of squared weight."""

    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("singular values must be a non-empty vector")
    if not 0 < threshold <= 1:
        raise ValueError("squared-weight threshold must lie in (0, 1]")
    weights = np.square(values)
    cumulative = np.cumsum(weights) / np.sum(weights)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def lift_gauge_diagnostics(
    hermitian: np.ndarray,
    multiplication: QuotientMultiplicationMap,
    right_inverses: Mapping[str, np.ndarray],
    *,
    squared_weight_thresholds: tuple[float, ...] = (0.99, 0.999, 0.9999),
) -> dict[str, dict[str, object]]:
    """Audit exactness and Schmidt sensitivity across right-inverse gauges."""

    source = np.asarray(hermitian, dtype=np.complex128)
    raw = multiplication.matrix.toarray()
    reports: dict[str, dict[str, object]] = {}
    for name, right_inverse in right_inverses.items():
        inverse = np.asarray(right_inverse, dtype=np.complex128)
        right_error = float(
            np.linalg.norm(raw @ inverse - np.eye(raw.shape[0]))
            / np.sqrt(raw.shape[0])
        )
        lifted = lift_hermitian_form(source, inverse)
        recovered = pull_back_lifted_hermitian(lifted, multiplication)
        recovery_error = float(
            np.linalg.norm(recovered - source)
            / max(np.linalg.norm(source), np.finfo(float).tiny)
        )
        spectrum = operator_schmidt_singular_values(
            lifted,
            left_count=multiplication.left_count,
            right_count=multiplication.right_count,
        )
        reports[str(name)] = {
            "right_inverse_relative_frobenius_error": right_error,
            "hermitian_recovery_relative_frobenius_error": recovery_error,
            "lifted_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(lifted))),
            "lifted_frobenius_norm": float(np.linalg.norm(lifted)),
            "operator_schmidt_rank": int(
                np.count_nonzero(
                    spectrum
                    > max(float(spectrum[0]), 1.0)
                    * np.finfo(np.float64).eps
                    * max(lifted.shape)
                )
            ),
            "rank_for_cumulative_squared_weight": {
                f"{threshold:.12g}": rank_for_squared_weight(spectrum, threshold)
                for threshold in squared_weight_thresholds
            },
            "leading_operator_schmidt_singular_values": spectrum[:16].tolist(),
        }
    return reports
