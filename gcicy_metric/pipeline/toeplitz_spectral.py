"""Matrix-free Berezin modes and affine Hermitian update diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def normalized_weights(weights: Array) -> Array:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or float(np.sum(values)) <= 0
    ):
        raise ValueError("weights must be finite, non-negative, and have positive mass")
    return values / np.sum(values)


def _positive_hermitian_eigh(matrix: Array) -> tuple[Array, Array]:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("expected a square Hermitian matrix")
    value = 0.5 * (value + value.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
        raise FloatingPointError("matrix is not positive definite")
    return eigenvalues, eigenvectors


def _hermitian_from_eigh(eigenvectors: Array, eigenvalues: Array) -> Array:
    value = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.conjugate().T
    return 0.5 * (value + value.conjugate().T)


def coherent_sections(section_values: Array, h_matrix: Array) -> Array:
    """Return row-wise unit coherent sections H^(1/2)s / ||H^(1/2)s||."""

    values = np.asarray(section_values, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("section values must be a point-by-section matrix")
    eigenvalues, eigenvectors = _positive_hermitian_eigh(h_matrix)
    if values.shape[1] != len(eigenvalues):
        raise ValueError("section values and H have incompatible shapes")
    square_root = _hermitian_from_eigh(eigenvectors, np.sqrt(eigenvalues))
    coherent = values @ square_root.T
    norms = np.linalg.norm(coherent, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise FloatingPointError("coherent section normalization failed")
    coherent /= norms[:, None]
    return coherent


def weighted_center(field: Array, weights: Array) -> Array:
    values = np.asarray(field, dtype=np.float64).reshape(-1)
    masses = normalized_weights(weights)
    if values.shape != masses.shape or not np.all(np.isfinite(values)):
        raise ValueError("field and weights must be finite aligned vectors")
    return values - float(np.sum(masses * values))


def weighted_norm(field: Array, weights: Array) -> float:
    values = np.asarray(field, dtype=np.float64).reshape(-1)
    masses = normalized_weights(weights)
    if values.shape != masses.shape or not np.all(np.isfinite(values)):
        raise ValueError("field and weights must be finite aligned vectors")
    return float(np.sqrt(np.sum(masses * values**2)))


def toeplitz_lift(coherent: Array, weights: Array, field: Array) -> Array:
    """Lift a real symbol to a trace-free Hermitian coherent-state moment."""

    vectors = np.asarray(coherent, dtype=np.complex128)
    masses = normalized_weights(weights)
    values = np.asarray(field, dtype=np.float64).reshape(-1)
    if vectors.ndim != 2 or len(vectors) != len(masses) or values.shape != masses.shape:
        raise ValueError("coherent sections, weights, and field are not aligned")
    weighted_field = masses * values
    moment = vectors.T @ (weighted_field[:, None] * vectors.conjugate())
    section_count = vectors.shape[1]
    lifted = section_count * moment - float(np.sum(weighted_field)) * np.eye(
        section_count,
        dtype=np.complex128,
    )
    return 0.5 * (lifted + lifted.conjugate().T)


def apply_berezin(coherent: Array, weights: Array, field: Array) -> Array:
    """Apply Bf(x)=N integral |<v(x),v(y)>|^2 f(y) dnu(y)."""

    vectors = np.asarray(coherent, dtype=np.complex128)
    masses = normalized_weights(weights)
    values = np.asarray(field, dtype=np.float64).reshape(-1)
    if vectors.ndim != 2 or len(vectors) != len(masses) or values.shape != masses.shape:
        raise ValueError("coherent sections, weights, and field are not aligned")
    moment = vectors.T @ ((masses * values)[:, None] * vectors.conjugate())
    result = vectors.shape[1] * np.einsum(
        "na,ab,nb->n",
        vectors.conjugate(),
        moment,
        vectors,
        optimize=True,
    ).real
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Berezin application produced non-finite values")
    return result


def balance_defect(coherent: Array, weights: Array) -> float:
    vectors = np.asarray(coherent, dtype=np.complex128)
    masses = normalized_weights(weights)
    if vectors.ndim != 2 or len(vectors) != len(masses):
        raise ValueError("coherent sections and weights are not aligned")
    moment = vectors.T @ (masses[:, None] * vectors.conjugate())
    defect = vectors.shape[1] * moment - np.eye(vectors.shape[1])
    return float(np.linalg.norm(defect, ord="fro") / np.sqrt(vectors.shape[1]))


def estimate_berezin_spectral_radius(
    coherent: Array,
    weights: Array,
    *,
    iterations: int = 24,
    seed: int = 20260717,
) -> float:
    """Estimate the largest eigenvalue in the quadrature-weighted L2 space."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    masses = normalized_weights(weights)
    rng = np.random.default_rng(seed)
    vector = 1.0 + 0.01 * rng.normal(size=len(masses))
    vector /= weighted_norm(vector, masses)
    for _ in range(iterations):
        candidate = apply_berezin(coherent, masses, vector)
        norm = weighted_norm(candidate, masses)
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Berezin power iteration failed")
        vector = candidate / norm
    applied = apply_berezin(coherent, masses, vector)
    radius = float(np.sum(masses * vector * applied))
    if not np.isfinite(radius) or radius <= 0:
        raise FloatingPointError("invalid Berezin spectral-radius estimate")
    return radius


def chebyshev_berezin_modes(
    coherent: Array,
    weights: Array,
    residual: Array,
    mode_count: int,
    *,
    spectral_radius: float | None = None,
    spectral_safety_factor: float = 1.02,
) -> tuple[Array, float]:
    """Construct centered, unit-RMS T_j(2B/rho-I) residual modes."""

    if mode_count <= 0:
        raise ValueError("mode_count must be positive")
    if spectral_safety_factor < 1.0:
        raise ValueError("spectral_safety_factor must be at least one")
    masses = normalized_weights(weights)
    centered = weighted_center(residual, masses)
    initial_norm = weighted_norm(centered, masses)
    if initial_norm <= 0:
        raise ValueError("residual has no non-constant component")
    radius = (
        estimate_berezin_spectral_radius(coherent, masses)
        if spectral_radius is None
        else float(spectral_radius)
    )
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("spectral radius must be finite and positive")
    scaled_radius = radius * spectral_safety_factor

    def scaled_apply(field: Array) -> Array:
        value = 2.0 * apply_berezin(coherent, masses, field) / scaled_radius - field
        return weighted_center(value, masses)

    raw_modes = [centered]
    if mode_count > 1:
        raw_modes.append(scaled_apply(centered))
    for _ in range(2, mode_count):
        raw_modes.append(2.0 * scaled_apply(raw_modes[-1]) - raw_modes[-2])

    modes = []
    for mode in raw_modes:
        mode = weighted_center(mode, masses)
        norm = weighted_norm(mode, masses)
        if norm <= 1.0e-14 * initial_norm:
            modes.append(np.zeros_like(mode))
        else:
            modes.append(mode / norm)
    return np.asarray(modes, dtype=np.float64), radius


def affine_log_tangent(current: Array, candidate: Array) -> Array:
    """Return centered log(H^-1/2 candidate H^-1/2)."""

    current_eigenvalues, current_eigenvectors = _positive_hermitian_eigh(current)
    inverse_square_root = _hermitian_from_eigh(
        current_eigenvectors,
        1.0 / np.sqrt(current_eigenvalues),
    )
    relative = inverse_square_root @ np.asarray(candidate) @ inverse_square_root
    relative_eigenvalues, relative_eigenvectors = _positive_hermitian_eigh(relative)
    tangent = _hermitian_from_eigh(relative_eigenvectors, np.log(relative_eigenvalues))
    tangent -= np.trace(tangent).real / len(tangent) * np.eye(len(tangent))
    return 0.5 * (tangent + tangent.conjugate().T)


def affine_exponential_update(current: Array, tangent: Array) -> Array:
    """Apply H^+=H^(1/2) exp(A) H^(1/2), removing the irrelevant scale."""

    current_eigenvalues, current_eigenvectors = _positive_hermitian_eigh(current)
    square_root = _hermitian_from_eigh(
        current_eigenvectors,
        np.sqrt(current_eigenvalues),
    )
    tangent_value = np.asarray(tangent, dtype=np.complex128)
    tangent_value = 0.5 * (tangent_value + tangent_value.conjugate().T)
    tangent_eigenvalues, tangent_eigenvectors = np.linalg.eigh(tangent_value)
    exponential = _hermitian_from_eigh(
        tangent_eigenvectors,
        np.exp(tangent_eigenvalues),
    )
    candidate = square_root @ exponential @ square_root
    candidate = 0.5 * (candidate + candidate.conjugate().T)
    return candidate * (len(candidate) / float(np.trace(candidate).real))


@dataclass(frozen=True)
class HermitianSpanFit:
    coefficients: Array
    reconstruction: Array
    relative_frobenius_error: float
    explained_squared_fraction: float
    gram_condition_number: float
    numerical_rank: int


@dataclass(frozen=True)
class DefectScaledSpectralFit:
    """Constant dimensionless gains for defect-scaled spectral coefficients."""

    gains: Array
    normalized_root_mean_square_error: float
    sample_count: int


def fit_defect_scaled_spectral_gains(
    defects: Array,
    coefficients: Array,
    *,
    sample_weights: Array | None = None,
) -> DefectScaledSpectralFit:
    """Fit ``coefficients ~= defect * gains`` with equal weight across scales.

    Dividing each target coefficient by its positive balance defect before the
    least-squares fit prevents early, large T-map steps from overwhelming the
    near-fixed-point dynamics that the controller is intended to learn.
    """

    defect_values = np.asarray(defects, dtype=np.float64).reshape(-1)
    coefficient_values = np.asarray(coefficients, dtype=np.float64)
    if coefficient_values.ndim == 1:
        coefficient_values = coefficient_values[:, None]
    if (
        defect_values.size == 0
        or coefficient_values.ndim != 2
        or coefficient_values.shape[0] != defect_values.size
        or not np.all(np.isfinite(defect_values))
        or not np.all(np.isfinite(coefficient_values))
        or np.any(defect_values <= 0)
    ):
        raise ValueError("defects and coefficients must be finite and aligned")
    if sample_weights is None:
        weights = np.full(defect_values.size, 1.0 / defect_values.size)
    else:
        weights = normalized_weights(sample_weights)
        if weights.shape != defect_values.shape:
            raise ValueError("sample weights must align with defects")

    normalized_targets = coefficient_values / defect_values[:, None]
    gains = np.sum(weights[:, None] * normalized_targets, axis=0)
    residuals = normalized_targets - gains[None, :]
    rms = float(np.sqrt(np.sum(weights[:, None] * residuals**2)))
    return DefectScaledSpectralFit(
        gains=np.asarray(gains, dtype=np.float64),
        normalized_root_mean_square_error=rms,
        sample_count=defect_values.size,
    )


def fit_hermitian_span(target: Array, basis: Array, *, rcond: float = 1.0e-12) -> HermitianSpanFit:
    """Least-squares fit in the real Frobenius inner product on Hermitian matrices."""

    target_value = np.asarray(target, dtype=np.complex128)
    basis_values = np.asarray(basis, dtype=np.complex128)
    if (
        target_value.ndim != 2
        or target_value.shape[0] != target_value.shape[1]
        or basis_values.ndim != 3
        or basis_values.shape[1:] != target_value.shape
        or len(basis_values) == 0
    ):
        raise ValueError("target and Hermitian basis have incompatible shapes")
    gram = np.real(np.einsum("iab,jba->ij", basis_values, basis_values, optimize=True))
    right = np.real(np.einsum("iab,ba->i", basis_values, target_value, optimize=True))
    coefficients, _, rank, singular_values = np.linalg.lstsq(gram, right, rcond=rcond)
    reconstruction = np.einsum("i,iab->ab", coefficients, basis_values, optimize=True)
    reconstruction = 0.5 * (reconstruction + reconstruction.conjugate().T)
    target_norm = float(np.linalg.norm(target_value, ord="fro"))
    error_norm = float(np.linalg.norm(target_value - reconstruction, ord="fro"))
    relative_error = error_norm / max(target_norm, np.finfo(float).tiny)
    positive_singular_values = singular_values[
        singular_values > rcond * singular_values[0]
    ]
    condition = (
        float(positive_singular_values[0] / positive_singular_values[-1])
        if len(positive_singular_values)
        else float("inf")
    )
    return HermitianSpanFit(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        reconstruction=reconstruction,
        relative_frobenius_error=float(relative_error),
        explained_squared_fraction=float(max(0.0, 1.0 - relative_error**2)),
        gram_condition_number=condition,
        numerical_rank=int(rank),
    )
