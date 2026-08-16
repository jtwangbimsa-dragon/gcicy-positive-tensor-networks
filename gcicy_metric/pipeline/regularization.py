"""Basis-covariant regularization paths for positive Hermitian H-metrics."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def _positive_hermitian(matrix: Array) -> Array:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise ValueError("matrix must be non-empty and square")
    value = 0.5 * (value + value.conjugate().T)
    eigenvalues = np.linalg.eigvalsh(value)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
        raise ValueError("matrix must be finite and positive definite")
    return value


def normalize_positive_hermitian(matrix: Array, *, trace: float | None = None) -> Array:
    """Hermitianize an SPD matrix and fix its irrelevant overall scale."""

    value = _positive_hermitian(matrix)
    target_trace = float(len(value) if trace is None else trace)
    observed_trace = float(np.trace(value).real)
    if not np.isfinite(target_trace) or target_trace <= 0 or observed_trace <= 0:
        raise ValueError("matrix trace normalization must be positive")
    return value * (target_trace / observed_trace)


def reference_relative_spectrum(reference: Array, candidate: Array) -> Array:
    """Return generalized eigenvalues of ``candidate`` relative to ``reference``."""

    reference_h = _positive_hermitian(reference)
    candidate_h = _positive_hermitian(candidate)
    if reference_h.shape != candidate_h.shape:
        raise ValueError("reference and candidate shapes differ")
    eigenvalues, eigenvectors = np.linalg.eigh(reference_h)
    inverse_root = (
        eigenvectors / np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.conjugate().T
    relative = inverse_root @ candidate_h @ inverse_root
    relative = 0.5 * (relative + relative.conjugate().T)
    output = np.linalg.eigvalsh(relative)
    if not np.all(np.isfinite(output)) or output[0] <= 0:
        raise FloatingPointError("reference-relative spectrum is not positive")
    # Overall H-matrix scales do not change the Kahler metric.  Removing the
    # geometric mean makes this spectrum invariant under those scales as well
    # as under a congruent change of section basis.
    return output / np.exp(np.mean(np.log(output)))


def reference_geodesic_interpolate(
    reference: Array,
    candidate: Array,
    alpha: float,
) -> Array:
    """Interpolate on the SPD cone from reference (0) to candidate (1).

    The construction uses generalized eigenvalues of the matrix pair, so it is
    equivariant under a change of section basis.  The output trace is fixed to
    the matrix dimension because an overall scale does not change the metric.
    """

    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    reference_h = _positive_hermitian(reference)
    candidate_h = _positive_hermitian(candidate)
    if reference_h.shape != candidate_h.shape:
        raise ValueError("reference and candidate shapes differ")

    eigenvalues, eigenvectors = np.linalg.eigh(reference_h)
    root = (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conjugate().T
    inverse_root = (
        eigenvectors / np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.conjugate().T
    relative = inverse_root @ candidate_h @ inverse_root
    relative = 0.5 * (relative + relative.conjugate().T)
    relative_values, relative_vectors = np.linalg.eigh(relative)
    if not np.all(np.isfinite(relative_values)) or relative_values[0] <= 0:
        raise FloatingPointError("reference-relative matrix is not positive")
    powered = (
        relative_vectors * relative_values ** float(alpha)
    ) @ relative_vectors.conjugate().T
    output = root @ powered @ root
    return normalize_positive_hermitian(output)


def reference_log_spectrum_clip(
    reference: Array,
    candidate: Array,
    max_abs_log_eigenvalue: float,
) -> Array:
    """Clip only extreme generalized H-eigenmodes relative to a reference.

    Generalized eigenvalues are first centered to geometric mean one because
    the overall H scale is physically irrelevant.  Moderate modes are retained
    exactly, while modes outside ``exp(+-max_abs_log_eigenvalue)`` are clipped.
    """

    if not np.isfinite(max_abs_log_eigenvalue) or max_abs_log_eigenvalue < 0:
        raise ValueError("max_abs_log_eigenvalue must be finite and non-negative")
    reference_h = _positive_hermitian(reference)
    candidate_h = _positive_hermitian(candidate)
    if reference_h.shape != candidate_h.shape:
        raise ValueError("reference and candidate shapes differ")

    eigenvalues, eigenvectors = np.linalg.eigh(reference_h)
    root = (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conjugate().T
    inverse_root = (
        eigenvectors / np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.conjugate().T
    relative = inverse_root @ candidate_h @ inverse_root
    relative = 0.5 * (relative + relative.conjugate().T)
    relative_values, relative_vectors = np.linalg.eigh(relative)
    if not np.all(np.isfinite(relative_values)) or relative_values[0] <= 0:
        raise FloatingPointError("reference-relative matrix is not positive")
    centered_logs = np.log(relative_values)
    centered_logs -= np.mean(centered_logs)
    clipped_values = np.exp(
        np.clip(
            centered_logs,
            -max_abs_log_eigenvalue,
            max_abs_log_eigenvalue,
        )
    )
    clipped = (
        relative_vectors * clipped_values[None, :]
    ) @ relative_vectors.conjugate().T
    output = root @ clipped @ root
    return normalize_positive_hermitian(output)


def targeted_section_rank_one_update(
    h_matrix: Array,
    section_values: Array,
    strength: float,
    *,
    preserve_trace: bool = True,
) -> Array:
    """Lift one learned section-norm well with a positive rank-one update."""

    h = _positive_hermitian(h_matrix)
    section = np.asarray(section_values, dtype=np.complex128).reshape(-1)
    if section.shape != (len(h),) or not np.all(np.isfinite(section)):
        raise ValueError("section_values must be a finite vector matching H")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("rank-one update strength must be finite and non-negative")
    h_section = h @ section
    denominator = float(np.real(np.vdot(section, h_section)))
    if not np.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("section H-norm must be positive")
    candidate = (
        h + strength * np.outer(h_section, np.conjugate(h_section)) / denominator
    )
    candidate = 0.5 * (candidate + candidate.conjugate().T)
    if preserve_trace:
        candidate *= float(np.trace(h).real) / float(np.trace(candidate).real)
    return _positive_hermitian(candidate)
