"""Finite-dimensional Ricci-flat metric fitting on the local gCICY patch.

The fit implemented here is a local, traditional Monge-Ampere iteration:

    g_phi = g_0 + partial partialbar phi,

where phi is a real polynomial in the six real local coordinates
(Re s, Re t, Re r, Im s, Im t, Im r).  At each step we solve the linearized
Monge-Ampere equation

    trace_g(partial partialbar delta_phi) ~= -(log det g - log |Omega|^2)_0

on sampled points, then keep only steps that preserve positive definiteness on
both training and validation samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .simple_patch import Array, holomorphic_volume_log_density, induced_metric


@dataclass(frozen=True)
class ResidualStats:
    """Centered Monge-Ampere residual statistics for a metric sample set."""

    rms: float
    max_abs: float
    min_eigenvalue: float
    mean_log_error: float


@dataclass(frozen=True)
class FitResult:
    """Result of the finite-dimensional Monge-Ampere fit."""

    coefficients: Array
    exponents: Array
    history: list[dict[str, float | int]]
    baseline_train: ResidualStats
    baseline_validation: ResidualStats
    best_train: ResidualStats
    best_validation: ResidualStats


def real_coordinates(params: Array) -> Array:
    """Return real coordinates ordered as (Re s,t,r, Im s,t,r)."""

    p = np.asarray(params, dtype=np.complex128)
    if p.ndim == 1:
        return np.concatenate([p.real, p.imag])
    return np.concatenate([p.real, p.imag], axis=1)


def monomial_exponents(num_vars: int = 6, min_degree: int = 2, max_degree: int = 4) -> Array:
    """Return all exponent tuples with total degree in [min_degree, max_degree]."""

    if min_degree < 0 or max_degree < min_degree:
        raise ValueError("Expected 0 <= min_degree <= max_degree.")

    exponents: list[tuple[int, ...]] = []
    for powers in product(range(max_degree + 1), repeat=num_vars):
        degree = sum(powers)
        if min_degree <= degree <= max_degree:
            exponents.append(powers)
    return np.asarray(exponents, dtype=np.int64)


def real_monomial_hessian(x: Array, exponent: Array) -> Array:
    """Real Hessian of x**exponent for one exponent vector."""

    x = np.asarray(x, dtype=float)
    alpha = np.asarray(exponent, dtype=np.int64)
    n_vars = x.size
    hessian = np.zeros((n_vars, n_vars), dtype=float)

    for row in range(n_vars):
        if alpha[row] >= 2:
            beta = alpha.copy()
            beta[row] -= 2
            hessian[row, row] = alpha[row] * (alpha[row] - 1) * float(np.prod(x**beta))
        for col in range(row + 1, n_vars):
            if alpha[row] >= 1 and alpha[col] >= 1:
                beta = alpha.copy()
                beta[row] -= 1
                beta[col] -= 1
                value = alpha[row] * alpha[col] * float(np.prod(x**beta))
                hessian[row, col] = value
                hessian[col, row] = value
    return hessian


def complex_hessian_from_real(real_hessian: Array) -> Array:
    """Convert a real Hessian to partial_i partial_bar_j Hessian."""

    h = np.asarray(real_hessian, dtype=float)
    if h.shape != (6, 6):
        raise ValueError(f"Expected a 6 x 6 real Hessian, got {h.shape}.")
    hxx = h[:3, :3]
    hxy = h[:3, 3:]
    hyx = h[3:, :3]
    hyy = h[3:, 3:]
    return 0.25 * (hxx + hyy) + 0.25j * (hxy - hyx)


def basis_hessians(params: Array, exponents: Array) -> Array:
    """Return basis partial partialbar Hessians for each sample and exponent."""

    x_values = real_coordinates(params)
    if x_values.ndim == 1:
        x_values = x_values[None, :]
    out = np.empty((x_values.shape[0], len(exponents), 3, 3), dtype=np.complex128)
    for point_idx, x in enumerate(x_values):
        for basis_idx, exponent in enumerate(exponents):
            out[point_idx, basis_idx] = complex_hessian_from_real(real_monomial_hessian(x, exponent))
    return out


def baseline_metrics(params: Array) -> Array:
    """Evaluate the starting pullback Fubini-Study metric on many points."""

    return np.asarray([induced_metric(point) for point in np.asarray(params, dtype=np.complex128)])


def apply_correction(metrics: Array, basis_tensor: Array, coefficients: Array) -> Array:
    """Apply partial partialbar phi to a metric batch."""

    return np.asarray(metrics, dtype=np.complex128) + np.einsum(
        "m,nmij->nij", np.asarray(coefficients, dtype=float), basis_tensor
    )


def residual_values(params: Array, metrics: Array) -> Array:
    """Return uncentered log det(g) - log |Omega|^2 values."""

    params = np.asarray(params, dtype=np.complex128)
    metrics = np.asarray(metrics, dtype=np.complex128)
    values = np.empty(params.shape[0], dtype=float)
    for idx, (point, metric) in enumerate(zip(params, metrics, strict=True)):
        eigvals = np.linalg.eigvalsh(metric)
        if eigvals[0] <= 0:
            values[idx] = np.nan
        else:
            values[idx] = float(np.sum(np.log(eigvals)) - holomorphic_volume_log_density(point))
    return values


def residual_stats(params: Array, metrics: Array) -> ResidualStats:
    """Compute centered Monge-Ampere residual statistics."""

    eigvals = np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128))
    min_eigenvalue = float(np.min(eigvals))
    values = residual_values(params, metrics)
    if not np.all(np.isfinite(values)):
        return ResidualStats(float("nan"), float("nan"), min_eigenvalue, float("nan"))
    centered = values - float(np.mean(values))
    return ResidualStats(
        rms=float(np.sqrt(np.mean(centered**2))),
        max_abs=float(np.max(np.abs(centered))),
        min_eigenvalue=min_eigenvalue,
        mean_log_error=float(np.mean(values)),
    )


def linearized_fit(
    train_params: Array,
    validation_params: Array,
    *,
    max_degree: int = 4,
    l2: float = 1.0,
    iterations: int = 8,
    min_eigenvalue: float = 1e-5,
) -> FitResult:
    """Fit a local Ricci-flat metric approximation by linearized MA steps."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if l2 < 0:
        raise ValueError("l2 must be nonnegative")

    exponents = monomial_exponents(max_degree=max_degree)
    train_base = baseline_metrics(train_params)
    validation_base = baseline_metrics(validation_params)
    train_basis = basis_hessians(train_params, exponents)
    validation_basis = basis_hessians(validation_params, exponents)

    coefficients = np.zeros(len(exponents), dtype=float)
    best_coefficients = coefficients.copy()
    train_metrics = train_base.copy()
    validation_metrics = validation_base.copy()
    baseline_train = residual_stats(train_params, train_metrics)
    baseline_validation = residual_stats(validation_params, validation_metrics)
    best_train = baseline_train
    best_validation = baseline_validation
    current_train = baseline_train
    history: list[dict[str, float | int]] = [
        {
            "iteration": -1,
            "step": 0.0,
            "train_rms": baseline_train.rms,
            "train_max_abs": baseline_train.max_abs,
            "train_min_eigenvalue": baseline_train.min_eigenvalue,
            "validation_rms": baseline_validation.rms,
            "validation_max_abs": baseline_validation.max_abs,
            "validation_min_eigenvalue": baseline_validation.min_eigenvalue,
        }
    ]

    for iteration in range(iterations):
        centered = residual_values(train_params, train_metrics)
        centered = centered - float(np.mean(centered))
        design = np.empty((len(train_params), len(exponents)), dtype=float)
        for point_idx, metric in enumerate(train_metrics):
            inverse_metric = np.linalg.inv(metric)
            design[point_idx] = [
                float(np.real(np.trace(inverse_metric @ train_basis[point_idx, basis_idx])))
                for basis_idx in range(len(exponents))
            ]
        design = design - np.mean(design, axis=0, keepdims=True)
        target = -centered
        lhs = design.T @ design + l2 * np.eye(len(exponents))
        rhs = design.T @ target
        delta = np.linalg.solve(lhs, rhs)

        best_step: tuple[float, ResidualStats, ResidualStats, Array, Array] | None = None
        for step in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
            candidate_coefficients = coefficients + step * delta
            candidate_train = apply_correction(train_base, train_basis, candidate_coefficients)
            candidate_validation = apply_correction(validation_base, validation_basis, candidate_coefficients)
            train_stats = residual_stats(train_params, candidate_train)
            validation_stats = residual_stats(validation_params, candidate_validation)
            if train_stats.min_eigenvalue <= min_eigenvalue:
                continue
            if validation_stats.min_eigenvalue <= min_eigenvalue:
                continue
            if not np.isfinite(train_stats.rms) or not np.isfinite(validation_stats.rms):
                continue
            if train_stats.rms > current_train.rms:
                continue
            if best_step is None or validation_stats.rms < best_step[2].rms:
                best_step = (step, train_stats, validation_stats, candidate_coefficients, candidate_train)

        if best_step is None:
            history.append(
                {
                    "iteration": iteration,
                    "step": 0.0,
                    "train_rms": current_train.rms,
                    "train_max_abs": current_train.max_abs,
                    "train_min_eigenvalue": current_train.min_eigenvalue,
                    "validation_rms": best_validation.rms,
                    "validation_max_abs": best_validation.max_abs,
                    "validation_min_eigenvalue": best_validation.min_eigenvalue,
                }
            )
            break

        step, train_stats, validation_stats, coefficients, train_metrics = best_step
        validation_metrics = apply_correction(validation_base, validation_basis, coefficients)
        current_train = train_stats
        if validation_stats.rms < best_validation.rms:
            best_coefficients = coefficients.copy()
            best_train = train_stats
            best_validation = validation_stats
        history.append(
            {
                "iteration": iteration,
                "step": step,
                "train_rms": train_stats.rms,
                "train_max_abs": train_stats.max_abs,
                "train_min_eigenvalue": train_stats.min_eigenvalue,
                "validation_rms": validation_stats.rms,
                "validation_max_abs": validation_stats.max_abs,
                "validation_min_eigenvalue": validation_stats.min_eigenvalue,
            }
        )

    return FitResult(
        coefficients=best_coefficients,
        exponents=exponents,
        history=history,
        baseline_train=baseline_train,
        baseline_validation=baseline_validation,
        best_train=best_train,
        best_validation=best_validation,
    )
