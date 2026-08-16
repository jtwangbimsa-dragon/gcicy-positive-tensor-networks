"""A local affine-patch model for the gCICY example in Anderson et al.

We use the configuration matrix

    [ P1 | 1 1 -1  1 ]
    [ P1 | 1 1  1 -1 ]
    [ P5 | 3 1  1  1 ]

from arXiv:1507.03235, Eq. (1.2).  This module studies one explicit affine
patch x0 = y0 = z0 = 1.  The construction keeps the negative-degree equations
as rational sections, but on this patch they become ordinary polynomial
conditions.

The metric computed here is the pullback of the product Fubini-Study metric
from P1 x P1 x P5.  It is not expected to be Ricci-flat; it is a baseline
Kahler metric and a useful starting point for later ML or PDE corrections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class MetricDiagnostics:
    """Numerical checks for one point on the local gCICY patch."""

    residual_norm: float
    min_eigenvalue: float
    max_eigenvalue: float
    logdet: float
    scalar_curvature: float | None = None


def _as_complex_vector(values: Array) -> Array:
    out = np.asarray(values, dtype=np.complex128)
    if out.shape != (3,):
        raise ValueError(f"Expected a vector of 3 complex parameters, got {out.shape}.")
    if abs(out[2]) < 1e-10:
        raise ValueError("The local parameter r = z4 must be nonzero on this patch.")
    return out


def p1_rest(a: complex, b: complex, w1: complex, w2: complex, w3: complex, w4: complex) -> complex:
    """The part of p1 not containing the linear w5 term.

    In homogeneous coordinates this corresponds to terms of multi-degree
    (1, 1, 3).  On the affine patch it becomes the polynomial below.
    """

    cubic = 0.20 * (w1**3 + 0.70 * w2**3 - 0.40 * w3**3 + 0.50 * w4**3)
    mixed = 0.11 * a * w2 * w3 - 0.09 * b * w1 * w4
    mixed += 0.07 * a * b * w2 * w4 + 0.05 * (a + b) * w1 * w2
    return 0.03 + cubic + mixed


def p1_rest_gradient(a: complex, b: complex, w1: complex, w2: complex, w3: complex, w4: complex) -> Array:
    """Gradient of p1_rest with respect to (a, b, w1, w2, w3, w4)."""

    return np.array(
        [
            0.11 * w2 * w3 + 0.07 * b * w2 * w4 + 0.05 * w1 * w2,
            -0.09 * w1 * w4 + 0.07 * a * w2 * w4 + 0.05 * w1 * w2,
            0.60 * w1**2 - 0.09 * b * w4 + 0.05 * (a + b) * w2,
            0.42 * w2**2 + 0.11 * a * w3 + 0.07 * a * b * w4 + 0.05 * (a + b) * w1,
            -0.24 * w3**2 + 0.11 * a * w2,
            0.30 * w4**2 - 0.09 * b * w1 + 0.07 * a * b * w2,
        ],
        dtype=np.complex128,
    )


def embedding(params: Array) -> Array:
    """Map local parameters (s, t, r) to affine ambient coordinates.

    Ambient affine coordinates are ordered as

        (a, b, w1, w2, w3, w4, w5)

    where a = x1/x0, b = y1/y0, and wi = zi/z0 in P5.

    The local gCICY equations are solved by

        w2 = s, w3 = t, w4 = r,
        w1 = s*t/r, a = -t/r, b = -s/r,
        w5 = -p1_rest.
    """

    s, t, r = _as_complex_vector(params)
    w2 = s
    w3 = t
    w4 = r
    w1 = s * t / r
    a = -t / r
    b = -s / r
    w5 = -p1_rest(a, b, w1, w2, w3, w4)
    return np.array([a, b, w1, w2, w3, w4, w5], dtype=np.complex128)


def embedding_jacobian(params: Array) -> Array:
    """Exact holomorphic Jacobian of embedding with respect to (s, t, r)."""

    s, t, r = _as_complex_vector(params)
    r2 = r**2
    jac = np.array(
        [
            [0.0, -1.0 / r, t / r2],
            [-1.0 / r, 0.0, s / r2],
            [t / r, s / r, -s * t / r2],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    a, b, w1, w2, w3, w4, _ = embedding(params)
    jac[6, :] = -p1_rest_gradient(a, b, w1, w2, w3, w4) @ jac[:6, :]
    return jac


def equations(ambient_coords: Array) -> Array:
    """Evaluate the four local defining equations (p1, p2, q1, q2)."""

    z = np.asarray(ambient_coords, dtype=np.complex128)
    if z.shape != (7,):
        raise ValueError(f"Expected 7 affine ambient coordinates, got {z.shape}.")

    a, b, w1, w2, w3, w4, w5 = z

    # p2 = x0*y0*z1 + x1*y0*z2 + x0*y1*z3 + x1*y1*z4
    p2 = w1 + a * w2 + b * w3 + a * b * w4

    # On x0 = y0 = 1, q1 = d1/x0 and q2 = c1/y0.
    q1 = w2 + b * w4
    q2 = w3 + a * w4

    p1 = w5 + p1_rest(a, b, w1, w2, w3, w4)
    return np.array([p1, p2, q1, q2], dtype=np.complex128)


def equation_jacobian(ambient_coords: Array) -> Array:
    """Jacobian of (p1, p2, q1, q2) with respect to ambient affine coordinates."""

    z = np.asarray(ambient_coords, dtype=np.complex128)
    if z.shape != (7,):
        raise ValueError(f"Expected 7 affine ambient coordinates, got {z.shape}.")

    a, b, w1, w2, w3, w4, _ = z
    grad = p1_rest_gradient(a, b, w1, w2, w3, w4)
    return np.array(
        [
            [grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], 1.0],
            [w2 + b * w4, w3 + a * w4, 1.0, a, b, a * b, 0.0],
            [0.0, w4, 0.0, 1.0, 0.0, b, 0.0],
            [w4, 0.0, 0.0, 0.0, 1.0, a, 0.0],
        ],
        dtype=np.complex128,
    )


def dependent_equation_jacobian(params: Array) -> Array:
    """Jacobian for residue coordinates (a, b, w1, w5).

    The independent local coordinates are (w2, w3, w4) = (s, t, r).
    For the residue holomorphic 3-form, the relevant denominator is

        det d(p1, p2, q1, q2) / d(a, b, w1, w5).

    On this patch the determinant simplifies to r**2, but returning the full
    matrix keeps the construction explicit and mirrors the complete-intersection
    residue formula used for ordinary CICYs.
    """

    a, b, w1, w2, w3, w4, _ = embedding(params)
    grad = p1_rest_gradient(a, b, w1, w2, w3, w4)
    return np.array(
        [
            [grad[0], grad[1], grad[2], 1.0],
            [w2 + b * w4, w3 + a * w4, 1.0, 0.0],
            [0.0, w4, 0.0, 0.0],
            [w4, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )


def residue_denominator(params: Array) -> complex:
    """Return the residue denominator for the local holomorphic 3-form."""

    _, _, r = _as_complex_vector(params)
    return r**2


def holomorphic_volume_log_density(params: Array) -> float:
    """Return log |Omega|^2 in the local coordinates (s, t, r)."""

    det = residue_denominator(params)
    if abs(det) < 1e-14:
        raise FloatingPointError("Residue denominator is too close to zero on this patch.")
    return float(-2.0 * np.log(abs(det)))


def monge_ampere_log_error(params: Array, metric: Array) -> float:
    """Return log det(g) - log |Omega|^2 before subtracting its mean."""

    eigvals = np.linalg.eigvalsh(np.asarray(metric, dtype=np.complex128))
    if eigvals[0] <= 0:
        raise FloatingPointError(f"Metric is not positive definite: eigenvalues={eigvals}.")
    return float(np.sum(np.log(eigvals)) - holomorphic_volume_log_density(params))


def fubini_study_metric(coords: Array) -> Array:
    """Return the affine Fubini-Study metric on P^n."""

    v = np.asarray(coords, dtype=np.complex128).reshape(-1)
    n = v.size
    rho = 1.0 + float(np.vdot(v, v).real)
    return (rho * np.eye(n, dtype=np.complex128) - np.outer(v, np.conjugate(v))) / rho**2


def ambient_fs_metric(ambient_coords: Array, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Array:
    """Product Fubini-Study metric on P1 x P1 x P5 in this affine patch."""

    z = np.asarray(ambient_coords, dtype=np.complex128)
    if z.shape != (7,):
        raise ValueError(f"Expected 7 affine ambient coordinates, got {z.shape}.")

    g = np.zeros((7, 7), dtype=np.complex128)
    g[0:1, 0:1] = weights[0] * fubini_study_metric(z[0:1])
    g[1:2, 1:2] = weights[1] * fubini_study_metric(z[1:2])
    g[2:7, 2:7] = weights[2] * fubini_study_metric(z[2:7])
    return g


def holomorphic_jacobian(fun, params: Array, step: float = 1e-6) -> Array:
    """Central finite-difference Jacobian for a holomorphic map."""

    p = _as_complex_vector(params)
    base = np.asarray(fun(p), dtype=np.complex128)
    jac = np.empty((base.size, p.size), dtype=np.complex128)
    for col in range(p.size):
        direction = np.zeros_like(p)
        direction[col] = step
        jac[:, col] = (fun(p + direction) - fun(p - direction)) / (2.0 * step)
    return jac


def induced_metric(params: Array, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Array:
    """Pull back the ambient product Fubini-Study metric to the gCICY patch."""

    z = embedding(params)
    g_ambient = ambient_fs_metric(z, weights=weights)
    jac = embedding_jacobian(params)
    g = jac.conjugate().T @ g_ambient @ jac
    return 0.5 * (g + g.conjugate().T)


def logdet_metric(params: Array) -> float:
    """Log determinant of the induced Hermitian metric."""

    eigvals = np.linalg.eigvalsh(induced_metric(params))
    if eigvals[0] <= 0:
        raise FloatingPointError(f"Metric is not positive definite: eigenvalues={eigvals}.")
    return float(np.sum(np.log(eigvals)))


def real_hessian(fun, x: Array, step: float = 2e-4) -> Array:
    """Central finite-difference Hessian for a real scalar function."""

    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.empty((n, n), dtype=float)
    f0 = float(fun(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step
        hess[i, i] = (float(fun(x + ei)) - 2.0 * f0 + float(fun(x - ei))) / step**2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step
            value = (
                float(fun(x + ei + ej))
                - float(fun(x + ei - ej))
                - float(fun(x - ei + ej))
                + float(fun(x - ei - ej))
            ) / (4.0 * step**2)
            hess[i, j] = value
            hess[j, i] = value
    return hess


def ricci_tensor(params: Array, step: float = 2e-4) -> Array:
    """Approximate R_{i bar j} = - partial_i partial_bar_j log det(g)."""

    p = _as_complex_vector(params)
    n = p.size
    x0 = np.concatenate([p.real, p.imag])

    def wrapped(x: Array) -> float:
        z = x[:n] + 1j * x[n:]
        return logdet_metric(z)

    hess = real_hessian(wrapped, x0, step=step)
    hxx = hess[:n, :n]
    hxy = hess[:n, n:]
    hyx = hess[n:, :n]
    hyy = hess[n:, n:]
    ricci = -0.25 * (hxx + hyy + 1j * (hxy - hyx))
    return 0.5 * (ricci + ricci.conjugate().T)


def scalar_curvature(params: Array, step: float = 2e-4) -> float:
    """Approximate scalar curvature of the induced metric."""

    g = induced_metric(params)
    ricci = ricci_tensor(params, step=step)
    return float(np.real(np.trace(np.linalg.solve(g, ricci))))


def metric_diagnostics(params: Array, include_curvature: bool = False) -> MetricDiagnostics:
    """Compute residuals and basic metric diagnostics at one local point."""

    z = embedding(params)
    residual = np.linalg.norm(equations(z))
    g = induced_metric(params)
    eigvals = np.linalg.eigvalsh(g)
    curvature = scalar_curvature(params) if include_curvature else None
    return MetricDiagnostics(
        residual_norm=float(residual),
        min_eigenvalue=float(eigvals[0]),
        max_eigenvalue=float(eigvals[-1]),
        logdet=float(np.sum(np.log(eigvals))),
        scalar_curvature=curvature,
    )


def random_parameters(n_points: int, seed: int = 0, scale: float = 0.35) -> Array:
    """Generate random local parameters away from the divisor r = 0."""

    rng = np.random.default_rng(seed)
    params = np.empty((n_points, 3), dtype=np.complex128)
    params[:, 0] = scale * (rng.normal(size=n_points) + 1j * rng.normal(size=n_points))
    params[:, 1] = scale * (rng.normal(size=n_points) + 1j * rng.normal(size=n_points))
    params[:, 2] = 1.0 + scale * (rng.normal(size=n_points) + 1j * rng.normal(size=n_points))

    too_small = np.abs(params[:, 2]) < 0.25
    while np.any(too_small):
        count = int(np.sum(too_small))
        params[too_small, 2] = 1.0 + scale * (rng.normal(size=count) + 1j * rng.normal(size=count))
        too_small = np.abs(params[:, 2]) < 0.25
    return params


def random_parameters_log_annulus(
    n_points: int,
    seed: int = 0,
    coordinate_scale: float = 0.3,
    r_min: float = 0.05,
    r_max: float = 1.5,
) -> Array:
    """Generate local parameters with log-uniform |r| and random r phase."""

    if r_min <= 0 or r_max <= r_min:
        raise ValueError("Expected 0 < r_min < r_max.")
    rng = np.random.default_rng(seed)
    params = np.empty((n_points, 3), dtype=np.complex128)
    params[:, 0] = coordinate_scale * (rng.normal(size=n_points) + 1j * rng.normal(size=n_points))
    params[:, 1] = coordinate_scale * (rng.normal(size=n_points) + 1j * rng.normal(size=n_points))
    log_radius = rng.uniform(np.log(r_min), np.log(r_max), size=n_points)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    params[:, 2] = np.exp(log_radius + 1j * phase)
    return params
