"""Section/H-matrix style Kahler metrics on the explicit local gCICY patch.

This module implements a fast diagonal section ansatz

    K = log(sum_m exp(theta_m) |s_m(s,t,r)|^2)

where the local sections are Laurent monomials

    s_m = s**a_m t**b_m r**c_m.

For this diagonal H-matrix model the Hermitian Hessian can be evaluated by a
weighted covariance formula, avoiding the slow per-point autograd Hessian used
in early experiments.
"""

from __future__ import annotations

import numpy as np

from .ricci_flat_fit import apply_correction, baseline_metrics, basis_hessians
from .global_sections import global_h_metrics

Array = np.ndarray


def laurent_exponents(
    *,
    max_weight: int = 4,
    max_st_degree: int | None = None,
    min_r_power: int = -3,
    max_r_power: int = 3,
    include_constant: bool = True,
) -> Array:
    """Return Laurent monomial exponents for ``s**a t**b r**c``.

    The basis keeps non-negative powers in ``s`` and ``t`` and allows bounded
    Laurent powers in ``r``.  The default weight cutoff
    ``a + b + abs(c) <= max_weight`` gives a compact local proxy for pullbacks
    of ambient projective sections after solving the gCICY constraints.
    """

    if max_weight < 0:
        raise ValueError("max_weight must be non-negative.")
    if max_st_degree is None:
        max_st_degree = max_weight
    if max_st_degree < 0:
        raise ValueError("max_st_degree must be non-negative.")
    if min_r_power > max_r_power:
        raise ValueError("Expected min_r_power <= max_r_power.")

    exponents: list[tuple[int, int, int]] = []
    for a in range(max_st_degree + 1):
        for b in range(max_st_degree + 1 - a):
            for c in range(min_r_power, max_r_power + 1):
                if a + b + abs(c) > max_weight:
                    continue
                if (a, b, c) == (0, 0, 0) and not include_constant:
                    continue
                exponents.append((a, b, c))
    exponents.sort(key=lambda item: (item[0] + item[1] + abs(item[2]), item[0], item[1], item[2]))
    return np.asarray(exponents, dtype=np.int64)


def _softmax(values: Array) -> Array:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def section_metric(
    params: Array,
    exponents: Array,
    log_weights: Array | None = None,
    *,
    epsilon: float = 1e-14,
    normalization: float = 1.0,
) -> Array:
    """Evaluate the diagonal section metric at one local point.

    If ``u_m = exp(theta_m) |s**a t**b r**c|^2`` and
    ``p_m = u_m / sum_l u_l``, then away from coordinate zeros

        partial partialbar log(sum_m u_m)

    is the covariance of ``(a/s, b/t, c/r)`` under the probability weights
    ``p_m``.  The small ``epsilon`` only guards floating-point division when a
    sampled coordinate is numerically zero.
    """

    z = np.asarray(params, dtype=np.complex128).reshape(-1)
    if z.shape != (3,):
        raise ValueError(f"Expected one vector of 3 complex parameters, got {z.shape}.")
    q = np.asarray(exponents, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"Expected exponents with shape (n, 3), got {q.shape}.")
    theta = np.zeros(q.shape[0], dtype=float) if log_weights is None else np.asarray(log_weights, dtype=float)
    if theta.shape != (q.shape[0],):
        raise ValueError(f"Expected {q.shape[0]} log weights, got {theta.shape}.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if normalization <= 0:
        raise ValueError("normalization must be positive.")

    abs_z = np.maximum(np.abs(z), epsilon)
    logits = theta + 2.0 * (q @ np.log(abs_z))
    probabilities = _softmax(logits)
    safe_z = np.where(np.abs(z) > epsilon, z, epsilon + 0.0j)
    logarithmic_derivatives = q / safe_z[None, :]
    mean = probabilities @ logarithmic_derivatives
    second = np.einsum(
        "m,mi,mj->ij",
        probabilities,
        logarithmic_derivatives,
        np.conjugate(logarithmic_derivatives),
    )
    metric = normalization * (second - np.outer(mean, np.conjugate(mean)))
    return 0.5 * (metric + metric.conjugate().T)


def section_metrics(
    params: Array,
    exponents: Array,
    log_weights: Array | None = None,
    *,
    epsilon: float = 1e-14,
    normalization: float = 1.0,
) -> Array:
    """Evaluate the diagonal section metric on a batch of local points."""

    points = np.asarray(params, dtype=np.complex128)
    if points.ndim == 1:
        points = points[None, :]
    return np.asarray(
        [
            section_metric(
                point,
                exponents,
                log_weights,
                epsilon=epsilon,
                normalization=normalization,
            )
            for point in points
        ],
        dtype=np.complex128,
    )


def artifact_metrics(params: Array, artifact) -> Array:
    """Evaluate a saved metric artifact on new local points.

    Existing polynomial-correction artifacts contain ``coefficients`` and
    ``exponents``.  Section/H-matrix artifacts contain ``section_theta`` and
    ``section_exponents``.  Keeping this dispatch here lets all verification
    scripts use the same evaluation path.
    """

    files = set(artifact.files)
    if {"global_section_exponents", "global_h_matrix"}.issubset(files):
        normalization = float(artifact["global_section_normalization"]) if "global_section_normalization" in files else 1.0
        return global_h_metrics(
            params,
            artifact["global_section_exponents"],
            artifact["global_h_matrix"],
            normalization=normalization,
        )
    if {"section_theta", "section_exponents"}.issubset(files):
        normalization = float(artifact["section_normalization"]) if "section_normalization" in files else 1.0
        return section_metrics(
            params,
            artifact["section_exponents"],
            artifact["section_theta"],
            normalization=normalization,
        )
    if {"coefficients", "exponents"}.issubset(files):
        base = baseline_metrics(params)
        return apply_correction(base, basis_hessians(params, artifact["exponents"]), artifact["coefficients"])
    raise ValueError("Unrecognized metric artifact format.")
