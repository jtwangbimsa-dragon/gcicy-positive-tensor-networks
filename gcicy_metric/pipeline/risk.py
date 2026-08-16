"""Weighted normalization and tail-risk primitives used by metric training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WeightedCVARAudit:
    value: float
    var_threshold: float
    selected_tail_mass: float
    strict_tail_mass: float
    boundary_mass: float
    selected_boundary_mass: float


def normalized_positive_weights_numpy(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.min(values) <= 0:
        raise ValueError("weights must be finite and positive")
    return values / np.sum(values)


def weighted_log_mean_exp_numpy(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    raw = np.asarray(values, dtype=np.float64)
    probabilities = normalized_positive_weights_numpy(weights)
    if raw.shape != probabilities.shape or not np.all(np.isfinite(raw)):
        raise ValueError("values and weights must be finite vectors of equal shape")
    maximum = float(np.max(raw))
    return float(maximum + np.log(np.sum(probabilities * np.exp(raw - maximum))))


def weighted_cvar_numpy(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    tail_fraction: float,
) -> WeightedCVARAudit:
    """Return upper-tail weighted CVaR with a fractional boundary point."""

    losses = np.asarray(values, dtype=np.float64)
    probabilities = normalized_positive_weights_numpy(weights)
    if losses.shape != probabilities.shape or not np.all(np.isfinite(losses)):
        raise ValueError("values and weights must be finite vectors of equal shape")
    if not np.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")

    order = np.argsort(-losses, kind="stable")
    sorted_losses = losses[order]
    sorted_probabilities = probabilities[order]
    cumulative = np.cumsum(sorted_probabilities)
    boundary_index = min(
        int(np.searchsorted(cumulative, tail_fraction, side="left")),
        len(sorted_losses) - 1,
    )
    mass_before = float(cumulative[boundary_index - 1]) if boundary_index else 0.0
    selected_boundary_mass = float(tail_fraction - mass_before)
    selected = np.zeros_like(sorted_probabilities)
    selected[:boundary_index] = sorted_probabilities[:boundary_index]
    selected[boundary_index] = selected_boundary_mass
    threshold = float(sorted_losses[boundary_index])
    strict_tail_mass = float(np.sum(probabilities[losses > threshold]))
    boundary_mass = float(np.sum(probabilities[losses == threshold]))
    return WeightedCVARAudit(
        value=float(np.sum(selected * sorted_losses) / tail_fraction),
        var_threshold=threshold,
        selected_tail_mass=float(np.sum(selected)),
        strict_tail_mass=strict_tail_mass,
        boundary_mass=boundary_mass,
        selected_boundary_mass=selected_boundary_mass,
    )


def weighted_log_mean_exp_torch(values: Any, weights: Any) -> Any:
    import torch

    positive = torch.clamp(weights, min=torch.finfo(weights.dtype).tiny)
    return torch.logsumexp(torch.log(positive) + values, dim=0) - torch.log(
        torch.sum(positive)
    )


def smooth_upper_log_ratio_excess_torch(
    log_ratio: Any,
    *,
    ratio_threshold: float,
    smooth_temperature: float,
) -> Any:
    """Return a stable smooth penalty for normalized ratios above a threshold."""

    import torch

    if not np.isfinite(ratio_threshold) or ratio_threshold <= 1.0:
        raise ValueError("ratio_threshold must be greater than one")
    if not np.isfinite(smooth_temperature) or smooth_temperature <= 0.0:
        raise ValueError("smooth_temperature must be positive")
    log_threshold = float(np.log(ratio_threshold))
    return (
        torch.nn.functional.softplus(
            (log_ratio - log_threshold) / smooth_temperature
        )
        * smooth_temperature
    )


def weighted_cvar_torch(
    values: Any,
    weights: Any,
    *,
    tail_fraction: float,
) -> Any:
    """Differentiable weighted-sort CVaR away from ordering ties."""

    import torch

    if not np.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    positive = torch.clamp(weights, min=torch.finfo(weights.dtype).tiny)
    probabilities = positive / torch.sum(positive)
    sorted_values, order = torch.sort(values, descending=True)
    sorted_weights = probabilities[order]
    cumulative_mass = torch.cumsum(sorted_weights, dim=0)
    mass_before = cumulative_mass - sorted_weights
    remaining_mass = torch.clamp(tail_fraction - mass_before, min=0.0)
    selected_mass = torch.minimum(sorted_weights, remaining_mass)
    return torch.sum(selected_mass * sorted_values) / tail_fraction


def weighted_cvar_fixed_threshold_linearization_torch(
    values: Any,
    weights: Any,
    *,
    tail_fraction: float,
    var_threshold: float,
    boundary_selection_fraction: float,
    total_weight: float | None = None,
) -> Any:
    """CVaR tail linearization with the weighted boundary point included."""

    import torch

    if not np.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if (
        not np.isfinite(boundary_selection_fraction)
        or not 0.0 <= boundary_selection_fraction <= 1.0
    ):
        raise ValueError("boundary_selection_fraction must lie in [0, 1]")
    positive = torch.clamp(weights, min=torch.finfo(weights.dtype).tiny)
    denominator = (
        torch.sum(positive)
        if total_weight is None
        else torch.as_tensor(total_weight, dtype=weights.dtype, device=weights.device)
    )
    selected = (values > var_threshold).to(values.dtype) + (
        boundary_selection_fraction * (values == var_threshold).to(values.dtype)
    )
    return weighted_cvar_selected_linearization_torch(
        values,
        positive,
        selected,
        tail_fraction=tail_fraction,
        total_weight=denominator,
    )


def weighted_cvar_selected_linearization_torch(
    values: Any,
    weights: Any,
    selected_fractions: Any,
    *,
    tail_fraction: float,
    total_weight: Any | None = None,
) -> Any:
    """Linearize CVaR using a frozen fractional tail selection."""

    import torch

    if not np.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if values.shape != weights.shape or values.shape != selected_fractions.shape:
        raise ValueError("values, weights, and selected_fractions must align")
    positive = torch.clamp(weights, min=torch.finfo(weights.dtype).tiny)
    denominator = (
        torch.sum(positive)
        if total_weight is None
        else torch.as_tensor(total_weight, dtype=weights.dtype, device=weights.device)
    )
    return torch.sum(positive * selected_fractions.detach() * values) / (
        tail_fraction * denominator
    )
