"""Positive residual tensor networks over an exact algebraic power lift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PositiveFeatureMoments:
    """A positive feature norm and its first and mixed derivatives."""

    norm: torch.Tensor
    holomorphic_gradient: torch.Tensor
    mixed_hessian: torch.Tensor


def dense_power_moments(
    hermitian: torch.Tensor,
    section_values: torch.Tensor,
    section_derivatives: torch.Tensor,
    *,
    power: int,
) -> PositiveFeatureMoments:
    """Return moments of ``(s^dagger H s)**power``."""

    if power <= 0:
        raise ValueError("the algebraic power must be positive")
    if (
        hermitian.ndim != 2
        or hermitian.shape[0] != hermitian.shape[1]
        or section_values.ndim != 2
        or section_values.shape[1] != hermitian.shape[0]
        or section_derivatives.ndim != 3
        or section_derivatives.shape[:2] != section_values.shape
    ):
        raise ValueError("dense power-lift inputs are not aligned")

    h_values = torch.einsum("ij,nj->ni", hermitian, section_values)
    h_derivatives = torch.einsum(
        "ij,nja->nia",
        hermitian,
        section_derivatives,
    )
    local_norm = torch.real(
        torch.einsum("ni,ni->n", torch.conj(section_values), h_values)
    )
    local_gradient = torch.einsum(
        "ni,nia->na",
        torch.conj(section_values),
        h_derivatives,
    )
    local_mixed = torch.einsum(
        "nia,nib->nab",
        torch.conj(section_derivatives),
        h_derivatives,
    )
    norm = torch.pow(local_norm, power)
    gradient = (
        power
        * torch.pow(local_norm, power - 1)[:, None]
        * local_gradient
    )
    mixed = (
        power
        * torch.pow(local_norm, power - 1)[:, None, None]
        * local_mixed
    )
    if power > 1:
        mixed = mixed + (
            power
            * (power - 1)
            * torch.pow(local_norm, power - 2)[:, None, None]
            * torch.conj(local_gradient)[:, :, None]
            * local_gradient[:, None, :]
        )
    return PositiveFeatureMoments(norm, gradient, mixed)


def _residual_moments(
    residual_model: torch.nn.Module,
    section_values: torch.Tensor,
    section_derivatives: torch.Tensor,
) -> PositiveFeatureMoments:
    if hasattr(residual_model, "feature_moments"):
        moments = residual_model.feature_moments(
            section_values,
            section_derivatives,
        )
    elif hasattr(residual_model, "learned_moments"):
        moments = residual_model.learned_moments(
            section_values,
            section_derivatives,
        )
    else:
        raise TypeError("residual model does not expose positive feature moments")
    return PositiveFeatureMoments(
        norm=moments.norm,
        holomorphic_gradient=moments.holomorphic_gradient,
        mixed_hessian=moments.mixed_hessian,
    )


class ExactPowerLiftResidualMetric(torch.nn.Module):
    """Add a positive trainable residual to an exact dense power lift.

    The represented positive function is

    ``F = F_base**power + scale * amplitude**2 * F_residual``.

    Setting ``amplitude`` to zero recovers the dense source metric exactly.
    The residual model is optimized only with the native geometric objective;
    the dense source is a fixed structured baseline, not a distillation target.
    """

    def __init__(
        self,
        baseline_h: np.ndarray,
        residual_model: torch.nn.Module,
        *,
        baseline_power: int,
        source_normalization: float,
        residual_scale: float = 1.0,
        residual_amplitude: float = 0.0,
        dtype: torch.dtype = torch.complex128,
        device: Any = None,
    ) -> None:
        super().__init__()
        baseline = np.asarray(baseline_h, dtype=np.complex128)
        baseline = 0.5 * (baseline + baseline.conj().T)
        eigenvalues = np.linalg.eigvalsh(baseline)
        if (
            baseline.ndim != 2
            or baseline.shape[0] != baseline.shape[1]
            or not np.all(np.isfinite(eigenvalues))
            or eigenvalues[0] <= 0
        ):
            raise ValueError("baseline H must be finite and positive definite")
        if baseline_power <= 0:
            raise ValueError("baseline power must be positive")
        if not np.isfinite(source_normalization) or source_normalization <= 0:
            raise ValueError("source normalization must be finite and positive")
        if not np.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual scale must be finite and positive")
        if not np.isfinite(residual_amplitude) or residual_amplitude < 0:
            raise ValueError("residual amplitude must be finite and non-negative")
        if dtype not in {torch.complex64, torch.complex128}:
            raise ValueError("metric dtype must be complex64 or complex128")

        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.residual_model = residual_model
        self.baseline_power = int(baseline_power)
        self.source_normalization = float(source_normalization)
        self.target_normalization = float(
            source_normalization / baseline_power
        )
        self.register_buffer(
            "baseline_h",
            torch.tensor(baseline, dtype=dtype, device=device),
        )
        self.register_buffer(
            "residual_scale",
            torch.tensor(residual_scale, dtype=real_dtype, device=device),
        )
        self.residual_amplitude = torch.nn.Parameter(
            torch.tensor(
                residual_amplitude,
                dtype=real_dtype,
                device=device,
            )
        )

    @property
    def residual_weight(self) -> torch.Tensor:
        return self.residual_scale * torch.square(self.residual_amplitude)

    @property
    def trainable_real_parameter_count(self) -> int:
        return int(
            sum(
                parameter.numel() * (2 if parameter.is_complex() else 1)
                for parameter in self.parameters()
            )
        )

    def set_residual_scale_(self, value: float) -> None:
        if not np.isfinite(value) or value <= 0:
            raise ValueError("residual scale must be finite and positive")
        with torch.no_grad():
            self.residual_scale.fill_(float(value))

    def set_residual_amplitude_(self, value: float) -> None:
        if not np.isfinite(value) or value < 0:
            raise ValueError("residual amplitude must be finite and non-negative")
        with torch.no_grad():
            self.residual_amplitude.fill_(float(value))

    def component_moments(
        self,
        residual_values: torch.Tensor,
        residual_derivatives: torch.Tensor,
        baseline_values: torch.Tensor,
        baseline_derivatives: torch.Tensor,
    ) -> tuple[PositiveFeatureMoments, PositiveFeatureMoments]:
        baseline = dense_power_moments(
            self.baseline_h,
            baseline_values,
            baseline_derivatives,
            power=self.baseline_power,
        )
        residual = _residual_moments(
            self.residual_model,
            residual_values,
            residual_derivatives,
        )
        return baseline, residual

    def residual_moments(
        self,
        residual_values: torch.Tensor,
        residual_derivatives: torch.Tensor,
    ) -> PositiveFeatureMoments:
        return _residual_moments(
            self.residual_model,
            residual_values,
            residual_derivatives,
        )

    def combine_moments(
        self,
        baseline: PositiveFeatureMoments,
        residual: PositiveFeatureMoments,
    ) -> PositiveFeatureMoments:
        weight = self.residual_weight
        return PositiveFeatureMoments(
            norm=baseline.norm + weight * residual.norm,
            holomorphic_gradient=(
                baseline.holomorphic_gradient
                + weight * residual.holomorphic_gradient
            ),
            mixed_hessian=(
                baseline.mixed_hessian
                + weight * residual.mixed_hessian
            ),
        )

    def moments(
        self,
        residual_values: torch.Tensor,
        residual_derivatives: torch.Tensor,
        baseline_values: torch.Tensor,
        baseline_derivatives: torch.Tensor,
    ) -> PositiveFeatureMoments:
        baseline, residual = self.component_moments(
            residual_values,
            residual_derivatives,
            baseline_values,
            baseline_derivatives,
        )
        return self.combine_moments(baseline, residual)

    def _potential_and_metric_from_moments(
        self,
        moments: PositiveFeatureMoments,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(torch.all(torch.isfinite(moments.norm) & (moments.norm > 0))):
            raise FloatingPointError("combined positive norm is invalid")
        inverse = torch.reciprocal(moments.norm)
        metric = moments.mixed_hessian * inverse[:, None, None]
        metric = metric - (
            torch.conj(moments.holomorphic_gradient)[:, :, None]
            * moments.holomorphic_gradient[:, None, :]
            * torch.square(inverse)[:, None, None]
        )
        metric = self.target_normalization * metric
        metric = 0.5 * (
            metric + torch.conj(torch.transpose(metric, 1, 2))
        )
        potential = self.target_normalization * torch.log(moments.norm)
        return potential, metric

    def potential_and_metric_from_cached_baseline(
        self,
        residual_values: torch.Tensor,
        residual_derivatives: torch.Tensor,
        baseline_norm: torch.Tensor,
        baseline_gradient: torch.Tensor,
        baseline_mixed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        baseline = PositiveFeatureMoments(
            baseline_norm,
            baseline_gradient,
            baseline_mixed,
        )
        residual = self.residual_moments(
            residual_values,
            residual_derivatives,
        )
        return self._potential_and_metric_from_moments(
            self.combine_moments(baseline, residual)
        )

    def potential_and_metric(
        self,
        residual_values: torch.Tensor,
        residual_derivatives: torch.Tensor,
        baseline_values: torch.Tensor,
        baseline_derivatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moments = self.moments(
            residual_values,
            residual_derivatives,
            baseline_values,
            baseline_derivatives,
        )
        return self._potential_and_metric_from_moments(moments)

    def forward(
        self,
        residual_values: torch.Tensor,
        residual_derivatives: torch.Tensor,
        baseline_values: torch.Tensor,
        baseline_derivatives: torch.Tensor,
    ) -> torch.Tensor:
        return self.potential_and_metric(
            residual_values,
            residual_derivatives,
            baseline_values,
            baseline_derivatives,
        )[1]


def calibrate_residual_scale(
    model: ExactPowerLiftResidualMetric,
    residual_values: torch.Tensor,
    residual_derivatives: torch.Tensor,
    baseline_values: torch.Tensor,
    baseline_derivatives: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Match weighted geometric means of baseline and residual norms."""

    with torch.no_grad():
        baseline, residual = model.component_moments(
            residual_values,
            residual_derivatives,
            baseline_values,
            baseline_derivatives,
        )
        if not bool(
            torch.all(baseline.norm > 0)
            and torch.all(residual.norm > 0)
            and torch.all(weights > 0)
        ):
            raise FloatingPointError("scale calibration received non-positive values")
        normalized = weights / torch.sum(weights)
        log_scale = torch.sum(
            normalized * (torch.log(baseline.norm) - torch.log(residual.norm))
        )
        value = float(torch.exp(log_scale).cpu())
    if not np.isfinite(value) or value <= 0:
        raise FloatingPointError("residual scale calibration failed")
    return value
