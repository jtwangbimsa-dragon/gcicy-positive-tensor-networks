"""Exact degree lifts of positive algebraic metrics by integer powers."""

from __future__ import annotations

import math

import torch


class AlgebraicMetricPowerLift(torch.nn.Module):
    """View ``F_k`` as ``F_k^p`` at degree ``p k`` without materializing H."""

    def __init__(
        self,
        source_model: torch.nn.Module,
        *,
        source_degree: int,
        target_degree: int,
    ) -> None:
        super().__init__()
        if source_degree <= 0 or target_degree <= 0:
            raise ValueError("algebraic degrees must be positive")
        if target_degree % source_degree:
            raise ValueError("the target degree must be an integer source-degree multiple")
        observed_normalization = float(getattr(source_model, "normalization"))
        if not math.isfinite(observed_normalization) or observed_normalization <= 0:
            raise ValueError("source model normalization must be finite and positive")
        self.source_model = source_model
        self.source_degree = int(source_degree)
        self.target_degree = int(target_degree)
        self.power = int(target_degree // source_degree)
        # If K = a log(F), then F -> F**p preserves K only when the target
        # normalization is a/p.  This covers both 1/k and 1/(pi k)
        # conventions without hard-coding either one.
        self.normalization = observed_normalization / self.power

    def source_log_feature(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> torch.Tensor:
        potential, _ = self.source_model.potential_and_metric(
            section_values, section_derivatives
        )
        return potential / float(self.source_model.normalization)

    def log_feature_and_metric(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        potential, metric = self.source_model.potential_and_metric(
            section_values, section_derivatives
        )
        source_log_feature = potential / float(self.source_model.normalization)
        return self.power * source_log_feature, metric

    def potential_and_metric(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_feature, metric = self.log_feature_and_metric(
            section_values, section_derivatives
        )
        return self.normalization * log_feature, metric

    def forward(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> torch.Tensor:
        return self.potential_and_metric(section_values, section_derivatives)[1]
