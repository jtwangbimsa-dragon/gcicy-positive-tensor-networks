"""Matrix-free Gauss-Newton operators for model-normalized native MA E2.

The residual is

``sqrt(w_i) * (r_i - 1)``,

where ``r_i = exp(ell_i) / sum_j w_j exp(ell_j)`` and
``ell_i = log det(g_i) - log |Omega_i|^2``.  The normalization is part of
the differentiation graph.  In particular, this module does not freeze a
teacher volume or the baseline model's normalization.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch.func import functional_call, jvp, vjp


ParameterUnpacker = Callable[[torch.Tensor], Mapping[str, torch.Tensor]]


def hermitian_log_volume(metric: torch.Tensor) -> torch.Tensor:
    """Return ``log det(g)`` through a positive Hermitian Cholesky factor."""

    if metric.ndim != 3 or metric.shape[1] != metric.shape[2]:
        raise ValueError("metric must have shape (points, dimension, dimension)")
    factor = torch.linalg.cholesky(metric)
    diagonal = torch.real(torch.diagonal(factor, dim1=1, dim2=2))
    return 2.0 * torch.sum(torch.log(diagonal), dim=1)


class MatrixFreeNormalizedE2Jacobian:
    """JVP/VJP operator for the exactly normalized native E2 residual."""

    def __init__(
        self,
        model: torch.nn.Module,
        unpack_parameters: ParameterUnpacker,
        theta: torch.Tensor,
        dataset: dict[str, Any],
        *,
        chunk_size: int,
    ) -> None:
        if theta.ndim != 1:
            raise ValueError("the parameter vector must be one-dimensional")
        if chunk_size <= 0:
            raise ValueError("chunk size must be positive")
        required = {"count", "values", "derivatives", "weights", "log_omega"}
        missing = required - set(dataset)
        if missing:
            raise ValueError(f"native E2 dataset is missing keys: {sorted(missing)}")
        if dataset["count"] <= 0:
            raise ValueError("native E2 dataset must be non-empty")
        if dataset["weights"].shape != (dataset["count"],):
            raise ValueError("native E2 weights are misaligned")
        if not bool(
            torch.all(
                torch.isfinite(dataset["weights"])
                & (dataset["weights"] > 0)
            )
        ):
            raise ValueError("native E2 weights must be finite and positive")

        self.model = model
        self.unpack_parameters = unpack_parameters
        self.theta = theta
        self.dataset = dataset
        self.chunk_size = int(chunk_size)
        self._slices = tuple(
            slice(start, min(start + chunk_size, dataset["count"]))
            for start in range(0, dataset["count"], chunk_size)
        )
        weights = dataset["weights"]
        self.weights = (weights / torch.sum(weights)).detach()
        self.square_root_weights = torch.sqrt(self.weights)

        with torch.no_grad():
            self.base_raw = self.raw(theta).detach()
            self.base_ratio = self._normalized_ratio(self.base_raw).detach()
            self.base_residual = (
                self.square_root_weights * (self.base_ratio - 1.0)
            ).detach()
            self.base_probability = (
                self.weights * self.base_ratio
            ).detach()

    def _chunk_raw_function(
        self,
        selection: slice,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        values = self.dataset["values"][selection]
        derivatives = self.dataset["derivatives"][selection]
        log_omega = self.dataset["log_omega"][selection]

        def raw(parameter_vector: torch.Tensor) -> torch.Tensor:
            metric = functional_call(
                self.model,
                dict(self.unpack_parameters(parameter_vector)),
                (values, derivatives),
                strict=False,
            )
            return hermitian_log_volume(metric) - log_omega

        return raw

    def raw(self, parameter_vector: torch.Tensor) -> torch.Tensor:
        """Evaluate unnormalized log volume ratios in bounded chunks."""

        return torch.cat(
            [
                self._chunk_raw_function(selection)(parameter_vector)
                for selection in self._slices
            ]
        )

    def _normalized_ratio(self, raw: torch.Tensor) -> torch.Tensor:
        log_mean = torch.logsumexp(
            torch.log(self.weights) + raw,
            dim=0,
        )
        return torch.exp(raw - log_mean)

    def residual(
        self,
        parameter_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the exact normalized residual at a candidate parameter."""

        if parameter_vector is None:
            return self.base_residual
        raw = self.raw(parameter_vector)
        return self.square_root_weights * (self._normalized_ratio(raw) - 1.0)

    def jvp(self, direction: torch.Tensor) -> torch.Tensor:
        """Apply the normalized residual Jacobian to a parameter direction."""

        if direction.shape != self.theta.shape:
            raise ValueError("native E2 JVP direction has the wrong shape")
        raw_tangent = torch.cat(
            [
                jvp(
                    self._chunk_raw_function(selection),
                    (self.theta,),
                    (direction,),
                )[1]
                for selection in self._slices
            ]
        )
        normalization_tangent = torch.sum(
            self.base_probability * raw_tangent
        )
        return (
            self.square_root_weights
            * self.base_ratio
            * (raw_tangent - normalization_tangent)
        ).detach()

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint normalized residual Jacobian."""

        if cotangent.shape != (self.dataset["count"],):
            raise ValueError("native E2 VJP cotangent has the wrong shape")
        scaled = (
            cotangent
            * self.square_root_weights
            * self.base_ratio
        )
        raw_cotangent = (
            scaled
            - self.base_probability * torch.sum(scaled)
        )
        result = torch.zeros_like(self.theta)
        for selection in self._slices:
            function = self._chunk_raw_function(selection)
            _, pullback = vjp(function, self.theta)
            result = result + pullback(raw_cotangent[selection])[0]
        return result.detach()

    def normal(self, cotangent: torch.Tensor) -> torch.Tensor:
        """Apply ``J J^T`` in residual space."""

        return self.jvp(self.vjp(cotangent))

    @property
    def native_e2(self) -> float:
        return float(torch.dot(self.base_residual, self.base_residual))
