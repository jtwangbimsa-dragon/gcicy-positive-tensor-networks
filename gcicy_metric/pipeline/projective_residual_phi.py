"""Projectively invariant residual-potential metrics over a fixed H metric."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .exact_lift_tree import dense_h_potential_and_metric


class ProjectiveResidualPhiMetric(torch.nn.Module):
    """Learn ``g = g_H + ddbar(phi)`` from global projective moment features.

    The final linear layer is initialized to zero, so round zero is exactly the
    supplied algebraic H metric.  The feature matrix

    ``rho = z z^* / (z^* z)``

    is invariant under nonzero complex rescaling of homogeneous coordinates.
    """

    def __init__(
        self,
        reference_h: np.ndarray,
        *,
        normalization: float,
        ambient_dimension: int = 5,
        hidden_width: int = 144,
        hidden_layers: int = 3,
        potential_scale: float = 1.0,
        complex_dtype: torch.dtype = torch.complex64,
        device: Any = None,
    ) -> None:
        super().__init__()
        matrix = np.asarray(reference_h, dtype=np.complex128)
        matrix = 0.5 * (matrix + matrix.conj().T)
        eigenvalues = np.linalg.eigvalsh(matrix)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))
            or eigenvalues[0] <= 0
        ):
            raise ValueError("reference H must be finite and positive definite")
        if normalization <= 0 or not np.isfinite(normalization):
            raise ValueError("normalization must be finite and positive")
        if ambient_dimension <= 0 or hidden_width <= 0 or hidden_layers <= 0:
            raise ValueError("network dimensions must be positive")
        if potential_scale <= 0 or not np.isfinite(potential_scale):
            raise ValueError("potential scale must be finite and positive")
        if complex_dtype not in {torch.complex64, torch.complex128}:
            raise ValueError("complex dtype must be complex64 or complex128")

        real_dtype = (
            torch.float32
            if complex_dtype == torch.complex64
            else torch.float64
        )
        self.normalization = float(normalization)
        self.ambient_dimension = int(ambient_dimension)
        self.potential_scale = float(potential_scale)
        self.complex_dtype = complex_dtype
        self.real_dtype = real_dtype
        self.register_buffer(
            "reference_h",
            torch.tensor(matrix, dtype=complex_dtype, device=device),
        )

        feature_dimension = 2 * ambient_dimension * ambient_dimension
        layers: list[torch.nn.Module] = []
        input_dimension = feature_dimension
        for _ in range(hidden_layers):
            layers.append(
                torch.nn.Linear(
                    input_dimension,
                    hidden_width,
                    dtype=real_dtype,
                    device=device,
                )
            )
            layers.append(torch.nn.GELU())
            input_dimension = hidden_width
        output = torch.nn.Linear(
            input_dimension,
            1,
            dtype=real_dtype,
            device=device,
        )
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = torch.nn.Sequential(*layers)

    @property
    def trainable_real_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def projective_features(
        self,
        real_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if (
            real_coordinates.ndim != 2
            or real_coordinates.shape[1] != 2 * self.ambient_dimension
        ):
            raise ValueError("real homogeneous coordinates have the wrong shape")
        z = torch.complex(
            real_coordinates[:, : self.ambient_dimension],
            real_coordinates[:, self.ambient_dimension :],
        )
        denominator = torch.sum(torch.square(torch.abs(z)), dim=1)
        denominator = torch.clamp(
            denominator,
            min=torch.finfo(real_coordinates.dtype).tiny,
        )
        moment = (
            z[:, :, None] * torch.conj(z[:, None, :])
            / denominator[:, None, None]
        )
        return torch.cat(
            (
                moment.real.reshape(len(moment), -1),
                moment.imag.reshape(len(moment), -1),
            ),
            dim=1,
        )

    def phi(self, real_coordinates: torch.Tensor) -> torch.Tensor:
        return self.potential_scale * self.network(
            self.projective_features(real_coordinates)
        ).squeeze(-1)

    def _phi_single(self, real_coordinate: torch.Tensor) -> torch.Tensor:
        return self.phi(real_coordinate[None])[0]

    def ambient_mixed_hessian(
        self,
        real_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``d_z d_zbar phi`` with anti-holomorphic index first.

        Metrics in this repository are stored as ``g[abar, b]``.  For
        ``z = x + i y``, this convention gives

        ``g[abar, b] = (H_xx + H_yy + i (H_yx - H_xy))[a, b] / 4``.
        """
        if not bool(torch.all(torch.isfinite(real_coordinates))):
            raise ValueError("homogeneous coordinates must be finite")
        coordinate_norm = torch.sum(torch.square(real_coordinates), dim=1)
        if not bool(torch.all(coordinate_norm > 0)):
            raise ValueError("homogeneous coordinates must be nonzero")
        real_hessian = torch.vmap(torch.func.hessian(self._phi_single))(
            real_coordinates
        )
        dimension = self.ambient_dimension
        xx = real_hessian[:, :dimension, :dimension]
        xy = real_hessian[:, :dimension, dimension:]
        yx = real_hessian[:, dimension:, :dimension]
        yy = real_hessian[:, dimension:, dimension:]
        mixed = 0.25 * torch.complex(xx + yy, yx - xy)
        return 0.5 * (mixed + torch.conj(torch.transpose(mixed, 1, 2)))

    def potential_and_metric(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
        real_coordinates: torch.Tensor,
        pullbacks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pullbacks.ndim != 3 or pullbacks.shape[2] != self.ambient_dimension:
            raise ValueError("pullbacks have the wrong ambient dimension")
        base_potential, base_metric = dense_h_potential_and_metric(
            self.reference_h,
            section_values,
            section_derivatives,
            normalization=self.normalization,
        )
        ambient = self.ambient_mixed_hessian(real_coordinates)
        correction = torch.einsum(
            "nai,nij,nbj->nab",
            torch.conj(pullbacks),
            ambient,
            pullbacks,
        )
        metric = base_metric + correction
        metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        return base_potential + self.phi(real_coordinates), metric

    def forward(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
        real_coordinates: torch.Tensor,
        pullbacks: torch.Tensor,
    ) -> torch.Tensor:
        return self.potential_and_metric(
            section_values,
            section_derivatives,
            real_coordinates,
            pullbacks,
        )[1]
