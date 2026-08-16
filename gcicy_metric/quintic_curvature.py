"""Local-coordinate curvature utilities for the Fermat quintic.

The cymetric point generator stores every point in an affine projective patch:
one homogeneous coordinate is exactly one, and one of the remaining
coordinates is eliminated with the quintic equation.  This module reconstructs
the same holomorphic chart and differentiates a metric without finite
differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FermatLocalChart:
    """A stable affine/implicit chart through one Fermat-quintic point."""

    patch_index: int
    dependent_index: int
    independent_indices: tuple[int, int, int]
    base_independent: np.ndarray
    exact_base_dependent: complex
    base_rhs: complex
    input_quintic_residual: float

    @classmethod
    def from_point(cls, point: np.ndarray) -> "FermatLocalChart":
        values = np.asarray(point, dtype=np.complex128)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError("a Fermat point must be a finite complex vector of length five")

        patch_candidates = np.flatnonzero(np.isclose(values, 1.0 + 0.0j, atol=2.0e-6))
        if len(patch_candidates) != 1:
            raise ValueError(
                "the cymetric affine convention requires exactly one coordinate equal to one"
            )
        patch_index = int(patch_candidates[0])
        available = [index for index in range(5) if index != patch_index]
        dependent_index = max(available, key=lambda index: abs(values[index]) ** 4)
        independent_indices = tuple(
            index for index in available if index != dependent_index
        )
        if len(independent_indices) != 3:
            raise AssertionError("a quintic threefold chart must have three coordinates")

        base_independent = values[list(independent_indices)]
        base_rhs = complex(-1.0 - np.sum(base_independent**5))
        original_dependent = complex(values[dependent_index])
        if abs(original_dependent) == 0:
            raise ValueError("the selected implicit coordinate has zero derivative")
        correction = np.exp(np.log(base_rhs / original_dependent**5) / 5.0)
        exact_base_dependent = original_dependent * correction
        return cls(
            patch_index=patch_index,
            dependent_index=dependent_index,
            independent_indices=independent_indices,
            base_independent=base_independent,
            exact_base_dependent=exact_base_dependent,
            base_rhs=base_rhs,
            input_quintic_residual=float(abs(np.sum(values**5))),
        )

    def real_coordinates(self, *, dtype=None, device=None):
        """Return ``(Re u, Im u)`` in the chart's three good coordinates."""

        import torch

        real_dtype = dtype or torch.float64
        coordinates = np.concatenate(
            (self.base_independent.real, self.base_independent.imag)
        )
        return torch.tensor(coordinates, dtype=real_dtype, device=device)

    def values_and_pullback(self, real_coordinates):
        """Evaluate the exact local branch and its holomorphic pullback.

        The fifth root is anchored at the supplied point.  Consequently this
        remains on the same local branch while avoiding any finite-difference
        branch selection.
        """

        import torch

        if real_coordinates.shape != (6,) or not real_coordinates.dtype.is_floating_point:
            raise ValueError("real coordinates must be a floating tensor of shape (6,)")
        u = torch.complex(real_coordinates[:3], real_coordinates[3:])
        complex_dtype = u.dtype
        device = u.device
        base_rhs = torch.tensor(self.base_rhs, dtype=complex_dtype, device=device)
        base_dependent = torch.tensor(
            self.exact_base_dependent, dtype=complex_dtype, device=device
        )
        rhs = -torch.ones((), dtype=complex_dtype, device=device) - torch.sum(u**5)
        dependent = base_dependent * torch.exp(torch.log(rhs / base_rhs) / 5.0)

        zero = torch.zeros((), dtype=complex_dtype, device=device)
        values: list[Any] = [zero for _ in range(5)]
        values[self.patch_index] = torch.ones((), dtype=complex_dtype, device=device)
        values[self.dependent_index] = dependent
        for local_index, ambient_index in enumerate(self.independent_indices):
            values[ambient_index] = u[local_index]
        homogeneous = torch.stack(values)

        zero_row = torch.zeros(3, dtype=complex_dtype, device=device)
        rows: list[Any] = [zero_row for _ in range(5)]
        rows[self.dependent_index] = -(u**4) / dependent**4
        for local_index, ambient_index in enumerate(self.independent_indices):
            row = torch.zeros(3, dtype=complex_dtype, device=device)
            row[local_index] = 1.0
            rows[ambient_index] = row
        pullback = torch.stack(rows)
        return homogeneous, pullback


def complex_hessian_from_real_hessian(real_hessian):
    """Convert a Hessian in ``(x_1,x_2,x_3,y_1,y_2,y_3)`` to ddbar."""

    import torch

    if real_hessian.shape != (6, 6):
        raise ValueError("the real Hessian must have shape (6, 6)")
    xx = real_hessian[:3, :3]
    xy = real_hessian[:3, 3:]
    yx = real_hessian[3:, :3]
    yy = real_hessian[3:, 3:]
    return 0.25 * (xx + yy + 1j * (xy - yx))


def metric_in_fermat_chart(model, chart: FermatLocalChart, real_coordinates):
    """Evaluate a positive tensor-network metric in one intrinsic chart."""

    values, pullback = chart.values_and_pullback(real_coordinates)
    _, metric = model.potential_and_metric(values[None, :], pullback[None, :, :])
    return metric[0]


def curvature_at_fermat_point(model, point: np.ndarray, *, vectorize: bool = False):
    """Compute exact-autodiff Ricci data at one stored Fermat point."""

    import torch

    chart = FermatLocalChart.from_point(point)
    parameter = next(model.parameters(), None)
    device = None if parameter is None else parameter.device
    coordinates = chart.real_coordinates(dtype=torch.float64, device=device)
    coordinates.requires_grad_(True)

    def log_determinant(current_coordinates):
        metric = metric_in_fermat_chart(model, chart, current_coordinates)
        _, log_absolute_determinant = torch.linalg.slogdet(metric)
        return log_absolute_determinant

    real_hessian = torch.autograd.functional.hessian(
        log_determinant,
        coordinates,
        create_graph=False,
        strict=False,
        vectorize=vectorize,
    )
    metric = metric_in_fermat_chart(model, chart, coordinates)
    # ``complex_hessian_from_real_hessian`` returns (holomorphic,
    # anti-holomorphic), whereas the TN backend stores metric rows as
    # anti-holomorphic and columns as holomorphic.  Put both tensors in the
    # same convention before contracting indices.
    ricci_tensor = complex_hessian_from_real_hessian(real_hessian).T
    ricci_tensor = 0.5 * (ricci_tensor + torch.conj(ricci_tensor.T))
    inverse_metric = torch.linalg.inv(metric)
    ricci_scalar = torch.real(torch.trace(inverse_metric @ ricci_tensor))

    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    inverse_square_root = (
        eigenvectors
        @ torch.diag(torch.rsqrt(eigenvalues)).to(metric.dtype)
        @ torch.conj(eigenvectors.T)
    )
    normalized_ricci = inverse_square_root @ ricci_tensor @ inverse_square_root
    ricci_tensor_norm = torch.sqrt(torch.sum(torch.abs(normalized_ricci) ** 2))
    determinant = torch.real(torch.linalg.det(metric))

    values, pullback = chart.values_and_pullback(coordinates)
    quintic_residual = torch.abs(torch.sum(values**5))
    tangent_residual = torch.max(
        torch.abs(torch.einsum("a,ai->i", 5.0 * values**4, pullback))
    )
    return {
        "chart": chart,
        "metric": metric.detach(),
        "determinant": float(determinant.detach().cpu()),
        "minimum_eigenvalue": float(torch.min(eigenvalues).detach().cpu()),
        "maximum_eigenvalue": float(torch.max(eigenvalues).detach().cpu()),
        "ricci_tensor": ricci_tensor.detach(),
        "ricci_scalar": float(ricci_scalar.detach().cpu()),
        "ricci_tensor_norm": float(ricci_tensor_norm.detach().cpu()),
        "corrected_quintic_residual": float(quintic_residual.detach().cpu()),
        "tangent_residual": float(tangent_residual.detach().cpu()),
    }
