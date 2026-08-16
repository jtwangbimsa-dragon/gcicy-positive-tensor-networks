from __future__ import annotations

import math

import numpy as np
import torch

from gcicy_metric.fermat_quintic import (
    fermat_quintic_quotient_basis,
    fermat_quotient_fubini_study_h,
)
from gcicy_metric.pipeline.algebraic_power_lift import AlgebraicMetricPowerLift
from gcicy_metric.pipeline.fermat_full_h import FermatSymmetricFullH


def test_f10_cubed_is_an_exact_degree_thirty_metric_lift() -> None:
    ambient, exponents, lift = fermat_quintic_quotient_basis(10)
    source = FermatSymmetricFullH(
        exponents,
        fermat_quotient_fubini_study_h(ambient, lift),
        normalization=1.0 / (10.0 * math.pi),
        device=torch.device("cpu"),
    )
    teacher = AlgebraicMetricPowerLift(
        source, source_degree=10, target_degree=30
    )
    rng = np.random.default_rng(202607257)
    with torch.no_grad():
        source.coordinates.add_(
            torch.tensor(
                rng.normal(scale=0.05, size=source.coordinates.shape),
                dtype=torch.float64,
            )
        )
    values = torch.tensor(
        rng.normal(size=(7, len(exponents)))
        + 1j * rng.normal(size=(7, len(exponents))),
        dtype=torch.complex128,
    )
    derivatives = torch.tensor(
        rng.normal(size=(7, len(exponents), 3))
        + 1j * rng.normal(size=(7, len(exponents), 3)),
        dtype=torch.complex128,
    )

    source_potential, source_metric = source.potential_and_metric(
        values, derivatives
    )
    target_log_feature, target_metric = teacher.log_feature_and_metric(
        values, derivatives
    )
    target_potential, target_metric_again = teacher.potential_and_metric(
        values, derivatives
    )

    torch.testing.assert_close(
        target_log_feature,
        3.0 * source_potential / source.normalization,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        target_potential, source_potential, rtol=2.0e-13, atol=2.0e-13
    )
    torch.testing.assert_close(
        target_metric, source_metric, rtol=2.0e-13, atol=2.0e-13
    )
    torch.testing.assert_close(
        target_metric_again, source_metric, rtol=2.0e-13, atol=2.0e-13
    )
