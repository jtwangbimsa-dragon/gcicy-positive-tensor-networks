from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_quintic_positive_tensor_network_equivalence as audit


def _payload(*, sites: int = 20, precision: str = "complex64") -> dict:
    return {
        "schema": audit.MODEL_SCHEMA,
        "architecture": "shared_local_dictionary",
        "source_degree": 1,
        "site_count": sites,
        "precision": precision,
    }


def test_model_pair_requires_explicit_k_and_precision_exceptions() -> None:
    with pytest.raises(ValueError, match="different site counts"):
        audit.validate_model_pair(
            _payload(),
            _payload(sites=40),
            allow_different_site_counts=False,
            allow_different_precisions=False,
        )
    with pytest.raises(ValueError, match="different precisions"):
        audit.validate_model_pair(
            _payload(),
            _payload(precision="complex128"),
            allow_different_site_counts=False,
            allow_different_precisions=False,
        )
    audit.validate_model_pair(
        _payload(),
        _payload(sites=40, precision="complex128"),
        allow_different_site_counts=True,
        allow_different_precisions=True,
    )


def test_statistics_use_pointwise_frobenius_relative_error() -> None:
    potentials_a = np.array([1.0, 2.0])
    potentials_b = np.array([1.0, 2.25])
    metrics_a = np.stack((np.eye(2), 2.0 * np.eye(2))).astype(np.complex128)
    metrics_b = metrics_a.copy()
    metrics_b[1, 0, 0] += 0.2
    result = audit.comparison_statistics(
        potentials_a, potentials_b, metrics_a, metrics_b
    )
    assert result["maximum_absolute_potential_difference"] == pytest.approx(0.25)
    assert result["maximum_absolute_metric_frobenius_difference"] == pytest.approx(0.2)
    assert result["maximum_relative_metric_frobenius_difference"] == pytest.approx(
        0.2 / np.sqrt(8.0)
    )


def test_claims_do_not_conflate_learned_norm_with_full_floor_metric() -> None:
    model_a_sha = "a" * 64
    payload_b = _payload(sites=40)
    payload_b.update(
        {
            "site_transfer_source_model_sha256": model_a_sha,
            "site_transfer_rule": "exact learned-norm MPS repetition with bridges",
        }
    )
    claims = audit.construction_claims(_payload(), payload_b, model_a_sha)
    assert claims["learned_norm_exact_by_construction"] is True
    assert claims["full_metric_audited_with_positive_floor"] is True
    assert "not asserted" in claims["full_metric_claim"]
