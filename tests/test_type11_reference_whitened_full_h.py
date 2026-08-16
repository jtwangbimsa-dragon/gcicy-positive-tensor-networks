import numpy as np
import pytest

from scripts.train_type11_k6_reference_whitened_full_h import (
    weighted_log_mean_exp,
)


def test_weighted_log_mean_exp_matches_direct_evaluation() -> None:
    values = np.asarray([-2.0, 0.5, 3.0], dtype=np.float64)
    weights = np.asarray([1.0, 4.0, 2.0], dtype=np.float64)

    expected = np.log(np.sum(weights * np.exp(values)) / np.sum(weights))

    assert weighted_log_mean_exp(values, weights) == pytest.approx(expected)


def test_weighted_log_mean_exp_is_stable_for_large_offsets() -> None:
    values = np.asarray([1000.0, 1001.0], dtype=np.float64)
    weights = np.asarray([2.0, 1.0], dtype=np.float64)

    shifted = np.asarray([0.0, 1.0], dtype=np.float64)
    expected = 1000.0 + np.log(
        np.sum(weights * np.exp(shifted)) / np.sum(weights)
    )

    assert weighted_log_mean_exp(values, weights) == pytest.approx(expected)


def test_weighted_log_mean_exp_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="weights must be nonnegative"):
        weighted_log_mean_exp(
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, -1.0]),
        )
