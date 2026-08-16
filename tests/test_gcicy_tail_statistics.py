from __future__ import annotations

import numpy as np
import pytest

from gcicy_metric.pipeline.tail import (
    bilateral_tail_metrics,
    paired_cluster_bootstrap,
)


def test_bilateral_tail_metrics_vanish_for_constant_log_eta() -> None:
    values = np.full(12, 2.5)
    weights = np.arange(1.0, 13.0)
    clusters = np.repeat(np.arange(3), 4)

    metrics = bilateral_tail_metrics(values, weights, clusters)

    assert metrics["sigma"] == pytest.approx(0.0, abs=1.0e-14)
    assert metrics["chi"] == pytest.approx(0.0, abs=1.0e-14)
    assert metrics["absolute_log_ratio_q999"] == pytest.approx(0.0, abs=1.0e-14)
    assert metrics["upper_log_ratio_cvar_1pct"] == pytest.approx(0.0, abs=1.0e-14)
    assert metrics["lower_log_ratio_cvar_1pct"] == pytest.approx(0.0, abs=1.0e-14)
    assert metrics["normalized_ratio_min"] == pytest.approx(1.0)
    assert metrics["normalized_ratio_max"] == pytest.approx(1.0)
    assert metrics["fibre_cluster_count"] == 3
    assert metrics["fibre_cluster_size_min"] == 4
    assert metrics["fibre_cluster_size_max"] == 4


def test_bilateral_tail_metrics_detect_both_sides() -> None:
    values = np.asarray([-4.0, -0.1, 0.0, 0.2, 3.0])
    weights = np.ones_like(values)

    metrics = bilateral_tail_metrics(values, weights)

    assert metrics["upper_log_ratio_q999"] > 0.0
    assert metrics["lower_log_ratio_q999"] > 0.0
    assert metrics["absolute_log_ratio_q999"] >= max(
        metrics["upper_log_ratio_q999"],
        metrics["lower_log_ratio_q999"],
    )
    assert metrics["normalized_ratio_min"] < 1.0 / 3.0
    assert metrics["normalized_ratio_max"] > 3.0
    assert metrics["normalized_ratio_below_one_third_point_count"] > 0
    assert metrics["normalized_ratio_above_3_point_count"] > 0


def test_paired_cluster_bootstrap_is_zero_for_identical_models() -> None:
    values = np.linspace(-0.5, 0.5, 24)
    weights = np.linspace(1.0, 2.0, 24)
    clusters = np.repeat(np.arange(6), 4)

    result = paired_cluster_bootstrap(
        values,
        values.copy(),
        weights,
        clusters,
        replicates=20,
        seed=7,
    )

    for comparison in result["comparisons"].values():
        assert comparison["improvement"] == pytest.approx(0.0, abs=1.0e-14)
        assert comparison["bootstrap_mean_improvement"] == pytest.approx(
            0.0, abs=1.0e-14
        )
        assert comparison["bootstrap_95pct_confidence_interval"] == pytest.approx(
            [0.0, 0.0], abs=1.0e-14
        )


def test_paired_cluster_bootstrap_prefers_constant_candidate() -> None:
    baseline = np.tile(np.asarray([-1.0, -0.2, 0.2, 1.0]), 8)
    candidate = np.zeros_like(baseline)
    weights = np.ones_like(baseline)
    clusters = np.repeat(np.arange(8), 4)

    result = paired_cluster_bootstrap(
        baseline,
        candidate,
        weights,
        clusters,
        replicates=30,
        seed=11,
    )

    assert result["comparisons"]["sigma"]["improvement"] > 0.0
    assert result["comparisons"]["chi"]["improvement"] > 0.0
    assert result["comparisons"]["absolute_log_ratio_q999"]["improvement"] > 0.0
    assert result["comparisons"]["sigma"]["bootstrap_win_probability"] == 1.0
