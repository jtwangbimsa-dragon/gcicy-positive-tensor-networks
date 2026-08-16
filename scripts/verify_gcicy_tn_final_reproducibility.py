#!/usr/bin/env python3
"""Recompute the paper's principal numerical claims from pointwise arrays.

This verifier is deliberately independent of the scripts that generated the
LaTeX tables.  It starts from saved log Monge--Ampere ratios, importance
weights, fibre identifiers and metric eigenvalues, then reconstructs the
reported bulk and tail statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
DEFAULT_ROOT = (
    SCRIPT.parent
    if (SCRIPT.parent / "outputs").is_dir()
    else SCRIPT.parents[1]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="repository or unpacked bundle root",
    )
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        help="override the final manuscript directory",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def close(
    observed: float,
    expected: float,
    *,
    relative_tolerance: float = 2.0e-12,
    absolute_tolerance: float = 2.0e-13,
) -> None:
    if not math.isclose(
        float(observed),
        float(expected),
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise AssertionError(
            f"expected {float(expected):.17g}, observed {float(observed):.17g}"
        )


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    threshold = probability * cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, threshold, side="left")),
        len(order) - 1,
    )
    return float(values[order[index]])


def weighted_upper_tail_mean(
    values: np.ndarray,
    weights: np.ndarray,
    tail_fraction: float,
) -> float:
    normalized = weights / np.sum(weights)
    order = np.argsort(values, kind="stable")[::-1]
    sorted_values = values[order]
    sorted_weights = normalized[order]
    mass_before = np.cumsum(sorted_weights) - sorted_weights
    selected = np.minimum(
        sorted_weights,
        np.maximum(tail_fraction - mass_before, 0.0),
    )
    return float(np.sum(selected * sorted_values) / tail_fraction)


def pointwise_metrics(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["model_log_eta"], dtype=np.float64).reshape(-1)
        weights = np.asarray(data["importance_weights"], dtype=np.float64).reshape(-1)
        clusters = np.asarray(data["sampling_cluster_ids"]).reshape(-1)
        eigenvalues = (
            np.asarray(data["metric_minimum_eigenvalues"], dtype=np.float64)
            if "metric_minimum_eigenvalues" in data.files
            else None
        )
    if raw.shape != weights.shape or raw.shape != clusters.shape:
        raise AssertionError(f"unaligned pointwise arrays: {path}")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(weights)):
        raise AssertionError(f"nonfinite pointwise arrays: {path}")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise AssertionError(f"invalid importance weights: {path}")

    normalized_weights = weights / np.sum(weights)
    maximum = float(np.max(raw))
    log_normalization = maximum + float(
        np.log(np.sum(normalized_weights * np.exp(raw - maximum)))
    )
    log_ratio = raw - log_normalization
    ratio = np.exp(log_ratio)
    residual = ratio - 1.0
    absolute_log = np.abs(log_ratio)
    squared_energy = float(np.sum(normalized_weights * residual**2))
    cluster_values, inverse = np.unique(clusters, return_inverse=True)
    cluster_weights = np.bincount(inverse, weights=weights)

    result: dict[str, Any] = {
        "point_count": int(raw.size),
        "sigma": float(np.sum(normalized_weights * np.abs(residual))),
        "chi": float(np.sqrt(squared_energy)),
        "squared_energy": squared_energy,
        "absolute_log_ratio_q999": weighted_quantile(
            absolute_log, weights, 0.999
        ),
        "absolute_log_ratio_cvar_1pct": weighted_upper_tail_mean(
            absolute_log, weights, 0.01
        ),
        "maximum_absolute_log_ratio": float(np.max(absolute_log)),
        "normalization_log_kappa": log_normalization,
        "importance_effective_sample_size": float(
            np.sum(weights) ** 2 / np.sum(weights**2)
        ),
        "fibre_cluster_count": int(cluster_values.size),
        "fibre_cluster_effective_sample_size": float(
            np.sum(cluster_weights) ** 2 / np.sum(cluster_weights**2)
        ),
        "normalized_ratio_min": float(np.min(ratio)),
        "normalized_ratio_max": float(np.max(ratio)),
    }
    if eigenvalues is not None:
        if eigenvalues.shape != raw.shape or not np.all(np.isfinite(eigenvalues)):
            raise AssertionError(f"invalid eigenvalue evidence: {path}")
        result["minimum_metric_eigenvalue"] = float(np.min(eigenvalues))
        result["nonpositive_metric_count"] = int(np.count_nonzero(eigenvalues <= 0.0))
    return result


def aligned_arrays(paths: list[Path]) -> None:
    reference_weights = None
    reference_clusters = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            weights = np.asarray(data["importance_weights"], dtype=np.float64)
            clusters = np.asarray(data["sampling_cluster_ids"])
        normalized = weights / np.sum(weights)
        if reference_weights is None:
            reference_weights = normalized
            reference_clusters = clusters
            continue
        if not np.allclose(reference_weights, normalized, rtol=1.0e-13, atol=1.0e-15):
            raise AssertionError(f"importance weights are not point-aligned: {path}")
        if not np.array_equal(reference_clusters, clusters):
            raise AssertionError(f"fibre identifiers are not point-aligned: {path}")


def compare_metric_dict(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in (
        "sigma",
        "chi",
        "absolute_log_ratio_q999",
        "absolute_log_ratio_cvar_1pct",
        "maximum_absolute_log_ratio",
    ):
        if key in expected:
            close(observed[key], expected[key])
    if expected.get("minimum_metric_eigenvalue") is not None:
        close(observed["minimum_metric_eigenvalue"], expected["minimum_metric_eigenvalue"])
        assert observed["nonpositive_metric_count"] == expected.get(
            "nonpositive_metric_count", 0
        )


def check_x11(root: Path) -> dict[str, Any]:
    base = Path("outputs/pipeline/type11_x11_equal_time_final_20260807")
    summary = load_json(root, f"{base}/final_summary.json")
    assert summary["point_count"] == 200000
    assert summary["replicate_count"] == 3
    models = {
        "positive_TN": "tn_arrays.npz",
        "source_density_residual_phi": "source_density_phi/blind_tail_arrays.npz",
        "cold_full_H6_equal_time_plateau": "full_h6_equal_time_plateau_arrays.npz",
    }
    reconstructed: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    for replicate in (1, 2, 3):
        directory = root / base / f"replicate_{replicate}"
        paths = [directory / relative for relative in models.values()]
        aligned_arrays(paths)
        row = summary["rows"][replicate - 1]
        for model, relative in models.items():
            metrics = pointwise_metrics(directory / relative)
            reconstructed[model].append(metrics)
            compare_metric_dict(metrics, row["models"][model])

    aggregate: dict[str, Any] = {}
    for model, rows in reconstructed.items():
        aggregate[model] = {}
        for metric in (
            "sigma",
            "chi",
            "absolute_log_ratio_q999",
            "absolute_log_ratio_cvar_1pct",
            "maximum_absolute_log_ratio",
            "minimum_metric_eigenvalue",
        ):
            values = [float(row[metric]) for row in rows]
            expected = summary["aggregate"][model][metric]
            close(statistics.mean(values), expected["mean"])
            close(statistics.stdev(values), expected["sample_standard_deviation"])
            aggregate[model][metric] = {
                "mean": statistics.mean(values),
                "sample_standard_deviation": statistics.stdev(values),
            }

    corrected = aggregate["source_density_residual_phi"][
        "maximum_absolute_log_ratio"
    ]
    close(corrected["mean"], 0.5776764333478125)
    close(corrected["sample_standard_deviation"], 0.05859113268970598)

    expected_improvements = {
        "tn_over_phi": {
            "sigma": 0.33828810338308013,
            "chi": 0.31676570377341284,
        },
        "tn_over_full_h6": {
            "sigma": 0.49172546540175754,
            "chi": 0.5952305935036253,
        },
    }
    for comparison, metrics in expected_improvements.items():
        for metric, expected in metrics.items():
            close(
                summary["comparisons"][comparison][metric][
                    "relative_improvement_of_means"
                ],
                expected,
            )
    assert summary["all_sampled_metrics_positive"] is True

    lift = load_json(
        root,
        "outputs/pipeline/type11_k6_full_h_same_degree_20260805/"
        "h2_cubic_lift_report.json",
    )
    assert lift["target_section_count"] == 871
    assert lift["holdout_relative_reconstruction_error"] < 3.0e-14
    assert lift["holdout_metric_relative_rms_difference"] < 2.0e-9
    materialized = load_json(
        root,
        "outputs/pipeline/type11_k6_full_h_same_degree_20260805/"
        "tn_replicate1_materialized_full_h_report.json",
    )
    assert materialized["holdout_metric_relative_rms_difference"] < 2.0e-9
    return {
        "sample_points": 200000,
        "replicates": 3,
        "aggregate": aggregate,
        "corrected_phi_maximum": corrected,
        "parameter_counts": {
            "positive_TN": 141750,
            "source_density_residual_phi": 142626,
            "full_H6": 871**2,
        },
        "H2_cubic_lift_metric_relative_rms": lift[
            "holdout_metric_relative_rms_difference"
        ],
        "TN_materialization_metric_relative_rms": materialized[
            "holdout_metric_relative_rms_difference"
        ],
    }


def check_x21(root: Path) -> dict[str, Any]:
    final_base = Path("outputs/pipeline/type21_q121_d12_final_blind_20260729")
    final = load_json(root, f"{final_base}/final_blind_summary.json")
    files = {
        "full_h4": "h4_final_blind_arrays.npz",
        "d8_source": "d8_final_blind_arrays.npz",
        "d12_selected": "d12_final_blind_arrays.npz",
    }
    paths = [root / final_base / relative for relative in files.values()]
    aligned_arrays(paths)
    reconstructed = {}
    for model, relative in files.items():
        metrics = pointwise_metrics(root / final_base / relative)
        compare_metric_dict(metrics, final["metrics"][model])
        reconstructed[model] = metrics
    sigma_gain = 1.0 - reconstructed["d8_source"]["sigma"] / reconstructed[
        "full_h4"
    ]["sigma"]
    chi_gain = 1.0 - reconstructed["d8_source"]["chi"] / reconstructed[
        "full_h4"
    ]["chi"]
    close(sigma_gain, 0.35457867956933875)
    close(chi_gain, 0.3643374249001504)

    control = load_json(
        root,
        "outputs/pipeline/type21_d8_continuation_capacity_control_20260730/summary.json",
    )
    assert control["preregistered_directional_gate_pass"] is False
    control_base = Path(
        "outputs/pipeline/type21_d8_continuation_capacity_control_20260730"
    )
    reconstructed_control: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ("source", "control", "d12")
    }
    relative_improvements: dict[str, list[float]] = {
        metric: [] for metric in ("sigma", "chi")
    }
    for seed in (86231, 86232, 86233):
        paths = {
            arm: root / control_base / f"seed_{seed}_{arm}_holdout_arrays.npz"
            for arm in reconstructed_control
        }
        aligned_arrays(list(paths.values()))
        for arm, path in paths.items():
            metrics = pointwise_metrics(path)
            compare_metric_dict(metrics, control["records"][str(seed)][arm])
            reconstructed_control[arm].append(metrics)
        for metric in relative_improvements:
            source_value = reconstructed_control["control"][-1][metric]
            candidate_value = reconstructed_control["d12"][-1][metric]
            relative_improvements[metric].append(
                1.0 - candidate_value / source_value
            )
    for arm, rows in reconstructed_control.items():
        for metric in (
            "sigma",
            "chi",
            "absolute_log_ratio_q999",
            "absolute_log_ratio_cvar_1pct",
            "minimum_metric_eigenvalue",
        ):
            close(
                statistics.mean(float(row[metric]) for row in rows),
                control["aggregate_means"][arm][metric],
            )
    for metric, values in relative_improvements.items():
        close(
            statistics.mean(values),
            control["paired_primary_improvements"][metric][
                "mean_relative_improvement"
            ],
        )
    initial_chi = statistics.mean(
        float(row["chi"]) for row in reconstructed_control["source"]
    )
    close(initial_chi, 0.014754661197808357)
    if math.isclose(initial_chi, reconstructed["d8_source"]["chi"], rel_tol=1.0e-4):
        raise AssertionError("distinct X21 experiments were accidentally conflated")

    ladder_base = Path(
        "outputs/pipeline/type21_fixed_d8_degree_scaling_floor1e14_20260807"
    )
    ladder = load_json(root, f"{ladder_base}/final_summary_k8_k20.json")
    assert ladder["degrees"] == [8, 12, 16, 20]
    assert ladder["replicates_per_degree"] == 3
    for row in ladder["rows"]:
        path = (
            root
            / ladder_base
            / f"k{row['degree']}/replicate_{row['replicate']}/final_common/arrays.npz"
        )
        metrics = pointwise_metrics(path)
        compare_metric_dict(metrics, row["metrics"])
    expected_means = {
        "8": 0.008766950850767199,
        "12": 0.005695955718862779,
        "16": 0.006357117557953068,
        "20": 0.006072882248982755,
    }
    for degree, expected in expected_means.items():
        close(ladder["by_degree"][degree]["metrics"]["sigma"]["mean"], expected)

    multiplication = load_json(
        root,
        "outputs/pipeline/type21_section_multiplication_audit_20260728/"
        "multiplication_rank_audit.json",
    )
    assert multiplication["section_counts"]["k4"] == 324
    assert all(
        set(seed["maps"]["mu_22"]["rank_by_relative_tolerance"].values())
        == {324}
        for seed in multiplication["seeds"]
    )
    return {
        "compression": {
            "metrics": reconstructed,
            "parameter_reduction_percent": 100.0 * (1.0 - 96800 / 104976),
            "sigma_reduction_percent": 100.0 * sigma_gain,
            "chi_reduction_percent": 100.0 * chi_gain,
        },
        "equal_update_initial_mean_chi": initial_chi,
        "equal_update_gate_pass": False,
        "equal_update_control_vs_d12": {
            metric: {
                "mean_relative_improvement": statistics.mean(values),
                "sample_standard_deviation": statistics.stdev(values),
            }
            for metric, values in relative_improvements.items()
        },
        "fixed_D8_mean_sigma": expected_means,
        "degree_four_multiplication_rank": 324,
    }


def check_x22(root: Path) -> dict[str, Any]:
    first_base = Path(
        "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719"
    )
    q289_base = Path("outputs/pipeline/type22_fixed_q289_capacity_20260806")
    models = {
        "q22_k4": root / first_base / "active_q22_k4_control_blind83703_n65532_arrays.npz",
        "q60_k4": root / first_base / "active_q60_k4_capacity_blind83703_n65532_arrays.npz",
        "q289_k4": root / q289_base / "q289_k4_capacity_blind83703_n65532_arrays.npz",
    }
    aligned_arrays(list(models.values()))
    metrics = {name: pointwise_metrics(path) for name, path in models.items()}
    bilateral = load_json(root, f"{first_base}/bilateral_evaluation_20260805.json")
    compare_metric_dict(metrics["q22_k4"], bilateral["models"]["q22_k4"]["metrics"])
    compare_metric_dict(metrics["q60_k4"], bilateral["models"]["q60_k4"]["metrics"])
    q289 = load_json(root, f"{q289_base}/q289_k4_capacity_blind83703_n65532.json")
    close(metrics["q289_k4"]["sigma"], q289["metrics"]["compressed_ma_errors"]["sigma"])
    close(metrics["q289_k4"]["chi"], q289["metrics"]["compressed_ma_errors"]["sqrt_squared_energy"])
    assert q289["trainable_real_parameter_count"] == 34680

    continuation_files = {
        "q60_k4": root / first_base / "active_q60_k4_on_common_blind83713_n65532_arrays.npz",
        "q60_k6": root / first_base / "active_q60_k6_on_common_blind83713_n65532_arrays.npz",
    }
    aligned_arrays(list(continuation_files.values()))
    continuation = {
        name: pointwise_metrics(path) for name, path in continuation_files.items()
    }
    common = load_json(root, f"{first_base}/k4_k6_common_bilateral_20260805.json")
    for model in continuation:
        compare_metric_dict(continuation[model], common["models"][model]["metrics"])
    close(continuation["q60_k4"]["sigma"], 0.07446793303829818)
    close(continuation["q60_k6"]["sigma"], 0.047375795505400664)
    return {
        "dictionary_metrics": metrics,
        "continuation_metrics": continuation,
        "parameter_counts": {"q22": 15356, "q60": 41880, "q289": 34680},
    }


def manuscript_text(directory: Path) -> str:
    paths = [directory / "gcicy_tn_paper.tex"]
    paths.extend(sorted((directory / "generated_tn").glob("*.tex")))
    return "\n".join(
        path.read_text(encoding="utf-8") for path in paths if path.is_file()
    )


def check_manuscript(root: Path, manuscript_dir: Path) -> dict[str, Any]:
    text = manuscript_text(manuscript_dir)
    supplement_table = root / "supplement/generated_tn/x11_final_common_holdout_20260807.tex"
    if supplement_table.is_file():
        text += supplement_table.read_text(encoding="utf-8")
    required = (
        "0.57768",
        "0.05859",
        "0.01480",
        "0.02237",
        "0.02912",
        "35.46",
        "36.43",
        "0.008767",
        "0.005696",
        "0.006357",
        "0.006073",
        "0.03853",
        "0.04738",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise AssertionError(f"final manuscript is missing values: {missing}")
    active = (manuscript_dir / "gcicy_tn_paper.tex").read_text(encoding="utf-8")
    active += (manuscript_dir / "generated_tn/x11_equal_time_final_20260807.tex").read_text(
        encoding="utf-8"
    )
    if "0.57906" in active:
        raise AssertionError("superseded X11 maximum remains in active manuscript inputs")
    for fragment in (
        r"raw output is multiplied by \(0.1\)",
        r"\((-15,10)\)",
        "fixes the trace convention",
        "metric level to approximately\n" + r"\(10^{-9}\)",
    ):
        if fragment not in active:
            raise AssertionError(f"documentation correction missing: {fragment}")
    return {"required_numeric_fragments": len(required), "superseded_values": 0}


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manuscript_dir = (
        args.manuscript_dir.expanduser().resolve()
        if args.manuscript_dir is not None
        else root / "manuscript"
    )
    if not manuscript_dir.is_dir():
        manuscript_dir = root / "release_work/final_repro_20260813/manuscript"
    report = {
        "schema": "gcicy-tn-final-reproducibility-check-v1",
        "x11": check_x11(root),
        "x21": check_x21(root),
        "x22": check_x22(root),
        "manuscript": check_manuscript(root, manuscript_dir),
        "status": "all pointwise reconstructions and manuscript checks passed",
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
