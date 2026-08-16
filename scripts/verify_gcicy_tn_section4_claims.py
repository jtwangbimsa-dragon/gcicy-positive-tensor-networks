#!/usr/bin/env python3
"""Recompute the numerical headline claims in Section 4.

This verifier is intentionally independent of the LaTeX table generators.  It
reads the frozen JSON evidence, recomputes means, sample standard deviations,
parameter counts and relative changes, and then checks that the rounded values
occur in the registered manuscript source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
EXPECTED_PDF_SHA256 = (
    "4c8c8435a349638a23c0eb0b5cdda1cd6271e56b6256102d41f535f1ee61667a"
)
DEFAULT_MANUSCRIPT_DIR = ROOT / "manuscript_v5"
if not DEFAULT_MANUSCRIPT_DIR.is_dir():
    DEFAULT_MANUSCRIPT_DIR = (
        ROOT / "outputs/manuscript_snapshots/gcicy_tn_paper_v5_current_20260810"
    )


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(observed: float, expected: float, tolerance: float = 5e-13) -> None:
    if not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"expected {expected:.16g}, observed {observed:.16g}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manuscript_text(directory: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(directory.rglob("*.tex"))
    )


def check_x11() -> dict:
    summary = load(
        "outputs/pipeline/type11_x11_equal_time_final_20260807/final_summary.json"
    )
    assert summary["point_count"] == 200000
    assert summary["replicate_count"] == 3
    assert summary["all_sampled_metrics_positive"] is True

    expected = {
        "positive_TN": (0.014802218204044865, 0.022596787046948813),
        "source_density_residual_phi": (
            0.022369581504765066,
            0.03307326223485543,
        ),
        "cold_full_H6_equal_time_plateau": (
            0.029122486366044375,
            0.05582632156551387,
        ),
    }
    for model, (sigma, chi) in expected.items():
        rows = [row["models"][model] for row in summary["rows"]]
        close(statistics.mean(row["sigma"] for row in rows), sigma)
        close(statistics.mean(row["chi"] for row in rows), chi)
        close(
            statistics.stdev(row["sigma"] for row in rows),
            summary["aggregate"][model]["sigma"]["sample_standard_deviation"],
        )

    rank = load(
        "outputs/pipeline/type11_k2_to_k6_multiplication_audit_20260805/"
        "multiplication_rank_audit.json"
    )
    assert rank["target_section_count"] == 871
    assert all(
        set(seed["coefficient_rank_by_relative_tolerance"].values()) == {871}
        for seed in rank["seeds"]
    )

    materialized_tn = load(
        "outputs/pipeline/type11_k6_full_h_same_degree_20260805/"
        "tn_replicate1_materialized_full_h_report.json"
    )
    assert materialized_tn["section_count"] == 871
    assert materialized_tn["complete_full_h_real_parameter_count"] == 871**2
    assert materialized_tn["holdout_points"] == 1024
    assert materialized_tn["holdout_metric_relative_rms_difference"] < 2e-9
    assert materialized_tn["holdout_centered_log_eta_rms_difference"] < 2e-9

    exact_lift = load(
        "outputs/pipeline/type11_k6_full_h_same_degree_20260805/"
        "h2_cubic_lift_report.json"
    )
    assert exact_lift["source_section_count"] == 45
    assert exact_lift["target_section_count"] == 871
    assert exact_lift["symmetric_product_count"] == 16215
    assert exact_lift["ordered_product_count"] == 91125
    assert exact_lift["holdout_relative_reconstruction_error"] < 3e-14
    assert exact_lift["holdout_metric_relative_rms_difference"] < 2e-9

    parameter_counts = {
        "positive_TN": 2 * 45**2 * (5 + 5**2 + 5),
        "full_H6": 871**2,
        "source_density_residual_phi": 142626,
    }
    assert parameter_counts == {
        "positive_TN": 141750,
        "full_H6": 758641,
        "source_density_residual_phi": 142626,
    }
    return {
        "point_count": summary["point_count"],
        "replicates": summary["replicate_count"],
        "parameter_counts": parameter_counts,
        "mean_sigma": {name: values[0] for name, values in expected.items()},
        "mean_chi": {name: values[1] for name, values in expected.items()},
        "multiplication_rank": 871,
        "representation_checks": {
            "TN_to_full_H_metric_relative_rms": materialized_tn[
                "holdout_metric_relative_rms_difference"
            ],
            "TN_to_full_H_centered_log_eta_rms": materialized_tn[
                "holdout_centered_log_eta_rms_difference"
            ],
            "H2_cubic_lift_function_relative_error": exact_lift[
                "holdout_relative_reconstruction_error"
            ],
            "H2_cubic_lift_metric_relative_rms": exact_lift[
                "holdout_metric_relative_rms_difference"
            ],
        },
    }


def check_x21() -> dict:
    final = load(
        "outputs/pipeline/type21_q121_d12_final_blind_20260729/"
        "final_blind_summary.json"
    )
    assert final["final_blind"]["point_count"] == 196608
    h4 = final["metrics"]["full_h4"]
    d8 = final["metrics"]["d8_source"]
    close(h4["sigma"], 0.01627280599236754)
    close(d8["sigma"], 0.010502815930705834)
    close(h4["chi"], 0.023968295955664777)
    close(d8["chi"], 0.015235748727933181)
    sigma_gain = 1.0 - d8["sigma"] / h4["sigma"]
    chi_gain = 1.0 - d8["chi"] / h4["chi"]
    parameter_gain = 1.0 - 96800 / 104976
    assert round(100 * sigma_gain, 2) == 35.46
    assert round(100 * chi_gain, 2) == 36.43
    assert round(100 * parameter_gain, 2) == 7.79

    control = load(
        "outputs/pipeline/type21_d8_continuation_capacity_control_20260730/"
        "summary.json"
    )
    sigma_changes = control["paired_primary_improvements"]["sigma"][
        "per_seed_relative_improvement"
    ]
    chi_changes = control["paired_primary_improvements"]["chi"][
        "per_seed_relative_improvement"
    ]
    close(statistics.mean(sigma_changes), 0.019201759170286686)
    assert round(100 * statistics.stdev(sigma_changes), 2) == 2.86
    close(statistics.mean(chi_changes), 0.010981994260257198)
    assert round(100 * statistics.stdev(chi_changes), 2) == 2.66
    assert control["preregistered_directional_gate_pass"] is False

    degree = load(
        "outputs/pipeline/type21_fixed_d8_degree_scaling_floor1e14_20260807/"
        "final_evidence_k8_k20/final_summary_k8_k20.json"
    )
    expected_sigma = {
        "8": 0.008766950850767199,
        "12": 0.005695955718862779,
        "16": 0.006357117557953068,
        "20": 0.006072882248982755,
    }
    for key, value in expected_sigma.items():
        close(degree["by_degree"][key]["metrics"]["sigma"]["mean"], value)
    assert degree["point_count"] == 196608
    assert degree["replicates_per_degree"] == 3

    multiplication = load(
        "outputs/pipeline/type21_section_multiplication_audit_20260728/"
        "multiplication_rank_audit.json"
    )
    assert multiplication["section_counts"]["k4"] == 324
    assert all(
        set(seed["maps"]["mu_22"]["rank_by_relative_tolerance"].values())
        == {324}
        for seed in multiplication["seeds"]
    )

    schmidt = load(
        "outputs/pipeline/type21_h4_lift_all_cut_schmidt_20260729/"
        "minimum_and_nonminimum_lifts_all_cuts.json"
    )
    assert schmidt["section_count_degree_one"] == 11
    assert schmidt["section_count_degree_four"] == 324
    assert schmidt["ordered_product_count"] == 11**4
    assert schmidt["minimum_norm_right_inverse_error"] < 2e-12
    assert schmidt["alternative_right_inverse_error"] < 2e-12
    for lift in ("minimum_norm", "nonminimum_null_perturbation"):
        for cut in ("cut_1_of_4", "cut_3_of_4"):
            ranks = schmidt["lifts"][lift][cut][
                "rank_by_relative_tolerance"
            ]
            assert set(ranks.values()) == {121}
    return {
        "compression": {
            "point_count": 196608,
            "full_H4_sigma": h4["sigma"],
            "TN_k8_D8_sigma": d8["sigma"],
            "parameter_reduction_percent": 100 * parameter_gain,
            "sigma_reduction_percent": 100 * sigma_gain,
            "chi_reduction_percent": 100 * chi_gain,
        },
        "equal_update": {
            "sigma_mean_percent": 100 * statistics.mean(sigma_changes),
            "sigma_sd_percent": 100 * statistics.stdev(sigma_changes),
            "chi_mean_percent": 100 * statistics.mean(chi_changes),
            "chi_sd_percent": 100 * statistics.stdev(chi_changes),
            "gate_pass": False,
        },
        "fixed_D8_mean_sigma": expected_sigma,
        "degree_four_multiplication_rank": 324,
        "lift_audit": {
            "right_inverse_errors": {
                "minimum_norm": schmidt["minimum_norm_right_inverse_error"],
                "nonminimum": schmidt["alternative_right_inverse_error"],
            },
            "numerical_edge_cut_operator_schmidt_rank": 121,
            "rank_tolerances": schmidt["rank_tolerances"],
        },
    }


def check_x22() -> dict:
    bilateral = load(
        "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/"
        "bilateral_evaluation_20260805.json"
    )["models"]
    q289 = load(
        "outputs/pipeline/type22_fixed_q289_capacity_20260806/"
        "q289_k4_capacity_blind83703_n65532.json"
    )
    values = {
        "q22_k4": (
            bilateral["q22_k4"]["metrics"]["sigma"],
            bilateral["q22_k4"]["metrics"]["chi"],
        ),
        "q60_k4": (
            bilateral["q60_k4"]["metrics"]["sigma"],
            bilateral["q60_k4"]["metrics"]["chi"],
        ),
        "q289_k4": (
            q289["metrics"]["compressed_ma_errors"]["sigma"],
            q289["metrics"]["compressed_ma_errors"]["sqrt_squared_energy"],
        ),
        "q60_k6": (
            bilateral["q60_k6"]["metrics"]["sigma"],
            bilateral["q60_k6"]["metrics"]["chi"],
        ),
    }
    expected = {
        "q22_k4": (0.10040243221407952, 0.13001501925730896),
        "q60_k4": (0.0739214034504444, 0.09739728479419245),
        "q289_k4": (0.038534152840527774, 0.0539933202834899),
        "q60_k6": (0.04737579550540058, 0.06653279134431639),
    }
    for key in expected:
        close(values[key][0], expected[key][0])
        close(values[key][1], expected[key][1])
    assert q289["points"] == 65532
    assert q289["trainable_real_parameter_count"] == 34680

    paired = load(
        "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/"
        "k4_k6_common_paired_bootstrap_20260805.json"
    )
    assert paired["point_count"] == 65532
    assert paired["fibre_cluster_count"] == 21844
    q999 = paired["comparisons"]["absolute_log_ratio_q999"]
    q999_interval = q999["bootstrap_95pct_confidence_interval"]
    assert q999_interval[0] < 0 < q999_interval[1]
    assert q999["bootstrap_win_probability"] == 0.9665
    cvar = paired["comparisons"]["absolute_log_ratio_cvar_1pct"]
    assert cvar["bootstrap_95pct_confidence_interval"][0] > 0
    return {
        "point_count": 65532,
        "metrics": values,
        "parameter_counts": {"q22": 15356, "q60": 41880, "q289": 34680},
        "k4_to_k6_paired_bootstrap": {
            "fibre_clusters": paired["fibre_cluster_count"],
            "absolute_log_ratio_q999_improvement_95pct_interval": q999_interval,
            "absolute_log_ratio_q999_win_probability": q999[
                "bootstrap_win_probability"
            ],
            "absolute_log_ratio_cvar_1pct_improvement_95pct_interval": cvar[
                "bootstrap_95pct_confidence_interval"
            ],
        },
    }


def check_manuscript(directory: Path) -> None:
    text = manuscript_text(directory)
    required = [
        "0.01480", "0.02237", "0.02912", "0.016273", "0.010503",
        "35.46", "36.43", "1.92", "2.86", "0.008767", "0.005696",
        "0.006357", "0.006073", "0.10040", "0.07392", "0.03853",
        "0.07447", "0.04738",
    ]
    missing = [value for value in required if value not in text]
    if missing:
        raise AssertionError(f"rounded Section 4 values absent from manuscript: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_DIR,
    )
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_manuscript(args.manuscript_dir)
    if args.pdf is not None and sha256(args.pdf) != EXPECTED_PDF_SHA256:
        raise AssertionError("manuscript PDF hash does not match registered v5")
    report = {
        "schema": "gcicy-tn-section4-independent-recheck-v1",
        "manuscript_pdf_sha256": EXPECTED_PDF_SHA256,
        "x11": check_x11(),
        "x21": check_x21(),
        "x22": check_x22(),
        "status": "all checks passed",
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
