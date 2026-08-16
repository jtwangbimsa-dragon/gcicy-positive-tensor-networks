#!/usr/bin/env python3
"""Build machine-checked tables for the multitype gCICY metric manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def workspace_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type11",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k24_tail_refined_audit.json",
    )
    parser.add_argument(
        "--type21",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p5p1_type21_k3_1223_k1234_large_audit.json",
    )
    parser.add_argument(
        "--type22",
        type=Path,
        default=ROOT / "outputs/pipeline/p1p1p5_type22_k123_audit.json",
    )
    parser.add_argument(
        "--out-of-sample",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_m4_k1_out_of_sample_audit.json",
    )
    parser.add_argument(
        "--spectrum-8192",
        type=Path,
        default=ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_scalar_metric_volume_cluster_8192.json",
    )
    parser.add_argument(
        "--spectrum-32768",
        type=Path,
        default=ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_scalar_metric_volume_cluster_32768.json",
    )
    parser.add_argument(
        "--spectrum-65536",
        type=Path,
        default=ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_scalar_metric_volume_cluster_65536.json",
    )
    for seed in (271828, 161803, 577215):
        parser.add_argument(
            f"--basis-seed-{seed}",
            dest=f"basis_seed_{seed}",
            type=Path,
            default=ROOT
            / "outputs/pipeline/"
            f"p4p1_type11_hirzebruch_scalar_basis_seed_{seed}_65536.json",
        )
    parser.add_argument(
        "--m4-smoothness-31991",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_m4_smoothness_p31991.json",
    )
    parser.add_argument(
        "--m4-smoothness-32003",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_m4_smoothness_p32003.json",
    )
    parser.add_argument(
        "--m4-smoothness-65521",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_m4_smoothness_p65521.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "outputs/multitype_publication_evidence.json",
    )
    parser.add_argument(
        "--out-markdown",
        type=Path,
        default=ROOT / "paper/generated/multitype_tables.md",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=ROOT / "paper/generated/multitype_tables.tex",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing evidence file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"multitype evidence gate failed: {message}")


def maximum_atlas_error(audit: dict) -> float:
    atlas = audit["atlas"]
    values = [
        atlas["max_baseline_projective_ma_error"],
        atlas["max_baseline_implicit_ma_error"],
        atlas["max_projective_importance_log_weight_error"],
        atlas["max_implicit_importance_log_weight_error"],
        *atlas["artifact_projective_ma_errors"].values(),
        *atlas["artifact_implicit_ma_errors"].values(),
    ]
    return float(max(values))


def metric_rows(label: str, type_label: str, audit: dict) -> list[dict]:
    output = []
    for artifact in audit["artifacts"]:
        output.append(
            {
                "geometry": label,
                "type": type_label,
                "degree": artifact["degree"],
                "section_count": artifact["section_count"],
                "mean_sigma": artifact["mean_sigma"],
                "sigma_95_percent_ci": artifact["sigma_95_percent_ci"],
                "min_metric_eigenvalue": artifact["min_metric_eigenvalue"],
                "min_importance_effective_sample_size": artifact[
                    "min_importance_effective_sample_size"
                ],
                "every_seed_improved": artifact["every_seed_sigma_improved"],
            }
        )
    return output


def coverage_row(label: str, type_label: str, audit: dict) -> dict:
    primary = audit["artifacts"][-1]
    atlas = audit["atlas"]
    return {
        "geometry": label,
        "type": type_label,
        "projective": [
            atlas["projective_charts_seen"],
            atlas["projective_charts_expected"],
        ],
        "implicit": [
            atlas["implicit_coordinate_choices_seen"],
            atlas["implicit_coordinate_choices_expected"],
        ],
        "max_atlas_error": maximum_atlas_error(audit),
        "minimum_jacobian_singular_value": audit[
            "minimum_sampled_jacobian_singular_value"
        ],
        "primary_sigma": primary["mean_sigma"],
        "minimum_metric_eigenvalue": audit["minimum_metric_eigenvalue"],
    }


def spectrum_rows(label: str, audit: dict) -> list[dict]:
    artifact = audit["artifacts"][-1]
    output = []
    for level in artifact["trial_levels"]:
        eigenvalues = level["eigenvalues"]
        output.append(
            {
                "sample": label,
                "points": audit["points_per_seed"],
                "level": level["level"],
                "raw_feature_count": level["raw_feature_count"],
                "retained_trial_ranks": level["retained_trial_ranks"],
                "lambda_1": eigenvalues[0]["mean"],
                "lambda_2": eigenvalues[1]["mean"],
                "lambda_3": eigenvalues[2]["mean"],
                "minimum_effective_sample_size": level[
                    "minimum_integration_effective_sample_size"
                ],
                "minimum_ess_per_rank": level[
                    "minimum_effective_samples_per_retained_direction"
                ],
                "minimum_cluster_ess_per_rank": level[
                    "minimum_effective_clusters_per_retained_direction"
                ],
                "maximum_cluster_jackknife_relative_standard_error": (
                    level.get("cluster_delete_group_jackknife", {}).get(
                        "maximum_within_seed_relative_standard_error"
                    )
                ),
            }
        )
    return output


def spectrum_means(audit: dict, level_number: int, count: int = 3) -> list[float]:
    levels = {
        int(level["level"]): level
        for level in audit["artifacts"][-1]["trial_levels"]
    }
    if level_number not in levels:
        raise SystemExit(f"missing scalar trial level {level_number}")
    return [
        float(row["mean"])
        for row in levels[level_number]["eigenvalues"][:count]
    ]


def markdown(summary: dict) -> str:
    lines = [
        "# Multitype publication evidence",
        "",
        "> Generated from accepted audit JSON files; do not edit by hand.",
        "",
        "## Held-out metric degree trends",
        "",
        "| Geometry | Type | Degree | Sections | Sigma | 95% CI | Min eig. | Min ESS |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["metric_rows"]:
        degree = ",".join(str(value) for value in row["degree"])
        interval = row["sigma_95_percent_ci"]
        lines.append(
            f"| {row['geometry']} | {row['type']} | {degree} | "
            f"{row['section_count']} | {row['mean_sigma']:.6f} | "
            f"[{interval[0]:.6f}, {interval[1]:.6f}] | "
            f"{row['min_metric_eigenvalue']:.3e} | "
            f"{row['min_importance_effective_sample_size']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Atlas coverage",
            "",
            "| Geometry | Type | Projective | Implicit | Max error | Min Jacobian sv |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["coverage_rows"]:
        lines.append(
            f"| {row['geometry']} | {row['type']} | "
            f"{row['projective'][0]}/{row['projective'][1]} | "
            f"{row['implicit'][0]}/{row['implicit'][1]} | "
            f"{row['max_atlas_error']:.3e} | "
            f"{row['minimum_jacobian_singular_value']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Finite-basis scalar Rayleigh--Ritz comparisons",
            "",
            "| Points | Level | Raw/rank | lambda1 | lambda2 | lambda3 | Min cluster-weight ESS/rank | Max JK rel. SE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["spectrum_rows"]:
        ranks = ",".join(str(value) for value in row["retained_trial_ranks"])
        jackknife = row["maximum_cluster_jackknife_relative_standard_error"]
        jackknife_text = "--" if jackknife is None else f"{100.0 * jackknife:.2f}%"
        lines.append(
            f"| {row['points']} | {row['level']} | "
            f"{row['raw_feature_count']}/{ranks} | {row['lambda_1']:.6f} | "
            f"{row['lambda_2']:.6f} | {row['lambda_3']:.6f} | "
            f"{row['minimum_cluster_ess_per_rank']:.2f} | {jackknife_text} |"
        )
    return "\n".join(lines) + "\n"


def latex(summary: dict) -> str:
    lines = [
        "% Generated by scripts/build_multitype_publication_evidence.py; do not edit.",
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\caption{Fresh-seed Monge--Ampere errors across gCICY types.}",
        "\\label{tab:multitype-metrics}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Geometry & type & $k$ & $N_k$ & $\\sigma$ & $\\lambda_{\\min}(g)$ & ESS \\\\",
        "\\midrule",
    ]
    for row in summary["metric_rows"]:
        k = row["degree"][0]
        lines.append(
            f"{row['geometry']} & {row['type']} & {k} & {row['section_count']} & "
            f"{row['mean_sigma']:.5f} & {row['min_metric_eigenvalue']:.2e} & "
            f"{row['min_importance_effective_sample_size']:.0f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Complete numerical atlas coverage for the primary metric on each geometry.}",
            "\\label{tab:multitype-atlas}",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "Geometry & type & projective & implicit & max. error & min. $s(J)$ \\\\",
            "\\midrule",
        ]
    )
    for row in summary["coverage_rows"]:
        lines.append(
            f"{row['geometry']} & {row['type']} & "
            f"{row['projective'][0]}/{row['projective'][1]} & "
            f"{row['implicit'][0]}/{row['implicit'][1]} & "
            f"{row['max_atlas_error']:.2e} & "
            f"{row['minimum_jacobian_singular_value']:.2e} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Finite-basis metric-volume sample-size and trial-space comparison for the first three scalar-Laplacian Monte Carlo Rayleigh--Ritz estimates on the finite-$k$ $m=3$ type-$(1,1)$ metric.}",
            "\\label{tab:scalar-spectrum}",
            "\\begin{tabular}{rrrrrrrr}",
            "\\toprule",
            "$N$ & level & raw/rank & $\\lambda_1$ & $\\lambda_2$ & $\\lambda_3$ & cwESS/rank & JK SE \\\\",
            "\\midrule",
        ]
    )
    for row in summary["spectrum_rows"]:
        ranks = ",".join(str(value) for value in row["retained_trial_ranks"])
        jackknife = row["maximum_cluster_jackknife_relative_standard_error"]
        jackknife_text = "--" if jackknife is None else f"{100.0 * jackknife:.2f}\\%"
        lines.append(
            f"{row['points']} & {row['level']} & "
            f"{row['raw_feature_count']}/{ranks} & {row['lambda_1']:.5f} & "
            f"{row['lambda_2']:.5f} & {row['lambda_3']:.5f} & "
            f"{row['minimum_cluster_ess_per_rank']:.2f} & {jackknife_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Random-cubic basis-seed robustness on identical $N=65536$ point samples.  All rows use 256 added rank-one cubic directions; the final two columns report the point- and cluster-weight ESS per retained direction.}",
            "\\label{tab:scalar-basis-seeds}",
            "\\begin{tabular}{rrrrrrr}",
            "\\toprule",
            "cubic seed & rank & $\\lambda_1$ & $\\lambda_2$ & $\\lambda_3$ & pESS/rank & cwESS/rank \\\\ ",
            "\\midrule",
        ]
    )
    for row in summary["basis_seed_rows"]:
        ranks = ",".join(str(value) for value in row["retained_trial_ranks"])
        lines.append(
            f"{row['cubic_feature_seed']} & {ranks} & {row['lambda_1']:.5f} & "
            f"{row['lambda_2']:.5f} & {row['lambda_3']:.5f} & "
            f"{row['minimum_effective_sample_size_per_rank']:.2f} & "
            f"{row['minimum_cluster_ess_per_rank']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    named_paths = {
        "type11": args.type11,
        "type21": args.type21,
        "type22": args.type22,
        "out_of_sample": args.out_of_sample,
        "spectrum_8192": args.spectrum_8192,
        "spectrum_32768": args.spectrum_32768,
        "spectrum_65536": args.spectrum_65536,
        "basis_seed_271828": args.basis_seed_271828,
        "basis_seed_161803": args.basis_seed_161803,
        "basis_seed_577215": args.basis_seed_577215,
        "m4_smoothness_31991": args.m4_smoothness_31991,
        "m4_smoothness_32003": args.m4_smoothness_32003,
        "m4_smoothness_65521": args.m4_smoothness_65521,
    }
    data = {key: load(path) for key, path in named_paths.items()}
    for key in ("type11", "type21", "type22", "out_of_sample"):
        require(data[key]["success"], f"{key} metric audit")
        require(all(data[key]["gates"].values()), f"{key} metric gates")
    for key in ("spectrum_8192", "spectrum_32768", "spectrum_65536"):
        require(data[key]["success"], f"{key} audit")
        require(all(data[key]["gates"].values()), f"{key} gates")
        require(
            data[key]["integration_measure"] == "metric_volume",
            f"{key} uses approximate-metric volume",
        )
    for key in (
        "m4_smoothness_31991",
        "m4_smoothness_32003",
        "m4_smoothness_65521",
    ):
        require(data[key]["complete_atlas_proved_smooth"], f"{key} complete atlas")
        require(
            data[key]["gcicy_charts_proved_smooth"] == 20,
            f"{key} final smooth charts",
        )
    require(
        data["spectrum_8192"]["trial_feature_configuration"][
            "level_3_kind"
        ]
        == "random_rank_one_cubic_enrichment",
        "documented level-3 construction",
    )
    level3_8192 = spectrum_means(data["spectrum_8192"], 3)
    level3_32768 = spectrum_means(data["spectrum_32768"], 3)
    level2_65536 = spectrum_means(data["spectrum_65536"], 2)
    level3_65536 = spectrum_means(data["spectrum_65536"], 3)
    sample_size_relative_changes_8192_to_32768 = [
        (target - source) / source
        for source, target in zip(level3_8192, level3_32768, strict=True)
    ]
    sample_size_relative_changes_32768_to_65536 = [
        (target - source) / source
        for source, target in zip(level3_32768, level3_65536, strict=True)
    ]
    trial_space_relative_changes = [
        (target - source) / source
        for source, target in zip(level2_65536, level3_65536, strict=True)
    ]
    require(
        max(abs(value) for value in sample_size_relative_changes_8192_to_32768)
        <= 0.03,
        "first-three-mode 8192-to-32768 sample-size stability",
    )
    require(
        max(abs(value) for value in sample_size_relative_changes_32768_to_65536)
        <= 0.02,
        "first-three-mode 32768-to-65536 sample-size stability",
    )
    require(
        max(abs(value) for value in trial_space_relative_changes) <= 0.01,
        "first-three-mode trial-space stability",
    )
    final_level3 = next(
        level
        for level in data["spectrum_65536"]["artifacts"][-1]["trial_levels"]
        if int(level["level"]) == 3
    )
    require(
        final_level3["minimum_effective_clusters_per_retained_direction"] >= 15.0,
        "final cluster ESS per retained direction",
    )
    require(
        final_level3["cluster_delete_group_jackknife"][
            "maximum_within_seed_relative_standard_error"
        ]
        <= 0.05,
        "final cluster-jackknife relative standard error",
    )
    require(
        all(
            row["all_returned_clusters_complete"]
            for row in data["spectrum_65536"]["sampling_diagnostics"]
        ),
        "numerically degree-complete four-root clusters in final Ritz sample",
    )

    basis_keys = (
        "spectrum_65536",
        "basis_seed_271828",
        "basis_seed_161803",
        "basis_seed_577215",
    )
    for key in basis_keys[1:]:
        require(data[key]["success"], f"{key} audit")
        require(all(data[key]["gates"].values()), f"{key} gates")
        require(data[key]["points_per_seed"] == 65536, f"{key} point count")
        require(data[key]["seeds"] == data["spectrum_65536"]["seeds"], f"{key} samples")
    basis_seed_rows = []
    basis_values = []
    for key in basis_keys:
        audit = data[key]
        values = spectrum_means(audit, 3)
        level = next(
            entry
            for entry in audit["artifacts"][-1]["trial_levels"]
            if int(entry["level"]) == 3
        )
        basis_values.append(values)
        basis_seed_rows.append(
            {
                "cubic_feature_seed": int(
                    audit["trial_feature_configuration"]["cubic_feature_seed"]
                ),
                "retained_trial_ranks": [
                    int(value) for value in level["retained_trial_ranks"]
                ],
                "lambda_1": values[0],
                "lambda_2": values[1],
                "lambda_3": values[2],
                "minimum_effective_sample_size_per_rank": float(
                    level["minimum_effective_samples_per_retained_direction"]
                ),
                "minimum_cluster_ess_per_rank": float(
                    level["minimum_effective_clusters_per_retained_direction"]
                ),
            }
        )
    basis_array = np.asarray(basis_values, dtype=float)
    basis_means = np.mean(basis_array, axis=0)
    basis_relative_standard_deviations = (
        np.std(basis_array, axis=0, ddof=1) / basis_means
    )
    basis_relative_ranges = (
        (np.max(basis_array, axis=0) - np.min(basis_array, axis=0)) / basis_means
    )
    require(
        float(np.max(basis_relative_standard_deviations)) <= 0.02,
        "cubic basis-seed relative standard deviation",
    )
    require(
        float(np.max(basis_relative_ranges)) <= 0.05,
        "cubic basis-seed relative range",
    )

    metrics = []
    coverage = []
    geometry_specs = (
        ("type11", "Hirzebruch $m=3$", "$(1,1)$"),
        ("type21", "K3-fibered", "$(2,1)$"),
        ("type22", "$\\PP^1\\times\\PP^1\\times\\PP^5$", "$(2,2)$"),
        ("out_of_sample", "Hirzebruch $m=4$", "$(1,1)$ transfer"),
    )
    for key, label, type_label in geometry_specs:
        metrics.extend(metric_rows(label, type_label, data[key]))
        coverage.append(coverage_row(label, type_label, data[key]))

    summary = {
        "description": (
            "Machine-checked multitype, separate transfer, and scalar Ritz "
            "evidence for the gCICY metric manuscript."
        ),
        "all_gates_passed": True,
        "source_files": [
            {
                "key": key,
                "path": workspace_path(path),
                "sha256": sha256(path),
            }
            for key, path in named_paths.items()
        ],
        "gates": {
            "three_primary_types": "passed",
            "separate_transfer_geometry": "passed",
            "metric_volume_sample_size_scaling": "passed",
            "random_cubic_trial_enrichment": "passed",
            "random_cubic_basis_seed_robustness": "passed_four_seeds",
            "fibre_cluster_statistics": "passed",
            "complete_root_sampling": "passed",
            "out_of_sample_exact_smoothness": "passed_60_chart_prime_checks",
        },
        "first_three_mode_relative_changes": {
            "level3_8192_to_32768": sample_size_relative_changes_8192_to_32768,
            "level3_32768_to_65536": sample_size_relative_changes_32768_to_65536,
            "level2_to_level3_at_65536": trial_space_relative_changes,
            "sample_size_gate_8192_to_32768": 0.03,
            "sample_size_gate_32768_to_65536": 0.02,
            "trial_space_gate": 0.01,
        },
        "metric_rows": metrics,
        "coverage_rows": coverage,
        "spectrum_rows": [
            row
            for key in ("spectrum_8192", "spectrum_32768", "spectrum_65536")
            for row in spectrum_rows(key.removeprefix("spectrum_"), data[key])
            if row["level"] in (2, 3)
        ],
        "basis_seed_rows": basis_seed_rows,
        "basis_seed_robustness": {
            "relative_standard_deviations": basis_relative_standard_deviations.tolist(),
            "relative_ranges": basis_relative_ranges.tolist(),
            "maximum_relative_standard_deviation_gate": 0.02,
            "maximum_relative_range_gate": 0.05,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_markdown.write_text(markdown(summary), encoding="utf-8")
    args.out_tex.write_text(latex(summary), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_markdown}")
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()
