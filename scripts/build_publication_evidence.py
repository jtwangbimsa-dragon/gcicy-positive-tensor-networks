#!/usr/bin/env python3
"""Build machine-checked publication tables from accepted audit JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def workspace_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gcicy-convergence",
        type=Path,
        default=ROOT / "outputs" / "gcicy_generic_h_convergence_k1_k2_k3_gpu_audit.json",
    )
    parser.add_argument(
        "--bicubic-convergence",
        type=Path,
        default=ROOT / "outputs" / "bicubic_h_convergence_k1_k2_k3_gpu_audit.json",
    )
    parser.add_argument(
        "--gcicy-k3-artifact",
        type=Path,
        default=ROOT / "outputs" / "gcicy_generic_global_h_metric_k3_rank64_whitened_gpu.npz",
    )
    parser.add_argument(
        "--gcicy-k3-summary",
        type=Path,
        default=ROOT / "outputs" / "gcicy_generic_global_h_metric_k3_rank64_whitened_gpu_summary.json",
    )
    parser.add_argument(
        "--gcicy-k3-audit",
        type=Path,
        default=ROOT / "outputs" / "gcicy_generic_global_h_metric_k3_rank64_whitened_gpu_audit.json",
    )
    parser.add_argument(
        "--bicubic-k3-artifact",
        type=Path,
        default=ROOT / "outputs" / "bicubic_global_h_metric_k3_k2xk1_gpu.npz",
    )
    parser.add_argument(
        "--bicubic-k3-summary",
        type=Path,
        default=ROOT / "outputs" / "bicubic_global_h_metric_k3_k2xk1_gpu_summary.json",
    )
    parser.add_argument(
        "--bicubic-k3-audit",
        type=Path,
        default=ROOT / "outputs" / "bicubic_global_h_metric_k3_k2xk1_gpu_audit.json",
    )
    parser.add_argument(
        "--bicubic-rejected-k3",
        type=Path,
        default=ROOT / "outputs" / "bicubic_h_convergence_audit.json",
    )
    parser.add_argument(
        "--family",
        type=Path,
        default=ROOT / "outputs" / "gcicy_model_family_k1_audit.json",
    )
    parser.add_argument(
        "--gcicy-observables",
        type=Path,
        default=ROOT / "outputs" / "gcicy_geometric_observables_k1_k2_k3_gpu_audit.json",
    )
    parser.add_argument(
        "--bicubic-observables",
        type=Path,
        default=ROOT / "outputs" / "bicubic_geometric_observables_k1_k2_k3_gpu_audit.json",
    )
    parser.add_argument(
        "--hard-region",
        type=Path,
        default=ROOT / "outputs" / "gcicy_k3_hard_region_audit.json",
    )
    parser.add_argument(
        "--hard-region-samples",
        type=Path,
        default=ROOT / "outputs" / "gcicy_k3_hard_region_samples.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "outputs" / "publication_evidence_summary.json",
    )
    parser.add_argument(
        "--out-markdown",
        type=Path,
        default=ROOT / "paper" / "generated" / "results_tables.md",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=ROOT / "paper" / "generated" / "results_tables.tex",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing evidence file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ci(values: list[float]) -> tuple[float, list[float]]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, [mean - 1.96 * standard_error, mean + 1.96 * standard_error]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"publication evidence gate failed: {message}")


def h_artifact_summary(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing evidence file: {path}")
    with np.load(path) as artifact:
        matrix = np.asarray(artifact["global_h_matrix"])
    antihermitian = matrix - matrix.conjugate().T
    value = matrix.astype(np.complex128)
    hermitian = 0.5 * (value + value.conjugate().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    summary = {
        "path": workspace_path(path),
        "sha256": file_sha256(path),
        "dtype": str(matrix.dtype),
        "shape": list(matrix.shape),
        "max_antihermitian_abs": float(np.max(np.abs(antihermitian))),
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
    }
    require(summary["max_antihermitian_abs"] <= 1e-12, f"non-Hermitian H artifact {path}")
    require(summary["min_eigenvalue"] > 0, f"non-positive H artifact {path}")
    return summary


def convergence_rows(label: str, audit: dict, accepted: bool = True) -> list[dict]:
    rows = []
    for item in audit["aggregate"]:
        rows.append(
            {
                "geometry": label,
                "k": item["k"],
                "sections": item["section_count"],
                "mean_rms": item["mean_rms"],
                "rms_95_percent_ci": item["rms_95_percent_ci"],
                "mean_weighted_rms": item["mean_weighted_rms"],
                "weighted_rms_95_percent_ci": item["weighted_rms_95_percent_ci"],
                "min_metric_eigenvalue": item["min_metric_eigenvalue"],
                "status": "accepted" if accepted else "rejected_diagnostic",
                "artifact": item["artifact"],
            }
        )
    return rows


def baseline_and_method_rows(label: str, audit: dict, method_names: dict[int, str]) -> list[dict]:
    baseline, baseline_ci = mean_ci([row["baseline_rms"] for row in audit["rows"]])
    baseline_weighted, baseline_weighted_ci = mean_ci(
        [row["baseline_weighted_rms"] for row in audit["rows"]]
    )
    rows = [
        {
            "geometry": label,
            "method": "ambient_fubini_study",
            "mean_rms": baseline,
            "rms_95_percent_ci": baseline_ci,
            "mean_weighted_rms": baseline_weighted,
            "weighted_rms_95_percent_ci": baseline_weighted_ci,
            "unweighted_reduction_from_baseline": 0.0,
            "weighted_reduction_from_baseline": 0.0,
            "status": "baseline",
        }
    ]
    for item in audit["aggregate"]:
        rows.append(
            {
                "geometry": label,
                "method": method_names[item["k"]],
                "mean_rms": item["mean_rms"],
                "rms_95_percent_ci": item["rms_95_percent_ci"],
                "mean_weighted_rms": item["mean_weighted_rms"],
                "weighted_rms_95_percent_ci": item["weighted_rms_95_percent_ci"],
                "unweighted_reduction_from_baseline": 1.0 - item["mean_rms"] / baseline,
                "weighted_reduction_from_baseline": 1.0
                - item["mean_weighted_rms"] / baseline_weighted,
                "status": "accepted",
            }
        )
    return rows


def observable_rows(label: str, audit: dict, candidate: str) -> list[dict]:
    require(candidate in audit["aggregate"], f"missing {label} observable candidate {candidate}")
    selected = audit["aggregate"][candidate]
    rows = []
    volume = selected["volume_ratio"]
    rows.append(
        {
            "geometry": label,
            "metric": candidate,
            "observable": "normalized_volume",
            "estimate": volume["mean"],
            "seed_standard_error": volume["seed_standard_error"],
            "normal_95_percent_ci": volume["normal_95_percent_ci"],
            "exact": volume["exact"],
            "relative_error": volume["relative_error"],
            "exact_in_95_percent_ci": (
                volume["normal_95_percent_ci"][0]
                <= volume["exact"]
                <= volume["normal_95_percent_ci"][1]
            ),
        }
    )
    for index, slope in enumerate(selected["slope_ratios"]):
        rows.append(
            {
                "geometry": label,
                "metric": candidate,
                "observable": f"normalized_slope_{index + 1}",
                "estimate": slope["mean"],
                "seed_standard_error": slope["seed_standard_error"],
                "normal_95_percent_ci": slope["normal_95_percent_ci"],
                "exact": slope["exact"],
                "relative_error": slope["relative_error"],
                "exact_in_95_percent_ci": (
                    slope["normal_95_percent_ci"][0]
                    <= slope["exact"]
                    <= slope["normal_95_percent_ci"][1]
                ),
            }
        )
    return rows


def hard_region_rows(audit: dict) -> list[dict]:
    rows = [
        {
            "region": "all points",
            "count": audit["total_points"],
            **audit["distributions"]["absolute_centered_residual"],
        }
    ]
    labels = {
        "bottom_1_percent_jacobian_min_singular_value": "bottom 1% Jacobian sigma",
        "bottom_1_percent_residue_denominator_abs": "bottom 1% residue denominator",
        "bottom_1_percent_metric_min_eigenvalue": "bottom 1% metric eigenvalue",
        "bottom_1_percent_rational_denominator_margin": "bottom 1% rational margin",
        "top_1_percent_importance_weight": "top 1% importance weight",
    }
    for key, label in labels.items():
        item = audit["hard_subsets"][key]
        rows.append(
            {
                "region": label,
                "count": item["count"],
                **item["absolute_centered_residual"],
            }
        )
    return rows


def fmt(value: float) -> str:
    return f"{value:.4f}"


def interval(values: list[float]) -> str:
    return f"[{values[0]:.4f}, {values[1]:.4f}]"


def markdown(summary: dict) -> str:
    lines = [
        "# Generated publication evidence",
        "",
        "> Generated only from machine-checked audit JSON files. Do not edit by hand.",
        "",
        "## Paired section-degree results",
        "",
        "| Geometry | k | Sections | RMS mean | RMS 95% CI | Weighted RMS | Weighted 95% CI | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["convergence_rows"]:
        lines.append(
            f"| {row['geometry']} | {row['k']} | {row['sections']} | {fmt(row['mean_rms'])} | "
            f"{interval(row['rms_95_percent_ci'])} | {fmt(row['mean_weighted_rms'])} | "
            f"{interval(row['weighted_rms_95_percent_ci'])} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Matched baseline comparison",
            "",
            "| Geometry | Method | RMS | Weighted RMS | RMS reduction | Weighted reduction |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["method_rows"]:
        lines.append(
            f"| {row['geometry']} | {row['method']} | {fmt(row['mean_rms'])} | "
            f"{fmt(row['mean_weighted_rms'])} | {row['unweighted_reduction_from_baseline']:.1%} | "
            f"{row['weighted_reduction_from_baseline']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## gCICY coefficient-family validation",
            "",
            "| Seed | Mean RMS | Weighted RMS | RMS reduction | Weighted reduction | Min eigenvalue |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["family_rows"]:
        lines.append(
            f"| {row['model_seed']} | {fmt(row['mean_candidate_rms'])} | "
            f"{fmt(row['mean_candidate_weighted_rms'])} | "
            f"{row['mean_unweighted_reduction_fraction']:.1%} | "
            f"{row['mean_weighted_reduction_fraction']:.1%} | "
            f"{row['min_metric_eigenvalue']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Geometric observables",
            "",
            "| Geometry | Metric | Observable | Estimate | Seed SE | 95% CI | Exact | In CI |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["observable_rows"]:
        lines.append(
            f"| {row['geometry']} | {row['metric']} | {row['observable']} | "
            f"{fmt(row['estimate'])} | {fmt(row['seed_standard_error'])} | "
            f"{interval(row['normal_95_percent_ci'])} | {fmt(row['exact'])} | "
            f"{'yes' if row['exact_in_95_percent_ci'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## gCICY k=3 hard-region stress test",
            "",
            "| Region | Points | Median | P95 | P99 | Maximum | RMS |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["hard_region_rows"]:
        lines.append(
            f"| {row['region']} | {row['count']} | {fmt(row['median'])} | "
            f"{fmt(row['p95'])} | {fmt(row['p99'])} | {fmt(row['max'])} | {fmt(row['rms'])} |"
        )
    lines.extend(
        [
            "",
            "## Completion gates",
            "",
            f"- gCICY k=1,2,3 paired convergence: **{summary['gates']['gcicy_k1_k2']}**",
            f"- gCICY k=3: **{summary['gates']['gcicy_k3']}**",
            f"- Five-model gCICY k=1 family: **{summary['gates']['gcicy_family']}**",
            f"- Bicubic k=1,2,3 paired convergence: **{summary['gates']['bicubic_k1_k2']}**",
            f"- Bicubic k=3: **{summary['gates']['bicubic_k3']}**",
            f"- Geometric observables: **{summary['gates']['observables']}**",
            f"- Hard-region stress audit: **{summary['gates']['hard_regions']}**",
            f"- Reproducible paper: **{summary['gates']['paper']}**",
            "",
        ]
    )
    return "\n".join(lines)


def latex(summary: dict) -> str:
    lines = [
        "% Generated by scripts/build_publication_evidence.py; do not edit.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Paired fresh-seed Monge--Ampere residuals.}",
        "\\label{tab:convergence}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Geometry & $k$ & $N_k$ & RMS & weighted RMS & status \\\\",
        "\\midrule",
    ]
    for row in summary["convergence_rows"]:
        status = {
            "accepted": "accepted",
            "rejected_diagnostic": "rejected diagnostic",
        }.get(row["status"], row["status"].replace("_", " "))
        lines.append(
            f"{row['geometry']} & {row['k']} & {row['sections']} & "
            f"{row['mean_rms']:.4f} & {row['mean_weighted_rms']:.4f} & {status} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Matched comparison with the ambient Fubini--Study baseline.}",
            "\\label{tab:methods}",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "Geometry & method & RMS & weighted RMS & reduction & weighted reduction \\\\",
            "\\midrule",
        ]
    )
    for row in summary["method_rows"]:
        method = {
            "ambient_fubini_study": "ambient FS",
            "global_H_k1_full": "full $H$, $k=1$",
            "global_H_k2_rank40": "rank-40 $H$, $k=2$",
            "global_H_k2_full": "full $H$, $k=2$",
            "global_H_k3_rank64_whitened": "rank-64 $H$, $k=3$",
            "global_H_k3_full_product_init": "full $H$, $k=3$",
        }.get(row["method"], row["method"].replace("_", " "))
        lines.append(
            f"{row['geometry']} & {method} & {row['mean_rms']:.4f} & "
            f"{row['mean_weighted_rms']:.4f} & "
            f"{100 * row['unweighted_reduction_from_baseline']:.1f}\\% & "
            f"{100 * row['weighted_reduction_from_baseline']:.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Fresh-seed $k=1$ metric audits across exact gCICY coefficient models.}",
            "\\label{tab:family}",
            "\\begin{tabular}{rrrrrr}",
            "\\toprule",
            "seed & RMS & weighted RMS & reduction & weighted reduction & $\\lambda_{\\min}$ \\\\",
            "\\midrule",
        ]
    )
    for row in summary["family_rows"]:
        lines.append(
            f"{row['model_seed']} & {row['mean_candidate_rms']:.4f} & "
            f"{row['mean_candidate_weighted_rms']:.4f} & "
            f"{100 * row['mean_unweighted_reduction_fraction']:.1f}\\% & "
            f"{100 * row['mean_weighted_reduction_fraction']:.1f}\\% & "
            f"{row['min_metric_eigenvalue']:.2e} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Exact-intersection observable checks for accepted $k=3$ metrics.}",
            "\\label{tab:observables}",
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "Geometry & observable & estimate & exact & relative error \\\\",
            "\\midrule",
        ]
    )
    for row in summary["observable_rows"]:
        observable = {
            "normalized_volume": "volume ratio",
            "normalized_slope_1": "slope ratio 1",
            "normalized_slope_2": "slope ratio 2",
            "normalized_slope_3": "slope ratio 3",
        }.get(row["observable"], row["observable"].replace("_", " "))
        lines.append(
            f"{row['geometry']} & {observable} & {row['estimate']:.4f} & "
            f"{row['exact']:.4f} & {100 * row['relative_error']:.2f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Absolute centered residuals in fixed-sample gCICY $k=3$ stress regions.}",
            "\\label{tab:hard-regions}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "region & $n$ & median & P95 & P99 & maximum \\\\",
            "\\midrule",
        ]
    )
    for row in summary["hard_region_rows"]:
        region = row["region"].replace("%", "\\%")
        lines.append(
            f"{region} & {row['count']} & {row['median']:.4f} & "
            f"{row['p95']:.4f} & {row['p99']:.4f} & {row['max']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    source_paths = [
        args.gcicy_convergence,
        args.bicubic_convergence,
        args.gcicy_k3_artifact,
        args.gcicy_k3_summary,
        args.gcicy_k3_audit,
        args.bicubic_k3_artifact,
        args.bicubic_k3_summary,
        args.bicubic_k3_audit,
        args.bicubic_rejected_k3,
        args.family,
        args.gcicy_observables,
        args.bicubic_observables,
        args.hard_region,
        args.hard_region_samples,
    ]
    gcicy = load(args.gcicy_convergence)
    bicubic = load(args.bicubic_convergence)
    gcicy_k3_audit = load(args.gcicy_k3_audit)
    bicubic_k3_audit = load(args.bicubic_k3_audit)
    bicubic_rejected = load(args.bicubic_rejected_k3)
    family = load(args.family)
    gcicy_observables = load(args.gcicy_observables)
    bicubic_observables = load(args.bicubic_observables)
    hard_region = load(args.hard_region)

    require(gcicy["strictly_decreasing_unweighted_on_every_seed"], "gCICY unweighted k1->k2->k3")
    require(gcicy["strictly_decreasing_weighted_on_every_seed"], "gCICY weighted k1->k2->k3")
    require([row["k"] for row in gcicy["aggregate"]] == [1, 2, 3], "gCICY accepted degrees")
    require(bicubic["strictly_decreasing_unweighted_on_every_seed"], "bicubic unweighted k1->k2->k3")
    require(bicubic["strictly_decreasing_weighted_on_every_seed"], "bicubic weighted k1->k2->k3")
    require([row["k"] for row in bicubic["aggregate"]] == [1, 2, 3], "bicubic accepted degrees")
    require(gcicy_k3_audit["all_passed"], "gCICY k3 fresh-seed audit")
    require(gcicy_k3_audit["projective_charts_seen"] == 24, "gCICY k3 projective atlas")
    require(gcicy_k3_audit["max_trained_h_chart_ma_error"] <= 1e-8, "gCICY k3 chart MA")
    require(gcicy_k3_audit["min_candidate_eigenvalue"] > 0, "gCICY k3 metric positivity")
    require(bicubic_k3_audit["all_passed"], "bicubic k3 fresh-seed audit")
    require(bicubic_k3_audit["projective_charts_seen"] == 9, "bicubic k3 projective atlas")
    require(bicubic_k3_audit["implicit_coordinate_choices_seen"] == 4, "bicubic k3 implicit atlas")
    require(bicubic_k3_audit["max_projective_ma_error"] <= 1e-8, "bicubic k3 projective MA")
    require(bicubic_k3_audit["max_implicit_ma_error"] <= 1e-8, "bicubic k3 implicit MA")
    require(not bicubic_rejected["strictly_decreasing_unweighted_on_every_seed"], "bicubic k3 rejection")
    require(not bicubic_rejected["strictly_decreasing_weighted_on_every_seed"], "bicubic weighted k3 rejection")
    require(family["all_models_passed"] and family["model_count"] >= 5, "five-model family")
    require(family["exact_chart_certificates"] >= 360, "complete exact family charts")
    require(family["total_fresh_seed_sets"] >= 40, "family fresh-seed coverage")
    require(hard_region["all_gates_passed"], "gCICY k3 hard-region geometry gates")
    require(hard_region["total_points"] >= 8192, "gCICY k3 hard-region sample size")

    accepted_convergence = convergence_rows("gCICY", gcicy) + convergence_rows("bicubic", bicubic)
    rejected_k3 = convergence_rows("bicubic", bicubic_rejected, accepted=False)[-1]
    convergence = accepted_convergence
    methods = baseline_and_method_rows(
        "gCICY",
        gcicy,
        {1: "global_H_k1_full", 2: "global_H_k2_rank40", 3: "global_H_k3_rank64_whitened"},
    ) + baseline_and_method_rows(
        "bicubic",
        bicubic,
        {1: "global_H_k1_full", 2: "global_H_k2_full", 3: "global_H_k3_full_product_init"},
    )
    observables = observable_rows("gCICY", gcicy_observables, "k3_3_3") + observable_rows(
        "bicubic", bicubic_observables, "k3"
    )
    require(all(row["exact_in_95_percent_ci"] for row in observables), "observable confidence intervals")
    stress_rows = hard_region_rows(hard_region)

    summary = {
        "description": "Machine-checked evidence index and generated result tables for the gCICY metric paper.",
        "source_files": [
            {"path": workspace_path(path), "sha256": file_sha256(path)}
            for path in source_paths
        ],
        "gates": {
            "gcicy_k1_k2": "passed_k1_k2_k3_on_every_paired_seed",
            "gcicy_k3": "passed_positive_complete_atlas",
            "gcicy_family": "passed_5_models_k1",
            "bicubic_k1_k2": "passed_k1_k2_k3_on_every_paired_seed",
            "bicubic_k3": "passed_positive_complete_atlas",
            "observables": "passed_exact_values_in_95_percent_ci",
            "hard_regions": "passed_8192_points_complete_atlas",
            "paper": "numerical_core_complete_manuscript_draft",
        },
        "convergence_rows": convergence,
        "rejected_diagnostics": [rejected_k3],
        "method_rows": methods,
        "k3_artifacts": {
            "gcicy": h_artifact_summary(args.gcicy_k3_artifact),
            "bicubic": h_artifact_summary(args.bicubic_k3_artifact),
        },
        "family_summary": {
            key: family[key]
            for key in (
                "model_count",
                "model_seeds",
                "exact_chart_certificates",
                "total_fresh_seed_sets",
                "candidate_rms_across_models",
                "candidate_weighted_rms_across_models",
                "minimum_metric_eigenvalue",
                "maximum_chart_ma_error",
            )
        },
        "family_rows": family["model_rows"],
        "observable_rows": observables,
        "hard_region_rows": stress_rows,
        "hard_region_summary": hard_region,
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
    print(json.dumps(summary["gates"], indent=2))


if __name__ == "__main__":
    main()
