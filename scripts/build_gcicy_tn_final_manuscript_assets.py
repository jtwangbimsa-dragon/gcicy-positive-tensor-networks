#!/usr/bin/env python3
"""Generate final paper tables directly from frozen numerical summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_X11_SUMMARY = (
    ROOT
    / "outputs/pipeline/type11_x11_equal_time_final_20260807/final_summary.json"
)
DEFAULT_X11_H2_START = DEFAULT_X11_SUMMARY.parent / "h2_start.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x11-summary", type=Path, default=DEFAULT_X11_SUMMARY)
    parser.add_argument("--x11-h2-start", type=Path, default=DEFAULT_X11_H2_START)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mean_sd(metric: dict) -> str:
    return (
        f"{float(metric['mean']):.5f}"
        f"\\pm{float(metric['sample_standard_deviation']):.5f}"
    )


def build_x11_table(summary: dict, h2_start: dict) -> str:
    if summary["point_count"] != 200_000 or summary["replicate_count"] != 3:
        raise ValueError("the final X11 table requires the frozen 3 x 200,000 protocol")
    if not summary["all_sampled_metrics_positive"]:
        raise ValueError("the X11 positivity check failed")
    if h2_start["point_count"] != summary["point_count"]:
        raise ValueError("the H2 start and learned models use different point counts")
    if not h2_start["metrics"]["metric_geometry_evidence_available"]:
        raise ValueError("the H2 start lacks metric-positivity evidence")

    models = summary["aggregate"]
    h2_metrics = h2_start["metrics"]
    rows = (
        (
            r"common $H_2$ start",
            "2,025",
            f"${float(h2_metrics['sigma']):.5f}$",
            f"${float(h2_metrics['chi']):.5f}$",
            f"${float(h2_metrics['minimum_metric_eigenvalue']):.5f}$",
        ),
        (
            r"positive $k=6,D=5$ TN",
            "141,750",
            f"${mean_sd(models['positive_TN']['sigma'])}$",
            f"${mean_sd(models['positive_TN']['chi'])}$",
            f"${mean_sd(models['positive_TN']['minimum_metric_eigenvalue'])}$",
        ),
        (
            r"neural $\phi$-model",
            "142,626",
            f"${mean_sd(models['source_density_residual_phi']['sigma'])}$",
            f"${mean_sd(models['source_density_residual_phi']['chi'])}$",
            f"${mean_sd(models['source_density_residual_phi']['minimum_metric_eigenvalue'])}$",
        ),
        (
            r"direct $H_6$, extended to plateau",
            "758,641",
            f"${mean_sd(models['cold_full_H6_equal_time_plateau']['sigma'])}$",
            f"${mean_sd(models['cold_full_H6_equal_time_plateau']['chi'])}$",
            f"${mean_sd(models['cold_full_H6_equal_time_plateau']['minimum_metric_eigenvalue'])}$",
        ),
    )

    tail_rows = (
        (
            r"positive $k=6,D=5$ TN",
            mean_sd(models["positive_TN"]["absolute_log_ratio_q999"]),
            mean_sd(models["positive_TN"]["absolute_log_ratio_cvar_1pct"]),
            mean_sd(models["positive_TN"]["maximum_absolute_log_ratio"]),
        ),
        (
            r"neural $\phi$-model",
            mean_sd(models["source_density_residual_phi"]["absolute_log_ratio_q999"]),
            mean_sd(models["source_density_residual_phi"]["absolute_log_ratio_cvar_1pct"]),
            mean_sd(models["source_density_residual_phi"]["maximum_absolute_log_ratio"]),
        ),
        (
            r"direct $H_6$, extended to plateau",
            mean_sd(models["cold_full_H6_equal_time_plateau"]["absolute_log_ratio_q999"]),
            mean_sd(models["cold_full_H6_equal_time_plateau"]["absolute_log_ratio_cvar_1pct"]),
            mean_sd(models["cold_full_H6_equal_time_plateau"]["maximum_absolute_log_ratio"]),
        ),
    )

    lines = [
        "% Generated from the frozen X11 equal-time final summary.",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"model & trainable real parameters & $\sigma$ & $\chi$ & $\lambda_{\min}(g)$ \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            "",
            r"\begin{tabular}{@{}lrrr@{}}",
            r"\toprule",
            r"model & $Q_{0.999}(|\log r|)$ & $\operatorname{CVaR}_{1\%}(|\log r|)$ & $\max |\log r|$ \\",
            r"\midrule",
        ]
    )
    lines.extend(
        f"{label} & ${q999}$ & ${cvar}$ & ${maximum}$ \\\\" 
        for label, q999, cvar, maximum in tail_rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.x11_summary.read_text(encoding="utf-8"))
    h2_start = json.loads(args.x11_h2_start.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "x11_equal_time_final_20260807.tex"
    path.write_text(build_x11_table(summary, h2_start), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
