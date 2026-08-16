#!/usr/bin/env python3
"""Summarize the blocked O(2) full-epoch energy-functional experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METRICS = (
    "sigma",
    "chi",
    "q999_abs_residual",
    "cvar99_abs_residual",
    "maximum_abs_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--l1-report", type=Path, required=True)
    parser.add_argument("--l2-report", type=Path, required=True)
    parser.add_argument("--continuation-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--tex-out", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def metric_row(statistics: dict[str, Any]) -> dict[str, float | int]:
    return {
        "n_points": int(statistics["n_points"]),
        "sigma": float(statistics["sigma_official_formula"]),
        "chi": float(statistics["weighted_rms_abs_residual"]),
        "q999_abs_residual": float(
            statistics["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99_abs_residual": float(
            statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum_abs_residual": float(
            statistics["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
        "nonpositive_metric_count": int(
            statistics["nonpositive_min_eigenvalue"]["count"]
        ),
    }


def experiment_row(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema") != "quintic-positive-tensor-network-same-points-v1":
        raise ValueError(f"unsupported TN report schema: {path}")
    architecture = report["architecture"]
    configuration = report["configuration"]
    if (
        int(architecture["source_degree"]) != 2
        or int(architecture["shared_dictionary_rank_q"]) != 375
        or int(architecture["bond_dimension_D"]) != 8
    ):
        raise RuntimeError(f"blocked O(2) architecture gate failed: {path}")
    if not report["training"].get("full_epoch_gradient"):
        raise RuntimeError(f"full-epoch gradient gate failed: {path}")
    if report["training"].get("fixed_log_kappa_source") != "registered_command_line_value":
        raise RuntimeError(f"registered-kappa gate failed: {path}")
    return {
        "source_degree": 2,
        "dictionary_rank": int(architecture["shared_dictionary_rank_q"]),
        "bond_dimension": int(architecture["bond_dimension_D"]),
        "stored_real_parameters": int(
            architecture["trainable_real_parameter_count"]
        ),
        "active_real_parameters": int(
            architecture["active_real_parameter_count"]
        ),
        "ma_loss_kind": report["training"]["ma_loss_kind"],
        "learning_rate": float(configuration["learning_rate"]),
        "best_epoch": int(report["training"]["best_epoch"]),
        "optimizer_steps": int(report["training"]["optimizer_steps"]),
        "fixed_log_kappa": float(report["training"]["fixed_log_kappa"]),
        "metrics": metric_row(report["blind_test"]["normalized_volume"]),
        "timing_seconds": report["timing_seconds"],
        "report": str(path),
        "report_sha256": sha256_file(path),
        "model_sha256": report["artifacts"]["model_sha256"],
        "blind_points_sha256": report["common_point_evidence"]["source_sha256"][
            "blind_points"
        ],
        "blind_pullbacks_sha256": report["common_point_evidence"][
            "pullback_sha256"
        ]["blind"],
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    base_path, base_summary = load(args.base_summary)
    l1_path, l1_report = load(args.l1_report)
    l2_path, l2_report = load(args.l2_report)
    continuation_path, continuation_report = load(args.continuation_report)
    if base_summary.get("schema") != "quintic-tn-residual-source-arms-summary-v1":
        raise ValueError("unexpected base summary schema")

    source_base = base_summary["models"]["base"]
    base = {
        "source_degree": int(source_base["source_degree"]),
        "stored_real_parameters": int(source_base["total_real_parameters"]),
        "active_real_parameters": int(source_base["total_real_parameters"]),
        "metrics": source_base["metrics"],
        "report": source_base["report"],
        "report_sha256": source_base["report_sha256"],
        "model_sha256": source_base["model_sha256"],
        "blind_points_sha256": source_base["blind_points_sha256"],
        "blind_pullbacks_sha256": source_base["blind_pullbacks_sha256"],
    }
    models = {
        "o1_base": base,
        "o2_full_epoch_l1": experiment_row(l1_path, l1_report),
        "o2_full_epoch_l2": experiment_row(l2_path, l2_report),
        "o2_l2_continuation": experiment_row(
            continuation_path, continuation_report
        ),
    }
    common_hashes = {
        (row["blind_points_sha256"], row["blind_pullbacks_sha256"])
        for row in models.values()
    }
    if len(common_hashes) != 1:
        raise RuntimeError("models do not share the registered blind arrays")
    point_counts = {int(row["metrics"]["n_points"]) for row in models.values()}
    if point_counts != {200000}:
        raise RuntimeError(f"unexpected blind point counts: {point_counts}")

    base_metrics = base["metrics"]
    relative = {
        name: {
            metric: float(row["metrics"][metric]) / float(base_metrics[metric]) - 1.0
            for metric in METRICS
        }
        for name, row in models.items()
        if name != "o1_base"
    }
    final = models["o2_l2_continuation"]["metrics"]
    interpretation = (
        "bulk_and_q999_improve_cvar_flat_max_worse"
        if all(float(final[key]) < float(base_metrics[key]) for key in METRICS[:3])
        and float(final["maximum_abs_residual"])
        > float(base_metrics["maximum_abs_residual"])
        else "mixed"
    )
    return {
        "schema": "quintic-tn-o2-full-epoch-energy-summary-v1",
        "base_summary": str(base_path),
        "base_summary_sha256": sha256_file(base_path),
        "common_blind": {
            "n_points": 200000,
            "blind_points_sha256": base["blind_points_sha256"],
            "blind_pullbacks_sha256": base["blind_pullbacks_sha256"],
        },
        "models": models,
        "relative_metric_changes_vs_o1_base": relative,
        "interpretation": interpretation,
    }


def markdown(summary: dict[str, Any]) -> str:
    labels = {
        "o1_base": "O1 base",
        "o2_full_epoch_l1": "blocked O2, full-epoch L1",
        "o2_full_epoch_l2": "blocked O2, full-epoch L2",
        "o2_l2_continuation": "blocked O2, L2 continuation",
    }
    lines = [
        "# Blocked O(2) full-epoch energy experiment",
        "",
        "| model | active params | sigma | chi | q999 | CVaR99 | maximum | nonpositive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["models"].items():
        value = row["metrics"]
        lines.append(
            f"| {labels[name]} | {row['active_real_parameters']:,} | "
            f"{value['sigma']:.8g} | {value['chi']:.8g} | "
            f"{value['q999_abs_residual']:.8g} | "
            f"{value['cvar99_abs_residual']:.8g} | "
            f"{value['maximum_abs_residual']:.8g} | "
            f"{value['nonpositive_metric_count']} |"
        )
    lines.extend(["", f"Interpretation: `{summary['interpretation']}`.", ""])
    return "\n".join(lines)


def latex(summary: dict[str, Any]) -> str:
    labels = {
        "o1_base": r"O1 base",
        "o2_full_epoch_l1": r"blocked O2, full-epoch $L^1$",
        "o2_full_epoch_l2": r"blocked O2, full-epoch $L^2$",
        "o2_l2_continuation": r"blocked O2, $L^2$ continuation",
    }
    rows = []
    for name, row in summary["models"].items():
        value = row["metrics"]
        rows.append(
            f"    {labels[name]} & {row['active_real_parameters']:,} & "
            f"{value['sigma']:.4g} & {value['chi']:.4g} & "
            f"{value['q999_abs_residual']:.4g} & "
            f"{value['cvar99_abs_residual']:.4g} & "
            f"{value['maximum_abs_residual']:.4g} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"  \centering",
            r"  \small",
            r"  \setlength{\tabcolsep}{4.5pt}",
            r"  \begin{tabular}{lrrrrrr}",
            r"    \toprule",
            r"    model & active parameters & $\sigma$ & $\chi$ & $q_{99.9}$ & CVaR$_{99}$ & max \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \caption{Blocked degree-two source experiments on the common 200,000-point blind pool. The registered volume normalization and full-epoch gradient remove the small-pool normalization bias and minibatch update noise.}",
            r"  \label{tab:o2-full-epoch-energy}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown(summary), encoding="utf-8")
    if args.tex_out is not None:
        args.tex_out.parent.mkdir(parents=True, exist_ok=True)
        args.tex_out.write_text(latex(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
