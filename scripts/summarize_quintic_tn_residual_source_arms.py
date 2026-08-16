#!/usr/bin/env python3
"""Summarize the preregistered matched-budget O(1)/O(2) residual arms."""

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
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--degree-one-report", type=Path, required=True)
    parser.add_argument("--degree-two-report", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--blocking-audit", type=Path, required=True)
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


def metrics(statistics: dict[str, Any]) -> dict[str, float | int]:
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


def arm_row(
    path: Path,
    report: dict[str, Any],
    *,
    expected_source_degree: int,
    expected_parameters: int,
) -> dict[str, Any]:
    if report.get("schema") != "quintic-positive-tensor-network-direct-sum-report-v1":
        raise ValueError(f"unsupported residual report schema: {path}")
    architecture = report["architecture"]
    configurations = architecture["branch_configurations"]
    if int(configurations[1]["source_degree"]) != expected_source_degree:
        raise RuntimeError(f"residual source-degree gate failed: {path}")
    if int(architecture["total_real_parameter_count"]) != expected_parameters:
        raise RuntimeError(f"residual parameter-count gate failed: {path}")
    nesting = report["training"]["zero_weight_nesting_check"]
    if nesting is None or max(
        float(nesting["maximum_absolute_potential_error"]),
        float(nesting["maximum_absolute_metric_entry_error"]),
    ) > float(nesting["tolerance"]):
        raise RuntimeError(f"zero-weight nesting gate failed: {path}")
    blind = report.get("blind_test")
    if blind is None:
        raise RuntimeError(f"formal residual arm has no blind audit: {path}")
    return {
        "source_degree": expected_source_degree,
        "total_real_parameters": int(architecture["total_real_parameter_count"]),
        "active_real_parameters": int(architecture["active_real_parameter_count"]),
        "best_epoch": int(report["training"]["best_epoch"]),
        "branch_weights": architecture["branch_weights"],
        "validation_residual_fraction_mean": float(
            report["training"]["validation_branch_fractions"]["weighted_means"][1]
        ),
        "zero_weight_nesting_check": nesting,
        "metrics": metrics(blind["normalized_volume"]),
        "optimizer_steps": int(report["training"]["optimizer_steps"]),
        "timing_seconds": report["timing_seconds"],
        "report": str(path),
        "report_sha256": sha256_file(path),
        "model_sha256": report["artifacts"]["model_sha256"],
        "blind_points_sha256": report["common_point_evidence"]["source_sha256"][
            "blind_points"
        ],
        "blind_pullbacks_sha256": report["common_point_evidence"]["pullback_sha256"][
            "blind"
        ],
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    base_path, base = load(args.base_report)
    one_path, one = load(args.degree_one_report)
    two_path, two = load(args.degree_two_report)
    prereg_path, prereg = load(args.preregistration)
    blocking_path, blocking = load(args.blocking_audit)
    if base.get("schema") != "quintic-tn-two-site-lm-v1":
        raise ValueError("base report is not the registered two-site model")
    if prereg.get("schema") != "quintic-tn-residual-source-arms-preregistration-v1":
        raise ValueError("unexpected residual-arm preregistration schema")
    if blocking.get("schema") != "positive-tensor-network-pair-blocking-audit-v1":
        raise ValueError("unexpected blocking-audit schema")

    base_metrics = metrics(base["benchmark"]["final"])
    base_row = {
        "source_degree": 1,
        "total_real_parameters": int(
            base["architecture"]["stored_learned_real_parameters"]
        ),
        "metrics": base_metrics,
        "report": str(base_path),
        "report_sha256": sha256_file(base_path),
        "model_sha256": base["artifacts"]["model_sha256"],
        "blind_points_sha256": base["source"]["blind_points_sha256"],
        "blind_pullbacks_sha256": base["source"]["blind_pullbacks_sha256"],
    }
    one_row = arm_row(
        one_path,
        one,
        expected_source_degree=1,
        expected_parameters=int(
            prereg["arms"]["degree_one_residual"]["total_real_parameters"]
        ),
    )
    two_row = arm_row(
        two_path,
        two,
        expected_source_degree=2,
        expected_parameters=int(
            prereg["arms"]["degree_two_rectangular_residual"][
                "total_real_parameters"
            ]
        ),
    )
    rows = {"base": base_row, "degree_one_residual": one_row, "degree_two_residual": two_row}
    hashes = {
        (row["blind_points_sha256"], row["blind_pullbacks_sha256"])
        for row in rows.values()
    }
    if len(hashes) != 1:
        raise RuntimeError("residual arms do not use the base blind arrays")
    if any(row["metrics"]["n_points"] != base_metrics["n_points"] for row in rows.values()):
        raise RuntimeError("residual-arm blind point counts differ")

    changes = {
        name: {
            key: float(row["metrics"][key]) / float(base_metrics[key]) - 1.0
            for key in METRICS
        }
        for name, row in rows.items()
        if name != "base"
    }
    two_beats_one_all = all(
        float(two_row["metrics"][key]) < float(one_row["metrics"][key])
        for key in METRICS
    )
    one_beats_base = float(one_row["metrics"]["sigma"]) < float(base_metrics["sigma"])
    two_beats_base = float(two_row["metrics"]["sigma"]) < float(base_metrics["sigma"])
    if two_beats_base and two_beats_one_all:
        interpretation = "degree_two_wins_bulk_and_tail"
    elif one_beats_base and float(one_row["metrics"]["sigma"]) <= float(
        two_row["metrics"]["sigma"]
    ):
        interpretation = "degree_one_matches_or_wins"
    else:
        interpretation = "neither_beats_base"

    return {
        "schema": "quintic-tn-residual-source-arms-summary-v1",
        "preregistration": str(prereg_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "blocking_audit": str(blocking_path),
        "blocking_audit_sha256": sha256_file(blocking_path),
        "common_blind": {
            "n_points": base_metrics["n_points"],
            "blind_points_sha256": base_row["blind_points_sha256"],
            "blind_pullbacks_sha256": base_row["blind_pullbacks_sha256"],
        },
        "models": rows,
        "relative_metric_changes_vs_base": changes,
        "registered_interpretation": interpretation,
        "blocking_preflight": {
            "exact_rectangular_rank": blocking["shared_dictionary"][
                "exact_rectangular_block"
            ]["numerical_rank"],
            "aggregate_antisymmetric_output_fraction": blocking["blocking"][
                "aggregate_antisymmetric_output_frobenius_fraction"
            ],
            "fixed_dictionary_exact_core_parameters": blocking["parameter_budget"][
                "exact_rectangular_core_real_parameters_with_fixed_dictionary"
            ],
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    labels = {
        "base": "frozen O1 base",
        "degree_one_residual": "+ complete O1 residual",
        "degree_two_residual": "+ complete rectangular O2 residual",
    }
    lines = [
        "# Matched-budget residual-source experiment",
        "",
        "| model | parameters | sigma | chi | q999 | CVaR99 | maximum | nonpositive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["models"].items():
        value = row["metrics"]
        lines.append(
            f"| {labels[name]} | {row['total_real_parameters']} | "
            f"{value['sigma']:.8g} | {value['chi']:.8g} | "
            f"{value['q999_abs_residual']:.8g} | "
            f"{value['cvar99_abs_residual']:.8g} | "
            f"{value['maximum_abs_residual']:.8g} | "
            f"{value['nonpositive_metric_count']} |"
        )
    lines.extend(
        [
            "",
            f"Registered interpretation: `{summary['registered_interpretation']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def latex(summary: dict[str, Any]) -> str:
    labels = {
        "base": "frozen O1 base",
        "degree_one_residual": r"$+$ complete O1 residual",
        "degree_two_residual": r"$+$ rectangular O2 residual",
    }
    rows = []
    for name, row in summary["models"].items():
        value = row["metrics"]
        rows.append(
            f"    {labels[name]} & {row['total_real_parameters']:,} & "
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
            r"  \begin{tabular}{lrrrrrr}",
            r"    \toprule",
            r"    model & parameters & $\sigma$ & $\chi$ & $q_{99.9}$ & CVaR$_{99}$ & max \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \caption{Preregistered residual-source comparison on the common blind pool. The O1 base is frozen; the two added branches have nearly matched parameter and update budgets.}",
            r"  \label{tab:residual-source-arms}",
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
    print(args.out.resolve())


if __name__ == "__main__":
    main()
