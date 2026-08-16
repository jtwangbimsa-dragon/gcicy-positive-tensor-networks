#!/usr/bin/env python3
"""Build the checked near-matched-budget TN/full-H diagnostic table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "sigma",
    "chi",
    "q999_abs_residual",
    "cvar99_abs_residual",
    "maximum_abs_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tn-report", type=Path, required=True)
    parser.add_argument("--full-h-training-report", type=Path, required=True)
    parser.add_argument("--full-h-audit-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--tex-out", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def extract_metrics(statistics: dict[str, Any]) -> dict[str, float | int]:
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


def build_comparison(
    tn_path: Path,
    tn_report: dict[str, Any],
    full_h_training_path: Path,
    full_h_training: dict[str, Any],
    full_h_audit_path: Path,
    full_h_audit: dict[str, Any],
) -> dict[str, Any]:
    if tn_report.get("schema") != "quintic-tn-two-site-lm-v1":
        raise ValueError("TN report is not the registered two-site result")
    if full_h_training.get("schema") != "quintic-h-cymetric-style-mechanism-v1":
        raise ValueError("full-H training report has an unsupported schema")
    if full_h_audit.get("schema") != "quintic-saved-h-common-blind-audit-v1":
        raise ValueError("full-H audit report has an unsupported schema")

    tn_metrics = extract_metrics(tn_report["benchmark"]["final"])
    full_h_metrics = extract_metrics(full_h_audit["blind"]["audited_h"])
    if tn_metrics["n_points"] != full_h_metrics["n_points"]:
        raise RuntimeError("TN and full-H blind point counts differ")

    tn_source = tn_report["source"]
    full_h_hashes = full_h_audit["artifacts_sha256"]
    common_hashes = {
        "blind_points_sha256": (
            tn_source["blind_points_sha256"], full_h_hashes["blind_points"]
        ),
        "blind_pullbacks_sha256": (
            tn_source["blind_pullbacks_sha256"], full_h_hashes["blind_pullbacks"]
        ),
    }
    mismatches = {
        key: values for key, values in common_hashes.items() if values[0] != values[1]
    }
    if mismatches:
        raise RuntimeError(f"common blind benchmark hashes differ: {mismatches}")

    selected_h_hash = full_h_training["artifacts"]["best_h_sha256"]
    audited_h_hash = full_h_hashes["h_artifact"]
    if selected_h_hash != audited_h_hash:
        raise RuntimeError("full-H blind audit does not use the selected best H")

    tn_architecture = tn_report["architecture"]
    full_h_basis = full_h_training["basis"]
    tn_parameters = int(tn_architecture["stored_learned_real_parameters"])
    full_h_parameters = int(full_h_basis["registered_real_parameter_count"])
    changes = {
        key: float(tn_metrics[key]) / float(full_h_metrics[key]) - 1.0
        for key in METRIC_KEYS
    }
    history = full_h_training["training"]["history"]
    best_epoch = int(full_h_training["training"]["best_epoch"])
    last_evaluated_epoch = int(history[-1]["epoch"])

    return {
        "schema": "quintic-tn-full-h-near-matched-budget-v1",
        "scientific_scope": {
            "kind": "near-matched stored-parameter diagnostic",
            "common_geometry_and_blind_measure": True,
            "common_blind_arrays_verified": True,
            "not_matched": [
                "polarization degree",
                "training objective",
                "optimization algorithm",
                "training path and compute budget",
            ],
            "full_h_convergence_warning": (
                "The selected full-H checkpoint occurs at the last evaluated epoch; "
                "this is a finite-budget baseline, not a certified optimum."
            ),
        },
        "common_benchmark": {
            "n_points": tn_metrics["n_points"],
            "blind_points_sha256": common_hashes["blind_points_sha256"][0],
            "blind_pullbacks_sha256": common_hashes["blind_pullbacks_sha256"][0],
        },
        "models": {
            "positive_tn": {
                "degree": int(tn_architecture["site_count"]),
                "stored_learned_real_parameters": tn_parameters,
                "active_real_parameters": int(
                    tn_architecture["active_real_parameters"]
                ),
                "metrics": tn_metrics,
                "report": str(tn_path),
                "report_sha256": sha256_file(tn_path),
                "model_artifact_sha256": tn_report.get("artifacts", {}).get(
                    "model_sha256"
                ),
            },
            "dense_full_h": {
                "degree": int(full_h_basis["degree"]),
                "section_count": int(full_h_basis["section_count"]),
                "registered_real_parameters": full_h_parameters,
                "metrics": full_h_metrics,
                "condition_number": float(
                    full_h_audit["h_spectrum"]["condition_number"]
                ),
                "best_epoch": best_epoch,
                "last_evaluated_epoch": last_evaluated_epoch,
                "best_is_last_evaluated_epoch": best_epoch == last_evaluated_epoch,
                "training_report": str(full_h_training_path),
                "training_report_sha256": sha256_file(full_h_training_path),
                "audit_report": str(full_h_audit_path),
                "audit_report_sha256": sha256_file(full_h_audit_path),
                "h_artifact_sha256": audited_h_hash,
            },
        },
        "comparison": {
            "tn_parameter_fraction_of_full_h": tn_parameters / full_h_parameters,
            "tn_relative_parameter_change_vs_full_h": (
                tn_parameters / full_h_parameters - 1.0
            ),
            "tn_relative_metric_changes_vs_full_h": changes,
        },
    }


def markdown_table(summary: dict[str, Any]) -> str:
    tn = summary["models"]["positive_tn"]
    full_h = summary["models"]["dense_full_h"]

    def row(label: str, model: dict[str, Any], parameter_key: str) -> str:
        metric = model["metrics"]
        return (
            f"| {label} | {model['degree']} | {model[parameter_key]} | "
            f"{metric['sigma']:.8g} | {metric['chi']:.8g} | "
            f"{metric['q999_abs_residual']:.8g} | "
            f"{metric['cvar99_abs_residual']:.8g} | "
            f"{metric['maximum_abs_residual']:.8g} | "
            f"{metric['nonpositive_metric_count']} |"
        )

    sigma_change = summary["comparison"]["tn_relative_metric_changes_vs_full_h"][
        "sigma"
    ]
    parameter_change = summary["comparison"][
        "tn_relative_parameter_change_vs_full_h"
    ]
    return "\n".join(
        [
            "# Near-matched-budget TN/full-H diagnostic",
            "",
            "| Model | degree | stored/registered parameters | sigma | chi | q999 | CVaR99 | maximum | nonpositive |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            row("positive TN, D12 + two-site", tn, "stored_learned_real_parameters"),
            row("dense full-H energy baseline", full_h, "registered_real_parameters"),
            "",
            f"The TN uses {abs(parameter_change):.2%} fewer stored learned parameters and "
            f"its blind sigma is {abs(sigma_change):.2%} lower in this finite-budget diagnostic.",
            "",
            "The blind points and pullbacks are byte-identical. The rows do not match "
            "degree, objective, optimizer, or training compute. The full-H best checkpoint "
            "is the last evaluated epoch, so this is not evidence against the converged "
            "dense energy optimum.",
        ]
    ) + "\n"


def latex_table(summary: dict[str, Any]) -> str:
    tn = summary["models"]["positive_tn"]
    full_h = summary["models"]["dense_full_h"]

    def values(model: dict[str, Any], parameter_key: str) -> str:
        metric = model["metrics"]
        return (
            f"{model['degree']} & {model[parameter_key]:,} & "
            f"{metric['sigma']:.4g} & {metric['chi']:.4g} & "
            f"{metric['q999_abs_residual']:.4g} & "
            f"{metric['cvar99_abs_residual']:.4g} & "
            f"{metric['maximum_abs_residual']:.4g}"
        )

    return "\n".join(
        [
            r"\begin{table}[H]",
            r"  \centering",
            r"  \small",
            r"  \begin{tabular}{lrrrrrrr}",
            r"    \toprule",
            r"    model & $k$ & parameters & $\sigma$ & $\chi$ & $q_{99.9}$ & CVaR$_{99}$ & max \\",
            r"    \midrule",
            "    TN, D12 + two-site & "
            + values(tn, "stored_learned_real_parameters")
            + r" \\",
            "    dense full-$H$ energy & "
            + values(full_h, "registered_real_parameters")
            + r" \\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \caption{Near-matched stored-parameter diagnostic on the same 200,000 blind points and pullbacks. Degrees, objectives, optimizers, and training compute are not matched. The full-$H$ checkpoint is a finite-budget baseline selected at its last evaluated epoch, not a converged dense optimum.}",
            r"  \label{tab:tn-full-h-budget}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    tn_path, tn_report = load_json(args.tn_report)
    training_path, training_report = load_json(args.full_h_training_report)
    audit_path, audit_report = load_json(args.full_h_audit_report)
    summary = build_comparison(
        tn_path,
        tn_report,
        training_path,
        training_report,
        audit_path,
        audit_report,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_table(summary), encoding="utf-8")
    if args.tex_out is not None:
        args.tex_out.parent.mkdir(parents=True, exist_ok=True)
        args.tex_out.write_text(latex_table(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
