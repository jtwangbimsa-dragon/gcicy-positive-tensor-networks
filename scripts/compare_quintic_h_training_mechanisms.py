#!/usr/bin/env python3
"""Summarize symmetry-free quintic H-metric training-mechanism runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="May be repeated; RUN_DIR must contain a mechanism-v1 report.json.",
    )
    parser.add_argument("--old-h-run-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_run_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"run must have LABEL=RUN_DIR form: {value!r}")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError(f"run must have LABEL=RUN_DIR form: {value!r}")
    return label, Path(raw_path).expanduser().resolve()


def best_history_row(report: dict[str, Any]) -> dict[str, Any]:
    best_epoch = int(report["training"]["best_epoch"])
    rows = [
        row
        for row in report["training"]["history"]
        if int(row["epoch"]) == best_epoch
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one history row for best epoch {best_epoch}")
    return rows[0]


def summarize_run(label: str, run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    report = load_json(report_path)
    if report.get("schema") != "quintic-h-cymetric-style-mechanism-v1":
        raise ValueError(f"unsupported report schema in {report_path}")
    if report["scientific_scope"].get("symmetry_usage") != "none":
        raise ValueError(f"mechanism run unexpectedly used symmetry: {report_path}")

    configuration = report["configuration"]
    training = report["training"]
    best = best_history_row(report)
    validation = best["validation"]["audit_normalized"]
    probe = best.get("diagnostic_probe")
    old_tail_probe = None
    if probe is not None:
        old_tail_probe = probe["selected_center_groups"]["raw_full_h"][
            "fixed_ratio"
        ]
    blind = report.get("blind")
    blind_summary = None
    if blind is not None:
        blind_audit = blind["audit_normalized"]
        blind_summary = {
            "sigma": blind_audit["sigma_official_formula"],
            "weighted_rms_abs_residual": blind_audit[
                "weighted_rms_abs_residual"
            ],
            "ratio_q0.999": blind_audit["ratio_unweighted_quantiles"][
                "q0.9990"
            ],
            "ratio_q0.9999": blind_audit["ratio_unweighted_quantiles"][
                "q0.9999"
            ],
            "ratio_maximum": blind_audit["ratio_unweighted_quantiles"][
                "q1.0000"
            ],
            "count_ratio_above_1.5": blind_audit["ratio_upper_tails"]["1.5"][
                "count"
            ],
        }
    train_count = int(report["data"]["train_count"])
    epochs = int(configuration["epochs"])
    return {
        "label": label,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "batch_size": int(configuration["batch_size"]),
        "epochs": epochs,
        "optimizer_updates": int(training["total_updates"]),
        "training_point_presentations": train_count * epochs,
        "learning_rate": configuration["learning_rate"],
        "gradient_clip": configuration["gradient_clip"],
        "target_normalization": configuration["target_normalization"],
        "loss_weighting": configuration["loss_weighting"],
        "batch_reduction": configuration["batch_reduction"],
        "best_epoch": int(training["best_epoch"]),
        "validation": {
            "sigma": validation["sigma_importance_weighted"],
            "l2": validation["l2_importance_weighted"],
            "ratio_q0.999": validation["ratio_unweighted_quantiles"]["q0.999"],
            "ratio_maximum": validation["ratio_unweighted_quantiles"]["maximum"],
            "h_condition_number": best["validation"]["h_matrix"][
                "condition_number"
            ],
        },
        "old_raw_h_tail_centers": (
            {
                "count": old_tail_probe["count_ratio_above_1.5"]
                + old_tail_probe["count_ratio_below_0.5"]
                if old_tail_probe is not None
                else None,
                "median_abs_residual": old_tail_probe[
                    "abs_residual_unweighted_quantiles"
                ]["median"],
                "maximum_abs_residual": old_tail_probe[
                    "abs_residual_unweighted_quantiles"
                ]["maximum"],
            }
            if old_tail_probe is not None
            else None
        ),
        "blind_200k": blind_summary,
        "timing_seconds": report["timing_seconds"],
    }


def summarize_old_h(run_dir: Path) -> dict[str, Any]:
    report = load_json(run_dir / "report.json")
    old = report["blind"]["our_full_h"]
    return {
        "label": "old_full_group_H",
        "run_dir": str(run_dir),
        "blind_200k": {
            "sigma": old["sigma_official_formula"],
            "weighted_rms_abs_residual": old["weighted_rms_abs_residual"],
            "ratio_q0.999": old["ratio_unweighted_quantiles"]["q0.9990"],
            "ratio_q0.9999": old["ratio_unweighted_quantiles"]["q0.9999"],
            "ratio_maximum": old["ratio_unweighted_quantiles"]["q1.0000"],
            "count_ratio_above_1.5": old["ratio_upper_tails"]["1.5"]["count"],
        },
    }


def markdown_table(
    runs: list[dict[str, Any]],
    references: dict[str, Any],
) -> str:
    lines = [
        "# Symmetry-free quintic H-training mechanism comparison",
        "",
        "All mechanism rows use the exact common Fermat-quintic data. Tail probes are diagnostic only.",
        "",
        "| Run | Batch | Updates | Point presentations | Val sigma | Blind sigma | Blind max r | Old-tail median | cond(H) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        blind = run["blind_200k"]
        probe = run["old_raw_h_tail_centers"]
        lines.append(
            "| {label} | {batch_size} | {optimizer_updates} | "
            "{training_point_presentations} | {val_sigma:.6g} | {blind_sigma:.6g} | "
            "{blind_max:.6g} | {probe_median:.6g} | {condition:.6g} |".format(
                **run,
                val_sigma=run["validation"]["sigma"],
                blind_sigma=blind["sigma"],
                blind_max=blind["ratio_maximum"],
                probe_median=probe["median_abs_residual"],
                condition=run["validation"]["h_condition_number"],
            )
        )
    if references:
        lines.extend(["", "## References", ""])
        old = references.get("old_full_group_H")
        if old is not None:
            blind = old["blind_200k"]
            lines.append(
                "- Old full-group H: blind sigma `{:.6g}`, maximum ratio `{:.6g}`, "
                "count `r>1.5` `{}`.".format(
                    blind["sigma"],
                    blind["ratio_maximum"],
                    blind["count_ratio_above_1.5"],
                )
            )
        cymetric = references.get("official_cymetric")
        if cymetric is not None:
            lines.append(
                "- Official cymetric: blind sigma `{:.6g}`, maximum ratio `{:.6g}`.".format(
                    cymetric["sigma"], cymetric["ratio_maximum"]
                )
            )
    lines.extend(
        [
            "",
            "The finite common audit is empirical and is not a global sup-norm certificate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    parsed = [parse_run_argument(value) for value in args.run]
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise ValueError("run labels must be unique")
    runs = [summarize_run(label, run_dir) for label, run_dir in parsed]
    runs.sort(key=lambda row: row["batch_size"])

    references: dict[str, Any] = {}
    if args.old_h_run_dir is not None:
        old_dir = args.old_h_run_dir.expanduser().resolve()
        references["old_full_group_H"] = summarize_old_h(old_dir)
    first_report = load_json(Path(runs[0]["report_path"]))
    official = first_report["official_cymetric_reference"]
    references["official_cymetric"] = {
        "sigma": official["sigma_official_formula"],
        "ratio_maximum": official["ratio_unweighted_quantiles"]["q1.0000"],
    }

    payload = {
        "schema": "quintic-h-training-mechanism-comparison-v1",
        "claim_scope": {
            "symmetry_usage": "none",
            "classification": "paired empirical mechanism comparison",
            "warning": "The common 200k audit is not a global supremum certificate.",
        },
        "runs": runs,
        "references": references,
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    if output_json.exists() or output_markdown.exists():
        raise FileExistsError("comparison outputs already exist")
    write_json(output_json, payload)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(
        markdown_table(runs, references),
        encoding="utf-8",
    )
    print(f"wrote {output_json}")
    print(f"wrote {output_markdown}")


if __name__ == "__main__":
    main()
