#!/usr/bin/env python3
"""Build the expanded X21 k-D development table and derived comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "outputs/pipeline/type21_kd_plateau_grid_floor1e10_20260731"
FOLLOWUPS = ROOT / "outputs/pipeline/type21_kd_plateau_followups_20260802"
NESTED_K = ROOT / "outputs/pipeline/type21_nested_k8d10_to_k16d10_20260802"
SUMMARY = ROOT / "outputs/pipeline/type21_kd_plateau_paper_summary_20260802.json"
TABLE = ROOT / "gcicy paper/generated_tn/x21_kd_plateau_nested_20260802.tex"


CONFIGS = (
    ("direct_k8_d8", "common origin", 8, 8, 96_800, GRID / "k8_d8"),
    ("direct_k12_d8", "common origin", 12, 8, 158_752, GRID / "k12_d8"),
    ("direct_k16_d8", "common origin", 16, 8, 220_704, GRID / "k16_d8"),
    ("direct_k8_d10", "common origin", 8, 10, 150_040, GRID / "k8_d10"),
    ("direct_k12_d10", "common origin", 12, 10, 246_840, FOLLOWUPS / "k12_d10"),
    ("direct_k16_d10", "common origin", 16, 10, 343_640, FOLLOWUPS / "k16_d10"),
    ("direct_k8_d12", "common origin", 8, 12, 214_896, GRID / "k8_d12"),
    ("direct_k16_d12", "common origin", 16, 12, 493_680, GRID / "k16_d12"),
    ("nested_d_k8", "from $(8,10)$", 8, 12, 214_896, FOLLOWUPS / "k8_d10_to_d12"),
    ("nested_d_k16", "from $(16,10)$", 16, 12, 493_680, FOLLOWUPS / "k16_d10_to_d12"),
    ("nested_k_d10", "from $(8,10)$", 16, 10, 343_640, NESTED_K),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_improvement(source: float, target: float) -> float:
    return (source - target) / source


def compact_metrics(report: dict[str, Any]) -> dict[str, Any]:
    models = report["models"]
    if len(models) != 1:
        raise ValueError("each bilateral report must contain exactly one model")
    metrics = next(iter(models.values()))["metrics"]
    keys = (
        "point_count",
        "sigma",
        "squared_energy",
        "chi",
        "absolute_log_ratio_q999",
        "absolute_log_ratio_cvar_1pct",
        "normalized_ratio_min",
        "normalized_ratio_max",
        "minimum_metric_eigenvalue",
        "nonpositive_metric_count",
    )
    return {key: metrics[key] for key in keys}


def comparison(rows: dict[str, dict[str, Any]], source: str, target: str) -> dict[str, Any]:
    source_metrics = rows[source]["metrics"]
    target_metrics = rows[target]["metrics"]
    keys = ("sigma", "chi", "absolute_log_ratio_q999", "absolute_log_ratio_cvar_1pct")
    return {
        "source": source,
        "target": target,
        "relative_improvements": {
            key: relative_improvement(source_metrics[key], target_metrics[key])
            for key in keys
        },
    }


def main() -> None:
    rows: dict[str, dict[str, Any]] = {}
    for key, initialization, degree, bond, parameters, directory in CONFIGS:
        report_path = directory / "development_confirmation_bilateral.json"
        equivalence_path = directory / "initial_equivalence.json"
        plateau_path = directory / "plateau_summary.json"
        for path in (report_path, equivalence_path, plateau_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        metrics = compact_metrics(load(report_path))
        equivalence = load(equivalence_path)
        plateau = load(plateau_path)
        if not equivalence["success"]:
            raise RuntimeError(f"epoch-zero equivalence failed for {key}")
        if metrics["nonpositive_metric_count"] != 0:
            raise RuntimeError(f"sampled positivity failed for {key}")
        if int(plateau["trainable_real_parameter_count"]) != parameters:
            raise RuntimeError(f"parameter-count mismatch for {key}")
        if plateau["termination_reason"] != "validation_plateau":
            raise RuntimeError(f"plateau stopping rule failed for {key}")
        rows[key] = {
            "initialization": initialization,
            "k": degree,
            "D": bond,
            "trainable_real_parameters": parameters,
            "metrics": metrics,
            "termination_reason": plateau["termination_reason"],
            "report": str(report_path.relative_to(ROOT)),
            "report_sha256": sha256(report_path),
            "equivalence_report": str(equivalence_path.relative_to(ROOT)),
            "equivalence_report_sha256": sha256(equivalence_path),
            "plateau_summary": str(plateau_path.relative_to(ROOT)),
            "plateau_summary_sha256": sha256(plateau_path),
        }

    comparisons = {
        "direct_D8_k8_to_k12": comparison(rows, "direct_k8_d8", "direct_k12_d8"),
        "direct_D8_k8_to_k16": comparison(rows, "direct_k8_d8", "direct_k16_d8"),
        "direct_D10_k8_to_k12": comparison(rows, "direct_k8_d10", "direct_k12_d10"),
        "direct_D10_k8_to_k16": comparison(rows, "direct_k8_d10", "direct_k16_d10"),
        "direct_k8_D8_to_D10": comparison(rows, "direct_k8_d8", "direct_k8_d10"),
        "direct_k8_D10_to_D12": comparison(rows, "direct_k8_d10", "direct_k8_d12"),
        "direct_k16_D8_to_D10": comparison(rows, "direct_k16_d8", "direct_k16_d10"),
        "direct_k16_D10_to_D12": comparison(rows, "direct_k16_d10", "direct_k16_d12"),
        "nested_D_k8": comparison(rows, "direct_k8_d10", "nested_d_k8"),
        "nested_D_k16": comparison(rows, "direct_k16_d10", "nested_d_k16"),
        "nested_D_k16_vs_direct_D12": comparison(rows, "direct_k16_d12", "nested_d_k16"),
        "nested_k_D10": comparison(rows, "direct_k8_d10", "nested_k_d10"),
        "nested_k_D10_vs_direct_k16": comparison(rows, "direct_k16_d10", "nested_k_d10"),
    }
    best_key = min(rows, key=lambda key: rows[key]["metrics"]["sigma"])
    protocols = (
        ROOT / "pipeline_specs/type21_kd_plateau_grid_20260731.json",
        ROOT / "pipeline_specs/type21_kd_plateau_followups_20260802.json",
        ROOT / "pipeline_specs/type21_nested_k8d10_to_k16d10_20260802.json",
    )
    summary = {
        "schema": "type21-kd-plateau-paper-summary-v1",
        "date": "2026-08-02",
        "evidence_role": "single-seed development landscape; not fresh final blind evidence",
        "rows": rows,
        "comparisons": comparisons,
        "best_sigma_row": best_key,
        "all_epoch_zero_equivalence_gates_pass": True,
        "all_stopping_rules_are_validation_plateaus": True,
        "all_sampled_metrics_positive": True,
        "protocols": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for path in protocols
        ],
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    table_rows = []
    for key, initialization, degree, bond, parameters, _ in CONFIGS:
        metrics = rows[key]["metrics"]
        table_rows.append(
            "{} & {} & {} & {:,} & {:.6f} & {:.6f} & {:.5f} & {:.5f} \\\\".format(
                initialization,
                degree,
                bond,
                parameters,
                metrics["sigma"],
                metrics["chi"],
                metrics["absolute_log_ratio_q999"],
                metrics["absolute_log_ratio_cvar_1pct"],
            )
        )
    table = "\n".join(
        [
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"initialization & $k$ & $D$ & $P$ & $\sigma$ & $\chi$ & "
            r"$Q_{0.999}(|\log r|)$ & $\CVaR_{1\%}(|\log r|)$ \\",
            r"\midrule",
            *table_rows[:8],
            r"\midrule",
            *table_rows[8:],
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(table, encoding="utf-8")
    print(SUMMARY)
    print(TABLE)
    print(f"best sigma row: {best_key} = {rows[best_key]['metrics']['sigma']:.9f}")


if __name__ == "__main__":
    main()
