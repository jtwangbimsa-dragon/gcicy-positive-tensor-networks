#!/usr/bin/env python3
"""Build review-revision tables and figures from frozen gCICY TN reports."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = (
    ROOT
    / "outputs/pipeline/type11_h2_matched_tn_phi_three_seed_plateau_20260731"
    / "training_reports"
)
X11_PROTOCOL = (
    ROOT
    / "pipeline_specs/type11_h2_matched_tn_phi_three_seed_plateau_final_blind_20260731.json"
)
X21_SUMMARY = ROOT / "outputs/pipeline/type21_kd_plateau_paper_summary_20260802.json"
X21_COMPRESSION = (
    ROOT
    / "outputs/pipeline/type21_q121_d12_final_blind_20260729/"
    "h4_vs_d8_final_blind_bootstrap_2000.json"
)
GENERATED = ROOT / "gcicy paper/generated_tn"
FIGURES = ROOT / "gcicy paper/figures"
LEDGER = TRAINING_DIR.parent / "training_ledger.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actual_epochs(report: dict[str, Any], architecture: str) -> int:
    if architecture == "tn":
        # TN history includes the epoch-zero validation entry.
        return len(report["history"]) - 1
    return len(report["training"]["history"])


def stage_record(
    replicate: int,
    architecture: str,
    stage: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    path = TRAINING_DIR / f"replicate_{replicate}_{architecture}_{stage}.json"
    report = load(path)
    epochs = actual_epochs(report, architecture)
    batch_size = int(protocol["plateau_schedule"]["batch_size"])
    train_points = int(protocol["frozen_data"]["train"]["points"])
    updates_per_epoch = train_points // batch_size
    if architecture == "tn":
        runtime = float(report["runtime_seconds"])
        peak = int(report["device_memory"]["maximum_allocated_bytes"])
        termination = report["termination_reason"]
        best_epoch = int(report["best_epoch"])
    else:
        runtime = float(report["timing_seconds"]["optimization"])
        peak = int(report["training_device_memory"]["maximum_allocated_bytes"])
        termination = report["training"]["termination_reason"]
        best_epoch = int(report["training"]["best_epoch"])
    if termination != "validation_plateau":
        raise RuntimeError(f"unexpected termination in {path}: {termination}")
    return {
        "replicate": replicate,
        "architecture": architecture,
        "stage": stage,
        "epochs": epochs,
        "optimizer_updates": epochs * updates_per_epoch,
        "optimization_wall_seconds": runtime,
        "peak_allocated_bytes": peak,
        "best_epoch": best_epoch,
        "termination_reason": termination,
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def build_x11_cost_assets() -> list[Path]:
    protocol = load(X11_PROTOCOL)
    stages = [
        stage_record(replicate, architecture, stage, protocol)
        for replicate in (1, 2, 3)
        for architecture in ("tn", "phi")
        for stage in ("primary", "precision")
    ]
    totals: list[dict[str, Any]] = []
    for replicate in (1, 2, 3):
        for architecture in ("tn", "phi"):
            selected = [
                row
                for row in stages
                if row["replicate"] == replicate
                and row["architecture"] == architecture
            ]
            totals.append(
                {
                    "replicate": replicate,
                    "architecture": architecture,
                    "epochs": sum(row["epochs"] for row in selected),
                    "optimizer_updates": sum(
                        row["optimizer_updates"] for row in selected
                    ),
                    "optimization_wall_seconds": sum(
                        row["optimization_wall_seconds"] for row in selected
                    ),
                    "peak_allocated_bytes": max(
                        row["peak_allocated_bytes"] for row in selected
                    ),
                    "termination_reason": "validation_plateau",
                }
            )

    aggregates: dict[str, Any] = {}
    for architecture in ("tn", "phi"):
        selected = [row for row in totals if row["architecture"] == architecture]
        aggregates[architecture] = {}
        for key in (
            "epochs",
            "optimizer_updates",
            "optimization_wall_seconds",
            "peak_allocated_bytes",
        ):
            mean, sd = mean_sd([float(row[key]) for row in selected])
            aggregates[architecture][key] = {
                "mean": mean,
                "sample_standard_deviation": sd,
            }

    ledger = {
        "schema": "gcicy-x11-plateau-training-ledger-v1",
        "scope": (
            "post-warm-start primary and precision continuation only; "
            "the common 20-epoch warm-start is excluded"
        ),
        "protocol": str(X11_PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(X11_PROTOCOL),
        "stages": stages,
        "replicate_totals": totals,
        "aggregate": aggregates,
    }
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    summary_rows = []
    labels = {"tn": r"positive TN", "phi": r"residual $\phi$"}
    for architecture in ("tn", "phi"):
        values = aggregates[architecture]
        epoch = values["epochs"]
        updates = values["optimizer_updates"]
        seconds = values["optimization_wall_seconds"]
        peak = values["peak_allocated_bytes"]
        summary_rows.append(
            "{} & ${:.1f}\\pm{:.1f}$ & ${:,.0f}\\pm{:,.0f}$ & "
            "${:.1f}\\pm{:.1f}$ & ${:.3f}\\pm{:.3f}$ & validation plateau \\\\".format(
                labels[architecture],
                epoch["mean"],
                epoch["sample_standard_deviation"],
                updates["mean"],
                updates["sample_standard_deviation"],
                seconds["mean"] / 60.0,
                seconds["sample_standard_deviation"] / 60.0,
                peak["mean"] / (1024**3),
                peak["sample_standard_deviation"] / (1024**3),
            )
        )
    summary = "\n".join(
        [
            r"\begin{tabular}{lrrrrl}",
            r"\toprule",
            r"model & epochs & updates & optimization min & peak GiB & stop \\",
            r"\midrule",
            *summary_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    summary_path = GENERATED / "x11_plateau_cost_summary_20260802.tex"
    summary_path.write_text(summary, encoding="utf-8")

    seed_rows = []
    for row in totals:
        primary = next(
            item
            for item in stages
            if item["replicate"] == row["replicate"]
            and item["architecture"] == row["architecture"]
            and item["stage"] == "primary"
        )
        precision = next(
            item
            for item in stages
            if item["replicate"] == row["replicate"]
            and item["architecture"] == row["architecture"]
            and item["stage"] == "precision"
        )
        seed_rows.append(
            "{} & {} & {} & {} & {:,} & {:.2f} & {:.3f} & validation plateau \\\\".format(
                row["replicate"],
                labels[row["architecture"]],
                primary["epochs"],
                precision["epochs"],
                row["optimizer_updates"],
                row["optimization_wall_seconds"] / 60.0,
                row["peak_allocated_bytes"] / (1024**3),
            )
        )
    seedwise = "\n".join(
        [
            r"\begin{tabular}{rlrrrrrl}",
            r"\toprule",
            r"seed & model & primary ep. & precision ep. & updates & min & peak GiB & stop \\",
            r"\midrule",
            *seed_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    seedwise_path = GENERATED / "x11_plateau_cost_seedwise_20260802.tex"
    seedwise_path.write_text(seedwise, encoding="utf-8")
    return [LEDGER, summary_path, seedwise_path]


def build_x21_path_figure() -> list[Path]:
    summary = load(X21_SUMMARY)
    rows = summary["rows"]
    colors = {8: "#007f73", 12: "#d97706", 16: "#335c9b"}
    markers = {8: "o", 10: "s", 12: "^"}

    fig, ax = plt.subplots(figsize=(7.25, 4.55), constrained_layout=True)
    for degree in (8, 12, 16):
        direct = sorted(
            (
                row
                for key, row in rows.items()
                if key.startswith("direct_") and row["k"] == degree
            ),
            key=lambda row: row["D"],
        )
        if direct:
            ax.plot(
                [row["trainable_real_parameters"] for row in direct],
                [1000 * row["metrics"]["sigma"] for row in direct],
                color=colors[degree],
                linewidth=1.2,
                alpha=0.65,
            )
        for row in direct:
            ax.scatter(
                row["trainable_real_parameters"],
                1000 * row["metrics"]["sigma"],
                color=colors[degree],
                marker=markers[row["D"]],
                s=58,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

    arrow_specs = (
        ("direct_k8_d10", "nested_d_k8", "nested $D$"),
        ("direct_k16_d10", "nested_d_k16", "nested $D$"),
        ("direct_k8_d10", "nested_k_d10", "nested $k$"),
    )
    for source_key, target_key, label in arrow_specs:
        source = rows[source_key]
        target = rows[target_key]
        color = colors[target["k"]]
        ax.annotate(
            "",
            xy=(target["trainable_real_parameters"], 1000 * target["metrics"]["sigma"]),
            xytext=(source["trainable_real_parameters"], 1000 * source["metrics"]["sigma"]),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5, "ls": "--"},
            zorder=2,
        )
        ax.scatter(
            target["trainable_real_parameters"],
            1000 * target["metrics"]["sigma"],
            facecolor="none",
            edgecolor=color,
            marker=markers[target["D"]],
            s=105,
            linewidth=1.8,
            zorder=4,
        )
        if target_key != "nested_d_k8":
            offset = (-72, 9) if target_key == "nested_d_k16" else (5, 8)
            ax.annotate(
                label,
                (target["trainable_real_parameters"], 1000 * target["metrics"]["sigma"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    best_direct = rows["direct_k16_d10"]
    ax.annotate(
        "best direct",
        (best_direct["trainable_real_parameters"], 1000 * best_direct["metrics"]["sigma"]),
        xytext=(-61, 11),
        textcoords="offset points",
        fontsize=8,
        color=colors[16],
    )
    ax.set_xlabel("trainable real parameters")
    ax.set_ylabel(r"$10^3\,\sigma_{\rm dev}$")
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    degree_handles = [
        Line2D([0], [0], color=colors[k], marker="o", lw=1.2, label=fr"$k={k}$")
        for k in (8, 12, 16)
    ]
    bond_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=markers[d],
            linestyle="None",
            label=fr"$D={d}$",
        )
        for d in (8, 10, 12)
    ]
    ax.legend(
        handles=degree_handles + bond_handles,
        ncol=3,
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES / "gcicy_tn_x21_path_dependence_20260802.pdf"
    png_path = FIGURES / "gcicy_tn_x21_path_dependence_20260802.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return [pdf_path, png_path]


def build_x21_compression_table() -> list[Path]:
    payload = load(X21_COMPRESSION)
    development_summary = load(X21_SUMMARY)
    comparisons = payload["comparisons"]
    h4_parameters = 324**2
    d8_parameters = int(
        development_summary["rows"]["direct_k8_d8"]["trainable_real_parameters"]
    )

    def metric(name: str, role: str) -> float:
        return float(comparisons[name][role])

    rows = [
        (
            r"full $H_4$",
            h4_parameters,
            metric("sigma", "baseline"),
            metric("chi", "baseline"),
            metric("absolute_log_ratio_q999", "baseline"),
            metric("absolute_log_ratio_cvar_1pct", "baseline"),
        ),
        (
            r"positive $k=8,D=8$ TN",
            d8_parameters,
            metric("sigma", "candidate"),
            metric("chi", "candidate"),
            metric("absolute_log_ratio_q999", "candidate"),
            metric("absolute_log_ratio_cvar_1pct", "candidate"),
        ),
    ]
    body = [
        "{} & {:,} & {:.6f} & {:.6f} & {:.3f} & {:.3f} \\\\".format(*row)
        for row in rows
    ]
    table = "\n".join(
        [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"model & $P$ & $\sigma$ & $\chi$ & $Q^{\rm abs}_{0.999}$ & $\CVaR^{\rm abs}_{1\%}$ \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    path = GENERATED / "x21_final_blind_compression_20260802.tex"
    path.write_text(table, encoding="utf-8")
    return [path]


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    outputs = (
        build_x11_cost_assets()
        + build_x21_path_figure()
        + build_x21_compression_table()
    )
    manifest = {
        "schema": "gcicy-tn-review-revision-assets-v1",
        "sources": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                X11_PROTOCOL,
                X21_SUMMARY,
                X21_COMPRESSION,
                *sorted(TRAINING_DIR.glob("*.json")),
            ]
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in outputs
        },
    }
    manifest_path = GENERATED / "review_revision_assets_20260802.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
