#!/usr/bin/env python3
"""Build figures, LaTeX tables, and an evidence manifest for the gCICY TN paper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "gcicy paper"
GENERATED = PAPER / "generated_tn"
FIGURES = PAPER / "figures"

INPUTS = {
    "publication_summary": ROOT / "outputs/gcicy_tn_publication_summary_20260719.json",
    "full_h_context": ROOT / "outputs/gcicy_paper_metric_tail_summary.json",
    "x11_k2": ROOT
    / "outputs/pipeline/type11_tensor_network_scaling_k2_20260717/"
    "model_d4_k2_energy_blind.json",
    "x11_k4": ROOT
    / "outputs/pipeline/type11_tensor_network_k4_energy_continuation2_20260717/"
    "model_d4_energy2_blind.json",
    "x11_k6": ROOT
    / "outputs/pipeline/type11_tensor_network_scaling_k6_d5_production_20260717/"
    "model_d5_k6_production_blind.json",
    "x11_ritz_k2": ROOT
    / "outputs/pipeline/type11_tensor_network_scaling_k2_20260717/"
    "model_d4_k2_energy_scalar_ritz_4x32768_l23.json",
    "x11_ritz_k4": ROOT
    / "outputs/pipeline/type11_tensor_network_k4_energy_continuation2_20260717/"
    "model_d4_energy2_scalar_ritz_fresh_4x32768_l23.json",
    "x11_ritz_k6": ROOT
    / "outputs/pipeline/type11_tensor_network_scaling_k6_d5_production_20260717/"
    "model_d5_k6_production_scalar_ritz_4x32768_l23.json",
    "x21_k4": ROOT
    / "outputs/pipeline/type21_tensor_network_q22_degree_ladder_20260719/"
    "q22_k4_teacher_free_blind83503_n65532.json",
    "x21_k6": ROOT
    / "outputs/pipeline/type21_tensor_network_q22_degree_ladder_20260719/"
    "q22_k6_teacher_free_blind83513_n65532.json",
    "x21_k8": ROOT
    / "outputs/pipeline/type21_tensor_network_q22_degree_ladder_20260719/"
    "q22_k8_teacher_free_blind83523_n65532.json",
    "x22_q22_k4": ROOT
    / "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/"
    "active_q22_k4_control_blind83703_n65532.json",
    "x22_q60_k4": ROOT
    / "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/"
    "active_q60_k4_capacity_blind83703_n65532.json",
    "x22_q60_k6": ROOT
    / "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/"
    "active_q60_k6_degree_control_blind83713_n65532.json",
    "kernel_scaling": ROOT
    / "outputs/pipeline/type11_tensor_network_fixed_d5_kernel_scaling_m1_m8_20260718.json",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def metric_row(report: dict[str, Any]) -> dict[str, float]:
    compressed = report["metrics"]["compressed_ma_errors"]
    return {
        "sigma": float(compressed["sigma"]),
        "chi": float(compressed["sqrt_squared_energy"]),
        "q999": float(compressed["positive_log_ratio_q999"]),
        "cvar": float(compressed["positive_log_ratio_cvar_1pct"]),
        "rmin": float(compressed["normalized_ratio_min"]),
        "rmax": float(compressed["normalized_ratio_max"]),
        "mineig": float(report["metrics"]["minimum_metric_eigenvalue"]),
    }


def teacher_row(report: dict[str, Any]) -> dict[str, float]:
    teacher = report["metrics"]["teacher_ma_errors"]
    if teacher is None:
        raise ValueError("The requested report has no same-point teacher metrics.")
    return {
        "sigma": float(teacher["sigma"]),
        "chi": float(teacher["sqrt_squared_energy"]),
        "q999": float(teacher["positive_log_ratio_q999"]),
        "cvar": float(teacher["positive_log_ratio_cvar_1pct"]),
        "rmin": float(teacher["normalized_ratio_min"]),
        "rmax": float(teacher["normalized_ratio_max"]),
    }


def write_text(name: str, content: str) -> Path:
    path = GENERATED / name
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def tabular_snippet(
    column_spec: str,
    header: str,
    rows: list[str],
    *,
    resize: bool = False,
) -> str:
    body = "\n".join(
        [
            f"\\begin{{tabular}}{{{column_spec}}}",
            "\\toprule",
            header + " \\tabularnewline",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    if resize:
        return "\\resizebox{\\textwidth}{!}{%\n" + body + "\n}"
    return body


def build_tables(data: dict[str, dict[str, Any]]) -> list[Path]:
    outputs: list[Path] = []
    summary = data["publication_summary"]

    headline_rows = []
    for case in summary["cases"]:
        metrics = case["blind_metrics"]
        degree = r"\(\times\)".join(str(x) for x in case["target_multidegree"])
        dictionary = case["physical_dictionary"]
        headline_rows.append(
            "{} & {} & {}/{} & {:,} & {:,} & {:.5f} & {:.5f} & {:.5f} & "
            "{:.5f} & [{:.3f},{:.3f}] & {:.4f} \\tabularnewline".format(
                case["type"],
                degree,
                dictionary["rank"],
                dictionary["complete_rank"],
                case["trainable_real_parameter_count"],
                case["sample_counts"]["blind"],
                metrics["sigma"],
                metrics["chi"],
                metrics["absolute_log_ratio_q999"],
                metrics["absolute_log_ratio_cvar_1pct"],
                metrics["normalized_ratio_min"],
                metrics["normalized_ratio_max"],
                metrics["minimum_metric_eigenvalue"],
            )
        )
    outputs.append(
        write_text(
            "headline_metrics.tex",
            tabular_snippet(
                "lcccccccccc",
                "type & target & $q/d^2$ & $P$ & blind $N$ & "
                "$\\sigma$ & $\\chi$ & $q_{99.9}$ & $\\CVaR_{1\\%}$ & "
                "$[r_{\\min},r_{\\max}]$ & $\\lambda_{\\min}(g)$",
                headline_rows,
                resize=True,
            ),
        )
    )

    x11_rows = []
    for degree, key in [(2, "x11_k2"), (4, "x11_k4"), (6, "x11_k6")]:
        report = data[key]
        tn = metric_row(report)
        teacher = teacher_row(report)
        params = int(report["trainable_real_parameter_count"])
        section_count = {2: 45, 4: 274, 6: 871}[degree]
        x11_rows.append(
            "{} & {:,} & {:,} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & "
            "{:.5f} & {:.5f} \\tabularnewline".format(
                degree,
                params,
                section_count * section_count,
                tn["sigma"],
                teacher["sigma"],
                tn["chi"],
                teacher["chi"],
                tn["q999"],
                teacher["q999"],
            )
        )
    outputs.append(
        write_text(
            "x11_paired_bulk.tex",
            tabular_snippet(
                "rrrrrrrrr",
                "$k$ & $P_{\\TN}$ & $P_{\\mathrm{full}\\text{-}H}$ & "
                "$\\sigma_{\\TN}$ & $\\sigma_H$ & $\\chi_{\\TN}$ & "
                "$\\chi_H$ & $q_{99.9,\\TN}$ & $q_{99.9,H}$",
                x11_rows,
                resize=True,
            ),
        )
    )

    x11_tail_rows = []
    for degree, key in [(2, "x11_k2"), (4, "x11_k4"), (6, "x11_k6")]:
        report = data[key]
        tn = metric_row(report)
        teacher = teacher_row(report)
        x11_tail_rows.append(
            "{} & {:.5f} & {:.5f} & {:.4f} & {:.4f} & {:.4f} \\tabularnewline".format(
                degree,
                tn["cvar"],
                teacher["cvar"],
                tn["rmax"],
                teacher["rmax"],
                tn["mineig"],
            )
        )
    outputs.append(
        write_text(
            "x11_paired_tail.tex",
            tabular_snippet(
                "rrrrrr",
                "$k$ & $\\CVaR_{\\TN}$ & $\\CVaR_H$ & "
                "$r_{\\max,\\TN}$ & $r_{\\max,H}$ & "
                "$\\lambda_{\\min}(g_{\\TN})$",
                x11_tail_rows,
            ),
        )
    )

    scaling_rows = []
    scaling_specs = [
        ("(1,1)", 2, "complete", "x11_k2"),
        ("(1,1)", 4, "complete", "x11_k4"),
        ("(1,1)", 6, "complete", "x11_k6"),
        ("(2,1)", 4, "q=22", "x21_k4"),
        ("(2,1)", 6, "q=22", "x21_k6"),
        ("(2,1)", 8, "q=22", "x21_k8"),
        ("(2,2)", 4, "q=22", "x22_q22_k4"),
        ("(2,2)", 4, "q=60", "x22_q60_k4"),
        ("(2,2)", 6, "q=60", "x22_q60_k6"),
    ]
    for geometry, degree, dictionary, key in scaling_specs:
        report = data[key]
        metrics = metric_row(report)
        scaling_rows.append(
            "{} & {} & {} & {:,} & {:.5f} & {:.5f} & {:.5f} & {:.5f} "
            "\\tabularnewline".format(
                geometry,
                degree,
                dictionary,
                int(report["trainable_real_parameter_count"]),
                metrics["sigma"],
                metrics["chi"],
                metrics["q999"],
                metrics["cvar"],
            )
        )
    outputs.append(
        write_text(
            "degree_dictionary_scaling.tex",
            tabular_snippet(
                "llcrrrrr",
                "type & degree & dictionary & $P$ & $\\sigma$ & $\\chi$ & "
                "$q_{99.9}$ & $\\CVaR_{1\\%}$",
                scaling_rows,
            ),
        )
    )

    dense_specs = [
        ("(1,1)", 6, 871, "x11_k6"),
        ("(2,1)", 8, 2440, "x21_k8"),
        ("(2,2)", 6, 1852, "x22_q60_k6"),
    ]
    parameter_rows = []
    for geometry, degree, section_count, key in dense_specs:
        report = data[key]
        p_tn = int(report["trainable_real_parameter_count"])
        p_dense = section_count * section_count
        parameter_rows.append(
            "{} & {} & {:,} & {:,} & {:,} & {:.1f} \\tabularnewline".format(
                geometry,
                degree,
                section_count,
                p_tn,
                p_dense,
                p_dense / p_tn,
            )
        )
    outputs.append(
        write_text(
            "parameter_comparison.tex",
            tabular_snippet(
                "lrrrrr",
                "type & target degree & $h^0(L^k)$ & $P_{\\TN}$ & "
                "$P_{\\mathrm{full}\\text{-}H}$ & full-$H$/TN",
                parameter_rows,
            ),
        )
    )

    protocol_rows = []
    for case in summary["cases"]:
        samples = case["sample_counts"]
        fibres = case["complete_fibre_sampling"]
        protocol_rows.append(
            "{} & {:,} & {:,} & {:,} & {:,} & {} & {:.0f} \\tabularnewline".format(
                case["type"],
                samples["train"],
                samples["validation"],
                samples["blind"],
                fibres["blind_fibres"],
                fibres["roots_per_fibre"],
                case["runtime_seconds"],
            )
        )
    outputs.append(
        write_text(
            "protocol_ledger.tex",
            tabular_snippet(
                "lrrrrrr",
                "type & train & validation & blind & blind fibres & "
                "roots/fibre & train s",
                protocol_rows,
            ),
        )
    )

    ritz_rows = []
    for degree, key in [(2, "x11_ritz_k2"), (4, "x11_ritz_k4"), (6, "x11_ritz_k6")]:
        report = data[key]
        tn = report["metric_summaries"]["tensor_network"]["trial_levels"]["3"]
        teacher = report["metric_summaries"]["full_h_teacher"]["trial_levels"]["3"]
        comparison = next(
            item
            for item in report["paired_ritz_comparisons"]
            if item["candidate"] == "tensor_network"
            and item["reference"] == "full_h_teacher"
            and int(item["trial_level"]) == 3
        )
        shifts = comparison["mean_relative_shifts"]
        ritz_rows.append(
            "{} & ({:.4f},{:.4f},{:.4f}) & ({:.4f},{:.4f},{:.4f}) & "
            "({:+.2f},{:+.2f},{:+.2f}) & {:.2f} \\tabularnewline".format(
                degree,
                *tn["mean_eigenvalues"],
                *teacher["mean_eigenvalues"],
                *(100.0 * value for value in shifts),
                100.0 * comparison["maximum_absolute_per_dataset_relative_shift"],
            )
        )
    outputs.append(
        write_text(
            "x11_ritz.tex",
            tabular_snippet(
                "rcccc",
                "$k$ & TN mean $(\\lambda_1,\\lambda_2,\\lambda_3)$ & "
                "teacher mean $(\\lambda_1,\\lambda_2,\\lambda_3)$ & "
                "mean shifts (\\%) & max. per-dataset shift (\\%)",
                ritz_rows,
                resize=True,
            ),
        )
    )

    full_h_rows = {
        (row["type"], tuple(row["degree"])): row
        for row in data["full_h_context"]["rows"]
    }
    context_specs = [
        ("(1,1)", (4, 4), "separate 8-seed full-$H$ audit"),
        ("(2,1)", (4, 4), "separate 8-seed full-$H$ audit"),
        ("(2,2)", (3, 3, 3), "separate 8-seed full-$H$ audit"),
    ]
    context_rows = []
    for geometry, degree, comparison in context_specs:
        row = full_h_rows[(geometry, degree)]
        context_rows.append(
            "{} & {} & {:,} & {:.5f} & {:.5f} & {:.3f} & {} \\tabularnewline".format(
                geometry,
                r"\(\times\)".join(str(x) for x in degree),
                row["section_count"],
                row["mean_sigma"],
                row["mean_chi"],
                row["maximum_normalized_ratio"],
                comparison,
            )
        )
    outputs.append(
        write_text(
            "full_h_context.tex",
            tabular_snippet(
                "llrrrrl",
                "type & degree & sections & mean $\\sigma$ & mean $\\chi$ & "
                "max $r$ & numerical comparison",
                context_rows,
                resize=True,
            ),
        )
    )

    kernel_rows = []
    for row in data["kernel_scaling"]["results"]:
        kernel_rows.append(
            "{} & {} & {:,} & {:.3f} & {:.2f} & {:.1f} \\tabularnewline".format(
                row["site_count"],
                row["polarization_degree_for_k0_2"],
                row["trainable_real_parameter_count"],
                1000.0 * row["median_forward_seconds"],
                row["forward_microseconds_per_point"],
                row["device_memory"]["incremental_peak_allocated_bytes"] / (1024.0**2),
            )
        )
    outputs.append(
        write_text(
            "kernel_scaling.tex",
            tabular_snippet(
                "rrrrrr",
                "sites $m$ & degree & parameters & median ms/batch & "
                "$\\mu$s/point & incremental peak MiB",
                kernel_rows,
            ),
        )
    )
    return outputs


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [FIGURES / f"{stem}.pdf", FIGURES / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def build_figures(data: dict[str, dict[str, Any]]) -> list[Path]:
    set_plot_style()
    outputs: list[Path] = []
    palette = {"X11": "#1b4965", "X21": "#2a9d8f", "X22": "#c14953"}

    series = {
        "X11": [(2, metric_row(data["x11_k2"])), (4, metric_row(data["x11_k4"])), (6, metric_row(data["x11_k6"]))],
        "X21": [(4, metric_row(data["x21_k4"])), (6, metric_row(data["x21_k6"])), (8, metric_row(data["x21_k8"]))],
        "X22": [(4, metric_row(data["x22_q60_k4"])), (6, metric_row(data["x22_q60_k6"]))],
    }
    metrics = [
        ("sigma", r"$\sigma$"),
        ("chi", r"$\chi$"),
        ("q999", r"$q_{99.9}((\log r)_+)$"),
        ("cvar", r"$\mathrm{CVaR}_{1\%}((\log r)_+)$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.1), sharex=False)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for name, values in series.items():
            x = [entry[0] for entry in values]
            y = [entry[1][metric] for entry in values]
            axis.plot(x, y, marker="o", linewidth=1.8, color=palette[name], label=name)
        axis.set_xlabel("target degree")
        axis.set_ylabel(label)
        axis.set_yscale("log")
        axis.set_xticks(sorted({entry[0] for values in series.values() for entry in values}))
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.suptitle("Independent-blind degree trends for the positive TN models", y=1.01)
    fig.tight_layout()
    outputs.extend(save_figure(fig, "gcicy_tn_degree_scaling"))

    labels = [r"$\sigma$", r"$\chi$", r"$q_{99.9}$", "CVaR", r"$r_{\max}$"]
    fig, axis = plt.subplots(figsize=(7.15, 3.65))
    x = np.arange(len(labels))
    width = 0.24
    for offset, (degree, key, color) in enumerate(
        [(2, "x11_k2", "#1b4965"), (4, "x11_k4", "#2a9d8f"), (6, "x11_k6", "#c14953")]
    ):
        tn = metric_row(data[key])
        teacher = teacher_row(data[key])
        improvements = [
            100.0 * (teacher[name] - tn[name]) / teacher[name]
            for name in ["sigma", "chi", "q999", "cvar", "rmax"]
        ]
        axis.bar(x + (offset - 1) * width, improvements, width, label=f"k={degree}", color=color)
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("same-point improvement over full-$H$ teacher (%)")
    axis.legend(frameon=False, ncol=3)
    axis.set_title(r"$X_{11}$ paired finite-budget comparison; positive is better")
    fig.tight_layout()
    outputs.extend(save_figure(fig, "gcicy_tn_x11_paired_improvement"))

    geometries = ["X11", "X21", "X22"]
    tn_parameters = np.array([141750, 12364, 47880], dtype=float)
    dense_parameters = np.array([871**2, 2440**2, 1852**2], dtype=float)
    fig, axis = plt.subplots(figsize=(7.15, 3.75))
    x = np.arange(len(geometries))
    width = 0.34
    axis.bar(x - width / 2, tn_parameters, width, label="positive TN", color="#2a9d8f")
    axis.bar(x + width / 2, dense_parameters, width, label="full-$H$ at same degree", color="#c14953")
    axis.set_yscale("log")
    axis.set_xticks(x, geometries)
    axis.set_ylabel("trainable real parameters")
    axis.set_title("Structured parameter counts at the reported target degrees")
    axis.legend(frameon=False)
    for index, ratio in enumerate(dense_parameters / tn_parameters):
        axis.text(index, dense_parameters[index] * 1.12, f"{ratio:.1f}x", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    outputs.extend(save_figure(fig, "gcicy_tn_parameter_comparison"))
    return outputs


def build_manifest(generated_files: list[Path]) -> Path:
    manifest = {
        "schema": "gcicy-tn-manuscript-assets-v1",
        "source_root": str(ROOT),
        "inputs": {
            key: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for key, path in INPUTS.items()
        },
        "generated": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path)}
            for path in generated_files
        },
        "comparison_scope": {
            "x11_teacher_rows": "same points and weights; finite-budget TN versus stored full-H teacher",
            "full_h_context_rows": "separate multi-seed protocols; context only, not paired rankings",
            "cymetric": "no controlled gCICY residual-potential baseline is claimed",
        },
    }
    path = GENERATED / "evidence_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing manuscript inputs:\n" + "\n".join(missing))
    data = {key: load_json(path) for key, path in INPUTS.items()}
    generated = build_tables(data)
    generated.extend(build_figures(data))
    manifest = build_manifest(generated)
    print(f"Generated {len(generated)} assets and {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
