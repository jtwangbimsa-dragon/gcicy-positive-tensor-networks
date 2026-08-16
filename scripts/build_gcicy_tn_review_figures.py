#!/usr/bin/env python3
"""Build the two headline figures used by the revised gCICY TN manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TN_COLOR = "#167D8D"
PHI_COLOR = "#B34A4A"
D8_COLOR = "#247BA0"
D12_COLOR = "#D17A22"
FULL_H_COLOR = "#343A40"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_log_ratio(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as arrays:
        log_eta = np.asarray(arrays["model_log_eta"], dtype=np.float64)
        weights = np.asarray(arrays["importance_weights"], dtype=np.float64)
    weights = weights / weights.sum()
    shift = float(np.max(log_eta))
    log_mean = shift + np.log(np.sum(weights * np.exp(log_eta - shift)))
    return log_eta - log_mean, weights


def weighted_survival(
    values: np.ndarray, weights: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    suffix = np.cumsum(sorted_weights[::-1])[::-1]
    indices = np.searchsorted(sorted_values, grid, side="left")
    result = np.zeros_like(grid)
    valid = indices < len(sorted_values)
    result[valid] = suffix[indices[valid]]
    return result


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_x11_tail_figure(run_dir: Path, out_dir: Path) -> None:
    records: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "TN": [],
        r"residual $\phi$": [],
    }
    for replicate in (1, 2, 3):
        replicate_dir = run_dir / f"replicate_{replicate}"
        tn_path = replicate_dir / "tn_arrays.npz"
        phi_path = replicate_dir / "source_density_phi" / "blind_tail_arrays.npz"
        if not tn_path.is_file():
            tn_path = replicate_dir / "tn_final_blind_arrays.npz"
        if not phi_path.is_file():
            phi_path = replicate_dir / "phi_final_blind" / "blind_tail_arrays.npz"
        records["TN"].append(
            normalized_log_ratio(tn_path)
        )
        records[r"residual $\phi$"].append(
            normalized_log_ratio(phi_path)
        )

    largest = max(
        float(np.max(np.abs(log_ratio)))
        for model_records in records.values()
        for log_ratio, _ in model_records
    )
    grid = np.linspace(0.0, max(0.9, largest * 1.03), 360)
    colors = {"TN": TN_COLOR, r"residual $\phi$": PHI_COLOR}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for axis, side, title in zip(
        axes,
        ("upper", "lower"),
        (r"Upper tail: $\log r>t$", r"Lower tail: $-\log r>t$"),
    ):
        for label, model_records in records.items():
            curves = []
            for log_ratio, weights in model_records:
                values = np.maximum(log_ratio, 0.0)
                if side == "lower":
                    values = np.maximum(-log_ratio, 0.0)
                curve = weighted_survival(values, weights, grid)
                curves.append(curve)
                axis.plot(
                    grid,
                    np.maximum(curve, 1e-7),
                    color=colors[label],
                    alpha=0.25,
                    linewidth=1.0,
                )
            stacked = np.stack(curves)
            mean = stacked.mean(axis=0)
            axis.plot(
                grid,
                np.maximum(mean, 1e-7),
                color=colors[label],
                linewidth=2.2,
                label=label,
            )
            axis.fill_between(
                grid,
                np.maximum(stacked.min(axis=0), 1e-7),
                np.maximum(stacked.max(axis=0), 1e-7),
                color=colors[label],
                alpha=0.10,
                linewidth=0,
            )
        axis.axhline(0.01, color="#777777", linestyle="--", linewidth=0.9)
        axis.set_yscale("log")
        axis.set_ylim(5e-6, 1.05)
        axis.set_xlim(0.0, grid[-1])
        axis.set_xlabel(r"threshold $t$")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel(r"weighted survival probability")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, out_dir, "gcicy_tn_x11_plateau_tail_survival")


def metric_from_bilateral(path: Path) -> dict[str, float]:
    report = load_json(path)
    metrics = next(iter(report["models"].values()))["metrics"]
    return {
        "sigma": float(metrics["sigma"]),
        "chi": float(metrics["chi"]),
        "cvar": float(metrics["absolute_log_ratio_cvar_1pct"]),
    }


def build_x21_capacity_figure(
    d8_paths: list[Path],
    d12_paths: list[Path],
    full_h_path: Path,
    final_blind_summary_path: Path | None,
    out_dir: Path,
) -> None:
    d8_rows = [metric_from_bilateral(path) for path in d8_paths]
    d8 = {
        key: np.array([row[key] for row in d8_rows])
        for key in ("sigma", "chi", "cvar")
    }
    d12_rows = [metric_from_bilateral(path) for path in d12_paths]
    d12 = {
        key: np.array([row[key] for row in d12_rows])
        for key in ("sigma", "chi", "cvar")
    }
    full_h = metric_from_bilateral(full_h_path)
    final_blind = (
        None
        if final_blind_summary_path is None
        else load_json(final_blind_summary_path)["metrics"]
    )

    panels = [
        ("sigma", r"$\sigma$"),
        ("chi", r"$\chi$"),
        ("cvar", r"$\mathrm{CVaR}_{1\%}(|\log r|)$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.55))
    for axis, (key, ylabel) in zip(axes, panels):
        axis.scatter(
            [104.976],
            [full_h[key]],
            color=FULL_H_COLOR,
            marker="D",
            s=72,
            label=r"full-$H_4$",
            zorder=4,
        )
        for x, values, color, marker, label in (
            (96.8, d8[key], D8_COLOR, "o", r"TN $D=8$"),
            (214.896, d12[key], D12_COLOR, "s", r"TN $D=12$"),
        ):
            offsets = np.linspace(-2.0, 2.0, len(values))
            axis.scatter(
                x + offsets,
                values,
                color=color,
                marker=marker,
                s=54,
                alpha=0.82,
                zorder=3,
            )
            axis.errorbar(
                [x],
                [values.mean()],
                yerr=[values.std(ddof=1)],
                color=color,
                marker=marker,
                markersize=8.5,
                capsize=4,
                linewidth=1.6,
                label=label,
                zorder=5,
            )
        axis.plot(
            [96.8, 214.896],
            [d8[key].mean(), d12[key].mean()],
            color="#666666",
            linewidth=1.0,
            linestyle=":",
            zorder=1,
        )
        if final_blind is not None:
            final_metric_key = {
                "sigma": "sigma",
                "chi": "chi",
                "cvar": "absolute_log_ratio_cvar_1pct",
            }[key]
            final_values = (
                (
                    104.976,
                    final_blind["full_h4"][final_metric_key],
                    FULL_H_COLOR,
                    "D",
                ),
                (
                    96.8,
                    final_blind["d8_source"][final_metric_key],
                    D8_COLOR,
                    "o",
                ),
                (
                    214.896,
                    final_blind["d12_selected"][final_metric_key],
                    D12_COLOR,
                    "s",
                ),
            )
            for final_index, (x, value, color, marker) in enumerate(final_values):
                axis.scatter(
                    [x],
                    [value],
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.25,
                    marker=marker,
                    s=105,
                    label=(
                        "frozen final blind"
                        if key == "sigma" and final_index == 0
                        else None
                    ),
                    zorder=7,
                )
        axis.set_xlabel(r"trainable real parameters ($10^3$)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.20, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir, "gcicy_tn_x21_capacity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x11-run-dir",
        type=Path,
        help="Optional X11 run directory; omit to rebuild only the X21 figure.",
    )
    parser.add_argument("--x21-d8-report", type=Path, action="append")
    parser.add_argument("--x21-d12-report", type=Path, action="append")
    parser.add_argument("--x21-full-h-report", type=Path)
    parser.add_argument("--x21-final-blind-summary", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.x11_run_dir is not None:
        build_x11_tail_figure(args.x11_run_dir, args.out_dir)
    x21_requested = any(
        value is not None
        for value in (
            args.x21_d8_report,
            args.x21_d12_report,
            args.x21_full_h_report,
            args.x21_final_blind_summary,
        )
    )
    if x21_requested:
        if (
            args.x21_d8_report is None
            or args.x21_d12_report is None
            or args.x21_full_h_report is None
            or len(args.x21_d8_report) != 3
            or len(args.x21_d12_report) != 3
        ):
            raise ValueError(
                "The X21 figure requires exactly three D=8 reports, three D=12 "
                "reports, and one full-H report."
            )
        build_x21_capacity_figure(
            args.x21_d8_report,
            args.x21_d12_report,
            args.x21_full_h_report,
            args.x21_final_blind_summary,
            args.out_dir,
        )


if __name__ == "__main__":
    main()
