#!/usr/bin/env python3
"""Plot the preregistered X21 equal-update continuation control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gcicy-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gcicy-cache")

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("source", "control", "d12")
ARM_LABELS = (
    r"source $D=8$",
    r"equal-update $D=8$",
    r"expanded $D=12$",
)
COLORS = ("#777777", "#2878B5", "#C63C32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    seeds = [str(seed) for seed in payload["seeds"]]
    x = np.arange(len(ARMS))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("sigma", "chi"),
        (r"Mean absolute MA error $\sigma$", r"RMS MA error $\chi$"),
        strict=True,
    ):
        matrix = np.asarray(
            [
                [
                    payload["records"][seed][arm][metric]
                    for arm in ARMS
                ]
                for seed in seeds
            ],
            dtype=float,
        )
        for seed, row in zip(seeds, matrix, strict=True):
            axis.plot(
                x,
                row,
                color="#A6A6A6",
                linewidth=1.0,
                marker="o",
                markersize=4,
                label=f"seed {seed}" if axis is axes[0] else None,
                zorder=1,
            )
        means = np.mean(matrix, axis=0)
        axis.plot(
            x,
            means,
            color="#111111",
            linewidth=2.0,
            marker="D",
            markersize=5,
            label="three-seed mean" if axis is axes[0] else None,
            zorder=3,
        )
        for location, value, color in zip(x, means, COLORS, strict=True):
            axis.scatter(
                [location],
                [value],
                color=color,
                edgecolor="white",
                linewidth=0.6,
                s=38,
                zorder=4,
            )
        paired_gain = payload["paired_primary_improvements"][metric][
            "mean_relative_improvement"
        ]
        axis.text(
            0.98,
            0.97,
            rf"mean $D=12$ gain: {100.0 * paired_gain:.2f}\%"
            "\npreregistered gate: failed",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        axis.set_xticks(x, ARM_LABELS)
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)

    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        r"$X_{21}$: continuation dominates the original source, "
        r"while the equal-update bond effect is small and seed dependent",
        fontsize=10,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
