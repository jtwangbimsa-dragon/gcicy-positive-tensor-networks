#!/usr/bin/env python3
"""Paired weighted k-convergence audit for exact bicubic H metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.bicubic_model import (  # noqa: E402
    bicubic_baseline_metrics,
    bicubic_global_h_metrics,
    bicubic_importance_weights,
    bicubic_residual_stats,
    make_exact_bicubic_model,
    sample_bicubic_points,
)


DEFAULT_ARTIFACTS = ",".join(
    str(ROOT / "outputs" / f"bicubic_global_h_metric_k{k}_weighted.npz") for k in (1, 2, 3)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--seeds", default="9201,9202,9203,9204,9205,9206,9207,9208")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "bicubic_h_convergence_audit.json",
    )
    return parser.parse_args()


def parse_paths(text: str) -> list[Path]:
    return [Path(value.strip()).expanduser().resolve() for value in text.split(",") if value.strip()]


def parse_seeds(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def confidence_interval(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array)))
    mean = float(np.mean(array))
    return [mean - 1.96 * standard_error, mean + 1.96 * standard_error]


def main() -> None:
    args = parse_args()
    artifacts = []
    for path in parse_paths(args.artifacts):
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        artifact = np.load(path)
        artifacts.append(
            {
                "path": path,
                "artifact": artifact,
                "k": int(artifact["global_section_degree"]),
                "exponents": artifact["global_section_exponents"],
                "h_matrix": artifact["global_h_matrix"],
                "normalization": float(artifact["global_section_normalization"]),
            }
        )
    artifacts.sort(key=lambda row: row["k"])
    if len({row["k"] for row in artifacts}) != len(artifacts):
        raise SystemExit("artifacts must have distinct degrees")
    model_seed = int(artifacts[0]["artifact"]["bicubic_model_seed"])
    model = make_exact_bicubic_model(model_seed)
    for row in artifacts:
        if int(row["artifact"]["bicubic_model_seed"]) != model_seed or not np.allclose(
            row["artifact"]["bicubic_coefficients"], model.coefficients
        ):
            raise SystemExit("all artifacts must use the same exact bicubic model")

    by_k = {row["k"]: [] for row in artifacts}
    rows = []
    for seed in parse_seeds(args.seeds):
        points = sample_bicubic_points(model, args.points, seed=seed)
        importance = bicubic_importance_weights(points)
        baseline_metrics = bicubic_baseline_metrics(points)
        baseline = bicubic_residual_stats(points, baseline_metrics)
        baseline_weighted = bicubic_residual_stats(points, baseline_metrics, importance)
        seed_row = {
            "seed": seed,
            "baseline_rms": baseline.rms,
            "baseline_weighted_rms": baseline_weighted.rms,
            "metrics": {},
        }
        for artifact in artifacts:
            metrics = bicubic_global_h_metrics(
                points,
                artifact["exponents"],
                artifact["h_matrix"],
                normalization=artifact["normalization"],
            )
            stats = bicubic_residual_stats(points, metrics)
            weighted = bicubic_residual_stats(points, metrics, importance)
            metric_row = {
                "rms": stats.rms,
                "weighted_rms": weighted.rms,
                "max_abs": stats.max_abs,
                "min_eigenvalue": stats.min_eigenvalue,
            }
            seed_row["metrics"][str(artifact["k"])] = metric_row
            by_k[artifact["k"]].append(metric_row)
        seed_row["strictly_decreasing_unweighted"] = all(
            seed_row["metrics"][str(left["k"])]["rms"]
            > seed_row["metrics"][str(right["k"])]["rms"]
            for left, right in zip(artifacts, artifacts[1:])
        )
        seed_row["strictly_decreasing_weighted"] = all(
            seed_row["metrics"][str(left["k"])]["weighted_rms"]
            > seed_row["metrics"][str(right["k"])]["weighted_rms"]
            for left, right in zip(artifacts, artifacts[1:])
        )
        rows.append(seed_row)

    aggregate = []
    for artifact in artifacts:
        metric_rows = by_k[artifact["k"]]
        unweighted = [row["rms"] for row in metric_rows]
        weighted = [row["weighted_rms"] for row in metric_rows]
        aggregate.append(
            {
                "k": artifact["k"],
                "artifact": str(artifact["path"]),
                "section_count": int(len(artifact["exponents"])),
                "mean_rms": float(np.mean(unweighted)),
                "rms_95_percent_ci": confidence_interval(unweighted),
                "mean_weighted_rms": float(np.mean(weighted)),
                "weighted_rms_95_percent_ci": confidence_interval(weighted),
                "max_seed_weighted_rms": float(np.max(weighted)),
                "min_metric_eigenvalue": float(
                    np.min([row["min_eigenvalue"] for row in metric_rows])
                ),
            }
        )
    ks = np.asarray([row["k"] for row in aggregate], dtype=float)
    unweighted_means = np.asarray([row["mean_rms"] for row in aggregate])
    weighted_means = np.asarray([row["mean_weighted_rms"] for row in aggregate])
    summary = {
        "description": "Paired fresh-seed k convergence of exact bicubic full-H metrics.",
        "model_seed": model_seed,
        "seeds": parse_seeds(args.seeds),
        "points_per_seed": args.points,
        "strictly_decreasing_unweighted_on_every_seed": all(
            row["strictly_decreasing_unweighted"] for row in rows
        ),
        "strictly_decreasing_weighted_on_every_seed": all(
            row["strictly_decreasing_weighted"] for row in rows
        ),
        "empirical_unweighted_log_log_slope": float(
            np.polyfit(np.log(ks), np.log(unweighted_means), 1)[0]
        ),
        "empirical_weighted_log_log_slope": float(
            np.polyfit(np.log(ks), np.log(weighted_means), 1)[0]
        ),
        "aggregate": aggregate,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in aggregate:
        print(
            f"k={row['k']}: sections={row['section_count']}, rms={row['mean_rms']:.6e}, "
            f"weighted={row['mean_weighted_rms']:.6e}, min eig={row['min_metric_eigenvalue']:.6e}"
        )
    print(
        "strict every seed: "
        f"unweighted={summary['strictly_decreasing_unweighted_on_every_seed']}, "
        f"weighted={summary['strictly_decreasing_weighted_on_every_seed']}"
    )
    print(f"wrote {args.out}")
    if any(row["min_metric_eigenvalue"] <= 0 for row in aggregate):
        raise SystemExit("a bicubic convergence artifact is not positive")


if __name__ == "__main__":
    main()
