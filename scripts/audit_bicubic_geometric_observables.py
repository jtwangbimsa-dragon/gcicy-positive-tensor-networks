#!/usr/bin/env python3
"""Audit bicubic metric volume and line-bundle slopes against exact intersections."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.bicubic_model import (  # noqa: E402
    bicubic_baseline_metrics,
    bicubic_factor_metrics,
    bicubic_global_h_metrics,
    bicubic_proposal_log_density,
    make_exact_bicubic_model,
    sample_bicubic_points,
)


DEFAULT_ARTIFACTS = ",".join(
    [
        str(ROOT / "outputs" / "bicubic_global_h_metric_k1_weighted.npz"),
        str(ROOT / "outputs" / "bicubic_global_h_metric_k2_weighted.npz"),
    ]
)
EXACT_VOLUME_RATIO = Fraction(2, 1)
EXACT_SLOPE_RATIOS = (Fraction(3, 1), Fraction(3, 1))
LINE_BUNDLES = ((1, 0), (0, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--seeds", default="9201,9202,9203,9204,9205,9206,9207,9208")
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "bicubic_geometric_observables_audit.json",
    )
    return parser.parse_args()


def parse_paths(text: str) -> list[Path]:
    return [Path(value.strip()).expanduser().resolve() for value in text.split(",") if value.strip()]


def parse_seeds(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def metric_observables(metrics: np.ndarray, proposal_density: np.ndarray, factors: np.ndarray) -> dict:
    eigenvalues = np.linalg.eigvalsh(metrics)
    if np.min(eigenvalues) <= 0:
        raise FloatingPointError("observable audit encountered a non-positive metric")
    volume_ratio = np.prod(eigenvalues, axis=1) / proposal_density
    inverse = np.linalg.inv(metrics)
    slopes = []
    for factor_index in range(2):
        contraction = np.real(np.einsum("nij,nji->n", inverse, factors[:, factor_index]))
        slopes.append(volume_ratio * contraction)
    return {
        "volume_samples": volume_ratio,
        "slope_samples": np.asarray(slopes).T,
        "min_metric_eigenvalue": float(np.min(eigenvalues)),
    }


def aggregate_seed_means(values: list[float], exact: float) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        "mean": mean,
        "seed_standard_error": standard_error,
        "normal_95_percent_ci": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "exact": exact,
        "relative_error": float(mean / exact - 1.0),
    }


def main() -> None:
    args = parse_args()
    paths = parse_paths(args.artifacts)
    if not paths:
        raise SystemExit("at least one artifact is required")
    artifacts = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        artifact = np.load(path)
        artifacts.append(
            {
                "path": path,
                "artifact": artifact,
                "label": f"k{int(artifact['global_section_degree'])}",
            }
        )

    first = artifacts[0]["artifact"]
    model_seed = int(first["bicubic_model_seed"])
    model = make_exact_bicubic_model(model_seed)
    for row in artifacts:
        artifact = row["artifact"]
        if int(artifact["bicubic_model_seed"]) != model_seed:
            raise SystemExit("all artifacts must use the same model seed")
        if not np.allclose(artifact["bicubic_coefficients"], model.coefficients):
            raise SystemExit(f"artifact coefficients do not match model: {row['path']}")

    candidates = [{"label": "ambient_fs", "path": None}] + artifacts
    seed_rows = []
    for seed in parse_seeds(args.seeds):
        points = sample_bicubic_points(model, args.points, seed=seed)
        proposal_density = np.exp(
            np.asarray([bicubic_proposal_log_density(point) for point in points], dtype=float)
        )
        factors = np.asarray([bicubic_factor_metrics(point) for point in points], dtype=np.complex128)
        row = {"seed": seed, "points": args.points, "candidates": {}}
        for candidate in candidates:
            if candidate["label"] == "ambient_fs":
                metrics = bicubic_baseline_metrics(points)
            else:
                artifact = candidate["artifact"]
                metrics = bicubic_global_h_metrics(
                    points,
                    artifact["global_section_exponents"],
                    artifact["global_h_matrix"],
                    normalization=float(artifact["global_section_normalization"]),
                )
            observable = metric_observables(metrics, proposal_density, factors)
            row["candidates"][candidate["label"]] = {
                "volume_ratio_mean": float(np.mean(observable["volume_samples"])),
                "slope_ratio_means": [
                    float(value) for value in np.mean(observable["slope_samples"], axis=0)
                ],
                "min_metric_eigenvalue": observable["min_metric_eigenvalue"],
            }
        seed_rows.append(row)
        print(f"seed {seed}: completed {len(candidates)} candidates", flush=True)

    exact_volume = float(EXACT_VOLUME_RATIO)
    exact_slopes = [float(value) for value in EXACT_SLOPE_RATIOS]
    aggregate = {}
    for candidate in candidates:
        label = candidate["label"]
        candidate_rows = [row["candidates"][label] for row in seed_rows]
        aggregate[label] = {
            "artifact": str(candidate["path"]) if candidate["path"] is not None else None,
            "volume_ratio": aggregate_seed_means(
                [row["volume_ratio_mean"] for row in candidate_rows], exact_volume
            ),
            "slope_ratios": [
                aggregate_seed_means(
                    [row["slope_ratio_means"][index] for row in candidate_rows], exact_slopes[index]
                )
                for index in range(2)
            ],
            "min_metric_eigenvalue": float(
                np.min([row["min_metric_eigenvalue"] for row in candidate_rows])
            ),
        }

    summary = {
        "description": "Metric volume and ambient line-bundle slope audit against exact bicubic intersections.",
        "model_seed": model_seed,
        "seeds": parse_seeds(args.seeds),
        "points_per_seed": args.points,
        "exact": {
            "volume_ratio": exact_volume,
            "volume_ratio_fraction": str(EXACT_VOLUME_RATIO),
            "line_bundles": [list(value) for value in LINE_BUNDLES],
            "slope_ratios": exact_slopes,
            "slope_ratio_fractions": [str(value) for value in EXACT_SLOPE_RATIOS],
        },
        "aggregate": aggregate,
        "rows": seed_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for label, row in aggregate.items():
        print(
            f"{label}: volume={row['volume_ratio']['mean']:.6f} "
            f"(exact {exact_volume:.6f}), slopes="
            + ",".join(f"{item['mean']:.6f}" for item in row["slope_ratios"])
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
