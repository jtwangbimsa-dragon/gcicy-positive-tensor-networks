#!/usr/bin/env python3
"""Evaluate literature-standard Monge-Ampere sigma and squared-energy errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    generic_global_h_metrics,
    generic_importance_weights,
    generic_residual_values,
    make_exact_generic_model,
    sample_generic_gcicy_points,
)
from gcicy_metric.bicubic_model import (  # noqa: E402
    bicubic_global_h_metrics,
    bicubic_importance_weights,
    bicubic_residual_values,
    make_exact_bicubic_model,
    sample_bicubic_points,
)


DEFAULT_GCICY_ARTIFACTS = ",".join(
    str(ROOT / "outputs" / name)
    for name in (
        "gcicy_generic_global_h_metric_weighted.npz",
        "gcicy_generic_global_h_metric_k2_rank40_weighted.npz",
        "gcicy_generic_global_h_metric_k3_rank64_whitened_gpu.npz",
    )
)
DEFAULT_BICUBIC_ARTIFACTS = ",".join(
    str(ROOT / "outputs" / name)
    for name in (
        "bicubic_global_h_metric_k1_weighted.npz",
        "bicubic_global_h_metric_k2_weighted.npz",
        "bicubic_global_h_metric_k3_k2xk1_gpu.npz",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcicy-artifacts", default=DEFAULT_GCICY_ARTIFACTS)
    parser.add_argument("--bicubic-artifacts", default=DEFAULT_BICUBIC_ARTIFACTS)
    parser.add_argument("--gcicy-seeds", default="11101,11102,11103,11104,11105,11106,11107,11108")
    parser.add_argument("--bicubic-seeds", default="12101,12102,12103,12104,12105,12106,12107,12108")
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "standard_sigma_comparison_audit.json",
    )
    return parser.parse_args()


def parse_paths(text: str) -> list[Path]:
    return [Path(item.strip()).expanduser().resolve() for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def confidence_interval(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return [mean, mean]
    half_width = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(len(array))
    return [mean - half_width, mean + half_width]


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights) ** 2 / np.sum(weights**2))


def standard_errors(log_eta: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Return normalized volume-ratio errors with Omega-volume quadrature weights."""

    log_eta = np.asarray(log_eta, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if log_eta.shape != weights.shape or not np.all(np.isfinite(log_eta)):
        raise ValueError("log_eta and weights must be finite arrays of equal shape")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    normalized_weights = weights / np.sum(weights)
    shifted_ratio = np.exp(log_eta - float(np.max(log_eta)))
    normalized_ratio = shifted_ratio / float(np.sum(normalized_weights * shifted_ratio))
    ratio_error = 1.0 - normalized_ratio
    inverse_ratio_error = 1.0 - 1.0 / normalized_ratio
    centered_log_eta = log_eta - float(np.sum(normalized_weights * log_eta))
    energy = float(np.sum(normalized_weights * ratio_error**2))
    return {
        "sigma": float(np.sum(normalized_weights * np.abs(ratio_error))),
        "inverse_sigma": float(np.sum(normalized_weights * np.abs(inverse_ratio_error))),
        "squared_energy": energy,
        "sqrt_squared_energy": float(np.sqrt(energy)),
        "weighted_centered_log_ma_rms": float(
            np.sqrt(np.sum(normalized_weights * centered_log_eta**2))
        ),
        "normalized_ratio_min": float(np.min(normalized_ratio)),
        "normalized_ratio_max": float(np.max(normalized_ratio)),
        "importance_effective_sample_size": effective_sample_size(weights),
    }


def aggregate_rows(rows: list[dict]) -> dict:
    result = {"seeds": rows}
    for key in (
        "sigma",
        "inverse_sigma",
        "squared_energy",
        "sqrt_squared_energy",
        "weighted_centered_log_ma_rms",
    ):
        values = [float(row[key]) for row in rows]
        result[f"mean_{key}"] = float(np.mean(values))
        result[f"{key}_95_percent_ci"] = confidence_interval(values)
        result[f"max_seed_{key}"] = float(np.max(values))
    result["min_normalized_ratio"] = float(np.min([row["normalized_ratio_min"] for row in rows]))
    result["max_normalized_ratio"] = float(np.max([row["normalized_ratio_max"] for row in rows]))
    result["mean_importance_effective_sample_size"] = float(
        np.mean([row["importance_effective_sample_size"] for row in rows])
    )
    return result


def load_gcicy_artifacts(paths: list[Path]) -> tuple[object, list[dict]]:
    artifacts = []
    for path in paths:
        artifact = np.load(path)
        degree = tuple(int(value) for value in artifact["global_section_degree"])
        if len(set(degree)) != 1:
            raise SystemExit(f"expected a (k,k,k) gCICY degree: {path}")
        artifacts.append(
            {
                "path": path,
                "artifact": artifact,
                "k": degree[0],
                "exponents": artifact["global_section_exponents"],
                "h_matrix": artifact["global_h_matrix"],
                "normalization": float(artifact["global_section_normalization"]),
            }
        )
    artifacts.sort(key=lambda row: row["k"])
    model_seed = int(artifacts[0]["artifact"]["generic_model_seed"])
    model = make_exact_generic_model(model_seed)
    for row in artifacts:
        artifact = row["artifact"]
        if int(artifact["generic_model_seed"]) != model_seed:
            raise SystemExit("all gCICY artifacts must use the same model seed")
        if not np.allclose(artifact["p1_coefficients"], model.p1_coefficients) or not np.allclose(
            artifact["p2_tensor"], model.p2_tensor
        ):
            raise SystemExit(f"gCICY coefficients do not match the exact model: {row['path']}")
    return model, artifacts


def load_bicubic_artifacts(paths: list[Path]) -> tuple[object, list[dict]]:
    artifacts = []
    for path in paths:
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
    model_seed = int(artifacts[0]["artifact"]["bicubic_model_seed"])
    model = make_exact_bicubic_model(model_seed)
    for row in artifacts:
        artifact = row["artifact"]
        if int(artifact["bicubic_model_seed"]) != model_seed or not np.allclose(
            artifact["bicubic_coefficients"], model.coefficients
        ):
            raise SystemExit(f"bicubic coefficients do not match the exact model: {row['path']}")
    return model, artifacts


def audit_gcicy(paths: list[Path], seeds: list[int], points_per_seed: int) -> dict:
    model, artifacts = load_gcicy_artifacts(paths)
    by_k = {row["k"]: [] for row in artifacts}
    for seed in seeds:
        points = sample_generic_gcicy_points(model, points_per_seed, seed=seed)
        weights = generic_importance_weights(points)
        for row in artifacts:
            metrics = generic_global_h_metrics(
                points, row["exponents"], row["h_matrix"], normalization=row["normalization"]
            )
            stats = standard_errors(generic_residual_values(points, metrics), weights)
            by_k[row["k"]].append({"seed": seed, **stats})
    return {
        "model_seed": int(artifacts[0]["artifact"]["generic_model_seed"]),
        "points_per_seed": points_per_seed,
        "artifacts": [
            {
                "k": row["k"],
                "section_count": int(len(row["exponents"])),
                "artifact": str(row["path"]),
                **aggregate_rows(by_k[row["k"]]),
            }
            for row in artifacts
        ],
    }


def audit_bicubic(paths: list[Path], seeds: list[int], points_per_seed: int) -> dict:
    model, artifacts = load_bicubic_artifacts(paths)
    by_k = {row["k"]: [] for row in artifacts}
    for seed in seeds:
        points = sample_bicubic_points(model, points_per_seed, seed=seed)
        weights = bicubic_importance_weights(points)
        for row in artifacts:
            metrics = bicubic_global_h_metrics(
                points, row["exponents"], row["h_matrix"], normalization=row["normalization"]
            )
            stats = standard_errors(bicubic_residual_values(points, metrics), weights)
            by_k[row["k"]].append({"seed": seed, **stats})
    return {
        "model_seed": int(artifacts[0]["artifact"]["bicubic_model_seed"]),
        "points_per_seed": points_per_seed,
        "artifacts": [
            {
                "k": row["k"],
                "section_count": int(len(row["exponents"])),
                "artifact": str(row["path"]),
                **aggregate_rows(by_k[row["k"]]),
            }
            for row in artifacts
        ],
    }


def main() -> None:
    args = parse_args()
    if args.points <= 0:
        raise SystemExit("--points must be positive")
    summary = {
        "description": "Fresh-seed literature-standard normalized Monge-Ampere error audit.",
        "definition": {
            "eta": "det(g) / |Omega|^2 in each exact local chart",
            "normalized_eta": "eta / Omega-weighted mean(eta)",
            "sigma": "Omega-weighted mean(abs(1 - normalized_eta))",
            "inverse_sigma": (
                "Omega-weighted mean(abs(1 - 1 / normalized_eta)); this is the convention "
                "used in cymetric Eq. (4.8)"
            ),
            "squared_energy": "Omega-weighted mean((1 - normalized_eta)^2)",
        },
        "gcicy": audit_gcicy(
            parse_paths(args.gcicy_artifacts), parse_ints(args.gcicy_seeds), args.points
        ),
        "bicubic": audit_bicubic(
            parse_paths(args.bicubic_artifacts), parse_ints(args.bicubic_seeds), args.points
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for geometry in ("gcicy", "bicubic"):
        print(geometry)
        for row in summary[geometry]["artifacts"]:
            print(
                f"  k={row['k']}: sigma={row['mean_sigma']:.6e}, "
                f"inverse-sigma={row['mean_inverse_sigma']:.6e} "
                f"CI={row['sigma_95_percent_ci']}, E={row['mean_squared_energy']:.6e}, "
                f"log-RMS={row['mean_weighted_centered_log_ma_rms']:.6e}"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
