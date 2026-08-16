#!/usr/bin/env python3
"""Scan a low-dimensional convex ensemble of trained Kähler metrics."""

from __future__ import annotations

import argparse
from math import comb
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, standard_errors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--exact-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--points-per-seed", type=int, required=True)
    parser.add_argument(
        "--simplex-denominator",
        type=int,
        default=4,
        help="Scan weights in multiples of 1/N on the probability simplex.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def integer_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield ordered non-negative integer compositions in deterministic order."""

    if total < 0 or parts <= 0:
        raise ValueError("total must be non-negative and parts must be positive")
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in integer_compositions(total - first, parts - 1):
            yield (first, *remainder)


def simplex_weight_grid(artifacts: int, denominator: int) -> list[np.ndarray]:
    """Return the lattice of convex weights with the requested denominator."""

    if artifacts <= 0 or denominator <= 0:
        raise ValueError("artifact count and denominator must be positive")
    weights = [
        np.asarray(composition, dtype=float) / denominator
        for composition in integer_compositions(denominator, artifacts)
    ]
    expected = comb(denominator + artifacts - 1, artifacts - 1)
    if len(weights) != expected:
        raise AssertionError("simplex lattice cardinality is inconsistent")
    return weights


def normalized_ratio_tail(log_eta: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    values = np.asarray(log_eta, dtype=float)
    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights /= np.sum(normalized_weights)
    shifted = np.exp(values - float(np.max(values)))
    ratio = shifted / float(np.sum(normalized_weights * shifted))
    contributions = normalized_weights * (1.0 - ratio) ** 2
    energy = float(np.sum(contributions))
    order = np.argsort(contributions)[::-1]
    if energy <= 0:
        cumulative = np.zeros_like(contributions)
    else:
        cumulative = np.cumsum(contributions[order]) / energy

    def count_for_fraction(fraction: float) -> int:
        if energy <= 0:
            return 0
        return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    return {
        "maximum_normalized_ratio": float(np.max(ratio)),
        "maximum_ratio_point_index": int(np.argmax(ratio)),
        "maximum_energy_contribution_point_index": int(order[0]),
        "energy_concentration": {
            str(count): (
                float(np.sum(contributions[order[:count]]) / energy)
                if energy > 0
                else 0.0
            )
            for count in (1, 4, 16, 64, 256)
            if count <= len(order)
        },
        "points_for_energy_fraction": {
            str(fraction): count_for_fraction(fraction) for fraction in (0.5, 0.9, 0.99)
        },
    }


def weight_label(weights: np.ndarray) -> str:
    active = [
        f"a{index}={weight:.6g}" for index, weight in enumerate(weights) if weight > 0
    ]
    return ",".join(active)


def main() -> None:
    args = parse_args()
    if args.points_per_seed <= 0:
        raise SystemExit("points-per-seed must be positive")
    if args.simplex_denominator <= 0:
        raise SystemExit("simplex-denominator must be positive")
    if len(args.artifacts) < 2:
        raise SystemExit("at least two artifacts are required")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("evaluation seeds must be distinct")

    started = time.perf_counter()
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    artifact_paths = [path.expanduser().resolve() for path in args.artifacts]
    artifacts = [adapter.load_h_artifact(path, model) for path in artifact_paths]
    degrees = {tuple(artifact.degree) for artifact in artifacts}
    normalizations = np.asarray(
        [artifact.normalization for artifact in artifacts], dtype=float
    )
    if len(degrees) != 1:
        raise SystemExit("all artifacts must use the same section degree")
    if not np.allclose(normalizations, normalizations[0], rtol=0, atol=1e-14):
        raise SystemExit("all artifacts must use the same metric normalization")

    candidate_weights = simplex_weight_grid(len(artifacts), args.simplex_denominator)
    rows: list[list[dict[str, Any]]] = [[] for _ in candidate_weights]
    sampling_diagnostics: dict[str, Any] = {}
    for seed in args.seeds:
        points, sampling = adapter.sample_points_with_diagnostics(
            model,
            args.points_per_seed,
            seed=seed,
        )
        sampling_diagnostics[str(seed)] = sampling
        importance_weights = adapter.importance_weights(points)
        log_omega = np.asarray(
            [adapter.holomorphic_volume_log_density(point) for point in points],
            dtype=float,
        )
        base_metrics = np.asarray(
            [adapter.h_metrics(points, artifact) for artifact in artifacts],
            dtype=np.complex128,
        )
        for index, weights in enumerate(candidate_weights):
            metrics = np.einsum("a,anij->nij", weights, base_metrics, optimize=True)
            eigenvalues = np.linalg.eigvalsh(metrics)
            if np.any(eigenvalues <= 0):
                raise FloatingPointError("convex metric ensemble is not positive")
            residuals = np.sum(np.log(eigenvalues), axis=1) - log_omega
            stats = standard_errors(residuals, importance_weights)
            rows[index].append(
                {
                    "seed": int(seed),
                    **stats,
                    "min_metric_eigenvalue": float(np.min(eigenvalues)),
                    "tail": normalized_ratio_tail(residuals, importance_weights),
                }
            )
        print(f"evaluated seed {seed}", flush=True)

    candidates = []
    for weights, seed_rows in zip(candidate_weights, rows, strict=True):
        candidates.append(
            {
                "label": weight_label(weights),
                "weights": [float(value) for value in weights],
                "active_artifact_count": int(np.count_nonzero(weights)),
                "mean_sigma": float(np.mean([row["sigma"] for row in seed_rows])),
                "maximum_sigma": float(np.max([row["sigma"] for row in seed_rows])),
                "mean_sqrt_squared_energy": float(
                    np.mean([row["sqrt_squared_energy"] for row in seed_rows])
                ),
                "maximum_sqrt_squared_energy": float(
                    np.max([row["sqrt_squared_energy"] for row in seed_rows])
                ),
                "maximum_normalized_ratio": float(
                    np.max([row["normalized_ratio_max"] for row in seed_rows])
                ),
                "minimum_metric_eigenvalue": float(
                    np.min([row["min_metric_eigenvalue"] for row in seed_rows])
                ),
                "seeds": seed_rows,
            }
        )

    output = {
        "schema_version": 1,
        "description": (
            "Low-dimensional convex combinations of positive Kähler metrics "
            "in one fixed Kähler class. Non-negative weights summing to one "
            "preserve positivity, closedness, and the class."
        ),
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "artifacts": [
            {
                "index": index,
                "path": str(path),
                "degree": list(artifact.degree),
                "section_count": artifact.section_count,
                "normalization": artifact.normalization,
            }
            for index, (path, artifact) in enumerate(
                zip(artifact_paths, artifacts, strict=True)
            )
        ],
        "simplex_denominator": args.simplex_denominator,
        "candidate_count": len(candidate_weights),
        "evaluation_seeds": args.seeds,
        "points_per_seed": args.points_per_seed,
        "sampling_diagnostics": sampling_diagnostics,
        "candidates": candidates,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
