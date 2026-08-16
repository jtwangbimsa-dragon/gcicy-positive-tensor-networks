#!/usr/bin/env python3
"""Scan a positive rank-one H repair anchored at one disclosed tail point."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    HMetricArtifact,
    get_adapter,
    standard_errors,
    targeted_section_rank_one_update,
)


def parse_batch(text: str) -> tuple[int, int]:
    try:
        seed_text, points_text = text.split(":", maxsplit=1)
        seed, points = int(seed_text), int(points_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("batches must have form seed:points") from exc
    if points <= 0:
        raise argparse.ArgumentTypeError("batch point count must be positive")
    return seed, points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--exact-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--target-point-index", type=int, required=True)
    parser.add_argument("--cluster-size", type=int, default=4)
    parser.add_argument(
        "--evaluation-batches", type=parse_batch, nargs="+", required=True
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=(0.0, 0.3, 1.0, 3.0, 10.0, 15.0, 30.0, 100.0),
    )
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def metrics_from_section_data(
    values: np.ndarray,
    derivatives: np.ndarray,
    h_matrix: np.ndarray,
    normalization: float,
) -> np.ndarray:
    h_values = np.einsum("ab,nb->na", h_matrix, values, optimize=True)
    denominator = np.real(
        np.einsum("na,na->n", np.conjugate(values), h_values, optimize=True)
    )
    if np.any(denominator <= 0):
        raise FloatingPointError("H-metric denominator is not positive")
    h_derivatives = np.einsum("ab,nbj->naj", h_matrix, derivatives, optimize=True)
    first = np.einsum(
        "nmi,nmj->nij",
        np.conjugate(derivatives),
        h_derivatives,
        optimize=True,
    )
    gradient = np.einsum(
        "nm,nmj->nj", np.conjugate(values), h_derivatives, optimize=True
    )
    metrics = first / denominator[:, None, None]
    metrics -= (
        np.conjugate(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metrics *= normalization
    return 0.5 * (metrics + np.conjugate(np.swapaxes(metrics, 1, 2)))


def tail_summary(residuals: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights /= np.sum(normalized_weights)
    shifted = np.exp(residuals - float(np.max(residuals)))
    ratio = shifted / float(np.sum(normalized_weights * shifted))
    contributions = normalized_weights * (1.0 - ratio) ** 2
    energy = float(np.sum(contributions))
    order = np.argsort(contributions)[::-1]
    ratio_index = int(np.argmax(ratio))
    error_index = int(order[0])
    return {
        "maximum_normalized_ratio": float(ratio[ratio_index]),
        "maximum_ratio_point_index": ratio_index,
        "maximum_error_point_index": error_index,
        "maximum_error_fraction": float(contributions[error_index] / energy),
        "energy_concentration": {
            str(count): float(np.sum(contributions[order[:count]]) / energy)
            for count in (1, 4, 16, 64, 256)
            if count <= len(order)
        },
    }


def save_candidate(
    source: Path, destination: Path, h_matrix: np.ndarray, strength: float
) -> None:
    with np.load(source, allow_pickle=False) as payload:
        output = {key: np.asarray(payload[key]).copy() for key in payload.files}
    output["global_h_matrix"] = np.asarray(h_matrix, dtype=np.complex128)
    output["targeted_rank_one_repair_strength"] = np.asarray(strength)
    output["targeted_rank_one_repair_source"] = np.asarray(source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output)


def main() -> None:
    args = parse_args()
    if args.target_point_index < 0 or args.cluster_size <= 0:
        raise SystemExit("target point index and cluster size are invalid")
    strengths = sorted(set(float(value) for value in args.strengths))
    if not strengths or min(strengths) < 0:
        raise SystemExit("strengths must be non-empty and non-negative")
    if len({seed for seed, _ in args.evaluation_batches}) != len(
        args.evaluation_batches
    ):
        raise SystemExit("evaluation batch seeds must be distinct")

    started = time.perf_counter()
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    source_path = args.artifact.expanduser().resolve()
    source = adapter.load_h_artifact(source_path, model)
    replay_count = (
        args.target_point_index // args.cluster_size + 1
    ) * args.cluster_size
    target_points, target_sampling = adapter.sample_points_with_diagnostics(
        model, replay_count, seed=args.target_seed
    )
    target_point = target_points[args.target_point_index]
    target_values, _ = adapter.section_values_and_jacobian(
        target_point, source.section_exponents
    )
    original_section_norm = float(
        np.real(np.vdot(target_values, source.h_matrix @ target_values))
    )

    candidates: list[tuple[float, HMetricArtifact]] = []
    candidate_paths: dict[float, str | None] = {}
    for strength in strengths:
        h_matrix = targeted_section_rank_one_update(
            source.h_matrix, target_values, strength
        )
        if args.candidate_dir is None:
            path = Path(f"<targeted-rank-one-{strength:.6g}>")
            candidate_paths[strength] = None
        else:
            token = f"{strength:.6g}".replace(".", "p")
            path = args.candidate_dir.expanduser().resolve() / (
                f"{source_path.stem}_targeted_rank_one_b{token}.npz"
            )
            save_candidate(source_path, path, h_matrix, strength)
            candidate_paths[strength] = str(path)
        candidates.append((strength, replace(source, path=path, h_matrix=h_matrix)))

    target_rows: dict[float, dict[str, Any]] = {}
    for strength, artifact in candidates:
        metric = adapter.h_metric(target_point, artifact)
        eigenvalues = np.linalg.eigvalsh(metric)
        section_norm = float(
            np.real(np.vdot(target_values, artifact.h_matrix @ target_values))
        )
        target_rows[strength] = {
            "raw_log_monge_ampere_ratio": float(
                adapter.monge_ampere_log_error(target_point, metric)
            ),
            "metric_eigenvalues": [float(value) for value in eigenvalues],
            "section_h_norm_squared": section_norm,
            "section_h_norm_ratio": section_norm / original_section_norm,
        }

    rows: dict[float, list[dict[str, Any]]] = {strength: [] for strength in strengths}
    sampling_diagnostics: dict[str, Any] = {}
    for seed, point_count in args.evaluation_batches:
        points, sampling = adapter.sample_points_with_diagnostics(
            model, point_count, seed=seed
        )
        sampling_diagnostics[str(seed)] = sampling
        importance_weights = adapter.importance_weights(points)
        log_omega = np.asarray(
            [adapter.holomorphic_volume_log_density(point) for point in points],
            dtype=float,
        )
        evaluated = [
            adapter.section_values_and_jacobian(point, source.section_exponents)
            for point in points
        ]
        values = np.asarray([item[0] for item in evaluated], dtype=np.complex128)
        derivatives = np.asarray([item[1] for item in evaluated], dtype=np.complex128)
        for strength, artifact in candidates:
            metrics = metrics_from_section_data(
                values,
                derivatives,
                artifact.h_matrix,
                artifact.normalization,
            )
            eigenvalues = np.linalg.eigvalsh(metrics)
            if np.any(eigenvalues <= 0):
                raise FloatingPointError("rank-one candidate metric is not positive")
            residuals = np.sum(np.log(eigenvalues), axis=1) - log_omega
            rows[strength].append(
                {
                    "seed": int(seed),
                    "points": int(point_count),
                    **standard_errors(residuals, importance_weights),
                    "min_metric_eigenvalue": float(np.min(eigenvalues)),
                    "tail": tail_summary(residuals, importance_weights),
                }
            )
        print(f"evaluated seed {seed}", flush=True)

    candidate_rows = []
    for strength, artifact in candidates:
        seed_rows = rows[strength]
        candidate_rows.append(
            {
                "strength": strength,
                "artifact": candidate_paths[strength],
                "h_condition_number": float(np.linalg.cond(artifact.h_matrix)),
                "target_point": target_rows[strength],
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
                "seeds": seed_rows,
            }
        )

    output = {
        "schema_version": 1,
        "description": (
            "Exploratory positive rank-one H repair targeted at one disclosed "
            "metric-tail point. No new blind seed is used."
        ),
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "source_artifact": str(source_path),
        "target_seed": args.target_seed,
        "target_point_index": args.target_point_index,
        "target_replay_count": replay_count,
        "target_sampling_diagnostics": target_sampling,
        "original_target_section_h_norm_squared": original_section_norm,
        "evaluation_batches": [
            {"seed": seed, "points": points} for seed, points in args.evaluation_batches
        ],
        "sampling_diagnostics": sampling_diagnostics,
        "candidates": candidate_rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
