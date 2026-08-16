#!/usr/bin/env python3
"""Jointly scan two sequential positive rank-one H repairs."""

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
from scripts.scan_targeted_rank_one_h_repair import (  # noqa: E402
    metrics_from_section_data,
    parse_batch,
    tail_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--exact-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--first-point-index", type=int, required=True)
    parser.add_argument("--second-point-index", type=int, required=True)
    parser.add_argument("--cluster-size", type=int, default=4)
    parser.add_argument("--first-strengths", type=float, nargs="+", required=True)
    parser.add_argument("--second-strengths", type=float, nargs="+", required=True)
    parser.add_argument(
        "--evaluation-batches", type=parse_batch, nargs="+", required=True
    )
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def save_candidate(
    source: Path,
    destination: Path,
    h_matrix: np.ndarray,
    first_strength: float,
    second_strength: float,
) -> None:
    with np.load(source, allow_pickle=False) as payload:
        output = {key: np.asarray(payload[key]).copy() for key in payload.files}
    output["global_h_matrix"] = np.asarray(h_matrix, dtype=np.complex128)
    output["targeted_rank_two_repair_first_strength"] = np.asarray(first_strength)
    output["targeted_rank_two_repair_second_strength"] = np.asarray(second_strength)
    output["targeted_rank_two_repair_source"] = np.asarray(source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output)


def point_row(adapter: Any, point: Any, artifact: HMetricArtifact) -> dict[str, Any]:
    metric = adapter.h_metric(point, artifact)
    eigenvalues = np.linalg.eigvalsh(metric)
    values, _ = adapter.section_values_and_jacobian(point, artifact.section_exponents)
    section_norm = float(np.real(np.vdot(values, artifact.h_matrix @ values)))
    return {
        "raw_log_monge_ampere_ratio": float(
            adapter.monge_ampere_log_error(point, metric)
        ),
        "metric_eigenvalues": [float(value) for value in eigenvalues],
        "section_h_norm_squared": section_norm,
    }


def main() -> None:
    args = parse_args()
    if min(args.first_point_index, args.second_point_index) < 0:
        raise SystemExit("point indices must be non-negative")
    if args.cluster_size <= 0 or args.first_point_index == args.second_point_index:
        raise SystemExit("cluster size or target-point inventory is invalid")
    first_strengths = sorted(set(float(value) for value in args.first_strengths))
    second_strengths = sorted(set(float(value) for value in args.second_strengths))
    if (
        not first_strengths
        or not second_strengths
        or min(first_strengths) < 0
        or min(second_strengths) < 0
    ):
        raise SystemExit("strength grids must be non-empty and non-negative")
    if len({seed for seed, _ in args.evaluation_batches}) != len(
        args.evaluation_batches
    ):
        raise SystemExit("evaluation batch seeds must be distinct")

    started = time.perf_counter()
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    source_path = args.artifact.expanduser().resolve()
    source = adapter.load_h_artifact(source_path, model)
    maximum_index = max(args.first_point_index, args.second_point_index)
    replay_count = (maximum_index // args.cluster_size + 1) * args.cluster_size
    target_points, target_sampling = adapter.sample_points_with_diagnostics(
        model, replay_count, seed=args.target_seed
    )
    first_point = target_points[args.first_point_index]
    second_point = target_points[args.second_point_index]
    first_values, _ = adapter.section_values_and_jacobian(
        first_point, source.section_exponents
    )
    second_values, _ = adapter.section_values_and_jacobian(
        second_point, source.section_exponents
    )

    candidates: list[tuple[tuple[float, float], HMetricArtifact]] = []
    candidate_paths: dict[tuple[float, float], str | None] = {}
    for first_strength in first_strengths:
        first_h = targeted_section_rank_one_update(
            source.h_matrix, first_values, first_strength
        )
        for second_strength in second_strengths:
            h_matrix = targeted_section_rank_one_update(
                first_h, second_values, second_strength
            )
            key = (first_strength, second_strength)
            if args.candidate_dir is None:
                path = Path(
                    f"<targeted-rank-two-{first_strength:.6g}-{second_strength:.6g}>"
                )
                candidate_paths[key] = None
            else:
                first_token = f"{first_strength:.6g}".replace(".", "p")
                second_token = f"{second_strength:.6g}".replace(".", "p")
                path = args.candidate_dir.expanduser().resolve() / (
                    f"{source_path.stem}_targeted_rank_two_"
                    f"b1_{first_token}_b2_{second_token}.npz"
                )
                save_candidate(
                    source_path,
                    path,
                    h_matrix,
                    first_strength,
                    second_strength,
                )
                candidate_paths[key] = str(path)
            candidates.append((key, replace(source, path=path, h_matrix=h_matrix)))

    rows: dict[tuple[float, float], list[dict[str, Any]]] = {
        key: [] for key, _ in candidates
    }
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
        for key, artifact in candidates:
            metrics = metrics_from_section_data(
                values,
                derivatives,
                artifact.h_matrix,
                artifact.normalization,
            )
            eigenvalues = np.linalg.eigvalsh(metrics)
            if np.any(eigenvalues <= 0):
                raise FloatingPointError("rank-two candidate metric is not positive")
            residuals = np.sum(np.log(eigenvalues), axis=1) - log_omega
            rows[key].append(
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
    for key, artifact in candidates:
        first_strength, second_strength = key
        seed_rows = rows[key]
        candidate_rows.append(
            {
                "first_strength": first_strength,
                "second_strength": second_strength,
                "artifact": candidate_paths[key],
                "h_condition_number": float(np.linalg.cond(artifact.h_matrix)),
                "first_target_point": point_row(adapter, first_point, artifact),
                "second_target_point": point_row(adapter, second_point, artifact),
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
            "Joint grid of two sequential positive rank-one H repairs on "
            "disclosed hard points. No new blind seed is used."
        ),
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "source_artifact": str(source_path),
        "target_seed": args.target_seed,
        "first_point_index": args.first_point_index,
        "second_point_index": args.second_point_index,
        "target_replay_count": replay_count,
        "target_sampling_diagnostics": target_sampling,
        "first_strengths": first_strengths,
        "second_strengths": second_strengths,
        "evaluation_batches": [
            {"seed": seed, "points": points} for seed, points in args.evaluation_batches
        ],
        "sampling_diagnostics": sampling_diagnostics,
        "candidate_count": len(candidates),
        "candidates": candidate_rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
