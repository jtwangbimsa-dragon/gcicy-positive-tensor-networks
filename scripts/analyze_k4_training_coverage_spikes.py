#!/usr/bin/env python3
"""Trace k=4 checkpoint spikes back to the frozen training fibres.

This diagnostic distinguishes three mechanisms that look identical in a tail
summary: a genuine sampling hole, a bad training point tolerated by the loss,
and a spike created during optimization.  It reproduces the exact train and
checkpoint seeds, evaluates the initial and selected H matrices, and probes
the selected checkpoint spikes on several local scales.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from gcicy_metric.pipeline.regularity import (  # noqa: E402
    estimate_h_metric_residual_gradient,
)


DEFAULT_TRAIN_SEEDS = tuple(range(72201, 72209))
DEFAULT_CHECKPOINT_SEEDS = tuple(range(72221, 72225))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="p4p1_type11_hirzebruch_x3")
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument(
        "--exact-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--initial-artifact", type=Path, required=True)
    parser.add_argument("--final-artifact", type=Path, required=True)
    parser.add_argument(
        "--train-seeds", type=int, nargs="+", default=DEFAULT_TRAIN_SEEDS
    )
    parser.add_argument(
        "--checkpoint-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINT_SEEDS,
    )
    parser.add_argument("--points-per-seed", type=int, default=32768)
    parser.add_argument("--sampling-workers", type=int, default=4)
    parser.add_argument("--sampling-cluster-size", type=int, default=4)
    parser.add_argument(
        "--sampling-backend", choices=("process", "thread"), default="process"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--nearest-query-batch-size", type=int, default=32)
    parser.add_argument("--bad-ratio-threshold", type=float, default=3.0)
    parser.add_argument("--control-points", type=int, default=512)
    parser.add_argument("--control-seed", type=int, default=919191)
    parser.add_argument("--neighborhood-points", type=int, default=96)
    parser.add_argument(
        "--neighborhood-radii",
        type=float,
        nargs="+",
        default=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    )
    parser.add_argument("--gradient-step", type=float, default=1e-3)
    parser.add_argument(
        "--gradients", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--point-payload", type=Path)
    return parser.parse_args()


def log_weighted_mean_exp(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape or len(values) == 0:
        raise ValueError("values and weights must be matching non-empty arrays")
    if np.min(weights) <= 0 or not np.all(np.isfinite(values)):
        raise FloatingPointError("log-mean inputs must be finite and positive")
    shift = float(np.max(values))
    return float(
        shift + np.log(np.sum(weights * np.exp(values - shift))) - np.log(np.sum(weights))
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)
    observed = np.quantile(np.asarray(values, dtype=float), probabilities)
    return {
        f"q{probability:g}": float(value)
        for probability, value in zip(probabilities, observed, strict=True)
    }


def point_coordinate_arrays(points: Sequence[Any]) -> tuple[np.ndarray, ...]:
    factors = []
    for factor_index in range(3):
        values = np.asarray(
            [point.coordinates[factor_index] for point in points],
            dtype=np.complex128,
        )
        norms = np.linalg.norm(values, axis=1)
        if np.min(norms) <= 0:
            raise FloatingPointError("homogeneous coordinate norm is zero")
        factors.append(values / norms[:, None])
    return tuple(factors)


def concatenate_coordinates(
    groups: Sequence[tuple[np.ndarray, ...]],
) -> tuple[np.ndarray, ...]:
    return tuple(np.concatenate([group[index] for group in groups], axis=0) for index in range(3))


def point_diagnostic(point: Any) -> dict[str, Any]:
    return {
        "projective_chart": [int(value) for value in point.projective_chart],
        "independent_indices": [int(value) for value in point.independent_indices],
        "jacobian_min_singular_value": float(point.jacobian_min_singular_value),
        "factor_coordinate_magnitudes": [
            np.abs(np.asarray(factor, dtype=np.complex128)).tolist()
            for factor in point.coordinates
        ],
    }


def prepare_features(
    adapter: Any,
    points: Sequence[Any],
    exponents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    for point in points:
        section_values, section_derivatives = adapter.section_values_and_jacobian(
            point, exponents
        )
        values.append(section_values)
        derivatives.append(section_derivatives)
    log_omega = np.asarray(
        [-2.0 * np.log(abs(point.residue_denominator)) for point in points],
        dtype=np.float32,
    )
    return (
        np.asarray(values, dtype=np.complex64),
        np.asarray(derivatives, dtype=np.complex64),
        log_omega,
    )


def evaluate_raw_residuals(
    torch: Any,
    values: np.ndarray,
    derivatives: np.ndarray,
    log_omega: np.ndarray,
    artifact: Any,
    *,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    h_matrix = torch.as_tensor(
        np.asarray(artifact.h_matrix, dtype=np.complex64),
        dtype=torch.complex64,
        device=device,
    )
    output = np.empty(len(values), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            section_values = torch.as_tensor(
                values[start:stop], dtype=torch.complex64, device=device
            )
            section_derivatives = torch.as_tensor(
                derivatives[start:stop], dtype=torch.complex64, device=device
            )
            omega = torch.as_tensor(
                log_omega[start:stop], dtype=torch.float32, device=device
            )
            h_values = torch.einsum("ab,nb->na", h_matrix, section_values)
            denominator = torch.real(
                torch.einsum("na,na->n", torch.conj(section_values), h_values)
            )
            if bool(torch.any(denominator <= 0)):
                raise FloatingPointError("H-metric denominator is not positive")
            h_derivatives = torch.einsum(
                "ab,nbj->naj", h_matrix, section_derivatives
            )
            first = torch.einsum(
                "nmi,nmj->nij", torch.conj(section_derivatives), h_derivatives
            )
            gradient = torch.einsum(
                "nm,nmj->nj", torch.conj(section_values), h_derivatives
            )
            metric = first / denominator[:, None, None]
            metric -= (
                torch.conj(gradient)[:, :, None]
                * gradient[:, None, :]
                / denominator[:, None, None] ** 2
            )
            metric *= float(artifact.normalization)
            metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
            eigenvalues = torch.linalg.eigvalsh(metric)
            if bool(torch.any(eigenvalues <= 0)):
                raise FloatingPointError("H-metric is not positive definite")
            raw = torch.sum(torch.log(eigenvalues), dim=1) - omega
            output[start:stop] = raw.detach().cpu().numpy().astype(np.float64)
    return output


def evaluate_points(
    adapter: Any,
    torch: Any,
    points: Sequence[Any],
    exponents: np.ndarray,
    artifacts: Sequence[Any],
    *,
    device: Any,
    batch_size: int,
) -> list[np.ndarray]:
    values, derivatives, log_omega = prepare_features(adapter, points, exponents)
    residuals = [
        evaluate_raw_residuals(
            torch,
            values,
            derivatives,
            log_omega,
            artifact,
            device=device,
            batch_size=batch_size,
        )
        for artifact in artifacts
    ]
    return residuals


def normalized_ratio(
    raw: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    log_normalization = log_weighted_mean_exp(raw, weights)
    ratio = np.exp(np.clip(raw - log_normalization, -700.0, 700.0))
    return ratio, log_normalization


def group_tail_summary(
    ratio: np.ndarray,
    weights: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    failure = ratio > threshold
    normalized_weights = weights / np.sum(weights)
    return {
        "ratio_quantiles": quantiles(ratio),
        "maximum_ratio": float(np.max(ratio)),
        "maximum_ratio_point_index": int(np.argmax(ratio)),
        "ratio_above_threshold_point_count": int(np.sum(failure)),
        "ratio_above_threshold_cluster_count": int(
            len(np.unique(cluster_ids[failure]))
        ),
        "ratio_above_threshold_weighted_mass": float(
            np.sum(normalized_weights[failure])
        ),
        "sigma": float(np.sum(normalized_weights * np.abs(1.0 - ratio))),
        "ratio_l2": float(
            np.sqrt(np.sum(normalized_weights * (1.0 - ratio) ** 2))
        ),
    }


def sample_seed(
    adapter: Any,
    *,
    model_seed: int,
    exact_model: bool,
    count: int,
    seed: int,
    workers: int,
    cluster_size: int,
    backend: str,
) -> tuple[list[Any], list[dict[str, int]]]:
    if workers > 1:
        return sample_points_parallel(
            adapter,
            model_seed=model_seed,
            exact_model=exact_model,
            count=count,
            seed=seed,
            workers=workers,
            cluster_size=cluster_size,
            backend=backend,
        )
    model = adapter.make_model(model_seed, exact=exact_model)
    points, _ = adapter.sample_points_with_diagnostics(model, count, seed=seed)
    return points, []


def nearest_product_fs(
    torch: Any,
    queries: tuple[np.ndarray, ...],
    references: tuple[np.ndarray, ...],
    *,
    device: Any,
    batch_size: int,
) -> dict[str, np.ndarray]:
    reference_tensors = [
        torch.as_tensor(factor, dtype=torch.complex64, device=device)
        for factor in references
    ]
    physical_distance = np.empty(len(queries[0]), dtype=float)
    physical_index = np.empty(len(queries[0]), dtype=np.int64)
    extended_distance = np.empty(len(queries[0]), dtype=float)
    extended_index = np.empty(len(queries[0]), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(queries[0]), batch_size):
            stop = min(start + batch_size, len(queries[0]))
            squared = None
            for factor_index, reference in enumerate(reference_tensors):
                query = torch.as_tensor(
                    queries[factor_index][start:stop],
                    dtype=torch.complex64,
                    device=device,
                )
                overlap = torch.abs(torch.conj(query) @ torch.transpose(reference, 0, 1))
                contribution = torch.acos(torch.clamp(overlap, 0.0, 1.0)) ** 2
                squared = contribution if squared is None else squared + contribution
                if factor_index == 1:
                    values, indices = torch.min(squared, dim=1)
                    physical_distance[start:stop] = torch.sqrt(values).cpu().numpy()
                    physical_index[start:stop] = indices.cpu().numpy()
            assert squared is not None
            values, indices = torch.min(squared, dim=1)
            extended_distance[start:stop] = torch.sqrt(values).cpu().numpy()
            extended_index[start:stop] = indices.cpu().numpy()
    return {
        "physical_distance": physical_distance,
        "physical_index": physical_index,
        "extended_distance": extended_distance,
        "extended_index": extended_index,
    }


def percentile_against_controls(value: float, controls: np.ndarray) -> float:
    return float(np.mean(np.asarray(controls, dtype=float) <= value))


def main() -> None:
    args = parse_args()
    positive_integers = (
        args.points_per_seed,
        args.sampling_workers,
        args.sampling_cluster_size,
        args.evaluation_batch_size,
        args.nearest_query_batch_size,
        args.control_points,
        args.neighborhood_points,
    )
    if min(positive_integers) <= 0:
        raise SystemExit("all point, worker, cluster, and batch counts must be positive")
    if args.points_per_seed % args.sampling_cluster_size:
        raise SystemExit("points-per-seed must be divisible by the cluster size")
    if args.bad_ratio_threshold <= 1 or min(args.neighborhood_radii) <= 0:
        raise SystemExit("tail threshold and neighborhood radii must be positive")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for this diagnostic") from exc
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    initial = adapter.load_h_artifact(args.initial_artifact, model)
    final = adapter.load_h_artifact(args.final_artifact, model)
    if not np.array_equal(initial.section_exponents, final.section_exponents):
        raise SystemExit("initial and final artifacts use different section bases")
    exponents = np.asarray(final.section_exponents, dtype=np.int64)

    train_coordinate_groups: list[tuple[np.ndarray, ...]] = []
    train_seed_values: list[np.ndarray] = []
    train_local_values: list[np.ndarray] = []
    train_cluster_values: list[np.ndarray] = []
    train_importance_values: list[np.ndarray] = []
    train_initial_raw_values: list[np.ndarray] = []
    train_final_raw_values: list[np.ndarray] = []
    train_initial_ratio_values: list[np.ndarray] = []
    train_final_ratio_values: list[np.ndarray] = []
    train_rows = []
    sampling_rows: dict[str, Any] = {}

    for seed in args.train_seeds:
        print(f"sampling train seed {seed}", flush=True)
        points, shards = sample_seed(
            adapter,
            model_seed=args.model_seed,
            exact_model=args.exact_model,
            count=args.points_per_seed,
            seed=seed,
            workers=args.sampling_workers,
            cluster_size=args.sampling_cluster_size,
            backend=args.sampling_backend,
        )
        initial_raw, final_raw = evaluate_points(
            adapter,
            torch,
            points,
            exponents,
            (initial, final),
            device=device,
            batch_size=args.evaluation_batch_size,
        )
        weights = adapter.importance_weights(points)
        initial_ratio, initial_log_normalization = normalized_ratio(initial_raw, weights)
        final_ratio, final_log_normalization = normalized_ratio(final_raw, weights)
        clusters = np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
        initial_summary = group_tail_summary(
            initial_ratio,
            weights,
            clusters,
            threshold=args.bad_ratio_threshold,
        )
        final_summary = group_tail_summary(
            final_ratio,
            weights,
            clusters,
            threshold=args.bad_ratio_threshold,
        )
        train_rows.append(
            {
                "seed": int(seed),
                "points": len(points),
                "fibres": int(len(np.unique(clusters))),
                "initial_log_normalization": initial_log_normalization,
                "final_log_normalization": final_log_normalization,
                "initial": initial_summary,
                "final": final_summary,
                "tail_point_intersection_count": int(
                    np.sum(
                        (initial_ratio > args.bad_ratio_threshold)
                        & (final_ratio > args.bad_ratio_threshold)
                    )
                ),
            }
        )
        sampling_rows[f"train_seed{seed}"] = shards
        train_coordinate_groups.append(point_coordinate_arrays(points))
        train_seed_values.append(np.full(len(points), seed, dtype=np.int64))
        train_local_values.append(np.arange(len(points), dtype=np.int64))
        train_cluster_values.append(clusters)
        train_importance_values.append(np.asarray(weights, dtype=float))
        train_initial_raw_values.append(initial_raw)
        train_final_raw_values.append(final_raw)
        train_initial_ratio_values.append(initial_ratio)
        train_final_ratio_values.append(final_ratio)
        print(
            f"train seed {seed}: final max={final_summary['maximum_ratio']:.6g}, "
            f"r>{args.bad_ratio_threshold:g} points="
            f"{final_summary['ratio_above_threshold_point_count']}",
            flush=True,
        )

    train_coordinates = concatenate_coordinates(train_coordinate_groups)
    train_seeds = np.concatenate(train_seed_values)
    train_local_indices = np.concatenate(train_local_values)
    train_clusters = np.concatenate(train_cluster_values)
    train_importance = np.concatenate(train_importance_values)
    train_initial_raw = np.concatenate(train_initial_raw_values)
    train_final_raw = np.concatenate(train_final_raw_values)
    train_initial_ratio = np.concatenate(train_initial_ratio_values)
    train_final_ratio = np.concatenate(train_final_ratio_values)

    checkpoint_coordinate_groups: list[tuple[np.ndarray, ...]] = []
    bad_points: list[Any] = []
    bad_rows: list[dict[str, Any]] = []
    checkpoint_rows = []
    checkpoint_offset = 0
    for seed in args.checkpoint_seeds:
        print(f"sampling checkpoint seed {seed}", flush=True)
        points, shards = sample_seed(
            adapter,
            model_seed=args.model_seed,
            exact_model=args.exact_model,
            count=args.points_per_seed,
            seed=seed,
            workers=args.sampling_workers,
            cluster_size=args.sampling_cluster_size,
            backend=args.sampling_backend,
        )
        initial_raw, final_raw = evaluate_points(
            adapter,
            torch,
            points,
            exponents,
            (initial, final),
            device=device,
            batch_size=args.evaluation_batch_size,
        )
        weights = adapter.importance_weights(points)
        initial_ratio, initial_log_normalization = normalized_ratio(initial_raw, weights)
        final_ratio, final_log_normalization = normalized_ratio(final_raw, weights)
        clusters = np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
        initial_summary = group_tail_summary(
            initial_ratio,
            weights,
            clusters,
            threshold=args.bad_ratio_threshold,
        )
        final_summary = group_tail_summary(
            final_ratio,
            weights,
            clusters,
            threshold=args.bad_ratio_threshold,
        )
        failure_indices = np.flatnonzero(final_ratio > args.bad_ratio_threshold)
        for local_index in failure_indices:
            cluster = int(clusters[local_index])
            members = np.flatnonzero(clusters == cluster)
            normalized_group_weights = weights / np.mean(weights)
            bad_points.append(points[int(local_index)])
            bad_rows.append(
                {
                    "bad_point_id": len(bad_rows),
                    "checkpoint_seed": int(seed),
                    "checkpoint_local_index": int(local_index),
                    "checkpoint_global_index": int(checkpoint_offset + local_index),
                    "checkpoint_cluster_id": cluster,
                    "final_raw_log_ratio": float(final_raw[local_index]),
                    "final_log_normalization": final_log_normalization,
                    "final_normalized_ratio": float(final_ratio[local_index]),
                    "initial_raw_log_ratio": float(initial_raw[local_index]),
                    "initial_log_normalization": initial_log_normalization,
                    "initial_normalized_ratio": float(initial_ratio[local_index]),
                    "created_or_amplified_by_training_factor": float(
                        final_ratio[local_index]
                        / max(initial_ratio[local_index], np.finfo(float).tiny)
                    ),
                    "normalized_proposal_importance_weight": float(
                        normalized_group_weights[local_index]
                    ),
                    "proposal_importance_weight_percentile": percentile_against_controls(
                        float(normalized_group_weights[local_index]),
                        normalized_group_weights,
                    ),
                    "same_fibre_local_indices": [int(value) for value in members],
                    "same_fibre_final_ratios": [
                        float(final_ratio[value]) for value in members
                    ],
                    "same_fibre_initial_ratios": [
                        float(initial_ratio[value]) for value in members
                    ],
                    **point_diagnostic(points[int(local_index)]),
                }
            )
        checkpoint_rows.append(
            {
                "seed": int(seed),
                "points": len(points),
                "fibres": int(len(np.unique(clusters))),
                "initial_log_normalization": initial_log_normalization,
                "final_log_normalization": final_log_normalization,
                "initial": initial_summary,
                "final": final_summary,
                "tail_point_intersection_count": int(
                    np.sum(
                        (initial_ratio > args.bad_ratio_threshold)
                        & (final_ratio > args.bad_ratio_threshold)
                    )
                ),
            }
        )
        sampling_rows[f"checkpoint_seed{seed}"] = shards
        checkpoint_coordinate_groups.append(point_coordinate_arrays(points))
        checkpoint_offset += len(points)
        print(
            f"checkpoint seed {seed}: final max={final_summary['maximum_ratio']:.6g}, "
            f"r>{args.bad_ratio_threshold:g} points={len(failure_indices)}",
            flush=True,
        )

    if not bad_points:
        raise SystemExit("no checkpoint bad points were found at the requested threshold")
    checkpoint_coordinates = concatenate_coordinates(checkpoint_coordinate_groups)
    bad_coordinates = point_coordinate_arrays(bad_points)

    print(
        f"finding exact train neighbours for {len(bad_points)} bad points", flush=True
    )
    bad_nearest = nearest_product_fs(
        torch,
        bad_coordinates,
        train_coordinates,
        device=device,
        batch_size=args.nearest_query_batch_size,
    )
    rng = np.random.default_rng(args.control_seed)
    bad_global_indices = {row["checkpoint_global_index"] for row in bad_rows}
    eligible = np.asarray(
        [
            index
            for index in range(len(checkpoint_coordinates[0]))
            if index not in bad_global_indices
        ],
        dtype=np.int64,
    )
    control_count = min(args.control_points, len(eligible))
    control_indices = rng.choice(eligible, size=control_count, replace=False)
    control_coordinates = tuple(factor[control_indices] for factor in checkpoint_coordinates)
    print(f"finding exact train neighbours for {control_count} controls", flush=True)
    control_nearest = nearest_product_fs(
        torch,
        control_coordinates,
        train_coordinates,
        device=device,
        batch_size=args.nearest_query_batch_size,
    )

    nearest_points: list[Any] = []
    for bad_index, row in enumerate(bad_rows):
        nearest_index = int(bad_nearest["physical_index"][bad_index])
        nearest_payload = {
            "coordinates_x": train_coordinates[0][nearest_index : nearest_index + 1],
            "coordinates_y": train_coordinates[1][nearest_index : nearest_index + 1],
            "coordinates_z": train_coordinates[2][nearest_index : nearest_index + 1],
        }
        nearest_point = adapter.points_from_storage_payload(model, nearest_payload)[0]
        nearest_points.append(nearest_point)
        source_log_normalization = float(row["final_log_normalization"])
        nearest_ratio_in_source_normalization = float(
            np.exp(
                np.clip(
                    train_final_raw[nearest_index] - source_log_normalization,
                    -700.0,
                    700.0,
                )
            )
        )
        distance = float(bad_nearest["physical_distance"][bad_index])
        row["nearest_training_point"] = {
            "physical_p4xp1_fubini_study_distance": distance,
            "physical_distance_control_percentile": percentile_against_controls(
                distance, control_nearest["physical_distance"]
            ),
            "extended_p4xp1xp1_fubini_study_distance": float(
                bad_nearest["extended_distance"][bad_index]
            ),
            "training_global_index": nearest_index,
            "training_seed": int(train_seeds[nearest_index]),
            "training_local_index": int(train_local_indices[nearest_index]),
            "training_cluster_id": int(train_clusters[nearest_index]),
            "final_ratio_in_own_training_group": float(
                train_final_ratio[nearest_index]
            ),
            "initial_ratio_in_own_training_group": float(
                train_initial_ratio[nearest_index]
            ),
            "final_ratio_using_bad_checkpoint_normalization": (
                nearest_ratio_in_source_normalization
            ),
            "final_raw_log_ratio": float(train_final_raw[nearest_index]),
            "initial_raw_log_ratio": float(train_initial_raw[nearest_index]),
            "normalized_proposal_importance_weight": float(
                train_importance[nearest_index]
            ),
            "secant_log_residual_slope": float(
                abs(row["final_raw_log_ratio"] - train_final_raw[nearest_index])
                / max(distance, np.finfo(float).tiny)
            ),
            **point_diagnostic(nearest_point),
        }

    print("probing local shells around checkpoint spikes", flush=True)
    for bad_index, (point, row) in enumerate(zip(bad_points, bad_rows, strict=True)):
        local_rows = []
        for radius_index, radius in enumerate(args.neighborhood_radii):
            local_seed = 880000 + 1000 * bad_index + radius_index
            try:
                local_points, diagnostics = adapter.sample_local_neighborhood(
                    model,
                    point,
                    args.neighborhood_points,
                    seed=local_seed,
                    radius=radius,
                )
                (local_raw,) = evaluate_points(
                    adapter,
                    torch,
                    local_points,
                    exponents,
                    (final,),
                    device=device,
                    batch_size=args.evaluation_batch_size,
                )
                local_log_ratio = local_raw - float(row["final_log_normalization"])
                local_ratio = np.exp(np.clip(local_log_ratio, -700.0, 700.0))
                local_rows.append(
                    {
                        "requested_intrinsic_radius": float(radius),
                        "sampling_seed": local_seed,
                        "ratio_quantiles": quantiles(local_ratio),
                        "log_ratio_quantiles": quantiles(local_log_ratio),
                        "ratio_above_threshold_fraction": float(
                            np.mean(local_ratio > args.bad_ratio_threshold)
                        ),
                        "ratio_above_half_center_fraction": float(
                            np.mean(
                                local_ratio
                                > 0.5 * float(row["final_normalized_ratio"])
                            )
                        ),
                        "sampling_diagnostics": diagnostics,
                    }
                )
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                local_rows.append(
                    {
                        "requested_intrinsic_radius": float(radius),
                        "sampling_seed": local_seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        row["local_shells"] = local_rows
        if args.gradients:
            try:
                row["empirical_tangent_gradient"] = (
                    estimate_h_metric_residual_gradient(
                        adapter,
                        model,
                        point,
                        final,
                        coarse_step=args.gradient_step,
                    ).to_dict()
                )
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                row["empirical_tangent_gradient"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        print(
            f"probed bad point {bad_index + 1}/{len(bad_points)}: "
            f"ratio={row['final_normalized_ratio']:.6g}",
            flush=True,
        )

    train_failure = train_final_ratio > args.bad_ratio_threshold
    initial_train_failure = train_initial_ratio > args.bad_ratio_threshold
    report = {
        "schema_version": 1,
        "description": (
            "Exact-seed diagnostic separating training coverage, loss tolerance, "
            "and optimization-created k=4 H-metric spikes."
        ),
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "initial_artifact": str(args.initial_artifact.expanduser().resolve()),
        "final_artifact": str(args.final_artifact.expanduser().resolve()),
        "section_count": int(len(exponents)),
        "bad_ratio_threshold": args.bad_ratio_threshold,
        "distance_conventions": {
            "physical": "product Fubini-Study distance on direct P4 x P1",
            "extended": (
                "product Fubini-Study distance on stabilized P4 x P1 x P1; "
                "reported separately because the third factor is auxiliary"
            ),
        },
        "training": {
            "seeds": [int(seed) for seed in args.train_seeds],
            "points_per_seed": args.points_per_seed,
            "total_points": int(len(train_final_ratio)),
            "total_fibres": int(
                len(args.train_seeds)
                * args.points_per_seed
                / args.sampling_cluster_size
            ),
            "per_seed": train_rows,
            "aggregate_final": {
                "ratio_quantiles": quantiles(train_final_ratio),
                "maximum_ratio": float(np.max(train_final_ratio)),
                "ratio_above_threshold_point_count": int(np.sum(train_failure)),
                "ratio_above_threshold_fibre_count": int(
                    len(
                        {
                            (int(seed), int(cluster))
                            for seed, cluster in zip(
                                train_seeds[train_failure],
                                train_clusters[train_failure],
                                strict=True,
                            )
                        }
                    )
                ),
            },
            "aggregate_initial": {
                "ratio_quantiles": quantiles(train_initial_ratio),
                "maximum_ratio": float(np.max(train_initial_ratio)),
                "ratio_above_threshold_point_count": int(
                    np.sum(initial_train_failure)
                ),
            },
            "tail_point_intersection_count": int(
                np.sum(train_failure & initial_train_failure)
            ),
        },
        "checkpoint": {
            "seeds": [int(seed) for seed in args.checkpoint_seeds],
            "points_per_seed": args.points_per_seed,
            "total_points": int(len(checkpoint_coordinates[0])),
            "per_seed": checkpoint_rows,
            "bad_point_count": len(bad_points),
        },
        "nearest_training_distance_controls": {
            "control_seed": args.control_seed,
            "control_count": control_count,
            "physical_distance_quantiles": quantiles(
                control_nearest["physical_distance"]
            ),
            "extended_distance_quantiles": quantiles(
                control_nearest["extended_distance"]
            ),
        },
        "bad_points": bad_rows,
        "sampling_shards": sampling_rows,
        "interpretation_limits": [
            "Nearest-neighbour distances diagnose this finite training design; they are not an epsilon-net certificate.",
            "Local-shell and tangent-gradient measurements are empirical local regularity diagnostics, not global Lipschitz bounds.",
            "Ratios are normalized independently within each original seed group to reproduce the training/checkpoint convention.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    payload_path = args.point_payload or args.out.with_suffix(".npz")
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    nearest_coordinates = point_coordinate_arrays(nearest_points)
    np.savez_compressed(
        payload_path,
        bad_coordinates_x=bad_coordinates[0],
        bad_coordinates_y=bad_coordinates[1],
        bad_coordinates_z=bad_coordinates[2],
        nearest_training_coordinates_x=nearest_coordinates[0],
        nearest_training_coordinates_y=nearest_coordinates[1],
        nearest_training_coordinates_z=nearest_coordinates[2],
        bad_checkpoint_seed=np.asarray(
            [row["checkpoint_seed"] for row in bad_rows], dtype=np.int64
        ),
        bad_checkpoint_local_index=np.asarray(
            [row["checkpoint_local_index"] for row in bad_rows], dtype=np.int64
        ),
        nearest_training_global_index=bad_nearest["physical_index"],
        control_checkpoint_global_index=control_indices,
        control_nearest_physical_distance=control_nearest["physical_distance"],
        control_nearest_extended_distance=control_nearest["extended_distance"],
    )
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {payload_path}", flush=True)


if __name__ == "__main__":
    main()
