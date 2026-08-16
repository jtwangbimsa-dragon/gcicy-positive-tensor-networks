#!/usr/bin/env python3
"""Scan a one-parameter, reference-regularized H-metric path."""

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
    reference_geodesic_interpolate,
    reference_log_spectrum_clip,
    reference_relative_spectrum,
    standard_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--exact-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--basis-seed", type=int, required=True)
    parser.add_argument("--basis-points", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--points-per-seed", type=int, required=True)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument(
        "--log-spectrum-clips",
        type=float,
        nargs="+",
        help=(
            "Scan caps on centered generalized log eigenvalues instead of the "
            "geodesic alpha path."
        ),
    )
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def normalized_ratio_tail(log_eta: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    values = np.asarray(log_eta, dtype=float)
    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights /= np.sum(normalized_weights)
    shifted = np.exp(values - float(np.max(values)))
    ratio = shifted / float(np.sum(normalized_weights * shifted))
    contributions = normalized_weights * (1.0 - ratio) ** 2
    energy = float(np.sum(contributions))
    order = np.argsort(contributions)[::-1]
    cumulative = np.cumsum(contributions[order]) / energy

    def count_for_fraction(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    return {
        "maximum_normalized_ratio": float(np.max(ratio)),
        "maximum_ratio_point_index": int(np.argmax(ratio)),
        "maximum_energy_contribution_point_index": int(order[0]),
        "energy_concentration": {
            str(count): float(np.sum(contributions[order[:count]]) / energy)
            for count in (1, 4, 16, 64, 256)
            if count <= len(order)
        },
        "points_for_energy_fraction": {
            str(fraction): count_for_fraction(fraction) for fraction in (0.5, 0.9, 0.99)
        },
    }


def aligned_reference_h(
    adapter: Any, model: Any, artifact_path: Path, basis_seed: int, basis_points: int
) -> tuple[np.ndarray, float, dict[str, Any]]:
    points, sampling = adapter.sample_points_with_diagnostics(
        model,
        basis_points,
        seed=basis_seed,
    )
    artifact = adapter.load_h_artifact(artifact_path, model)
    generated = adapter.restricted_section_basis(points, artifact.degree)
    with np.load(artifact_path, allow_pickle=False) as payload:
        if "basis_selected_indices" not in payload.files:
            raise ValueError("artifact does not record basis_selected_indices")
        selected_indices = np.asarray(payload["basis_selected_indices"], dtype=np.int64)
    if selected_indices.shape != (artifact.section_count,):
        raise ValueError("artifact basis index count is inconsistent")
    selected_exponents = np.asarray(generated.ambient_exponents)[selected_indices]
    if not np.array_equal(selected_exponents, artifact.section_exponents):
        raise ValueError(
            "artifact exponents do not match the reconstructed ambient basis"
        )
    aligned = replace(
        generated,
        selected_indices=selected_indices,
        selected_exponents=artifact.section_exponents,
        numerical_rank=artifact.section_count,
    )
    reference_h, relation_error = adapter.restricted_fubini_study_h_matrix(
        points,
        aligned,
    )
    return np.asarray(reference_h, dtype=np.complex128), float(relation_error), sampling


def save_candidate(
    source: Path,
    destination: Path,
    h_matrix: np.ndarray,
    *,
    path_kind: str,
    parameter: float,
) -> None:
    with np.load(source, allow_pickle=False) as payload:
        output = {key: np.asarray(payload[key]).copy() for key in payload.files}
    output["global_h_matrix"] = np.asarray(h_matrix, dtype=np.complex128)
    output["reference_regularization_path"] = np.asarray(path_kind)
    output["reference_regularization_parameter"] = np.asarray(float(parameter))
    output["reference_regularization_source"] = np.asarray(source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output)


def main() -> None:
    args = parse_args()
    if args.basis_points <= 0 or args.points_per_seed <= 0:
        raise SystemExit("basis-points and points-per-seed must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("evaluation seeds must be distinct")
    if args.log_spectrum_clips is None:
        path_kind = "spd_geodesic"
        parameters = sorted(set(float(value) for value in args.alphas))
        if not parameters or min(parameters) < 0 or max(parameters) > 1:
            raise SystemExit("alphas must be a non-empty subset of [0, 1]")
    else:
        path_kind = "centered_generalized_log_spectrum_clip"
        parameters = sorted(set(float(value) for value in args.log_spectrum_clips))
        if not parameters or min(parameters) < 0:
            raise SystemExit("log-spectrum-clips must be non-empty and non-negative")

    started = time.perf_counter()
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    source_path = args.artifact.expanduser().resolve()
    source_artifact = adapter.load_h_artifact(source_path, model)
    reference_h, relation_error, basis_sampling = aligned_reference_h(
        adapter,
        model,
        source_path,
        args.basis_seed,
        args.basis_points,
    )
    relative_spectrum = reference_relative_spectrum(
        reference_h,
        source_artifact.h_matrix,
    )

    candidates: list[tuple[float, HMetricArtifact]] = []
    candidate_paths: dict[float, str | None] = {}
    for parameter in parameters:
        if path_kind == "spd_geodesic":
            h_matrix = reference_geodesic_interpolate(
                reference_h,
                source_artifact.h_matrix,
                parameter,
            )
            token_prefix = "a"
            metadata_key = "reference_geodesic_alpha"
        else:
            h_matrix = reference_log_spectrum_clip(
                reference_h,
                source_artifact.h_matrix,
                parameter,
            )
            token_prefix = "c"
            metadata_key = "reference_log_spectrum_clip"
        if args.candidate_dir is None:
            path = Path(f"<reference-regularization-{parameter:.6f}>")
            candidate_paths[parameter] = None
        else:
            token = f"{parameter:.6f}".rstrip("0").rstrip(".").replace(".", "p")
            path = args.candidate_dir.expanduser().resolve() / (
                f"{source_path.stem}_reference_regularized_{token_prefix}{token}.npz"
            )
            save_candidate(
                source_path,
                path,
                h_matrix,
                path_kind=path_kind,
                parameter=parameter,
            )
            candidate_paths[parameter] = str(path)
        candidates.append(
            (
                parameter,
                HMetricArtifact(
                    path=path,
                    degree=source_artifact.degree,
                    section_exponents=source_artifact.section_exponents,
                    h_matrix=h_matrix,
                    normalization=source_artifact.normalization,
                    metadata={
                        **source_artifact.metadata,
                        metadata_key: parameter,
                    },
                ),
            )
        )

    rows: dict[float, list[dict[str, Any]]] = {
        parameter: [] for parameter in parameters
    }
    sampling_diagnostics: dict[str, Any] = {}
    for seed in args.seeds:
        points, sampling = adapter.sample_points_with_diagnostics(
            model,
            args.points_per_seed,
            seed=seed,
        )
        sampling_diagnostics[str(seed)] = sampling
        weights = adapter.importance_weights(points)
        for parameter, artifact in candidates:
            metrics = adapter.h_metrics(points, artifact)
            residuals = adapter.residual_values(points, metrics)
            stats = standard_errors(residuals, weights)
            rows[parameter].append(
                {
                    "seed": int(seed),
                    **stats,
                    "min_metric_eigenvalue": float(np.min(np.linalg.eigvalsh(metrics))),
                    "tail": normalized_ratio_tail(residuals, weights),
                }
            )
        print(f"evaluated seed {seed}", flush=True)

    candidate_rows = []
    for parameter, artifact in candidates:
        seed_rows = rows[parameter]
        candidate_relative_spectrum = reference_relative_spectrum(
            reference_h,
            artifact.h_matrix,
        )
        parameter_fields = (
            {"alpha": parameter}
            if path_kind == "spd_geodesic"
            else {"max_abs_generalized_log_eigenvalue": parameter}
        )
        candidate_rows.append(
            {
                **parameter_fields,
                "artifact": candidate_paths[parameter],
                "raw_h_condition_number": float(
                    np.linalg.cond(np.asarray(artifact.h_matrix, dtype=np.complex128))
                ),
                "reference_relative_spectrum": {
                    "minimum": float(candidate_relative_spectrum[0]),
                    "maximum": float(candidate_relative_spectrum[-1]),
                    "condition_number": float(
                        candidate_relative_spectrum[-1]
                        / candidate_relative_spectrum[0]
                    ),
                    "maximum_absolute_log_eigenvalue": float(
                        np.max(np.abs(np.log(candidate_relative_spectrum)))
                    ),
                },
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
            "One-parameter, basis-covariant regularization scan relative to "
            "the reference Fubini--Study H matrix."
        ),
        "regularization_path": path_kind,
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "source_artifact": str(source_path),
        "basis_seed": args.basis_seed,
        "basis_points": args.basis_points,
        "basis_relation_error": relation_error,
        "basis_sampling_diagnostics": basis_sampling,
        "evaluation_seeds": args.seeds,
        "points_per_seed": args.points_per_seed,
        "reference_relative_spectrum": {
            "minimum": float(relative_spectrum[0]),
            "maximum": float(relative_spectrum[-1]),
            "condition_number": float(relative_spectrum[-1] / relative_spectrum[0]),
            "log_rms": float(np.sqrt(np.mean(np.log(relative_spectrum) ** 2))),
            "maximum_absolute_log_eigenvalue": float(
                np.max(np.abs(np.log(relative_spectrum)))
            ),
        },
        "sampling_diagnostics": sampling_diagnostics,
        "candidates": candidate_rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
