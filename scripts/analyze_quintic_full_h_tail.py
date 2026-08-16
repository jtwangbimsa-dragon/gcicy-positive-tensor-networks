#!/usr/bin/env python3
"""Diagnose conditioning and clustering of a quintic full-H blind-test tail."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--tail-threshold", type=float, default=1.5)
    parser.add_argument("--control-count", type=int, default=64)
    parser.add_argument("--control-seed", type=int, default=202607162)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--symmetrized-h-output", type=Path)
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.05, 0.5, 0.95, 1.0)
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {}
    return {
        f"q{probability:.2f}": float(value)
        for probability, value in zip(probabilities, np.quantile(values, probabilities))
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    left = np.asarray(left[finite], dtype=np.float64)
    right = np.asarray(right[finite], dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def projective_nearest_distances(
    normalized_points: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    distances = np.empty(len(indices), dtype=np.float64)
    for output_index, point_index in enumerate(indices):
        overlaps = np.abs(normalized_points @ np.conj(normalized_points[point_index]))
        overlaps[point_index] = -np.inf
        maximum_overlap = float(np.clip(np.max(overlaps), 0.0, 1.0))
        distances[output_index] = math.sqrt(max(0.0, 1.0 - maximum_overlap**2))
    return distances


def within_group_nearest_distances(
    normalized_points: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    if len(indices) < 2:
        return np.empty(0, dtype=np.float64)
    selected = normalized_points[indices]
    overlaps = np.abs(selected @ np.conj(selected.T))
    np.fill_diagonal(overlaps, -np.inf)
    maximum_overlaps = np.clip(np.max(overlaps, axis=1), 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - maximum_overlaps**2))


def coordinate_profile(normalized_points: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    selected = np.abs(normalized_points[indices]) ** 2
    ordered = np.sort(selected, axis=1)
    top_pairs = np.sort(np.argpartition(selected, -2, axis=1)[:, -2:], axis=1)
    pair_counts: dict[str, int] = {}
    for pair in top_pairs:
        key = f"{int(pair[0])},{int(pair[1])}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    return {
        "smallest_coordinate_fraction": quantiles(ordered[:, 0]),
        "second_largest_coordinate_fraction": quantiles(ordered[:, -2]),
        "largest_coordinate_fraction": quantiles(ordered[:, -1]),
        "outside_two_largest_fraction": quantiles(
            1.0 - ordered[:, -2] - ordered[:, -1]
        ),
        "largest_minus_second_largest_fraction": quantiles(
            ordered[:, -1] - ordered[:, -2]
        ),
        "dominant_coordinate_pair_counts": pair_counts,
    }


def permutation_average(
    matrix: np.ndarray,
    exponents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {tuple(exponent): index for index, exponent in enumerate(exponents)}
    accumulated = np.zeros_like(matrix, dtype=np.complex128)
    defects = []
    norm = np.linalg.norm(matrix)
    for permutation in itertools.permutations(range(exponents.shape[1])):
        permuted = exponents[:, permutation]
        indices = np.asarray([lookup[tuple(row)] for row in permuted], dtype=np.int64)
        transformed = matrix[np.ix_(indices, indices)]
        accumulated += transformed
        defects.append(np.linalg.norm(matrix - transformed) / norm)
    average = accumulated / math.factorial(exponents.shape[1])
    average = 0.5 * (average + average.conj().T)
    average *= np.trace(matrix).real / np.trace(average).real
    return average, np.asarray(defects, dtype=np.float64)


def dominant_pair_strata(
    normalized_points: np.ndarray,
    ratios: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    fractions = np.abs(normalized_points) ** 2
    pairs = np.sort(np.argpartition(fractions, -2, axis=1)[:, -2:], axis=1)
    ordered = np.sort(fractions, axis=1)
    outside = 1.0 - ordered[:, -2] - ordered[:, -1]
    result: dict[str, Any] = {}
    for cutoff in (0.02, 0.05, 0.10):
        rows: dict[str, Any] = {}
        for left in range(fractions.shape[1]):
            for right in range(left + 1, fractions.shape[1]):
                mask = (
                    (outside <= cutoff)
                    & (pairs[:, 0] == left)
                    & (pairs[:, 1] == right)
                )
                if not np.any(mask):
                    continue
                selected_weights = weights[mask]
                selected_ratios = ratios[mask]
                rows[f"{left},{right}"] = {
                    "count": int(np.count_nonzero(mask)),
                    "weighted_sigma": float(
                        np.sum(selected_weights * np.abs(1.0 - selected_ratios))
                        / np.sum(selected_weights)
                    ),
                    "maximum_ratio": float(np.max(selected_ratios)),
                    "minimum_ratio": float(np.min(selected_ratios)),
                    "count_above_1.5": int(np.count_nonzero(selected_ratios > 1.5)),
                }
        result[f"outside_two_largest_at_most_{cutoff:.2f}"] = rows
    return result


def matrix_spectrum(matrix: np.ndarray) -> dict[str, Any]:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    maximum = float(eigenvalues[-1])
    minimum = float(eigenvalues[0])
    relative = eigenvalues / maximum
    return {
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "condition_number": float(maximum / minimum),
        "log_condition_number": float(math.log(maximum / minimum)),
        "eigenvalue_quantiles": quantiles(eigenvalues),
        "count_relative_below_1e-6": int(np.count_nonzero(relative < 1.0e-6)),
        "count_relative_below_1e-8": int(np.count_nonzero(relative < 1.0e-8)),
        "relative_hermiticity_error": float(
            np.linalg.norm(matrix - matrix.conj().T) / np.linalg.norm(matrix)
        ),
    }


def main() -> None:
    args = parse_args()
    model_dir = args.model_run_dir.resolve()
    source_dir = args.source_run_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else model_dir / "tail_diagnostics.json"
    )
    symmetrized_h_path = (
        args.symmetrized_h_output.resolve()
        if args.symmetrized_h_output is not None
        else model_dir / "symmetrized_h_metric.npz"
    )

    h_path = model_dir / "best_h_metric.npz"
    tail_path = model_dir / "blind_test_tail_arrays.npz"
    points_path = source_dir / "blind_points.npz"
    report_path = model_dir / "report.json"
    for path in (h_path, tail_path, points_path, report_path):
        if not path.exists():
            raise FileNotFoundError(path)

    h_artifact = np.load(h_path, allow_pickle=False)
    global_h = np.asarray(h_artifact["global_h_matrix"], dtype=np.complex128)
    initial_h = np.asarray(h_artifact["initial_h_matrix"], dtype=np.complex128)
    exponents = np.asarray(h_artifact["exponents"], dtype=np.int64)
    symmetrized_h, trained_symmetry_defects = permutation_average(global_h, exponents)
    _, initial_symmetry_defects = permutation_average(initial_h, exponents)
    np.savez_compressed(
        symmetrized_h_path,
        degree=np.asarray(h_artifact["degree"]),
        exponents=exponents,
        source_global_h_matrix=global_h,
        symmetrized_global_h_matrix=symmetrized_h,
        source_h_metric_sha256=np.asarray(sha256_file(h_path)),
    )
    arrays = np.load(tail_path, allow_pickle=False)
    ratios = np.asarray(arrays["normalized_ratio"], dtype=np.float64)
    weights = np.asarray(arrays["weights"], dtype=np.float64)
    minimum_eigenvalues = np.asarray(arrays["min_eigenvalue"], dtype=np.float64)
    phi_ratios = np.asarray(
        arrays["cymetric_phi_normalized_ratio"], dtype=np.float64
    )
    points_artifact = np.load(points_path, allow_pickle=False)
    if "complex_points" in points_artifact.files:
        complex_points = np.asarray(points_artifact["complex_points"])
    else:
        x_values = np.asarray(points_artifact["X"], dtype=np.float64)
        n_coordinates = x_values.shape[1] // 2
        complex_points = x_values[:, :n_coordinates] + 1j * x_values[:, n_coordinates:]
    if len(complex_points) != len(ratios):
        raise RuntimeError("blind point and ratio counts differ")
    if not np.array_equal(weights, np.asarray(points_artifact["weights"])):
        raise RuntimeError("blind point weights differ from the tail artifact")

    normalized_points = complex_points / np.linalg.norm(
        complex_points, axis=1, keepdims=True
    )
    tail_indices = np.flatnonzero(ratios > args.tail_threshold)
    worst_indices = np.argsort(ratios)[-64:][::-1]
    rng = np.random.default_rng(args.control_seed)
    eligible = np.setdiff1d(
        np.arange(len(ratios), dtype=np.int64), tail_indices, assume_unique=True
    )
    control_count = min(args.control_count, len(eligible))
    control_indices = rng.choice(eligible, size=control_count, replace=False)

    tail_nearest_all = projective_nearest_distances(normalized_points, tail_indices)
    control_nearest_all = projective_nearest_distances(normalized_points, control_indices)
    tail_nearest_tail = within_group_nearest_distances(normalized_points, tail_indices)
    worst_nearest_worst = within_group_nearest_distances(
        normalized_points, worst_indices
    )

    total_weight = float(np.sum(weights))
    thresholds = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0)
    worst = int(np.argmax(ratios))
    residual = np.abs(1.0 - ratios)
    report = {
        "schema": "quintic-full-h-tail-diagnostics-v1",
        "model_run_dir": model_dir,
        "source_run_dir": source_dir,
        "tail_threshold": args.tail_threshold,
        "h_spectrum": {
            "initial_fubini_study": matrix_spectrum(initial_h),
            "trained_full_h": matrix_spectrum(global_h),
            "permutation_averaged_full_h": matrix_spectrum(symmetrized_h),
        },
        "s5_permutation_symmetry": {
            "group_order": math.factorial(exponents.shape[1]),
            "initial_h_relative_defect_quantiles": quantiles(initial_symmetry_defects),
            "trained_h_relative_defect_quantiles": quantiles(
                trained_symmetry_defects
            ),
            "symmetrized_h_path": symmetrized_h_path,
        },
        "tail": {
            "count": int(len(tail_indices)),
            "point_fraction": float(len(tail_indices) / len(ratios)),
            "weighted_mass": float(np.sum(weights[tail_indices]) / total_weight),
            "indices": tail_indices,
            "ratios": ratios[tail_indices],
            "cymetric_phi_ratios_at_same_points": phi_ratios[tail_indices],
            "metric_minimum_eigenvalues": minimum_eigenvalues[tail_indices],
            "ambient_projective_chordal_nearest_all": quantiles(tail_nearest_all),
            "ambient_projective_chordal_nearest_tail": quantiles(tail_nearest_tail),
            "coordinate_profile": coordinate_profile(normalized_points, tail_indices),
        },
        "random_control": {
            "count": int(control_count),
            "ambient_projective_chordal_nearest_all": quantiles(control_nearest_all),
            "coordinate_profile": coordinate_profile(normalized_points, control_indices),
        },
        "worst_64": {
            "ambient_projective_chordal_nearest_within_worst_64": quantiles(
                worst_nearest_worst
            )
        },
        "worst_point": {
            "index": worst,
            "ratio": float(ratios[worst]),
            "cymetric_phi_ratio": float(phi_ratios[worst]),
            "weight": float(weights[worst]),
            "metric_minimum_eigenvalue": float(minimum_eigenvalues[worst]),
            "homogeneous_coordinates": complex_points[worst],
        },
        "thresholds": {
            str(threshold): {
                "count": int(np.count_nonzero(ratios > threshold)),
                "weighted_mass": float(np.sum(weights[ratios > threshold]) / total_weight),
            }
            for threshold in thresholds
        },
        "dominant_coordinate_pair_strata": dominant_pair_strata(
            normalized_points, ratios, weights
        ),
        "linear_correlations": {
            "abs_residual_vs_log_weight": correlation(residual, np.log(weights)),
            "abs_residual_vs_metric_minimum_eigenvalue": correlation(
                residual, minimum_eigenvalues
            ),
            "abs_residual_vs_cymetric_phi_abs_residual": correlation(
                residual, np.abs(1.0 - phi_ratios)
            ),
        },
        "artifacts_sha256": {
            "h_metric": sha256_file(h_path),
            "tail_arrays": sha256_file(tail_path),
            "blind_points": sha256_file(points_path),
            "model_report": sha256_file(report_path),
            "analysis_script": sha256_file(Path(__file__).resolve()),
            "symmetrized_h_metric": sha256_file(symmetrized_h_path),
        },
    }
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "trained_h_condition_number": report["h_spectrum"]["trained_full_h"][
                    "condition_number"
                ],
                "symmetrized_h_condition_number": report["h_spectrum"][
                    "permutation_averaged_full_h"
                ]["condition_number"],
                "trained_h_s5_defect": report["s5_permutation_symmetry"][
                    "trained_h_relative_defect_quantiles"
                ],
                "tail": report["tail"],
                "worst_point": report["worst_point"],
            },
            default=json_value,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
