#!/usr/bin/env python3
"""Audit one or more H artifacts on an existing cached type-(1,1) point set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--phi-arrays", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Named artifact in NAME=PATH form; repeat for multiple artifacts.",
    )
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrays-out", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_artifacts(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--artifact values must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in names:
            raise ValueError("artifact names must be non-empty and distinct")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        names.add(name)
        output.append((name, path))
    return output


def weighted_log_mean_exp(values: np.ndarray, weights: np.ndarray) -> float:
    maximum = float(np.max(values))
    return float(
        maximum
        + np.log(np.sum(weights * np.exp(values - maximum)) / np.sum(weights))
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= np.sum(sorted_weights)
    return float(
        np.interp(
            probability,
            cumulative,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
    )


def tail_row(
    raw: np.ndarray,
    minimum: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    valid = np.isfinite(raw) & np.isfinite(minimum) & (minimum > 0)
    normalization = weighted_log_mean_exp(raw[valid], weights[valid])
    ratio = np.full(len(raw), np.nan, dtype=np.float64)
    ratio[valid] = np.exp(np.clip(raw[valid] - normalization, -745.0, 709.0))
    valid_weights = weights[valid]
    residual = 1.0 - ratio[valid]

    def threshold(value: float) -> dict[str, Any]:
        mask = ratio > value
        return {
            "point_count": int(np.sum(mask)),
            "weighted_mass": float(np.sum(weights[mask]) / np.sum(weights)),
            "cluster_count": int(len(np.unique(clusters[mask]))) if np.any(mask) else 0,
        }

    return (
        {
            "sigma": float(np.sum(valid_weights * np.abs(residual)) / np.sum(valid_weights)),
            "chi_l2": float(
                np.sqrt(np.sum(valid_weights * residual**2) / np.sum(valid_weights))
            ),
            "normalization_log_kappa": normalization,
            "ratio_q99": weighted_quantile(ratio[valid], valid_weights, 0.99),
            "ratio_q999": weighted_quantile(ratio[valid], valid_weights, 0.999),
            "ratio_q9999": weighted_quantile(ratio[valid], valid_weights, 0.9999),
            "max_ratio": float(np.nanmax(ratio)),
            "ratio_above_3": threshold(3.0),
            "ratio_above_10": threshold(10.0),
            "ratio_above_100": threshold(100.0),
            "min_metric_eigenvalue": float(np.min(minimum)),
            "nonpositive_or_nonfinite_count": int(np.sum(~valid)),
        },
        ratio,
    )


def complex_pairs(values: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    named_paths = parse_artifacts(args.artifact)
    point_path = args.points.expanduser().resolve()
    phi_path = args.phi_arrays.expanduser().resolve()
    if not point_path.exists() or not phi_path.exists():
        raise SystemExit("point and Phi array files must exist")

    with np.load(point_path, allow_pickle=False) as payload:
        coordinates_x = np.asarray(payload["coordinates_x"], dtype=np.complex128)
        coordinates_y = np.asarray(payload["coordinates_y"], dtype=np.complex128)
        coordinates_z = np.asarray(payload["coordinates_z"], dtype=np.complex128)
        log_omega = np.asarray(payload["log_omega"], dtype=np.float64)
        weights = np.asarray(payload["importance_weights"], dtype=np.float64)
        clusters = np.asarray(payload["cluster_ids"], dtype=np.int64)
        point_metadata = json.loads(str(payload["metadata_json"]))
    with np.load(phi_path, allow_pickle=False) as payload:
        phi_ratio = np.asarray(payload["phi_normalized_ratio"], dtype=np.float64)
    count = len(weights)
    if any(len(array) != count for array in (coordinates_y, coordinates_z, log_omega, clusters, phi_ratio)):
        raise ValueError("cached point and Phi arrays have inconsistent lengths")

    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    model = adapter.make_model(args.model_seed, exact=True)
    output_arrays: dict[str, np.ndarray] = {
        "importance_weights": weights,
        "cluster_ids": clusters,
        "phi_normalized_ratio": phi_ratio,
    }
    rows = {}
    started = time.perf_counter()
    for name, path in named_paths:
        artifact = adapter.load_h_artifact(path, model)
        raw_rows = []
        minimum_rows = []
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            points = adapter.points_from_storage_payload(
                model,
                {
                    "coordinates_x": coordinates_x[start:stop],
                    "coordinates_y": coordinates_y[start:stop],
                    "coordinates_z": coordinates_z[start:stop],
                },
            )
            metrics = adapter.h_metrics(points, artifact)
            eigenvalues = np.linalg.eigvalsh(metrics)
            minimum = eigenvalues[:, 0]
            raw = np.full(stop - start, np.nan, dtype=np.float64)
            valid = minimum > 0
            raw[valid] = np.sum(np.log(eigenvalues[valid]), axis=1) - log_omega[start:stop][valid]
            raw_rows.append(raw)
            minimum_rows.append(minimum)
        raw = np.concatenate(raw_rows)
        minimum = np.concatenate(minimum_rows)
        statistics, ratio = tail_row(raw, minimum, weights, clusters)
        worst = int(np.nanargmax(ratio))
        worst_point = adapter.points_from_storage_payload(
            model,
            {
                "coordinates_x": coordinates_x[[worst]],
                "coordinates_y": coordinates_y[[worst]],
                "coordinates_z": coordinates_z[[worst]],
            },
        )[0]
        section, derivatives = adapter.section_values_and_jacobian(
            worst_point, artifact.section_exponents
        )
        h_matrix = np.asarray(artifact.h_matrix, dtype=np.complex128)
        h_values = h_matrix @ section
        denominator = float(np.real(np.vdot(section, h_values)))
        section_norm_squared = float(np.real(np.vdot(section, section)))
        h_eigenvalues, h_eigenvectors = np.linalg.eigh(
            0.5 * (h_matrix + h_matrix.conjugate().T)
        )
        spectral_mass = np.abs(h_eigenvectors.conjugate().T @ section) ** 2
        spectral_mass /= np.sum(spectral_mass)
        rows[name] = {
            "artifact": str(path),
            "artifact_sha256": sha256_file(path),
            "statistics": statistics,
            "h_spectrum": {
                "minimum": float(h_eigenvalues[0]),
                "maximum": float(h_eigenvalues[-1]),
                "condition_number": float(h_eigenvalues[-1] / h_eigenvalues[0]),
            },
            "worst_point": {
                "index": worst,
                "cluster_id": int(clusters[worst]),
                "h_ratio": float(ratio[worst]),
                "phi_ratio_at_same_point": float(phi_ratio[worst]),
                "importance_weight": float(weights[worst]),
                "section_h_norm_squared": denominator,
                "section_euclidean_norm_squared": section_norm_squared,
                "section_h_rayleigh_quotient": denominator / section_norm_squared,
                "section_spectral_mass_lowest_1_percent": float(
                    np.sum(spectral_mass[: max(1, len(spectral_mass) // 100)])
                ),
                "section_spectral_mass_lowest_10_percent": float(
                    np.sum(spectral_mass[: max(1, len(spectral_mass) // 10)])
                ),
                "section_derivative_frobenius_norm": float(np.linalg.norm(derivatives)),
                "x": complex_pairs(coordinates_x[worst]),
                "y": complex_pairs(coordinates_y[worst]),
                "z": complex_pairs(coordinates_z[worst]),
            },
        }
        output_arrays[f"{name}_raw_log_ratio"] = raw
        output_arrays[f"{name}_normalized_ratio"] = ratio
        output_arrays[f"{name}_min_eigenvalue"] = minimum
        print(
            f"{name}: sigma={statistics['sigma']:.6e}, chi={statistics['chi_l2']:.6e}, "
            f"max={statistics['max_ratio']:.6e}, r>10={statistics['ratio_above_10']['point_count']}",
            flush=True,
        )

    arrays_path = (
        args.arrays_out.expanduser().resolve()
        if args.arrays_out is not None
        else args.out.expanduser().resolve().with_suffix(".npz")
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **output_arrays)
    report = {
        "schema": "cached-gcicy-h-artifact-tail-comparison-v1",
        "model_seed": args.model_seed,
        "point_cache": str(point_path),
        "point_cache_sha256": sha256_file(point_path),
        "point_metadata": point_metadata,
        "phi_arrays": str(phi_path),
        "phi_arrays_sha256": sha256_file(phi_path),
        "point_count": count,
        "independent_four_root_fibre_count": len(np.unique(clusters)),
        "artifacts": rows,
        "arrays": str(arrays_path),
        "arrays_sha256": sha256_file(arrays_path),
        "runtime_seconds": time.perf_counter() - started,
        "claim_limit": "This is a fixed finite-sample comparison, not a global sup-norm certificate.",
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
