#!/usr/bin/env python3
"""Compare local tail neighborhoods on the common Fermat-quintic point set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_KEYS = ("raw_full_h", "s5_full_h", "cymetric_phi", "fs_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--raw-h-run-dir", type=Path, required=True)
    parser.add_argument("--symmetrized-h-run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-out", type=Path)
    parser.add_argument("--top-centers", type=int, default=32)
    parser.add_argument(
        "--neighbor-counts",
        type=int,
        nargs="+",
        default=(8, 32, 128, 512),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
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


def quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    observed = np.quantile(array, [0.0, 0.5, 0.9, 0.99, 1.0])
    return {
        label: float(value)
        for label, value in zip(
            ("minimum", "median", "q90", "q99", "maximum"),
            observed,
            strict=True,
        )
    }


def normalized_projective_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.complex128)
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("projective points must have finite positive norms")
    return values / norms[:, None]


def projective_distances(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    overlap = np.abs(points @ np.conj(center))
    return np.arccos(np.clip(overlap, 0.0, 1.0))


def nearest_indices(distances: np.ndarray, count: int) -> np.ndarray:
    retain = min(int(count), len(distances))
    indices = np.argpartition(distances, retain - 1)[:retain]
    return indices[np.argsort(distances[indices], kind="stable")]


def local_model_profile(
    ratio: np.ndarray,
    center_index: int,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> dict[str, Any]:
    center_log = float(np.log(ratio[center_index]))
    neighbor_log = np.log(ratio[neighbor_indices])
    absolute_residual = np.abs(1.0 - ratio[neighbor_indices])
    slopes = np.abs(neighbor_log - center_log) / np.maximum(
        neighbor_distances,
        np.finfo(np.float64).tiny,
    )
    return {
        "center_ratio": float(ratio[center_index]),
        "center_abs_residual": float(abs(1.0 - ratio[center_index])),
        "neighbor_ratio": quantiles(ratio[neighbor_indices]),
        "neighbor_abs_residual": quantiles(absolute_residual),
        "absolute_log_ratio_secant_slope": quantiles(slopes),
    }


def aggregate_center_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "center_count": len(rows),
        "nearest_training_distance": quantiles(
            np.asarray([row["nearest_training_distance"] for row in rows])
        ),
        "nearest_blind_distance": quantiles(
            np.asarray([row["nearest_blind_distance"] for row in rows])
        ),
        "two_largest_coordinate_mass_fraction": quantiles(
            np.asarray(
                [row["two_largest_coordinate_mass_fraction"] for row in rows]
            )
        ),
    }
    output["models_at_centers"] = {
        model: {
            "ratio": quantiles(
                np.asarray([row["model_values"][model]["ratio"] for row in rows])
            ),
            "abs_residual": quantiles(
                np.asarray(
                    [row["model_values"][model]["abs_residual"] for row in rows]
                )
            ),
        }
        for model in MODEL_KEYS
    }
    return output


def main() -> None:
    args = parse_args()
    if args.top_centers <= 0:
        raise ValueError("top-centers must be positive")
    neighbor_counts = tuple(sorted(set(int(value) for value in args.neighbor_counts)))
    if not neighbor_counts or neighbor_counts[0] <= 0:
        raise ValueError("neighbor-counts must be positive")

    source_dir = args.source_run_dir.expanduser().resolve()
    raw_dir = args.raw_h_run_dir.expanduser().resolve()
    sym_dir = args.symmetrized_h_run_dir.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    probe_path = (
        args.probe_out.expanduser().resolve()
        if args.probe_out is not None
        else output_path.with_name(output_path.stem + "_probes.npz")
    )
    if output_path.exists() or probe_path.exists():
        raise FileExistsError("neighborhood outputs already exist")

    dataset_path = source_dir / "training_data" / "dataset.npz"
    blind_path = source_dir / "blind_points.npz"
    source_tail_path = source_dir / "blind_test_tail_arrays.npz"
    raw_tail_path = raw_dir / "blind_test_tail_arrays.npz"
    sym_tail_path = sym_dir / "blind_test_tail_arrays.npz"
    for path in (
        dataset_path,
        blind_path,
        source_tail_path,
        raw_tail_path,
        sym_tail_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(dataset_path, allow_pickle=False) as dataset:
        train_x = np.asarray(dataset["X_train"], dtype=np.float64)
    with np.load(blind_path, allow_pickle=False) as blind:
        blind_points = np.asarray(blind["complex_points"], dtype=np.complex128)
        blind_weights = np.asarray(blind["weights"], dtype=np.float64)
    with np.load(source_tail_path, allow_pickle=False) as source_tail:
        cymetric_ratio = np.asarray(source_tail["normalized_ratio"], dtype=np.float64)
        fs_ratio = np.asarray(source_tail["fs_normalized_ratio"], dtype=np.float64)
        source_weights = np.asarray(source_tail["weights"], dtype=np.float64)
    with np.load(raw_tail_path, allow_pickle=False) as raw_tail:
        raw_ratio = np.asarray(raw_tail["normalized_ratio"], dtype=np.float64)
        raw_weights = np.asarray(raw_tail["weights"], dtype=np.float64)
    with np.load(sym_tail_path, allow_pickle=False) as sym_tail:
        sym_ratio = np.asarray(sym_tail["normalized_ratio"], dtype=np.float64)
        sym_weights = np.asarray(sym_tail["weights"], dtype=np.float64)

    point_count = len(blind_points)
    arrays = (blind_weights, source_weights, raw_weights, sym_weights)
    if any(len(values) != point_count for values in arrays):
        raise ValueError("common blind arrays have inconsistent lengths")
    if not all(
        np.array_equal(blind_weights, values)
        for values in (source_weights, raw_weights, sym_weights)
    ):
        raise ValueError("common blind arrays are not pointwise weight matched")
    ratios = {
        "raw_full_h": raw_ratio,
        "s5_full_h": sym_ratio,
        "cymetric_phi": cymetric_ratio,
        "fs_baseline": fs_ratio,
    }
    if any(
        len(values) != point_count
        or np.any(~np.isfinite(values))
        or np.any(values <= 0)
        for values in ratios.values()
    ):
        raise ValueError("all common-point ratios must be finite and positive")

    n_coordinates = train_x.shape[1] // 2
    train_complex = train_x[:, :n_coordinates] + 1j * train_x[:, n_coordinates:]
    normalized_train = normalized_projective_points(train_complex)
    normalized_blind = normalized_projective_points(blind_points)

    center_sources: dict[int, list[str]] = {}
    for model in ("raw_full_h", "s5_full_h", "cymetric_phi"):
        residual = np.abs(1.0 - ratios[model])
        count = min(args.top_centers, len(residual))
        selected = np.argsort(residual, kind="stable")[-count:][::-1]
        for index in selected:
            center_sources.setdefault(int(index), []).append(model)

    maximum_neighbors = neighbor_counts[-1]
    center_rows = []
    probe_indices: set[int] = set(center_sources)
    training_probe_indices: set[int] = set()
    for ordinal, center_index in enumerate(sorted(center_sources), start=1):
        center = normalized_blind[center_index]
        blind_distances = projective_distances(normalized_blind, center)
        blind_distances[center_index] = np.inf
        blind_neighbors = nearest_indices(blind_distances, maximum_neighbors)
        train_distances = projective_distances(normalized_train, center)
        train_neighbors = nearest_indices(train_distances, maximum_neighbors)
        probe_indices.update(int(value) for value in blind_neighbors)
        training_probe_indices.update(int(value) for value in train_neighbors)

        coordinate_mass = np.square(np.abs(center))
        two_largest = np.sort(coordinate_mass)[-2:]
        row: dict[str, Any] = {
            "center_index": center_index,
            "selection_sources": sorted(center_sources[center_index]),
            "nearest_training_distance": float(train_distances[train_neighbors[0]]),
            "nearest_blind_distance": float(blind_distances[blind_neighbors[0]]),
            "two_largest_coordinate_indices": np.argsort(coordinate_mass)[-2:][::-1],
            "two_largest_coordinate_mass_fraction": float(np.sum(two_largest)),
            "model_values": {
                model: {
                    "ratio": float(values[center_index]),
                    "abs_residual": float(abs(1.0 - values[center_index])),
                }
                for model, values in ratios.items()
            },
            "neighborhoods": {},
        }
        for count in neighbor_counts:
            indices = blind_neighbors[:count]
            distances = blind_distances[indices]
            row["neighborhoods"][str(count)] = {
                "distance": quantiles(distances),
                "models": {
                    model: local_model_profile(
                        values,
                        center_index,
                        indices,
                        distances,
                    )
                    for model, values in ratios.items()
                },
            }
        center_rows.append(row)
        if ordinal % 10 == 0 or ordinal == len(center_sources):
            print(f"centers={ordinal}/{len(center_sources)}", flush=True)

    by_source = {}
    for source in ("raw_full_h", "s5_full_h", "cymetric_phi"):
        source_rows = [row for row in center_rows if source in row["selection_sources"]]
        by_source[source] = aggregate_center_rows(source_rows)

    overlap = {
        f"{first}__{second}": len(
            {
                row["center_index"]
                for row in center_rows
                if first in row["selection_sources"]
            }
            & {
                row["center_index"]
                for row in center_rows
                if second in row["selection_sources"]
            }
        )
        for first, second in (
            ("raw_full_h", "s5_full_h"),
            ("raw_full_h", "cymetric_phi"),
            ("s5_full_h", "cymetric_phi"),
        )
    }
    report = {
        "schema": "quintic-common-point-local-neighborhood-v1",
        "claim_scope": {
            "classification": "paired_empirical_local_diagnostic",
            "distance": "projective Fubini-Study distance arccos(|<z,w>|)",
            "warning": (
                "Nearest neighbors in finite train and blind samples do not form an "
                "epsilon-net and do not certify a global Lipschitz bound."
            ),
        },
        "inputs": {
            "source_run_dir": str(source_dir),
            "raw_h_run_dir": str(raw_dir),
            "symmetrized_h_run_dir": str(sym_dir),
            "sha256": {
                str(path): sha256_file(path)
                for path in (
                    dataset_path,
                    blind_path,
                    source_tail_path,
                    raw_tail_path,
                    sym_tail_path,
                )
            },
        },
        "inventory": {
            "training_point_count": len(normalized_train),
            "blind_point_count": len(normalized_blind),
            "top_centers_per_model": args.top_centers,
            "unique_center_count": len(center_rows),
            "neighbor_counts": list(neighbor_counts),
            "probe_blind_point_count": len(probe_indices),
            "probe_training_point_count": len(training_probe_indices),
        },
        "top_center_overlap": overlap,
        "aggregate_by_selection_source": by_source,
        "centers": center_rows,
        "probe_artifact": str(probe_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probe_path,
        blind_indices=np.asarray(sorted(probe_indices), dtype=np.int64),
        training_indices=np.asarray(sorted(training_probe_indices), dtype=np.int64),
        center_indices=np.asarray(sorted(center_sources), dtype=np.int64),
        center_source_labels=np.asarray(
            ["+".join(sorted(center_sources[index])) for index in sorted(center_sources)]
        ),
        source_blind_points_sha256=np.asarray(sha256_file(blind_path)),
        source_training_dataset_sha256=np.asarray(sha256_file(dataset_path)),
    )
    write_json(output_path, report)
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {probe_path}", flush=True)


if __name__ == "__main__":
    main()
