#!/usr/bin/env python3
"""Probe larger on-manifold shells around saved k=4 checkpoint spikes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--point-payload", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--points-per-shell", type=int, default=128)
    parser.add_argument(
        "--radii", type=float, nargs="+", default=(0.05, 0.1, 0.15, 0.2, 0.3)
    )
    parser.add_argument("--seed", type=int, default=881000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-point-payload", type=Path)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    observed = np.quantile(np.asarray(values, dtype=float), probabilities)
    return {
        f"q{probability:g}": float(value)
        for probability, value in zip(probabilities, observed, strict=True)
    }


def main() -> None:
    args = parse_args()
    if args.points_per_shell <= 0 or min(args.radii) <= 0:
        raise SystemExit("shell point count and radii must be positive")
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    adapter = get_adapter(source["adapter"])
    model = adapter.make_model(
        int(source["model_seed"]), exact=bool(source["exact_model"])
    )
    artifact = adapter.load_h_artifact(args.artifact, model)
    with np.load(args.point_payload, allow_pickle=False) as payload:
        point_payload = {
            "coordinates_x": np.asarray(payload["bad_coordinates_x"]),
            "coordinates_y": np.asarray(payload["bad_coordinates_y"]),
            "coordinates_z": np.asarray(payload["bad_coordinates_z"]),
        }
    centers = adapter.points_from_storage_payload(model, point_payload)
    source_rows = source["bad_points"]
    if len(centers) != len(source_rows):
        raise SystemExit("saved centers do not match the source report")

    rows: list[dict[str, Any]] = []
    top_points: list[Any] = []
    top_point_radii: list[float] = []
    top_point_ratios: list[float] = []
    top_point_log_normalizations: list[float] = []
    for center_index, (center, source_row) in enumerate(
        zip(centers, source_rows, strict=True)
    ):
        shell_rows = []
        log_normalization = float(source_row["final_log_normalization"])
        center_ratio = float(source_row["final_normalized_ratio"])
        maximum_ratio = float("-inf")
        maximum_point = None
        maximum_radius = float("nan")
        maximum_metric_eigenvalues = None
        for radius_index, radius in enumerate(args.radii):
            seed = args.seed + 1000 * center_index + radius_index
            try:
                points, diagnostics = adapter.sample_local_neighborhood(
                    model,
                    center,
                    args.points_per_shell,
                    seed=seed,
                    radius=radius,
                )
                metrics = adapter.h_metrics(points, artifact)
                raw = np.asarray(adapter.residual_values(points, metrics), dtype=float)
                log_ratio = raw - log_normalization
                ratio = np.exp(np.clip(log_ratio, -700.0, 700.0))
                minimum_eigenvalues = np.linalg.eigvalsh(metrics)[:, 0]
                local_maximum_index = int(np.argmax(ratio))
                if float(ratio[local_maximum_index]) > maximum_ratio:
                    maximum_ratio = float(ratio[local_maximum_index])
                    maximum_point = points[local_maximum_index]
                    maximum_radius = float(radius)
                    maximum_metric_eigenvalues = np.linalg.eigvalsh(
                        metrics[local_maximum_index]
                    )
                shell_rows.append(
                    {
                        "requested_intrinsic_radius": float(radius),
                        "seed": seed,
                        "ratio_quantiles": quantiles(ratio),
                        "log_ratio_quantiles": quantiles(log_ratio),
                        "ratio_above_3_fraction": float(np.mean(ratio > 3.0)),
                        "ratio_above_half_center_fraction": float(
                            np.mean(ratio > 0.5 * center_ratio)
                        ),
                        "minimum_metric_eigenvalue": float(
                            np.min(minimum_eigenvalues)
                        ),
                        "sampling_diagnostics": diagnostics,
                    }
                )
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                shell_rows.append(
                    {
                        "requested_intrinsic_radius": float(radius),
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if maximum_point is None or maximum_metric_eigenvalues is None:
            raise RuntimeError(
                f"no valid shell point was produced for center {center_index}"
            )
        top_points.append(maximum_point)
        top_point_radii.append(maximum_radius)
        top_point_ratios.append(maximum_ratio)
        top_point_log_normalizations.append(log_normalization)
        rows.append(
            {
                "bad_point_id": int(source_row["bad_point_id"]),
                "checkpoint_seed": int(source_row["checkpoint_seed"]),
                "checkpoint_local_index": int(
                    source_row["checkpoint_local_index"]
                ),
                "center_ratio": center_ratio,
                "maximum_observed_point": {
                    "ratio": maximum_ratio,
                    "requested_intrinsic_radius": maximum_radius,
                    "actual_product_fubini_study_distance": float(
                        adapter.point_distance(center, maximum_point)
                    ),
                    "metric_eigenvalues": [
                        float(value) for value in maximum_metric_eigenvalues
                    ],
                    "projective_chart": [
                        int(value) for value in maximum_point.projective_chart
                    ],
                    "independent_indices": [
                        int(value) for value in maximum_point.independent_indices
                    ],
                    "jacobian_min_singular_value": float(
                        maximum_point.jacobian_min_singular_value
                    ),
                },
                "shells": shell_rows,
            }
        )
        print(
            f"probed center {center_index + 1}/{len(centers)}: r={center_ratio:.6g}",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "description": (
            "Large-radius empirical shells around exact saved checkpoint spikes; "
            "these are not a global covering certificate."
        ),
        "source_report": str(args.source_report.expanduser().resolve()),
        "point_payload": str(args.point_payload.expanduser().resolve()),
        "artifact": str(args.artifact.expanduser().resolve()),
        "points_per_shell": args.points_per_shell,
        "radii": [float(radius) for radius in args.radii],
        "bad_points": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    payload_path = args.top_point_payload or args.out.with_suffix(".npz")
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        payload_path,
        **adapter.point_storage_payload(top_points),
        bad_point_id=np.asarray(
            [row["bad_point_id"] for row in source_rows], dtype=np.int64
        ),
        source_checkpoint_seed=np.asarray(
            [row["checkpoint_seed"] for row in source_rows], dtype=np.int64
        ),
        requested_intrinsic_radius=np.asarray(top_point_radii, dtype=float),
        observed_ratio=np.asarray(top_point_ratios, dtype=float),
        source_log_normalization=np.asarray(
            top_point_log_normalizations, dtype=float
        ),
    )
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {payload_path}", flush=True)


if __name__ == "__main__":
    main()
