#!/usr/bin/env python3
"""Audit saved H-metric points across arithmetic modes and local atlases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_h_training_precision import numpy_forward, torch_forward  # noqa: E402
from audit_selected_metric_point import (  # noqa: E402
    atlas_consistency,
    point_summary,
)
from gcicy_metric.pipeline import get_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="p4p1_type11_hirzebruch_x3")
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument(
        "--exact-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--point-payload", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--minimum-selected-coordinate", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_selected_coordinate <= 0:
        raise SystemExit("minimum-selected-coordinate must be positive")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the arithmetic replay") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    artifact = adapter.load_h_artifact(args.artifact, model)
    with np.load(args.point_payload, allow_pickle=False) as payload:
        coordinates = {
            key: np.asarray(payload[key])
            for key in ("coordinates_x", "coordinates_y", "coordinates_z")
        }
        labels = {
            key: np.asarray(payload[key])
            for key in (
                "bad_point_id",
                "source_checkpoint_seed",
                "requested_intrinsic_radius",
                "observed_ratio",
                "source_log_normalization",
            )
        }
    points = adapter.points_from_storage_payload(model, coordinates)
    if any(len(values) != len(points) for values in labels.values()):
        raise SystemExit("point labels do not match the saved coordinate count")

    source_h = np.asarray(artifact.h_matrix, dtype=np.complex128)
    source_h = 0.5 * (source_h + source_h.conjugate().T)
    source_h *= len(source_h) / float(np.trace(source_h).real)
    source_cholesky = np.linalg.cholesky(source_h)
    device = torch.device(args.device)
    cholesky32 = torch.as_tensor(
        source_cholesky, dtype=torch.complex64, device=device
    )
    training_h32 = cholesky32 @ torch.conj(cholesky32.T)
    training_h32 *= len(source_h) / torch.real(torch.trace(training_h32))
    source_h128_t = torch.as_tensor(
        source_h, dtype=torch.complex128, device=device
    )

    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        values, derivatives = adapter.section_values_and_jacobian(
            point, artifact.section_exponents
        )
        values_array = np.asarray(values, dtype=np.complex128)[None, :]
        derivatives_array = np.asarray(derivatives, dtype=np.complex128)[None, :, :]
        log_omega = np.asarray(
            [adapter.holomorphic_volume_log_density(point)], dtype=np.float64
        )
        reference = numpy_forward(
            values_array,
            derivatives_array,
            log_omega,
            source_h,
            float(artifact.normalization),
        )
        torch128 = torch_forward(
            values_array,
            derivatives_array,
            log_omega,
            source_h128_t,
            float(artifact.normalization),
        )
        torch64 = torch_forward(
            values_array,
            derivatives_array,
            log_omega,
            training_h32,
            float(artifact.normalization),
        )
        log_normalization = float(labels["source_log_normalization"][index])
        reference_ratio = float(np.exp(reference["raw"][0] - log_normalization))
        torch64_ratio = float(np.exp(torch64["raw"][0] - log_normalization))
        recorded_ratio = float(labels["observed_ratio"][index])
        rows.append(
            {
                "index": index,
                "bad_point_id": int(labels["bad_point_id"][index]),
                "source_checkpoint_seed": int(
                    labels["source_checkpoint_seed"][index]
                ),
                "requested_intrinsic_radius": float(
                    labels["requested_intrinsic_radius"][index]
                ),
                "recorded_ratio": recorded_ratio,
                "numpy_complex128_ratio": reference_ratio,
                "recorded_ratio_relative_error": float(
                    abs(reference_ratio - recorded_ratio) / recorded_ratio
                ),
                "torch_complex64_ratio": torch64_ratio,
                "torch64_absolute_log_ratio_error": float(
                    abs(torch64["raw"][0] - reference["raw"][0])
                ),
                "torch128_absolute_log_ratio_error": float(
                    abs(torch128["raw"][0] - reference["raw"][0])
                ),
                "numpy_complex128_minimum_metric_eigenvalue": float(
                    reference["minimum_eigenvalue"][0]
                ),
                "torch_complex64_minimum_metric_eigenvalue": float(
                    torch64["minimum_eigenvalue"][0]
                ),
                "section_formula_cancellation_amplification": float(
                    reference["cancellation_amplification"][0]
                ),
                "point": point_summary(adapter, model, point, artifact),
                "atlas_consistency": atlas_consistency(
                    adapter,
                    model,
                    point,
                    artifact,
                    minimum_selected_coordinate=args.minimum_selected_coordinate,
                ),
            }
        )
        print(f"audited point {index + 1}/{len(points)}", flush=True)

    output = {
        "schema_version": 1,
        "description": (
            "Fixed-coordinate complex128/complex64 and all-atlas audit of "
            "large-shell H-metric tail candidates."
        ),
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "artifact": str(args.artifact.expanduser().resolve()),
        "point_payload": str(args.point_payload.expanduser().resolve()),
        "point_count": len(points),
        "device": str(device),
        "maximum_recorded_ratio_relative_error": float(
            max(row["recorded_ratio_relative_error"] for row in rows)
        ),
        "maximum_torch64_absolute_log_ratio_error": float(
            max(row["torch64_absolute_log_ratio_error"] for row in rows)
        ),
        "maximum_torch128_absolute_log_ratio_error": float(
            max(row["torch128_absolute_log_ratio_error"] for row in rows)
        ),
        "maximum_projective_chart_log_ratio_error": float(
            max(
                row["atlas_consistency"][
                    "maximum_projective_candidate_ma_error"
                ]
                for row in rows
            )
        ),
        "maximum_implicit_coordinate_log_ratio_error": float(
            max(
                row["atlas_consistency"][
                    "maximum_implicit_candidate_ma_error"
                ]
                for row in rows
            )
        ),
        "minimum_recharted_metric_eigenvalue": float(
            min(
                row["atlas_consistency"]["minimum_recharted_metric_eigenvalue"]
                for row in rows
            )
        ),
        "points": rows,
        "claim_limit": (
            "This validates the saved candidates across implemented charts and "
            "floating-point modes; it is not a global sup-norm certificate."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
