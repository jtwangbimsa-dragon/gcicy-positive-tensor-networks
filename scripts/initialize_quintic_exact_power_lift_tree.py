#!/usr/bin/env python3
"""Freeze and certify an exact low-degree-teacher power-lift tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.exact_lift_tree import (  # noqa: E402
    build_exact_power_lift_tree,
    exact_lift_identity_errors,
    exact_power_lift_payload,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    prepare_features,
    ratio_statistics,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h", type=Path, required=True)
    parser.add_argument("--pool-points", type=Path, required=True)
    parser.add_argument("--pool-pullbacks", type=Path, required=True)
    parser.add_argument("--power", type=int, required=True)
    parser.add_argument("--bond-dimension", type=int, default=1)
    parser.add_argument("--positive-floor", type=float, default=1.0e-8)
    parser.add_argument("--verification-size", type=int, default=4096)
    parser.add_argument("--verification-seed", type=int, default=202607386)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=("complex64", "complex128"),
        default="complex128",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def tail_summary(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "q999": float(
            statistics["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99": float(
            statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum": float(
            statistics["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
    }


def main() -> None:
    args = parse_args()
    if (
        args.power <= 0
        or args.bond_dimension <= 0
        or args.verification_size <= 0
        or args.feature_batch_size <= 0
        or args.eval_batch_size <= 0
        or not 0 <= args.positive_floor < 1
    ):
        raise ValueError("power, rank, sample, and batch values must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    started = time.perf_counter()
    source_path = args.source_h.expanduser().resolve()
    points_path = args.pool_points.expanduser().resolve()
    pullbacks_path = args.pool_pullbacks.expanduser().resolve()
    output_model = args.output_model.expanduser().resolve()
    report_path = args.out.expanduser().resolve()
    status_path = report_path.parent / "status.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(status_path, {"state": "running", "phase": "loading"})
    try:
        source = np.load(source_path, allow_pickle=False)
        source_h = np.asarray(source["global_h_matrix"], dtype=np.complex128)
        source_degree = int(source["degree"])
        source_exponents = np.asarray(source["exponents"], dtype=np.int64)
        if source_h.shape != (len(source_exponents), len(source_exponents)):
            raise ValueError("source H and section basis dimensions disagree")
        source_normalization = 1.0 / (math.pi * source_degree)

        model = build_exact_power_lift_tree(
            source_h,
            source_normalization=source_normalization,
            power=args.power,
            bond_dimension=args.bond_dimension,
            positive_floor=args.positive_floor,
            precision=args.precision,
            device=args.device,
        )
        payload = exact_power_lift_payload(
            model,
            source_degree=source_degree,
            source_exponents=source_exponents,
            source_normalization=source_normalization,
            source_artifact_sha256=sha256_file(source_path),
        )
        output_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_model)

        points = np.load(points_path, allow_pickle=False)
        x_values = np.asarray(points["X"], dtype=np.float32)
        weights = np.asarray(points["weights"], dtype=np.float64)
        omega_squared = np.asarray(
            points["omega_squared"],
            dtype=np.float64,
        )
        pullbacks = np.load(pullbacks_path, mmap_mode="r")
        if not (
            len(x_values)
            == len(weights)
            == len(omega_squared)
            == len(pullbacks)
        ):
            raise RuntimeError("verification pool arrays are not aligned")
        if args.verification_size > len(x_values):
            raise ValueError("verification size exceeds the point pool")
        indices = np.random.default_rng(args.verification_seed).choice(
            len(x_values),
            size=args.verification_size,
            replace=False,
        )
        feature_values, feature_derivatives = prepare_features(
            x_values[indices],
            np.asarray(pullbacks[indices]),
            source_exponents,
            args.feature_batch_size,
            complex_dtype=(
                np.dtype(np.complex64)
                if args.precision == "complex64"
                else np.dtype(np.complex128)
            ),
        )
        complex_dtype = (
            torch.complex64
            if args.precision == "complex64"
            else torch.complex128
        )
        values = torch.tensor(
            feature_values,
            dtype=complex_dtype,
            device=args.device,
        )
        derivatives = torch.tensor(
            feature_derivatives,
            dtype=complex_dtype,
            device=args.device,
        )
        write_json(status_path, {"state": "running", "phase": "verifying"})

        identity_rows = []
        raw_rows = []
        minimum_rows = []
        with torch.no_grad():
            for start in range(0, len(values), args.eval_batch_size):
                stop = min(start + args.eval_batch_size, len(values))
                block_values = values[start:stop]
                block_derivatives = derivatives[start:stop]
                identity_rows.append(
                    exact_lift_identity_errors(
                        model,
                        source_h,
                        block_values,
                        block_derivatives,
                        source_normalization=source_normalization,
                    )
                )
                _, metric = model.potential_and_metric(
                    block_values,
                    block_derivatives,
                )
                eigenvalues = torch.linalg.eigvalsh(metric)
                if not bool(torch.all(torch.isfinite(eigenvalues))):
                    raise FloatingPointError("lift metric has nonfinite eigenvalues")
                raw_rows.append(
                    (
                        torch.sum(torch.log(eigenvalues), dim=1)
                        - torch.log(
                            torch.tensor(
                                omega_squared[indices[start:stop]],
                                dtype=eigenvalues.dtype,
                                device=eigenvalues.device,
                            )
                        )
                    ).cpu().numpy()
                )
                minimum_rows.append(
                    torch.min(eigenvalues, dim=1).values.cpu().numpy()
                )

        raw = np.concatenate(raw_rows).astype(np.float64)
        minimum = np.concatenate(minimum_rows).astype(np.float64)
        normalized_weights = weights[indices] / np.sum(weights[indices])
        statistics, _ = ratio_statistics(raw, normalized_weights, minimum)
        identity = {
            key: max(row[key] for row in identity_rows)
            for key in identity_rows[0]
        }
        if args.precision == "complex128":
            if (
                identity["potential_maximum_absolute_error"] > 2.0e-11
                or identity["metric_maximum_absolute_error"] > 2.0e-11
                or identity["metric_relative_frobenius_error"] > 2.0e-11
            ):
                raise RuntimeError("exact lift failed its FP64 identity gate")
        if statistics["nonpositive_min_eigenvalue"]["count"]:
            raise RuntimeError("exact lift failed positivity verification")

        report = {
            "schema": "quintic-exact-power-lift-tree-initialization-v1",
            "source": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "degree": source_degree,
                "section_count": len(source_exponents),
            },
            "tree": {
                "power": args.power,
                "target_degree": source_degree * args.power,
                "site_count": args.power,
                "bond_dimension": args.bond_dimension,
                "trainable_real_parameter_count": (
                    model.trainable_real_parameter_count
                ),
                "positive_floor": args.positive_floor,
                "teacher_runtime_dependency": False,
                "initialization_package": str(output_model),
                "initialization_package_sha256": sha256_file(output_model),
            },
            "verification": {
                "pool_points": str(points_path),
                "pool_points_sha256": sha256_file(points_path),
                "pool_pullbacks": str(pullbacks_path),
                "pool_pullbacks_sha256": sha256_file(pullbacks_path),
                "sample_size": args.verification_size,
                "seed": args.verification_seed,
                "identity": identity,
                "statistics": statistics,
                "tail": tail_summary(statistics),
            },
            "runtime_contract": (
                "The native trainer reads only the initialization package; "
                "the source teacher path is provenance, not a runtime input."
            ),
            "timing_seconds": time.perf_counter() - started,
        }
        report_path.write_text(
            json.dumps(json_value(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "timing_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps(json_value(report), indent=2), flush=True)
    except Exception as exc:
        write_json(
            status_path,
            {
                "state": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
