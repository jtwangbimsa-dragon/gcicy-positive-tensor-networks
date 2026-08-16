#!/usr/bin/env python3
"""Evaluate frozen power-lift models once on a common final blind pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.exact_lift_tree import (  # noqa: E402
    power_lift_tree_from_payload,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    load_pool_arrays,
    make_dataset,
    paired_improvement,
    statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="frozen model in LABEL=PATH form; repeat for each model",
    )
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--pool-points", type=Path, required=True)
    parser.add_argument("--pool-pullbacks", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=202607283)
    parser.add_argument("--feature-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--expected-points-sha256")
    parser.add_argument("--expected-pullbacks-sha256")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model_specs(values: list[str]) -> list[tuple[str, Path]]:
    rows = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise ValueError("each model must use LABEL=PATH")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in labels:
            raise ValueError("model labels must be nonempty and unique")
        labels.add(label)
        rows.append((label, Path(raw_path).expanduser().resolve()))
    return rows


def state_bytes(model: torch.nn.Module) -> int:
    return int(
        sum(
            value.numel() * value.element_size()
            for value in model.state_dict().values()
        )
    )


def main() -> None:
    args = parse_args()
    if min(args.size, args.feature_batch_size, args.eval_batch_size) <= 0:
        raise ValueError("sample and batch sizes must be positive")
    models = parse_model_specs(args.model)
    labels = {label for label, _ in models}
    if args.reference_label not in labels:
        raise ValueError("reference label is absent from the model list")

    started = time.perf_counter()
    points_path = args.pool_points.expanduser().resolve()
    pullbacks_path = args.pool_pullbacks.expanduser().resolve()
    points_sha256 = sha256_file(points_path)
    pullbacks_sha256 = sha256_file(pullbacks_path)
    if (
        args.expected_points_sha256
        and points_sha256 != args.expected_points_sha256
    ):
        raise RuntimeError("blind point-pool hash does not match the frozen value")
    if (
        args.expected_pullbacks_sha256
        and pullbacks_sha256 != args.expected_pullbacks_sha256
    ):
        raise RuntimeError(
            "blind pullback-pool hash does not match the frozen value"
        )
    arrays, indices = load_pool_arrays(
        points_path,
        pullbacks_path,
        size=args.size,
        seed=args.seed,
    )

    payload_rows: dict[str, dict[str, Any]] = {}
    model_rows: dict[str, torch.nn.Module] = {}
    common_exponents = None
    for label, path in models:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("teacher_runtime_dependency") is not False:
            raise RuntimeError(f"{label} is not a teacher-free frozen package")
        exponents = np.asarray(payload["source_exponents"], dtype=np.int64)
        if common_exponents is None:
            common_exponents = exponents
        elif not np.array_equal(exponents, common_exponents):
            raise RuntimeError("frozen models do not share a section basis")
        payload_rows[label] = payload
        model_rows[label] = power_lift_tree_from_payload(
            payload,
            device="cpu",
        )

    datasets: dict[torch.dtype, dict[str, Any]] = {}
    feature_timing: dict[str, float] = {}
    for dtype in {next(model.parameters()).dtype for model in model_rows.values()}:
        feature_started = time.perf_counter()
        datasets[dtype] = make_dataset(
            arrays,
            common_exponents,
            feature_batch_size=args.feature_batch_size,
            complex_dtype=dtype,
            device=torch.device("cpu"),
        )
        feature_timing[str(dtype)] = time.perf_counter() - feature_started

    results = {}
    ratios = {}
    for label, path in models:
        model = model_rows[label]
        dtype = next(model.parameters()).dtype
        evaluation_started = time.perf_counter()
        model_statistics, ratio = statistics(
            model,
            datasets[dtype],
            chunk_size=args.eval_batch_size,
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
        payload = payload_rows[label]
        results[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema": payload["schema"],
            "source_degree": payload["source_degree"],
            "target_degree": payload["target_degree"],
            "site_count": int(payload["site_count"]),
            "bond_dimension": int(payload["bond_dimension"]),
            "precision": payload["precision"],
            "trainable_real_parameter_count": (
                model.trainable_real_parameter_count
            ),
            "state_bytes": state_bytes(model),
            "evaluation_seconds": evaluation_seconds,
            "points_per_second": args.size / evaluation_seconds,
            "statistics": model_statistics,
        }
        ratios[label] = ratio

    reference_ratio = ratios[args.reference_label]
    reference_dtype = next(
        model_rows[args.reference_label].parameters()
    ).dtype
    reference_weights = datasets[reference_dtype]["weights_numpy"]
    comparisons = {}
    for label, _ in models:
        if label == args.reference_label:
            continue
        comparisons[label] = {
            "reference_label": args.reference_label,
            "paired_improvement": paired_improvement(
                reference_ratio,
                ratios[label],
                reference_weights,
            ),
        }

    report = {
        "schema": "quintic-power-lift-final-blind-comparison-v1",
        "protocol": {
            "final_blind_opened_once": True,
            "models_frozen_before_evaluation": True,
            "post_blind_model_selection_allowed": False,
            "reference_label": args.reference_label,
        },
        "pool": {
            "points": str(points_path),
            "points_sha256": points_sha256,
            "pullbacks": str(pullbacks_path),
            "pullbacks_sha256": pullbacks_sha256,
            "size": args.size,
            "seed": args.seed,
            "indices_sha256": hashlib.sha256(
                np.asarray(indices, dtype=np.int64).tobytes()
            ).hexdigest(),
        },
        "feature_timing_seconds": feature_timing,
        "models": results,
        "comparisons_to_reference": comparisons,
        "total_timing_seconds": time.perf_counter() - started,
    }
    output_path = args.out.expanduser().resolve()
    write_json(output_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
