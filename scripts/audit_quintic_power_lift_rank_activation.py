#!/usr/bin/env python3
"""Audit the metric perturbation introduced while activating dormant lift ranks."""

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
    dataset_subset,
    load_pool_arrays,
    make_dataset,
    normalized_native_e2,
    paired_improvement,
    statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--pool-points", type=Path, required=True)
    parser.add_argument("--pool-pullbacks", type=Path, required=True)
    parser.add_argument("--size", type=int, default=20000)
    parser.add_argument("--gradient-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=202607270)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_jets(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    potential_rows = []
    metric_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            potential, metric = model.potential_and_metric(
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
            )
            potential_rows.append(
                potential.detach().cpu().to(torch.float64).numpy()
            )
            metric_rows.append(
                metric.detach().cpu().to(torch.complex128).numpy()
            )
    return np.concatenate(potential_rows), np.concatenate(metric_rows)


def load_model_and_dataset(
    path: Path,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    feature_batch_size: int,
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("teacher_runtime_dependency") is not False:
        raise RuntimeError(f"{path} is not a teacher-free package")
    model = power_lift_tree_from_payload(payload, device="cpu")
    complex_dtype = next(model.parameters()).dtype
    dataset = make_dataset(
        arrays,
        np.asarray(payload["source_exponents"], dtype=np.int64),
        feature_batch_size=feature_batch_size,
        complex_dtype=complex_dtype,
        device=torch.device("cpu"),
    )
    return payload, model, dataset


def gradient_audit(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    reference_payload: dict[str, Any],
    *,
    gradient_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    count = min(gradient_size, dataset["count"])
    sample = dataset_subset(dataset, np.arange(count, dtype=np.int64))
    model.zero_grad(set_to_none=True)
    loss, _ = normalized_native_e2(
        model,
        sample,
        chunk_size=chunk_size,
    )
    loss.backward()
    reference_state = reference_payload["state_dict"]
    rows = []
    total_squared = 0.0
    dormant_squared = 0.0
    for site, core in enumerate(model.cores):
        gradient = core.grad
        if gradient is None:
            raise RuntimeError(f"site {site} has no native E2 gradient")
        old_shape = reference_state[f"cores.{site}"].shape
        dormant = gradient.detach().clone()
        old_slices = tuple(slice(0, size) for size in old_shape)
        dormant[old_slices] = 0
        total_norm = float(torch.linalg.vector_norm(gradient))
        dormant_norm = float(torch.linalg.vector_norm(dormant))
        rows.append(
            {
                "site": site,
                "total_gradient_norm": total_norm,
                "dormant_gradient_norm": dormant_norm,
            }
        )
        total_squared += total_norm * total_norm
        dormant_squared += dormant_norm * dormant_norm
    model.zero_grad(set_to_none=True)
    return {
        "sample_size": count,
        "native_e2": float(loss.detach().cpu()),
        "total_gradient_norm": float(np.sqrt(total_squared)),
        "dormant_gradient_norm": float(np.sqrt(dormant_squared)),
        "sites": rows,
    }


def main() -> None:
    args = parse_args()
    if (
        min(
            args.size,
            args.gradient_size,
            args.feature_batch_size,
            args.eval_batch_size,
        )
        <= 0
    ):
        raise ValueError("sample and batch sizes must be positive")
    started = time.perf_counter()
    reference_path = args.reference.expanduser().resolve()
    candidate_paths = [
        path.expanduser().resolve() for path in args.candidate
    ]
    arrays, indices = load_pool_arrays(
        args.pool_points.expanduser().resolve(),
        args.pool_pullbacks.expanduser().resolve(),
        size=args.size,
        seed=args.seed,
    )
    reference_payload, reference_model, reference_dataset = (
        load_model_and_dataset(
            reference_path,
            arrays,
            feature_batch_size=args.feature_batch_size,
        )
    )
    reference_statistics, reference_ratio = statistics(
        reference_model,
        reference_dataset,
        chunk_size=args.eval_batch_size,
    )
    reference_potential, reference_metric = model_jets(
        reference_model,
        reference_dataset,
        chunk_size=args.eval_batch_size,
    )

    rows = []
    for path in candidate_paths:
        payload, model, dataset = load_model_and_dataset(
            path,
            arrays,
            feature_batch_size=args.feature_batch_size,
        )
        if not np.array_equal(
            np.asarray(payload["source_exponents"], dtype=np.int64),
            np.asarray(
                reference_payload["source_exponents"],
                dtype=np.int64,
            ),
        ):
            raise RuntimeError("candidate and reference section bases differ")
        candidate_statistics, candidate_ratio = statistics(
            model,
            dataset,
            chunk_size=args.eval_batch_size,
        )
        candidate_potential, candidate_metric = model_jets(
            model,
            dataset,
            chunk_size=args.eval_batch_size,
        )
        potential_difference = candidate_potential - reference_potential
        metric_difference = candidate_metric - reference_metric
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "precision": payload["precision"],
                "bond_dimension": int(payload["bond_dimension"]),
                "trainable_real_parameter_count": (
                    model.trainable_real_parameter_count
                ),
                "bond_expansion": payload.get("bond_expansion"),
                "statistics": candidate_statistics,
                "paired_change_from_reference": paired_improvement(
                    reference_ratio,
                    candidate_ratio,
                    reference_dataset["weights_numpy"],
                ),
                "jet_difference": {
                    "potential_rms": float(
                        np.sqrt(np.mean(np.square(potential_difference)))
                    ),
                    "potential_maximum_absolute": float(
                        np.max(np.abs(potential_difference))
                    ),
                    "metric_relative_frobenius": float(
                        np.linalg.norm(metric_difference)
                        / max(
                            np.linalg.norm(reference_metric),
                            np.finfo(float).tiny,
                        )
                    ),
                    "metric_maximum_absolute": float(
                        np.max(np.abs(metric_difference))
                    ),
                },
                "gradient_audit": gradient_audit(
                    model,
                    dataset,
                    reference_payload,
                    gradient_size=args.gradient_size,
                    chunk_size=args.eval_batch_size,
                ),
            }
        )

    report = {
        "schema": "quintic-power-lift-rank-activation-audit-v1",
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "precision": reference_payload["precision"],
            "bond_dimension": int(reference_payload["bond_dimension"]),
            "statistics": reference_statistics,
        },
        "pool": {
            "points": str(args.pool_points.expanduser().resolve()),
            "pullbacks": str(args.pool_pullbacks.expanduser().resolve()),
            "size": args.size,
            "gradient_size": min(args.gradient_size, args.size),
            "seed": args.seed,
            "indices_sha256": hashlib.sha256(
                np.asarray(indices, dtype=np.int64).tobytes()
            ).hexdigest(),
        },
        "candidates": rows,
        "timing_seconds": time.perf_counter() - started,
    }
    write_json(args.out.expanduser().resolve(), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
