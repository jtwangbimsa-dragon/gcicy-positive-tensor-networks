#!/usr/bin/env python3
"""Confirm one frozen generic-quintic compiled-tree proposal against its parent."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_generic_quintic_h4_architecture_arms import (  # noqa: E402
    build_checkpoint_model,
    infer_architecture,
)
from scripts.refine_generic_quintic_compiled_tree_native_gn import (  # noqa: E402
    load_disjoint_splits,
    load_excluded_indices,
)
from scripts.train_generic_quintic_compiled_tree import (  # noqa: E402
    metric_row,
    sha256_file,
    whiten_dataset,
)
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    make_dataset,
    paired_improvement,
    tail_guard,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmation-points", type=Path, required=True)
    parser.add_argument("--confirmation-pullbacks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-indices-file",
        type=Path,
        nargs="*",
        default=(),
    )
    parser.add_argument("--confirmation-size", type=int, default=5000)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=202607441)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def checkpoint_contract(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    configuration = payload["configuration"]
    return (
        np.asarray(configuration["exponents"], dtype=np.int64),
        np.asarray(configuration["whitening"], dtype=np.complex128),
        str(configuration["precision"]),
    )


def main() -> None:
    args = parse_args()
    if min(
        args.confirmation_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.threads,
    ) <= 0:
        raise ValueError("confirmation and batch sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a confirmation run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    started = time.perf_counter()

    try:
        baseline_path = args.baseline_checkpoint.expanduser().resolve()
        candidate_path = args.candidate_checkpoint.expanduser().resolve()
        baseline_payload = torch.load(
            baseline_path,
            map_location="cpu",
            weights_only=False,
        )
        candidate_payload = torch.load(
            candidate_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            infer_architecture(baseline_payload) != "compiled-tree"
            or infer_architecture(candidate_payload) != "compiled-tree"
        ):
            raise ValueError("both checkpoints must be compiled trees")
        if bool(
            baseline_payload.get("teacher_runtime_dependency", False)
            or candidate_payload.get("teacher_runtime_dependency", False)
        ):
            raise ValueError("confirmation checkpoints must be teacher-independent")
        baseline_contract = checkpoint_contract(baseline_payload)
        candidate_contract = checkpoint_contract(candidate_payload)
        if (
            baseline_contract[2] != candidate_contract[2]
            or not np.array_equal(
                baseline_contract[0],
                candidate_contract[0],
            )
            or not np.allclose(
                baseline_contract[1],
                candidate_contract[1],
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise ValueError("baseline and candidate feature contracts differ")
        exponents, whitening, precision = baseline_contract
        dtype = (
            torch.complex64
            if precision == "complex64"
            else torch.complex128
        )

        excluded = load_excluded_indices(args.exclude_indices_file)
        arrays, indices = load_disjoint_splits(
            {
                "confirmation": (
                    args.confirmation_points,
                    args.confirmation_pullbacks,
                    args.confirmation_size,
                )
            },
            seed=args.seed,
            exclusions=excluded,
        )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **indices)
        dataset = whiten_dataset(
            make_dataset(
                arrays["confirmation"],
                exponents,
                feature_batch_size=args.feature_batch_size,
                complex_dtype=dtype,
                device=device,
            ),
            whitening,
        )
        baseline_model = build_checkpoint_model(
            baseline_payload,
            device=device,
        )
        candidate_model = build_checkpoint_model(
            candidate_payload,
            device=device,
        )
        baseline_statistics, baseline_ratio, baseline_tail = metric_row(
            baseline_model,
            dataset,
            chunk_size=args.eval_batch_size,
        )
        candidate_statistics, candidate_ratio, candidate_tail = metric_row(
            candidate_model,
            dataset,
            chunk_size=args.eval_batch_size,
        )
        paired = paired_improvement(
            baseline_ratio,
            candidate_ratio,
            dataset["weights_numpy"],
        )
        passes = bool(
            candidate_statistics["sigma_official_formula"]
            < baseline_statistics["sigma_official_formula"]
            and candidate_statistics["weighted_rms_abs_residual"]
            < baseline_statistics["weighted_rms_abs_residual"]
            and paired["e2"]["ci95_low"] > 0
            and paired["sigma"]["ci95_low"] > 0
            and tail_guard(
                candidate_tail,
                baseline_tail,
                relative_degradation=0.0,
            )
        )
        output_payload = copy.deepcopy(
            candidate_payload if passes else baseline_payload
        )
        output_payload["confirmation_passes"] = passes
        output_payload["selection_only"] = False
        output_payload["teacher_runtime_dependency"] = False
        output_payload["confirmation_parent_sha256"] = sha256_file(
            baseline_path
        )
        output_checkpoint = output_dir / (
            "accepted.pt" if passes else "baseline_retained.pt"
        )
        torch.save(output_payload, output_checkpoint)

        report = {
            "schema": "generic-quintic-compiled-tree-candidate-confirmation-v1",
            "teacher_runtime_dependency": False,
            "configuration": {
                **vars(args),
                "baseline_checkpoint": str(baseline_path),
                "candidate_checkpoint": str(candidate_path),
                "output_dir": str(output_dir),
            },
            "baseline": baseline_statistics,
            "candidate": candidate_statistics,
            "baseline_tail": baseline_tail,
            "candidate_tail": candidate_tail,
            "paired_improvement": paired,
            "passes": passes,
            "models": {
                "baseline_edge_dimensions": list(
                    baseline_model.edge_dimensions
                ),
                "candidate_edge_dimensions": list(
                    candidate_model.edge_dimensions
                ),
                "baseline_real_parameter_count": (
                    baseline_model.trainable_real_parameter_count
                ),
                "candidate_real_parameter_count": (
                    candidate_model.trainable_real_parameter_count
                ),
            },
            "data": {
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "excluded_counts": {
                    name: int(len(values))
                    for name, values in excluded.items()
                },
            },
            "checkpoint": str(output_checkpoint),
            "checkpoint_sha256": sha256_file(output_checkpoint),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "passes": passes,
                "checkpoint": str(output_checkpoint),
            },
        )
        print(status_path.read_text(encoding="utf-8"), flush=True)
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
