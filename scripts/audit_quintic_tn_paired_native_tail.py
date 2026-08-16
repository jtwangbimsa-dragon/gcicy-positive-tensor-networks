#!/usr/bin/env python3
"""Compare two quintic TN metrics on identical native MA points and tails."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.audit_quintic_tn_reynolds_derivative_fidelity import (  # noqa: E402
    weighted_paired_summary,
)
from scripts.refine_quintic_hard_symmetry_channel_gn import (  # noqa: E402
    candidate_used_indices,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    ratio_statistics,
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    tensor_split,
    training_log_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--exclude-report", type=Path, nargs="*", default=())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-start", type=int, default=60_000)
    parser.add_argument("--validation-limit", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.validation_start < 0:
        raise ValueError("validation start cannot be negative")
    if args.validation_limit <= 0 or args.batch_size <= 0:
        raise ValueError("validation limit and batch size must be positive")


def evaluate_model(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_rows = []
    minimum_rows = []
    model.requires_grad_(False).eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], batch_size):
            stop = min(start + batch_size, dataset["count"])
            metric = model(
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            logdet = training_log_volume(metric, "cholesky")
            raw_rows.append(
                (
                    logdet - dataset["log_omega"][start:stop]
                ).detach().cpu().numpy().astype(np.float64)
            )
            minimum_rows.append(
                torch.min(eigenvalues, dim=1).values.detach().cpu().numpy().astype(
                    np.float64
                )
            )
    return np.concatenate(raw_rows), np.concatenate(minimum_rows)


def tail_values(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "abs_residual_q999": statistics["abs_residual_weighted_quantiles"][
            "q0.9990"
        ],
        "abs_residual_max": statistics["abs_residual_weighted_quantiles"][
            "q1.0000"
        ],
        "abs_residual_cvar99": statistics["abs_residual_weighted_cvar"][
            "cvar_0.9900"
        ],
        "abs_residual_cvar999": statistics["abs_residual_weighted_cvar"][
            "cvar_0.9990"
        ],
        "ratio_max": statistics["ratio_weighted_quantiles"]["q1.0000"],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output = args.output.expanduser().resolve()
    status = output.with_suffix(output.suffix + ".status.json")
    arrays_path = output.with_suffix(output.suffix + ".arrays.npz")
    indices_path = output.with_suffix(output.suffix + ".indices.npz")
    if (
        output.exists()
        or status.exists()
        or arrays_path.exists()
        or indices_path.exists()
    ):
        raise FileExistsError("refusing to overwrite a paired native-tail audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(status, {"state": "running", "phase": "loading"})

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    baseline_path = args.baseline_model.expanduser().resolve()
    candidate_path = args.candidate_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    exclude_reports = tuple(path.expanduser().resolve() for path in args.exclude_report)
    for path in (
        baseline_path,
        candidate_path,
        dataset_path,
        pullbacks_path,
        *exclude_reports,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

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
    for key in ("source_degree", "site_count", "total_degree"):
        if baseline_payload.get(key) != candidate_payload.get(key):
            raise ValueError(f"baseline and candidate {key} differ")
    source_degree = int(baseline_payload.get("source_degree", 1))
    reference_h = np.asarray(
        baseline_payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    candidate_reference_h = np.asarray(
        candidate_payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    if not np.allclose(reference_h, candidate_reference_h, rtol=0.0, atol=1.0e-12):
        raise ValueError("baseline and candidate reference H differ")
    precision = (
        "complex128"
        if "complex128"
        in {str(baseline_payload["precision"]), str(candidate_payload["precision"])}
        else "complex64"
    )
    complex_dtype = torch.complex128 if precision == "complex128" else torch.complex64
    real_dtype = torch.float64 if precision == "complex128" else torch.float32

    data = np.load(dataset_path, allow_pickle=False)
    validation_stop = args.validation_start + args.validation_limit
    if validation_stop > len(data["X_val"]):
        raise ValueError("requested validation window exceeds the dataset")
    indices = np.arange(args.validation_start, validation_stop, dtype=np.int64)
    excluded_indices = []
    excluded_artifacts = []
    for report_path in exclude_reports:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        values, artifact = candidate_used_indices(
            report,
            report_path=report_path,
        )
        excluded_indices.append(values)
        excluded_artifacts.append(artifact)
    previous = (
        np.unique(np.concatenate(excluded_indices))
        if excluded_indices
        else np.empty(0, dtype=np.int64)
    )
    overlap = np.intersect1d(previous, indices, assume_unique=False)
    if overlap.size:
        raise ValueError(
            f"native-tail audit overlaps preceding development sets at "
            f"{overlap.size} indices"
        )

    validation_pullbacks = np.load(pullbacks_path, mmap_mode="r")[indices]
    validation = tensor_split(
        np.asarray(data["X_val"][indices], dtype=np.float32),
        np.asarray(validation_pullbacks),
        np.asarray(data["y_val"][indices], dtype=np.float64),
        source_degree=source_degree,
        complex_dtype=complex_dtype,
        real_dtype=real_dtype,
        device=device,
    )
    baseline = positive_tensor_network_from_artifact_payload(
        reference_h,
        baseline_payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=complex_dtype)
    write_json(status, {"state": "running", "phase": "baseline"})
    baseline_raw, baseline_minimum = evaluate_model(
        baseline,
        validation,
        batch_size=args.batch_size,
    )
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    candidate = positive_tensor_network_from_artifact_payload(
        reference_h,
        candidate_payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=complex_dtype)
    write_json(status, {"state": "running", "phase": "candidate"})
    candidate_raw, candidate_minimum = evaluate_model(
        candidate,
        validation,
        batch_size=args.batch_size,
    )

    weights = validation["weights_numpy"].astype(np.float64)
    weights = weights / np.sum(weights)
    baseline_statistics, baseline_ratio = ratio_statistics(
        baseline_raw,
        weights,
        baseline_minimum,
    )
    candidate_statistics, candidate_ratio = ratio_statistics(
        candidate_raw,
        weights,
        candidate_minimum,
    )
    baseline_residual = np.abs(1.0 - baseline_ratio)
    candidate_residual = np.abs(1.0 - candidate_ratio)
    paired = {
        "absolute_residual": weighted_paired_summary(
            baseline_residual,
            candidate_residual,
            weights,
        ),
        "squared_residual": weighted_paired_summary(
            np.square(baseline_residual),
            np.square(candidate_residual),
            weights,
        ),
    }
    baseline_tail = tail_values(baseline_statistics)
    candidate_tail = tail_values(candidate_statistics)
    tail_nonworse = {
        key: bool(candidate_tail[key] <= baseline_tail[key])
        for key in baseline_tail
    }
    bulk_gate = bool(
        paired["absolute_residual"]["ci95_lower"] > 0
        and paired["squared_residual"]["ci95_lower"] > 0
        and candidate_statistics["nonpositive_min_eigenvalue"]["count"] == 0
    )
    tail_gate = bool(all(tail_nonworse.values()))

    np.savez_compressed(
        arrays_path,
        weights=weights,
        baseline_raw=baseline_raw,
        candidate_raw=candidate_raw,
        baseline_ratio=baseline_ratio,
        candidate_ratio=candidate_ratio,
        baseline_minimum_eigenvalue=baseline_minimum,
        candidate_minimum_eigenvalue=candidate_minimum,
    )
    np.savez_compressed(indices_path, audit_indices=indices)
    report = {
        "schema": "quintic-tn-paired-native-tail-v1",
        "scientific_scope": {
            "purpose": (
                "Compare the baseline and candidate on identical native "
                "Monge-Ampere points after separate weighted volume normalization."
            ),
            "limitation": (
                "The current dataset has no fibre IDs, so confidence intervals "
                "treat points as independent and are not cluster-bootstrap claims."
            ),
        },
        "configuration": {
            "device": str(device),
            "precision": precision,
            "validation_start": args.validation_start,
            "validation_limit": args.validation_limit,
            "batch_size": args.batch_size,
        },
        "source": {
            "baseline_model": str(baseline_path),
            "baseline_model_sha256": sha256_file(baseline_path),
            "candidate_model": str(candidate_path),
            "candidate_model_sha256": sha256_file(candidate_path),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "validation_pullbacks": str(pullbacks_path),
            "validation_pullbacks_sha256": sha256_file(pullbacks_path),
            "excluded_reports": [str(path) for path in exclude_reports],
            "excluded_index_artifacts": [
                str(path) for path in excluded_artifacts
            ],
            "indices": str(indices_path),
            "indices_sha256": sha256_file(indices_path),
            "arrays": str(arrays_path),
            "arrays_sha256": sha256_file(arrays_path),
        },
        "data_isolation": {
            "preceding_used_index_count": int(previous.size),
            "audit_index_count": int(indices.size),
            "overlap_count": int(overlap.size),
        },
        "baseline": baseline_statistics,
        "candidate": candidate_statistics,
        "paired_improvement": paired,
        "tail_comparison": {
            "baseline": baseline_tail,
            "candidate": candidate_tail,
            "candidate_nonworse": tail_nonworse,
        },
        "bulk_gate_passed": bulk_gate,
        "tail_gate_passed": tail_gate,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output, report)
    write_json(status, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "output": str(output),
                "baseline_sigma": baseline_statistics["sigma_official_formula"],
                "candidate_sigma": candidate_statistics["sigma_official_formula"],
                "baseline_chi": baseline_statistics[
                    "weighted_rms_abs_residual"
                ],
                "candidate_chi": candidate_statistics[
                    "weighted_rms_abs_residual"
                ],
                "bulk_gate_passed": bulk_gate,
                "tail_gate_passed": tail_gate,
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
