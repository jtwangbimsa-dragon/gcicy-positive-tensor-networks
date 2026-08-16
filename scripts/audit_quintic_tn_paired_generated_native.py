#!/usr/bin/env python3
"""Audit a quintic TN candidate on frozen fresh confirmation and tail splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.audit_quintic_tn_paired_native_tail import (  # noqa: E402
    evaluate_model,
    tail_values,
)
from scripts.audit_quintic_tn_reynolds_derivative_fidelity import (  # noqa: E402
    weighted_paired_summary,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    ratio_statistics,
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    tensor_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmation-count", type=int, default=5000)
    parser.add_argument("--tail-count", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-baseline-sha256")
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--expected-points-sha256")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def summarize_split(
    baseline_raw: np.ndarray,
    candidate_raw: np.ndarray,
    baseline_minimum: np.ndarray,
    candidate_minimum: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    resolved_weights = np.asarray(weights, dtype=np.float64)
    resolved_weights = resolved_weights / np.sum(resolved_weights)
    baseline_statistics, baseline_ratio = ratio_statistics(
        baseline_raw,
        resolved_weights,
        baseline_minimum,
    )
    candidate_statistics, candidate_ratio = ratio_statistics(
        candidate_raw,
        resolved_weights,
        candidate_minimum,
    )
    baseline_residual = np.abs(1.0 - baseline_ratio)
    candidate_residual = np.abs(1.0 - candidate_ratio)
    paired = {
        "absolute_residual": weighted_paired_summary(
            baseline_residual,
            candidate_residual,
            resolved_weights,
        ),
        "squared_residual": weighted_paired_summary(
            np.square(baseline_residual),
            np.square(candidate_residual),
            resolved_weights,
        ),
    }
    baseline_tail = tail_values(baseline_statistics)
    candidate_tail = tail_values(candidate_statistics)
    nonworse = {
        key: bool(candidate_tail[key] <= baseline_tail[key])
        for key in baseline_tail
    }
    bulk_gate = bool(
        paired["absolute_residual"]["ci95_lower"] > 0
        and paired["squared_residual"]["ci95_lower"] > 0
        and candidate_statistics["nonpositive_min_eigenvalue"]["count"] == 0
    )
    tail_gate = bool(all(nonworse.values()))
    return {
        "baseline": baseline_statistics,
        "candidate": candidate_statistics,
        "paired_improvement": paired,
        "tail_comparison": {
            "baseline": baseline_tail,
            "candidate": candidate_tail,
            "candidate_nonworse": nonworse,
        },
        "bulk_gate_passed": bulk_gate,
        "tail_gate_passed": tail_gate,
    }


def verify_expected(label: str, observed: str, expected: str | None) -> None:
    if expected is not None and observed != expected:
        raise RuntimeError(f"{label} SHA-256 gate failed: {observed} != {expected}")


def main() -> None:
    args = parse_args()
    if min(args.confirmation_count, args.tail_count, args.batch_size) <= 0:
        raise ValueError("split counts and batch size must be positive")
    output = args.output.expanduser().resolve()
    status = output.with_suffix(output.suffix + ".status.json")
    arrays_path = output.with_suffix(output.suffix + ".arrays.npz")
    if output.exists() or status.exists() or arrays_path.exists():
        raise FileExistsError("refusing to overwrite a generated native audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(status, {"state": "running", "phase": "loading"})

    try:
        device = torch.device(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else ("cpu" if args.device == "auto" else args.device)
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        baseline_path = args.baseline_model.expanduser().resolve()
        candidate_path = args.candidate_model.expanduser().resolve()
        generated_dir = args.generated_dir.expanduser().resolve()
        points_path = generated_dir / "points.npz"
        pullbacks_path = generated_dir / "pullbacks.npy"
        generation_report_path = generated_dir / "report.json"
        for path in (
            baseline_path,
            candidate_path,
            points_path,
            pullbacks_path,
            generation_report_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        baseline_hash = sha256_file(baseline_path)
        candidate_hash = sha256_file(candidate_path)
        points_hash = sha256_file(points_path)
        verify_expected(
            "baseline model",
            baseline_hash,
            args.expected_baseline_sha256,
        )
        verify_expected(
            "candidate model",
            candidate_hash,
            args.expected_candidate_sha256,
        )
        verify_expected("points", points_hash, args.expected_points_sha256)

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
        reference_h = np.asarray(
            baseline_payload["state_dict"]["reference_h"].detach().cpu(),
            dtype=np.complex128,
        )
        candidate_reference_h = np.asarray(
            candidate_payload["state_dict"]["reference_h"].detach().cpu(),
            dtype=np.complex128,
        )
        if not np.allclose(
            reference_h,
            candidate_reference_h,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("baseline and candidate reference H differ")
        precision = (
            "complex128"
            if "complex128"
            in {
                str(baseline_payload["precision"]),
                str(candidate_payload["precision"]),
            }
            else "complex64"
        )
        complex_dtype = (
            torch.complex128 if precision == "complex128" else torch.complex64
        )
        real_dtype = (
            torch.float64 if precision == "complex128" else torch.float32
        )
        source_degree = int(baseline_payload.get("source_degree", 1))

        stored = np.load(points_path, allow_pickle=False)
        points = np.asarray(stored["X"], dtype=np.float32)
        weights = np.asarray(stored["weights"], dtype=np.float64)
        omega = np.asarray(stored["omega_squared"], dtype=np.float64)
        pullbacks = np.load(pullbacks_path, mmap_mode="r")
        count = args.confirmation_count + args.tail_count
        if {
            len(points),
            len(weights),
            len(omega),
            len(pullbacks),
        } != {count}:
            raise RuntimeError(
                "generated arrays must contain exactly confirmation + tail points"
            )
        labels = np.column_stack((weights, omega))
        dataset = tensor_split(
            points,
            np.asarray(pullbacks),
            labels,
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
        baseline.force_indexed_block_contraction = True
        write_json(status, {"state": "running", "phase": "baseline"})
        baseline_raw, baseline_minimum = evaluate_model(
            baseline,
            dataset,
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
        candidate.force_indexed_block_contraction = True
        write_json(status, {"state": "running", "phase": "candidate"})
        candidate_raw, candidate_minimum = evaluate_model(
            candidate,
            dataset,
            batch_size=args.batch_size,
        )

        confirmation_stop = args.confirmation_count
        confirmation = summarize_split(
            baseline_raw[:confirmation_stop],
            candidate_raw[:confirmation_stop],
            baseline_minimum[:confirmation_stop],
            candidate_minimum[:confirmation_stop],
            weights[:confirmation_stop],
        )
        tail = summarize_split(
            baseline_raw[confirmation_stop:],
            candidate_raw[confirmation_stop:],
            baseline_minimum[confirmation_stop:],
            candidate_minimum[confirmation_stop:],
            weights[confirmation_stop:],
        )
        accepted = bool(
            confirmation["bulk_gate_passed"]
            and tail["bulk_gate_passed"]
            and tail["tail_gate_passed"]
        )
        np.savez_compressed(
            arrays_path,
            weights=weights,
            baseline_raw=baseline_raw,
            candidate_raw=candidate_raw,
            baseline_minimum_eigenvalue=baseline_minimum,
            candidate_minimum_eigenvalue=candidate_minimum,
        )
        generation_report = json.loads(
            generation_report_path.read_text(encoding="utf-8")
        )
        report = {
            "schema": "quintic-tn-paired-generated-native-v1",
            "scientific_scope": {
                "purpose": (
                    "One preregistered fresh point set, with the first split used "
                    "once for confirmation and the second split used once for a "
                    "larger bulk-and-tail gate."
                ),
                "limitation": (
                    "Confidence intervals treat generated points as independent; "
                    "the generator does not expose fibre-cluster identifiers."
                ),
            },
            "configuration": {
                "device": str(device),
                "precision": precision,
                "confirmation_count": args.confirmation_count,
                "tail_count": args.tail_count,
                "batch_size": args.batch_size,
            },
            "source": {
                "baseline_model": baseline_path,
                "baseline_model_sha256": baseline_hash,
                "candidate_model": candidate_path,
                "candidate_model_sha256": candidate_hash,
                "points": points_path,
                "points_sha256": points_hash,
                "pullbacks": pullbacks_path,
                "pullbacks_sha256": sha256_file(pullbacks_path),
                "generation_report": generation_report_path,
                "generation_report_sha256": sha256_file(generation_report_path),
                "generation_configuration": generation_report.get(
                    "configuration"
                ),
                "arrays": arrays_path,
                "arrays_sha256": sha256_file(arrays_path),
            },
            "data_isolation": {
                "fit_or_selection_use": False,
                "confirmation_slice": [0, confirmation_stop],
                "tail_slice": [confirmation_stop, count],
                "point_seed_frozen_before_model_evaluation": True,
            },
            "confirmation": confirmation,
            "tail": tail,
            "accepted": accepted,
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output, report)
        write_json(
            status,
            {
                "state": "complete",
                "phase": "complete",
                "accepted": accepted,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "accepted": accepted,
                    "confirmation_baseline_sigma": confirmation["baseline"][
                        "sigma_official_formula"
                    ],
                    "confirmation_candidate_sigma": confirmation["candidate"][
                        "sigma_official_formula"
                    ],
                    "tail_baseline_sigma": tail["baseline"][
                        "sigma_official_formula"
                    ],
                    "tail_candidate_sigma": tail["candidate"][
                        "sigma_official_formula"
                    ],
                    "wall_seconds": report["wall_seconds"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            status,
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
