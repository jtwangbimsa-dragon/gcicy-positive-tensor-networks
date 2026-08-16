#!/usr/bin/env python3
"""Cross-fit full-batch teacher-metric gradients for a quintic TN."""

from __future__ import annotations

import argparse
import copy
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

from gcicy_metric.pipeline import (  # noqa: E402
    positive_tensor_network_from_artifact_payload,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    ComplexParameterVectorizer,
    real_inner,
    vector_norm,
)
from scripts.distill_quintic_tn_from_full_h_oracle import (  # noqa: E402
    array_slice,
    full_h_teacher_targets,
    load_teacher,
)
from scripts.refine_quintic_tn_reynolds_metric import (  # noqa: E402
    evaluate,
    model_feature_jet,
    save_model,
    whitened_metric_squared,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    tensor_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--teacher-h-artifact", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-degree", type=int, default=10)
    parser.add_argument("--target-degree", type=int, default=30)
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-limit", type=int, default=20000)
    parser.add_argument("--holdout-start", type=int, default=0)
    parser.add_argument("--holdout-limit", type=int, default=20000)
    parser.add_argument("--teacher-feature-batch-size", type=int, default=128)
    parser.add_argument("--teacher-eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument(
        "--relative-steps",
        type=float,
        nargs="+",
        default=(1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3),
    )
    parser.add_argument(
        "--direction-mode",
        choices=("global-gradient", "block-balanced-gradient"),
        default="global-gradient",
    )
    parser.add_argument("--expected-active-parameters", type=int, default=0)
    parser.add_argument("--expected-stored-parameters", type=int, default=0)
    parser.add_argument(
        "--mixed-canonical-center",
        type=int,
        default=-1,
        help="canonicalize around this site before auditing; negative disables it",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.teacher_degree,
        args.target_degree,
        args.fit_limit,
        args.holdout_limit,
        args.teacher_feature_batch_size,
        args.teacher_eval_batch_size,
        args.gradient_batch_size,
        args.eval_batch_size,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("degrees, sample sizes, and batch sizes must be positive")
    if args.target_degree % args.teacher_degree:
        raise ValueError("target degree must be a multiple of teacher degree")
    if args.fit_start < 0 or args.holdout_start < 0:
        raise ValueError("dataset starts cannot be negative")
    if args.mixed_canonical_center < -1:
        raise ValueError("mixed canonical center must be negative or a site index")
    if not args.relative_steps or any(value <= 0 for value in args.relative_steps):
        raise ValueError("relative steps must be positive")
    if args.expected_active_parameters < 0 or args.expected_stored_parameters < 0:
        raise ValueError("expected parameter counts cannot be negative")


def copy_vector_into_model(
    model: torch.nn.Module,
    vectorizer: ComplexParameterVectorizer,
    vector: torch.Tensor,
) -> None:
    replacements = vectorizer.unpack(vector)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in replacements.items():
            named[name].copy_(value)


def metric_loss_and_gradient(
    model: torch.nn.Module,
    vectorizer: ComplexParameterVectorizer,
    dataset: dict[str, Any],
    target_metric: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, torch.Tensor, list[dict[str, Any]]]:
    model.zero_grad(set_to_none=True)
    target_cholesky = torch.linalg.cholesky(target_metric)
    total = 0.0
    for start in range(0, dataset["count"], batch_size):
        stop = min(start + batch_size, dataset["count"])
        _, _, metric = model_feature_jet(
            model,
            dataset["values"][start:stop],
            dataset["derivatives"][start:stop],
        )
        squared = whitened_metric_squared(
            metric,
            target_metric[start:stop],
            target_cholesky[start:stop],
        )
        loss = torch.sum(dataset["weights"][start:stop] * squared)
        loss.backward()
        total += float(loss.detach())

    named = dict(model.named_parameters())
    gradient_parts = []
    rows = []
    for name, count in zip(vectorizer.names, vectorizer.counts, strict=True):
        parameter = named[name]
        if parameter.grad is None:
            raise RuntimeError(f"parameter {name!r} has no oracle gradient")
        part = parameter.grad.detach().reshape(-1).clone()
        if part.numel() != count:
            raise RuntimeError("gradient shape changed during accumulation")
        gradient_parts.append(part)
        rows.append(
            {
                "name": name,
                "complex_parameters": int(count),
                "gradient_norm": float(vector_norm(part)),
            }
        )
    return total, torch.cat(gradient_parts), rows


def gradient_alignment(
    fit_gradient: torch.Tensor,
    holdout_gradient: torch.Tensor,
) -> dict[str, float]:
    fit_norm = float(vector_norm(fit_gradient))
    holdout_norm = float(vector_norm(holdout_gradient))
    inner = float(real_inner(fit_gradient, holdout_gradient))
    denominator = max(fit_norm * holdout_norm, np.finfo(float).tiny)
    return {
        "fit_norm": fit_norm,
        "holdout_norm": holdout_norm,
        "real_inner": inner,
        "cosine": inner / denominator,
    }


def block_balanced_descent_direction(
    theta: torch.Tensor,
    gradient: torch.Tensor,
    counts: tuple[int, ...],
) -> torch.Tensor:
    if theta.shape != gradient.shape or theta.ndim != 1:
        raise ValueError("theta and gradient must be matching vectors")
    if sum(counts) != theta.numel():
        raise ValueError("parameter block counts do not cover the vector")
    rows = []
    offset = 0
    for count in counts:
        theta_part = theta[offset : offset + count]
        gradient_part = gradient[offset : offset + count]
        theta_norm = float(vector_norm(theta_part))
        gradient_norm = float(vector_norm(gradient_part))
        if gradient_norm == 0.0:
            rows.append(torch.zeros_like(gradient_part))
        else:
            rows.append(-gradient_part * (theta_norm / gradient_norm))
        offset += count
    return torch.cat(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_path = args.model.expanduser().resolve()
    teacher_path = args.teacher_h_artifact.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    train_pullbacks_path = pullbacks_dir / "train_pullbacks.npy"
    validation_pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    for path in (
        model_path,
        teacher_path,
        dataset_path,
        train_pullbacks_path,
        validation_pullbacks_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    source_degree = int(payload.get("source_degree", 1))
    student_dtype = (
        torch.complex64
        if str(payload["precision"]) == "complex64"
        else torch.complex128
    )
    student_real_dtype = (
        torch.float32 if student_dtype == torch.complex64 else torch.float64
    )

    data = np.load(dataset_path, allow_pickle=False)
    fit_x = array_slice(
        np.asarray(data["X_train"], dtype=np.float32),
        start=args.fit_start,
        count=args.fit_limit,
    )
    fit_y = array_slice(
        np.asarray(data["y_train"], dtype=np.float64),
        start=args.fit_start,
        count=args.fit_limit,
    )
    holdout_x = array_slice(
        np.asarray(data["X_val"], dtype=np.float32),
        start=args.holdout_start,
        count=args.holdout_limit,
    )
    holdout_y = array_slice(
        np.asarray(data["y_val"], dtype=np.float64),
        start=args.holdout_start,
        count=args.holdout_limit,
    )
    fit_pullbacks = array_slice(
        np.load(train_pullbacks_path, mmap_mode="r"),
        start=args.fit_start,
        count=args.fit_limit,
    )
    holdout_pullbacks = array_slice(
        np.load(validation_pullbacks_path, mmap_mode="r"),
        start=args.holdout_start,
        count=args.holdout_limit,
    )

    write_json(status_path, {"state": "running", "phase": "teacher_targets"})
    teacher, teacher_exponents = load_teacher(
        teacher_path,
        matrix_key="global_h_matrix",
        source_degree=args.teacher_degree,
        target_degree=args.target_degree,
        device=device,
    )
    fit_target_log, fit_target_metric = full_h_teacher_targets(
        teacher,
        fit_x,
        fit_pullbacks,
        teacher_exponents,
        feature_batch_size=args.teacher_feature_batch_size,
        eval_batch_size=args.teacher_eval_batch_size,
        device=device,
    )
    holdout_target_log, holdout_target_metric = full_h_teacher_targets(
        teacher,
        holdout_x,
        holdout_pullbacks,
        teacher_exponents,
        feature_batch_size=args.teacher_feature_batch_size,
        eval_batch_size=args.teacher_eval_batch_size,
        device=device,
    )
    del teacher

    fit = tensor_split(
        fit_x,
        fit_pullbacks,
        fit_y,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=student_real_dtype,
        device=device,
    )
    holdout = tensor_split(
        holdout_x,
        holdout_pullbacks,
        holdout_y,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=student_real_dtype,
        device=device,
    )
    fit_target_log = fit_target_log.to(dtype=student_real_dtype)
    holdout_target_log = holdout_target_log.to(dtype=student_real_dtype)
    fit_target_metric = fit_target_metric.to(dtype=student_dtype)
    holdout_target_metric = holdout_target_metric.to(dtype=student_dtype)

    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=student_dtype)
    canonicalization = None
    if args.mixed_canonical_center >= 0:
        if args.mixed_canonical_center >= model.site_count:
            raise ValueError("mixed canonical center is outside the student chain")
        check_count = min(8, fit["count"])
        with torch.no_grad():
            metric_before = model(
                fit["values"][:check_count],
                fit["derivatives"][:check_count],
            )
        model.mixed_canonicalize_coefficient_cores_(
            args.mixed_canonical_center
        )
        removed_scale = model.normalize_coefficient_chain_scale_(
            site_index=args.mixed_canonical_center
        )
        with torch.no_grad():
            metric_after = model(
                fit["values"][:check_count],
                fit["derivatives"][:check_count],
            )
        difference = torch.linalg.vector_norm(metric_after - metric_before)
        reference = torch.linalg.vector_norm(metric_before)
        relative_difference = float(
            difference / torch.clamp(reference, min=torch.finfo(reference.dtype).tiny)
        )
        tolerance = 2.0e-10 if student_dtype == torch.complex128 else 2.0e-5
        if relative_difference > tolerance:
            raise RuntimeError("mixed canonicalization changed the student metric")
        canonicalization = {
            "center": args.mixed_canonical_center,
            "removed_scale": float(removed_scale),
            "metric_relative_difference": relative_difference,
            "tolerance": tolerance,
        }
    vectorizer = ComplexParameterVectorizer.from_module(model)
    theta = vectorizer.pack(model)
    active_parameters = int(2 * theta.numel())
    fixed_dictionary_parameters = 0
    if payload.get("trainable_physical_dictionary", False):
        dictionary = model.physical_dictionary
        fixed_dictionary_parameters = int(
            dictionary.numel() * (2 if dictionary.is_complex() else 1)
        )
    stored_parameters = active_parameters + fixed_dictionary_parameters
    if (
        args.expected_active_parameters
        and active_parameters != args.expected_active_parameters
    ):
        raise RuntimeError("active parameter-count gate failed")
    if (
        args.expected_stored_parameters
        and stored_parameters != args.expected_stored_parameters
    ):
        raise RuntimeError("stored parameter-count gate failed")

    write_json(status_path, {"state": "running", "phase": "gradients"})
    baseline_fit = evaluate(
        model,
        fit,
        fit_target_log,
        fit_target_metric,
        batch_size=args.eval_batch_size,
    )
    baseline_holdout = evaluate(
        model,
        holdout,
        holdout_target_log,
        holdout_target_metric,
        batch_size=args.eval_batch_size,
    )
    fit_loss, fit_gradient, fit_rows = metric_loss_and_gradient(
        model,
        vectorizer,
        fit,
        fit_target_metric,
        batch_size=args.gradient_batch_size,
    )
    holdout_loss, holdout_gradient, holdout_rows = metric_loss_and_gradient(
        model,
        vectorizer,
        holdout,
        holdout_target_metric,
        batch_size=args.gradient_batch_size,
    )
    alignment = gradient_alignment(fit_gradient, holdout_gradient)

    per_parameter = []
    offset = 0
    for fit_row, holdout_row, count in zip(
        fit_rows, holdout_rows, vectorizer.counts, strict=True
    ):
        fit_part = fit_gradient[offset : offset + count]
        holdout_part = holdout_gradient[offset : offset + count]
        per_parameter.append(
            {
                "name": fit_row["name"],
                "complex_parameters": int(count),
                **gradient_alignment(fit_part, holdout_part),
            }
        )
        offset += count

    theta_norm = float(vector_norm(theta))
    gradient_norm = float(vector_norm(fit_gradient))
    if args.direction_mode == "global-gradient":
        direction = -fit_gradient
        direction_scale = theta_norm / max(
            gradient_norm, np.finfo(float).tiny
        )
    else:
        direction = block_balanced_descent_direction(
            theta,
            fit_gradient,
            vectorizer.counts,
        )
        direction_scale = 1.0
    direction_norm = float(vector_norm(direction))
    candidates = []
    for index, relative_step in enumerate(args.relative_steps, start=1):
        scale = relative_step * direction_scale
        candidate = theta + scale * direction
        copy_vector_into_model(model, vectorizer, candidate)
        fit_metrics = evaluate(
            model,
            fit,
            fit_target_log,
            fit_target_metric,
            batch_size=args.eval_batch_size,
        )
        holdout_metrics = evaluate(
            model,
            holdout,
            holdout_target_log,
            holdout_target_metric,
            batch_size=args.eval_batch_size,
        )
        predicted_holdout_change = scale * float(
            real_inner(holdout_gradient, direction)
        )
        row = {
            "relative_step": float(relative_step),
            "scale": float(scale),
            "actual_parameter_relative_step": float(
                scale * direction_norm / max(theta_norm, np.finfo(float).tiny)
            ),
            "predicted_first_order_holdout_loss_change": (
                predicted_holdout_change
            ),
            "fit": fit_metrics,
            "holdout": holdout_metrics,
        }
        candidates.append(row)
        print(
            f"candidate={index}/{len(args.relative_steps)} "
            f"relative_step={relative_step:.3e} "
            f"holdout_metric_rms="
            f"{holdout_metrics['teacher_whitened_metric_frobenius_rms']:.6e} "
            f"holdout_sigma="
            f"{holdout_metrics['normalized_volume']['sigma_official_formula']:.6e}",
            flush=True,
        )

    baseline_rms = baseline_holdout["teacher_whitened_metric_frobenius_rms"]
    best = min(
        candidates,
        key=lambda row: row["holdout"][
            "teacher_whitened_metric_frobenius_rms"
        ],
    )
    accepted = bool(
        best["holdout"]["teacher_whitened_metric_frobenius_rms"] < baseline_rms
    )
    copy_vector_into_model(model, vectorizer, theta)
    if accepted:
        best_theta = theta + best["scale"] * direction
        copy_vector_into_model(model, vectorizer, best_theta)
        save_model(
            output_dir / "best_tensor_network.pt",
            payload,
            copy.deepcopy(model.state_dict()),
            {
                "schema": "quintic-full-h-oracle-gradient-crossfit-v1",
                "relative_step": best["relative_step"],
                "gradient_alignment": alignment,
                "active_real_parameters": active_parameters,
                "stored_learned_real_parameters": stored_parameters,
            },
        )

    report = {
        "schema": "quintic-full-h-oracle-gradient-crossfit-v1",
        "configuration": {
            **vars(args),
            "model": str(model_path),
            "teacher_h_artifact": str(teacher_path),
            "source_run_dir": str(source_dir),
            "pullbacks_dir": str(pullbacks_dir),
            "output_dir": str(output_dir),
            "device": str(device),
        },
        "source_sha256": {
            "model": sha256_file(model_path),
            "teacher_h_artifact": sha256_file(teacher_path),
            "dataset": sha256_file(dataset_path),
            "train_pullbacks": sha256_file(train_pullbacks_path),
            "validation_pullbacks": sha256_file(validation_pullbacks_path),
        },
        "parameters": {
            "active_real": active_parameters,
            "fixed_learned_dictionary_real": fixed_dictionary_parameters,
            "stored_learned_real": stored_parameters,
            "theta_norm": theta_norm,
        },
        "canonicalization": canonicalization,
        "baseline": {"fit": baseline_fit, "holdout": baseline_holdout},
        "metric_loss": {"fit": fit_loss, "holdout": holdout_loss},
        "gradient_alignment": alignment,
        "direction": {
            "mode": args.direction_mode,
            "norm": direction_norm,
            "scale_per_relative_step": direction_scale,
        },
        "per_parameter_alignment": per_parameter,
        "candidates": candidates,
        "selection": {"accepted": accepted, "best": best},
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "report.json", report)
    write_json(status_path, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "accepted": accepted,
                "gradient_cosine": alignment["cosine"],
                "baseline_holdout_metric_rms": baseline_rms,
                "best_relative_step": best["relative_step"],
                "best_holdout_metric_rms": best["holdout"][
                    "teacher_whitened_metric_frobenius_rms"
                ],
                "best_holdout_sigma": best["holdout"]["normalized_volume"][
                    "sigma_official_formula"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
