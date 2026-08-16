#!/usr/bin/env python3
"""Audit function and derivative fidelity to a Reynolds-projected TN teacher."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    ratio_statistics,
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    fixed_fermat_actions_torch,
    tensor_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-start", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=5000)
    parser.add_argument("--group-samples", type=int, default=64)
    parser.add_argument(
        "--teacher-mode",
        choices=("reynolds", "direct"),
        default="reynolds",
        help=(
            "compare against a Reynolds-projected teacher or evaluate the "
            "stored teacher directly on the same inputs"
        ),
    )
    parser.add_argument(
        "--action-batch-size",
        type=int,
        default=0,
        help="number of group images evaluated together; zero uses all actions",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-seed", type=int, default=202607232)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("validation_limit", "group_samples", "batch_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.action_batch_size < 0:
        raise ValueError("--action-batch-size cannot be negative")
    if args.validation_start < 0:
        raise ValueError("--validation-start cannot be negative")


def reference_normalize(
    model: torch.nn.Module,
    values: torch.Tensor,
    derivatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one common pointwise normalization to a section jet."""

    h_values = torch.einsum("ab,nb->na", model.reference_h, values)
    reference_norm = torch.real(
        torch.einsum("na,na->n", torch.conj(values), h_values)
    )
    if not bool(torch.all(reference_norm > 0)):
        raise FloatingPointError("reference section norm is not positive")
    inverse_scale = torch.rsqrt(reference_norm)
    return (
        values * inverse_scale[:, None],
        derivatives * inverse_scale[:, None, None],
        model.site_count * torch.log(reference_norm),
    )


def jet_from_moments(
    model: torch.nn.Module,
    norm: torch.Tensor,
    gradient: torch.Tensor,
    mixed_hessian: torch.Tensor,
    log_norm_correction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct log F, dK, and ddbar K from linearly composable F moments."""

    if not bool(torch.all(norm > 0)):
        raise FloatingPointError("feature norm is not positive")
    inverse_norm = torch.reciprocal(norm)
    log_feature = torch.log(norm) + log_norm_correction
    potential_gradient = (
        model.target_normalization * gradient * inverse_norm[:, None]
    )
    metric = mixed_hessian * inverse_norm[:, None, None]
    metric -= (
        torch.conj(gradient)[:, :, None]
        * gradient[:, None, :]
        * torch.square(inverse_norm)[:, None, None]
    )
    metric *= model.target_normalization
    metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
    return log_feature, potential_gradient, metric


def model_feature_jet(
    model: torch.nn.Module,
    values: torch.Tensor,
    derivatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_values, normalized_derivatives, correction = reference_normalize(
        model, values, derivatives
    )
    moments = model.feature_moments(normalized_values, normalized_derivatives)
    return jet_from_moments(
        model,
        moments.norm,
        moments.holomorphic_gradient,
        moments.mixed_hessian,
        correction,
    )


def reynolds_feature_jet(
    model: torch.nn.Module,
    values: torch.Tensor,
    derivatives: torch.Tensor,
    actions: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    *,
    action_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average F and both of its jets before constructing the metric."""

    normalized_values, normalized_derivatives, correction = reference_normalize(
        model, values, derivatives
    )
    norm_sum = None
    gradient_sum = None
    mixed_sum = None
    chunk_size = action_chunk_size or len(actions)
    if chunk_size <= 0:
        raise ValueError("action chunk size must be positive")
    batch_size = len(values)
    for action_start in range(0, len(actions), chunk_size):
        action_block = actions[action_start : action_start + chunk_size]
        transformed_values = torch.cat(
            [
                normalized_values[:, permutation] * phases[None, :]
                for permutation, phases in action_block
            ],
            dim=0,
        )
        transformed_derivatives = torch.cat(
            [
                normalized_derivatives[:, permutation, :] * phases[None, :, None]
                for permutation, phases in action_block
            ],
            dim=0,
        )
        moments = model.feature_moments(
            transformed_values,
            transformed_derivatives,
        )
        block_count = len(action_block)
        block_norm = moments.norm.reshape(block_count, batch_size).sum(dim=0)
        block_gradient = moments.holomorphic_gradient.reshape(
            block_count, batch_size, -1
        ).sum(dim=0)
        block_mixed = moments.mixed_hessian.reshape(
            block_count,
            batch_size,
            moments.mixed_hessian.shape[-2],
            moments.mixed_hessian.shape[-1],
        ).sum(dim=0)
        if norm_sum is None:
            norm_sum = block_norm
            gradient_sum = block_gradient
            mixed_sum = block_mixed
        else:
            norm_sum = norm_sum + block_norm
            gradient_sum = gradient_sum + block_gradient
            mixed_sum = mixed_sum + block_mixed
    if norm_sum is None or gradient_sum is None or mixed_sum is None:
        raise ValueError("at least one Reynolds action is required")
    inverse_count = 1.0 / len(actions)
    return jet_from_moments(
        model,
        norm_sum * inverse_count,
        gradient_sum * inverse_count,
        mixed_sum * inverse_count,
        correction,
    )


def invariant_derivative_errors(
    student_gradient: torch.Tensor,
    student_metric: torch.Tensor,
    teacher_gradient: torch.Tensor,
    teacher_metric: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return coordinate-invariant first/second derivative errors and spectra."""

    teacher_eigenvalues = torch.linalg.eigvalsh(teacher_metric)
    student_eigenvalues = torch.linalg.eigvalsh(student_metric)
    if not bool(torch.all(teacher_eigenvalues > 0)):
        minimum = float(torch.min(teacher_eigenvalues).detach().cpu())
        count = int(torch.count_nonzero(teacher_eigenvalues <= 0).detach().cpu())
        raise FloatingPointError(
            "Reynolds teacher metric is not positive: "
            f"minimum={minimum:.9e}, nonpositive_eigenvalues={count}"
        )
    if not bool(torch.all(student_eigenvalues > 0)):
        minimum = float(torch.min(student_eigenvalues).detach().cpu())
        count = int(torch.count_nonzero(student_eigenvalues <= 0).detach().cpu())
        raise FloatingPointError(
            "student metric is not positive: "
            f"minimum={minimum:.9e}, nonpositive_eigenvalues={count}"
        )

    gradient_delta = student_gradient - teacher_gradient
    solved_gradient = torch.linalg.solve(
        teacher_metric,
        gradient_delta.unsqueeze(-1),
    ).squeeze(-1)
    gradient_squared = torch.real(
        torch.sum(torch.conj(gradient_delta) * solved_gradient, dim=1)
    ).clamp_min(0.0)

    metric_delta = student_metric - teacher_metric
    relative_metric = torch.linalg.solve(teacher_metric, metric_delta)
    metric_squared = torch.real(
        torch.einsum("nij,nji->n", relative_metric, relative_metric)
    ).clamp_min(0.0)
    student_logdet = torch.sum(torch.log(student_eigenvalues), dim=1)
    teacher_logdet = torch.sum(torch.log(teacher_eigenvalues), dim=1)
    return (
        gradient_squared,
        metric_squared,
        student_logdet,
        teacher_logdet,
        torch.min(student_eigenvalues, dim=1).values,
    )


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(float(np.sum(weights * np.square(values))))


def weighted_paired_summary(
    before: np.ndarray,
    after: np.ndarray,
    weights: np.ndarray,
    *,
    z_score: float = 1.959963984540054,
) -> dict[str, float]:
    """Summarize paired before-minus-after improvement with a weighted CI."""

    delta = np.asarray(before, dtype=np.float64) - np.asarray(after, dtype=np.float64)
    resolved_weights = np.asarray(weights, dtype=np.float64)
    resolved_weights = resolved_weights / np.sum(resolved_weights)
    mean = float(np.sum(resolved_weights * delta))
    sum_squared_weights = float(np.sum(np.square(resolved_weights)))
    effective_count = 1.0 / sum_squared_weights
    denominator = max(1.0 - sum_squared_weights, np.finfo(float).tiny)
    variance = float(
        np.sum(resolved_weights * np.square(delta - mean)) / denominator
    )
    standard_error = math.sqrt(max(0.0, variance / effective_count))
    return {
        "mean_improvement": mean,
        "standard_error": standard_error,
        "ci95_lower": mean - z_score * standard_error,
        "ci95_upper": mean + z_score * standard_error,
        "effective_sample_count": effective_count,
        "weighted_fraction_improved": float(
            np.sum(resolved_weights * (delta > 0.0))
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    started = time.perf_counter()
    write_json(status_path, {"state": "running", "phase": "loading"})

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    student_path = args.student_model.expanduser().resolve()
    teacher_path = args.teacher_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    validation_pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    for path in (
        student_path,
        teacher_path,
        dataset_path,
        validation_pullbacks_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    student_payload = torch.load(student_path, map_location="cpu", weights_only=False)
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
    student_saved_precision = str(student_payload["precision"])
    teacher_saved_precision = str(teacher_payload["precision"])
    precision = (
        "complex128"
        if "complex128" in {student_saved_precision, teacher_saved_precision}
        else "complex64"
    )
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    real_dtype = torch.float32 if precision == "complex64" else torch.float64
    source_degree = int(student_payload.get("source_degree", 1))
    metadata_keys = ("source_degree", "site_count", "total_degree")
    for key in metadata_keys:
        student_value = int(student_payload.get(key, 1 if key == "source_degree" else -1))
        teacher_value = int(teacher_payload.get(key, 1 if key == "source_degree" else -1))
        if student_value != teacher_value:
            raise ValueError(f"student and teacher {key} differ")

    reference_h = np.asarray(
        student_payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    teacher_reference_h = np.asarray(
        teacher_payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    if not np.allclose(reference_h, teacher_reference_h, rtol=0.0, atol=1.0e-12):
        raise ValueError("student and teacher reference H differ")
    if not np.allclose(reference_h, np.eye(5), rtol=0.0, atol=1.0e-12):
        raise ValueError("Fermat Reynolds audit requires the invariant FS reference H")

    student = positive_tensor_network_from_artifact_payload(
        reference_h,
        student_payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=complex_dtype)
    teacher = positive_tensor_network_from_artifact_payload(
        reference_h,
        teacher_payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=complex_dtype)
    student.requires_grad_(False).eval()
    teacher.requires_grad_(False).eval()
    if not math.isclose(
        float(student.target_normalization),
        float(teacher.target_normalization),
        rel_tol=1.0e-12,
        abs_tol=1.0e-14,
    ):
        raise ValueError("student and teacher potential normalizations differ")

    data = np.load(dataset_path, allow_pickle=False)
    validation_stop = args.validation_start + args.validation_limit
    if validation_stop > len(data["X_val"]):
        raise ValueError("requested validation window exceeds the dataset")
    validation_slice = slice(args.validation_start, validation_stop)
    validation_x = np.asarray(data["X_val"][validation_slice], dtype=np.float32)
    validation_y = np.asarray(data["y_val"][validation_slice], dtype=np.float64)
    validation_pullbacks = np.load(validation_pullbacks_path, mmap_mode="r")[
        validation_slice
    ]
    validation = tensor_split(
        validation_x,
        validation_pullbacks,
        validation_y,
        source_degree=source_degree,
        complex_dtype=complex_dtype,
        real_dtype=real_dtype,
        device=device,
    )
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.torch_seed + 1009)
    actions = (
        fixed_fermat_actions_torch(
            args.group_samples,
            generator=action_generator,
            complex_dtype=complex_dtype,
            device=device,
        )
        if args.teacher_mode == "reynolds"
        else ()
    )

    delta_log_feature_rows: list[np.ndarray] = []
    gradient_squared_rows: list[np.ndarray] = []
    metric_squared_rows: list[np.ndarray] = []
    logdet_delta_rows: list[np.ndarray] = []
    student_raw_rows: list[np.ndarray] = []
    teacher_raw_rows: list[np.ndarray] = []
    student_minimum_rows: list[np.ndarray] = []
    teacher_minimum_rows: list[np.ndarray] = []

    write_json(status_path, {"state": "running", "phase": "derivative_audit"})
    with torch.no_grad():
        for start in range(0, validation["count"], args.batch_size):
            stop = min(start + args.batch_size, validation["count"])
            values = validation["values"][start:stop]
            derivatives = validation["derivatives"][start:stop]
            student_log_feature, student_gradient, student_metric = model_feature_jet(
                student, values, derivatives
            )
            if args.teacher_mode == "reynolds":
                teacher_log_feature, teacher_gradient, teacher_metric = (
                    reynolds_feature_jet(
                        teacher,
                        values,
                        derivatives,
                        actions,
                        action_chunk_size=args.action_batch_size or len(actions),
                    )
                )
            else:
                teacher_log_feature, teacher_gradient, teacher_metric = (
                    model_feature_jet(teacher, values, derivatives)
                )
            (
                gradient_squared,
                metric_squared,
                student_logdet,
                teacher_logdet,
                student_minimum,
            ) = invariant_derivative_errors(
                student_gradient,
                student_metric,
                teacher_gradient,
                teacher_metric,
            )
            teacher_minimum = torch.min(
                torch.linalg.eigvalsh(teacher_metric), dim=1
            ).values
            log_omega = validation["log_omega"][start:stop]

            def as_numpy(row: torch.Tensor) -> np.ndarray:
                return row.detach().cpu().numpy().astype(np.float64)

            delta_log_feature_rows.append(
                as_numpy(student_log_feature - teacher_log_feature)
            )
            gradient_squared_rows.append(as_numpy(gradient_squared))
            metric_squared_rows.append(as_numpy(metric_squared))
            logdet_delta_rows.append(as_numpy(student_logdet - teacher_logdet))
            student_raw_rows.append(as_numpy(student_logdet - log_omega))
            teacher_raw_rows.append(as_numpy(teacher_logdet - log_omega))
            student_minimum_rows.append(as_numpy(student_minimum))
            teacher_minimum_rows.append(as_numpy(teacher_minimum))
            if stop == validation["count"] or stop % (10 * args.batch_size) == 0:
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "derivative_audit",
                        "completed": stop,
                        "total": validation["count"],
                    },
                )

    weights = validation["weights_numpy"]
    delta_log_feature = np.concatenate(delta_log_feature_rows)
    centered_delta_log_feature = delta_log_feature - float(
        np.sum(weights * delta_log_feature)
    )
    gradient_squared = np.concatenate(gradient_squared_rows)
    metric_squared = np.concatenate(metric_squared_rows)
    logdet_delta = np.concatenate(logdet_delta_rows)
    student_raw = np.concatenate(student_raw_rows)
    teacher_raw = np.concatenate(teacher_raw_rows)
    student_minimum = np.concatenate(student_minimum_rows)
    teacher_minimum = np.concatenate(teacher_minimum_rows)
    potential_scale = float(student.target_normalization)

    fidelity = {
        "centered_log_feature_rms_L0": weighted_rms(
            centered_delta_log_feature, weights
        ),
        "centered_potential_rms_L0": potential_scale
        * weighted_rms(centered_delta_log_feature, weights),
        "teacher_metric_gradient_rms_L1": math.sqrt(
            float(np.sum(weights * gradient_squared))
        ),
        "teacher_whitened_metric_frobenius_rms_L2": math.sqrt(
            float(np.sum(weights * metric_squared))
        ),
        "log_determinant_rms_Ldet": weighted_rms(logdet_delta, weights),
        "log_determinant_max_abs": float(np.max(np.abs(logdet_delta))),
    }
    student_volume, _ = ratio_statistics(student_raw, weights, student_minimum)
    teacher_volume, _ = ratio_statistics(teacher_raw, weights, teacher_minimum)
    student_ratio = np.exp(
        student_raw - float(student_volume["log_mean_unnormalized_ratio"])
    )
    teacher_ratio = np.exp(
        teacher_raw - float(teacher_volume["log_mean_unnormalized_ratio"])
    )
    paired_native_volume = {
        "absolute_residual": weighted_paired_summary(
            np.abs(1.0 - teacher_ratio),
            np.abs(1.0 - student_ratio),
            weights,
        ),
        "squared_residual": weighted_paired_summary(
            np.square(1.0 - teacher_ratio),
            np.square(1.0 - student_ratio),
            weights,
        ),
    }
    report = {
        "schema": "quintic-tn-reynolds-derivative-fidelity-v1",
        "configuration": {
            "student_model": str(student_path),
            "student_model_sha256": sha256_file(student_path),
            "teacher_model": str(teacher_path),
            "teacher_model_sha256": sha256_file(teacher_path),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "validation_pullbacks": str(validation_pullbacks_path),
            "validation_pullbacks_sha256": sha256_file(
                validation_pullbacks_path
            ),
            "validation_points": validation["count"],
            "validation_start": args.validation_start,
            "teacher_mode": args.teacher_mode,
            "group_samples": len(actions),
            "action_batch_size": (
                args.action_batch_size or len(actions) if actions else 0
            ),
            "batch_size": args.batch_size,
            "torch_seed": args.torch_seed,
            "student_saved_precision": student_saved_precision,
            "teacher_saved_precision": teacher_saved_precision,
            "evaluation_precision": precision,
            "device": str(device),
        },
        "fidelity": fidelity,
        "student_normalized_volume": student_volume,
        "reynolds_teacher_normalized_volume": teacher_volume,
        "paired_native_volume_improvement": paired_native_volume,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "report.json", report)
    write_json(status_path, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "fidelity": fidelity,
                "student_sigma": student_volume["sigma_official_formula"],
                "reynolds_teacher_sigma": teacher_volume[
                    "sigma_official_formula"
                ],
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
