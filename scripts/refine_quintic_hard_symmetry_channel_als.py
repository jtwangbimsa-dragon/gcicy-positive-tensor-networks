#!/usr/bin/env python3
"""Fit both factors of one exact-nested hard-symmetry channel by GN/ALS."""

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
from scripts.audit_quintic_hard_symmetry_channel_candidates import (  # noqa: E402
    split_candidate_indices,
)
from scripts.audit_quintic_tn_metric_tangent_reachability import (  # noqa: E402
    MaskedComplexParameterVectorizer,
    MatrixFreeTeacherMetricJacobian,
    capture_fraction,
    residual_metrics,
)
from scripts.audit_quintic_tn_reynolds_derivative_fidelity import (  # noqa: E402
    weighted_paired_summary,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    lanczos_tridiagonal,
    real_inner,
    ridge_dual_from_lanczos,
    vector_norm,
)
from scripts.initialize_quintic_hard_symmetry_channel import (  # noqa: E402
    one_sided_channel_masks,
    seed_one_sided_channel_,
)
from scripts.refine_quintic_hard_symmetry_channel_gn import (  # noqa: E402
    candidate_used_indices,
)
from scripts.refine_quintic_tn_reynolds_metric import teacher_targets  # noqa: E402
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    fixed_fermat_actions_torch,
    tensor_split,
    training_log_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--exclude-report", type=Path, nargs="*", default=())
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--bond-index", type=int, default=5)
    parser.add_argument("--candidate-start", type=int, default=40_000)
    parser.add_argument("--candidate-limit", type=int, default=15_000)
    parser.add_argument("--fit-size", type=int, default=4096)
    parser.add_argument("--selection-size", type=int, default=4096)
    parser.add_argument("--confirmation-size", type=int, default=5000)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--group-samples", type=int, default=64)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-action-batch-size", type=int, default=64)
    parser.add_argument("--operator-chunk-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument(
        "--solver",
        choices=("steepest", "lanczos"),
        default="steepest",
    )
    parser.add_argument("--lanczos-steps", type=int, default=6)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(30.0, 10.0, 3.0, 1.0),
    )
    parser.add_argument(
        "--step-scales",
        type=float,
        nargs="+",
        default=(0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--nonlinear-shortlist", type=int, default=3)
    parser.add_argument("--relative-seed-scale", type=float, default=1.0)
    parser.add_argument(
        "--seed-reference-scale",
        choices=("inherited_rms", "unit_rms"),
        default="unit_rms",
    )
    parser.add_argument("--expected-parameter-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202607248)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.candidate_limit,
        args.fit_size,
        args.selection_size,
        args.confirmation_size,
        args.sweeps,
        args.group_samples,
        args.teacher_batch_size,
        args.teacher_action_batch_size,
        args.operator_chunk_size,
        args.evaluation_batch_size,
        args.lanczos_steps,
        args.nonlinear_shortlist,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all sample, batch, iteration, and shortlist counts are positive")
    if args.candidate_start < 0 or args.expected_parameter_count < 0:
        raise ValueError("candidate start and expected parameter count cannot be negative")
    if any(not np.isfinite(value) or value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be finite and positive")
    if any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in args.step_scales
    ):
        raise ValueError("step scales must lie in (0, 1]")
    if (
        not np.isfinite(args.relative_seed_scale)
        or args.relative_seed_scale <= 0
    ):
        raise ValueError("relative seed scale must be finite and positive")


def parameter_side(
    model: torch.nn.Module,
    *,
    bond_index: int,
    side: str,
    left_mask: torch.Tensor,
    right_mask: torch.Tensor,
) -> tuple[str, torch.Tensor]:
    if side == "left":
        return f"blocked_two_site_orbit_parameters.{bond_index}", left_mask
    if side == "right":
        return f"blocked_two_site_orbit_parameters.{bond_index + 1}", right_mask
    raise ValueError("channel side must be left or right")


def choose_nonlinear_candidate(
    rows: list[dict[str, Any]],
    deltas: list[torch.Tensor],
    *,
    selection_operator: MatrixFreeTeacherMetricJacobian,
    selection_residual: torch.Tensor,
    theta: torch.Tensor,
    shortlist: int,
) -> tuple[dict[str, Any] | None, torch.Tensor | None]:
    """Evaluate only the strongest linearly predicted candidates nonlinearly."""

    ordered = sorted(
        range(len(rows)),
        key=lambda index: rows[index]["selection_linear_capture"],
        reverse=True,
    )
    selected_index = None
    for index in ordered[:shortlist]:
        row = rows[index]
        try:
            actual = selection_operator.residual(theta + deltas[index])
            row["selection_actual_capture"] = capture_fraction(
                selection_residual,
                actual,
            )
            row["selection_actual_residual"] = residual_metrics(actual)
            row["selection_linearization_gap"] = (
                row["selection_linear_capture"]
                - row["selection_actual_capture"]
            )
            row["finite_actual_evaluation"] = True
        except (RuntimeError, FloatingPointError) as error:
            row["selection_actual_capture"] = -1.0e30
            row["finite_actual_evaluation"] = False
            row["evaluation_error"] = str(error)
        if (
            selected_index is None
            or row["selection_actual_capture"]
            > rows[selected_index]["selection_actual_capture"]
        ):
            selected_index = index
    if selected_index is None:
        return None, None
    selected = rows[selected_index]
    if selected["selection_actual_capture"] <= 0:
        return None, None
    return selected, deltas[selected_index]


def logdet_residual_and_positivity(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    target_metric: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[np.ndarray, float, int]:
    """Return unweighted logdet residuals and a metric positivity audit."""

    target_logdet = training_log_volume(target_metric, "cholesky")
    residuals = []
    minimum_eigenvalue = math.inf
    nonpositive = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], batch_size):
            stop = min(start + batch_size, dataset["count"])
            metric = model(
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(torch.min(eigenvalues).detach().cpu()),
            )
            nonpositive += int(
                torch.count_nonzero(eigenvalues <= 0).detach().cpu()
            )
            observed = training_log_volume(metric, "cholesky")
            residuals.append(
                (observed - target_logdet[start:stop]).detach().cpu()
            )
    return (
        torch.cat(residuals).numpy().astype(np.float64),
        minimum_eigenvalue,
        nonpositive,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_model = args.output_model.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    status = output_report.with_suffix(output_report.suffix + ".status.json")
    if output_model.exists() or output_report.exists() or status.exists():
        raise FileExistsError("refusing to overwrite an ALS channel artifact")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(status, {"state": "running", "phase": "loading"})

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model_path = args.expanded_model.expanduser().resolve()
    teacher_path = args.teacher_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    exclude_reports = tuple(path.expanduser().resolve() for path in args.exclude_report)
    for path in (
        model_path,
        teacher_path,
        dataset_path,
        pullbacks_path,
        *exclude_reports,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
    expansion = payload.get("phase_multiplicity_expansion")
    if not isinstance(expansion, dict):
        raise ValueError("input is not an exact multiplicity expansion")
    if not bool(expansion.get("function_preserving_by_construction", False)):
        raise ValueError("expanded model was not function preserving")
    source_multiplicity = int(expansion["source_multiplicity"])
    target_multiplicity = int(expansion["target_multiplicity"])
    source_degree = int(payload.get("source_degree", 1))
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )

    data = np.load(dataset_path, allow_pickle=False)
    candidate_stop = min(
        args.candidate_start + args.candidate_limit,
        len(data["X_val"]),
    )
    candidate_count = candidate_stop - args.candidate_start
    if candidate_count <= 0:
        raise ValueError("candidate index window is empty")
    local_fit, local_selection, local_confirmation = split_candidate_indices(
        candidate_count,
        args.fit_size,
        args.selection_size,
        args.confirmation_size,
        seed=args.seed,
    )
    fit_indices = local_fit + args.candidate_start
    selection_indices = local_selection + args.candidate_start
    confirmation_indices = local_confirmation + args.candidate_start
    used_indices = np.concatenate(
        (fit_indices, selection_indices, confirmation_indices)
    )
    excluded_index_paths = []
    excluded_indices = []
    for report_path in exclude_reports:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        values, indices_path = candidate_used_indices(
            report,
            report_path=report_path,
        )
        excluded_indices.append(values)
        excluded_index_paths.append(indices_path)
    previous_indices = (
        np.unique(np.concatenate(excluded_indices))
        if excluded_indices
        else np.empty(0, dtype=np.int64)
    )
    overlap = np.intersect1d(previous_indices, used_indices, assume_unique=False)
    if overlap.size:
        raise ValueError(
            f"ALS points overlap preceding screens at {overlap.size} indices"
        )

    validation_pullbacks = np.load(pullbacks_path, mmap_mode="r")

    def arrays(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(data["X_val"][indices], dtype=np.float32),
            np.asarray(validation_pullbacks[indices]),
            np.asarray(data["y_val"][indices], dtype=np.float64),
        )

    fit_arrays = arrays(fit_indices)
    selection_arrays = arrays(selection_indices)
    confirmation_arrays = arrays(confirmation_indices)

    target_fit = tensor_split(
        *fit_arrays,
        source_degree=source_degree,
        complex_dtype=torch.complex128,
        real_dtype=torch.float64,
        device=device,
    )
    target_selection = tensor_split(
        *selection_arrays,
        source_degree=source_degree,
        complex_dtype=torch.complex128,
        real_dtype=torch.float64,
        device=device,
    )
    teacher = positive_tensor_network_from_artifact_payload(
        reference_h,
        teacher_payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=torch.complex128)
    teacher.requires_grad_(False).eval()
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.seed + 1009)
    actions = fixed_fermat_actions_torch(
        args.group_samples,
        generator=action_generator,
        complex_dtype=torch.complex128,
        device=device,
    )
    write_json(status, {"state": "running", "phase": "teacher_fit_selection"})
    _, fit_target_metric = teacher_targets(
        teacher,
        target_fit,
        actions,
        batch_size=args.teacher_batch_size,
        action_batch_size=args.teacher_action_batch_size,
    )
    _, selection_target_metric = teacher_targets(
        teacher,
        target_selection,
        actions,
        batch_size=args.teacher_batch_size,
        action_batch_size=args.teacher_action_batch_size,
    )
    del teacher, actions, target_fit, target_selection

    student_dtype = (
        torch.complex64 if str(payload["precision"]) == "complex64" else torch.complex128
    )
    real_dtype = torch.float32 if student_dtype == torch.complex64 else torch.float64
    fit = tensor_split(
        *fit_arrays,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=real_dtype,
        device=device,
    )
    selection = tensor_split(
        *selection_arrays,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=real_dtype,
        device=device,
    )
    fit_target_metric = fit_target_metric.to(dtype=student_dtype)
    selection_target_metric = selection_target_metric.to(dtype=student_dtype)

    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=student_dtype)
    model.force_indexed_block_contraction = True
    parameter_count = int(model.trainable_real_parameter_count)
    if args.expected_parameter_count and parameter_count != args.expected_parameter_count:
        raise RuntimeError(
            f"parameter-count gate failed: {parameter_count} != "
            f"{args.expected_parameter_count}"
        )
    left_mask, right_mask = one_sided_channel_masks(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
    )
    seed_generator = torch.Generator(device=device)
    seed_generator.manual_seed(args.seed + 2003)
    activation = seed_one_sided_channel_(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        relative_scale=args.relative_seed_scale,
        reference_scale=args.seed_reference_scale,
        generator=seed_generator,
    )
    model.requires_grad_(False).eval()

    half_steps = []
    accepted_steps = 0
    stop_early = False
    initial_selection_squared = None
    for sweep in range(args.sweeps):
        for side in ("left", "right"):
            if stop_early:
                break
            write_json(
                status,
                {
                    "state": "running",
                    "phase": "als",
                    "sweep": sweep,
                    "side": side,
                },
            )
            parameter_name, mask = parameter_side(
                model,
                bond_index=args.bond_index,
                side=side,
                left_mask=left_mask,
                right_mask=right_mask,
            )
            parameter = dict(model.named_parameters())[parameter_name]
            parameter.requires_grad_(True)
            vectorizer = MaskedComplexParameterVectorizer.from_module(
                model,
                parameter_name,
                mask.reshape(parameter.shape),
            )
            theta = vectorizer.pack(model)
            model.requires_grad_(False).eval()
            fit_operator = MatrixFreeTeacherMetricJacobian(
                model,
                vectorizer,
                theta,
                fit,
                fit_target_metric,
                mode="logdet",
                chunk_size=args.operator_chunk_size,
            )
            selection_operator = MatrixFreeTeacherMetricJacobian(
                model,
                vectorizer,
                theta,
                selection,
                selection_target_metric,
                mode="logdet",
                chunk_size=args.operator_chunk_size,
            )
            fit_residual = fit_operator.residual()
            selection_residual = selection_operator.residual()
            if initial_selection_squared is None:
                initial_selection_squared = float(
                    real_inner(selection_residual, selection_residual)
                )
            gradient = fit_operator.vjp(fit_residual)
            gradient_norm = float(vector_norm(gradient))
            fit_squared = float(real_inner(fit_residual, fit_residual))
            rayleigh_scale = gradient_norm**2 / max(
                fit_squared,
                np.finfo(float).tiny,
            )
            if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
                half_steps.append(
                    {
                        "sweep": sweep,
                        "side": side,
                        "accepted": False,
                        "reason": "no finite residual-aligned tangent",
                    }
                )
                if side == "left" and accepted_steps == 0:
                    stop_early = True
                continue

            def progress(iteration: int, alpha: float, beta: float) -> None:
                if not args.quiet:
                    print(
                        f"sweep={sweep} side={side} "
                        f"lanczos={iteration}/{args.lanczos_steps} "
                        f"alpha={alpha:.6e} beta={beta:.6e}",
                        flush=True,
                    )

            rows = []
            deltas = []
            lanczos_report = None
            if args.solver == "steepest":
                direction = -gradient / max(
                    gradient_norm,
                    np.finfo(float).tiny,
                )
                unit_tangent = fit_operator.jvp(direction)
                tangent_squared = float(real_inner(unit_tangent, unit_tangent))
                if not np.isfinite(tangent_squared) or tangent_squared <= 0:
                    raise FloatingPointError(
                        "ALS steepest direction has no finite tangent response"
                    )
                optimal_alpha = -float(
                    real_inner(fit_residual, unit_tangent)
                ) / tangent_squared
                candidate_directions = [
                    {
                        "solver": "steepest",
                        "ridge_factor": None,
                        "ridge": None,
                        "full_delta": optimal_alpha * direction,
                        "fit_tangent": optimal_alpha * unit_tangent,
                        "optimal_linear_alpha": optimal_alpha,
                    }
                ]
            else:
                lanczos = lanczos_tridiagonal(
                    fit_operator.normal,
                    fit_residual,
                    steps=args.lanczos_steps,
                    callback=progress,
                )
                lanczos_report = {
                    "steps_completed": int(lanczos.tridiagonal.shape[0]),
                    "breakdown": bool(lanczos.breakdown),
                    "basis_orthogonality_error": lanczos.orthogonality_error,
                }
                candidate_directions = []
                for ridge_factor in args.ridge_factors:
                    ridge = float(ridge_factor * rayleigh_scale)
                    dual = ridge_dual_from_lanczos(lanczos, ridge)
                    candidate_directions.append(
                        {
                            "solver": "lanczos",
                            "ridge_factor": float(ridge_factor),
                            "ridge": ridge,
                            "full_delta": -fit_operator.vjp(dual),
                            "fit_tangent": None,
                            "optimal_linear_alpha": None,
                        }
                    )

            for direction_row in candidate_directions:
                full_delta = direction_row["full_delta"]
                fit_tangent = direction_row["fit_tangent"]
                if fit_tangent is None:
                    fit_tangent = fit_operator.jvp(full_delta)
                selection_tangent = selection_operator.jvp(full_delta)
                for step_scale in args.step_scales:
                    delta = float(step_scale) * full_delta
                    linear_fit = fit_residual + float(step_scale) * fit_tangent
                    linear_selection = (
                        selection_residual
                        + float(step_scale) * selection_tangent
                    )
                    rows.append(
                        {
                            "solver": direction_row["solver"],
                            "ridge_factor": direction_row["ridge_factor"],
                            "ridge": direction_row["ridge"],
                            "optimal_linear_alpha": direction_row[
                                "optimal_linear_alpha"
                            ],
                            "step_scale": float(step_scale),
                            "step_rms": float(
                                torch.sqrt(torch.mean(torch.abs(delta) ** 2))
                            ),
                            "fit_linear_capture": capture_fraction(
                                fit_residual,
                                linear_fit,
                            ),
                            "selection_linear_capture": capture_fraction(
                                selection_residual,
                                linear_selection,
                            ),
                            "fit_linear_residual": residual_metrics(linear_fit),
                            "selection_linear_residual": residual_metrics(
                                linear_selection
                            ),
                        }
                    )
                    deltas.append(delta.detach())
            selected, selected_delta = choose_nonlinear_candidate(
                rows,
                deltas,
                selection_operator=selection_operator,
                selection_residual=selection_residual,
                theta=theta,
                shortlist=args.nonlinear_shortlist,
            )
            step_report = {
                "sweep": sweep,
                "side": side,
                "parameter_name": parameter_name,
                "active_complex_parameter_count": int(theta.numel()),
                "gradient_norm": gradient_norm,
                "residual_rayleigh_scale": rayleigh_scale,
                "lanczos": lanczos_report,
                "fit_baseline": residual_metrics(fit_residual),
                "selection_baseline": residual_metrics(selection_residual),
                "rows": rows,
                "accepted": False,
                "selected": selected,
            }
            if selected is not None and selected_delta is not None:
                fit_actual = fit_operator.residual(theta + selected_delta)
                selected["fit_actual_capture"] = capture_fraction(
                    fit_residual,
                    fit_actual,
                )
                selected["fit_actual_residual"] = residual_metrics(fit_actual)
                if selected["fit_actual_capture"] > 0:
                    vectorizer.commit_(model, theta + selected_delta)
                    accepted_steps += 1
                    step_report["accepted"] = True
                    if not args.quiet:
                        print(
                            f"sweep={sweep} side={side} accepted "
                            f"selection={selected['selection_actual_capture']:.6f}",
                            flush=True,
                        )
                else:
                    step_report["rejection_reason"] = (
                        "nonlinear fit residual did not improve"
                    )
                    if side == "left" and accepted_steps == 0:
                        stop_early = True
            elif side == "left" and accepted_steps == 0:
                stop_early = True
            half_steps.append(step_report)
            del fit_operator, selection_operator
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if stop_early:
            break

    selection_candidate_exists = accepted_steps > 0
    confirmation_opened = bool(selection_candidate_exists)
    confirmation = {
        "state": "skipped",
        "reason": "no ALS half-step improved nonlinear selection residual",
    }
    accepted = False
    if confirmation_opened:
        write_json(
            status,
            {"state": "running", "phase": "one_time_confirmation"},
        )
        target_confirmation = tensor_split(
            *confirmation_arrays,
            source_degree=source_degree,
            complex_dtype=torch.complex128,
            real_dtype=torch.float64,
            device=device,
        )
        teacher = positive_tensor_network_from_artifact_payload(
            reference_h,
            teacher_payload,
            device=device,
            trainable_physical_dictionary=False,
        ).to(device=device, dtype=torch.complex128)
        teacher.requires_grad_(False).eval()
        action_generator = torch.Generator(device=device)
        action_generator.manual_seed(args.seed + 1009)
        actions = fixed_fermat_actions_torch(
            args.group_samples,
            generator=action_generator,
            complex_dtype=torch.complex128,
            device=device,
        )
        _, confirmation_target_metric = teacher_targets(
            teacher,
            target_confirmation,
            actions,
            batch_size=args.teacher_batch_size,
            action_batch_size=args.teacher_action_batch_size,
        )
        del teacher, actions, target_confirmation
        confirmation_data = tensor_split(
            *confirmation_arrays,
            source_degree=source_degree,
            complex_dtype=student_dtype,
            real_dtype=real_dtype,
            device=device,
        )
        confirmation_target_metric = confirmation_target_metric.to(
            dtype=student_dtype
        )
        baseline_model = positive_tensor_network_from_artifact_payload(
            reference_h,
            payload,
            device=device,
            trainable_physical_dictionary=False,
        ).to(device=device, dtype=student_dtype)
        baseline_model.force_indexed_block_contraction = True
        baseline_model.requires_grad_(False).eval()
        before, before_minimum, before_nonpositive = (
            logdet_residual_and_positivity(
                baseline_model,
                confirmation_data,
                confirmation_target_metric,
                batch_size=args.evaluation_batch_size,
            )
        )
        after, after_minimum, after_nonpositive = logdet_residual_and_positivity(
            model,
            confirmation_data,
            confirmation_target_metric,
            batch_size=args.evaluation_batch_size,
        )
        weights = (
            confirmation_data["weights"].detach().cpu().numpy().astype(np.float64)
        )
        weights = weights / np.sum(weights)
        before_squared = float(np.sum(weights * np.square(before)))
        after_squared = float(np.sum(weights * np.square(after)))
        squared_summary = weighted_paired_summary(
            np.square(before),
            np.square(after),
            weights,
        )
        absolute_summary = weighted_paired_summary(
            np.abs(before),
            np.abs(after),
            weights,
        )
        confirmation_capture = 1.0 - after_squared / max(
            before_squared,
            np.finfo(float).tiny,
        )
        accepted = bool(
            confirmation_capture > 0
            and squared_summary["ci95_lower"] > 0
            and absolute_summary["ci95_lower"] > 0
            and after_nonpositive == 0
        )
        confirmation = {
            "state": "evaluated",
            "capture": confirmation_capture,
            "baseline_logdet_rms": math.sqrt(before_squared),
            "candidate_logdet_rms": math.sqrt(after_squared),
            "absolute_residual_paired": absolute_summary,
            "squared_residual_paired": squared_summary,
            "baseline_minimum_metric_eigenvalue": before_minimum,
            "candidate_minimum_metric_eigenvalue": after_minimum,
            "baseline_nonpositive_metric_count": before_nonpositive,
            "candidate_nonpositive_metric_count": after_nonpositive,
            "acceptance_gate_passed": accepted,
        }

    model_saved = False
    if accepted:
        output_payload = dict(payload)
        output_payload["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        output_payload["adaptive_channel_als"] = {
            "schema": "quintic-hard-symmetry-channel-als-v1",
            "source_model": str(model_path),
            "source_model_sha256": sha256_file(model_path),
            "bond_index": args.bond_index,
            "accepted_half_steps": accepted_steps,
            "activation": activation,
            "confirmation": confirmation,
        }
        temporary_model = output_model.with_suffix(output_model.suffix + ".tmp")
        torch.save(output_payload, temporary_model)
        temporary_model.replace(output_model)
        model_saved = True

    indices_path = output_report.with_suffix(output_report.suffix + ".indices.npz")
    np.savez_compressed(
        indices_path,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
        confirmation_indices=confirmation_indices,
    )
    source = {
        "expanded_model": str(model_path),
        "expanded_model_sha256": sha256_file(model_path),
        "teacher_model": str(teacher_path),
        "teacher_model_sha256": sha256_file(teacher_path),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "validation_pullbacks": str(pullbacks_path),
        "validation_pullbacks_sha256": sha256_file(pullbacks_path),
        "indices": str(indices_path),
        "indices_sha256": sha256_file(indices_path),
        "excluded_reports": [str(path) for path in exclude_reports],
        "excluded_index_artifacts": [
            str(path) for path in excluded_index_paths
        ],
    }
    if model_saved:
        source["output_model"] = str(output_model)
        source["output_model_sha256"] = sha256_file(output_model)
    report = {
        "schema": "quintic-hard-symmetry-channel-als-v1",
        "scientific_scope": {
            "purpose": (
                "Alternately optimize both factors of one exact-nested virtual "
                "channel instead of freezing a random right factor."
            ),
            "selection_rule": (
                "Directions are learned on fit points; ridge, step, and ALS "
                "checkpoint decisions use selection points only."
            ),
            "confirmation_rule": (
                "The confirmation set is evaluated once after the ALS candidate "
                "is frozen. Both paired absolute and squared improvements must "
                "have positive 95% lower confidence bounds."
            ),
            "claim_limit": (
                "Passing this bulk gate creates a candidate for a separate tail "
                "audit; it does not itself accept the model for final reporting."
            ),
        },
        "configuration": {
            "device": str(device),
            "bond_index": args.bond_index,
            "candidate_start": args.candidate_start,
            "candidate_limit": candidate_count,
            "fit_size": args.fit_size,
            "selection_size": args.selection_size,
            "confirmation_size": args.confirmation_size,
            "sweeps": args.sweeps,
            "group_samples": args.group_samples,
            "operator_chunk_size": args.operator_chunk_size,
            "evaluation_batch_size": args.evaluation_batch_size,
            "solver": args.solver,
            "lanczos_steps": args.lanczos_steps,
            "ridge_factors": list(args.ridge_factors),
            "step_scales": list(args.step_scales),
            "nonlinear_shortlist": args.nonlinear_shortlist,
            "relative_seed_scale": args.relative_seed_scale,
            "seed_reference_scale": args.seed_reference_scale,
            "source_multiplicity": source_multiplicity,
            "target_multiplicity": target_multiplicity,
            "seed": args.seed,
        },
        "source": source,
        "data_isolation": {
            "preceding_used_index_count": int(previous_indices.size),
            "als_used_index_count": int(used_indices.size),
            "overlap_count": int(overlap.size),
        },
        "parameter_count": parameter_count,
        "activation": activation,
        "accepted_half_steps": accepted_steps,
        "initial_selection_squared_norm": initial_selection_squared,
        "half_steps": half_steps,
        "confirmation_opened": confirmation_opened,
        "confirmation": confirmation,
        "model_saved": model_saved,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(status, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "output_report": str(output_report),
                "output_model": str(output_model) if model_saved else None,
                "bond_index": args.bond_index,
                "accepted_half_steps": accepted_steps,
                "confirmation_opened": confirmation_opened,
                "confirmation_capture": confirmation.get("capture"),
                "accepted": accepted,
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
