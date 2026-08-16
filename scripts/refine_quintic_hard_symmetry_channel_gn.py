#!/usr/bin/env python3
"""Fit one cross-fit-selected hard-symmetry channel with damped GN/LM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.audit_quintic_tn_metric_tangent_reachability import (  # noqa: E402
    MaskedComplexParameterVectorizer,
    MatrixFreeTeacherMetricJacobian,
    capture_fraction,
    residual_metrics,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    lanczos_tridiagonal,
    real_inner,
    ridge_dual_from_lanczos,
    select_disjoint_indices,
    vector_norm,
)
from scripts.initialize_quintic_hard_symmetry_channel import (  # noqa: E402
    one_sided_channel_masks,
    seed_one_sided_channel_,
)
from scripts.refine_quintic_tn_reynolds_metric import teacher_targets  # noqa: E402
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    fixed_fermat_actions_torch,
    tensor_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-model", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--candidate-start", type=int, default=10_000)
    parser.add_argument("--candidate-limit", type=int, default=10_000)
    parser.add_argument("--fit-size", type=int, default=4096)
    parser.add_argument("--selection-size", type=int, default=4096)
    parser.add_argument("--group-samples", type=int, default=64)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-action-batch-size", type=int, default=64)
    parser.add_argument("--operator-chunk-size", type=int, default=4)
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
        default=(0.25, 0.5, 1.0),
    )
    parser.add_argument("--slq-probes", type=int, default=2)
    parser.add_argument("--slq-steps", type=int, default=6)
    parser.add_argument("--expected-parameter-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202607247)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.candidate_limit,
        args.fit_size,
        args.selection_size,
        args.group_samples,
        args.teacher_batch_size,
        args.teacher_action_batch_size,
        args.operator_chunk_size,
        args.lanczos_steps,
        args.slq_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sample, batch, chunk, and iteration counts must be positive")
    if args.candidate_start < 0 or args.expected_parameter_count < 0:
        raise ValueError("candidate start and expected parameter count cannot be negative")
    if args.slq_probes < 0:
        raise ValueError("SLQ probe count cannot be negative")
    if any(not np.isfinite(value) or value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be finite and positive")
    if any(not np.isfinite(value) or not 0 < value <= 1 for value in args.step_scales):
        raise ValueError("step scales must lie in (0, 1]")


def selected_candidate(report: dict[str, Any]) -> dict[str, Any]:
    candidate = report.get("selected_candidate")
    if candidate is None:
        candidate = report.get("probe_selected_candidate")
    if not isinstance(candidate, dict):
        raise ValueError("candidate report contains no selected candidate")
    return candidate


def candidate_used_indices(
    report: dict[str, Any],
    *,
    report_path: Path,
) -> tuple[np.ndarray, Path]:
    """Load every point index consumed by the preceding candidate screen."""

    source = report.get("source", {})
    configured = source.get("indices") if isinstance(source, dict) else None
    indices_path = (
        Path(str(configured)).expanduser()
        if configured
        else report_path.with_suffix(report_path.suffix + ".indices.npz")
    )
    if not indices_path.is_absolute():
        indices_path = (report_path.parent / indices_path).resolve()
    if not indices_path.exists():
        raise FileNotFoundError(f"candidate index artifact is missing: {indices_path}")
    expected_hash = source.get("indices_sha256") if isinstance(source, dict) else None
    if expected_hash and str(expected_hash) != sha256_file(indices_path):
        raise ValueError("candidate index artifact hash does not match its report")
    with np.load(indices_path, allow_pickle=False) as archive:
        arrays = [
            np.asarray(archive[name], dtype=np.int64).reshape(-1)
            for name in archive.files
            if name.endswith("_indices")
        ]
    if not arrays:
        raise ValueError("candidate index artifact contains no index arrays")
    return np.unique(np.concatenate(arrays)), indices_path


def actual_capture(
    operator: MatrixFreeTeacherMetricJacobian,
    baseline: torch.Tensor,
    parameter_vector: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    residual = operator.residual(parameter_vector)
    return capture_fraction(baseline, residual), residual_metrics(residual)


def slq_effective_dimension(
    operator: Callable[[torch.Tensor], torch.Tensor],
    *,
    output_count: int,
    ridge: float,
    probes: int,
    steps: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    quiet: bool,
) -> dict[str, Any] | None:
    """Estimate tr[A(A+ridge I)^-1] for A=JJ^T by stochastic Lanczos."""

    if probes == 0:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    estimates = []
    diagnostics = []
    for probe_index in range(probes):
        signs = torch.randint(
            0,
            2,
            (output_count,),
            generator=generator,
            dtype=torch.int64,
            device=device,
        )
        initial = (2 * signs - 1).to(dtype=dtype)

        def progress(iteration: int, alpha: float, beta: float) -> None:
            if not quiet:
                print(
                    f"slq={probe_index + 1}/{probes} "
                    f"lanczos={iteration}/{steps} alpha={alpha:.6e} "
                    f"beta={beta:.6e}",
                    flush=True,
                )

        result = lanczos_tridiagonal(
            operator,
            initial,
            steps=steps,
            callback=progress,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(result.tridiagonal)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        weights = torch.square(eigenvectors[0])
        estimate = float(
            output_count
            * torch.sum(weights * eigenvalues / (eigenvalues + ridge))
        )
        estimates.append(estimate)
        diagnostics.append(
            {
                "estimate": estimate,
                "steps_completed": int(result.tridiagonal.shape[0]),
                "breakdown": bool(result.breakdown),
                "basis_orthogonality_error": result.orthogonality_error,
            }
        )
    mean = float(np.mean(estimates))
    standard_error = (
        float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
        if len(estimates) > 1
        else 0.0
    )
    return {
        "ridge": float(ridge),
        "estimate": mean,
        "standard_error": standard_error,
        "fit_points_per_effective_dimension": (
            float(output_count / mean) if mean > 0 else math.inf
        ),
        "probes": probes,
        "steps": steps,
        "probe_diagnostics": diagnostics,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_model = args.output_model.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    status = output_report.with_suffix(output_report.suffix + ".status.json")
    if output_model.exists() or output_report.exists() or status.exists():
        raise FileExistsError("refusing to overwrite a channel GN artifact")
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
    candidate_path = args.candidate_report.expanduser().resolve()
    teacher_path = args.teacher_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    for path in (
        model_path,
        candidate_path,
        teacher_path,
        dataset_path,
        pullbacks_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
    with candidate_path.open("r", encoding="utf-8") as handle:
        candidate_report = json.load(handle)
    candidate = selected_candidate(candidate_report)
    if candidate_report["source"]["expanded_model_sha256"] != sha256_file(model_path):
        raise ValueError("candidate report and expanded model hashes differ")
    preflight_confirmation = candidate.get("holdout_confirmation", {})
    if float(candidate.get("candidate_rank_score", -math.inf)) <= 0:
        raise ValueError("candidate did not improve its independent selection set")
    if candidate_report.get("confirmation_opened") is False:
        raise ValueError("candidate report explicitly skipped confirmation")
    if float(preflight_confirmation.get("linear_capture", -math.inf)) <= 0:
        raise ValueError("candidate did not pass its one-time linear preflight")
    previous_indices, previous_indices_path = candidate_used_indices(
        candidate_report,
        report_path=candidate_path,
    )

    expansion = payload.get("phase_multiplicity_expansion")
    if not isinstance(expansion, dict):
        raise ValueError("input model is not an exact multiplicity expansion")
    source_multiplicity = int(expansion["source_multiplicity"])
    target_multiplicity = int(expansion["target_multiplicity"])
    source_degree = int(payload.get("source_degree", 1))
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )

    data = np.load(dataset_path, allow_pickle=False)
    candidate_stop = min(args.candidate_start + args.candidate_limit, len(data["X_val"]))
    candidate_count = candidate_stop - args.candidate_start
    if candidate_count <= 0:
        raise ValueError("candidate index window is empty")
    local_fit, local_selection = select_disjoint_indices(
        candidate_count,
        args.fit_size,
        args.selection_size,
        seed=args.seed,
    )
    fit_indices = local_fit + args.candidate_start
    selection_indices = local_selection + args.candidate_start
    stage_indices = np.concatenate((fit_indices, selection_indices))
    overlap = np.intersect1d(previous_indices, stage_indices, assume_unique=False)
    if overlap.size:
        raise ValueError(
            "GN fit/selection points overlap the preceding candidate screen: "
            f"{overlap.size} duplicated indices"
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
    write_json(status, {"state": "running", "phase": "teacher_targets"})
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
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    bond_index = int(candidate["bond_index"])
    candidate_seed = int(candidate["candidate_seed"])
    preflight_configuration = candidate_report["configuration"]
    seed_generator = torch.Generator(device=device)
    seed_generator.manual_seed(candidate_seed)
    activation = seed_one_sided_channel_(
        model,
        bond_index=bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        relative_scale=float(preflight_configuration["relative_seed_scale"]),
        reference_scale=str(
            preflight_configuration.get("seed_reference_scale", "inherited_rms")
        ),
        generator=seed_generator,
    )
    left_mask, _ = one_sided_channel_masks(
        model,
        bond_index=bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
    )
    parameter_name = f"blocked_two_site_orbit_parameters.{bond_index}"
    parameter = dict(model.named_parameters())[parameter_name]
    vectorizer = MaskedComplexParameterVectorizer.from_module(
        model,
        parameter_name,
        left_mask.reshape(parameter.shape),
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
    write_json(status, {"state": "running", "phase": "gn_lanczos"})
    fit_residual = fit_operator.residual()
    selection_residual = selection_operator.residual()
    gradient = fit_operator.vjp(fit_residual)
    gradient_norm = float(vector_norm(gradient))
    fit_squared = float(real_inner(fit_residual, fit_residual))
    rayleigh_scale = gradient_norm**2 / max(fit_squared, np.finfo(float).tiny)
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        raise FloatingPointError("selected channel has no finite GN scale")

    def gn_progress(iteration: int, alpha: float, beta: float) -> None:
        if not args.quiet:
            print(
                f"gn_lanczos={iteration}/{args.lanczos_steps} "
                f"alpha={alpha:.6e} beta={beta:.6e}",
                flush=True,
            )

    rows = []
    deltas = []
    if args.solver == "steepest":
        direction = -gradient / max(gradient_norm, np.finfo(float).tiny)
        unit_tangent = fit_operator.jvp(direction)
        tangent_squared = float(real_inner(unit_tangent, unit_tangent))
        if not np.isfinite(tangent_squared) or tangent_squared <= 0:
            raise FloatingPointError(
                "selected channel has no finite steepest tangent response"
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
        lanczos_report = None
    else:
        lanczos = lanczos_tridiagonal(
            fit_operator.normal,
            fit_residual,
            steps=args.lanczos_steps,
            callback=gn_progress,
        )
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
        lanczos_report = {
            "steps_completed": int(lanczos.tridiagonal.shape[0]),
            "breakdown": bool(lanczos.breakdown),
            "basis_orthogonality_error": lanczos.orthogonality_error,
        }

    for direction_row in candidate_directions:
        full_delta = direction_row["full_delta"]
        fit_tangent = direction_row["fit_tangent"]
        if fit_tangent is None:
            fit_tangent = fit_operator.jvp(full_delta)
        selection_tangent = selection_operator.jvp(full_delta)
        for step_scale in args.step_scales:
            delta = float(step_scale) * full_delta
            step_rms = float(torch.sqrt(torch.mean(torch.abs(delta) ** 2)))
            inherited_rms = max(
                float(activation["adjacent_inherited_rms"]),
                np.finfo(float).tiny,
            )
            linear_fit = fit_residual + float(step_scale) * fit_tangent
            linear_selection = (
                selection_residual + float(step_scale) * selection_tangent
            )
            row: dict[str, Any] = {
                "solver": direction_row["solver"],
                "ridge_factor": direction_row["ridge_factor"],
                "ridge": direction_row["ridge"],
                "optimal_linear_alpha": direction_row[
                    "optimal_linear_alpha"
                ],
                "step_scale": float(step_scale),
                "step_rms": step_rms,
                "step_rms_over_adjacent_inherited_rms": step_rms / inherited_rms,
                "fit_linear_capture": capture_fraction(fit_residual, linear_fit),
                "selection_linear_capture": capture_fraction(
                    selection_residual, linear_selection
                ),
                "fit_linear_residual": residual_metrics(linear_fit),
                "selection_linear_residual": residual_metrics(linear_selection),
            }
            try:
                fit_actual, fit_actual_metrics = actual_capture(
                    fit_operator,
                    fit_residual,
                    theta + delta,
                )
                selection_actual, selection_actual_metrics = actual_capture(
                    selection_operator,
                    selection_residual,
                    theta + delta,
                )
                row.update(
                    {
                        "fit_actual_capture": fit_actual,
                        "selection_actual_capture": selection_actual,
                        "fit_actual_residual": fit_actual_metrics,
                        "selection_actual_residual": selection_actual_metrics,
                        "selection_linearization_gap": (
                            row["selection_linear_capture"] - selection_actual
                        ),
                        "positive_metric": True,
                    }
                )
            except (RuntimeError, FloatingPointError) as error:
                row.update(
                    {
                        "fit_actual_capture": -1.0e30,
                        "selection_actual_capture": -1.0e30,
                        "positive_metric": False,
                        "evaluation_error": str(error),
                    }
                )
            rows.append(row)
            deltas.append(delta.detach().clone())
            if not args.quiet:
                print(
                    f"solver={row['solver']} "
                    f"ridge_factor={row['ridge_factor']} step={step_scale:g} "
                    f"selection_linear={row['selection_linear_capture']:.4f} "
                    f"selection_actual={row['selection_actual_capture']:.4f}",
                    flush=True,
                )

    viable = [
        index
        for index, row in enumerate(rows)
        if row["fit_actual_capture"] > 0
        and row["selection_actual_capture"] > 0
        and row["positive_metric"]
    ]
    if not viable:
        raise RuntimeError(
            "no channel candidate improved both nonlinear fit and selection residuals"
        )
    selected_index = max(
        viable,
        key=lambda index: rows[index]["selection_actual_capture"],
    )
    selected = rows[selected_index]
    best_delta = deltas[selected_index]
    write_json(status, {"state": "running", "phase": "effective_dimension"})
    effective_dimension = (
        {
            "estimate": 1.0,
            "interpretation": (
                "The accepted preconditioner exposes one fit-derived steepest "
                "direction; no multi-direction ridge solve was used."
            ),
        }
        if args.solver == "steepest"
        else slq_effective_dimension(
            fit_operator.normal,
            output_count=fit_operator.output_count,
            ridge=float(selected["ridge"]),
            probes=args.slq_probes,
            steps=args.slq_steps,
            seed=args.seed + 3001,
            dtype=fit_residual.dtype,
            device=device,
            quiet=args.quiet,
        )
    )

    selected_vector = theta + best_delta
    vectorizer.commit_(model, selected_vector)
    output_payload = dict(payload)
    output_payload["state_dict"] = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    output_payload["adaptive_channel_gn"] = {
        "schema": "quintic-hard-symmetry-channel-gn-v1",
        "source_model": str(model_path),
        "source_model_sha256": sha256_file(model_path),
        "candidate_report": str(candidate_path),
        "candidate_report_sha256": sha256_file(candidate_path),
        "bond_index": bond_index,
        "candidate_seed": candidate_seed,
        "fit_size": args.fit_size,
        "selection_size": args.selection_size,
        "selected_hyperparameters": selected,
        "effective_dimension": effective_dimension,
        "activation": activation,
    }
    temporary_model = output_model.with_suffix(output_model.suffix + ".tmp")
    torch.save(output_payload, temporary_model)
    temporary_model.replace(output_model)

    indices_path = output_report.with_suffix(output_report.suffix + ".indices.npz")
    np.savez_compressed(
        indices_path,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
    )
    report = {
        "schema": "quintic-hard-symmetry-channel-gn-selection-v1",
        "scientific_scope": {
            "selection_rule": (
                "Fit GN directions on fit points and select ridge/step only on "
                "independent selection points. This is not a confirmation gate."
            ),
            "next_gate": (
                "Use a fresh paired nonlinear bulk confirmation and a separate "
                "tail audit before committing the channel to the accepted model."
            ),
        },
        "configuration": {
            "device": str(device),
            "candidate_start": args.candidate_start,
            "candidate_limit": candidate_count,
            "fit_size": args.fit_size,
            "selection_size": args.selection_size,
            "group_samples": args.group_samples,
            "operator_chunk_size": args.operator_chunk_size,
            "solver": args.solver,
            "lanczos_steps": args.lanczos_steps,
            "ridge_factors": list(args.ridge_factors),
            "step_scales": list(args.step_scales),
            "seed": args.seed,
        },
        "source": {
            "expanded_model": str(model_path),
            "expanded_model_sha256": sha256_file(model_path),
            "candidate_report": str(candidate_path),
            "candidate_report_sha256": sha256_file(candidate_path),
            "candidate_indices": str(previous_indices_path),
            "candidate_indices_sha256": sha256_file(previous_indices_path),
            "teacher_model": str(teacher_path),
            "teacher_model_sha256": sha256_file(teacher_path),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "validation_pullbacks": str(pullbacks_path),
            "validation_pullbacks_sha256": sha256_file(pullbacks_path),
            "indices": str(indices_path),
            "indices_sha256": sha256_file(indices_path),
            "output_model": str(output_model),
            "output_model_sha256": sha256_file(output_model),
        },
        "candidate": candidate,
        "data_isolation": {
            "preceding_used_index_count": int(previous_indices.size),
            "gn_used_index_count": int(stage_indices.size),
            "overlap_count": int(overlap.size),
        },
        "activation": activation,
        "fit_baseline": residual_metrics(fit_residual),
        "selection_baseline": residual_metrics(selection_residual),
        "gradient_norm": gradient_norm,
        "residual_rayleigh_scale": rayleigh_scale,
        "lanczos": lanczos_report,
        "selected": selected,
        "effective_dimension": effective_dimension,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(status, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "output_model": str(output_model),
                "output_report": str(output_report),
                "bond_index": bond_index,
                "selection_actual_capture": selected["selection_actual_capture"],
                "effective_dimension": effective_dimension,
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
