#!/usr/bin/env python3
"""Run rank-adaptive mixed-canonical two-site GN/LM updates on a quintic TN."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_quintic_tn_scaling_preflight import (  # noqa: E402
    build_model,
    cast_artifact_precision,
    load_numpy_split,
    make_tensor_split,
    resolve_inputs,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    ComplexParameterVectorizer,
    MatrixFreeResidualJacobian,
    fixed_residual_metrics,
    lanczos_tridiagonal,
    real_inner,
    ridge_dual_from_lanczos,
    vector_norm,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    evaluate_raw,
)
from scripts.train_quintic_tn_one_site_lm import (  # noqa: E402
    direct_empirical_statistics,
    load_training_arrays,
    random_subset,
    real_tensor_parameter_count,
    tail_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--blind-reference-run-dir", type=Path)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--pairs", type=int, nargs="+")
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=20000)
    parser.add_argument("--blind-limit", type=int, default=0)
    parser.add_argument("--operator-chunk-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lanczos-steps", type=int, default=4)
    parser.add_argument(
        "--ridge-factors", type=float, nargs="+", default=(1000.0, 100.0, 10.0)
    )
    parser.add_argument(
        "--line-search-alphas", type=float, nargs="+", default=(0.125, 0.25, 0.5)
    )
    parser.add_argument("--maximum-relative-step", type=float, default=2.0e-2)
    parser.add_argument(
        "--maximum-validation-tail-relative-degradation",
        type=float,
        default=2.0e-3,
    )
    parser.add_argument("--minimum-validation-chi-improvement", type=float, default=0.0)
    parser.add_argument("--maximum-bond-dimension", type=int, default=0)
    parser.add_argument("--relative-singular-value-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--maximum-discarded-relative-weight", type=float, default=1.0)
    parser.add_argument(
        "--split-direction",
        choices=("left", "right", "symmetric"),
        default="right",
    )
    parser.add_argument("--seed", type=int, default=202607219)
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-blind", action="store_true")
    parser.add_argument(
        "--audit-unsplit-candidates",
        action="store_true",
        help=(
            "also evaluate each full two-site candidate before the mandatory "
            "rank-D SVD split; audit-only and never used for acceptance"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_size,
        args.validation_size,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.lanczos_steps,
        args.maximum_relative_step,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sample, chunk, Lanczos, and step limits must be positive")
    if args.blind_limit < 0 or args.maximum_bond_dimension < 0:
        raise ValueError("blind limit and maximum bond dimension must be non-negative")
    if (
        args.minimum_validation_chi_improvement < 0
        or args.maximum_validation_tail_relative_degradation < 0
        or args.relative_singular_value_cutoff < 0
        or not 0 <= args.maximum_discarded_relative_weight <= 1
    ):
        raise ValueError("improvement, tail, SVD, and discarded-weight limits are invalid")
    if not args.ridge_factors or any(value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be positive")
    if not args.line_search_alphas or any(
        value <= 0 or value > 1 for value in args.line_search_alphas
    ):
        raise ValueError("line-search alphas must lie in (0, 1]")


def solver_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "precision": args.precision,
        "operator_chunk_size": int(args.operator_chunk_size),
        "eval_batch_size": int(args.eval_batch_size),
        "lanczos_steps": int(args.lanczos_steps),
        "ridge_factors": [float(value) for value in args.ridge_factors],
        "line_search_alphas": [float(value) for value in args.line_search_alphas],
        "maximum_relative_step": float(args.maximum_relative_step),
        "maximum_validation_tail_relative_degradation": float(
            args.maximum_validation_tail_relative_degradation
        ),
        "minimum_validation_chi_improvement": float(
            args.minimum_validation_chi_improvement
        ),
        "maximum_bond_dimension": int(args.maximum_bond_dimension),
        "relative_singular_value_cutoff": float(
            args.relative_singular_value_cutoff
        ),
        "maximum_discarded_relative_weight": float(
            args.maximum_discarded_relative_weight
        ),
        "split_direction": args.split_direction,
        "audit_unsplit_candidates": bool(args.audit_unsplit_candidates),
    }


def mixed_pair_gauge_errors(model: torch.nn.Module, start: int) -> dict[str, float]:
    left_errors = []
    right_errors = []
    for core in model.coefficient_cores[:start]:
        left, right, physical = core.shape
        matrix = core.permute(0, 2, 1).reshape(left * physical, right)
        gram = torch.conj(matrix.T) @ matrix
        left_errors.append(
            float(
                torch.linalg.matrix_norm(
                    gram - torch.eye(right, dtype=gram.dtype, device=gram.device),
                    ord=2,
                )
            )
        )
    for core in model.coefficient_cores[start + 2 :]:
        left, right, physical = core.shape
        matrix = core.permute(0, 2, 1).reshape(left, physical * right)
        gram = matrix @ torch.conj(matrix.T)
        right_errors.append(
            float(
                torch.linalg.matrix_norm(
                    gram - torch.eye(left, dtype=gram.dtype, device=gram.device),
                    ord=2,
                )
            )
        )
    return {
        "maximum_left_error": max(left_errors, default=0.0),
        "maximum_right_error": max(right_errors, default=0.0),
    }


def fixed_model_residual_metrics(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    fixed_log_kappa: float,
    chunk_size: int,
) -> dict[str, float]:
    raw, _ = evaluate_raw(model, dataset, chunk_size=chunk_size)
    ratio = np.exp(np.clip(raw - fixed_log_kappa, -20.0, 20.0))
    residual = np.sqrt(dataset["weights_numpy"]) * (ratio - 1.0)
    squared_norm = float(np.dot(residual, residual))
    return {
        "squared_norm": squared_norm,
        "rms": math.sqrt(max(0.0, squared_norm)),
        "maximum_weighted_component": float(np.max(np.abs(residual))),
    }


def copy_factors_(model: torch.nn.Module, start: int, left, right) -> None:
    with torch.no_grad():
        model.coefficient_cores[start].copy_(left)
        model.coefficient_cores[start + 1].copy_(right)


def optimize_pair(
    model: torch.nn.Module,
    start: int,
    train: dict[str, Any],
    validation: dict[str, Any],
    *,
    fixed_log_kappa: float,
    run_baseline_tail: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    model.requires_grad_(False)
    preservation_count = min(128, validation["count"])
    preservation = {
        "count": preservation_count,
        "values": validation["values"][:preservation_count],
        "derivatives": validation["derivatives"][:preservation_count],
        "log_omega": validation["log_omega"][:preservation_count],
    }
    before_raw, _ = evaluate_raw(model, preservation, chunk_size=preservation_count)
    floor_before = float(model.positive_floor)
    model.mixed_canonicalize_coefficient_pair_(start)
    global_scale = model.normalize_coefficient_chain_scale_(site_index=start)
    after_raw, _ = evaluate_raw(model, preservation, chunk_size=preservation_count)

    baseline_left = model.coefficient_cores[start].detach().clone()
    baseline_right = model.coefficient_cores[start + 1].detach().clone()
    active = model.activate_two_site_coefficient_core_(start)
    active.requires_grad_(True)
    vectorizer = ComplexParameterVectorizer.from_module(
        model, parameter_names=("two_site_core",)
    )
    theta = vectorizer.pack(model)
    baseline_validation = direct_empirical_statistics(
        model, validation, chunk_size=args.eval_batch_size
    )
    operator = MatrixFreeResidualJacobian(
        model,
        vectorizer,
        theta,
        train,
        fixed_log_kappa=fixed_log_kappa,
        chunk_size=args.operator_chunk_size,
    )
    residual = operator.residual()
    baseline_train = fixed_residual_metrics(residual)
    baseline_energy = baseline_train["squared_norm"]
    gradient = operator.vjp(residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(
        baseline_energy, np.finfo(float).tiny
    )
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        raise FloatingPointError("active two-site core has no finite descent direction")
    bond_capacity = int(model.coefficient_cores[start].shape[1])
    maximum_bond = args.maximum_bond_dimension or bond_capacity
    if maximum_bond > bond_capacity:
        raise ValueError("requested two-site rank exceeds the preallocated bond")
    with torch.no_grad():
        _, _, baseline_split = model.split_two_site_coefficient_core(
            theta.reshape_as(active),
            maximum_bond_dimension=maximum_bond,
            relative_singular_value_cutoff=args.relative_singular_value_cutoff,
            direction=args.split_direction,
        )

    def progress(iteration: int, alpha: float, beta: float) -> None:
        if not args.quiet:
            print(
                f"pair={start}:{start + 1} lanczos={iteration}/{args.lanczos_steps} "
                f"alpha={alpha:.6e} beta={beta:.6e}",
                flush=True,
            )

    lanczos = lanczos_tridiagonal(
        operator.normal,
        residual,
        steps=args.lanczos_steps,
        callback=progress,
    )
    theta_norm = float(vector_norm(theta))
    candidates: list[dict[str, Any]] = []
    for factor in args.ridge_factors:
        ridge = factor * rayleigh_scale
        dual = ridge_dual_from_lanczos(lanczos, ridge)
        delta = -operator.vjp(dual)
        predicted = residual + operator.jvp(delta)
        predicted_capture = 1.0 - float(real_inner(predicted, predicted)) / max(
            baseline_energy, np.finfo(float).tiny
        )
        for line_alpha in args.line_search_alphas:
            candidate = theta + line_alpha * delta
            relative_step = float(
                line_alpha
                * vector_norm(delta)
                / max(theta_norm, np.finfo(float).tiny)
            )
            with torch.no_grad():
                unsplit_train_metrics = None
                unsplit_validation_metrics = None
                if getattr(args, "audit_unsplit_candidates", False):
                    active.copy_(candidate.reshape_as(active))
                    model.use_two_site_core = True
                    unsplit_train_metrics = fixed_model_residual_metrics(
                        model,
                        train,
                        fixed_log_kappa=fixed_log_kappa,
                        chunk_size=args.eval_batch_size,
                    )
                    unsplit_validation_metrics = direct_empirical_statistics(
                        model,
                        validation,
                        chunk_size=args.eval_batch_size,
                    )
                left, right, split = model.split_two_site_coefficient_core(
                    candidate.reshape_as(active),
                    maximum_bond_dimension=maximum_bond,
                    relative_singular_value_cutoff=(
                        args.relative_singular_value_cutoff
                    ),
                    direction=args.split_direction,
                )
                copy_factors_(model, start, left, right)
                model.use_two_site_core = False
                train_metrics = fixed_model_residual_metrics(
                    model,
                    train,
                    fixed_log_kappa=fixed_log_kappa,
                    chunk_size=args.eval_batch_size,
                )
                validation_metrics = direct_empirical_statistics(
                    model, validation, chunk_size=args.eval_batch_size
                )
                model.use_two_site_core = True
                copy_factors_(model, start, baseline_left, baseline_right)
            candidates.append(
                {
                    "ridge_factor": float(factor),
                    "ridge": float(ridge),
                    "alpha": float(line_alpha),
                    "relative_step": relative_step,
                    "predicted_capture_at_alpha_1": predicted_capture,
                    "train_fixed": train_metrics,
                    "validation": validation_metrics,
                    "unsplit_train_fixed": unsplit_train_metrics,
                    "unsplit_validation": unsplit_validation_metrics,
                    "unsplit_validation_tail": (
                        None
                        if unsplit_validation_metrics is None
                        else tail_summary(unsplit_validation_metrics)
                    ),
                    "split": split,
                    "eligible": bool(
                        train_metrics["squared_norm"] < baseline_energy
                        and relative_step <= args.maximum_relative_step
                        and split["discarded_relative_weight"]
                        <= args.maximum_discarded_relative_weight
                    ),
                    "parameter_vector": candidate,
                    "left_factor": left.detach().clone(),
                    "right_factor": right.detach().clone(),
                }
            )

    baseline_chi = float(baseline_validation["weighted_rms_abs_residual"])
    baseline_tail = tail_summary(baseline_validation)
    tail_limit = 1.0 + args.maximum_validation_tail_relative_degradation
    for row in candidates:
        observed_tail = tail_summary(row["validation"])
        row["validation_tail"] = observed_tail
        row["local_tail_guard_passed"] = all(
            observed_tail[key] <= tail_limit * baseline_tail[key]
            for key in baseline_tail
        )
        row["cumulative_tail_guard_passed"] = all(
            observed_tail[key] <= tail_limit * run_baseline_tail[key]
            for key in run_baseline_tail
        )
        row["tail_guard_passed"] = bool(
            row["local_tail_guard_passed"]
            and row["cumulative_tail_guard_passed"]
        )
        row["eligible"] = bool(row["eligible"] and row["tail_guard_passed"])
    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda row: row["validation"]["weighted_rms_abs_residual"],
        )
        if eligible
        else None
    )
    selected_chi = (
        float(selected["validation"]["weighted_rms_abs_residual"])
        if selected is not None
        else float("inf")
    )
    accepted = bool(
        selected is not None
        and baseline_chi - selected_chi > args.minimum_validation_chi_improvement
    )
    if accepted:
        final_left = selected["left_factor"]
        final_right = selected["right_factor"]
    else:
        final_left = baseline_left
        final_right = baseline_right
    model.commit_two_site_coefficient_split_(final_left, final_right)
    model.requires_grad_(False)
    final_train = direct_empirical_statistics(
        model, train, chunk_size=args.eval_batch_size
    )
    final_validation = direct_empirical_statistics(
        model, validation, chunk_size=args.eval_batch_size
    )

    hidden = {"parameter_vector", "left_factor", "right_factor"}
    serializable_candidates = [
        {key: value for key, value in row.items() if key not in hidden}
        for row in candidates
    ]
    selected_summary = (
        None
        if selected is None
        else {key: value for key, value in selected.items() if key not in hidden}
    )
    return {
        "pair_start": start,
        "pair": [start, start + 1],
        "active_parameter_name": "two_site_core",
        "active_complex_parameters": int(theta.numel()),
        "active_real_parameters": int(2 * theta.numel()),
        "gauge": {
            "global_scale": global_scale,
            "positive_floor_before": floor_before,
            "positive_floor_after": float(model.positive_floor),
            "maximum_raw_log_volume_change": float(
                np.max(np.abs(after_raw - before_raw))
            ),
            **mixed_pair_gauge_errors(model, start),
        },
        "baseline": {
            "train_fixed": baseline_train,
            "validation": baseline_validation,
            "validation_tail": baseline_tail,
            "run_validation_tail": run_baseline_tail,
            "jt_residual_norm": gradient_norm,
            "residual_rayleigh_scale": rayleigh_scale,
            "split": baseline_split,
        },
        "lanczos": {
            "steps_completed": lanczos.operator_applications,
            "breakdown": lanczos.breakdown,
            "orthogonality_error": lanczos.orthogonality_error,
            "eigenvalue_minimum": float(
                torch.min(torch.linalg.eigvalsh(lanczos.tridiagonal))
            ),
            "eigenvalue_maximum": float(
                torch.max(torch.linalg.eigvalsh(lanczos.tridiagonal))
            ),
        },
        "candidates": serializable_candidates,
        "selected": selected_summary,
        "accepted": accepted,
        "validation_chi_change": (
            selected_chi - baseline_chi if selected is not None else None
        ),
        "final_train_empirical": final_train,
        "final_validation_empirical": final_validation,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})
    try:
        paths = resolve_inputs(args)
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        payload = torch.load(paths["model"], map_location="cpu", weights_only=False)
        payload = cast_artifact_precision(payload, args.precision)
        source_degree = int(payload.get("source_degree", 1))
        model = build_model(payload, device)
        if model.architecture != "shared_local_dictionary":
            raise ValueError("two-site updates require a shared local dictionary")
        pairs = args.pairs or [max(0, model.site_count // 2 - 1)]
        if any(start < 0 or start + 1 >= model.site_count for start in pairs):
            raise ValueError("two-site pair is outside the coefficient chain")

        resume_path = (
            None
            if args.resume_checkpoint is None
            else args.resume_checkpoint.expanduser().resolve()
        )
        resume_payload = None
        completed_pairs: list[int] = []
        if resume_path is not None:
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            if resume_payload.get("schema") != "quintic-tn-two-site-lm-checkpoint-v1":
                raise ValueError("unsupported two-site LM resume checkpoint")
            if resume_payload.get("solver_protocol") != solver_protocol(args):
                raise RuntimeError("resume checkpoint used another solver protocol")
            if resume_payload.get("parent_model_sha256") != sha256_file(paths["model"]):
                raise RuntimeError("resume checkpoint belongs to another parent model")
            completed_pairs = [int(value) for value in resume_payload["completed_pairs"]]
            if pairs[: len(completed_pairs)] != completed_pairs:
                raise ValueError("completed resume pairs are not a requested prefix")
        pairs_to_run = pairs[len(completed_pairs) :]

        training_arrays, train_indices = random_subset(
            load_training_arrays(paths), args.train_size, seed=args.seed
        )
        validation_arrays, validation_indices = random_subset(
            load_numpy_split(paths, "validation", 0),
            args.validation_size,
            seed=args.seed + 1,
        )
        train = make_tensor_split(
            *training_arrays,
            source_degree=source_degree,
            precision=args.precision,
            device=device,
        )
        validation = make_tensor_split(
            *validation_arrays,
            source_degree=source_degree,
            precision=args.precision,
            device=device,
        )
        indices_path = output_dir / "optimization_indices.npz"
        if resume_path is not None:
            previous_indices = resume_path.parent / "optimization_indices.npz"
            if not previous_indices.exists():
                raise FileNotFoundError(previous_indices)
            with np.load(previous_indices, allow_pickle=False) as previous:
                if not (
                    np.array_equal(previous["train_indices"], train_indices)
                    and np.array_equal(previous["validation_indices"], validation_indices)
                ):
                    raise RuntimeError("resume checkpoint used different data indices")
        np.savez_compressed(
            indices_path,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )

        blind_baseline = None
        if not args.skip_blind:
            write_json(status_path, {"state": "running", "phase": "baseline_blind"})
            blind_arrays = load_numpy_split(paths, "blind", args.blind_limit)
            blind = make_tensor_split(
                *blind_arrays,
                source_degree=source_degree,
                precision=args.precision,
                device=device,
            )
            blind_baseline = direct_empirical_statistics(
                model, blind, chunk_size=args.eval_batch_size
            )
            del blind, blind_arrays
            if device.type == "cuda":
                torch.cuda.empty_cache()

        fixed_log_kappa = float(payload["fixed_log_kappa"])
        validation_baseline = direct_empirical_statistics(
            model, validation, chunk_size=args.eval_batch_size
        )
        run_baseline_tail = tail_summary(validation_baseline)
        rows: list[dict[str, Any]] = []
        if resume_payload is not None and resume_path is not None:
            checkpoint_tail = {
                key: float(value)
                for key, value in resume_payload["run_baseline_tail"].items()
            }
            if any(
                not np.isclose(
                    checkpoint_tail[key],
                    run_baseline_tail[key],
                    rtol=2.0e-12,
                    atol=2.0e-14,
                )
                for key in run_baseline_tail
            ):
                raise RuntimeError("resume baseline tail does not match the parent model")
            history = json.loads(
                (resume_path.parent / "optimization_history.json").read_text(
                    encoding="utf-8"
                )
            )
            rows = list(history["rows"])
            if [int(row["pair_start"]) for row in rows] != completed_pairs:
                raise RuntimeError("resume history and checkpoint pairs disagree")
            model.load_state_dict(resume_payload["state_dict"], strict=True)
            model.positive_floor = float(resume_payload["positive_floor"])

        for step, start in enumerate(
            pairs_to_run, start=len(completed_pairs) + 1
        ):
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "two_site_lm",
                    "step": step,
                    "steps": len(pairs),
                    "pair": [start, start + 1],
                },
            )
            row = optimize_pair(
                model,
                start,
                train,
                validation,
                fixed_log_kappa=fixed_log_kappa,
                run_baseline_tail=run_baseline_tail,
                args=args,
            )
            rows.append(row)
            write_json(output_dir / "optimization_history.json", {"rows": rows})
            torch.save(
                {
                    "schema": "quintic-tn-two-site-lm-checkpoint-v1",
                    "parent_model": str(paths["model"]),
                    "parent_model_sha256": sha256_file(paths["model"]),
                    "completed_pairs": [entry["pair_start"] for entry in rows],
                    "solver_protocol": solver_protocol(args),
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "positive_floor": float(model.positive_floor),
                    "run_baseline_tail": run_baseline_tail,
                },
                output_dir / "optimization_checkpoint.pt",
            )
            if not args.quiet:
                selected = row["selected"]
                selected_rank = (
                    None if selected is None else selected["split"]["retained_rank"]
                )
                selected_discarded = (
                    None
                    if selected is None
                    else f"{selected['split']['discarded_relative_weight']:.3e}"
                )
                print(
                    f"pair={start}:{start + 1} accepted={row['accepted']} "
                    f"validation_chi={row['final_validation_empirical']['weighted_rms_abs_residual']:.6e} "
                    f"rank={selected_rank} discarded={selected_discarded}",
                    flush=True,
                )

        blind_final = None
        if not args.skip_blind:
            write_json(status_path, {"state": "running", "phase": "final_blind"})
            blind_arrays = load_numpy_split(paths, "blind", args.blind_limit)
            blind = make_tensor_split(
                *blind_arrays,
                source_degree=source_degree,
                precision=args.precision,
                device=device,
            )
            blind_final = direct_empirical_statistics(
                model, blind, chunk_size=args.eval_batch_size
            )

        artifact = copy.deepcopy(payload)
        artifact["state_dict"] = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        artifact["precision"] = args.precision
        artifact["positive_floor"] = float(model.positive_floor)
        artifact["optimization_protocol"] = {
            "name": "mixed-canonical rank-adaptive two-site damped GN/Lanczos",
            "parent_model": str(paths["model"]),
            "parent_model_sha256": sha256_file(paths["model"]),
            "pairs": pairs,
            "fixed_log_kappa": fixed_log_kappa,
            "maximum_bond_dimension": int(
                args.maximum_bond_dimension or model.bond_dimension
            ),
        }
        model_path = output_dir / "best_tensor_network.pt"
        torch.save(artifact, model_path)

        active_real_parameters = real_tensor_parameter_count(model.parameters())
        fixed_dictionary_real_parameters = 0
        if payload.get("trainable_physical_dictionary", False):
            dictionary = model.physical_dictionary
            fixed_dictionary_real_parameters = dictionary.numel() * (
                2 if dictionary.is_complex() else 1
            )
        report = {
            "schema": "quintic-tn-two-site-lm-v1",
            "scientific_scope": {
                "purpose": (
                    "Rank-adaptive optimizer ablation on a preallocated TN bond; "
                    "the common blind pool is not a sealed final test."
                ),
                "objective": "sampled Headrick--Nassar sum w_i (r_i-1)^2",
                "acceptance": (
                    "post-SVD fixed-kappa training energy must decrease; validation "
                    "chi selects candidates subject to local and cumulative tail guards"
                ),
            },
            "configuration": {
                **vars(args),
                "run_dir": str(args.run_dir),
                "output_dir": str(output_dir),
                "resume_checkpoint": None if resume_path is None else str(resume_path),
                "source_run_dir": str(paths["source"]),
                "blind_reference_run_dir": str(paths["blind"]),
                "pullbacks_dir": str(paths["pullbacks"]),
                "model": str(paths["model"]),
                "device": str(device),
                "pairs": pairs,
            },
            "source": {
                "model_sha256": sha256_file(paths["model"]),
                "run_report_sha256": sha256_file(paths["report"]),
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "precompleted_pairs": completed_pairs,
                "source_run_dir": str(paths["source"]),
                "blind_reference_run_dir": str(paths["blind"]),
                "pullbacks_dir": str(paths["pullbacks"]),
                "blind_points_sha256": sha256_file(paths["blind"] / "blind_points.npz"),
                "blind_pullbacks_sha256": sha256_file(
                    paths["pullbacks"] / "blind_pullbacks.npy"
                ),
            },
            "architecture": {
                "site_count": int(model.site_count),
                "bond_dimension": int(model.bond_dimension),
                "active_real_parameters": active_real_parameters,
                "fixed_learned_dictionary_real_parameters": (
                    fixed_dictionary_real_parameters
                ),
                "stored_learned_real_parameters": (
                    active_real_parameters + fixed_dictionary_real_parameters
                ),
            },
            "optimization": {
                "fixed_log_kappa": fixed_log_kappa,
                "validation_baseline": validation_baseline,
                "rows": rows,
            },
            "benchmark": {"baseline": blind_baseline, "final": blind_final},
            "artifacts": {
                "model": str(model_path),
                "model_sha256": sha256_file(model_path),
                "optimization_checkpoint": str(
                    output_dir / "optimization_checkpoint.pt"
                ),
                "optimization_checkpoint_sha256": sha256_file(
                    output_dir / "optimization_checkpoint.pt"
                ),
            },
            "timing_seconds": time.perf_counter() - started,
        }
        report_path = output_dir / "report.json"
        write_json(report_path, report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "wall_seconds": report["timing_seconds"],
            },
        )
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "accepted_pairs": [
                        row["pair"] for row in rows if row["accepted"]
                    ],
                    "blind_baseline_sigma": (
                        None
                        if blind_baseline is None
                        else blind_baseline["sigma_official_formula"]
                    ),
                    "blind_final_sigma": (
                        None
                        if blind_final is None
                        else blind_final["sigma_official_formula"]
                    ),
                    "timing_seconds": report["timing_seconds"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "exception",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
