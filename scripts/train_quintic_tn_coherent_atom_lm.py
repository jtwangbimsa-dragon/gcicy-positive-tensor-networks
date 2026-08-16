#!/usr/bin/env python3
"""Refine one coherent residual atom with canonical one-site GN/LM sweeps."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    PositiveTensorNetworkCoherentSumMetric,
    PositiveTensorNetworkPositiveSumMetric,
)
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
from scripts.train_quintic_full_h_same_points import sha256_file, write_json  # noqa: E402
from scripts.train_quintic_tn_one_site_lm import (  # noqa: E402
    copy_parameter_vector_into_model,
    direct_empirical_statistics,
    evaluate_validation_candidate,
    load_training_arrays,
    mixed_gauge_errors,
    random_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--coherent-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--blind-reference-run-dir", type=Path)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--centers", type=int, nargs="+")
    parser.add_argument("--atom-index", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=20000)
    parser.add_argument("--blind-limit", type=int, default=0)
    parser.add_argument("--operator-chunk-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lanczos-steps", type=int, default=8)
    parser.add_argument(
        "--ridge-factors", type=float, nargs="+", default=(1000.0, 100.0, 10.0, 1.0)
    )
    parser.add_argument(
        "--line-search-alphas", type=float, nargs="+", default=(0.125, 0.25, 0.5, 1.0)
    )
    parser.add_argument("--maximum-relative-step", type=float, default=5.0e-2)
    parser.add_argument("--maximum-gate-magnitude", type=float, default=1.0)
    parser.add_argument(
        "--maximum-validation-tail-relative-degradation",
        type=float,
        default=2.0e-3,
    )
    parser.add_argument("--minimum-validation-chi-improvement", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=202607215)
    parser.add_argument("--precision", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--freeze-gate", action="store_true")
    parser.add_argument("--skip-blind", action="store_true")
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
        args.maximum_gate_magnitude,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sample, solver, and trust-region values must be positive")
    if (
        args.blind_limit < 0
        or args.minimum_validation_chi_improvement < 0
        or args.maximum_validation_tail_relative_degradation < 0
    ):
        raise ValueError("blind limit and validation tolerances must be non-negative")
    if args.atom_index <= 0:
        raise ValueError("the leading branch cannot be selected as a residual atom")
    if not args.ridge_factors or any(value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be positive")
    if not args.line_search_alphas or any(
        value <= 0 or value > 1 for value in args.line_search_alphas
    ):
        raise ValueError("line-search alphas must lie in (0, 1]")


def load_coherent_model(
    run_dir: Path, *, precision: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    artifact_path = run_dir.expanduser().resolve() / "coherent_model.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    schema = artifact.get("schema")
    model_class = {
        "quintic-positive-tensor-network-coherent-sum-v1": (
            PositiveTensorNetworkCoherentSumMetric
        ),
        "quintic-positive-tensor-network-positive-sum-v1": (
            PositiveTensorNetworkPositiveSumMetric
        ),
    }.get(schema)
    if model_class is None:
        raise ValueError("unsupported multi-atom tensor-network artifact")
    states = artifact.get("branch_states")
    metadata = artifact.get("branch_metadata")
    if not isinstance(states, list) or not isinstance(metadata, list):
        raise ValueError("coherent artifact has no registered branch states")
    if len(states) != len(metadata) or len(states) < 2:
        raise ValueError("coherent artifact branch metadata are inconsistent")

    branches = []
    for state, branch_metadata in zip(states, metadata, strict=True):
        source_path = Path(branch_metadata["source_model"]).expanduser().resolve()
        if sha256_file(source_path) != branch_metadata["source_model_sha256"]:
            raise RuntimeError(f"source branch hash changed: {source_path}")
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        payload = cast_artifact_precision(payload, precision)
        branch = build_model(payload, device)
        cast_state = {
            name: (
                value.to(
                    dtype=(
                        torch.complex128
                        if precision == "complex128"
                        else torch.complex64
                    )
                    if value.is_complex()
                    else value.dtype,
                    device=device,
                )
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in state.items()
        }
        branch.load_state_dict(cast_state, strict=True)
        branches.append(branch)

    gates = artifact["new_gates"]
    if isinstance(gates, torch.Tensor):
        gates = gates.detach().cpu().numpy()
    model = model_class(
        tuple(branches),
        positive_floor=float(artifact["positive_floor"]),
        initial_new_gates=tuple(complex(value) for value in gates),
    )
    model.to(device)
    return model, artifact, artifact_path


def tail_summary(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "q999": float(statistics["abs_residual_weighted_quantiles"]["q0.9990"]),
        "cvar99": float(statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]),
        "maximum": float(statistics["abs_residual_weighted_quantiles"]["q1.0000"]),
    }


def real_tensor_parameter_count(parameters: Any) -> int:
    return int(
        sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in parameters
        )
    )


def fixed_learned_dictionary_parameter_count(
    source_artifact: dict[str, Any], branches: Any
) -> int:
    count = 0
    for metadata, branch in zip(
        source_artifact["branch_metadata"], branches, strict=True
    ):
        payload = torch.load(
            Path(metadata["source_model"]).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("trainable_physical_dictionary", False):
            dictionary = branch.physical_dictionary
            count += dictionary.numel() * (2 if dictionary.is_complex() else 1)
    return int(count)


def canonicalize_atom_preserving_coherent_amplitude_(
    model: torch.nn.Module, atom_index: int, center: int
) -> dict[str, float]:
    atom = model.branches[atom_index]
    atom.mixed_canonicalize_coefficient_cores_(center)
    scale = atom.normalize_coefficient_chain_scale_(site_index=center)
    with torch.no_grad():
        model.new_gates[atom_index - 1].div_(scale)
    return {
        "atom_scale": float(scale),
        "gate_real_after_compensation": float(
            torch.real(model.new_gates[atom_index - 1])
        ),
        "gate_imaginary_after_compensation": float(
            torch.imag(model.new_gates[atom_index - 1])
        ),
        **mixed_gauge_errors(atom, center),
    }


def optimize_atom_center(
    model: torch.nn.Module,
    atom_index: int,
    center: int,
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
        "weights": validation["weights"][:preservation_count],
        "weights_numpy": validation["weights_numpy"][:preservation_count],
    }
    before = direct_empirical_statistics(
        model, preservation, chunk_size=preservation_count
    )
    gauge = canonicalize_atom_preserving_coherent_amplitude_(
        model, atom_index, center
    )
    after = direct_empirical_statistics(
        model, preservation, chunk_size=preservation_count
    )
    gauge["preservation_sigma_change"] = float(
        after["sigma_official_formula"] - before["sigma_official_formula"]
    )
    gauge["preservation_chi_change"] = float(
        after["weighted_rms_abs_residual"] - before["weighted_rms_abs_residual"]
    )

    active_core = f"branches.{atom_index}.coefficient_cores.{center}"
    model.branches[atom_index].coefficient_cores[center].requires_grad_(True)
    active_names = [active_core]
    if not args.freeze_gate:
        model.new_gates.requires_grad_(True)
        active_names.insert(0, "new_gates")
    vectorizer = ComplexParameterVectorizer.from_module(
        model, parameter_names=tuple(active_names)
    )
    theta = vectorizer.pack(model)
    baseline_validation = evaluate_validation_candidate(
        model,
        vectorizer,
        theta,
        validation,
        chunk_size=args.eval_batch_size,
    )
    baseline_tail = tail_summary(baseline_validation)
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
        raise FloatingPointError("active coherent atom core has no descent direction")

    def progress(iteration: int, alpha: float, beta: float) -> None:
        if not args.quiet:
            print(
                f"center={center} lanczos={iteration}/{args.lanczos_steps} "
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
    candidates = []
    tail_limit = 1.0 + args.maximum_validation_tail_relative_degradation
    for factor in args.ridge_factors:
        ridge = factor * rayleigh_scale
        dual = ridge_dual_from_lanczos(lanczos, ridge)
        delta = -operator.vjp(dual)
        predicted = residual + operator.jvp(delta)
        predicted_capture = 1.0 - float(real_inner(predicted, predicted)) / max(
            baseline_energy, np.finfo(float).tiny
        )
        for alpha in args.line_search_alphas:
            candidate = theta + alpha * delta
            candidate_parts = vectorizer.unpack(candidate)
            gate = candidate_parts.get("new_gates", model.new_gates.detach())
            gate_magnitude = float(torch.abs(gate[atom_index - 1]))
            relative_step = float(
                alpha * vector_norm(delta) / max(theta_norm, np.finfo(float).tiny)
            )
            train_residual = operator.residual(candidate)
            train_metrics = fixed_residual_metrics(train_residual)
            validation_metrics = evaluate_validation_candidate(
                model,
                vectorizer,
                candidate,
                validation,
                chunk_size=args.eval_batch_size,
            )
            observed_tail = tail_summary(validation_metrics)
            local_tail_guard = all(
                observed_tail[key] <= tail_limit * baseline_tail[key]
                for key in baseline_tail
            )
            cumulative_tail_guard = all(
                observed_tail[key] <= tail_limit * run_baseline_tail[key]
                for key in run_baseline_tail
            )
            tail_guard = local_tail_guard and cumulative_tail_guard
            candidates.append(
                {
                    "ridge_factor": float(factor),
                    "ridge": float(ridge),
                    "alpha": float(alpha),
                    "relative_step": relative_step,
                    "gate_magnitude": gate_magnitude,
                    "predicted_capture_at_alpha_1": predicted_capture,
                    "train_fixed": train_metrics,
                    "validation": validation_metrics,
                    "validation_tail": observed_tail,
                    "local_tail_guard_passed": local_tail_guard,
                    "cumulative_tail_guard_passed": cumulative_tail_guard,
                    "tail_guard_passed": tail_guard,
                    "eligible": bool(
                        train_metrics["squared_norm"] < baseline_energy
                        and relative_step <= args.maximum_relative_step
                        and gate_magnitude <= args.maximum_gate_magnitude
                        and tail_guard
                    ),
                    "parameter_vector": candidate,
                }
            )

    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda row: row["validation"]["weighted_rms_abs_residual"],
        )
        if eligible
        else None
    )
    baseline_chi = float(baseline_validation["weighted_rms_abs_residual"])
    selected_chi = (
        float(selected["validation"]["weighted_rms_abs_residual"])
        if selected is not None
        else float("inf")
    )
    accepted = bool(
        selected is not None
        and baseline_chi - selected_chi
        > args.minimum_validation_chi_improvement
    )
    if accepted:
        copy_parameter_vector_into_model(model, vectorizer, selected["parameter_vector"])
    model.requires_grad_(False)

    return {
        "center": center,
        "atom_index": atom_index,
        "active_parameter_names": vectorizer.names,
        "active_complex_parameters": int(theta.numel()),
        "active_real_parameters": int(2 * theta.numel()),
        "gauge": gauge,
        "baseline": {
            "train_fixed": baseline_train,
            "validation": baseline_validation,
            "validation_tail": baseline_tail,
            "run_validation_tail": run_baseline_tail,
            "jt_residual_norm": gradient_norm,
            "residual_rayleigh_scale": rayleigh_scale,
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
        "candidates": [
            {key: value for key, value in row.items() if key != "parameter_vector"}
            for row in candidates
        ],
        "selected": (
            None
            if selected is None
            else {
                key: value
                for key, value in selected.items()
                if key != "parameter_vector"
            }
        ),
        "accepted": accepted,
        "validation_chi_change": (
            selected_chi - baseline_chi if selected is not None else None
        ),
        "final_gate": {
            "real": float(torch.real(model.new_gates[atom_index - 1])),
            "imaginary": float(torch.imag(model.new_gates[atom_index - 1])),
            "magnitude": float(torch.abs(model.new_gates[atom_index - 1])),
        },
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
        model, source_artifact, source_artifact_path = load_coherent_model(
            args.coherent_run_dir,
            precision=args.precision,
            device=device,
        )
        if args.atom_index >= len(model.branches):
            raise ValueError("atom index exceeds the coherent branch count")
        atom = model.branches[args.atom_index]
        centers = args.centers or [atom.site_count // 2]
        if any(center < 0 or center >= atom.site_count for center in centers):
            raise ValueError("one-site center is outside the atom chain")
        resume_checkpoint_path = (
            None
            if args.resume_checkpoint is None
            else args.resume_checkpoint.expanduser().resolve()
        )
        resume_payload = None
        completed_centers: list[int] = []
        if resume_checkpoint_path is not None:
            resume_payload = torch.load(
                resume_checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                resume_payload.get("schema")
                != "quintic-tn-multi-atom-one-site-lm-checkpoint-v1"
            ):
                raise ValueError("unsupported multi-atom LM resume checkpoint")
            if (
                resume_payload.get("source_artifact_sha256")
                != sha256_file(source_artifact_path)
            ):
                raise RuntimeError("resume checkpoint belongs to another source artifact")
            completed_centers = [
                int(center) for center in resume_payload["completed_centers"]
            ]
            if centers[: len(completed_centers)] != completed_centers:
                raise ValueError(
                    "completed resume centers are not a prefix of the requested sweep"
                )
        centers_to_run = centers[len(completed_centers) :]
        if float(torch.abs(model.new_gates[args.atom_index - 1])) == 0.0:
            raise ValueError("atom LM requires a nonzero accepted coherent gate")

        training_arrays, train_indices = random_subset(
            load_training_arrays(paths), args.train_size, seed=args.seed
        )
        validation_arrays, validation_indices = random_subset(
            load_numpy_split(paths, "validation", 0),
            args.validation_size,
            seed=args.seed + 1,
        )
        source_degree = int(source_artifact["source_degree"])
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
        if resume_checkpoint_path is not None:
            resume_indices_path = resume_checkpoint_path.parent / "optimization_indices.npz"
            if not resume_indices_path.exists():
                raise FileNotFoundError(resume_indices_path)
            with np.load(resume_indices_path, allow_pickle=False) as resume_indices:
                if not (
                    np.array_equal(resume_indices["train_indices"], train_indices)
                    and np.array_equal(
                        resume_indices["validation_indices"], validation_indices
                    )
                ):
                    raise RuntimeError("resume checkpoint used different data indices")
        np.savez_compressed(
            indices_path,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )

        validation_baseline = direct_empirical_statistics(
            model, validation, chunk_size=args.eval_batch_size
        )
        source_tail_reference = source_artifact.get("validation_tail_reference")
        if source_tail_reference is None:
            run_baseline_tail = tail_summary(validation_baseline)
            run_baseline_tail_source = "multi-atom source model"
        else:
            reference_indices = np.asarray(
                source_tail_reference["validation_indices"], dtype=np.int64
            )
            if not np.array_equal(reference_indices, validation_indices):
                raise RuntimeError(
                    "registered leading-tail reference used different validation indices"
                )
            observed_initial_tail = tail_summary(validation_baseline)
            stored_initial_tail = {
                key: float(value)
                for key, value in source_tail_reference["initial"].items()
            }
            if any(
                not np.isclose(
                    observed_initial_tail[key],
                    stored_initial_tail[key],
                    rtol=2.0e-10,
                    atol=2.0e-13,
                )
                for key in stored_initial_tail
            ):
                raise RuntimeError(
                    "registered multi-atom initialization tail was not reproduced"
                )
            run_baseline_tail = {
                key: float(value)
                for key, value in source_tail_reference["leading"].items()
            }
            initial_tail_limit = (
                1.0 + args.maximum_validation_tail_relative_degradation
            )
            if any(
                observed_initial_tail[key]
                > initial_tail_limit * run_baseline_tail[key]
                for key in run_baseline_tail
            ):
                raise RuntimeError(
                    "multi-atom initialization already exceeds the leading-model tail guard"
                )
            run_baseline_tail_source = "registered leading branch on identical indices"
        blind = None
        blind_baseline = None
        if not args.skip_blind:
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

        rows: list[dict[str, Any]] = []
        if resume_payload is not None and resume_checkpoint_path is not None:
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
                raise RuntimeError("resume baseline tail does not match the source run")
            history_path = resume_checkpoint_path.parent / "optimization_history.json"
            history = json.loads(history_path.read_text())
            rows = list(history["rows"])
            if [int(row["center"]) for row in rows] != completed_centers:
                raise RuntimeError("resume history and checkpoint centers disagree")
            states = resume_payload["branch_states"]
            if len(states) != len(model.branches):
                raise RuntimeError("resume checkpoint branch count changed")
            for branch, state in zip(model.branches, states, strict=True):
                branch.load_state_dict(state, strict=True)
            with torch.no_grad():
                model.new_gates.copy_(
                    resume_payload["new_gates"].to(
                        device=model.new_gates.device,
                        dtype=model.new_gates.dtype,
                    )
                )

        for step, center in enumerate(
            centers_to_run, start=len(completed_centers) + 1
        ):
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "coherent_atom_one_site_lm",
                    "step": step,
                    "steps": len(centers),
                    "center": center,
                },
            )
            rows.append(
                optimize_atom_center(
                    model,
                    args.atom_index,
                    center,
                    train,
                    validation,
                    fixed_log_kappa=float(source_artifact["fixed_log_kappa"]),
                    run_baseline_tail=run_baseline_tail,
                    args=args,
                )
            )
            write_json(output_dir / "optimization_history.json", {"rows": rows})
            torch.save(
                {
                    "schema": "quintic-tn-multi-atom-one-site-lm-checkpoint-v1",
                    "source_artifact": str(source_artifact_path),
                    "source_artifact_sha256": sha256_file(source_artifact_path),
                    "completed_centers": [row["center"] for row in rows],
                    "branch_states": [
                        {
                            key: value.detach().cpu()
                            for key, value in branch.state_dict().items()
                        }
                        for branch in model.branches
                    ],
                    "new_gates": model.new_gates.detach().cpu(),
                    "run_baseline_tail": run_baseline_tail,
                },
                output_dir / "optimization_checkpoint.pt",
            )

        validation_final = direct_empirical_statistics(
            model, validation, chunk_size=args.eval_batch_size
        )
        blind_final = (
            None
            if blind is None
            else direct_empirical_statistics(
                model, blind, chunk_size=args.eval_batch_size
            )
        )

        artifact = dict(source_artifact)
        artifact["branch_states"] = [
            {
                key: value.detach().cpu()
                for key, value in branch.state_dict().items()
            }
            for branch in model.branches
        ]
        artifact["new_gates"] = model.new_gates.detach().cpu()
        artifact["parent_coherent_artifact"] = str(source_artifact_path)
        artifact["parent_coherent_artifact_sha256"] = sha256_file(
            source_artifact_path
        )
        artifact["optimization"] = {
            "method": "mixed-canonical one-site damped Gauss-Newton/LM",
            "atom_index": args.atom_index,
            "centers": centers,
        }
        model_path = output_dir / "coherent_model.pt"
        torch.save(artifact, model_path)

        active_real_parameters = real_tensor_parameter_count(model.parameters())
        fixed_dictionary_real_parameters = fixed_learned_dictionary_parameter_count(
            source_artifact, model.branches
        )

        report = {
            "schema": "quintic-tn-multi-atom-one-site-lm-v1",
            "scientific_scope": {
                "purpose": "residual-informed multi-atom refinement",
                "composition": (
                    "coherent amplitude sum"
                    if source_artifact["schema"]
                    == "quintic-positive-tensor-network-coherent-sum-v1"
                    else "incoherent positive norm sum"
                ),
                "selection_metric": "independent validation chi with tail guard",
                "pool_status": "existing architecture benchmark, not sealed final test",
            },
            "configuration": {
                **{
                    key: value
                    for key, value in vars(args).items()
                    if not isinstance(value, Path)
                },
                "run_dir": str(args.run_dir),
                "coherent_run_dir": str(args.coherent_run_dir),
                "output_dir": str(output_dir),
                "resume_checkpoint": (
                    None
                    if resume_checkpoint_path is None
                    else str(resume_checkpoint_path)
                ),
                "device": str(device),
            },
            "source": {
                "coherent_artifact": str(source_artifact_path),
                "coherent_artifact_sha256": sha256_file(source_artifact_path),
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "source_run_dir": str(paths["source"]),
                "blind_reference_run_dir": str(paths["blind"]),
                "pullbacks_dir": str(paths["pullbacks"]),
                "blind_points_sha256": sha256_file(
                    paths["blind"] / "blind_points.npz"
                ),
                "blind_pullbacks_sha256": sha256_file(
                    paths["pullbacks"] / "blind_pullbacks.npy"
                ),
                "precompleted_centers": completed_centers,
                "run_baseline_tail_source": run_baseline_tail_source,
                "run_baseline_tail": run_baseline_tail,
            },
            "architecture": {
                "composition_schema": source_artifact["schema"],
                "branch_count": len(model.branches),
                "bond_dimensions": [
                    int(branch.bond_dimension) for branch in model.branches
                ],
                "active_real_parameters": active_real_parameters,
                "fixed_learned_dictionary_real_parameters": (
                    fixed_dictionary_real_parameters
                ),
                "stored_learned_real_parameters": (
                    active_real_parameters + fixed_dictionary_real_parameters
                ),
                "final_gate": {
                    "real": float(torch.real(model.new_gates[args.atom_index - 1])),
                    "imaginary": float(torch.imag(model.new_gates[args.atom_index - 1])),
                    "magnitude": float(torch.abs(model.new_gates[args.atom_index - 1])),
                },
            },
            "validation": {
                "baseline": validation_baseline,
                "final": validation_final,
            },
            "steps": rows,
            "benchmark": {
                "baseline": blind_baseline,
                "final": blind_final,
            },
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
            },
        )
        print(
            json.dumps(
                {
                    "accepted_centers": [
                        row["center"] for row in rows if row["accepted"]
                    ],
                    "validation_sigma_baseline": validation_baseline[
                        "sigma_official_formula"
                    ],
                    "validation_sigma_final": validation_final[
                        "sigma_official_formula"
                    ],
                    "validation_chi_baseline": validation_baseline[
                        "weighted_rms_abs_residual"
                    ],
                    "validation_chi_final": validation_final[
                        "weighted_rms_abs_residual"
                    ],
                    "report": str(report_path),
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
            },
        )
        raise


if __name__ == "__main__":
    main()
