#!/usr/bin/env python3
"""Fit the exact nested complex gate of a two-atom coherent quintic TN."""

from __future__ import annotations

import argparse
import copy
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
    real_inner,
)
from scripts.train_quintic_full_h_same_points import sha256_file, write_json  # noqa: E402
from scripts.train_quintic_tn_one_site_lm import (  # noqa: E402
    direct_empirical_statistics,
    evaluate_validation_candidate,
    load_training_arrays,
    random_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--atom-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--blind-reference-run-dir", type=Path)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=20000)
    parser.add_argument("--blind-limit", type=int, default=0)
    parser.add_argument("--operator-chunk-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--damping-factors", type=float, nargs="+", default=(100.0, 10.0, 1.0, 0.1)
    )
    parser.add_argument(
        "--line-search-alphas", type=float, nargs="+", default=(0.125, 0.25, 0.5, 1.0)
    )
    parser.add_argument("--maximum-gate-magnitude", type=float, default=1.0)
    parser.add_argument(
        "--maximum-validation-tail-relative-degradation",
        type=float,
        default=2.0e-3,
    )
    parser.add_argument("--canonical-center", type=int)
    parser.add_argument("--atom-relative-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=202607213)
    parser.add_argument("--precision", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-blind", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(
        args.train_size,
        args.validation_size,
        args.operator_chunk_size,
        args.eval_batch_size,
    ) <= 0:
        raise ValueError("sample and batch sizes must be positive")
    if args.blind_limit < 0 or args.maximum_validation_tail_relative_degradation < 0:
        raise ValueError("blind limit and tail tolerance must be non-negative")
    if args.maximum_gate_magnitude <= 0 or not np.isfinite(args.maximum_gate_magnitude):
        raise ValueError("maximum gate magnitude must be finite and positive")
    if args.atom_relative_noise < 0 or not np.isfinite(args.atom_relative_noise):
        raise ValueError("atom noise must be finite and non-negative")
    if not args.damping_factors or any(value < 0 for value in args.damping_factors):
        raise ValueError("damping factors must be non-negative")
    if not args.line_search_alphas or any(
        value <= 0 or value > 1 for value in args.line_search_alphas
    ):
        raise ValueError("line-search alphas must lie in (0, 1]")


def validate_branch_payloads(
    leading: dict[str, Any], atom: dict[str, Any]
) -> None:
    keys = (
        "source_degree",
        "total_degree",
        "site_count",
        "bond_dimension",
        "output_dimension",
        "target_normalization",
        "architecture",
    )
    mismatches = {
        key: (leading.get(key), atom.get(key))
        for key in keys
        if leading.get(key) != atom.get(key)
        and not (
            key == "output_dimension"
            and leading.get(key, 5) == atom.get(key, 5)
        )
    }
    if mismatches:
        raise ValueError(f"coherent branch metadata mismatch: {mismatches}")


def add_relative_atom_noise_(
    atom: torch.nn.Module, relative_noise: float, *, seed: int
) -> None:
    if relative_noise == 0:
        return
    generator = torch.Generator(device=atom.reference_h.device)
    generator.manual_seed(seed)
    with torch.no_grad():
        for core in atom.coefficient_cores:
            rms = torch.linalg.vector_norm(core) / np.sqrt(core.numel())
            real = torch.randn(
                core.shape,
                generator=generator,
                dtype=core.real.dtype,
                device=core.device,
            )
            imaginary = torch.randn(
                core.shape,
                generator=generator,
                dtype=core.real.dtype,
                device=core.device,
            )
            core.add_(relative_noise * rms * (real + 1j * imaginary) / np.sqrt(2.0))


def tail_summary(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "q999": float(statistics["abs_residual_weighted_quantiles"]["q0.9990"]),
        "cvar99": float(statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]),
        "maximum": float(statistics["abs_residual_weighted_quantiles"]["q1.0000"]),
    }


def fit_gate(
    model: torch.nn.Module,
    train: dict[str, Any],
    validation: dict[str, Any],
    *,
    fixed_log_kappa: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.requires_grad_(False)
    model.new_gates.requires_grad_(True)
    vectorizer = ComplexParameterVectorizer.from_module(
        model, parameter_names=("new_gates",)
    )
    theta = vectorizer.pack(model)
    if theta.numel() != 1 or float(torch.abs(theta[0])) != 0.0:
        raise ValueError("gate preflight requires one exactly zero new gate")
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
    directions = (
        torch.ones_like(theta),
        1j * torch.ones_like(theta),
    )
    columns = torch.stack([operator.jvp(direction) for direction in directions], dim=1)
    normal = torch.transpose(columns, 0, 1) @ columns
    gradient = torch.stack(
        [real_inner(columns[:, index], residual) for index in range(2)]
    )
    scale = float(torch.trace(normal) / 2.0)
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError("coherent atom has no visible gate tangent")
    tail_limit = 1.0 + args.maximum_validation_tail_relative_degradation
    candidates = []
    for factor in args.damping_factors:
        damping = factor * scale
        system = normal + damping * torch.eye(
            2, dtype=normal.dtype, device=normal.device
        )
        delta_components = torch.linalg.solve(system, -gradient)
        delta = delta_components[0].to(theta.dtype) + 1j * delta_components[1].to(
            theta.dtype
        )
        for alpha in args.line_search_alphas:
            candidate = theta + alpha * delta.reshape_as(theta)
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
            tail_guard = all(
                observed_tail[key] <= tail_limit * baseline_tail[key]
                for key in baseline_tail
            )
            candidates.append(
                {
                    "damping_factor": float(factor),
                    "damping": float(damping),
                    "alpha": float(alpha),
                    "gate_real": float(torch.real(candidate[0])),
                    "gate_imaginary": float(torch.imag(candidate[0])),
                    "gate_magnitude": float(torch.abs(candidate[0])),
                    "train_fixed": train_metrics,
                    "validation": validation_metrics,
                    "validation_tail": observed_tail,
                    "tail_guard_passed": tail_guard,
                    "eligible": bool(
                        train_metrics["squared_norm"]
                        < baseline_train["squared_norm"]
                        and float(torch.abs(candidate[0]))
                        <= args.maximum_gate_magnitude
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
    accepted = bool(
        selected is not None
        and float(selected["validation"]["weighted_rms_abs_residual"])
        < baseline_chi
    )
    if accepted:
        with torch.no_grad():
            model.new_gates.copy_(selected["parameter_vector"])
    model.requires_grad_(False)
    return {
        "baseline_train_fixed": baseline_train,
        "baseline_validation": baseline_validation,
        "baseline_validation_tail": baseline_tail,
        "gate_jacobian_normal_matrix": normal.detach().cpu().numpy().tolist(),
        "gate_jacobian_gradient": gradient.detach().cpu().numpy().tolist(),
        "gate_jacobian_scale": scale,
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
        "final_gate": {
            "real": float(torch.real(model.new_gates[0])),
            "imaginary": float(torch.imag(model.new_gates[0])),
            "magnitude": float(torch.abs(model.new_gates[0])),
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
        atom_path = args.atom_model.expanduser().resolve()
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        leading_payload = torch.load(
            paths["model"], map_location="cpu", weights_only=False
        )
        atom_payload = torch.load(atom_path, map_location="cpu", weights_only=False)
        validate_branch_payloads(leading_payload, atom_payload)
        leading_payload = cast_artifact_precision(leading_payload, args.precision)
        atom_payload = cast_artifact_precision(atom_payload, args.precision)
        leading = build_model(leading_payload, device)
        atom = build_model(atom_payload, device)
        center = (
            leading.site_count // 2
            if args.canonical_center is None
            else args.canonical_center
        )
        if center < 0 or center >= leading.site_count:
            raise ValueError("canonical center is outside the branch chain")
        leading.requires_grad_(False)
        atom.requires_grad_(False)
        leading.mixed_canonicalize_coefficient_cores_(center)
        leading_scale = leading.normalize_coefficient_chain_scale_(site_index=center)
        add_relative_atom_noise_(atom, args.atom_relative_noise, seed=args.seed)
        atom.mixed_canonicalize_coefficient_cores_(center)
        atom_scale = atom.normalize_coefficient_chain_scale_(site_index=center)
        coherent = PositiveTensorNetworkCoherentSumMetric(
            (leading, atom),
            positive_floor=float(leading.positive_floor),
            initial_new_gates=(0.0j,),
        )

        source_degree = int(leading_payload.get("source_degree", 1))
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
        indices_path = output_dir / "gate_fit_indices.npz"
        np.savez_compressed(
            indices_path,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )

        write_json(status_path, {"state": "running", "phase": "gate_fit"})
        gate_fit = fit_gate(
            coherent,
            train,
            validation,
            fixed_log_kappa=float(leading_payload["fixed_log_kappa"]),
            args=args,
        )
        final_validation = direct_empirical_statistics(
            coherent, validation, chunk_size=args.eval_batch_size
        )

        blind_baseline = None
        blind_final = None
        if not args.skip_blind:
            write_json(status_path, {"state": "running", "phase": "blind_audit"})
            blind_arrays = load_numpy_split(paths, "blind", args.blind_limit)
            blind = make_tensor_split(
                *blind_arrays,
                source_degree=source_degree,
                precision=args.precision,
                device=device,
            )
            saved_gate = coherent.new_gates.detach().clone()
            coherent.set_new_gates_((0.0j,))
            blind_baseline = direct_empirical_statistics(
                coherent, blind, chunk_size=args.eval_batch_size
            )
            coherent.set_new_gates_(saved_gate)
            blind_final = direct_empirical_statistics(
                coherent, blind, chunk_size=args.eval_batch_size
            )

        artifact = {
            "schema": "quintic-positive-tensor-network-coherent-sum-v1",
            "geometry": leading_payload.get("geometry"),
            "source_degree": source_degree,
            "total_degree": int(leading_payload["total_degree"]),
            "site_count": int(leading_payload["site_count"]),
            "target_normalization": float(leading_payload["target_normalization"]),
            "positive_floor": float(coherent.positive_floor),
            "precision": args.precision,
            "fixed_log_kappa": float(leading_payload["fixed_log_kappa"]),
            "branch_states": [
                {key: value.detach().cpu() for key, value in branch.state_dict().items()}
                for branch in coherent.branches
            ],
            "branch_metadata": [
                {
                    "source_model": str(path),
                    "source_model_sha256": sha256_file(path),
                    "bond_dimension": int(payload["bond_dimension"]),
                    "architecture": payload["architecture"],
                    "physical_dictionary_rank": payload.get("physical_dictionary_rank"),
                    "output_dimension": payload.get("output_dimension", 5),
                }
                for path, payload in (
                    (paths["model"], leading_payload),
                    (atom_path, atom_payload),
                )
            ],
            "new_gates": coherent.new_gates.detach().cpu(),
            "leading_scale": leading_scale,
            "atom_scale": atom_scale,
            "atom_relative_noise": args.atom_relative_noise,
        }
        model_path = output_dir / "coherent_model.pt"
        torch.save(artifact, model_path)
        full_active_parameters = sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for branch in coherent.branches
            for parameter in branch.parameters()
        ) + 2 * coherent.new_gates.numel()
        report = {
            "schema": "quintic-tn-coherent-gate-fit-v1",
            "scientific_scope": {
                "purpose": "exact nested coherent-atom gate preflight",
                "identity_at_zero_gate": True,
                "training_stage": (
                    "gate only; atom cores remain frozen until a nonzero gate is accepted"
                ),
                "pool_status": "existing architecture benchmark, not sealed final test",
            },
            "configuration": {
                **{
                    key: value
                    for key, value in vars(args).items()
                    if not isinstance(value, Path)
                },
                "run_dir": str(args.run_dir),
                "atom_model": str(atom_path),
                "output_dir": str(output_dir),
                "leading_model": str(paths["model"]),
                "device": str(device),
                "canonical_center": center,
            },
            "architecture": {
                "branch_count": 2,
                "bond_dimension_per_branch": int(leading_payload["bond_dimension"]),
                "stored_real_parameters_excluding_fixed_dictionaries": full_active_parameters,
                "gate_fit_active_real_parameters": 2,
                "leading_scale": leading_scale,
                "atom_scale": atom_scale,
                "positive_floor_after_scale": float(coherent.positive_floor),
            },
            "source": {
                "leading_model_sha256": sha256_file(paths["model"]),
                "atom_model_sha256": sha256_file(atom_path),
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
            },
            "gate_fit": gate_fit,
            "validation_final": final_validation,
            "benchmark": {
                "baseline": blind_baseline,
                "final": blind_final,
            },
            "artifacts": {
                "model": str(model_path),
                "model_sha256": sha256_file(model_path),
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
                    "accepted": gate_fit["accepted"],
                    "final_gate": gate_fit["final_gate"],
                    "validation_sigma": final_validation["sigma_official_formula"],
                    "blind_baseline_sigma": (
                        None if blind_baseline is None else blind_baseline["sigma_official_formula"]
                    ),
                    "blind_final_sigma": (
                        None if blind_final is None else blind_final["sigma_official_formula"]
                    ),
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
