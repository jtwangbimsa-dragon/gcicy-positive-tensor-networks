#!/usr/bin/env python3
"""Audit residual reachability of exact Fermat-invariant two-site directions."""

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

from gcicy_metric.pipeline import (  # noqa: E402
    fermat_root_virtual_charges,
    matrix_unit_phase_charges,
    phase_charge_s5_orbit_key,
    phase_s5_two_site_orbit_labels,
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
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_tn_one_site_lm import (  # noqa: E402
    direct_empirical_statistics,
    load_training_arrays,
    random_subset,
    tail_summary,
)
from scripts.train_quintic_tn_two_site_lm import (  # noqa: E402
    fixed_model_residual_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--blind-reference-run-dir", type=Path)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-start", type=int, default=14)
    parser.add_argument("--probe-size", type=int, default=2000)
    parser.add_argument("--holdout-size", type=int, default=2000)
    parser.add_argument("--operator-chunk-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lanczos-steps", type=int, default=8)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(100.0, 10.0, 1.0),
    )
    parser.add_argument(
        "--line-search-alphas",
        type=float,
        nargs="+",
        default=(0.125, 0.25, 0.5),
    )
    parser.add_argument("--maximum-relative-step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=202607223)
    parser.add_argument(
        "--precision",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.probe_size,
        args.holdout_size,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.lanczos_steps,
        args.maximum_relative_step,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sample, batch, Lanczos, and step sizes must be positive")
    if not args.ridge_factors or any(value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be positive")
    if not args.line_search_alphas or any(
        value <= 0 or value > 1 for value in args.line_search_alphas
    ):
        raise ValueError("line-search alphas must lie in (0, 1]")


def phase_block_spectrum(
    core: torch.Tensor,
    *,
    left_boundary: bool,
    right_boundary: bool,
    multiplicity: int,
    relative_cutoff: float,
) -> dict[str, object]:
    """Compute the Schmidt spectrum separately in every intermediate charge."""

    boundary = np.zeros((1, 4), dtype=np.int64)
    virtual = fermat_root_virtual_charges(multiplicity)
    left_charges = boundary if left_boundary else virtual
    right_charges = boundary if right_boundary else virtual
    local = matrix_unit_phase_charges()
    row_groups: dict[tuple[int, ...], list[int]] = {}
    for left_index, left_charge in enumerate(left_charges):
        for local_index, local_charge in enumerate(local):
            charge = tuple(int(value) for value in (left_charge + local_charge) % 5)
            row_groups.setdefault(charge, []).append(left_index * 25 + local_index)
    column_groups: dict[tuple[int, ...], list[int]] = {}
    right_count = len(right_charges)
    for local_index, local_charge in enumerate(local):
        for right_index, right_charge in enumerate(right_charges):
            charge = tuple(int(value) for value in (right_charge - local_charge) % 5)
            column_groups.setdefault(charge, []).append(
                local_index * right_count + right_index
            )
    matrix = core.permute(0, 2, 3, 1).reshape(
        len(left_charges) * 25,
        25 * len(right_charges),
    )
    rows = []
    total_squared_weight = 0.0
    retained_squared_weight = 0.0
    total_rank = 0
    for charge in sorted(set(row_groups) & set(column_groups)):
        row_indices = torch.as_tensor(
            row_groups[charge], dtype=torch.int64, device=core.device
        )
        column_indices = torch.as_tensor(
            column_groups[charge], dtype=torch.int64, device=core.device
        )
        block = matrix.index_select(0, row_indices).index_select(1, column_indices)
        singular_values = torch.linalg.svdvals(block)
        largest = float(singular_values[0]) if singular_values.numel() else 0.0
        threshold = relative_cutoff * largest
        numerical_rank = int(torch.count_nonzero(singular_values > threshold))
        squared = torch.square(singular_values).detach().cpu().numpy().astype(float)
        total_squared_weight += float(np.sum(squared))
        retained_squared_weight += float(np.sum(squared[:numerical_rank]))
        total_rank += numerical_rank
        rows.append(
            {
                "charge": list(charge),
                "s5_orbit_key": list(phase_charge_s5_orbit_key(charge)),
                "block_shape": [len(row_indices), len(column_indices)],
                "numerical_rank": numerical_rank,
                "singular_values": [float(value) for value in singular_values],
                "squared_weight": float(np.sum(squared)),
            }
        )
    orbit_summary: dict[tuple[int, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row["s5_orbit_key"])
        summary = orbit_summary.setdefault(
            key,
            {
                "s5_orbit_key": list(key),
                "sector_count": 0,
                "numerical_ranks": [],
                "total_squared_weight": 0.0,
            },
        )
        summary["sector_count"] += 1
        summary["numerical_ranks"].append(row["numerical_rank"])
        summary["total_squared_weight"] += row["squared_weight"]
    return {
        "relative_singular_value_cutoff": relative_cutoff,
        "intermediate_sector_count": len(rows),
        "total_retained_rank": total_rank,
        "retained_squared_weight_fraction": (
            1.0
            if total_squared_weight == 0.0
            else retained_squared_weight / total_squared_weight
        ),
        "s5_orbits": list(orbit_summary.values()),
        "sectors": rows,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    paths = resolve_inputs(args)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    payload = torch.load(paths["model"], map_location="cpu", weights_only=False)
    payload = cast_artifact_precision(payload, args.precision)
    multiplicity = int(payload.get("fermat_phase_charge_multiplicity", 0))
    if multiplicity <= 0 or not bool(payload.get("fermat_s5_orbit_tying", False)):
        raise ValueError("the model is not a hard Fermat phase-and-S5 tensor network")
    model = build_model(payload, device)
    if args.pair_start < 0 or args.pair_start + 1 >= model.site_count:
        raise ValueError("two-site pair is outside the coefficient chain")
    model.eval()
    model.requires_grad_(False)

    probe_arrays, probe_indices = random_subset(
        load_training_arrays(paths),
        args.probe_size,
        seed=args.seed,
    )
    holdout_arrays, holdout_indices = random_subset(
        load_numpy_split(paths, "validation", 0),
        args.holdout_size,
        seed=args.seed + 1,
    )
    source_degree = int(payload.get("source_degree", 1))
    probe = make_tensor_split(
        *probe_arrays,
        source_degree=source_degree,
        precision=args.precision,
        device=device,
    )
    holdout = make_tensor_split(
        *holdout_arrays,
        source_degree=source_degree,
        precision=args.precision,
        device=device,
    )
    fixed_log_kappa = float(payload["fixed_log_kappa"])
    baseline_probe = fixed_model_residual_metrics(
        model,
        probe,
        fixed_log_kappa=fixed_log_kappa,
        chunk_size=args.eval_batch_size,
    )
    baseline_holdout = direct_empirical_statistics(
        model,
        holdout,
        chunk_size=args.eval_batch_size,
    )
    baseline_tail = tail_summary(baseline_holdout)

    labels = phase_s5_two_site_orbit_labels(
        left_boundary=args.pair_start == 0,
        right_boundary=args.pair_start + 1 == model.site_count - 1,
        multiplicity=multiplicity,
    )
    active, projection_error = model.activate_two_site_coefficient_orbits_(
        args.pair_start,
        labels,
    )
    active.requires_grad_(True)
    tolerance = 2.0e-5 if args.precision == "complex64" else 2.0e-12
    if projection_error > tolerance:
        raise RuntimeError(
            "the factorized hard-symmetry baseline is not in the two-site "
            f"orbit space: relative error {projection_error}"
        )
    vectorizer = ComplexParameterVectorizer.from_module(
        model,
        parameter_names=("two_site_orbit_parameters",),
    )
    theta = vectorizer.pack(model)
    operator = MatrixFreeResidualJacobian(
        model,
        vectorizer,
        theta,
        probe,
        fixed_log_kappa=fixed_log_kappa,
        chunk_size=args.operator_chunk_size,
    )
    residual = operator.residual()
    baseline_fixed = fixed_residual_metrics(residual)
    gradient = operator.vjp(residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(
        baseline_fixed["squared_norm"],
        np.finfo(float).tiny,
    )
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        raise FloatingPointError("two-site orbit space has no finite descent direction")

    def progress(iteration: int, alpha: float, beta: float) -> None:
        if not args.quiet:
            print(
                f"lanczos={iteration}/{args.lanczos_steps} "
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
    for factor in args.ridge_factors:
        ridge = factor * rayleigh_scale
        dual = ridge_dual_from_lanczos(lanczos, ridge)
        delta = -operator.vjp(dual)
        predicted = residual + operator.jvp(delta)
        predicted_capture = 1.0 - float(real_inner(predicted, predicted)) / max(
            baseline_fixed["squared_norm"],
            np.finfo(float).tiny,
        )
        for alpha in args.line_search_alphas:
            candidate = theta + alpha * delta
            relative_step = float(
                alpha
                * vector_norm(delta)
                / max(theta_norm, np.finfo(float).tiny)
            )
            with torch.no_grad():
                active.copy_(candidate)
                probe_fixed = fixed_model_residual_metrics(
                    model,
                    probe,
                    fixed_log_kappa=fixed_log_kappa,
                    chunk_size=args.eval_batch_size,
                )
                holdout_metrics = direct_empirical_statistics(
                    model,
                    holdout,
                    chunk_size=args.eval_batch_size,
                )
                active.copy_(theta)
            observed_tail = tail_summary(holdout_metrics)
            candidates.append(
                {
                    "ridge_factor": float(factor),
                    "ridge": float(ridge),
                    "alpha": float(alpha),
                    "relative_step": relative_step,
                    "predicted_probe_capture_at_alpha_1": predicted_capture,
                    "probe_fixed": probe_fixed,
                    "holdout": holdout_metrics,
                    "holdout_tail": observed_tail,
                    "bulk_improved": bool(
                        holdout_metrics["weighted_rms_abs_residual"]
                        < baseline_holdout["weighted_rms_abs_residual"]
                    ),
                    "all_tail_metrics_nonworse": bool(
                        all(observed_tail[key] <= baseline_tail[key] for key in baseline_tail)
                    ),
                    "step_gate_passed": relative_step <= args.maximum_relative_step,
                    "parameter_vector": candidate.detach().clone(),
                }
            )
    with torch.no_grad():
        active.copy_(theta)
    selected = min(
        (
            row
            for row in candidates
            if row["bulk_improved"]
            and row["all_tail_metrics_nonworse"]
            and row["step_gate_passed"]
        ),
        key=lambda row: row["holdout"]["weighted_rms_abs_residual"],
        default=None,
    )

    hidden = {"parameter_vector"}
    serializable_candidates = [
        {key: value for key, value in row.items() if key not in hidden}
        for row in candidates
    ]
    serializable_selected = (
        None
        if selected is None
        else {key: value for key, value in selected.items() if key not in hidden}
    )
    spectrum = None
    if selected is not None:
        with torch.no_grad():
            active.copy_(selected["parameter_vector"])
            selected_core = model.active_two_site_coefficient_core().detach().clone()
            active.copy_(theta)
        cutoff = 1.0e-5 if args.precision == "complex64" else 1.0e-11
        spectrum = {
            "baseline": phase_block_spectrum(
                model._materialize_orbit_tensor(theta, model.two_site_orbit_labels),
                left_boundary=args.pair_start == 0,
                right_boundary=args.pair_start + 1 == model.site_count - 1,
                multiplicity=multiplicity,
                relative_cutoff=cutoff,
            ),
            "selected": phase_block_spectrum(
                selected_core,
                left_boundary=args.pair_start == 0,
                right_boundary=args.pair_start + 1 == model.site_count - 1,
                multiplicity=multiplicity,
                relative_cutoff=cutoff,
            ),
        }

    report = {
        "schema": "quintic-tn-symmetry-blocked-two-site-audit-v1",
        "scientific_scope": {
            "purpose": (
                "Measure whether the complete local Fermat-invariant two-site "
                "space contains residual-aligned directions beyond a factorized pair."
            ),
            "claim_limit": (
                "This is a tangent-cone candidate audit; no enlarged bond is "
                "committed and no rank-improvement claim follows without a split."
            ),
        },
        "configuration": {
            **vars(args),
            "run_dir": str(args.run_dir),
            "output": str(args.output),
            "source_run_dir": str(paths["source"]),
            "pullbacks_dir": str(paths["pullbacks"]),
            "model": str(paths["model"]),
            "device": str(device),
        },
        "source": {
            "model_sha256": sha256_file(paths["model"]),
            "phase_charge_multiplicity": multiplicity,
            "probe_indices": probe_indices.tolist(),
            "holdout_indices": holdout_indices.tolist(),
        },
        "candidate_space": {
            "allowed_dense_complex_entries": int(np.count_nonzero(labels >= 0)),
            "independent_complex_orbits": int(active.numel()),
            "independent_real_parameters": int(2 * active.numel()),
            "factorized_baseline_projection_relative_error": projection_error,
        },
        "baseline": {
            "probe_fixed_before_activation": baseline_probe,
            "probe_fixed": baseline_fixed,
            "holdout": baseline_holdout,
            "holdout_tail": baseline_tail,
            "jt_residual_norm": gradient_norm,
            "residual_rayleigh_scale": rayleigh_scale,
        },
        "lanczos": {
            "steps_completed": lanczos.operator_applications,
            "breakdown": lanczos.breakdown,
            "orthogonality_error": lanczos.orthogonality_error,
        },
        "candidates": serializable_candidates,
        "selected_pareto_candidate": serializable_selected,
        "phase_block_schmidt_spectrum": spectrum,
        "timing_seconds": time.perf_counter() - started,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "independent_complex_orbits": int(active.numel()),
                "projection_error": projection_error,
                "baseline_holdout_chi": baseline_holdout[
                    "weighted_rms_abs_residual"
                ],
                "selected": serializable_selected,
                "timing_seconds": report["timing_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
