#!/usr/bin/env python3
"""Train a tiny defect-scaled spectral controller and run a guarded rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    affine_exponential_update,
    affine_log_tangent,
    fit_defect_scaled_spectral_gains,
    get_adapter,
)
from scripts.audit_type11_toeplitz_krylov_rank import (  # noqa: E402
    error_row,
    load_checkpoint,
    parse_integers,
    torch_berezin_geometry,
    torch_berezin_modes_and_lifts,
    torch_coherent_sections,
)
from scripts.audit_type11_toeplitz_krylov_two_pool import (  # noqa: E402
    evaluate_h_on_pool,
    prepare_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--train-iterations", default="18,27,37,46,55")
    parser.add_argument("--holdout-iterations", default="9,64")
    parser.add_argument("--start-iteration", type=int, default=9)
    parser.add_argument("--mode-count", type=int, default=2)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--validation-points", type=int)
    parser.add_argument("--blind-seed", type=int, default=72121)
    parser.add_argument("--blind-points", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--maximum-steps", type=int, default=80)
    parser.add_argument("--balance-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--operator-norm-cap", type=float, default=0.2)
    parser.add_argument("--maximum-step-multiplier", type=float, default=1.0)
    parser.add_argument("--maximum-backtracks", type=int, default=8)
    parser.add_argument("--minimum-relative-defect-improvement", type=float, default=1.0e-5)
    parser.add_argument("--validation-chi-factor", type=float, default=1.25)
    parser.add_argument("--validation-cvar-slack", type=float, default=0.15)
    parser.add_argument("--validation-max-r-factor", type=float, default=1.5)
    parser.add_argument("--power-iterations", type=int, default=24)
    parser.add_argument("--spectral-safety-factor", type=float, default=1.02)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def oracle_rows_by_iteration(oracle: dict) -> dict[int, dict]:
    return {int(row["current_iteration"]): row for row in oracle["pairs"]}


def mode_fit(row: dict, mode_count: int) -> dict:
    fits = row["residual_source_audits"]["balance"]["fit_curve"]
    matches = [fit for fit in fits if int(fit["mode_count"]) == mode_count]
    if len(matches) != 1:
        raise ValueError(f"oracle lacks a unique balance fit for {mode_count} modes")
    return matches[0]


def coefficient_targets(
    rows: dict[int, dict], iterations: list[int], mode_count: int
) -> tuple[np.ndarray, np.ndarray]:
    missing = sorted(set(iterations) - set(rows))
    if missing:
        raise ValueError(f"oracle lacks requested iterations: {missing}")
    defects = []
    coefficients = []
    for iteration in iterations:
        row = rows[iteration]
        defects.append(float(row["balance_defect_frobenius_per_sqrt_n"]))
        coefficients.append(mode_fit(row, mode_count)["coefficients"])
    return np.asarray(defects), np.asarray(coefficients)


def evaluate_pool(pool: dict, h_matrix, *, normalization: float) -> dict:
    import torch

    h_tensor = torch.as_tensor(
        h_matrix,
        dtype=pool["values"].dtype,
        device=pool["values"].device,
    )
    log_eta, minimum = evaluate_h_on_pool(
        pool,
        h_tensor,
        normalization=normalization,
    )
    return error_row(log_eta, pool["weights_numpy"], minimum)


def validation_guard(candidate: dict, current: dict, args: argparse.Namespace) -> bool:
    return bool(
        candidate["sqrt_squared_energy"]
        <= args.validation_chi_factor * current["sqrt_squared_energy"]
        and candidate["positive_log_ratio_cvar_1pct"]
        <= current["positive_log_ratio_cvar_1pct"] + args.validation_cvar_slack
        and candidate["normalized_ratio_max"]
        <= args.validation_max_r_factor * current["normalized_ratio_max"]
    )


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the spectral rollout") from exc

    args = parse_args()
    train_iterations = parse_integers(
        args.train_iterations, label="train-iterations", allow_zero=True
    )
    holdout_iterations = parse_integers(
        args.holdout_iterations, label="holdout-iterations", allow_zero=True
    )
    if set(train_iterations) & set(holdout_iterations):
        raise SystemExit("training and holdout iterations must be disjoint")
    if (
        args.mode_count <= 0
        or args.workers <= 0
        or args.maximum_steps <= 0
        or args.maximum_backtracks < 0
        or args.blind_points <= 0
        or args.blind_points % 4
    ):
        raise SystemExit("invalid positive rollout argument")
    if (
        args.balance_tolerance <= 0
        or args.operator_norm_cap <= 0
        or args.maximum_step_multiplier <= 0
        or not 0 <= args.minimum_relative_defect_improvement < 1
        or min(
            args.validation_chi_factor,
            args.validation_max_r_factor,
            args.spectral_safety_factor,
        )
        < 1
        or args.validation_cvar_slack < 0
    ):
        raise SystemExit("invalid trust-region or validation guard")

    started = time.perf_counter()
    artifact_path = args.artifact.expanduser().resolve()
    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    oracle_path = args.oracle.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    rows = oracle_rows_by_iteration(oracle)
    train_defects, train_coefficients = coefficient_targets(
        rows, train_iterations, args.mode_count
    )
    controller_fit = fit_defect_scaled_spectral_gains(
        train_defects,
        train_coefficients,
    )
    holdout_defects, holdout_coefficients = coefficient_targets(
        rows, holdout_iterations, args.mode_count
    )
    holdout_normalized_targets = holdout_coefficients / holdout_defects[:, None]
    holdout_predictions = holdout_defects[:, None] * controller_fit.gains[None, :]
    holdout_relative_errors = np.linalg.norm(
        holdout_coefficients - holdout_predictions, axis=1
    ) / np.maximum(np.linalg.norm(holdout_coefficients, axis=1), np.finfo(float).tiny)

    mode_seed = int(oracle["mode_pool"]["seed"])
    mode_points = int(oracle["mode_pool"]["points"])
    validation_seed = (
        int(oracle["independent_evaluation_pool"]["seed"])
        if args.validation_seed is None
        else args.validation_seed
    )
    validation_points = (
        int(oracle["independent_evaluation_pool"]["points"])
        if args.validation_points is None
        else args.validation_points
    )
    point_counts = (mode_points, validation_points, args.blind_points)
    if min(point_counts) <= 0 or any(count % 4 for count in point_counts):
        raise SystemExit("all point counts must be positive multiples of four")
    if len({mode_seed, validation_seed, args.blind_seed}) != 3:
        raise SystemExit("mode, validation, and blind seeds must be distinct")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    real_dtype = torch.float32 if args.precision == "complex64" else torch.float64
    complex_dtype = torch.complex64 if args.precision == "complex64" else torch.complex128

    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    geometry = adapter.make_model(args.model_seed, exact=True)
    artifact = adapter.load_h_artifact(artifact_path, geometry)
    if oracle["artifact_sha256"] != sha256_file(artifact_path):
        raise ValueError("oracle and artifact hashes do not match")
    checkpoint_path = trajectory_dir / f"iteration_{args.start_iteration:04d}.npz"
    loaded_iteration, current_h, checkpoint_metadata = load_checkpoint(checkpoint_path)
    if loaded_iteration != args.start_iteration:
        raise ValueError("start checkpoint iteration mismatch")

    print(
        f"controller gains={controller_fit.gains.tolist()} "
        f"train_rms={controller_fit.normalized_root_mean_square_error:.6e}",
        flush=True,
    )
    print(f"preparing mode pool seed={mode_seed} points={mode_points}", flush=True)
    mode_pool = prepare_pool(
        adapter,
        artifact,
        model_seed=args.model_seed,
        count=mode_points,
        seed=mode_seed,
        workers=args.workers,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    print(
        f"preparing validation pool seed={validation_seed} points={validation_points}",
        flush=True,
    )
    validation_pool = prepare_pool(
        adapter,
        artifact,
        model_seed=args.model_seed,
        count=validation_points,
        seed=validation_seed,
        workers=args.workers,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    print(
        f"preparing blind pool seed={args.blind_seed} points={args.blind_points}",
        flush=True,
    )
    blind_pool = prepare_pool(
        adapter,
        artifact,
        model_seed=args.model_seed,
        count=args.blind_points,
        seed=args.blind_seed,
        workers=args.workers,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )

    normalization = float(artifact.normalization)
    initial_h = np.array(current_h, copy=True)
    initial_validation = evaluate_pool(
        validation_pool, current_h, normalization=normalization
    )
    current_validation = initial_validation
    rollout = []
    termination = "maximum_steps"

    for step in range(args.maximum_steps):
        current_h_t = torch.tensor(current_h, dtype=complex_dtype, device=device)
        coherent = torch_coherent_sections(mode_pool["values"], current_h_t)
        radius, balance_residual, defect_t = torch_berezin_geometry(
            coherent,
            mode_pool["weights"],
            power_iterations=args.power_iterations,
        )
        defect = float(defect_t.detach().cpu())
        if defect <= args.balance_tolerance:
            termination = "balance_tolerance"
            break
        _, lifts_t = torch_berezin_modes_and_lifts(
            coherent,
            mode_pool["weights"],
            balance_residual,
            mode_count=args.mode_count,
            spectral_radius=radius,
            spectral_safety_factor=args.spectral_safety_factor,
        )
        coefficients = defect * controller_fit.gains
        tangent_t = torch.einsum(
            "i,iab->ab",
            torch.tensor(coefficients, dtype=complex_dtype, device=device),
            lifts_t,
        )
        tangent = tangent_t.detach().cpu().numpy().astype(np.complex128)
        tangent_eigenvalues = np.linalg.eigvalsh(tangent)
        operator_norm = float(np.max(np.abs(tangent_eigenvalues)))
        trust_scale = min(
            args.maximum_step_multiplier,
            args.operator_norm_cap / max(operator_norm, 1.0e-300),
        )

        defect_candidates = []
        for backtrack in range(args.maximum_backtracks + 1):
            scale = trust_scale * 0.5**backtrack
            candidate_h = affine_exponential_update(current_h, scale * tangent)
            candidate_h_t = torch.tensor(
                candidate_h, dtype=complex_dtype, device=device
            )
            candidate_coherent = torch_coherent_sections(
                mode_pool["values"], candidate_h_t
            )
            _, _, candidate_defect_t = torch_berezin_geometry(
                candidate_coherent,
                mode_pool["weights"],
                power_iterations=args.power_iterations,
            )
            candidate_defect = float(candidate_defect_t.detach().cpu())
            required = defect * (1.0 - args.minimum_relative_defect_improvement)
            if candidate_defect >= required:
                continue
            defect_candidates.append(
                (candidate_h, candidate_h_t, candidate_defect, scale, backtrack)
            )

        accepted_candidates = []
        for (
            candidate_h,
            candidate_h_t,
            candidate_defect,
            scale,
            backtrack,
        ) in sorted(defect_candidates, key=lambda candidate: candidate[2]):
            try:
                candidate_validation = evaluate_pool(
                    validation_pool,
                    candidate_h_t,
                    normalization=normalization,
                )
            except FloatingPointError:
                continue
            if not validation_guard(candidate_validation, current_validation, args):
                continue
            accepted_candidates.append(
                (
                    candidate_h,
                    candidate_defect,
                    candidate_validation,
                    scale,
                    backtrack,
                )
            )
            break

        if not accepted_candidates:
            termination = "trust_region_rejected"
            break
        candidate_h, candidate_defect, candidate_validation, scale, backtrack = min(
            accepted_candidates,
            key=lambda candidate: candidate[1],
        )
        rollout.append(
            {
                "step": step,
                "balance_defect_before": defect,
                "balance_defect_after": candidate_defect,
                "berezin_spectral_radius": float(radius.detach().cpu()),
                "predicted_coefficients": coefficients.tolist(),
                "tangent_frobenius_norm": float(np.linalg.norm(tangent, ord="fro")),
                "tangent_operator_norm": operator_norm,
                "accepted_scale": scale,
                "backtracks": backtrack,
                "validation_ma_errors": candidate_validation,
            }
        )
        print(
            f"step={step} defect={defect:.6e}->{candidate_defect:.6e} "
            f"scale={scale:.3e} chi={candidate_validation['sqrt_squared_energy']:.6e} "
            f"max_r={candidate_validation['normalized_ratio_max']:.6e}",
            flush=True,
        )
        current_h = candidate_h
        current_validation = candidate_validation
        del current_h_t, coherent, lifts_t, tangent_t

    final_h_t = torch.tensor(current_h, dtype=complex_dtype, device=device)
    final_coherent = torch_coherent_sections(mode_pool["values"], final_h_t)
    _, _, final_defect_t = torch_berezin_geometry(
        final_coherent,
        mode_pool["weights"],
        power_iterations=args.power_iterations,
    )
    final_defect = float(final_defect_t.detach().cpu())
    if final_defect <= args.balance_tolerance:
        termination = "balance_tolerance"

    initial_blind = evaluate_pool(blind_pool, initial_h, normalization=normalization)
    final_blind = evaluate_pool(blind_pool, final_h_t, normalization=normalization)
    teacher_h_t = torch.tensor(
        artifact.h_matrix, dtype=complex_dtype, device=device
    )
    teacher_blind = evaluate_pool(blind_pool, teacher_h_t, normalization=normalization)
    distance_to_teacher = float(
        np.linalg.norm(affine_log_tangent(artifact.h_matrix, current_h), ord="fro")
        / np.sqrt(artifact.section_count)
    )
    distance_from_start = float(
        np.linalg.norm(affine_log_tangent(initial_h, current_h), ord="fro")
        / np.sqrt(artifact.section_count)
    )

    holdout_rows = []
    for index, iteration in enumerate(holdout_iterations):
        holdout_rows.append(
            {
                "iteration": iteration,
                "balance_defect": float(holdout_defects[index]),
                "normalized_target_coefficients": holdout_normalized_targets[
                    index
                ].tolist(),
                "predicted_coefficients": holdout_predictions[index].tolist(),
                "relative_coefficient_error": float(holdout_relative_errors[index]),
                "explained_squared_fraction": float(
                    mode_fit(rows[iteration], args.mode_count)[
                        "explained_squared_fraction"
                    ]
                ),
            }
        )

    report = {
        "schema": "type11-defect-scaled-toeplitz-controller-rollout-v1",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "trajectory_directory": str(trajectory_dir),
        "trajectory_metadata": checkpoint_metadata,
        "oracle": str(oracle_path),
        "oracle_sha256": sha256_file(oracle_path),
        "controller": {
            "family": "constant defect-scaled balance-residual spectral gains",
            "mode_count": args.mode_count,
            "trainable_parameter_count": args.mode_count,
            "gains": controller_fit.gains.tolist(),
            "training_iterations": train_iterations,
            "training_normalized_rms": controller_fit.normalized_root_mean_square_error,
            "holdout": holdout_rows,
        },
        "start_iteration": args.start_iteration,
        "mode_pool": {
            "seed": mode_seed,
            "points": mode_points,
            "role": "controller state and trust-region balance defect",
        },
        "validation_pool": {
            "seed": validation_seed,
            "points": validation_points,
            "role": "tail guard only; excluded from coefficient fitting",
        },
        "blind_pool": {
            "seed": args.blind_seed,
            "points": args.blind_points,
            "role": "evaluated only after rollout",
        },
        "trust_region": {
            "operator_norm_cap": args.operator_norm_cap,
            "maximum_step_multiplier": args.maximum_step_multiplier,
            "maximum_backtracks": args.maximum_backtracks,
            "minimum_relative_defect_improvement": args.minimum_relative_defect_improvement,
            "validation_chi_factor": args.validation_chi_factor,
            "validation_cvar_slack": args.validation_cvar_slack,
            "validation_max_r_factor": args.validation_max_r_factor,
        },
        "termination_reason": termination,
        "accepted_steps": len(rollout),
        "initial_balance_defect": float(
            rows[args.start_iteration]["balance_defect_frobenius_per_sqrt_n"]
        ),
        "final_balance_defect": final_defect,
        "affine_distance_per_sqrt_n_to_t_map_teacher": distance_to_teacher,
        "affine_distance_per_sqrt_n_from_start": distance_from_start,
        "validation": {
            "initial": initial_validation,
            "final": current_validation,
        },
        "blind": {
            "initial": initial_blind,
            "controller_final": final_blind,
            "full_t_map_teacher": teacher_blind,
        },
        "rollout": rollout,
        "device": str(device),
        "precision": args.precision,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(out_path, report)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
