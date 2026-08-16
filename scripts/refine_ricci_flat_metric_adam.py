#!/usr/bin/env python3
"""Refine the local gCICY Ricci-flat approximation with Adam on the MA loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    apply_correction,
    baseline_metrics,
    basis_hessians,
    embedding,
    holomorphic_volume_log_density,
    linearized_fit,
    random_parameters,
    residual_stats,
    residual_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--validation", type=int, default=256)
    parser.add_argument("--export", type=int, default=256)
    parser.add_argument("--guard-points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--validation-seed", type=int, default=222)
    parser.add_argument("--export-seed", type=int, default=333)
    parser.add_argument("--guard-seeds", default="444,555,666,777,888,999")
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--linearized-iterations", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--validation-weight", type=float, default=0.35)
    parser.add_argument("--barrier-weight", type=float, default=100.0)
    parser.add_argument("--drift-weight", type=float, default=1e-4)
    parser.add_argument("--min-eigenvalue", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_metric_adam_refined.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_adam_refined_summary.json")
    return parser.parse_args()


def stats_dict(stats) -> dict[str, float]:
    return {
        "rms": stats.rms,
        "max_abs": stats.max_abs,
        "min_eigenvalue": stats.min_eigenvalue,
        "mean_log_error": stats.mean_log_error,
    }


def centered(values: np.ndarray) -> np.ndarray:
    return values - float(np.mean(values))


def parse_guard_seeds(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def summarize_guard(stats_list) -> dict[str, float | int]:
    if not stats_list:
        return {"count": 0, "mean_rms": float("nan"), "max_rms": float("nan"), "min_eigenvalue": float("nan")}
    finite_rms = [stats.rms for stats in stats_list if np.isfinite(stats.rms)]
    return {
        "count": len(stats_list),
        "finite_count": len(finite_rms),
        "mean_rms": float(np.mean(finite_rms)) if finite_rms else float("nan"),
        "max_rms": float(np.max(finite_rms)) if finite_rms else float("nan"),
        "min_eigenvalue": float(np.min([stats.min_eigenvalue for stats in stats_list])),
    }


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for Adam refinement.") from exc

    args = parse_args()
    torch.set_default_dtype(torch.float64)

    train_params = random_parameters(args.train, seed=args.seed, scale=args.scale)
    validation_params = random_parameters(args.validation, seed=args.validation_seed, scale=args.scale)
    export_params = random_parameters(args.export, seed=args.export_seed, scale=args.scale)
    guard_seeds = parse_guard_seeds(args.guard_seeds)
    guard_params = [random_parameters(args.guard_points, seed=seed, scale=args.scale) for seed in guard_seeds]

    initial = linearized_fit(
        train_params,
        validation_params,
        max_degree=args.degree,
        l2=args.l2,
        iterations=args.linearized_iterations,
        min_eigenvalue=1e-5,
    )
    exponents = initial.exponents
    initial_coefficients = initial.coefficients.astype(float)

    train_base = baseline_metrics(train_params)
    validation_base = baseline_metrics(validation_params)
    export_base = baseline_metrics(export_params)
    guard_base = [baseline_metrics(params) for params in guard_params]
    train_basis = basis_hessians(train_params, exponents)
    validation_basis = basis_hessians(validation_params, exponents)
    export_basis = basis_hessians(export_params, exponents)
    guard_basis = [basis_hessians(params, exponents) for params in guard_params]

    train_log_omega = np.asarray([holomorphic_volume_log_density(point) for point in train_params], dtype=float)
    validation_log_omega = np.asarray(
        [holomorphic_volume_log_density(point) for point in validation_params], dtype=float
    )

    train_base_t = torch.tensor(train_base, dtype=torch.complex128)
    validation_base_t = torch.tensor(validation_base, dtype=torch.complex128)
    train_basis_t = torch.tensor(train_basis, dtype=torch.complex128)
    validation_basis_t = torch.tensor(validation_basis, dtype=torch.complex128)
    train_log_omega_t = torch.tensor(train_log_omega)
    validation_log_omega_t = torch.tensor(validation_log_omega)
    initial_coefficients_t = torch.tensor(initial_coefficients)
    coefficients = torch.nn.Parameter(initial_coefficients_t.clone())
    optimizer = torch.optim.Adam([coefficients], lr=args.lr)

    def torch_residual(metric_base, basis, log_omega, coeffs):
        metric = metric_base + torch.einsum("m,nmij->nij", coeffs.to(torch.complex128), basis)
        eigvals = torch.linalg.eigvalsh(metric)
        logdet = torch.log(torch.clamp(eigvals, min=1e-12)).sum(dim=1)
        raw = logdet - log_omega
        residual = raw - raw.mean()
        barrier = torch.nn.functional.softplus((args.min_eigenvalue - eigvals) * 50.0).mean() / 50.0
        return residual, eigvals, barrier

    def evaluate(coeffs: np.ndarray):
        train_metric = apply_correction(train_base, train_basis, coeffs)
        validation_metric = apply_correction(validation_base, validation_basis, coeffs)
        export_metric = apply_correction(export_base, export_basis, coeffs)
        guard_stats = [
            residual_stats(params, apply_correction(base, basis, coeffs))
            for params, base, basis in zip(guard_params, guard_base, guard_basis, strict=True)
        ]
        return (
            residual_stats(train_params, train_metric),
            residual_stats(validation_params, validation_metric),
            residual_stats(export_params, export_metric),
            guard_stats,
        )

    best_coefficients = initial_coefficients.copy()
    best_train, best_validation, best_export, best_guard = evaluate(best_coefficients)
    best_guard_summary = summarize_guard(best_guard)
    initial_guard_summary = dict(best_guard_summary)
    history: list[dict[str, float | int | bool]] = [
        {
            "epoch": -1,
            "accepted": True,
            "train_rms": best_train.rms,
            "validation_rms": best_validation.rms,
            "export_rms": best_export.rms,
            "train_min_eigenvalue": best_train.min_eigenvalue,
            "validation_min_eigenvalue": best_validation.min_eigenvalue,
            "export_min_eigenvalue": best_export.min_eigenvalue,
            "guard_mean_rms": best_guard_summary["mean_rms"],
            "guard_max_rms": best_guard_summary["max_rms"],
            "guard_min_eigenvalue": best_guard_summary["min_eigenvalue"],
        }
    ]

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        train_residual, _, train_barrier = torch_residual(
            train_base_t, train_basis_t, train_log_omega_t, coefficients
        )
        validation_residual, _, validation_barrier = torch_residual(
            validation_base_t, validation_basis_t, validation_log_omega_t, coefficients
        )
        drift = torch.mean((coefficients - initial_coefficients_t) ** 2)
        loss = (
            torch.mean(train_residual**2)
            + args.validation_weight * torch.mean(validation_residual**2)
            + args.barrier_weight * (train_barrier + validation_barrier)
            + args.drift_weight * drift
        )
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every != 0:
            continue

        current_coefficients = coefficients.detach().numpy().astype(float)
        train_stats, validation_stats, export_stats, guard_stats = evaluate(current_coefficients)
        guard_summary = summarize_guard(guard_stats)
        accepted = (
            validation_stats.min_eigenvalue > args.min_eigenvalue
            and export_stats.min_eigenvalue > args.min_eigenvalue
            and guard_summary["finite_count"] == guard_summary["count"]
            and guard_summary["min_eigenvalue"] > args.min_eigenvalue
            and np.isfinite(validation_stats.rms)
            and np.isfinite(export_stats.rms)
            and validation_stats.rms < best_validation.rms
            and export_stats.rms < best_export.rms
            and guard_summary["mean_rms"] < best_guard_summary["mean_rms"]
        )
        if accepted:
            best_coefficients = current_coefficients.copy()
            best_train = train_stats
            best_validation = validation_stats
            best_export = export_stats
            best_guard = guard_stats
            best_guard_summary = guard_summary
        history.append(
            {
                "epoch": epoch,
                "accepted": bool(accepted),
                "loss": float(loss.detach().numpy()),
                "train_rms": train_stats.rms,
                "validation_rms": validation_stats.rms,
                "export_rms": export_stats.rms,
                "train_min_eigenvalue": train_stats.min_eigenvalue,
                "validation_min_eigenvalue": validation_stats.min_eigenvalue,
                "export_min_eigenvalue": export_stats.min_eigenvalue,
                "guard_mean_rms": guard_summary["mean_rms"],
                "guard_max_rms": guard_summary["max_rms"],
                "guard_min_eigenvalue": guard_summary["min_eigenvalue"],
            }
        )

    export_corrected = apply_correction(export_base, export_basis, best_coefficients)
    baseline_export_values = residual_values(export_params, export_base)
    corrected_export_values = residual_values(export_params, export_corrected)
    ambient_coords = np.asarray([embedding(point) for point in export_params], dtype=np.complex128)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        params=export_params,
        ambient_coords=ambient_coords,
        baseline_metrics=export_base,
        corrected_metrics=export_corrected,
        correction_metrics=export_corrected - export_base,
        initial_coefficients=initial_coefficients,
        coefficients=best_coefficients,
        exponents=exponents,
        baseline_log_ma=baseline_export_values,
        baseline_centered_log_ma=centered(baseline_export_values),
        corrected_log_ma=corrected_export_values,
        corrected_centered_log_ma=centered(corrected_export_values),
    )

    initial_export = residual_stats(export_params, apply_correction(export_base, export_basis, initial_coefficients))
    summary = {
        "description": "Adam refinement of a local gCICY Ricci-flat metric approximation.",
        "method": "2022-style MA-loss optimization of an algebraic Kahler-potential ansatz",
        "ansatz": "g = g0 + partial partialbar phi, with phi a real polynomial in local coordinates",
        "is_exact_ricci_flat": False,
        "is_kahler_local": True,
        "train_points": args.train,
        "validation_points": args.validation,
        "export_points": args.export,
        "guard_points_per_seed": args.guard_points,
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "export_seed": args.export_seed,
        "guard_seeds": guard_seeds,
        "scale": args.scale,
        "degree": args.degree,
        "basis_size": int(len(exponents)),
        "linearized_l2": args.l2,
        "linearized_iterations": args.linearized_iterations,
        "adam_epochs": args.epochs,
        "adam_lr": args.lr,
        "min_eigenvalue_gate": args.min_eigenvalue,
        "initial_train": stats_dict(initial.best_train),
        "initial_validation": stats_dict(initial.best_validation),
        "initial_export": stats_dict(initial_export),
        "initial_guard": initial_guard_summary,
        "refined_train": stats_dict(best_train),
        "refined_validation": stats_dict(best_validation),
        "refined_export": stats_dict(best_export),
        "refined_guard": best_guard_summary,
        "history": history,
        "npz_file": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(f"validation MA rms: {initial.best_validation.rms:.6e} -> {best_validation.rms:.6e}")
    print(f"export MA rms: {initial_export.rms:.6e} -> {best_export.rms:.6e}")
    print(f"refined export min eigenvalue: {best_export.min_eigenvalue:.6e}")
    print("metric type: local Kahler MA-loss approximation, not an exact certified Ricci-flat metric")


if __name__ == "__main__":
    main()
