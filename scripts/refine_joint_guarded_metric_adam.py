#!/usr/bin/env python3
"""Adam-refine the conservative joint metric while preserving all audit gates."""

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
    holomorphic_volume_log_density,
    random_parameters,
    random_parameters_log_annulus,
    residual_stats,
    residual_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_metric_joint_guarded.npz")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_metric_joint_guarded_adam.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_joint_guarded_adam_summary.json")
    return parser.parse_args()


def local(n_points: int, seed: int) -> np.ndarray:
    return random_parameters(n_points, seed=seed, scale=0.35)


def radial(n_points: int, seed: int) -> np.ndarray:
    return random_parameters_log_annulus(n_points, seed=seed, coordinate_scale=0.3, r_min=0.05, r_max=1.5)


def mixed(local_seed: int, radial_seed: int) -> np.ndarray:
    return np.vstack([local(256, local_seed), radial(512, radial_seed)])


def prepare(params: np.ndarray, exponents: np.ndarray):
    return (
        params,
        baseline_metrics(params),
        basis_hessians(params, exponents),
        np.asarray([holomorphic_volume_log_density(point) for point in params], dtype=float),
    )


def stats_dict(stats) -> dict[str, float]:
    return {
        "rms": stats.rms,
        "max_abs": stats.max_abs,
        "min_eigenvalue": stats.min_eigenvalue,
        "mean_log_error": stats.mean_log_error,
    }


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for Adam refinement.") from exc

    args = parse_args()
    torch.set_default_dtype(torch.float64)
    artifact = np.load(args.base)
    initial_coefficients = artifact["coefficients"].astype(float)
    exponents = artifact["exponents"]

    train = prepare(np.vstack([local(1024, 11), radial(2048, 12), mixed(333, 334), mixed(444, 445)]), exponents)
    validation = prepare(np.vstack([mixed(222, 223), mixed(101, 202)]), exponents)
    checks = [
        ("local444", prepare(local(256, 444), exponents)),
        ("radial101", prepare(radial(256, 101), exponents)),
        ("mixed222_223", prepare(mixed(222, 223), exponents)),
        ("mixed333_334", prepare(mixed(333, 334), exponents)),
    ]
    export = prepare(np.vstack([mixed(222, 223), mixed(333, 334), local(256, 444), radial(256, 101)]), exponents)

    coefficients = torch.nn.Parameter(torch.tensor(initial_coefficients.copy()))
    initial_coefficients_t = torch.tensor(initial_coefficients.copy())
    optimizer = torch.optim.Adam([coefficients], lr=args.lr)

    def dataset_loss(dataset):
        _, base, basis, log_omega = dataset
        metric = torch.tensor(base, dtype=torch.complex128)
        basis_t = torch.tensor(basis, dtype=torch.complex128)
        log_omega_t = torch.tensor(log_omega)
        corrected = metric + torch.einsum("m,nmij->nij", coefficients.to(torch.complex128), basis_t)
        eigvals = torch.linalg.eigvalsh(corrected)
        raw = torch.log(torch.clamp(eigvals, min=1e-12)).sum(dim=1) - log_omega_t
        centered = raw - raw.mean()
        barrier = torch.nn.functional.softplus((1e-4 - eigvals) * 80.0).mean() / 80.0
        return torch.mean(centered**2), barrier

    def evaluate(dataset, coeffs: np.ndarray):
        params, base, basis, _ = dataset
        return residual_stats(params, apply_correction(base, basis, coeffs))

    best_coefficients = initial_coefficients.copy()
    best_check_stats = [evaluate(dataset, best_coefficients) for _, dataset in checks]
    history: list[dict[str, float | int | bool]] = []

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        train_loss, train_barrier = dataset_loss(train)
        validation_loss, validation_barrier = dataset_loss(validation)
        drift = torch.mean((coefficients - initial_coefficients_t) ** 2)
        loss = train_loss + 0.75 * validation_loss + 200.0 * (train_barrier + validation_barrier) + 1e-5 * drift
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every != 0:
            continue

        current_coefficients = coefficients.detach().numpy().astype(float)
        check_stats = [evaluate(dataset, current_coefficients) for _, dataset in checks]
        accepted = all(
            np.isfinite(current.rms)
            and current.min_eigenvalue > 1e-5
            and current.rms < previous.rms
            for current, previous in zip(check_stats, best_check_stats, strict=True)
        )
        if accepted:
            best_coefficients = current_coefficients.copy()
            best_check_stats = check_stats
        row = {"epoch": epoch, "accepted": bool(accepted), "loss": float(loss.detach().numpy())}
        for (name, _), stats in zip(checks, check_stats, strict=True):
            row[f"{name}_rms"] = stats.rms
            row[f"{name}_min_eigenvalue"] = stats.min_eigenvalue
        history.append(row)

    export_base_stats = residual_stats(export[0], export[1])
    export_corrected = apply_correction(export[1], export[2], best_coefficients)
    export_stats = residual_stats(export[0], export_corrected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        params=export[0],
        baseline_metrics=export[1],
        corrected_metrics=export_corrected,
        correction_metrics=export_corrected - export[1],
        coefficients=best_coefficients,
        exponents=exponents,
        baseline_log_ma=residual_values(export[0], export[1]),
        corrected_log_ma=residual_values(export[0], export_corrected),
    )
    summary = {
        "description": "Early-stopped Adam refinement of the conservative joint guarded metric.",
        "base_artifact": str(args.base),
        "epochs": args.epochs,
        "lr": args.lr,
        "baseline_export": stats_dict(export_base_stats),
        "corrected_export": stats_dict(export_stats),
        "final_check_stats": {
            name: stats_dict(stats) for (name, _), stats in zip(checks, best_check_stats, strict=True)
        },
        "history": history,
        "npz_file": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(f"export MA rms: {export_base_stats.rms:.6e} -> {export_stats.rms:.6e}")
    print(f"corrected export min eigenvalue: {export_stats.min_eigenvalue:.6e}")


if __name__ == "__main__":
    main()
