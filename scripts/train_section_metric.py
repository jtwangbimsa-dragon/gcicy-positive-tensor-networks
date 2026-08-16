#!/usr/bin/env python3
"""Train a fast diagonal section/H-matrix metric candidate.

The ansatz is

    K = log(sum_m exp(theta_m) |s**a_m t**b_m r**c_m|^2),

with integer Laurent exponents.  This is a local diagonal-H analogue of the
Donaldson/2022-style section metric.  The metric Hessian is computed by the
analytic covariance formula in ``gcicy_metric.section_ansatz`` rather than by
per-point automatic differentiation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    baseline_metrics,
    holomorphic_volume_log_density,
    laurent_exponents,
    random_parameters,
    random_parameters_log_annulus,
    residual_stats,
    residual_values,
    section_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-weight", type=int, default=5)
    parser.add_argument("--min-r-power", type=int, default=-4)
    parser.add_argument("--max-r-power", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--min-eigenvalue", type=float, default=1e-5)
    parser.add_argument("--local-loss-weight", type=float, default=3.0)
    parser.add_argument("--radial-loss-weight", type=float, default=1.0)
    parser.add_argument("--mixed-loss-weight", type=float, default=1.0)
    parser.add_argument("--validation-loss-weight", type=float, default=0.75)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_section_metric_diagonal.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_section_metric_diagonal_summary.json")
    return parser.parse_args()


def local(n_points: int, seed: int) -> np.ndarray:
    return random_parameters(n_points, seed=seed, scale=0.35)


def radial(n_points: int, seed: int) -> np.ndarray:
    return random_parameters_log_annulus(n_points, seed=seed, coordinate_scale=0.3, r_min=0.05, r_max=1.5)


def mixed(local_seed: int, radial_seed: int) -> np.ndarray:
    return np.vstack([local(256, local_seed), radial(512, radial_seed)])


def prepare(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        params,
        baseline_metrics(params),
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
        raise SystemExit("PyTorch is required for section-metric training.") from exc

    args = parse_args()
    torch.set_default_dtype(torch.float64)

    exponents = laurent_exponents(
        max_weight=args.max_weight,
        min_r_power=args.min_r_power,
        max_r_power=args.max_r_power,
    )
    train_local = prepare(
        np.vstack([local(768, 11), local(256, 444), local(256, 555), local(256, 666), local(256, 777)])
    )
    train_radial = prepare(np.vstack([radial(1536, 12), radial(512, 101), radial(512, 202)]))
    train_mixed = prepare(np.vstack([mixed(333, 334), mixed(444, 445)]))
    validation_local = prepare(np.vstack([local(256, 888), local(256, 999), local(256, 1111), local(256, 1222)]))
    validation_radial = prepare(np.vstack([radial(256, 303), radial(256, 404)]))
    validation_mixed = prepare(np.vstack([mixed(222, 223), mixed(101, 202)]))
    checks = [(f"local{seed}", prepare(local(256, seed))) for seed in (444, 555, 666, 777, 888, 999, 1111, 1222)]
    checks += [(f"radial{seed}", prepare(radial(256, seed))) for seed in (101, 202, 303, 404)]
    checks += [
        (f"mixed{local_seed}_{radial_seed}", prepare(mixed(local_seed, radial_seed)))
        for local_seed, radial_seed in ((222, 223), (333, 334), (444, 445), (101, 202))
    ]
    export = prepare(np.vstack([mixed(222, 223), mixed(333, 334), local(256, 444), radial(256, 101)]))

    exponents_t = torch.tensor(exponents, dtype=torch.float64)
    theta = torch.nn.Parameter(torch.zeros(len(exponents), dtype=torch.float64))
    optimizer = torch.optim.Adam([theta], lr=args.lr)

    def section_metric_torch(params: np.ndarray):
        z = torch.tensor(params, dtype=torch.complex128)
        abs_z = torch.clamp(torch.abs(z), min=1e-14)
        logits = theta[None, :] + 2.0 * (torch.log(abs_z) @ exponents_t.T)
        probabilities = torch.softmax(logits, dim=1)
        safe_z = torch.where(abs_z > 1e-14, z, torch.full_like(z, 1e-14 + 0.0j))
        derivatives = exponents_t.to(torch.complex128)[None, :, :] / safe_z[:, None, :]
        mean = torch.einsum("nm,nmi->ni", probabilities.to(torch.complex128), derivatives)
        second = torch.einsum(
            "nm,nmi,nmj->nij",
            probabilities.to(torch.complex128),
            derivatives,
            torch.conj(derivatives),
        )
        metric = second - mean[:, :, None] * torch.conj(mean[:, None, :])
        return 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))

    def dataset_loss(dataset):
        params, _, log_omega = dataset
        log_omega_t = torch.tensor(log_omega)
        metric = section_metric_torch(params)
        eigvals = torch.linalg.eigvalsh(metric)
        raw = torch.log(torch.clamp(eigvals, min=1e-12)).sum(dim=1) - log_omega_t
        centered = raw - raw.mean()
        barrier = torch.nn.functional.softplus((args.min_eigenvalue - eigvals) * 80.0).mean() / 80.0
        return torch.mean(centered**2), barrier

    def evaluate(dataset, log_weights: np.ndarray):
        params, base, _ = dataset
        base_stats = residual_stats(params, base)
        metric = section_metrics(params, exponents, log_weights)
        section_stats = residual_stats(params, metric)
        return base_stats, section_stats

    best_theta = theta.detach().numpy().copy()
    best_score = float("inf")
    best_passed_training_gates = False
    history: list[dict[str, float | int | bool]] = []

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        train_local_loss, train_local_barrier = dataset_loss(train_local)
        train_radial_loss, train_radial_barrier = dataset_loss(train_radial)
        train_mixed_loss, train_mixed_barrier = dataset_loss(train_mixed)
        validation_local_loss, validation_local_barrier = dataset_loss(validation_local)
        validation_radial_loss, validation_radial_barrier = dataset_loss(validation_radial)
        validation_mixed_loss, validation_mixed_barrier = dataset_loss(validation_mixed)
        centered_theta = theta - theta.mean()
        train_loss = (
            args.local_loss_weight * train_local_loss
            + args.radial_loss_weight * train_radial_loss
            + args.mixed_loss_weight * train_mixed_loss
        )
        validation_loss = (
            args.local_loss_weight * validation_local_loss
            + args.radial_loss_weight * validation_radial_loss
            + args.mixed_loss_weight * validation_mixed_loss
        )
        barrier = (
            train_local_barrier
            + train_radial_barrier
            + train_mixed_barrier
            + validation_local_barrier
            + validation_radial_barrier
            + validation_mixed_barrier
        )
        loss = (
            train_loss
            + args.validation_loss_weight * validation_loss
            + 200.0 * barrier
            + 1e-5 * torch.mean(centered_theta**2)
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta -= theta.mean()

        if epoch % args.eval_every != 0:
            continue

        current_theta = theta.detach().numpy().astype(float)
        check_rows = []
        all_passed = True
        finite_positive = True
        for name, dataset in checks:
            baseline, section = evaluate(dataset, current_theta)
            passed = (
                np.isfinite(section.rms)
                and section.min_eigenvalue > args.min_eigenvalue
                and section.rms < baseline.rms
            )
            finite_positive = finite_positive and np.isfinite(section.rms) and section.min_eigenvalue > args.min_eigenvalue
            all_passed = all_passed and passed
            check_rows.append((name, baseline, section, passed))

        finite_rms = [row[2].rms for row in check_rows if np.isfinite(row[2].rms)]
        score = float(np.mean(finite_rms)) if finite_rms else float("inf")
        accepted = bool((all_passed and score < best_score) or (not best_passed_training_gates and finite_positive and score < best_score))
        if accepted:
            best_theta = current_theta.copy()
            best_score = score
            best_passed_training_gates = bool(all_passed)

        row: dict[str, float | int | bool] = {
            "epoch": epoch,
            "accepted": accepted,
            "passed_training_gates": bool(all_passed),
            "loss": float(loss.detach().numpy()),
            "mean_check_rms": score,
        }
        for name, _, section, passed in check_rows:
            row[f"{name}_rms"] = section.rms
            row[f"{name}_min_eigenvalue"] = section.min_eigenvalue
            row[f"{name}_passed"] = bool(passed)
        history.append(row)
        print(
            f"epoch {epoch}: loss={row['loss']:.6e}, mean_check_rms={score:.6e}, "
            f"passed_gates={all_passed}, accepted={accepted}"
        )

    export_base_stats = residual_stats(export[0], export[1])
    export_section = section_metrics(export[0], exponents, best_theta)
    export_section_stats = residual_stats(export[0], export_section)
    check_summary = {}
    for name, dataset in checks:
        baseline, section = evaluate(dataset, best_theta)
        check_summary[name] = {
            "baseline": stats_dict(baseline),
            "section": stats_dict(section),
            "passed": bool(
                np.isfinite(section.rms)
                and section.min_eigenvalue > args.min_eigenvalue
                and section.rms < baseline.rms
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        params=export[0],
        baseline_metrics=export[1],
        corrected_metrics=export_section,
        correction_metrics=export_section - export[1],
        section_theta=best_theta,
        section_exponents=exponents,
        section_normalization=np.asarray(1.0),
        baseline_log_ma=residual_values(export[0], export[1]),
        corrected_log_ma=residual_values(export[0], export_section),
    )
    summary = {
        "description": "Fast diagonal Laurent section/H-matrix metric training.",
        "max_weight": args.max_weight,
        "min_r_power": args.min_r_power,
        "max_r_power": args.max_r_power,
        "basis_size": int(len(exponents)),
        "epochs": args.epochs,
        "lr": args.lr,
        "local_loss_weight": args.local_loss_weight,
        "radial_loss_weight": args.radial_loss_weight,
        "mixed_loss_weight": args.mixed_loss_weight,
        "validation_loss_weight": args.validation_loss_weight,
        "min_eigenvalue_gate": args.min_eigenvalue,
        "passed_training_gates": bool(best_passed_training_gates),
        "baseline_export": stats_dict(export_base_stats),
        "section_export": stats_dict(export_section_stats),
        "checks": check_summary,
        "history": history,
        "npz_file": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(f"export MA rms: {export_base_stats.rms:.6e} -> {export_section_stats.rms:.6e}")
    print(f"section export min eigenvalue: {export_section_stats.min_eigenvalue:.6e}")
    print(f"passed_training_gates={best_passed_training_gates}")


if __name__ == "__main__":
    main()
