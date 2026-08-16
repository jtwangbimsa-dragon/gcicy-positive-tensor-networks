#!/usr/bin/env python3
"""Try a neural-network Kahler-potential correction on the local gCICY patch."""

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
    residual_stats,
    residual_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_metric_adam_refined.npz")
    parser.add_argument("--train", type=int, default=96)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--export", type=int, default=96)
    parser.add_argument("--guard-points", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--validation-seed", type=int, default=222)
    parser.add_argument("--export-seed", type=int, default=333)
    parser.add_argument("--guard-seeds", default="444,555,666,777")
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--nn-scale", type=float, default=0.03)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--validation-weight", type=float, default=0.8)
    parser.add_argument("--barrier-weight", type=float, default=50.0)
    parser.add_argument("--min-eigenvalue", type=float, default=1e-5)
    parser.add_argument("--min-rms-improvement", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_nn_kahler_attempt.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_nn_kahler_attempt_summary.json")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "gcicy_nn_kahler_attempt.pt")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def stats_dict(stats) -> dict[str, float]:
    return {
        "rms": stats.rms,
        "max_abs": stats.max_abs,
        "min_eigenvalue": stats.min_eigenvalue,
        "mean_log_error": stats.mean_log_error,
    }


def summarize_guard(stats_list) -> dict[str, float | int]:
    if not stats_list:
        return {"count": 0, "finite_count": 0, "mean_rms": float("nan"), "max_rms": float("nan"), "min_eigenvalue": float("nan")}
    finite = [stats.rms for stats in stats_list if np.isfinite(stats.rms)]
    return {
        "count": len(stats_list),
        "finite_count": len(finite),
        "mean_rms": float(np.mean(finite)) if finite else float("nan"),
        "max_rms": float(np.max(finite)) if finite else float("nan"),
        "min_eigenvalue": float(np.min([stats.min_eigenvalue for stats in stats_list])),
    }


def centered(values: np.ndarray) -> np.ndarray:
    return values - float(np.mean(values))


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for NN Kahler-potential training.") from exc

    args = parse_args()
    torch.set_default_dtype(torch.float64)

    base_artifact = np.load(args.base_artifact)
    coefficients = base_artifact["coefficients"]
    exponents = base_artifact["exponents"]

    def make_dataset(n_points: int, seed: int):
        params = random_parameters(n_points, seed=seed, scale=args.scale)
        base = baseline_metrics(params)
        basis = basis_hessians(params, exponents)
        metric = apply_correction(base, basis, coefficients)
        real_coords = np.concatenate([params.real, params.imag], axis=1)
        log_omega = np.asarray([holomorphic_volume_log_density(point) for point in params], dtype=float)
        return params, real_coords, metric, log_omega

    train = make_dataset(args.train, args.seed)
    validation = make_dataset(args.validation, args.validation_seed)
    export = make_dataset(args.export, args.export_seed)
    guards = [make_dataset(args.guard_points, seed) for seed in parse_seeds(args.guard_seeds)]

    class PotentialNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(6, args.hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(args.hidden, args.hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(args.hidden, 1),
            )
            torch.nn.init.normal_(self.net[-1].weight, std=1e-5)
            torch.nn.init.zeros_(self.net[-1].bias)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = PotentialNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def correction_batch(real_coords, *, create_graph: bool):
        corrections = []
        for idx in range(real_coords.shape[0]):
            point = real_coords[idx]

            def potential(one_point):
                return model(one_point[None, :]).sum()

            hessian = torch.autograd.functional.hessian(potential, point, create_graph=create_graph)
            complex_hessian = 0.25 * (hessian[:3, :3] + hessian[3:, 3:]).to(torch.complex128)
            complex_hessian = complex_hessian + 0.25j * (hessian[:3, 3:] - hessian[3:, :3]).to(torch.complex128)
            corrections.append(complex_hessian)
        return torch.stack(corrections)

    def torch_dataset_loss(dataset):
        _, real_coords, metrics, log_omega = dataset
        real_coords_t = torch.tensor(real_coords, requires_grad=True)
        metrics_t = torch.tensor(metrics, dtype=torch.complex128)
        log_omega_t = torch.tensor(log_omega)
        corrected = metrics_t + args.nn_scale * correction_batch(real_coords_t, create_graph=True)
        eigvals = torch.linalg.eigvalsh(corrected)
        logdet = torch.log(torch.clamp(eigvals, min=1e-12)).sum(dim=1)
        raw = logdet - log_omega_t
        residual = raw - raw.mean()
        barrier = torch.nn.functional.softplus((args.min_eigenvalue - eigvals) * 60.0).mean() / 60.0
        return torch.mean(residual**2), barrier

    def corrected_metrics(dataset):
        _, real_coords, metrics, _ = dataset
        real_coords_t = torch.tensor(real_coords, requires_grad=True)
        with torch.enable_grad():
            correction = correction_batch(real_coords_t, create_graph=False).detach().numpy()
        return metrics + args.nn_scale * correction

    def evaluate(dataset):
        params, _, metrics, _ = dataset
        return residual_stats(params, corrected_metrics(dataset))

    def guard_stats():
        return [evaluate(dataset) for dataset in guards]

    base_train = residual_stats(train[0], train[2])
    base_validation = residual_stats(validation[0], validation[2])
    base_export = residual_stats(export[0], export[2])
    base_guard = [residual_stats(dataset[0], dataset[2]) for dataset in guards]
    best_epoch = -1
    best_train = base_train
    best_validation = base_validation
    best_export = base_export
    best_guard_summary = summarize_guard(base_guard)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float | int | bool]] = [
        {
            "epoch": -1,
            "accepted": True,
            "train_rms": base_train.rms,
            "validation_rms": base_validation.rms,
            "export_rms": base_export.rms,
            "guard_mean_rms": best_guard_summary["mean_rms"],
            "guard_min_eigenvalue": best_guard_summary["min_eigenvalue"],
        }
    ]

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        train_loss, train_barrier = torch_dataset_loss(train)
        validation_loss, validation_barrier = torch_dataset_loss(validation)
        loss = (
            train_loss
            + args.validation_weight * validation_loss
            + args.barrier_weight * (train_barrier + validation_barrier)
        )
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every != 0:
            continue

        train_stats = evaluate(train)
        validation_stats = evaluate(validation)
        export_stats = evaluate(export)
        current_guard = guard_stats()
        current_guard_summary = summarize_guard(current_guard)
        accepted = (
            np.isfinite(validation_stats.rms)
            and np.isfinite(export_stats.rms)
            and validation_stats.min_eigenvalue > args.min_eigenvalue
            and export_stats.min_eigenvalue > args.min_eigenvalue
            and current_guard_summary["finite_count"] == current_guard_summary["count"]
            and current_guard_summary["min_eigenvalue"] > args.min_eigenvalue
            and validation_stats.rms < best_validation.rms - args.min_rms_improvement
            and export_stats.rms < best_export.rms - args.min_rms_improvement
            and current_guard_summary["mean_rms"] < best_guard_summary["mean_rms"] - args.min_rms_improvement
        )
        if accepted:
            best_epoch = epoch
            best_train = train_stats
            best_validation = validation_stats
            best_export = export_stats
            best_guard_summary = current_guard_summary
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
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
                "guard_mean_rms": current_guard_summary["mean_rms"],
                "guard_min_eigenvalue": current_guard_summary["min_eigenvalue"],
            }
        )

    model.load_state_dict(best_state)
    export_corrected = corrected_metrics(export)
    export_values = residual_values(export[0], export_corrected)
    base_values = residual_values(export[0], export[2])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        params=export[0],
        base_metrics=export[2],
        corrected_metrics=export_corrected,
        nn_correction_metrics=export_corrected - export[2],
        baseline_log_ma=base_values,
        baseline_centered_log_ma=centered(base_values),
        corrected_log_ma=export_values,
        corrected_centered_log_ma=centered(export_values),
        base_artifact=str(args.base_artifact),
    )
    torch.save({"state_dict": best_state, "args": vars(args)}, args.model)

    accepted_any = best_epoch >= 0
    summary = {
        "description": "Neural-network Kahler-potential correction attempt for the local gCICY patch.",
        "method": "NN potential phi_NN trained with centered Monge-Ampere loss; accepted only if held-out gates improve.",
        "base_artifact": str(args.base_artifact),
        "accepted_any_nn_checkpoint": bool(accepted_any),
        "best_epoch": best_epoch,
        "is_exact_ricci_flat": False,
        "is_kahler_local": True,
        "nn_scale": args.nn_scale,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "min_rms_improvement": args.min_rms_improvement,
        "train_points": args.train,
        "validation_points": args.validation,
        "export_points": args.export,
        "guard_points": args.guard_points,
        "guard_seeds": parse_seeds(args.guard_seeds),
        "base_train": stats_dict(base_train),
        "base_validation": stats_dict(base_validation),
        "base_export": stats_dict(base_export),
        "base_guard": summarize_guard(base_guard),
        "best_train": stats_dict(best_train),
        "best_validation": stats_dict(best_validation),
        "best_export": stats_dict(best_export),
        "best_guard": best_guard_summary,
        "history": history,
        "npz_file": str(args.out),
        "model_file": str(args.model),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.model}")
    print(f"accepted_any_nn_checkpoint={accepted_any}")
    print(f"validation MA rms: {base_validation.rms:.6e} -> {best_validation.rms:.6e}")
    print(f"export MA rms: {base_export.rms:.6e} -> {best_export.rms:.6e}")
    print(f"guard mean MA rms: {summarize_guard(base_guard)['mean_rms']:.6e} -> {best_guard_summary['mean_rms']:.6e}")


if __name__ == "__main__":
    main()
