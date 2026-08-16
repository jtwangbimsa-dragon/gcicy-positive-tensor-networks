#!/usr/bin/env python3
"""Train a positive full Hermitian H-matrix on genuine global sections."""

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
    global_h_metrics,
    holomorphic_volume_log_density,
    random_parameters,
    random_parameters_log_annulus,
    random_projective_coverage_points,
    reference_section_values_and_jacobian,
    residual_stats,
    residual_values,
    restricted_ambient_basis,
    restricted_fubini_study_h_matrix,
)


def parse_degree(text: str) -> tuple[int, int, int]:
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 3 or min(values) < 0:
        raise argparse.ArgumentTypeError("degree must be three non-negative comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=parse_degree, default=(1, 1, 1))
    parser.add_argument("--basis-points", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--min-eigenvalue", type=float, default=1e-5)
    parser.add_argument("--local-weight", type=float, default=3.0)
    parser.add_argument("--radial-weight", type=float, default=1.0)
    parser.add_argument("--mixed-weight", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--validation-weight", type=float, default=0.75)
    parser.add_argument("--drift-weight", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_global_h_metric.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_global_h_metric_summary.json")
    return parser.parse_args()


def local(n_points: int, seed: int) -> np.ndarray:
    return random_parameters(n_points, seed=seed, scale=0.35)


def radial(n_points: int, seed: int) -> np.ndarray:
    return random_parameters_log_annulus(n_points, seed=seed, coordinate_scale=0.3, r_min=0.05, r_max=1.5)


def mixed(n_local: int, local_seed: int, n_radial: int, radial_seed: int) -> np.ndarray:
    return np.vstack([local(n_local, local_seed), radial(n_radial, radial_seed)])


def coverage(n_points: int, seed: int) -> np.ndarray:
    return random_projective_coverage_points(n_points, seed=seed)[0]


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
        raise SystemExit("PyTorch is required for full H-matrix training.") from exc

    args = parse_args()
    torch.set_default_dtype(torch.float64)

    basis_params = local(args.basis_points, 1001)
    basis = restricted_ambient_basis(basis_params, args.degree, relative_rank_threshold=1e-10)
    initial_h, relation_error = restricted_fubini_study_h_matrix(basis_params, basis)
    initial_h = initial_h * (len(initial_h) / float(np.trace(initial_h).real))
    initial_cholesky = np.linalg.cholesky(initial_h)
    exponents = basis.selected_exponents

    def prepare(params: np.ndarray, *, absolute_eigenvalue_gate: float | None = None):
        values = []
        derivatives = []
        for point in params:
            section_values, section_derivatives = reference_section_values_and_jacobian(point, exponents)
            values.append(section_values)
            derivatives.append(section_derivatives)
        return {
            "params": params,
            "values": np.asarray(values, dtype=np.complex128),
            "derivatives": np.asarray(derivatives, dtype=np.complex128),
            "log_omega": np.asarray([holomorphic_volume_log_density(point) for point in params], dtype=float),
            "baseline": baseline_metrics(params),
            "absolute_eigenvalue_gate": (
                args.min_eigenvalue if absolute_eigenvalue_gate is None else absolute_eigenvalue_gate
            ),
        }

    train_sets = [
        (args.local_weight, prepare(local(768, 1101))),
        (args.radial_weight, prepare(radial(1536, 1102))),
        (args.mixed_weight, prepare(mixed(256, 1103, 512, 1104))),
        (args.coverage_weight, prepare(coverage(1024, 1105), absolute_eigenvalue_gate=0.0)),
    ]
    validation_sets = [
        (args.local_weight, prepare(local(512, 1201))),
        (args.radial_weight, prepare(radial(768, 1202))),
        (args.mixed_weight, prepare(mixed(256, 1203, 512, 1204))),
        (args.coverage_weight, prepare(coverage(768, 1205), absolute_eigenvalue_gate=0.0)),
    ]
    checks = [
        ("local1301", prepare(local(256, 1301))),
        ("radial1302", prepare(radial(256, 1302))),
        ("mixed1303_1304", prepare(mixed(256, 1303, 512, 1304))),
        ("coverage1305", prepare(coverage(512, 1305), absolute_eigenvalue_gate=0.0)),
    ]
    export = prepare(
        np.vstack([mixed(512, 1401, 1024, 1402), coverage(512, 1403)]),
        absolute_eigenvalue_gate=0.0,
    )

    diagonal_log = torch.nn.Parameter(torch.log(torch.tensor(np.diag(initial_cholesky).real)))
    lower_real = torch.nn.Parameter(torch.tensor(initial_cholesky.real))
    lower_imag = torch.nn.Parameter(torch.tensor(initial_cholesky.imag))
    optimizer = torch.optim.Adam([diagonal_log, lower_real, lower_imag], lr=args.lr)
    lower_mask = torch.tril(torch.ones_like(lower_real), diagonal=-1)
    initial_h_t = torch.tensor(initial_h, dtype=torch.complex128)

    def current_h_matrix():
        lower = lower_mask * lower_real + 1j * lower_mask * lower_imag
        cholesky = lower.to(torch.complex128) + torch.diag(torch.exp(diagonal_log)).to(torch.complex128)
        h_matrix = cholesky @ torch.conj(cholesky.T)
        return h_matrix * (len(initial_h) / torch.real(torch.trace(h_matrix)))

    torch_cache: dict[int, tuple] = {}

    def as_torch(dataset):
        key = id(dataset)
        if key not in torch_cache:
            torch_cache[key] = (
                torch.tensor(dataset["values"], dtype=torch.complex128),
                torch.tensor(dataset["derivatives"], dtype=torch.complex128),
                torch.tensor(dataset["log_omega"]),
            )
        return torch_cache[key]

    def raw_residual_and_barrier(dataset, h_matrix):
        values, derivatives, log_omega = as_torch(dataset)
        h_values = torch.einsum("ab,nb->na", h_matrix, values)
        denominator = torch.real(torch.einsum("na,na->n", torch.conj(values), h_values))
        h_derivatives = torch.einsum("ab,nbj->naj", h_matrix, derivatives)
        first = torch.einsum("nmi,nmj->nij", torch.conj(derivatives), h_derivatives)
        gradient = torch.einsum("nm,nmj->nj", torch.conj(values), h_derivatives)
        metric = first / denominator[:, None, None]
        metric -= torch.conj(gradient)[:, :, None] * gradient[:, None, :] / denominator[:, None, None] ** 2
        metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        eigenvalues = torch.linalg.eigvalsh(metric)
        raw = torch.log(torch.clamp(eigenvalues, min=1e-12)).sum(dim=1) - log_omega
        spectral_scale = torch.clamp(torch.mean(torch.abs(eigenvalues), dim=1, keepdim=True), min=1e-14)
        relative_eigenvalues = eigenvalues / spectral_scale
        barrier = torch.nn.functional.softplus((1e-8 - relative_eigenvalues) * 80.0).mean() / 80.0
        return raw, barrier

    def grouped_loss(weighted_datasets, h_matrix):
        pieces = []
        barriers = []
        total_weight = 0.0
        weighted_mean = torch.tensor(0.0)
        for weight, dataset in weighted_datasets:
            raw, barrier = raw_residual_and_barrier(dataset, h_matrix)
            pieces.append((weight, raw))
            barriers.append(barrier)
            weighted_mean = weighted_mean + weight * raw.mean()
            total_weight += weight
        weighted_mean = weighted_mean / total_weight
        loss = sum(weight * torch.mean((raw - weighted_mean) ** 2) for weight, raw in pieces) / total_weight
        return loss, sum(barriers)

    def evaluate(dataset, h_matrix: np.ndarray):
        metrics = global_h_metrics(dataset["params"], exponents, h_matrix)
        return residual_stats(dataset["params"], dataset["baseline"]), residual_stats(dataset["params"], metrics)

    initial_check_stats = [evaluate(dataset, initial_h)[1] for _, dataset in checks]
    best_h = initial_h.copy()
    best_score = float("inf")
    best_passed_gates = False
    history: list[dict[str, float | int | bool]] = []

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        h_matrix = current_h_matrix()
        train_loss, train_barrier = grouped_loss(train_sets, h_matrix)
        validation_loss, validation_barrier = grouped_loss(validation_sets, h_matrix)
        drift = torch.mean(torch.abs(h_matrix - initial_h_t) ** 2)
        loss = (
            train_loss
            + args.validation_weight * validation_loss
            + 200.0 * (train_barrier + validation_barrier)
            + args.drift_weight * drift
        )
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every != 0:
            continue

        candidate_h = current_h_matrix().detach().numpy()
        check_rows = []
        all_passed = True
        finite_positive = True
        for name, dataset in checks:
            baseline, candidate = evaluate(dataset, candidate_h)
            passed = (
                np.isfinite(candidate.rms)
                and candidate.min_eigenvalue > dataset["absolute_eigenvalue_gate"]
                and candidate.rms < baseline.rms
            )
            finite_positive = (
                finite_positive
                and np.isfinite(candidate.rms)
                and candidate.min_eigenvalue > dataset["absolute_eigenvalue_gate"]
            )
            all_passed = all_passed and passed
            check_rows.append((name, baseline, candidate, passed))

        score = float(np.mean([row[2].rms for row in check_rows if np.isfinite(row[2].rms)]))
        accepted = bool(
            (all_passed and score < best_score)
            or (not best_passed_gates and finite_positive and score < best_score)
        )
        if accepted:
            best_h = candidate_h.copy()
            best_score = score
            best_passed_gates = bool(all_passed)

        row: dict[str, float | int | bool] = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "accepted": accepted,
            "passed_internal_gates": bool(all_passed),
            "mean_check_rms": score,
        }
        for name, _, candidate, passed in check_rows:
            row[f"{name}_rms"] = candidate.rms
            row[f"{name}_min_eigenvalue"] = candidate.min_eigenvalue
            row[f"{name}_passed"] = bool(passed)
        history.append(row)
        print(
            f"epoch {epoch}: loss={row['loss']:.6e}, mean_check_rms={score:.6e}, "
            f"passed_gates={all_passed}, accepted={accepted}"
        )

    export_baseline = residual_stats(export["params"], export["baseline"])
    export_metrics = global_h_metrics(export["params"], exponents, best_h)
    export_stats = residual_stats(export["params"], export_metrics)
    check_summary = {}
    for name, dataset in checks:
        baseline, candidate = evaluate(dataset, best_h)
        check_summary[name] = {
            "baseline": stats_dict(baseline),
            "global_h": stats_dict(candidate),
            "passed": bool(
                np.isfinite(candidate.rms)
                and candidate.min_eigenvalue > dataset["absolute_eigenvalue_gate"]
                and candidate.rms < baseline.rms
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        params=export["params"],
        baseline_metrics=export["baseline"],
        corrected_metrics=export_metrics,
        global_section_degree=np.asarray(args.degree, dtype=np.int64),
        global_section_exponents=exponents,
        global_h_matrix=best_h,
        global_section_normalization=np.asarray(1.0),
        basis_selected_indices=basis.selected_indices,
        basis_relation_error=np.asarray(relation_error),
        baseline_log_ma=residual_values(export["params"], export["baseline"]),
        corrected_log_ma=residual_values(export["params"], export_metrics),
    )
    summary = {
        "description": "Full positive Hermitian H-matrix training on genuine restricted ambient global sections.",
        "degree": list(args.degree),
        "ambient_section_count": int(len(basis.ambient_exponents)),
        "restricted_section_count": int(basis.numerical_rank),
        "basis_relation_error": relation_error,
        "epochs": args.epochs,
        "lr": args.lr,
        "passed_internal_gates": bool(best_passed_gates),
        "initial_check_stats": {
            name: stats_dict(stats) for (name, _), stats in zip(checks, initial_check_stats, strict=True)
        },
        "baseline_export": stats_dict(export_baseline),
        "global_h_export": stats_dict(export_stats),
        "checks": check_summary,
        "history": history,
        "npz_file": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(f"export MA rms: {export_baseline.rms:.6e} -> {export_stats.rms:.6e}")
    print(f"global H export min eigenvalue: {export_stats.min_eigenvalue:.6e}")
    print(f"passed_internal_gates={best_passed_gates}")


if __name__ == "__main__":
    main()
