#!/usr/bin/env python3
"""Train a structured low-rank global H-matrix at higher section degree."""

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
    parser.add_argument("--degree", type=parse_degree, default=(2, 2, 2))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--basis-points", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--local-weight", type=float, default=3.0)
    parser.add_argument("--radial-weight", type=float, default=1.0)
    parser.add_argument("--mixed-weight", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--validation-weight", type=float, default=0.75)
    parser.add_argument("--factor-regularization", type=float, default=1e-5)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "gcicy_global_h_metric_k2_low_rank.npz",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "outputs" / "gcicy_global_h_metric_k2_low_rank_summary.json",
    )
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
        raise SystemExit("PyTorch is required for low-rank H-matrix training.") from exc

    args = parse_args()
    if args.rank <= 0:
        raise SystemExit("rank must be positive")
    if args.epsilon <= 0:
        raise SystemExit("epsilon must be positive")
    torch.set_default_dtype(torch.float64)

    degree_scale = args.degree[0] if args.degree[0] == args.degree[1] == args.degree[2] else 1
    normalization = 1.0 / max(1, degree_scale)
    basis_params = local(args.basis_points, 2001)
    basis = restricted_ambient_basis(basis_params, args.degree, relative_rank_threshold=1e-10)
    if args.rank >= basis.numerical_rank:
        raise SystemExit("rank must be smaller than the restricted section dimension")
    initial_h, relation_error = restricted_fubini_study_h_matrix(basis_params, basis)
    initial_h = initial_h * (len(initial_h) / float(np.trace(initial_h).real))
    exponents = basis.selected_exponents

    def prepare(params: np.ndarray, *, absolute_eigenvalue_gate: float | None = None):
        values = []
        derivatives = []
        for point in params:
            section_values, section_derivatives = reference_section_values_and_jacobian(point, exponents)
            values.append(section_values)
            derivatives.append(section_derivatives)
        values_array = np.asarray(values, dtype=np.complex128)
        derivatives_array = np.asarray(derivatives, dtype=np.complex128)
        h0_values = np.einsum("ab,nb->na", initial_h, values_array)
        h0_derivatives = np.einsum("ab,nbj->naj", initial_h, derivatives_array)
        return {
            "params": params,
            "values": values_array,
            "derivatives": derivatives_array,
            "h0_values": h0_values,
            "h0_derivatives": h0_derivatives,
            "log_omega": np.asarray([holomorphic_volume_log_density(point) for point in params], dtype=float),
            "baseline": baseline_metrics(params),
            "absolute_eigenvalue_gate": (
                1e-5 if absolute_eigenvalue_gate is None else absolute_eigenvalue_gate
            ),
        }

    train_sets = [
        (args.local_weight, prepare(local(768, 2101))),
        (args.radial_weight, prepare(radial(1536, 2102))),
        (args.mixed_weight, prepare(mixed(256, 2103, 512, 2104))),
        (args.coverage_weight, prepare(coverage(1024, 2105), absolute_eigenvalue_gate=0.0)),
    ]
    validation_sets = [
        (args.local_weight, prepare(local(512, 2201))),
        (args.radial_weight, prepare(radial(768, 2202))),
        (args.mixed_weight, prepare(mixed(256, 2203, 512, 2204))),
        (args.coverage_weight, prepare(coverage(768, 2205), absolute_eigenvalue_gate=0.0)),
    ]
    checks = [
        ("local2301", prepare(local(256, 2301))),
        ("radial2302", prepare(radial(256, 2302))),
        ("mixed2303_2304", prepare(mixed(256, 2303, 512, 2304))),
        ("coverage2305", prepare(coverage(512, 2305), absolute_eigenvalue_gate=0.0)),
    ]
    export = prepare(
        np.vstack([mixed(512, 2401, 1024, 2402), coverage(512, 2403)]),
        absolute_eigenvalue_gate=0.0,
    )

    section_count = basis.numerical_rank
    rng = np.random.default_rng(2002)
    random_frame = rng.normal(size=(section_count, args.rank)) + 1j * rng.normal(size=(section_count, args.rank))
    initial_v, _ = np.linalg.qr(random_frame)
    initial_v = initial_v[:, : args.rank]
    u_real = torch.nn.Parameter(torch.zeros((section_count, args.rank)))
    u_imag = torch.nn.Parameter(torch.zeros((section_count, args.rank)))
    v_real = torch.nn.Parameter(torch.tensor(initial_v.real))
    v_imag = torch.nn.Parameter(torch.tensor(initial_v.imag))
    optimizer = torch.optim.Adam([u_real, u_imag, v_real, v_imag], lr=args.lr)
    h0_t = torch.tensor(initial_h, dtype=torch.complex128)
    identity_t = torch.eye(section_count, dtype=torch.complex128)
    initial_v_t = torch.tensor(initial_v, dtype=torch.complex128)

    def factors():
        return u_real.to(torch.complex128) + 1j * u_imag, v_real.to(torch.complex128) + 1j * v_imag

    def dense_h_matrix():
        u, v = factors()
        transform = identity_t + u @ torch.conj(v.T)
        h_matrix = transform @ h0_t @ torch.conj(transform.T) + args.epsilon * h0_t
        return h_matrix * (section_count / torch.real(torch.trace(h_matrix)))

    torch_cache: dict[int, tuple] = {}

    def as_torch(dataset):
        key = id(dataset)
        if key not in torch_cache:
            torch_cache[key] = (
                torch.tensor(dataset["values"], dtype=torch.complex128),
                torch.tensor(dataset["derivatives"], dtype=torch.complex128),
                torch.tensor(dataset["h0_values"], dtype=torch.complex128),
                torch.tensor(dataset["h0_derivatives"], dtype=torch.complex128),
                torch.tensor(dataset["log_omega"]),
            )
        return torch_cache[key]

    def apply_structured_h(values, h0_values, u, v):
        u_dagger_values = torch.einsum("mr,nm...->nr...", torch.conj(u), values)
        h0_v = h0_t @ v
        transformed_h0 = h0_values + torch.einsum("mr,nr...->nm...", h0_v, u_dagger_values)
        v_dagger_transformed = torch.einsum("mr,nm...->nr...", torch.conj(v), transformed_h0)
        return (
            transformed_h0
            + torch.einsum("mr,nr...->nm...", u, v_dagger_transformed)
            + args.epsilon * h0_values
        )

    def raw_residual_and_barrier(dataset):
        values, derivatives, h0_values, h0_derivatives, log_omega = as_torch(dataset)
        u, v = factors()
        h_values = apply_structured_h(values, h0_values, u, v)
        h_derivatives = apply_structured_h(derivatives, h0_derivatives, u, v)
        denominator = torch.real(torch.einsum("na,na->n", torch.conj(values), h_values))
        first = torch.einsum("nmi,nmj->nij", torch.conj(derivatives), h_derivatives)
        gradient = torch.einsum("nm,nmj->nj", torch.conj(values), h_derivatives)
        metric = first / denominator[:, None, None]
        metric -= torch.conj(gradient)[:, :, None] * gradient[:, None, :] / denominator[:, None, None] ** 2
        metric *= normalization
        metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        eigenvalues = torch.linalg.eigvalsh(metric)
        raw = torch.log(torch.clamp(eigenvalues, min=1e-12)).sum(dim=1) - log_omega
        spectral_scale = torch.clamp(torch.mean(torch.abs(eigenvalues), dim=1, keepdim=True), min=1e-14)
        relative_eigenvalues = eigenvalues / spectral_scale
        barrier = torch.nn.functional.softplus((1e-8 - relative_eigenvalues) * 80.0).mean() / 80.0
        return raw, barrier

    def grouped_loss(weighted_datasets):
        pieces = []
        barriers = []
        total_weight = 0.0
        weighted_mean = torch.tensor(0.0)
        for weight, dataset in weighted_datasets:
            raw, barrier = raw_residual_and_barrier(dataset)
            pieces.append((weight, raw))
            barriers.append(barrier)
            weighted_mean = weighted_mean + weight * raw.mean()
            total_weight += weight
        weighted_mean = weighted_mean / total_weight
        loss = sum(weight * torch.mean((raw - weighted_mean) ** 2) for weight, raw in pieces) / total_weight
        return loss, sum(barriers)

    def evaluate(dataset, h_matrix: np.ndarray):
        metrics = global_h_metrics(
            dataset["params"],
            exponents,
            h_matrix,
            normalization=normalization,
        )
        return residual_stats(dataset["params"], dataset["baseline"]), residual_stats(dataset["params"], metrics)

    initial_effective_h = (1.0 + args.epsilon) * initial_h
    best_h = initial_effective_h.copy()
    best_score = float("inf")
    best_passed_gates = False
    history: list[dict[str, float | int | bool]] = []

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        train_loss, train_barrier = grouped_loss(train_sets)
        validation_loss, validation_barrier = grouped_loss(validation_sets)
        u, v = factors()
        factor_penalty = torch.mean(torch.abs(u) ** 2) + torch.mean(torch.abs(v - initial_v_t) ** 2)
        loss = (
            train_loss
            + args.validation_weight * validation_loss
            + 200.0 * (train_barrier + validation_barrier)
            + args.factor_regularization * factor_penalty
        )
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every != 0:
            continue

        candidate_h = dense_h_matrix().detach().numpy()
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
    export_metrics = global_h_metrics(export["params"], exponents, best_h, normalization=normalization)
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
        global_section_normalization=np.asarray(normalization),
        low_rank=np.asarray(args.rank),
        basis_selected_indices=basis.selected_indices,
        basis_relation_error=np.asarray(relation_error),
        baseline_log_ma=residual_values(export["params"], export["baseline"]),
        corrected_log_ma=residual_values(export["params"], export_metrics),
    )
    summary = {
        "description": "Structured low-rank congruence training on genuine restricted ambient global sections.",
        "degree": list(args.degree),
        "normalization": normalization,
        "rank": args.rank,
        "epsilon": args.epsilon,
        "ambient_section_count": int(len(basis.ambient_exponents)),
        "restricted_section_count": int(basis.numerical_rank),
        "basis_relation_error": relation_error,
        "epochs": args.epochs,
        "lr": args.lr,
        "passed_internal_gates": bool(best_passed_gates),
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
