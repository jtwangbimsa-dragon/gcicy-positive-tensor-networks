#!/usr/bin/env python3
"""Train a full global-section H-matrix on the generic gCICY model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    generic_baseline_metrics,
    generic_global_h_metrics,
    generic_holomorphic_volume_log_density,
    generic_importance_weights,
    generic_effective_sample_size,
    generic_lift_h_matrix_power,
    generic_residual_stats,
    generic_residual_values,
    generic_restricted_ambient_basis,
    generic_restricted_fubini_study_h_matrix,
    generic_section_values_and_jacobian,
    generic_weighted_residual_stats,
    make_exact_generic_model,
    make_generic_model,
    sample_generic_gcicy_points,
)


def parse_degree(text: str) -> tuple[int, int, int]:
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 3 or min(values) <= 0:
        raise argparse.ArgumentTypeError("degree must be three positive comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=20260711)
    parser.add_argument("--exact-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--degree", type=parse_degree, default=(1, 1, 1))
    parser.add_argument("--initial-artifact", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument(
        "--importance-weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train and gate on the Omega-volume importance-weighted MA residual",
    )
    parser.add_argument("--basis-points", type=int, default=512)
    parser.add_argument("--train-points", type=int, default=2048)
    parser.add_argument("--validation-points", type=int, default=1024)
    parser.add_argument("--check-points", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience-evals", type=int, default=6)
    parser.add_argument("--validation-weight", type=float, default=0.75)
    parser.add_argument("--drift-weight", type=float, default=1e-5)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def default_output_paths(degree: tuple[int, int, int], importance_weighted: bool) -> tuple[Path, Path]:
    if degree == (1, 1, 1):
        stem = "gcicy_generic_global_h_metric"
    elif degree[0] == degree[1] == degree[2]:
        stem = f"gcicy_generic_global_h_metric_k{degree[0]}"
    else:
        stem = "gcicy_generic_global_h_metric_" + "_".join(str(value) for value in degree)
    if importance_weighted:
        stem += "_weighted"
    return ROOT / "outputs" / f"{stem}.npz", ROOT / "outputs" / f"{stem}_summary.json"


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
        raise SystemExit("PyTorch is required for generic full-H training.") from exc

    args = parse_args()
    default_out, default_summary = default_output_paths(args.degree, args.importance_weighted)
    args.out = args.out or default_out
    args.summary = args.summary or default_summary
    if args.degree[0] != args.degree[1] or args.degree[1] != args.degree[2]:
        raise SystemExit("The convergence trainer currently requires degree (k,k,k).")
    normalization = 1.0 / args.degree[0]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    real_dtype = torch.float32 if args.precision == "complex64" else torch.float64
    complex_dtype = torch.complex64 if args.precision == "complex64" else torch.complex128
    torch.set_default_dtype(real_dtype)
    model = make_exact_generic_model(args.model_seed) if args.exact_model else make_generic_model(args.model_seed)
    basis_points = sample_generic_gcicy_points(model, args.basis_points, seed=3201)
    basis = generic_restricted_ambient_basis(basis_points, args.degree)
    initial_h, relation_error = generic_restricted_fubini_study_h_matrix(basis_points, basis)
    initialization = "ambient_fubini_study"
    if args.initial_artifact is not None:
        source = np.load(args.initial_artifact)
        if int(source["generic_model_seed"]) != model.seed:
            raise SystemExit("initial artifact uses a different model seed")
        if not np.allclose(source["p1_coefficients"], model.p1_coefficients) or not np.allclose(
            source["p2_tensor"], model.p2_tensor
        ):
            raise SystemExit("initial artifact coefficients do not match the target model")
        initial_h, relation_error = generic_lift_h_matrix_power(
            basis_points,
            source["global_section_exponents"],
            source["global_h_matrix"],
            basis,
        )
        initialization = f"lifted_power_of:{args.initial_artifact.resolve()}"
    initial_h = initial_h * (len(initial_h) / float(np.trace(initial_h).real))
    initial_cholesky = np.linalg.cholesky(initial_h)
    exponents = basis.selected_exponents

    def prepare(points):
        values = []
        derivatives = []
        for point in points:
            section_values, section_derivatives = generic_section_values_and_jacobian(point, exponents)
            values.append(section_values)
            derivatives.append(section_derivatives)
        importance_weights = generic_importance_weights(points)
        return {
            "points": points,
            "values": np.asarray(values, dtype=np.complex128),
            "derivatives": np.asarray(derivatives, dtype=np.complex128),
            "log_omega": np.asarray(
                [generic_holomorphic_volume_log_density(point) for point in points],
                dtype=float,
            ),
            "baseline": generic_baseline_metrics(points),
            "importance_weights": importance_weights,
            "effective_sample_size": generic_effective_sample_size(importance_weights),
        }

    train = prepare(sample_generic_gcicy_points(model, args.train_points, seed=3202))
    validation = prepare(sample_generic_gcicy_points(model, args.validation_points, seed=3203))
    checks = [
        (f"seed{seed}", prepare(sample_generic_gcicy_points(model, args.check_points, seed=seed)))
        for seed in (3301, 3302, 3303, 3304)
    ]
    export = prepare(sample_generic_gcicy_points(model, 1024, seed=3401))

    diagonal_log = torch.nn.Parameter(
        torch.log(torch.tensor(np.diag(initial_cholesky).real, dtype=real_dtype, device=device))
    )
    lower_real = torch.nn.Parameter(torch.tensor(initial_cholesky.real, dtype=real_dtype, device=device))
    lower_imag = torch.nn.Parameter(torch.tensor(initial_cholesky.imag, dtype=real_dtype, device=device))
    lower_mask = torch.tril(torch.ones_like(lower_real), diagonal=-1)
    optimizer = torch.optim.Adam([diagonal_log, lower_real, lower_imag], lr=args.lr)
    initial_h_t = torch.tensor(initial_h, dtype=complex_dtype, device=device)

    def current_h_matrix():
        lower = lower_mask * lower_real + 1j * lower_mask * lower_imag
        cholesky = lower.to(complex_dtype) + torch.diag(torch.exp(diagonal_log)).to(complex_dtype)
        h_matrix = cholesky @ torch.conj(cholesky.T)
        return h_matrix * (len(initial_h) / torch.real(torch.trace(h_matrix)))

    cache = {}

    def as_torch(dataset):
        key = id(dataset)
        if key not in cache:
            cache[key] = (
                torch.tensor(dataset["values"], dtype=complex_dtype, device=device),
                torch.tensor(dataset["derivatives"], dtype=complex_dtype, device=device),
                torch.tensor(dataset["log_omega"], dtype=real_dtype, device=device),
                torch.tensor(dataset["importance_weights"], dtype=real_dtype, device=device),
            )
        return cache[key]

    def raw_and_barrier(dataset, h_matrix):
        values, derivatives, log_omega, importance_weights = as_torch(dataset)
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
        scale = torch.clamp(torch.mean(torch.abs(eigenvalues), dim=1, keepdim=True), min=1e-14)
        relative_eigenvalues = eigenvalues / scale
        barrier = torch.nn.functional.softplus((1e-8 - relative_eigenvalues) * 80.0).mean() / 80.0
        weights = importance_weights if args.importance_weighted else torch.ones_like(importance_weights)
        return raw, barrier, weights

    def evaluate(dataset, h_matrix):
        metrics = generic_global_h_metrics(
            dataset["points"], exponents, h_matrix, normalization=normalization
        )
        if args.importance_weighted:
            return generic_weighted_residual_stats(
                dataset["points"], dataset["baseline"], dataset["importance_weights"]
            ), generic_weighted_residual_stats(dataset["points"], metrics, dataset["importance_weights"])
        return generic_residual_stats(dataset["points"], dataset["baseline"]), generic_residual_stats(dataset["points"], metrics)

    initial_rows = []
    for name, dataset in checks:
        baseline, candidate = evaluate(dataset, initial_h)
        passed = np.isfinite(candidate.rms) and candidate.min_eigenvalue > 0 and candidate.rms < baseline.rms
        initial_rows.append((name, candidate, passed))
    best_h = initial_h.copy()
    best_score = float(np.mean([row[1].rms for row in initial_rows]))
    best_passed = bool(all(row[2] for row in initial_rows))
    history = [
        {
            "epoch": -1,
            "loss": None,
            "accepted": True,
            "passed_internal_gates": best_passed,
            "mean_check_rms": best_score,
        }
    ]
    print(
        f"initialization={initialization}, device={device}, precision={args.precision}, "
        f"mean_check_rms={best_score:.6e}",
        flush=True,
    )
    stale_evaluations = 0

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        h_matrix = current_h_matrix()
        train_raw, train_barrier, train_weights = raw_and_barrier(train, h_matrix)
        validation_raw, validation_barrier, validation_weights = raw_and_barrier(validation, h_matrix)
        train_mean = torch.sum(train_weights * train_raw) / torch.sum(train_weights)
        validation_mean = torch.sum(validation_weights * validation_raw) / torch.sum(validation_weights)
        shared_mean = (train_mean + args.validation_weight * validation_mean) / (
            1.0 + args.validation_weight
        )
        train_loss = torch.sum(train_weights * (train_raw - shared_mean) ** 2) / torch.sum(train_weights)
        validation_loss = torch.sum(validation_weights * (validation_raw - shared_mean) ** 2) / torch.sum(validation_weights)
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
        candidate_h = current_h_matrix().detach().cpu().numpy()
        rows = []
        all_passed = True
        for name, dataset in checks:
            baseline, candidate = evaluate(dataset, candidate_h)
            passed = np.isfinite(candidate.rms) and candidate.min_eigenvalue > 0 and candidate.rms < baseline.rms
            all_passed = all_passed and passed
            rows.append((name, baseline, candidate, passed))
        score = float(np.mean([row[2].rms for row in rows]))
        accepted = bool(all_passed and score < best_score)
        if accepted:
            best_h = candidate_h.copy()
            best_score = score
            best_passed = True
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        history_row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "accepted": accepted,
            "passed_internal_gates": bool(all_passed),
            "mean_check_rms": score,
        }
        for name, _, candidate, passed in rows:
            history_row[f"{name}_rms"] = candidate.rms
            history_row[f"{name}_min_eigenvalue"] = candidate.min_eigenvalue
            history_row[f"{name}_passed"] = bool(passed)
        history.append(history_row)
        print(
            f"epoch {epoch}: loss={history_row['loss']:.6e}, mean_check_rms={score:.6e}, "
            f"passed_gates={all_passed}, accepted={accepted}",
            flush=True,
        )
        if args.patience_evals > 0 and stale_evaluations >= args.patience_evals:
            print(f"early stopping after {stale_evaluations} evaluations without improvement", flush=True)
            break

    export_baseline = generic_residual_stats(export["points"], export["baseline"])
    export_metrics = generic_global_h_metrics(
        export["points"], exponents, best_h, normalization=normalization
    )
    export_stats = generic_residual_stats(export["points"], export_metrics)
    export_baseline_weighted = generic_weighted_residual_stats(
        export["points"], export["baseline"], export["importance_weights"]
    )
    export_stats_weighted = generic_weighted_residual_stats(
        export["points"], export_metrics, export["importance_weights"]
    )
    check_summary = {}
    for name, dataset in checks:
        baseline, candidate = evaluate(dataset, best_h)
        check_summary[name] = {
            "baseline": stats_dict(baseline),
            "global_h": stats_dict(candidate),
            "effective_sample_size": dataset["effective_sample_size"],
            "passed": bool(np.isfinite(candidate.rms) and candidate.min_eigenvalue > 0 and candidate.rms < baseline.rms),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        generic_model_seed=np.asarray(model.seed),
        exact_integer_coefficients=np.asarray(args.exact_model),
        importance_weighted_training=np.asarray(args.importance_weighted),
        p1_exponents=model.p1_exponents,
        p1_coefficients=model.p1_coefficients,
        p2_exponents=model.p2_exponents,
        p2_coefficients=model.p2_coefficients,
        p2_tensor=model.p2_tensor,
        global_section_degree=np.asarray(args.degree, dtype=np.int64),
        global_section_exponents=exponents,
        global_h_matrix=best_h,
        global_section_normalization=np.asarray(normalization),
        basis_selected_indices=basis.selected_indices,
        basis_relation_error=np.asarray(relation_error),
        baseline_log_ma=generic_residual_values(export["points"], export["baseline"]),
        corrected_log_ma=generic_residual_values(export["points"], export_metrics),
        importance_weights=export["importance_weights"],
    )
    summary = {
        "description": "Full positive Hermitian H-matrix on the deterministic exact gCICY model.",
        "model_seed": model.seed,
        "exact_integer_coefficients": args.exact_model,
        "degree": list(args.degree),
        "normalization": normalization,
        "initialization": initialization,
        "device": str(device),
        "precision": args.precision,
        "importance_weighted_training": args.importance_weighted,
        "ambient_section_count": int(len(basis.ambient_exponents)),
        "restricted_section_count": int(basis.numerical_rank),
        "basis_relation_error": relation_error,
        "epochs": args.epochs,
        "lr": args.lr,
        "passed_internal_gates": best_passed,
        "export_effective_sample_size": export["effective_sample_size"],
        "baseline_export": stats_dict(export_baseline_weighted if args.importance_weighted else export_baseline),
        "global_h_export": stats_dict(export_stats_weighted if args.importance_weighted else export_stats),
        "baseline_export_unweighted": stats_dict(export_baseline),
        "global_h_export_unweighted": stats_dict(export_stats),
        "baseline_export_weighted": stats_dict(export_baseline_weighted),
        "global_h_export_weighted": stats_dict(export_stats_weighted),
        "checks": check_summary,
        "history": history,
        "npz_file": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    primary_baseline = export_baseline_weighted if args.importance_weighted else export_baseline
    primary_stats = export_stats_weighted if args.importance_weighted else export_stats
    print(f"generic export MA rms: {primary_baseline.rms:.6e} -> {primary_stats.rms:.6e}")
    print(f"generic global H export min eigenvalue: {export_stats.min_eigenvalue:.6e}")
    print(f"passed_internal_gates={best_passed}")


if __name__ == "__main__":
    main()
