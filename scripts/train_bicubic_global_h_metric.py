#!/usr/bin/env python3
"""Train a weighted full-H metric on the exact ordinary bicubic control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.bicubic_model import (  # noqa: E402
    bicubic_baseline_metrics,
    bicubic_fubini_study_h_matrix,
    bicubic_global_h_metrics,
    bicubic_holomorphic_volume_log_density,
    bicubic_importance_weights,
    bicubic_lift_h_matrix_product,
    bicubic_lift_h_matrix_power,
    bicubic_residual_stats,
    bicubic_residual_values,
    bicubic_restricted_basis,
    bicubic_section_values_and_jacobian,
    make_exact_bicubic_model,
    sample_bicubic_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=20260721)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--initial-artifact", type=Path)
    parser.add_argument("--multiply-initial-artifact", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--importance-weighted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--basis-points", type=int, default=512)
    parser.add_argument("--train-points", type=int, default=2048)
    parser.add_argument("--validation-points", type=int, default=1024)
    parser.add_argument("--check-points", type=int, default=512)
    parser.add_argument("--export-points", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience-evals", type=int, default=6)
    parser.add_argument("--validation-weight", type=float, default=0.75)
    parser.add_argument("--drift-weight", type=float, default=1e-5)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def default_paths(degree: int, weighted: bool) -> tuple[Path, Path]:
    stem = f"bicubic_global_h_metric_k{degree}"
    if weighted:
        stem += "_weighted"
    return ROOT / "outputs" / f"{stem}.npz", ROOT / "outputs" / f"{stem}_summary.json"


def stats_dict(stats) -> dict[str, float]:
    return {
        "rms": stats.rms,
        "max_abs": stats.max_abs,
        "min_eigenvalue": stats.min_eigenvalue,
        "mean_log_error": stats.mean_log_error,
    }


def hermitian_projection(matrix: np.ndarray) -> np.ndarray:
    """Remove accelerator roundoff before evaluating or exporting an H matrix."""

    value = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (value + value.conjugate().T)


def effective_sample_size(weights: np.ndarray) -> float:
    return float(np.sum(weights) ** 2 / np.sum(weights**2))


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for bicubic full-H training") from exc

    args = parse_args()
    if args.degree <= 0:
        raise SystemExit("degree must be positive")
    default_out, default_summary = default_paths(args.degree, args.importance_weighted)
    args.out = args.out or default_out
    args.summary = args.summary or default_summary
    normalization = 1.0 / args.degree

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    real_dtype = torch.float32 if args.precision == "complex64" else torch.float64
    complex_dtype = torch.complex64 if args.precision == "complex64" else torch.complex128
    torch.set_default_dtype(real_dtype)
    torch.manual_seed(8200 + args.degree)

    model = make_exact_bicubic_model(args.model_seed)
    basis_points = sample_bicubic_points(model, args.basis_points, seed=8201)
    basis = bicubic_restricted_basis(basis_points, args.degree)
    initial_h, relation_error = bicubic_fubini_study_h_matrix(basis_points, basis)
    initialization = "ambient_fubini_study"
    if args.multiply_initial_artifact is not None and args.initial_artifact is None:
        raise SystemExit("--multiply-initial-artifact requires --initial-artifact")

    def load_matching_artifact(path: Path):
        source = np.load(path)
        if int(source["bicubic_model_seed"]) != model.seed or not np.allclose(
            source["bicubic_coefficients"], model.coefficients
        ):
            raise SystemExit(f"initial artifact does not match the bicubic model: {path}")
        return source

    if args.initial_artifact is not None:
        source = load_matching_artifact(args.initial_artifact)
        if args.multiply_initial_artifact is None:
            initial_h, relation_error = bicubic_lift_h_matrix_power(
                basis_points,
                source["global_section_exponents"],
                source["global_h_matrix"],
                basis,
            )
            initialization = f"lifted_power_of:{args.initial_artifact.resolve()}"
        else:
            multiplier = load_matching_artifact(args.multiply_initial_artifact)
            initial_h, relation_error = bicubic_lift_h_matrix_product(
                basis_points,
                source["global_section_exponents"],
                source["global_h_matrix"],
                multiplier["global_section_exponents"],
                multiplier["global_h_matrix"],
                basis,
            )
            initialization = (
                f"product_of:{args.initial_artifact.resolve()}"
                f"*{args.multiply_initial_artifact.resolve()}"
            )
    initial_h *= len(initial_h) / float(np.trace(initial_h).real)
    initial_cholesky = np.linalg.cholesky(initial_h)
    exponents = basis.selected_exponents

    def prepare(points):
        values = []
        derivatives = []
        for point in points:
            section_values, section_derivatives = bicubic_section_values_and_jacobian(point, exponents)
            values.append(section_values)
            derivatives.append(section_derivatives)
        importance = bicubic_importance_weights(points)
        return {
            "points": points,
            "values": np.asarray(values, dtype=np.complex128),
            "derivatives": np.asarray(derivatives, dtype=np.complex128),
            "log_omega": np.asarray(
                [bicubic_holomorphic_volume_log_density(point) for point in points], dtype=float
            ),
            "baseline": bicubic_baseline_metrics(points),
            "importance_weights": importance,
            "effective_sample_size": effective_sample_size(importance),
        }

    train = prepare(sample_bicubic_points(model, args.train_points, seed=8202))
    validation = prepare(sample_bicubic_points(model, args.validation_points, seed=8203))
    checks = [
        (f"seed{seed}", prepare(sample_bicubic_points(model, args.check_points, seed=seed)))
        for seed in (8301, 8302, 8303, 8304)
    ]
    export = prepare(sample_bicubic_points(model, args.export_points, seed=8401))

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
        values, derivatives, log_omega, importance = as_torch(dataset)
        h_values = torch.einsum("ab,nb->na", h_matrix, values)
        denominator = torch.real(torch.einsum("na,na->n", torch.conj(values), h_values))
        h_derivatives = torch.einsum("ab,nbj->naj", h_matrix, derivatives)
        first = torch.einsum("nmi,nmj->nij", torch.conj(derivatives), h_derivatives)
        gradient = torch.einsum("nm,nmj->nj", torch.conj(values), h_derivatives)
        metric = first / denominator[:, None, None]
        metric -= torch.conj(gradient)[:, :, None] * gradient[:, None, :] / denominator[:, None, None] ** 2
        metric *= normalization
        metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        eigenvalues = torch.linalg.eigvalsh(metric)
        raw = torch.log(torch.clamp(eigenvalues, min=1e-12)).sum(dim=1) - log_omega
        scale = torch.clamp(torch.mean(torch.abs(eigenvalues), dim=1, keepdim=True), min=1e-14)
        barrier = torch.nn.functional.softplus((1e-8 - eigenvalues / scale) * 80.0).mean() / 80.0
        weights = importance if args.importance_weighted else torch.ones_like(importance)
        return raw, barrier, weights

    def evaluate(dataset, h_matrix):
        metrics = bicubic_global_h_metrics(
            dataset["points"], exponents, h_matrix, normalization=normalization
        )
        weights = dataset["importance_weights"] if args.importance_weighted else None
        return bicubic_residual_stats(dataset["points"], dataset["baseline"], weights), bicubic_residual_stats(
            dataset["points"], metrics, weights
        )

    initial_rows = []
    for name, dataset in checks:
        baseline, candidate = evaluate(dataset, initial_h)
        passed = np.isfinite(candidate.rms) and candidate.min_eigenvalue > 0 and candidate.rms < baseline.rms
        initial_rows.append((name, candidate, passed))
    best_h = hermitian_projection(initial_h)
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
    stale_evaluations = 0
    print(
        f"initialization={initialization}, sections={basis.numerical_rank}, device={device}, "
        f"precision={args.precision}, mean_check_rms={best_score:.6e}",
        flush=True,
    )

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        h_matrix = current_h_matrix()
        train_raw, train_barrier, train_weights = raw_and_barrier(train, h_matrix)
        validation_raw, validation_barrier, validation_weights = raw_and_barrier(validation, h_matrix)
        train_mean = torch.sum(train_weights * train_raw) / torch.sum(train_weights)
        validation_mean = torch.sum(validation_weights * validation_raw) / torch.sum(validation_weights)
        shared_mean = (train_mean + args.validation_weight * validation_mean) / (1 + args.validation_weight)
        train_loss = torch.sum(train_weights * (train_raw - shared_mean) ** 2) / torch.sum(train_weights)
        validation_loss = torch.sum(validation_weights * (validation_raw - shared_mean) ** 2) / torch.sum(
            validation_weights
        )
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
        candidate_h = hermitian_projection(current_h_matrix().detach().cpu().numpy())
        rows = []
        all_passed = True
        for name, dataset in checks:
            baseline, candidate = evaluate(dataset, candidate_h)
            passed = np.isfinite(candidate.rms) and candidate.min_eigenvalue > 0 and candidate.rms < baseline.rms
            all_passed = all_passed and passed
            rows.append((name, candidate, passed))
        score = float(np.mean([row[1].rms for row in rows]))
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
        for name, candidate, passed in rows:
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
            print(f"early stopping after {stale_evaluations} stale evaluations", flush=True)
            break

    export_metrics = bicubic_global_h_metrics(
        export["points"], exponents, best_h, normalization=normalization
    )
    export_baseline_unweighted = bicubic_residual_stats(export["points"], export["baseline"])
    export_unweighted = bicubic_residual_stats(export["points"], export_metrics)
    export_baseline_weighted = bicubic_residual_stats(
        export["points"], export["baseline"], export["importance_weights"]
    )
    export_weighted = bicubic_residual_stats(export["points"], export_metrics, export["importance_weights"])
    primary_baseline = export_baseline_weighted if args.importance_weighted else export_baseline_unweighted
    primary = export_weighted if args.importance_weighted else export_unweighted

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
        bicubic_model_seed=np.asarray(model.seed),
        bicubic_exponents=model.exponents,
        bicubic_coefficients=model.coefficients,
        global_section_degree=np.asarray(args.degree),
        global_section_exponents=exponents,
        global_h_matrix=best_h,
        global_section_normalization=np.asarray(normalization),
        importance_weighted_training=np.asarray(args.importance_weighted),
        basis_selected_indices=basis.selected_indices,
        basis_relation_error=np.asarray(relation_error),
        baseline_log_ma=bicubic_residual_values(export["points"], export["baseline"]),
        corrected_log_ma=bicubic_residual_values(export["points"], export_metrics),
        importance_weights=export["importance_weights"],
    )
    summary = {
        "description": "Weighted full positive H metric on an exact ordinary bicubic control.",
        "model_seed": model.seed,
        "degree": args.degree,
        "normalization": normalization,
        "initialization": initialization,
        "device": str(device),
        "precision": args.precision,
        "importance_weighted_training": args.importance_weighted,
        "ambient_section_count": int(len(basis.ambient_exponents)),
        "restricted_section_count": basis.numerical_rank,
        "basis_relation_error": relation_error,
        "export_effective_sample_size": export["effective_sample_size"],
        "baseline_export": stats_dict(primary_baseline),
        "global_h_export": stats_dict(primary),
        "baseline_export_unweighted": stats_dict(export_baseline_unweighted),
        "global_h_export_unweighted": stats_dict(export_unweighted),
        "baseline_export_weighted": stats_dict(export_baseline_weighted),
        "global_h_export_weighted": stats_dict(export_weighted),
        "passed_internal_gates": best_passed,
        "checks": check_summary,
        "history": history,
        "npz_file": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(f"bicubic export MA rms: {primary_baseline.rms:.6e} -> {primary.rms:.6e}")
    print(f"bicubic H export min eigenvalue: {primary.min_eigenvalue:.6e}")


if __name__ == "__main__":
    main()
