#!/usr/bin/env python3
"""Train a scalable positive low-rank H-matrix on an exact gCICY model."""

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
from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402


def parse_degree(text: str) -> tuple[int, int, int]:
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 3 or min(values) <= 0:
        raise argparse.ArgumentTypeError("degree must be three positive comma-separated integers")
    return values


def parse_seeds(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("check seeds must be a non-empty distinct list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=20260711)
    parser.add_argument("--exact-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--degree", type=parse_degree, default=(2, 2, 2))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--initial-artifact", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument(
        "--importance-weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train and gate on the Omega-volume importance-weighted MA residual",
    )
    parser.add_argument("--basis-points", type=int, default=1024)
    parser.add_argument("--train-points", type=int, default=2048)
    parser.add_argument("--validation-points", type=int, default=1024)
    parser.add_argument("--check-points", type=int, default=512)
    parser.add_argument("--export-points", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience-evals", type=int, default=6)
    parser.add_argument("--validation-weight", type=float, default=0.75)
    parser.add_argument("--factor-regularization", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--parameterization",
        choices=("whitened", "direct"),
        default="whitened",
        help="apply the low-rank congruence in H0-whitened or raw section coordinates",
    )
    parser.add_argument("--torch-seed", type=int)
    parser.add_argument("--factor-seed", type=int)
    parser.add_argument("--basis-seed", type=int, default=5201)
    parser.add_argument("--train-seed", type=int, default=5202)
    parser.add_argument("--validation-seed", type=int, default=5203)
    parser.add_argument(
        "--check-seeds",
        type=parse_seeds,
        default=(5301, 5302, 5303, 5304),
    )
    parser.add_argument("--export-seed", type=int, default=5401)
    parser.add_argument("--sampling-workers", type=int, default=1)
    parser.add_argument("--sampling-cluster-size", type=int, default=3)
    parser.add_argument(
        "--sampling-backend",
        choices=("process", "thread"),
        default="process",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def default_output_paths(
    degree: tuple[int, int, int], rank: int, importance_weighted: bool
) -> tuple[Path, Path]:
    degree_label = "k" + str(degree[0]) if degree[0] == degree[1] == degree[2] else "_".join(map(str, degree))
    stem = f"gcicy_generic_global_h_metric_{degree_label}_rank{rank}"
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


def hermitian_projection(matrix: np.ndarray) -> np.ndarray:
    """Remove accelerator roundoff before evaluating or exporting an H matrix."""

    value = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (value + value.conjugate().T)


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for low-rank H-matrix training.") from exc

    args = parse_args()
    if args.degree[0] != args.degree[1] or args.degree[1] != args.degree[2]:
        raise SystemExit("The convergence trainer currently requires degree (k,k,k).")
    if args.rank <= 0 or args.epsilon <= 0:
        raise SystemExit("rank and epsilon must be positive")
    if args.sampling_workers <= 0 or args.sampling_cluster_size <= 0:
        raise SystemExit("sampling-workers and sampling-cluster-size must be positive")
    if args.sampling_workers > 1:
        point_counts = (
            args.basis_points,
            args.train_points,
            args.validation_points,
            args.check_points,
            args.export_points,
        )
        if any(count % args.sampling_cluster_size for count in point_counts):
            raise SystemExit(
                "parallel sampling point counts must be divisible by sampling-cluster-size"
            )
    default_out, default_summary = default_output_paths(args.degree, args.rank, args.importance_weighted)
    args.out = args.out or default_out
    args.summary = args.summary or default_summary
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
    torch_seed = args.torch_seed if args.torch_seed is not None else 5200 + args.degree[0]
    factor_seed = args.factor_seed if args.factor_seed is not None else 5204 + args.degree[0]
    torch.manual_seed(torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(torch_seed)
    model = make_exact_generic_model(args.model_seed) if args.exact_model else make_generic_model(args.model_seed)
    adapter = get_adapter("p1p1p5_type22")
    sampling_shards: dict[str, list[dict[str, int]]] = {}

    def sample(label: str, count: int, seed: int):
        if args.sampling_workers <= 1:
            return sample_generic_gcicy_points(model, count, seed=seed)
        points, metadata = sample_points_parallel(
            adapter,
            model_seed=args.model_seed,
            exact_model=args.exact_model,
            count=count,
            seed=seed,
            workers=args.sampling_workers,
            cluster_size=args.sampling_cluster_size,
            backend=args.sampling_backend,
        )
        sampling_shards[label] = metadata
        return points

    basis_points = sample("basis", args.basis_points, args.basis_seed)
    basis = generic_restricted_ambient_basis(basis_points, args.degree)
    if args.rank >= basis.numerical_rank:
        raise SystemExit("rank must be smaller than the restricted section dimension")
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
    initial_h *= len(initial_h) / float(np.trace(initial_h).real)
    initial_cholesky = np.linalg.cholesky(initial_h)
    exponents = basis.selected_exponents

    def prepare(points):
        values = []
        derivatives = []
        for point in points:
            section_values, section_derivatives = generic_section_values_and_jacobian(point, exponents)
            values.append(section_values)
            derivatives.append(section_derivatives)
        values_array = np.asarray(values, dtype=np.complex128)
        derivatives_array = np.asarray(derivatives, dtype=np.complex128)
        whitened_values = np.einsum("ab,nb->na", initial_cholesky.conjugate().T, values_array)
        whitened_derivatives = np.einsum(
            "ab,nbj->naj", initial_cholesky.conjugate().T, derivatives_array
        )
        importance_weights = generic_importance_weights(points)
        return {
            "points": points,
            "values": values_array,
            "derivatives": derivatives_array,
            "h0_values": np.einsum("ab,nb->na", initial_h, values_array),
            "h0_derivatives": np.einsum("ab,nbj->naj", initial_h, derivatives_array),
            "whitened_values": whitened_values,
            "whitened_derivatives": whitened_derivatives,
            "log_omega": np.asarray(
                [generic_holomorphic_volume_log_density(point) for point in points], dtype=float
            ),
            "baseline": generic_baseline_metrics(points),
            "importance_weights": importance_weights,
            "effective_sample_size": generic_effective_sample_size(importance_weights),
        }

    train = prepare(sample("train", args.train_points, args.train_seed))
    validation = prepare(
        sample("validation", args.validation_points, args.validation_seed)
    )
    checks = [
        (f"seed{seed}", prepare(sample(f"check_seed{seed}", args.check_points, seed)))
        for seed in args.check_seeds
    ]
    export = prepare(sample("export", args.export_points, args.export_seed))

    section_count = basis.numerical_rank
    rng = np.random.default_rng(factor_seed)
    frame = rng.normal(size=(section_count, args.rank)) + 1j * rng.normal(size=(section_count, args.rank))
    initial_v, _ = np.linalg.qr(frame)
    initial_v = initial_v[:, : args.rank]
    u_real = torch.nn.Parameter(torch.zeros((section_count, args.rank), dtype=real_dtype, device=device))
    u_imag = torch.nn.Parameter(torch.zeros((section_count, args.rank), dtype=real_dtype, device=device))
    v_real = torch.nn.Parameter(torch.tensor(initial_v.real, dtype=real_dtype, device=device))
    v_imag = torch.nn.Parameter(torch.tensor(initial_v.imag, dtype=real_dtype, device=device))
    optimizer = torch.optim.Adam([u_real, u_imag, v_real, v_imag], lr=args.lr)
    h0_t = torch.tensor(initial_h, dtype=complex_dtype, device=device)
    cholesky_t = torch.tensor(initial_cholesky, dtype=complex_dtype, device=device)
    identity_t = torch.eye(section_count, dtype=complex_dtype, device=device)
    initial_v_t = torch.tensor(initial_v, dtype=complex_dtype, device=device)

    def factors():
        u = u_real.to(complex_dtype) + 1j * u_imag
        v = v_real.to(complex_dtype) + 1j * v_imag
        return u, v

    def dense_h_matrix():
        u, v = factors()
        transform = identity_t + u @ torch.conj(v.T)
        if args.parameterization == "whitened":
            transformed_cholesky = cholesky_t @ transform
            h_matrix = transformed_cholesky @ torch.conj(transformed_cholesky.T) + args.epsilon * h0_t
        else:
            h_matrix = transform @ h0_t @ torch.conj(transform.T) + args.epsilon * h0_t
        return h_matrix * (section_count / torch.real(torch.trace(h_matrix)))

    torch_cache: dict[int, tuple] = {}

    def as_torch(dataset):
        key = id(dataset)
        if key not in torch_cache:
            torch_cache[key] = (
                torch.tensor(dataset["values"], dtype=complex_dtype, device=device),
                torch.tensor(dataset["derivatives"], dtype=complex_dtype, device=device),
                torch.tensor(dataset["h0_values"], dtype=complex_dtype, device=device),
                torch.tensor(dataset["h0_derivatives"], dtype=complex_dtype, device=device),
                torch.tensor(dataset["whitened_values"], dtype=complex_dtype, device=device),
                torch.tensor(dataset["whitened_derivatives"], dtype=complex_dtype, device=device),
                torch.tensor(dataset["log_omega"], dtype=real_dtype, device=device),
                torch.tensor(dataset["importance_weights"], dtype=real_dtype, device=device),
            )
        return torch_cache[key]

    def apply_structured_h(values, h0_values, whitened_values, u, v):
        if args.parameterization == "whitened":
            u_dagger_values = torch.einsum("mr,nm...->nr...", torch.conj(u), whitened_values)
            transformed = whitened_values + torch.einsum("mr,nr...->nm...", v, u_dagger_values)
            v_dagger_transformed = torch.einsum("mr,nm...->nr...", torch.conj(v), transformed)
            return (
                h0_values
                + torch.einsum("mr,nr...->nm...", cholesky_t @ v, u_dagger_values)
                + torch.einsum("mr,nr...->nm...", cholesky_t @ u, v_dagger_transformed)
                + args.epsilon * h0_values
            )
        u_dagger_values = torch.einsum("mr,nm...->nr...", torch.conj(u), values)
        transformed_h0 = h0_values + torch.einsum("mr,nr...->nm...", h0_t @ v, u_dagger_values)
        v_dagger_transformed = torch.einsum("mr,nm...->nr...", torch.conj(v), transformed_h0)
        return transformed_h0 + torch.einsum("mr,nr...->nm...", u, v_dagger_transformed) + args.epsilon * h0_values

    def raw_and_barrier(dataset):
        (
            values,
            derivatives,
            h0_values,
            h0_derivatives,
            whitened_values,
            whitened_derivatives,
            log_omega,
            importance_weights,
        ) = as_torch(dataset)
        u, v = factors()
        h_values = apply_structured_h(values, h0_values, whitened_values, u, v)
        h_derivatives = apply_structured_h(
            derivatives, h0_derivatives, whitened_derivatives, u, v
        )
        denominator = torch.real(torch.einsum("na,na->n", torch.conj(values), h_values))
        first = torch.einsum("nmi,nmj->nij", torch.conj(derivatives), h_derivatives)
        gradient = torch.einsum("nm,nmj->nj", torch.conj(values), h_derivatives)
        metric = first / denominator[:, None, None]
        metric -= torch.conj(gradient)[:, :, None] * gradient[:, None, :] / denominator[:, None, None] ** 2
        metric *= normalization
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

    best_h = hermitian_projection((1.0 + args.epsilon) * initial_h)
    initial_rows = []
    for name, dataset in checks:
        baseline, candidate = evaluate(dataset, best_h)
        passed = np.isfinite(candidate.rms) and candidate.min_eigenvalue > 0 and candidate.rms < baseline.rms
        initial_rows.append((name, candidate, passed))
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
        train_raw, train_barrier, train_weights = raw_and_barrier(train)
        validation_raw, validation_barrier, validation_weights = raw_and_barrier(validation)
        train_mean = torch.sum(train_weights * train_raw) / torch.sum(train_weights)
        validation_mean = torch.sum(validation_weights * validation_raw) / torch.sum(validation_weights)
        shared_mean = (train_mean + args.validation_weight * validation_mean) / (
            1.0 + args.validation_weight
        )
        train_loss = torch.sum(train_weights * (train_raw - shared_mean) ** 2) / torch.sum(train_weights)
        validation_loss = torch.sum(validation_weights * (validation_raw - shared_mean) ** 2) / torch.sum(validation_weights)
        u, v = factors()
        factor_penalty = torch.mean(torch.abs(u) ** 2) + torch.mean(torch.abs(v - initial_v_t) ** 2)
        loss = (
            train_loss
            + args.validation_weight * validation_loss
            + 200.0 * (train_barrier + validation_barrier)
            + args.factor_regularization * factor_penalty
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([u_real, u_imag, v_real, v_imag], max_norm=10.0)
        optimizer.step()

        if epoch % args.eval_every != 0:
            continue
        candidate_h = hermitian_projection(dense_h_matrix().detach().cpu().numpy())
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
        low_rank=np.asarray(args.rank),
        low_rank_epsilon=np.asarray(args.epsilon),
        basis_selected_indices=basis.selected_indices,
        basis_relation_error=np.asarray(relation_error),
        baseline_log_ma=generic_residual_values(export["points"], export["baseline"]),
        corrected_log_ma=generic_residual_values(export["points"], export_metrics),
        importance_weights=export["importance_weights"],
    )
    summary = {
        "description": "Scalable positive low-rank H-matrix on the deterministic exact gCICY model.",
        "model_seed": model.seed,
        "exact_integer_coefficients": args.exact_model,
        "degree": list(args.degree),
        "normalization": normalization,
        "initialization": initialization,
        "device": str(device),
        "precision": args.precision,
        "parameterization": args.parameterization,
        "importance_weighted_training": args.importance_weighted,
        "rank": args.rank,
        "epsilon": args.epsilon,
        "ambient_section_count": int(len(basis.ambient_exponents)),
        "restricted_section_count": int(basis.numerical_rank),
        "basis_relation_error": relation_error,
        "point_counts": {
            "basis": args.basis_points,
            "train": args.train_points,
            "validation": args.validation_points,
            "check_per_seed": args.check_points,
            "export": args.export_points,
        },
        "seeds": {
            "torch": torch_seed,
            "factor": factor_seed,
            "basis": args.basis_seed,
            "train": args.train_seed,
            "validation": args.validation_seed,
            "checks": list(args.check_seeds),
            "export": args.export_seed,
        },
        "parallel_sampling": {
            "workers": args.sampling_workers,
            "cluster_size": args.sampling_cluster_size,
            "backend": args.sampling_backend,
            "seed_derivation": "numpy.random.SeedSequence(base_seed).spawn(active_workers)",
            "shards": sampling_shards,
        },
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
    print(f"generic low-rank H export min eigenvalue: {export_stats.min_eigenvalue:.6e}")
    print(f"passed_internal_gates={best_passed}")


if __name__ == "__main__":
    main()
