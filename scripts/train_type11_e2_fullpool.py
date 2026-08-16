#!/usr/bin/env python3
"""Minimize the fixed-normalization squared MA energy on one full point pool."""

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

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--basis-points", type=int, default=4096)
    parser.add_argument("--basis-seed", type=int, default=70499)
    parser.add_argument("--train-points", type=int, default=8192)
    parser.add_argument("--train-seed", type=int, default=72001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_h(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    value = 0.5 * (value + value.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    scale = float(eigenvalues[-1])
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError("H has no positive spectral scale")
    eigenvalues = np.maximum(eigenvalues, 1.0e-12 * scale)
    value = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.conjugate().T
    return value * (len(value) / float(np.trace(value).real))


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for full-pool E2 training") from exc

    args = parse_args()
    if args.degree <= 0 or args.epochs <= 0 or args.learning_rate <= 0:
        raise SystemExit("degree, epochs, and learning rate must be positive")
    if args.eval_every <= 0 or args.workers <= 0:
        raise SystemExit("eval-every and workers must be positive")
    if (
        args.basis_points <= 0
        or args.train_points <= 0
        or args.basis_points % 4
        or args.train_points % 4
    ):
        raise SystemExit("point counts must be positive multiples of four")

    started = time.perf_counter()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    real_dtype = torch.float32 if args.precision == "complex64" else torch.float64
    complex_dtype = torch.complex64 if args.precision == "complex64" else torch.complex128
    torch.manual_seed(20260717)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260717)
        torch.cuda.reset_peak_memory_stats(device)

    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    model = adapter.make_model(args.model_seed, exact=True)
    degree = (args.degree, args.degree)
    normalization = 1.0 / adapter.configuration.kahler_power(degree)

    print("sampling basis points and constructing the restricted section space", flush=True)
    basis_points, basis_shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.basis_points,
        seed=args.basis_seed,
        workers=args.workers,
        cluster_size=4,
        backend="process",
    )
    basis = adapter.restricted_section_basis(basis_points, degree)
    initial_h, relation_error = adapter.restricted_fubini_study_h_matrix(
        basis_points, basis
    )
    initial_h = normalize_h(initial_h)
    initial_cholesky = np.linalg.cholesky(initial_h)
    exponents = np.asarray(basis.selected_exponents, dtype=np.int64)
    print(
        f"basis rank={basis.numerical_rank}, relation_error={relation_error:.3e}",
        flush=True,
    )

    print("sampling and preparing the fixed full-pool E2 integration set", flush=True)
    train_points, train_shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.train_points,
        seed=args.train_seed,
        workers=args.workers,
        cluster_size=4,
        backend="process",
    )
    evaluated = [
        adapter.section_values_and_jacobian(point, exponents) for point in train_points
    ]
    prepared_complex = np.complex64 if args.precision == "complex64" else np.complex128
    values = np.asarray([row[0] for row in evaluated], dtype=prepared_complex)
    derivatives = np.asarray([row[1] for row in evaluated], dtype=prepared_complex)
    log_omega = np.asarray(
        [adapter.holomorphic_volume_log_density(point) for point in train_points],
        dtype=np.float32 if args.precision == "complex64" else np.float64,
    )
    weights = np.asarray(adapter.importance_weights(train_points), dtype=np.float64)
    weights /= np.sum(weights)
    effective_sample_size = float(1.0 / np.sum(weights**2))

    values_t = torch.tensor(values, dtype=complex_dtype, device=device)
    derivatives_t = torch.tensor(derivatives, dtype=complex_dtype, device=device)
    log_omega_t = torch.tensor(log_omega, dtype=real_dtype, device=device)
    weights_t = torch.tensor(weights, dtype=real_dtype, device=device)
    initial_cholesky_t = torch.tensor(
        initial_cholesky, dtype=complex_dtype, device=device
    )
    section_count = len(initial_h)
    lower_mask = torch.tril(
        torch.ones((section_count, section_count), dtype=real_dtype, device=device),
        diagonal=-1,
    )
    diagonal_log = torch.nn.Parameter(
        torch.zeros(section_count, dtype=real_dtype, device=device)
    )
    lower_real = torch.nn.Parameter(
        torch.zeros((section_count, section_count), dtype=real_dtype, device=device)
    )
    lower_imag = torch.nn.Parameter(torch.zeros_like(lower_real))
    optimizer = torch.optim.Adam(
        [diagonal_log, lower_real, lower_imag], lr=args.learning_rate
    )

    def current_h():
        lower = lower_mask * lower_real + 1j * lower_mask * lower_imag
        relative_factor = lower.to(complex_dtype) + torch.diag(
            torch.exp(diagonal_log)
        ).to(complex_dtype)
        factor = initial_cholesky_t @ relative_factor
        candidate = factor @ torch.conj(factor.T)
        return candidate * (section_count / torch.real(torch.trace(candidate)))

    def raw_log_eta(h_matrix):
        h_values = torch.einsum("ab,nb->na", h_matrix, values_t)
        denominator = torch.real(
            torch.einsum("na,na->n", torch.conj(values_t), h_values)
        )
        h_derivatives = torch.einsum("ab,nbj->naj", h_matrix, derivatives_t)
        first = torch.einsum(
            "nmi,nmj->nij", torch.conj(derivatives_t), h_derivatives
        )
        gradient = torch.einsum(
            "nm,nmj->nj", torch.conj(values_t), h_derivatives
        )
        metric = first / denominator[:, None, None]
        metric -= (
            torch.conj(gradient)[:, :, None]
            * gradient[:, None, :]
            / denominator[:, None, None] ** 2
        )
        metric = normalization * 0.5 * (
            metric + torch.conj(torch.transpose(metric, 1, 2))
        )
        eigenvalues = torch.linalg.eigvalsh(metric)
        raw = torch.log(torch.clamp(eigenvalues, min=1.0e-12)).sum(dim=1)
        return raw - log_omega_t, eigenvalues

    def weighted_log_mean_exp(raw):
        maximum = torch.max(raw)
        return maximum + torch.log(torch.sum(weights_t * torch.exp(raw - maximum)))

    with torch.no_grad():
        initial_raw, _ = raw_log_eta(current_h())
        fixed_log_kappa = weighted_log_mean_exp(initial_raw).detach()
    print(f"fixed training-pool log(kappa)={float(fixed_log_kappa.cpu()):.12e}", flush=True)

    history: list[dict[str, float | int]] = []
    best_energy = float("inf")
    best_h = initial_h.copy()
    termination = "completed_requested_epochs"
    for epoch in range(args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        h_matrix = current_h()
        raw, eigenvalues = raw_log_eta(h_matrix)
        ratio = torch.exp(raw - fixed_log_kappa)
        energy = torch.sum(weights_t * (ratio - 1.0) ** 2)
        if not bool(torch.isfinite(energy)):
            termination = "nonfinite_energy"
            print(f"epoch {epoch}: non-finite E2; stopping", flush=True)
            break
        energy_value = float(energy.detach().cpu())
        if energy_value < best_energy:
            best_energy = energy_value
            best_h = normalize_h(h_matrix.detach().cpu().numpy())
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            sigma = torch.sum(weights_t * torch.abs(ratio - 1.0))
            row = {
                "epoch": epoch,
                "squared_energy": energy_value,
                "sqrt_squared_energy": float(np.sqrt(energy_value)),
                "sigma": float(sigma.detach().cpu()),
                "maximum_ratio": float(torch.max(ratio).detach().cpu()),
                "minimum_ratio": float(torch.min(ratio).detach().cpu()),
                "minimum_metric_eigenvalue": float(
                    torch.min(eigenvalues).detach().cpu()
                ),
            }
            history.append(row)
            print(
                f"epoch {epoch}: E2={row['squared_energy']:.6e}, "
                f"sigma={row['sigma']:.6e}, max_r={row['maximum_ratio']:.6e}, "
                f"min_metric_eig={row['minimum_metric_eigenvalue']:.3e}",
                flush=True,
            )
        if epoch == args.epochs:
            break
        energy.backward()
        optimizer.step()

    artifact_path = args.out.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **adapter.artifact_model_payload(model, exact=True),
        "pipeline_schema_version": np.asarray(1),
        "pipeline_adapter": np.asarray(adapter.key),
        "importance_weighted_training": np.asarray(True),
        "global_section_degree": np.asarray(degree, dtype=np.int64),
        "global_section_exponents": exponents,
        "global_h_matrix": best_h,
        "global_h_positive_relative_floor": np.asarray(1.0e-12),
        "global_section_normalization": np.asarray(normalization),
        "basis_selected_indices": np.asarray(basis.selected_indices, dtype=np.int64),
        "basis_relation_error": np.asarray(relation_error),
        "h_parameterization": np.asarray("reference_whitened_fullpool_e2"),
        "e2_fixed_log_kappa": np.asarray(float(fixed_log_kappa.cpu())),
        "e2_train_seed": np.asarray(args.train_seed),
        "e2_train_points": np.asarray(args.train_points),
        "e2_best_squared_energy": np.asarray(best_energy),
    }
    np.savez_compressed(artifact_path, **payload)
    device_memory = None
    if device.type == "cuda":
        device_memory = {
            "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    summary = {
        "schema": "type11-fullpool-squared-ma-energy-v1",
        "artifact": str(artifact_path),
        "model_seed": args.model_seed,
        "degree": list(degree),
        "normalization": normalization,
        "device": str(device),
        "precision": args.precision,
        "device_memory": device_memory,
        "basis_points": args.basis_points,
        "basis_seed": args.basis_seed,
        "basis_shards": basis_shards,
        "basis_rank": basis.numerical_rank,
        "basis_relation_error": relation_error,
        "train_points": args.train_points,
        "train_seed": args.train_seed,
        "train_shards": train_shards,
        "importance_effective_sample_size": effective_sample_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "fixed_log_kappa": float(fixed_log_kappa.cpu()),
        "best_squared_energy": best_energy,
        "termination_reason": termination,
        "history": history,
        "runtime_seconds": time.perf_counter() - started,
        "objective_note": (
            "This is the target-measure weighted empirical integral of "
            "(eta/kappa - 1)^2 with kappa fixed from the initial FS metric on "
            "the same integration pool. No CVaR, spectral penalty, replay, or "
            "gradient clipping is used."
        ),
    }
    summary["artifact_sha256"] = sha256_file(artifact_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {artifact_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
