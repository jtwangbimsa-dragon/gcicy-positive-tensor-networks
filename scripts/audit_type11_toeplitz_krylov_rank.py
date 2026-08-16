#!/usr/bin/env python3
"""Audit whether O(k) Berezin/Krylov modes recover type-(1,1) T-map updates."""

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
from gcicy_metric.pipeline.audit import (  # noqa: E402
    normalized_volume_ratios,
    standard_errors,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from gcicy_metric.pipeline.toeplitz_spectral import (  # noqa: E402
    affine_exponential_update,
    affine_log_tangent,
    fit_hermitian_span,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--seed", type=int, default=72111)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode-counts", default="2,4,6,8,10,12")
    parser.add_argument(
        "--residual-sources",
        default="ma,balance,target",
        help="comma-separated subset of ma,balance,target",
    )
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument(
        "--start-iterations",
        help="optional comma-separated starting iterations; each requires iteration+1",
    )
    parser.add_argument("--power-iterations", type=int, default=24)
    parser.add_argument("--spectral-safety-factor", type=float, default=1.02)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_integers(
    text: str,
    *,
    label: str,
    allow_zero: bool = False,
) -> list[int]:
    try:
        values = sorted({int(piece.strip()) for piece in text.split(",") if piece.strip()})
    except ValueError as exc:
        raise SystemExit(f"{label} must be comma-separated integers") from exc
    minimum = 0 if allow_zero else 1
    if not values or values[0] < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SystemExit(f"{label} must contain {qualifier} integers")
    return values


def parse_residual_sources(text: str) -> list[str]:
    values = list(dict.fromkeys(piece.strip() for piece in text.split(",") if piece.strip()))
    allowed = {"ma", "balance", "target"}
    invalid = sorted(set(values).difference(allowed))
    if not values or invalid:
        raise SystemExit(
            "residual-sources must be a non-empty subset of ma,balance,target; "
            f"invalid={invalid}"
        )
    return values


def load_checkpoint(path: Path) -> tuple[int, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "schema",
            "iteration",
            "h_matrix",
            "model_seed",
            "degree",
            "train_seed",
            "train_points",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"trajectory checkpoint {path} lacks {missing}")
        metadata = {
            "schema": str(payload["schema"]),
            "model_seed": int(payload["model_seed"]),
            "degree": [int(value) for value in np.asarray(payload["degree"]).tolist()],
            "train_seed": int(payload["train_seed"]),
            "train_points": int(payload["train_points"]),
        }
        iteration = int(payload["iteration"])
        h_matrix = np.asarray(payload["h_matrix"], dtype=np.complex128)
    return iteration, h_matrix, metadata


def select_adjacent_pairs(
    checkpoints: dict[int, Path],
    *,
    starts: list[int] | None,
    maximum: int,
) -> list[tuple[int, int]]:
    available = sorted(
        (iteration, iteration + 1)
        for iteration in checkpoints
        if iteration + 1 in checkpoints
    )
    if starts is not None:
        requested = [(iteration, iteration + 1) for iteration in starts]
        missing = [pair for pair in requested if pair not in available]
        if missing:
            raise ValueError(f"trajectory lacks requested adjacent pairs: {missing}")
        return requested
    if not available:
        raise ValueError("trajectory has no adjacent iteration checkpoints")
    if len(available) <= maximum:
        return available
    indices = np.unique(
        np.rint(np.linspace(0, len(available) - 1, maximum)).astype(np.int64)
    )
    return [available[int(index)] for index in indices]


def torch_coherent_sections(values, h_matrix):
    import torch

    h_matrix = 0.5 * (h_matrix + torch.conj(h_matrix.T))
    eigenvalues, eigenvectors = torch.linalg.eigh(h_matrix)
    if not bool(torch.all(eigenvalues > 0)):
        raise FloatingPointError("H is not positive during coherent-state evaluation")
    square_root = (eigenvectors * torch.sqrt(eigenvalues)[None, :]) @ torch.conj(
        eigenvectors.T
    )
    coherent = values @ square_root.T
    norms = torch.linalg.vector_norm(coherent, dim=1)
    if not bool(torch.all(norms > 0)):
        raise FloatingPointError("coherent section normalization failed")
    return coherent / norms[:, None]


def torch_apply_berezin(coherent, weights, field):
    import torch

    moment = (coherent.T * (weights * field)[None, :]) @ torch.conj(coherent)
    return coherent.shape[1] * torch.real(
        torch.einsum("na,ab,nb->n", torch.conj(coherent), moment, coherent)
    )


def torch_weighted_center(field, weights):
    import torch

    return field - torch.sum(weights * field)


def torch_weighted_norm(field, weights):
    import torch

    return torch.sqrt(torch.sum(weights * field**2))


def torch_berezin_geometry(
    coherent,
    weights,
    *,
    power_iterations: int,
):
    import torch

    vector = torch.ones_like(weights)
    vector += 0.01 * torch.sin(
        torch.arange(len(vector), dtype=vector.dtype, device=vector.device)
    )
    vector /= torch_weighted_norm(vector, weights)
    for _ in range(power_iterations):
        candidate = torch_apply_berezin(coherent, weights, vector)
        vector = candidate / torch_weighted_norm(candidate, weights)
    applied = torch_apply_berezin(coherent, weights, vector)
    radius = torch.sum(weights * vector * applied)

    identity = torch.eye(
        coherent.shape[1], dtype=coherent.dtype, device=coherent.device
    )
    balanced_moment = (coherent.T * weights[None, :]) @ torch.conj(coherent)
    balance_operator = coherent.shape[1] * balanced_moment - identity
    balance_residual = torch.real(
        torch.einsum(
            "na,ab,nb->n",
            torch.conj(coherent),
            balance_operator,
            coherent,
        )
    )
    balance_defect = torch.linalg.matrix_norm(balance_operator) / np.sqrt(
        coherent.shape[1]
    )
    return radius, balance_residual, balance_defect


def torch_berezin_modes_and_lifts(
    coherent,
    weights,
    residual,
    *,
    mode_count: int,
    spectral_radius,
    spectral_safety_factor: float,
):
    import torch

    scaled_radius = spectral_safety_factor * spectral_radius

    def scaled_apply(field):
        value = 2.0 * torch_apply_berezin(coherent, weights, field) / scaled_radius
        return torch_weighted_center(value - field, weights)

    residual = torch_weighted_center(residual, weights)
    initial_norm = torch_weighted_norm(residual, weights)
    raw_modes = [residual]
    if mode_count > 1:
        raw_modes.append(scaled_apply(residual))
    for _ in range(2, mode_count):
        raw_modes.append(2.0 * scaled_apply(raw_modes[-1]) - raw_modes[-2])

    modes = []
    lifts = []
    identity = torch.eye(
        coherent.shape[1], dtype=coherent.dtype, device=coherent.device
    )
    for raw_mode in raw_modes:
        mode = torch_weighted_center(raw_mode, weights)
        norm = torch_weighted_norm(mode, weights)
        if bool(norm <= 1.0e-14 * initial_norm):
            mode = torch.zeros_like(mode)
        else:
            mode = mode / norm
        weighted_field = weights * mode
        moment = (coherent.T * weighted_field[None, :]) @ torch.conj(coherent)
        lifted = coherent.shape[1] * moment - torch.sum(weighted_field) * identity
        lifted = 0.5 * (lifted + torch.conj(lifted.T))
        modes.append(mode)
        lifts.append(lifted)

    return torch.stack(modes), torch.stack(lifts)


def torch_log_eta(values, derivatives, h_matrix, normalization: float, omega_log):
    import torch

    h_values = torch.einsum("ab,nb->na", h_matrix, values)
    denominator = torch.real(
        torch.einsum("na,na->n", torch.conj(values), h_values)
    )
    if not bool(torch.all(denominator > 0)):
        raise FloatingPointError("non-positive H section norm")
    h_derivatives = torch.einsum("ab,nbj->naj", h_matrix, derivatives)
    first = torch.einsum(
        "nmi,nmj->nij", torch.conj(derivatives), h_derivatives
    )
    gradient = torch.einsum("nm,nmj->nj", torch.conj(values), h_derivatives)
    metric = first / denominator[:, None, None]
    metric -= (
        torch.conj(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metric = normalization * 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
    eigenvalues = torch.linalg.eigvalsh(metric)
    if not bool(torch.all(eigenvalues > 0)):
        raise FloatingPointError("candidate metric is not positive on audit points")
    log_eta = torch.sum(torch.log(eigenvalues), dim=1) - omega_log
    return log_eta.detach().cpu().numpy(), float(torch.min(eigenvalues).detach().cpu())


def error_row(log_eta: np.ndarray, weights: np.ndarray, minimum_eigenvalue: float) -> dict:
    row = standard_errors(log_eta, weights)
    row["minimum_metric_eigenvalue"] = minimum_eigenvalue
    return row


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the accelerated audit") from exc

    args = parse_args()
    mode_counts = parse_integers(args.mode_counts, label="mode-counts")
    residual_sources = parse_residual_sources(args.residual_sources)
    start_iterations = (
        parse_integers(
            args.start_iterations,
            label="start-iterations",
            allow_zero=True,
        )
        if args.start_iterations is not None
        else None
    )
    if args.points <= 0 or args.points % 4 or args.workers <= 0:
        raise SystemExit("points must be a positive multiple of four")
    if args.max_pairs <= 0 or args.power_iterations <= 0:
        raise SystemExit("max-pairs and power-iterations must be positive")
    if args.spectral_safety_factor < 1.0:
        raise SystemExit("spectral-safety-factor must be at least one")

    started = time.perf_counter()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    real_dtype = torch.float32 if args.precision == "complex64" else torch.float64
    complex_dtype = torch.complex64 if args.precision == "complex64" else torch.complex128

    artifact_path = args.artifact.expanduser().resolve()
    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    model = adapter.make_model(args.model_seed, exact=True)
    artifact = adapter.load_h_artifact(artifact_path, model)
    sampling_backend = "thread" if args.workers == 1 else "process"

    checkpoints = {
        int(path.stem.split("_")[-1]): path
        for path in trajectory_dir.glob("iteration_*.npz")
    }
    pairs = select_adjacent_pairs(
        checkpoints,
        starts=start_iterations,
        maximum=args.max_pairs,
    )
    print(f"selected trajectory pairs: {pairs}", flush=True)

    first_iteration, first_h, metadata = load_checkpoint(checkpoints[pairs[0][0]])
    if first_iteration != pairs[0][0]:
        raise ValueError("checkpoint filename and stored iteration disagree")
    if metadata["model_seed"] != args.model_seed or tuple(metadata["degree"]) != artifact.degree:
        raise ValueError("trajectory metadata does not match the metric artifact")
    if first_h.shape != artifact.h_matrix.shape:
        raise ValueError("trajectory and artifact section dimensions disagree")

    print(f"sampling {args.points} registered audit points", flush=True)
    points, point_shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.points,
        seed=args.seed,
        workers=args.workers,
        cluster_size=4,
        backend=sampling_backend,
    )
    evaluated = [
        adapter.section_values_and_jacobian(point, artifact.section_exponents)
        for point in points
    ]
    values = np.asarray([row[0] for row in evaluated], dtype=np.complex128)
    derivatives = np.asarray([row[1] for row in evaluated], dtype=np.complex128)
    weights = np.asarray(adapter.importance_weights(points), dtype=np.float64)
    weights /= np.sum(weights)
    omega_log = np.asarray(
        [adapter.holomorphic_volume_log_density(point) for point in points],
        dtype=np.float64,
    )

    values_t = torch.tensor(values, dtype=complex_dtype, device=device)
    derivatives_t = torch.tensor(derivatives, dtype=complex_dtype, device=device)
    weights_t = torch.tensor(weights, dtype=real_dtype, device=device)
    omega_log_t = torch.tensor(omega_log, dtype=real_dtype, device=device)

    pair_rows = []
    for pair_index, (current_iteration, next_iteration) in enumerate(pairs, start=1):
        print(
            f"pair {pair_index}/{len(pairs)}: {current_iteration}->{next_iteration}",
            flush=True,
        )
        loaded_current, current_h, current_metadata = load_checkpoint(
            checkpoints[current_iteration]
        )
        loaded_next, next_h, next_metadata = load_checkpoint(checkpoints[next_iteration])
        if loaded_current != current_iteration or loaded_next != next_iteration:
            raise ValueError("checkpoint filename and iteration mismatch")
        if current_metadata != metadata or next_metadata != metadata:
            raise ValueError("trajectory checkpoint metadata changed within one run")

        current_h_t = torch.tensor(current_h, dtype=complex_dtype, device=device)
        next_h_t = torch.tensor(next_h, dtype=complex_dtype, device=device)
        current_log_eta, current_minimum = torch_log_eta(
            values_t,
            derivatives_t,
            current_h_t,
            float(artifact.normalization),
            omega_log_t,
        )
        next_log_eta, next_minimum = torch_log_eta(
            values_t,
            derivatives_t,
            next_h_t,
            float(artifact.normalization),
            omega_log_t,
        )
        coherent_t = torch_coherent_sections(values_t, current_h_t)
        spectral_radius_t, balance_residual_t, balance_defect_t = (
            torch_berezin_geometry(
                coherent_t,
                weights_t,
                power_iterations=args.power_iterations,
            )
        )
        target_tangent = affine_log_tangent(current_h, next_h)
        target_tangent_t = torch.tensor(
            target_tangent,
            dtype=complex_dtype,
            device=device,
        )
        target_residual_t = torch.real(
            torch.einsum(
                "na,ab,nb->n",
                torch.conj(coherent_t),
                target_tangent_t,
                coherent_t,
            )
        )
        _, current_log_ratio = normalized_volume_ratios(current_log_eta, weights)
        ma_residual = current_log_ratio - float(np.sum(weights * current_log_ratio))
        source_residuals = {
            "ma": torch.tensor(ma_residual, dtype=real_dtype, device=device),
            "balance": balance_residual_t,
            "target": target_residual_t,
        }

        source_rows = {}
        for source in residual_sources:
            print(f"  residual source: {source}", flush=True)
            modes_t, lifts_t = torch_berezin_modes_and_lifts(
                coherent_t,
                weights_t,
                source_residuals[source],
                mode_count=mode_counts[-1],
                spectral_radius=spectral_radius_t,
                spectral_safety_factor=args.spectral_safety_factor,
            )
            lifts = lifts_t.detach().cpu().numpy().astype(np.complex128)
            fit_curve = []
            final_fit = None
            for mode_count in mode_counts:
                fit = fit_hermitian_span(target_tangent, lifts[:mode_count])
                fit_curve.append(
                    {
                        "mode_count": mode_count,
                        "relative_frobenius_error": fit.relative_frobenius_error,
                        "explained_squared_fraction": fit.explained_squared_fraction,
                        "gram_condition_number": fit.gram_condition_number,
                        "numerical_rank": fit.numerical_rank,
                        "coefficients": fit.coefficients.tolist(),
                    }
                )
                final_fit = fit
            assert final_fit is not None
            candidate_h = affine_exponential_update(current_h, final_fit.reconstruction)
            candidate_h_t = torch.tensor(candidate_h, dtype=complex_dtype, device=device)
            candidate_log_eta, candidate_minimum = torch_log_eta(
                values_t,
                derivatives_t,
                candidate_h_t,
                float(artifact.normalization),
                omega_log_t,
            )
            source_rows[source] = {
                "fit_curve": fit_curve,
                "oracle_spectral_candidate_ma_errors": error_row(
                    candidate_log_eta,
                    weights,
                    candidate_minimum,
                ),
            }
            del candidate_h_t, modes_t, lifts_t

        pair_rows.append(
            {
                "current_iteration": current_iteration,
                "next_iteration": next_iteration,
                "target_tangent_frobenius_norm": float(
                    np.linalg.norm(target_tangent, ord="fro")
                ),
                "berezin_spectral_radius": float(spectral_radius_t.detach().cpu()),
                "balance_defect_frobenius_per_sqrt_n": float(
                    balance_defect_t.detach().cpu()
                ),
                "ma_errors": {
                    "current": error_row(current_log_eta, weights, current_minimum),
                    "true_t_map_next": error_row(next_log_eta, weights, next_minimum),
                },
                "residual_source_audits": source_rows,
            }
        )
        del current_h_t, next_h_t, coherent_t, target_tangent_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema": "type11-toeplitz-krylov-intrinsic-rank-audit-v1",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "trajectory_directory": str(trajectory_dir),
        "trajectory_metadata": metadata,
        "model_seed": args.model_seed,
        "degree": list(artifact.degree),
        "section_count": artifact.section_count,
        "seed": args.seed,
        "point_count": args.points,
        "point_shards": point_shards,
        "importance_effective_sample_size": float(1.0 / np.sum(weights**2)),
        "device": str(device),
        "precision": args.precision,
        "mode_counts": mode_counts,
        "residual_sources": residual_sources,
        "power_iterations": args.power_iterations,
        "spectral_safety_factor": args.spectral_safety_factor,
        "pairs": pair_rows,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation_gate": (
            "A neural controller is attempted only if an O(k) prefix recovers "
            "most affine-tangent energy and preserves the true T-map MA/tail change."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
