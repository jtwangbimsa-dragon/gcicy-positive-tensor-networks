#!/usr/bin/env python3
"""Two-pool oracle audit of Berezin modes against saved type-(1,1) T-map steps."""

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
from gcicy_metric.pipeline.audit import normalized_volume_ratios  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from gcicy_metric.pipeline.toeplitz_spectral import (  # noqa: E402
    affine_exponential_update,
    affine_log_tangent,
    fit_hermitian_span,
)
from scripts.audit_type11_toeplitz_krylov_rank import (  # noqa: E402
    error_row,
    load_checkpoint,
    parse_integers,
    parse_residual_sources,
    select_adjacent_pairs,
    torch_berezin_geometry,
    torch_berezin_modes_and_lifts,
    torch_coherent_sections,
    torch_log_eta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--mode-seed", type=int, default=72001)
    parser.add_argument("--mode-points", type=int, default=32768)
    parser.add_argument("--evaluation-seed", type=int, default=72111)
    parser.add_argument("--evaluation-points", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--mode-counts", default="2,4,6,8,10,12")
    parser.add_argument(
        "--residual-sources",
        default="ma,balance,target",
        help="comma-separated subset of ma,balance,target",
    )
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--start-iterations")
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


def prepare_pool(
    adapter,
    artifact,
    *,
    model_seed: int,
    count: int,
    seed: int,
    workers: int,
    device,
    real_dtype,
    complex_dtype,
) -> dict:
    import torch

    backend = "thread" if workers == 1 else "process"
    points, shards = sample_points_parallel(
        adapter,
        model_seed=model_seed,
        exact_model=True,
        count=count,
        seed=seed,
        workers=workers,
        cluster_size=4,
        backend=backend,
    )
    evaluated = [
        adapter.section_values_and_jacobian(point, artifact.section_exponents)
        for point in points
    ]
    values = torch.tensor(
        np.asarray([row[0] for row in evaluated]),
        dtype=complex_dtype,
        device=device,
    )
    derivatives = torch.tensor(
        np.asarray([row[1] for row in evaluated]),
        dtype=complex_dtype,
        device=device,
    )
    weights_numpy = np.asarray(adapter.importance_weights(points), dtype=np.float64)
    weights_numpy /= np.sum(weights_numpy)
    omega_log_numpy = np.asarray(
        [adapter.holomorphic_volume_log_density(point) for point in points],
        dtype=np.float64,
    )
    return {
        "count": count,
        "seed": seed,
        "shards": shards,
        "values": values,
        "derivatives": derivatives,
        "weights": torch.tensor(weights_numpy, dtype=real_dtype, device=device),
        "weights_numpy": weights_numpy,
        "omega_log": torch.tensor(omega_log_numpy, dtype=real_dtype, device=device),
        "importance_effective_sample_size": float(1.0 / np.sum(weights_numpy**2)),
    }


def evaluate_h_on_pool(pool: dict, h_matrix, *, normalization: float):
    return torch_log_eta(
        pool["values"],
        pool["derivatives"],
        h_matrix,
        normalization,
        pool["omega_log"],
    )


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the accelerated two-pool audit") from exc

    args = parse_args()
    mode_counts = parse_integers(args.mode_counts, label="mode-counts")
    residual_sources = parse_residual_sources(args.residual_sources)
    starts = (
        parse_integers(args.start_iterations, label="start-iterations", allow_zero=True)
        if args.start_iterations is not None
        else None
    )
    counts = (args.mode_points, args.evaluation_points)
    if min(counts) <= 0 or any(count % 4 for count in counts) or args.workers <= 0:
        raise SystemExit("point counts must be positive multiples of four")
    if args.mode_seed == args.evaluation_seed:
        raise SystemExit("mode and evaluation seeds must be independent")
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
    geometry = adapter.make_model(args.model_seed, exact=True)
    artifact = adapter.load_h_artifact(artifact_path, geometry)
    checkpoints = {
        int(path.stem.split("_")[-1]): path
        for path in trajectory_dir.glob("iteration_*.npz")
    }
    pairs = select_adjacent_pairs(checkpoints, starts=starts, maximum=args.max_pairs)
    first_iteration, first_h, metadata = load_checkpoint(checkpoints[pairs[0][0]])
    if first_iteration != pairs[0][0] or first_h.shape != artifact.h_matrix.shape:
        raise ValueError("trajectory does not match the artifact section dimension")
    if metadata["model_seed"] != args.model_seed or tuple(metadata["degree"]) != artifact.degree:
        raise ValueError("trajectory metadata does not match the artifact")

    print(
        f"preparing mode pool seed={args.mode_seed} points={args.mode_points}",
        flush=True,
    )
    mode_pool = prepare_pool(
        adapter,
        artifact,
        model_seed=args.model_seed,
        count=args.mode_points,
        seed=args.mode_seed,
        workers=args.workers,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    print(
        f"preparing independent evaluation pool seed={args.evaluation_seed} "
        f"points={args.evaluation_points}",
        flush=True,
    )
    evaluation_pool = prepare_pool(
        adapter,
        artifact,
        model_seed=args.model_seed,
        count=args.evaluation_points,
        seed=args.evaluation_seed,
        workers=args.workers,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )

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
            raise ValueError("trajectory metadata changed within a run")
        current_h_t = torch.tensor(current_h, dtype=complex_dtype, device=device)
        next_h_t = torch.tensor(next_h, dtype=complex_dtype, device=device)

        mode_current_log_eta, mode_current_minimum = evaluate_h_on_pool(
            mode_pool,
            current_h_t,
            normalization=float(artifact.normalization),
        )
        mode_next_log_eta, mode_next_minimum = evaluate_h_on_pool(
            mode_pool,
            next_h_t,
            normalization=float(artifact.normalization),
        )
        evaluation_current_log_eta, evaluation_current_minimum = evaluate_h_on_pool(
            evaluation_pool,
            current_h_t,
            normalization=float(artifact.normalization),
        )
        evaluation_next_log_eta, evaluation_next_minimum = evaluate_h_on_pool(
            evaluation_pool,
            next_h_t,
            normalization=float(artifact.normalization),
        )

        coherent_t = torch_coherent_sections(mode_pool["values"], current_h_t)
        spectral_radius_t, balance_residual_t, balance_defect_t = torch_berezin_geometry(
            coherent_t,
            mode_pool["weights"],
            power_iterations=args.power_iterations,
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
        _, mode_current_log_ratio = normalized_volume_ratios(
            mode_current_log_eta,
            mode_pool["weights_numpy"],
        )
        ma_residual = mode_current_log_ratio - float(
            np.sum(mode_pool["weights_numpy"] * mode_current_log_ratio)
        )
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
                mode_pool["weights"],
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
            mode_candidate_log_eta, mode_candidate_minimum = evaluate_h_on_pool(
                mode_pool,
                candidate_h_t,
                normalization=float(artifact.normalization),
            )
            evaluation_candidate_log_eta, evaluation_candidate_minimum = evaluate_h_on_pool(
                evaluation_pool,
                candidate_h_t,
                normalization=float(artifact.normalization),
            )
            source_rows[source] = {
                "fit_curve": fit_curve,
                "mode_pool_candidate_ma_errors": error_row(
                    mode_candidate_log_eta,
                    mode_pool["weights_numpy"],
                    mode_candidate_minimum,
                ),
                "independent_candidate_ma_errors": error_row(
                    evaluation_candidate_log_eta,
                    evaluation_pool["weights_numpy"],
                    evaluation_candidate_minimum,
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
                "mode_pool_ma_errors": {
                    "current": error_row(
                        mode_current_log_eta,
                        mode_pool["weights_numpy"],
                        mode_current_minimum,
                    ),
                    "true_t_map_next": error_row(
                        mode_next_log_eta,
                        mode_pool["weights_numpy"],
                        mode_next_minimum,
                    ),
                },
                "independent_ma_errors": {
                    "current": error_row(
                        evaluation_current_log_eta,
                        evaluation_pool["weights_numpy"],
                        evaluation_current_minimum,
                    ),
                    "true_t_map_next": error_row(
                        evaluation_next_log_eta,
                        evaluation_pool["weights_numpy"],
                        evaluation_next_minimum,
                    ),
                },
                "residual_source_audits": source_rows,
            }
        )
        del current_h_t, next_h_t, coherent_t, target_tangent_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema": "type11-toeplitz-krylov-two-pool-rank-audit-v1",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "trajectory_directory": str(trajectory_dir),
        "trajectory_metadata": metadata,
        "model_seed": args.model_seed,
        "degree": list(artifact.degree),
        "section_count": artifact.section_count,
        "mode_pool": {
            "seed": args.mode_seed,
            "points": args.mode_points,
            "shards": mode_pool["shards"],
            "importance_effective_sample_size": mode_pool[
                "importance_effective_sample_size"
            ],
            "role": "same quadrature registration as the saved T-map trajectory",
        },
        "independent_evaluation_pool": {
            "seed": args.evaluation_seed,
            "points": args.evaluation_points,
            "shards": evaluation_pool["shards"],
            "importance_effective_sample_size": evaluation_pool[
                "importance_effective_sample_size"
            ],
            "role": "never used to build or fit spectral modes",
        },
        "device": str(device),
        "precision": args.precision,
        "mode_counts": mode_counts,
        "residual_sources": residual_sources,
        "power_iterations": args.power_iterations,
        "spectral_safety_factor": args.spectral_safety_factor,
        "pairs": pair_rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
