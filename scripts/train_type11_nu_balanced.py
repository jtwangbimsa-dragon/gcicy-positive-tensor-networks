#!/usr/bin/env python3
"""Compute a fixed-sample nu-balanced gCICY H metric by T-iteration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eigvalsh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="p4p1_type11_hirzebruch_x3",
        help="registered pipeline adapter key",
    )
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument(
        "--degree",
        type=int,
        default=4,
        help="positive tensor power of the adapter's Kahler line bundle",
    )
    parser.add_argument("--basis-points", type=int, default=4096)
    parser.add_argument("--basis-seed", type=int, default=70499)
    parser.add_argument("--train-points", type=int, default=8192)
    parser.add_argument("--train-seed", type=int, default=72001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--sampling-cluster-size",
        type=int,
        default=4,
        help="number of roots retained from each complete sampling fibre",
    )
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument("--eigenvalue-floor", type=float, default=1.0e-9)
    parser.add_argument(
        "--whiten-reference-basis",
        action="store_true",
        help="run T-iteration after congruence-whitening the initial FS H",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        help="optional directory for atomic per-iteration H checkpoints",
    )
    parser.add_argument(
        "--trajectory-every",
        type=int,
        default=1,
        help="checkpoint every this many T-map iterations",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="continue from a trajectory iteration_XXXX.npz checkpoint",
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


def save_npz_atomic(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def polarization_degree(adapter, power: int) -> tuple[int, ...]:
    """Return the multidegree of a positive tensor power of the polarization."""

    if power <= 0:
        raise ValueError("polarization power must be positive")
    return tuple(
        power * int(value)
        for value in adapter.configuration.kahler_line_bundle
    )


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


def positive_hermitian_square_root(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    value = 0.5 * (value + value.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues[0] <= 0 or not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("reference H must be positive definite")
    square_root = (
        eigenvectors * np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.conjugate().T
    return 0.5 * (square_root + square_root.conjugate().T)


def transform_section_values(values: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply t=R s to row-major section values."""

    sections = np.asarray(values, dtype=np.complex128)
    basis_transform = np.asarray(transform, dtype=np.complex128)
    if sections.ndim != 2 or basis_transform.shape != (
        sections.shape[1],
        sections.shape[1],
    ):
        raise ValueError("section values and basis transform are not aligned")
    return np.einsum("ab,nb->na", basis_transform, sections, optimize=True)


def h_matrix_to_original_basis(
    matrix: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Convert an H acting on t=R s back to the original s basis."""

    value = np.asarray(matrix, dtype=np.complex128)
    basis_transform = np.asarray(transform, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("H must be square")
    if basis_transform.shape != value.shape:
        raise ValueError("H and basis transform are not aligned")
    converted = basis_transform.conjugate().T @ value @ basis_transform
    return 0.5 * (converted + converted.conjugate().T)


def save_trajectory_checkpoint(
    directory: Path,
    *,
    iteration: int,
    h_matrix: np.ndarray,
    args: argparse.Namespace,
    step: dict[str, float | int] | None,
    degree: tuple[int, ...] | None = None,
) -> dict[str, object]:
    selected_degree = degree or (int(args.degree), int(args.degree))
    adapter_key = getattr(args, "adapter", "p4p1_type11_hirzebruch_x3")
    path = directory / f"iteration_{iteration:04d}.npz"
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray("nu-balanced-trajectory-v2"),
        "adapter": np.asarray(adapter_key),
        "iteration": np.asarray(iteration, dtype=np.int64),
        "h_matrix": np.asarray(h_matrix, dtype=np.complex128),
        "model_seed": np.asarray(args.model_seed, dtype=np.int64),
        "degree": np.asarray(selected_degree, dtype=np.int64),
        "basis_seed": np.asarray(args.basis_seed, dtype=np.int64),
        "basis_points": np.asarray(args.basis_points, dtype=np.int64),
        "train_seed": np.asarray(args.train_seed, dtype=np.int64),
        "train_points": np.asarray(args.train_points, dtype=np.int64),
        "precision": np.asarray(args.precision),
        "reference_basis_whitened": np.asarray(
            bool(getattr(args, "whiten_reference_basis", False))
        ),
    }
    if step is not None:
        for key, value in step.items():
            payload[f"step_{key}"] = np.asarray(value)
    save_npz_atomic(path, **payload)
    return {
        "iteration": iteration,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def load_resume_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    section_count: int,
    degree: tuple[int, ...] | None = None,
) -> tuple[int, np.ndarray]:
    selected_degree = degree or (int(args.degree), int(args.degree))
    adapter_key = getattr(args, "adapter", "p4p1_type11_hirzebruch_x3")
    checkpoint = path.expanduser().resolve()
    with np.load(checkpoint, allow_pickle=False) as payload:
        required = {
            "schema",
            "iteration",
            "h_matrix",
            "model_seed",
            "degree",
            "basis_seed",
            "basis_points",
            "train_seed",
            "train_points",
            "precision",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"resume checkpoint lacks fields: {missing}")
        expected = {
            "adapter": adapter_key,
            "model_seed": args.model_seed,
            "basis_seed": args.basis_seed,
            "basis_points": args.basis_points,
            "train_seed": args.train_seed,
            "train_points": args.train_points,
            "precision": args.precision,
            "reference_basis_whitened": bool(
                getattr(args, "whiten_reference_basis", False)
            ),
        }
        observed = {
            "adapter": (
                str(payload["adapter"])
                if "adapter" in payload.files
                else "p4p1_type11_hirzebruch_x3"
            ),
            "model_seed": int(payload["model_seed"]),
            "basis_seed": int(payload["basis_seed"]),
            "basis_points": int(payload["basis_points"]),
            "train_seed": int(payload["train_seed"]),
            "train_points": int(payload["train_points"]),
            "precision": str(payload["precision"]),
            "reference_basis_whitened": bool(
                payload["reference_basis_whitened"]
            )
            if "reference_basis_whitened" in payload.files
            else False,
        }
        if observed != expected:
            raise ValueError(
                f"resume checkpoint metadata mismatch: expected {expected}, got {observed}"
            )
        checkpoint_degree = tuple(
            int(value) for value in np.asarray(payload["degree"]).tolist()
        )
        if checkpoint_degree != selected_degree:
            raise ValueError("resume checkpoint degree does not match the request")
        h_matrix = np.asarray(payload["h_matrix"], dtype=np.complex128)
        if h_matrix.shape != (section_count, section_count):
            raise ValueError("resume checkpoint H has the wrong section dimension")
        iteration = int(payload["iteration"])
    if iteration < 0:
        raise ValueError("resume checkpoint iteration cannot be negative")
    return iteration, normalize_h(h_matrix)


def relative_step_statistics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    generalized = eigvalsh(candidate, reference, check_finite=False)
    if generalized[0] <= 0:
        raise FloatingPointError("T-map step is not positive relative to its input")
    centered = np.log(generalized)
    centered -= np.mean(centered)
    return {
        "maximum_absolute_centered_log_eigenvalue": float(np.max(np.abs(centered))),
        "centered_log_eigenvalue_span": float(np.ptp(centered)),
    }


def nu_balanced_t_step_torch(values, weights, h_matrix, *, eigenvalue_floor: float):
    """Apply one inverse-H convention T-map update to torch tensors."""

    import torch

    h_values = torch.einsum("ab,nb->na", h_matrix, values)
    denominator = torch.real(
        torch.einsum("na,na->n", torch.conj(values), h_values)
    )
    if not bool(torch.all(denominator > 0)):
        raise FloatingPointError("non-positive section norm during T-iteration")
    coefficients = weights / denominator
    hilbert = (values.T * coefficients[None, :]) @ torch.conj(values)
    hilbert = 0.5 * (hilbert + torch.conj(hilbert.T))
    hilbert_eigenvalues, hilbert_eigenvectors = torch.linalg.eigh(hilbert)
    maximum = torch.max(hilbert_eigenvalues)
    floor = eigenvalue_floor * maximum
    clipped = torch.clamp(hilbert_eigenvalues, min=floor)
    next_h = (hilbert_eigenvectors * (1.0 / clipped)[None, :]) @ torch.conj(
        hilbert_eigenvectors.T
    )
    next_h = 0.5 * (next_h + torch.conj(next_h.T))
    next_h *= len(h_matrix) / torch.real(torch.trace(next_h))
    return next_h, hilbert_eigenvalues, floor


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for accelerated T-iteration") from exc

    args = parse_args()
    if args.degree <= 0 or args.iterations <= 0 or args.tolerance <= 0:
        raise SystemExit("degree, iterations, and tolerance must be positive")
    if (
        args.basis_points <= 0
        or args.train_points <= 0
        or args.sampling_cluster_size <= 0
        or args.basis_points % args.sampling_cluster_size
        or args.train_points % args.sampling_cluster_size
        or args.workers <= 0
    ):
        raise SystemExit(
            "point counts must be positive multiples of sampling-cluster-size"
        )
    if not 0 < args.eigenvalue_floor < 1:
        raise SystemExit("eigenvalue floor must lie in (0, 1)")
    if args.trajectory_every <= 0:
        raise SystemExit("trajectory-every must be positive")

    started = time.perf_counter()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    real_dtype = torch.float32 if args.precision == "complex64" else torch.float64
    complex_dtype = torch.complex64 if args.precision == "complex64" else torch.complex128

    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=True)
    degree = polarization_degree(adapter, args.degree)
    normalization = 1.0 / adapter.configuration.kahler_power(degree)
    sampling_backend = "thread" if args.workers == 1 else "process"

    print("sampling basis points and constructing the restricted section space", flush=True)
    basis_points, basis_shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.basis_points,
        seed=args.basis_seed,
        workers=args.workers,
        cluster_size=args.sampling_cluster_size,
        backend=sampling_backend,
    )
    basis = adapter.restricted_section_basis(basis_points, degree)
    reference_h, relation_error = adapter.restricted_fubini_study_h_matrix(
        basis_points, basis
    )
    reference_h = normalize_h(reference_h)
    if args.whiten_reference_basis:
        section_basis_transform = positive_hermitian_square_root(reference_h)
        internal_reference_h = np.eye(len(reference_h), dtype=np.complex128)
    else:
        section_basis_transform = np.eye(len(reference_h), dtype=np.complex128)
        internal_reference_h = reference_h
    exponents = np.asarray(basis.selected_exponents, dtype=np.int64)
    print(
        f"basis rank={basis.numerical_rank}, relation_error={relation_error:.3e}",
        flush=True,
    )

    print("sampling the fixed nu-balanced integration set", flush=True)
    train_points, train_shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.train_points,
        seed=args.train_seed,
        workers=args.workers,
        cluster_size=args.sampling_cluster_size,
        backend=sampling_backend,
    )
    values = np.asarray(
        [adapter.section_values_and_jacobian(point, exponents)[0] for point in train_points],
        dtype=np.complex128,
    )
    if args.whiten_reference_basis:
        values = transform_section_values(values, section_basis_transform)
    values = values.astype(
        np.complex64 if args.precision == "complex64" else np.complex128,
        copy=False,
    )
    weights = np.asarray(adapter.importance_weights(train_points), dtype=np.float64)
    weights /= np.sum(weights)
    effective_sample_size = float(1.0 / np.sum(weights**2))

    values_t = torch.tensor(values, dtype=complex_dtype, device=device)
    weights_t = torch.tensor(weights, dtype=real_dtype, device=device)
    section_count = len(reference_h)
    starting_iteration = 0
    starting_h = internal_reference_h
    if args.resume_checkpoint is not None:
        starting_iteration, starting_h = load_resume_checkpoint(
            args.resume_checkpoint,
            args=args,
            degree=degree,
            section_count=section_count,
        )
        print(
            f"resuming from iteration {starting_iteration}: "
            f"{args.resume_checkpoint.expanduser().resolve()}",
            flush=True,
        )
    h_t = torch.tensor(starting_h, dtype=complex_dtype, device=device)
    history: list[dict[str, float | int]] = []
    trajectory_checkpoints: list[dict[str, object]] = []
    trajectory_dir = (
        args.trajectory_dir.expanduser().resolve()
        if args.trajectory_dir is not None
        else None
    )
    if trajectory_dir is not None:
        trajectory_checkpoints.append(
            save_trajectory_checkpoint(
                trajectory_dir,
                iteration=starting_iteration,
                h_matrix=starting_h,
                degree=degree,
                args=args,
                step=None,
            )
        )
    converged = False

    final_iteration = starting_iteration
    for offset in range(1, args.iterations + 1):
        iteration = starting_iteration + offset
        next_h, hilbert_eigenvalues, floor = nu_balanced_t_step_torch(
            values_t,
            weights_t,
            h_t,
            eigenvalue_floor=args.eigenvalue_floor,
        )
        maximum = torch.max(hilbert_eigenvalues)

        current_np = normalize_h(h_t.detach().cpu().numpy())
        next_np = normalize_h(next_h.detach().cpu().numpy())
        step = relative_step_statistics(current_np, next_np)
        row = {
            "iteration": iteration,
            **step,
            "hilbert_minimum_relative_eigenvalue": float(
                (hilbert_eigenvalues[0] / maximum).detach().cpu()
            ),
            "hilbert_clipped_eigenvalue_count": int(
                torch.count_nonzero(hilbert_eigenvalues < floor).detach().cpu()
            ),
        }
        history.append(row)
        final_iteration = iteration
        print(
            f"iteration {iteration}: step_radius="
            f"{step['maximum_absolute_centered_log_eigenvalue']:.6e}, "
            f"step_span={step['centered_log_eigenvalue_span']:.6e}, "
            f"hilbert_rel_min={row['hilbert_minimum_relative_eigenvalue']:.3e}",
            flush=True,
        )
        h_t = torch.tensor(next_np, dtype=complex_dtype, device=device)
        should_save = (
            trajectory_dir is not None
            and (
                iteration % args.trajectory_every == 0
                or offset == args.iterations
                or step["maximum_absolute_centered_log_eigenvalue"] <= args.tolerance
            )
        )
        if should_save and trajectory_dir is not None:
            trajectory_checkpoints.append(
                save_trajectory_checkpoint(
                    trajectory_dir,
                    iteration=iteration,
                    h_matrix=next_np,
                    degree=degree,
                    args=args,
                    step=row,
                )
            )
        if step["maximum_absolute_centered_log_eigenvalue"] <= args.tolerance:
            converged = True
            break

    final_internal_h = normalize_h(h_t.detach().cpu().numpy())
    final_h = normalize_h(
        h_matrix_to_original_basis(final_internal_h, section_basis_transform)
    )
    final_relative = eigvalsh(final_h, reference_h, check_finite=False)
    final_relative /= np.exp(np.mean(np.log(final_relative)))
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
        "global_h_matrix": final_h,
        "global_h_positive_relative_floor": np.asarray(1.0e-12),
        "global_section_normalization": np.asarray(normalization),
        "basis_selected_indices": np.asarray(basis.selected_indices, dtype=np.int64),
        "basis_relation_error": np.asarray(relation_error),
        "h_parameterization": np.asarray("nu_balanced_t_iteration"),
        "nu_balanced_train_seed": np.asarray(args.train_seed),
        "nu_balanced_train_points": np.asarray(args.train_points),
        "nu_balanced_starting_iteration": np.asarray(starting_iteration),
        "nu_balanced_iterations": np.asarray(final_iteration),
        "nu_balanced_converged": np.asarray(converged),
        "nu_balanced_tolerance": np.asarray(args.tolerance),
        "nu_balanced_eigenvalue_floor": np.asarray(args.eigenvalue_floor),
        "nu_balanced_effective_sample_size": np.asarray(effective_sample_size),
        "nu_balanced_reference_basis_whitened": np.asarray(
            args.whiten_reference_basis
        ),
    }
    save_npz_atomic(artifact_path, **payload)
    summary = {
        "schema": "nu-balanced-t-iteration-v2",
        "adapter": adapter.key,
        "artifact": str(artifact_path),
        "model_seed": args.model_seed,
        "degree": list(degree),
        "normalization": normalization,
        "device": str(device),
        "precision": args.precision,
        "reference_basis_whitened": args.whiten_reference_basis,
        "basis_points": args.basis_points,
        "basis_seed": args.basis_seed,
        "basis_shards": basis_shards,
        "basis_rank": basis.numerical_rank,
        "basis_relation_error": relation_error,
        "train_points": args.train_points,
        "train_seed": args.train_seed,
        "train_shards": train_shards,
        "sampling_cluster_size": args.sampling_cluster_size,
        "importance_effective_sample_size": effective_sample_size,
        "requested_iterations": args.iterations,
        "starting_iteration": starting_iteration,
        "requested_additional_iterations": args.iterations,
        "final_iteration": final_iteration,
        "completed_iterations": len(history),
        "tolerance": args.tolerance,
        "converged": converged,
        "final_relative_to_fs_spectrum": {
            "minimum": float(final_relative[0]),
            "maximum": float(final_relative[-1]),
            "condition_number": float(final_relative[-1] / final_relative[0]),
            "log_eigenvalue_span": float(np.ptp(np.log(final_relative))),
        },
        "final_internal_h_condition_number": float(
            np.linalg.cond(final_internal_h)
        ),
        "history": history,
        "trajectory": {
            "directory": str(trajectory_dir) if trajectory_dir is not None else None,
            "checkpoint_every": args.trajectory_every,
            "resume_checkpoint": (
                str(args.resume_checkpoint.expanduser().resolve())
                if args.resume_checkpoint is not None
                else None
            ),
            "checkpoints_written_this_run": trajectory_checkpoints,
        },
        "runtime_seconds": time.perf_counter() - started,
        "method_note": (
            "For H equal to the inverse section metric, each update forms the "
            "nu-weighted Hilbert matrix integral and then inverts it. Optional "
            "reference-basis whitening is an exactly congruent section-basis "
            "change; the released artifact is converted back to the original basis."
        ),
    }
    write_text_atomic(summary_path, json.dumps(summary, indent=2) + "\n")
    summary["artifact_sha256"] = sha256_file(artifact_path)
    write_text_atomic(summary_path, json.dumps(summary, indent=2) + "\n")
    if trajectory_dir is not None:
        manifest = {
            "schema": "nu-balanced-trajectory-manifest-v2",
            "adapter": adapter.key,
            "artifact": str(artifact_path),
            "artifact_sha256": summary["artifact_sha256"],
            "summary": str(summary_path),
            "model_seed": args.model_seed,
            "degree": list(degree),
            "basis_seed": args.basis_seed,
            "basis_points": args.basis_points,
            "train_seed": args.train_seed,
            "train_points": args.train_points,
            "precision": args.precision,
            "reference_basis_whitened": args.whiten_reference_basis,
            "starting_iteration": starting_iteration,
            "final_iteration": final_iteration,
            "converged": converged,
            "checkpoints_written_this_run": trajectory_checkpoints,
        }
        write_text_atomic(
            trajectory_dir / "manifest.json",
            json.dumps(manifest, indent=2) + "\n",
        )
    print(json.dumps(summary["final_relative_to_fs_spectrum"], indent=2), flush=True)
    print(f"wrote {artifact_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
