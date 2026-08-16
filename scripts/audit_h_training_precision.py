#!/usr/bin/env python3
"""Compare complex64 training arithmetic with complex128 H-metric evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.active_set import load_active_point_pool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="p4p1_type11_hirzebruch_x3")
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--active-pool", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrays-out", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {key: float("nan") for key in ("min", "median", "p90", "p99", "max", "mean")}
    return {
        "min": float(np.min(data)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def hermitian_spectrum(matrix: np.ndarray) -> dict[str, float]:
    value = np.asarray(matrix, dtype=np.complex128)
    value = 0.5 * (value + value.conjugate().T)
    eigenvalues = np.linalg.eigvalsh(value)
    return {
        "minimum": float(eigenvalues[0]),
        "maximum": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "nonpositive_count": int(np.sum(eigenvalues <= 0)),
    }


def numpy_forward(
    values: np.ndarray,
    derivatives: np.ndarray,
    log_omega: np.ndarray,
    h_matrix: np.ndarray,
    normalization: float,
) -> dict[str, np.ndarray]:
    h_values = np.einsum("ab,nb->na", h_matrix, values, optimize=True)
    denominator = np.real(np.einsum("na,na->n", np.conjugate(values), h_values))
    h_derivatives = np.einsum("ab,nbj->naj", h_matrix, derivatives, optimize=True)
    first = np.einsum(
        "nmi,nmj->nij", np.conjugate(derivatives), h_derivatives, optimize=True
    )
    gradient = np.einsum(
        "nm,nmj->nj", np.conjugate(values), h_derivatives, optimize=True
    )
    first_term = first / denominator[:, None, None]
    projector_term = (
        np.conjugate(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metric = normalization * (first_term - projector_term)
    metric = 0.5 * (metric + np.conjugate(np.swapaxes(metric, 1, 2)))
    eigenvalues = np.linalg.eigvalsh(metric)
    raw = np.full(len(values), np.nan, dtype=np.float64)
    positive = eigenvalues[:, 0] > 0
    raw[positive] = np.sum(np.log(eigenvalues[positive]), axis=1) - log_omega[positive]
    cancellation = (
        np.linalg.norm(first_term, axis=(1, 2))
        + np.linalg.norm(projector_term, axis=(1, 2))
    ) / np.maximum(np.linalg.norm(first_term - projector_term, axis=(1, 2)), 1e-300)
    return {
        "raw": raw,
        "minimum_eigenvalue": eigenvalues[:, 0],
        "denominator": denominator,
        "cancellation_amplification": cancellation,
    }


def torch_forward(
    values: np.ndarray,
    derivatives: np.ndarray,
    log_omega: np.ndarray,
    h_matrix: torch.Tensor,
    normalization: float,
) -> dict[str, np.ndarray]:
    device = h_matrix.device
    complex_dtype = h_matrix.dtype
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
    values_t = torch.as_tensor(values, dtype=complex_dtype, device=device)
    derivatives_t = torch.as_tensor(derivatives, dtype=complex_dtype, device=device)
    log_omega_t = torch.as_tensor(log_omega, dtype=real_dtype, device=device)
    h_values = torch.einsum("ab,nb->na", h_matrix, values_t)
    denominator = torch.real(
        torch.einsum("na,na->n", torch.conj(values_t), h_values)
    )
    h_derivatives = torch.einsum("ab,nbj->naj", h_matrix, derivatives_t)
    first = torch.einsum(
        "nmi,nmj->nij", torch.conj(derivatives_t), h_derivatives
    )
    gradient = torch.einsum("nm,nmj->nj", torch.conj(values_t), h_derivatives)
    first_term = first / denominator[:, None, None]
    projector_term = (
        torch.conj(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metric = normalization * (first_term - projector_term)
    metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
    eigenvalues = torch.linalg.eigvalsh(metric)
    raw = torch.sum(torch.log(torch.clamp(eigenvalues, min=1e-12)), dim=1) - log_omega_t
    cancellation = (
        torch.linalg.matrix_norm(first_term)
        + torch.linalg.matrix_norm(projector_term)
    ) / torch.clamp(torch.linalg.matrix_norm(first_term - projector_term), min=1e-30)
    return {
        "raw": raw.detach().cpu().to(torch.float64).numpy(),
        "minimum_eigenvalue": eigenvalues[:, 0].detach().cpu().to(torch.float64).numpy(),
        "denominator": denominator.detach().cpu().to(torch.float64).numpy(),
        "cancellation_amplification": cancellation.detach().cpu().to(torch.float64).numpy(),
    }


def mode_summary(raw: np.ndarray, minimum: np.ndarray, reference_log_normalization: float) -> dict[str, Any]:
    ratio = np.exp(np.clip(raw - reference_log_normalization, -745.0, 709.0))
    valid = np.isfinite(raw) & np.isfinite(minimum) & (minimum > 0)
    return {
        "raw_log_ratio": distribution(raw),
        "minimum_metric_eigenvalue": distribution(minimum),
        "nonpositive_or_nonfinite_count": int(np.sum(~valid)),
        "fixed_normalization_ratio": {
            "maximum": float(np.nanmax(ratio)),
            "above_3_count": int(np.sum(ratio > 3.0)),
            "above_10_count": int(np.sum(ratio > 10.0)),
        },
    }


def comparison_summary(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    raw_error = candidate["raw"] - reference["raw"]
    minimum_error = candidate["minimum_eigenvalue"] - reference["minimum_eigenvalue"]
    denominator_relative_error = np.abs(
        candidate["denominator"] - reference["denominator"]
    ) / np.maximum(np.abs(reference["denominator"]), 1e-300)
    return {
        "absolute_raw_log_ratio_error": distribution(np.abs(raw_error)),
        "signed_raw_log_ratio_error": distribution(raw_error),
        "absolute_minimum_metric_eigenvalue_error": distribution(np.abs(minimum_error)),
        "relative_section_denominator_error": distribution(denominator_relative_error),
        "raw_error_correlation_with_log10_cancellation": correlation(
            np.abs(raw_error),
            np.log10(np.maximum(reference["cancellation_amplification"], 1.0)),
        ),
        "worst_absolute_raw_error_index": int(np.nanargmax(np.abs(raw_error))),
    }


def concatenate(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([row[key] for row in rows])
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    artifact_path = args.artifact.expanduser().resolve()
    pool_path = args.active_pool.expanduser().resolve()
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=True)
    artifact = adapter.load_h_artifact(artifact_path, model)
    pool = load_active_point_pool(
        pool_path,
        adapter,
        model,
        expected_model_seed=args.model_seed,
        expected_exact_model=True,
    )
    reference_log_normalization = float(pool.metadata["global_log_normalization"])
    normalization = float(artifact.normalization)
    source_h = np.asarray(artifact.h_matrix, dtype=np.complex128)
    source_h = 0.5 * (source_h + source_h.conjugate().T)
    source_h *= len(source_h) / float(np.trace(source_h).real)
    source_cholesky = np.linalg.cholesky(source_h)

    device = torch.device(args.device)
    cholesky32 = torch.as_tensor(source_cholesky, dtype=torch.complex64, device=device)
    training_h32 = cholesky32 @ torch.conj(cholesky32.T)
    training_h32 *= len(source_h) / torch.real(torch.trace(training_h32))
    source_h128_t = torch.as_tensor(source_h, dtype=torch.complex128, device=device)
    training_h32_numpy = np.asarray(
        training_h32.detach().cpu().to(torch.complex128).numpy(), dtype=np.complex128
    )

    modes: dict[str, list[dict[str, np.ndarray]]] = {
        "numpy_complex128_source_h": [],
        "numpy_complex128_training_h32": [],
        "torch_complex128_source_h": [],
        "torch_complex64_training_h32": [],
    }
    started = time.perf_counter()
    for start in range(0, len(pool.points), args.batch_size):
        stop = min(start + args.batch_size, len(pool.points))
        points = pool.points[start:stop]
        values = []
        derivatives = []
        for point in points:
            section, jacobian = adapter.section_values_and_jacobian(
                point, artifact.section_exponents
            )
            values.append(section)
            derivatives.append(jacobian)
        values_array = np.asarray(values, dtype=np.complex128)
        derivatives_array = np.asarray(derivatives, dtype=np.complex128)
        log_omega = np.asarray(
            [adapter.holomorphic_volume_log_density(point) for point in points],
            dtype=np.float64,
        )
        modes["numpy_complex128_source_h"].append(
            numpy_forward(values_array, derivatives_array, log_omega, source_h, normalization)
        )
        modes["numpy_complex128_training_h32"].append(
            numpy_forward(
                values_array,
                derivatives_array,
                log_omega,
                training_h32_numpy,
                normalization,
            )
        )
        modes["torch_complex128_source_h"].append(
            torch_forward(
                values_array,
                derivatives_array,
                log_omega,
                source_h128_t,
                normalization,
            )
        )
        modes["torch_complex64_training_h32"].append(
            torch_forward(
                values_array,
                derivatives_array,
                log_omega,
                training_h32,
                normalization,
            )
        )
        print(f"prepared and evaluated {stop}/{len(pool.points)} points", flush=True)

    combined = {name: concatenate(rows) for name, rows in modes.items()}
    reference = combined["numpy_complex128_source_h"]
    comparisons = {
        "torch128_formula_vs_numpy128": comparison_summary(
            combined["torch_complex128_source_h"], reference
        ),
        "h32_reconstruction_vs_source_h_in_numpy128": comparison_summary(
            combined["numpy_complex128_training_h32"], reference
        ),
        "torch64_arithmetic_vs_same_h32_in_numpy128": comparison_summary(
            combined["torch_complex64_training_h32"],
            combined["numpy_complex128_training_h32"],
        ),
        "total_training64_vs_reference128": comparison_summary(
            combined["torch_complex64_training_h32"], reference
        ),
    }
    worst_tail = int(np.nanargmax(reference["raw"]))
    worst_precision = comparisons["total_training64_vs_reference128"][
        "worst_absolute_raw_error_index"
    ]

    arrays_path = (
        args.arrays_out.expanduser().resolve()
        if args.arrays_out is not None
        else args.out.expanduser().resolve().with_suffix(".npz")
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "center_ids": np.asarray(pool.center_ids, dtype=np.int64),
        "radii": np.asarray(pool.radii, dtype=np.float64),
        "source_center_log_ratios": np.asarray(
            pool.source_center_log_ratios, dtype=np.float64
        ),
    }
    for mode_name, values in combined.items():
        for key, array in values.items():
            arrays[f"{mode_name}_{key}"] = array
    np.savez_compressed(arrays_path, **arrays)

    report = {
        "schema_version": 1,
        "description": "Fixed-point precision replay of the H-metric training forward pass.",
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "active_pool": str(pool_path),
        "active_pool_sha256": sha256_file(pool_path),
        "point_count": len(pool.points),
        "device": str(device),
        "batch_size": args.batch_size,
        "normalization": normalization,
        "reference_log_normalization": reference_log_normalization,
        "h_matrices": {
            "source_complex128": hermitian_spectrum(source_h),
            "training_cholesky_complex64_reconstructed": hermitian_spectrum(
                training_h32_numpy
            ),
            "relative_frobenius_change": float(
                np.linalg.norm(training_h32_numpy - source_h) / np.linalg.norm(source_h)
            ),
        },
        "modes": {
            name: {
                **mode_summary(
                    values["raw"],
                    values["minimum_eigenvalue"],
                    reference_log_normalization,
                ),
                "section_denominator": distribution(values["denominator"]),
                "cancellation_amplification": distribution(
                    values["cancellation_amplification"]
                ),
            }
            for name, values in combined.items()
        },
        "comparisons": comparisons,
        "selected_points": {
            "largest_reference_tail": {
                "index": worst_tail,
                "center_id": int(pool.center_ids[worst_tail]),
                "radius": float(pool.radii[worst_tail]),
                "reference_raw_log_ratio": float(reference["raw"][worst_tail]),
                "training64_raw_log_ratio": float(
                    combined["torch_complex64_training_h32"]["raw"][worst_tail]
                ),
                "absolute_precision_error": float(
                    abs(
                        combined["torch_complex64_training_h32"]["raw"][worst_tail]
                        - reference["raw"][worst_tail]
                    )
                ),
            },
            "largest_precision_disagreement": {
                "index": worst_precision,
                "center_id": int(pool.center_ids[worst_precision]),
                "radius": float(pool.radii[worst_precision]),
                "reference_raw_log_ratio": float(reference["raw"][worst_precision]),
                "training64_raw_log_ratio": float(
                    combined["torch_complex64_training_h32"]["raw"][worst_precision]
                ),
                "absolute_precision_error": float(
                    abs(
                        combined["torch_complex64_training_h32"]["raw"][worst_precision]
                        - reference["raw"][worst_precision]
                    )
                ),
            },
        },
        "arrays": str(arrays_path),
        "arrays_sha256": sha256_file(arrays_path),
        "runtime_seconds": time.perf_counter() - started,
        "claim_limit": "This isolates arithmetic on a fixed empirical active pool; it is not a global metric certificate.",
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
