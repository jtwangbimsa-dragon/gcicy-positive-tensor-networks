#!/usr/bin/env python3
"""Build an X11 degree-six full-H artifact representing the H2 metric exactly."""

from __future__ import annotations

import argparse
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, load_common_point_pool  # noqa: E402
from gcicy_metric.pipeline.common_point_pool import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--target-basis-artifact", type=Path, required=True)
    parser.add_argument("--common-pool", type=Path, required=True)
    parser.add_argument("--adapter", default="p4p1_type11_hirzebruch_x3")
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--fit-points", type=int, default=2048)
    parser.add_argument("--holdout-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=86401)
    parser.add_argument("--rcond", type=float, default=1.0e-12)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--product-chunk-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--artifact-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    return parser.parse_args()


def artifact_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def evaluate_values(
    adapter: object,
    points: list[object],
    exponents: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    rows = []
    for start in range(0, len(points), batch_size):
        values, _ = adapter.section_values_and_jacobian_batch(  # type: ignore[attr-defined]
            points[start : start + batch_size],
            exponents,
        )
        rows.append(np.asarray(values, dtype=np.complex128))
    return np.concatenate(rows, axis=0)


def product_values(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.prod(values[:, indices], axis=2)


def apply_kronecker_factor(tensor, matrix, axis: int):
    import torch

    order = [axis, *(index for index in range(tensor.ndim) if index != axis)]
    inverse = np.argsort(order).tolist()
    moved = tensor.permute(order).contiguous()
    shape = moved.shape
    transformed = matrix @ moved.reshape(shape[0], -1)
    return transformed.reshape(shape).permute(inverse).contiguous()


def relative_centered_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    difference = np.asarray(candidate) - np.asarray(reference)
    difference -= np.mean(difference)
    return float(np.sqrt(np.mean(np.square(np.abs(difference)))))


def main() -> None:
    args = parse_args()
    if args.fit_points <= 0 or args.holdout_points <= 0:
        raise SystemExit("fit and holdout point counts must be positive")
    if args.rcond <= 0 or args.product_chunk_size <= 0:
        raise SystemExit("rcond and product chunk size must be positive")

    started = time.perf_counter()
    source_path = args.source_artifact.expanduser().resolve()
    target_path = args.target_basis_artifact.expanduser().resolve()
    source_payload = artifact_arrays(source_path)
    target_payload = artifact_arrays(target_path)
    source_degree = tuple(
        int(value) for value in source_payload["global_section_degree"]
    )
    target_degree = tuple(
        int(value) for value in target_payload["global_section_degree"]
    )
    if target_degree != tuple(3 * value for value in source_degree):
        raise ValueError("the target basis is not the cubic source degree")
    source_exponents = np.asarray(
        source_payload["global_section_exponents"], dtype=np.int64
    )
    target_exponents = np.asarray(
        target_payload["global_section_exponents"], dtype=np.int64
    )
    source_h = np.asarray(source_payload["global_h_matrix"], dtype=np.complex128)
    source_h = 0.5 * (source_h + source_h.conj().T)
    source_h *= len(source_h) / np.trace(source_h).real

    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=True)
    pool = load_common_point_pool(
        args.common_pool,
        adapter,
        model,
        expected_model_seed=args.model_seed,
        expected_exact_model=True,
        expected_split="train",
    )
    sample_count = args.fit_points + args.holdout_points
    if sample_count > len(pool.points):
        raise ValueError("the common point pool is too small")
    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(len(pool.points), size=sample_count, replace=False)
    selected_points = [pool.points[int(index)] for index in selected_indices]
    source_values = evaluate_values(
        adapter,
        selected_points,
        source_exponents,
        batch_size=args.evaluation_batch_size,
    )
    target_values = evaluate_values(
        adapter,
        selected_points,
        target_exponents,
        batch_size=args.evaluation_batch_size,
    )
    row_scale = np.maximum(
        np.linalg.norm(target_values, axis=1), np.finfo(np.float64).tiny
    )
    target_values = target_values / row_scale[:, None]
    source_values = source_values / row_scale[:, None] ** (1.0 / 3.0)
    source_fit = source_values[: args.fit_points]
    source_holdout = source_values[args.fit_points :]
    target_fit = target_values[: args.fit_points]
    target_holdout = target_values[args.fit_points :]

    product_indices = np.asarray(
        list(combinations_with_replacement(range(len(source_exponents)), 3)),
        dtype=np.int64,
    )
    print(
        f"source_sections={len(source_exponents)} "
        f"symmetric_products={len(product_indices)} "
        f"target_sections={len(target_exponents)}",
        flush=True,
    )
    target_pseudoinverse = np.linalg.pinv(target_fit, rcond=args.rcond)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    pseudoinverse_t = torch.as_tensor(
        target_pseudoinverse, dtype=torch.complex128, device=device
    )
    coefficients = np.empty(
        (len(target_exponents), len(product_indices)), dtype=np.complex128
    )
    fit_numerator = 0.0
    fit_denominator = 0.0
    holdout_numerator = 0.0
    holdout_denominator = 0.0
    for start in range(0, len(product_indices), args.product_chunk_size):
        stop = min(start + args.product_chunk_size, len(product_indices))
        index_chunk = product_indices[start:stop]
        products_fit = product_values(source_fit, index_chunk)
        products_holdout = product_values(source_holdout, index_chunk)
        coefficient_chunk = (
            pseudoinverse_t
            @ torch.as_tensor(products_fit, dtype=torch.complex128, device=device)
        ).cpu().numpy()
        coefficients[:, start:stop] = coefficient_chunk
        fit_difference = target_fit @ coefficient_chunk - products_fit
        holdout_difference = target_holdout @ coefficient_chunk - products_holdout
        fit_numerator += float(np.linalg.norm(fit_difference) ** 2)
        fit_denominator += float(np.linalg.norm(products_fit) ** 2)
        holdout_numerator += float(np.linalg.norm(holdout_difference) ** 2)
        holdout_denominator += float(np.linalg.norm(products_holdout) ** 2)
        print(f"coefficient_products={stop}/{len(product_indices)}", flush=True)
    fit_error = float(np.sqrt(fit_numerator / fit_denominator))
    holdout_error = float(np.sqrt(holdout_numerator / holdout_denominator))

    symmetric_lookup = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(product_indices)
    }
    ordered_to_symmetric = np.fromiter(
        (
            symmetric_lookup[tuple(sorted(indices))]
            for indices in product(range(len(source_exponents)), repeat=3)
        ),
        dtype=np.int64,
        count=len(source_exponents) ** 3,
    )
    ordered_coefficients = coefficients[:, ordered_to_symmetric].T
    del coefficients
    ordered_t = torch.as_tensor(
        ordered_coefficients, dtype=torch.complex128, device=device
    )
    source_h_t = torch.as_tensor(source_h, dtype=torch.complex128, device=device)
    transformed = ordered_t.reshape(
        len(source_exponents),
        len(source_exponents),
        len(source_exponents),
        len(target_exponents),
    )
    for axis in range(3):
        transformed = apply_kronecker_factor(transformed, source_h_t, axis)
        print(f"applied_source_h_axis={axis}", flush=True)
    target_h = (
        torch.conj(ordered_t.T)
        @ transformed.reshape(len(source_exponents) ** 3, len(target_exponents))
    ).cpu().numpy()
    del ordered_t, transformed
    target_h = 0.5 * (target_h + target_h.conj().T)
    target_h *= len(target_h) / np.trace(target_h).real
    eigenvalues = np.linalg.eigvalsh(target_h)
    if eigenvalues[0] <= 0:
        raise FloatingPointError(
            f"the cubic lift is not positive definite: minimum={eigenvalues[0]:.3e}"
        )

    artifact_output = args.artifact_out.expanduser().resolve()
    artifact_output.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        **adapter.artifact_model_payload(model, exact=True),
        "pipeline_schema_version": np.asarray(1, dtype=np.int64),
        "pipeline_adapter": np.asarray(adapter.key),
        "importance_weighted_training": np.asarray(True),
        "global_section_degree": np.asarray(target_degree, dtype=np.int64),
        "global_section_exponents": target_exponents,
        "global_h_matrix": target_h,
        "global_h_positive_relative_floor": np.asarray(
            eigenvalues[0] / eigenvalues[-1], dtype=np.float64
        ),
        "global_section_normalization": np.asarray(1.0 / 6.0),
        "basis_selected_indices": np.arange(len(target_exponents), dtype=np.int64),
        "basis_relation_error": np.asarray(holdout_error, dtype=np.float64),
        "h_parameterization": np.asarray("exact_cubic_lift_of_h2"),
        "cubic_lift_source_sha256": np.asarray(file_sha256(source_path)),
        "cubic_lift_target_basis_sha256": np.asarray(file_sha256(target_path)),
        "cubic_lift_fit_index_sha256": np.asarray(
            __import__("hashlib").sha256(selected_indices.tobytes()).hexdigest()
        ),
    }
    temporary = artifact_output.with_suffix(artifact_output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **artifact_payload)
    temporary.replace(artifact_output)

    loaded = adapter.load_h_artifact(artifact_output, model)
    holdout_points = selected_points[args.fit_points :]
    source_artifact = adapter.load_h_artifact(source_path, model)
    source_metrics = adapter.h_metrics(holdout_points, source_artifact)
    target_metrics = adapter.h_metrics(holdout_points, loaded)
    metric_relative_rms = float(
        np.linalg.norm(target_metrics - source_metrics)
        / np.linalg.norm(source_metrics)
    )
    source_log_eta = adapter.residual_values(holdout_points, source_metrics)
    target_log_eta = adapter.residual_values(holdout_points, target_metrics)
    centered_log_eta_rms = relative_centered_error(target_log_eta, source_log_eta)

    report = {
        "schema": "type11-h2-cubic-full-h-lift-v1",
        "source_artifact": str(source_path),
        "source_sha256": file_sha256(source_path),
        "target_basis_artifact": str(target_path),
        "target_basis_sha256": file_sha256(target_path),
        "artifact": str(artifact_output),
        "artifact_sha256": file_sha256(artifact_output),
        "source_degree": list(source_degree),
        "target_degree": list(target_degree),
        "source_section_count": int(len(source_exponents)),
        "symmetric_product_count": int(len(product_indices)),
        "ordered_product_count": int(len(source_exponents) ** 3),
        "target_section_count": int(len(target_exponents)),
        "fit_points": int(args.fit_points),
        "holdout_points": int(args.holdout_points),
        "fit_relative_reconstruction_error": fit_error,
        "holdout_relative_reconstruction_error": holdout_error,
        "target_h_minimum_eigenvalue": float(eigenvalues[0]),
        "target_h_maximum_eigenvalue": float(eigenvalues[-1]),
        "target_h_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "holdout_metric_relative_rms_difference": metric_relative_rms,
        "holdout_centered_log_eta_rms_difference": centered_log_eta_rms,
        "wall_seconds": time.perf_counter() - started,
    }
    report_output = args.report_out.expanduser().resolve()
    report_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_output.with_suffix(report_output.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n")
    temporary_report.replace(report_output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
