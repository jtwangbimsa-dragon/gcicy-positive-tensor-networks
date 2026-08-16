#!/usr/bin/env python3
"""Materialize a three-site X11 TN as a complete degree-six Hermitian form."""

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

from gcicy_metric.pipeline import (  # noqa: E402
    get_adapter,
    load_common_point_pool,
    positive_tensor_network_from_artifact_payload,
)
from gcicy_metric.pipeline.common_point_pool import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-basis-artifact", type=Path, required=True)
    parser.add_argument("--common-pool", type=Path, required=True)
    parser.add_argument("--adapter", default="p4p1_type11_hirzebruch_x3")
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--fit-points", type=int, default=2048)
    parser.add_argument("--holdout-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=86411)
    parser.add_argument("--rcond", type=float, default=1.0e-12)
    parser.add_argument("--product-chunk-size", type=int, default=512)
    parser.add_argument("--target-chunk-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--artifact-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def evaluate_values(adapter, points, exponents, *, batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(points), batch_size):
        values, _ = adapter.section_values_and_jacobian_batch(
            points[start : start + batch_size], exponents
        )
        rows.append(np.asarray(values, dtype=np.complex128))
    return np.concatenate(rows, axis=0)


def apply_matrix_on_axis(tensor, matrix, axis: int):
    order = [axis, *(index for index in range(tensor.ndim) if index != axis)]
    inverse = np.argsort(order).tolist()
    moved = tensor.permute(order).contiguous()
    shape = moved.shape
    transformed = matrix @ moved.reshape(shape[0], -1)
    return transformed.reshape(shape).permute(inverse).contiguous()


def centered_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    difference = np.asarray(candidate) - np.asarray(reference)
    difference -= np.mean(difference)
    return float(np.sqrt(np.mean(np.square(np.abs(difference)))))


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if min(
        args.fit_points,
        args.holdout_points,
        args.product_chunk_size,
        args.target_chunk_size,
        args.evaluation_batch_size,
    ) <= 0:
        raise ValueError("point counts and chunk sizes must be positive")

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model_path = args.model.expanduser().resolve()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "type11-positive-tensor-network-v1":
        raise ValueError("unrecognized X11 tensor-network artifact")
    if int(payload["site_count"]) != 3:
        raise ValueError("materialization requires a three-site model")
    source_path = Path(payload["source_artifact"]).expanduser().resolve()
    target_path = args.target_basis_artifact.expanduser().resolve()
    target_payload = load_npz(target_path)

    adapter = get_adapter(args.adapter)
    geometry = adapter.make_model(args.model_seed, exact=True)
    source = adapter.load_h_artifact(source_path, geometry)
    target_exponents = np.asarray(
        target_payload["global_section_exponents"], dtype=np.int64
    )
    if len(source.h_matrix) != 45 or len(target_exponents) != 871:
        raise ValueError("unexpected X11 source or target section count")
    model = positive_tensor_network_from_artifact_payload(
        source.h_matrix, payload, device=device
    ).to(dtype=torch.complex128)
    model.eval()
    cores = tuple(core.detach() for core in model.materialized_cores())
    if tuple(core.shape for core in cores) != (
        (1, 5, 45, 45),
        (5, 5, 45, 45),
        (5, 1, 45, 45),
    ):
        raise ValueError(f"unexpected materialized core shapes: {[c.shape for c in cores]}")

    pool = load_common_point_pool(
        args.common_pool,
        adapter,
        geometry,
        expected_model_seed=args.model_seed,
        expected_exact_model=True,
        expected_split="train",
    )
    sample_count = args.fit_points + args.holdout_points
    if sample_count > len(pool.points):
        raise ValueError("the common point pool is too small")
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(pool.points), size=sample_count, replace=False)
    points = [pool.points[int(index)] for index in selected]
    source_values = evaluate_values(
        adapter, points, source.section_exponents, batch_size=args.evaluation_batch_size
    )
    target_values = evaluate_values(
        adapter, points, target_exponents, batch_size=args.evaluation_batch_size
    )
    row_scale = np.maximum(
        np.linalg.norm(target_values, axis=1), np.finfo(np.float64).tiny
    )
    target_scaled = target_values / row_scale[:, None]
    source_scaled = source_values / row_scale[:, None] ** (1.0 / 3.0)
    target_fit = target_scaled[: args.fit_points]
    source_fit = source_scaled[: args.fit_points]
    target_pseudoinverse = np.linalg.pinv(target_fit, rcond=args.rcond)

    symmetric_indices = np.asarray(
        list(combinations_with_replacement(range(45), 3)), dtype=np.int64
    )
    coefficients = np.empty((871, len(symmetric_indices)), dtype=np.complex128)
    pseudoinverse_t = torch.as_tensor(
        target_pseudoinverse, dtype=torch.complex128, device=device
    )
    for start in range(0, len(symmetric_indices), args.product_chunk_size):
        stop = min(start + args.product_chunk_size, len(symmetric_indices))
        products_fit = np.prod(source_fit[:, symmetric_indices[start:stop]], axis=2)
        coefficients[:, start:stop] = (
            pseudoinverse_t
            @ torch.as_tensor(products_fit, dtype=torch.complex128, device=device)
        ).cpu().numpy()
        print(f"multiplication_coefficients={stop}/{len(symmetric_indices)}", flush=True)

    lookup = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(symmetric_indices)
    }
    ordered_to_symmetric = np.fromiter(
        (
            lookup[tuple(sorted(indices))]
            for indices in product(range(45), repeat=3)
        ),
        dtype=np.int64,
        count=45**3,
    )
    ordered_coefficients = coefficients[:, ordered_to_symmetric].T
    del coefficients
    ordered_t = torch.as_tensor(
        ordered_coefficients, dtype=torch.complex128, device=device
    )

    output_map = torch.empty((45**3, 871), dtype=torch.complex128, device=device)
    first = cores[0][0]
    second = cores[1]
    third = cores[2][:, 0]
    for start in range(0, 871, args.target_chunk_size):
        stop = min(start + args.target_chunk_size, 871)
        lifted = ordered_t[:, start:stop].reshape(45, 45, 45, stop - start)
        left = torch.einsum("api,ijkt->apjkt", first, lifted)
        middle = torch.einsum("abqj,apjkt->bpqkt", second, left)
        output = torch.einsum("brk,bpqkt->pqrt", third, middle)
        output_map[:, start:stop] = output.reshape(45**3, stop - start)
        print(f"tn_target_columns={stop}/871", flush=True)
    learned_h = torch.conj(output_map.T) @ output_map
    del output_map

    reference_h = torch.as_tensor(
        np.asarray(source.h_matrix, dtype=np.complex128),
        dtype=torch.complex128,
        device=device,
    )
    transformed = ordered_t.reshape(45, 45, 45, 871)
    for axis in range(3):
        transformed = apply_matrix_on_axis(transformed, reference_h, axis)
    reference_lift_h = torch.conj(ordered_t.T) @ transformed.reshape(45**3, 871)
    del transformed, ordered_t
    target_h_t = learned_h + float(model.positive_floor) * reference_lift_h
    target_h = target_h_t.cpu().numpy()
    target_h = 0.5 * (target_h + target_h.conj().T)
    target_h *= len(target_h) / np.trace(target_h).real
    eigenvalues = np.linalg.eigvalsh(target_h)
    if eigenvalues[0] <= 0:
        raise FloatingPointError("materialized TN Hermitian form is not positive")

    artifact_out = args.artifact_out.expanduser().resolve()
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        **adapter.artifact_model_payload(geometry, exact=True),
        "pipeline_schema_version": np.asarray(1, dtype=np.int64),
        "pipeline_adapter": np.asarray(adapter.key),
        "importance_weighted_training": np.asarray(True),
        "global_section_degree": np.asarray((6, 6), dtype=np.int64),
        "global_section_exponents": target_exponents,
        "global_h_matrix": target_h,
        "global_h_positive_relative_floor": np.asarray(
            eigenvalues[0] / eigenvalues[-1], dtype=np.float64
        ),
        "global_section_normalization": np.asarray(1.0 / 6.0),
        "basis_selected_indices": np.arange(871, dtype=np.int64),
        "basis_relation_error": np.asarray(0.0, dtype=np.float64),
        "h_parameterization": np.asarray("materialized_three_site_positive_tn"),
        "materialized_tn_sha256": np.asarray(file_sha256(model_path)),
    }
    temporary = artifact_out.with_suffix(artifact_out.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **artifact_payload)
    temporary.replace(artifact_out)

    holdout_points = points[args.fit_points :]
    holdout_source_values, holdout_source_derivatives = (
        adapter.section_values_and_jacobian_batch(
            holdout_points, source.section_exponents
        )
    )
    with torch.no_grad():
        tn_metric = model(
            torch.as_tensor(
                holdout_source_values, dtype=torch.complex128, device=device
            ),
            torch.as_tensor(
                holdout_source_derivatives, dtype=torch.complex128, device=device
            ),
        ).cpu().numpy()
    loaded = adapter.load_h_artifact(artifact_out, geometry)
    full_h_metric = adapter.h_metrics(holdout_points, loaded)
    metric_relative_rms = float(
        np.linalg.norm(full_h_metric - tn_metric) / np.linalg.norm(tn_metric)
    )
    tn_log_eta = adapter.residual_values(holdout_points, tn_metric)
    full_h_log_eta = adapter.residual_values(holdout_points, full_h_metric)

    report = {
        "schema": "type11-k6-tn-materialized-full-h-v1",
        "model": str(model_path),
        "model_sha256": file_sha256(model_path),
        "source_artifact": str(source_path),
        "target_basis_artifact": str(target_path),
        "artifact": str(artifact_out),
        "artifact_sha256": file_sha256(artifact_out),
        "section_count": 871,
        "complete_full_h_real_parameter_count": 871**2,
        "positive_floor": float(model.positive_floor),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "holdout_points": args.holdout_points,
        "holdout_metric_relative_rms_difference": metric_relative_rms,
        "holdout_centered_log_eta_rms_difference": centered_rms(
            full_h_log_eta, tn_log_eta
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    report_out = args.report_out.expanduser().resolve()
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
