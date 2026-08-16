#!/usr/bin/env python3
"""Chunked GPU blind-tail audit of a saved gCICY H metric."""

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

from gcicy_metric.pipeline import get_adapter, sampling_cluster_statistics  # noqa: E402
from gcicy_metric.pipeline.audit import (  # noqa: E402
    normalized_volume_ratios,
    standard_errors,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    dense_h_potential_and_metric,
    section_values_and_jacobian_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sampling-cluster-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument("--top-points", type=int, default=64)
    parser.add_argument("--arrays-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the GPU H-metric audit") from exc

    args = parse_args()
    if (
        min(
            args.points,
            args.workers,
            args.sampling_cluster_size,
            args.chunk_size,
            args.top_points,
        )
        <= 0
        or args.points % args.sampling_cluster_size
    ):
        raise SystemExit(
            "points must be a positive multiple of sampling-cluster-size"
        )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    complex_dtype = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }[args.precision]

    started = time.perf_counter()
    adapter = get_adapter(args.adapter)
    geometry = adapter.make_model(args.model_seed, exact=True)
    artifact_path = args.artifact.expanduser().resolve()
    artifact = adapter.load_h_artifact(artifact_path, geometry)
    points, shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.points,
        seed=args.seed,
        workers=args.workers,
        cluster_size=args.sampling_cluster_size,
        backend="thread" if args.workers == 1 else "process",
    )
    weights = np.asarray(adapter.importance_weights(points), dtype=np.float64)
    weights /= np.sum(weights)
    cluster_ids = np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
    unique_clusters, cluster_counts = np.unique(cluster_ids, return_counts=True)
    if cluster_ids.shape != (args.points,):
        raise ValueError("sampling cluster ids do not align with points")
    if np.any(cluster_counts != args.sampling_cluster_size):
        raise ValueError("blind sample does not contain complete fibres")

    h_matrix = torch.tensor(artifact.h_matrix, dtype=complex_dtype, device=device)
    log_eta_chunks = []
    minimum_metric_eigenvalue = float("inf")
    nonpositive_metric_count = 0
    chunk_count = (args.points + args.chunk_size - 1) // args.chunk_size
    progress_interval = max(1, chunk_count // 10)
    for chunk_index, start in enumerate(
        range(0, args.points, args.chunk_size), start=1
    ):
        stop = min(start + args.chunk_size, args.points)
        values_numpy, derivatives_numpy = section_values_and_jacobian_arrays(
            adapter,
            points[start:stop],
            artifact.section_exponents,
        )
        values = torch.tensor(values_numpy, dtype=complex_dtype, device=device)
        derivatives = torch.tensor(
            derivatives_numpy, dtype=complex_dtype, device=device
        )
        with torch.no_grad():
            _, metric = dense_h_potential_and_metric(
                values,
                derivatives,
                h_matrix,
                float(artifact.normalization),
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            minimum_metric_eigenvalue = min(
                minimum_metric_eigenvalue,
                float(torch.min(eigenvalues).detach().cpu()),
            )
            nonpositive_metric_count += int(
                torch.count_nonzero(torch.min(eigenvalues, dim=1).values <= 0)
                .detach()
                .cpu()
            )
            if bool(torch.any(eigenvalues <= 0)):
                raise FloatingPointError("H metric is nonpositive on the blind pool")
            log_omega = torch.tensor(
                [
                    adapter.holomorphic_volume_log_density(point)
                    for point in points[start:stop]
                ],
                dtype=eigenvalues.dtype,
                device=device,
            )
            log_eta_chunks.append(
                (
                    torch.sum(torch.log(eigenvalues), dim=1) - log_omega
                ).detach().cpu().numpy()
            )
        del values, derivatives, metric, eigenvalues, log_omega
        if chunk_index == chunk_count or chunk_index % progress_interval == 0:
            print(
                f"evaluated metric chunks {chunk_index}/{chunk_count}",
                flush=True,
            )

    log_eta = np.concatenate(log_eta_chunks)
    errors = standard_errors(log_eta, weights)
    normalized_ratio, log_normalized_ratio = normalized_volume_ratios(
        log_eta, weights
    )
    metric_volume_weights = weights * normalized_ratio
    metric_volume_weights /= np.sum(metric_volume_weights)
    omega_cluster_statistics = sampling_cluster_statistics(weights, cluster_ids)
    metric_cluster_statistics = sampling_cluster_statistics(
        metric_volume_weights, cluster_ids
    )
    worst_order = np.argsort(normalized_ratio)[::-1]
    top_count = min(args.top_points, args.points)
    top_rows = [
        {
            "point_index": int(index),
            "cluster_id": int(cluster_ids[index]),
            "normalized_ratio": float(normalized_ratio[index]),
            "log_normalized_ratio": float(log_normalized_ratio[index]),
            "importance_weight": float(weights[index]),
            "metric_volume_weight": float(metric_volume_weights[index]),
            "projective_chart": [
                int(value) for value in adapter.point_projective_chart(points[index])
            ],
            "minimum_jacobian_singular_value": float(
                adapter.point_jacobian_min_singular_value(points[index])
            ),
        }
        for index in worst_order[:top_count]
    ]

    arrays_metadata = None
    if args.arrays_out is not None:
        arrays_path = args.arrays_out.expanduser().resolve()
        try:
            point_payload = {
                key: np.asarray(value)
                for key, value in adapter.point_storage_payload(points).items()
            }
            point_payload_available = True
        except NotImplementedError:
            point_payload = {}
            point_payload_available = False
        save_npz_atomic(
            arrays_path,
            schema=np.asarray("h-metric-blind-tail-arrays-v1"),
            adapter=np.asarray(adapter.key),
            model_seed=np.asarray(args.model_seed),
            seed=np.asarray(args.seed),
            log_eta=log_eta,
            normalized_ratio=normalized_ratio,
            importance_weights=weights,
            metric_volume_weights=metric_volume_weights,
            sampling_cluster_ids=cluster_ids,
            **point_payload,
        )
        arrays_metadata = {
            "path": str(arrays_path),
            "sha256": sha256_file(arrays_path),
            "point_storage_payload_available": point_payload_available,
        }

    device_metadata = {"kind": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_metadata.update(
            {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    report = {
        "schema": "h-metric-blind-tail-audit-v1",
        "adapter": adapter.key,
        "model_seed": args.model_seed,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "degree": list(artifact.degree),
        "section_count": artifact.section_count,
        "normalization": artifact.normalization,
        "seed": args.seed,
        "points": args.points,
        "sampling_cluster_size": args.sampling_cluster_size,
        "sampling_cluster_count": int(len(unique_clusters)),
        "complete_sampling_clusters": True,
        "point_shards": shards,
        "precision": args.precision,
        "device": device_metadata,
        "minimum_metric_eigenvalue": minimum_metric_eigenvalue,
        "nonpositive_metric_count": nonpositive_metric_count,
        "errors": errors,
        "omega_cluster_statistics": omega_cluster_statistics,
        "metric_volume_cluster_statistics": metric_cluster_statistics,
        "top_positive_ratio_points": top_rows,
        "point_arrays": arrays_metadata,
        "runtime_seconds": time.perf_counter() - started,
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(json.dumps(errors, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
