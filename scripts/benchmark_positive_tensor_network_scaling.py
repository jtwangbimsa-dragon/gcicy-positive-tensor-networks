#!/usr/bin/env python3
"""Benchmark native fixed-bond metric evaluation across polarization degrees."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import PositiveTensorNetworkMetric, get_adapter  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    section_values_and_jacobian_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--degrees", type=int, nargs="+", required=True)
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--seed", type=int, default=72971)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--timed-repeats", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import torch

    args = parse_args()
    if len(args.models) != len(args.degrees):
        raise SystemExit("models and degrees must have equal lengths")
    if args.points <= 0 or args.points % 4:
        raise SystemExit("points must be a positive multiple of four")
    if args.warmup_repeats < 0 or args.timed_repeats <= 0:
        raise SystemExit("repeat counts are invalid")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    geometry = adapter.make_model(args.model_seed, exact=True)
    points, shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.points,
        seed=args.seed,
        workers=args.workers,
        cluster_size=4,
        backend="thread" if args.workers == 1 else "process",
    )
    results = []
    for degree, raw_model_path in zip(args.degrees, args.models, strict=True):
        model_path = raw_model_path.expanduser().resolve()
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "type11-positive-tensor-network-v1":
            raise ValueError(f"unrecognized model artifact {model_path}")
        source_path = Path(payload["source_artifact"]).expanduser().resolve()
        if sha256_file(source_path) != payload["source_artifact_sha256"]:
            raise ValueError(f"source artifact hash mismatch for degree {degree}")
        source = adapter.load_h_artifact(source_path, geometry)
        section_started = time.perf_counter()
        values_numpy, derivatives_numpy = section_values_and_jacobian_arrays(
            adapter, points, source.section_exponents
        )
        section_seconds = time.perf_counter() - section_started
        precision = str(payload["precision"])
        complex_dtype = (
            torch.complex64 if precision == "complex64" else torch.complex128
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        values = torch.tensor(values_numpy, dtype=complex_dtype, device=device)
        derivatives = torch.tensor(
            derivatives_numpy, dtype=complex_dtype, device=device
        )
        model = PositiveTensorNetworkMetric(
            source.h_matrix,
            site_count=int(payload["site_count"]),
            bond_dimension=int(payload["bond_dimension"]),
            target_normalization=float(payload["target_normalization"]),
            positive_floor=float(payload["positive_floor"]),
            initialization_noise=0.0,
            dtype=complex_dtype,
            device=device,
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()

        warmup_potential = None
        warmup_metric = None
        with torch.no_grad():
            for _ in range(args.warmup_repeats):
                warmup_potential, warmup_metric = model.potential_and_metric(
                    values, derivatives
                )
        del warmup_potential, warmup_metric
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
        else:
            baseline_allocated = None
            baseline_reserved = None

        timings = []
        minimum_metric_eigenvalue = float("inf")
        with torch.no_grad():
            for _ in range(args.timed_repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                potential, metric = model.potential_and_metric(values, derivatives)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - started)
            minimum_metric_eigenvalue = float(
                torch.min(torch.linalg.eigvalsh(metric)).detach().cpu()
            )
        peak_memory = None
        if device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            peak_memory = {
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": peak_allocated
                - baseline_allocated,
                "incremental_peak_reserved_bytes": peak_reserved - baseline_reserved,
            }
        median_seconds = statistics.median(timings)
        results.append(
            {
                "degree": degree,
                "model": str(model_path),
                "model_sha256": sha256_file(model_path),
                "site_count": model.site_count,
                "bond_dimension": model.bond_dimension,
                "trainable_real_parameter_count": model.trainable_real_parameter_count,
                "parameter_storage_bytes": int(
                    sum(value.numel() * value.element_size() for value in model.parameters())
                ),
                "source_section_count": source.section_count,
                "source_section_evaluation_seconds": section_seconds,
                "timed_repeats": args.timed_repeats,
                "forward_seconds": timings,
                "median_forward_seconds": median_seconds,
                "mean_forward_seconds": statistics.fmean(timings),
                "forward_microseconds_per_point": 1.0e6
                * median_seconds
                / args.points,
                "points_per_second": args.points / median_seconds,
                "minimum_metric_eigenvalue": minimum_metric_eigenvalue,
                "device_memory": peak_memory,
                "target_degree_dense_h_materialized": False,
            }
        )
        del potential, metric, model, values, derivatives
        if device.type == "cuda":
            torch.cuda.empty_cache()

    device_metadata = {"kind": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_metadata.update(
            {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        )
    report = {
        "schema": "positive-tensor-network-fixed-bond-scaling-benchmark-v1",
        "model_seed": args.model_seed,
        "seed": args.seed,
        "points": args.points,
        "point_shards": shards,
        "device": device_metadata,
        "results": results,
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
