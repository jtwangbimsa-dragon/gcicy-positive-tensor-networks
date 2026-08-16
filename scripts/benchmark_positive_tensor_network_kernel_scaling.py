#!/usr/bin/env python3
"""Benchmark fixed-bond tensor-network metric kernels versus site count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import PositiveTensorNetworkMetric  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-counts", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--section-count", type=int, default=45)
    parser.add_argument("--bond-dimension", type=int, default=5)
    parser.add_argument(
        "--physical-dictionary-rank",
        type=int,
        help="use a fixed shared local dictionary with this rank",
    )
    parser.add_argument(
        "--train-physical-dictionary",
        action="store_true",
        help="count the shared dictionary as learned parameters",
    )
    parser.add_argument("--coordinate-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=72972)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--timed-repeats", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument(
        "--transfer-implementation",
        choices=("scalar", "vectorized"),
        default="vectorized",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    if any(site_count <= 0 for site_count in args.site_counts):
        raise SystemExit("site counts must be positive")
    if min(args.section_count, args.bond_dimension, args.coordinate_count) <= 0:
        raise SystemExit("tensor dimensions must be positive")
    if (
        args.physical_dictionary_rank is not None
        and not 0 < args.physical_dictionary_rank <= args.section_count**2
    ):
        raise SystemExit("physical dictionary rank must lie in [1, section_count^2]")
    if args.train_physical_dictionary and args.physical_dictionary_rank is None:
        raise SystemExit("a trainable dictionary requires a dictionary rank")
    if args.batch_size <= 0 or args.warmup_repeats < 0 or args.timed_repeats <= 0:
        raise SystemExit("batch size and repeat counts are invalid")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    complex_dtype = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }[args.precision]

    rng = np.random.default_rng(args.seed)
    scale = 1.0 / np.sqrt(2.0 * args.section_count)
    values_numpy = scale * (
        rng.normal(size=(args.batch_size, args.section_count))
        + 1j * rng.normal(size=(args.batch_size, args.section_count))
    )
    derivatives_numpy = scale * (
        rng.normal(
            size=(args.batch_size, args.section_count, args.coordinate_count)
        )
        + 1j
        * rng.normal(
            size=(args.batch_size, args.section_count, args.coordinate_count)
        )
    )
    values = torch.tensor(values_numpy, dtype=complex_dtype, device=device)
    derivatives = torch.tensor(
        derivatives_numpy, dtype=complex_dtype, device=device
    )
    reference_h = np.eye(args.section_count, dtype=np.complex128)
    physical_dictionary = None
    if args.physical_dictionary_rank is not None:
        raw_dictionary = rng.normal(
            size=(args.physical_dictionary_rank, args.section_count**2)
        ) + 1j * rng.normal(
            size=(args.physical_dictionary_rank, args.section_count**2)
        )
        orthonormal_columns, _ = np.linalg.qr(raw_dictionary.T)
        physical_dictionary = orthonormal_columns.T.reshape(
            args.physical_dictionary_rank,
            args.section_count,
            args.section_count,
        )

    results = []
    for site_count in args.site_counts:
        model = PositiveTensorNetworkMetric(
            reference_h,
            site_count=site_count,
            bond_dimension=args.bond_dimension,
            target_normalization=1.0 / (2.0 * site_count),
            positive_floor=1.0e-4,
            initialization_noise=1.0e-3,
            physical_dictionary=physical_dictionary,
            trainable_physical_dictionary=args.train_physical_dictionary,
            transfer_implementation=args.transfer_implementation,
            seed=args.seed + site_count,
            dtype=complex_dtype,
            device=device,
        )
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
        with torch.no_grad():
            for _ in range(args.timed_repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                potential, metric = model.potential_and_metric(values, derivatives)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - started)
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
        local_dimension = (
            args.section_count**2
            if args.physical_dictionary_rank is None
            else args.physical_dictionary_rank
        )
        if site_count == 1:
            expected_parameters = 2 * local_dimension
        else:
            expected_parameters = 2 * local_dimension * (
                2 * args.bond_dimension
                + (site_count - 2) * args.bond_dimension**2
            )
        if args.train_physical_dictionary:
            expected_parameters += (
                2 * args.physical_dictionary_rank * args.section_count**2
            )
        results.append(
            {
                "site_count": site_count,
                "polarization_degree_for_k0_2": 2 * site_count,
                "trainable_real_parameter_count": model.trainable_real_parameter_count,
                "expected_real_parameter_count": expected_parameters,
                "parameter_storage_bytes": int(
                    sum(
                        parameter.numel() * parameter.element_size()
                        for parameter in model.parameters()
                    )
                ),
                "median_forward_seconds": median_seconds,
                "mean_forward_seconds": statistics.fmean(timings),
                "forward_microseconds_per_point": 1.0e6
                * median_seconds
                / args.batch_size,
                "device_memory": peak_memory,
            }
        )
        del potential, metric, model
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
        "schema": "positive-tensor-network-kernel-scaling-benchmark-v1",
        "seed": args.seed,
        "section_count": args.section_count,
        "bond_dimension": args.bond_dimension,
        "architecture": (
            "dense_local_cores"
            if args.physical_dictionary_rank is None
            else "shared_local_dictionary"
        ),
        "physical_dictionary_rank": args.physical_dictionary_rank,
        "trainable_physical_dictionary": args.train_physical_dictionary,
        "coordinate_count": args.coordinate_count,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "transfer_implementation": args.transfer_implementation,
        "warmup_repeats": args.warmup_repeats,
        "timed_repeats": args.timed_repeats,
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
