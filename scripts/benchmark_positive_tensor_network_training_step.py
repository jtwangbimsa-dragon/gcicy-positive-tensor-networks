#!/usr/bin/env python3
"""Benchmark TN inference and a native training update in one code path."""

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
from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    distillation_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-count", type=int, default=3)
    parser.add_argument("--section-count", type=int, default=45)
    parser.add_argument("--bond-dimension", type=int, default=5)
    parser.add_argument("--coordinate-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--warmup-repeats", type=int, default=5)
    parser.add_argument("--timed-repeats", type=int, default=30)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "seconds": values,
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.fmean(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def main() -> None:
    import torch

    args = parse_args()
    positive = (
        args.site_count,
        args.section_count,
        args.bond_dimension,
        args.coordinate_count,
        args.batch_size,
        args.timed_repeats,
    )
    if min(positive) <= 0 or args.warmup_repeats < 0:
        raise SystemExit("all dimensions and repeat counts must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    complex_dtype = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }[args.precision]
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    scale = 1.0 / np.sqrt(2.0 * args.section_count)
    values = torch.tensor(
        scale
        * (
            rng.normal(size=(args.batch_size, args.section_count))
            + 1j * rng.normal(size=(args.batch_size, args.section_count))
        ),
        dtype=complex_dtype,
        device=device,
    )
    derivatives = torch.tensor(
        scale
        * (
            rng.normal(
                size=(
                    args.batch_size,
                    args.section_count,
                    args.coordinate_count,
                )
            )
            + 1j
            * rng.normal(
                size=(
                    args.batch_size,
                    args.section_count,
                    args.coordinate_count,
                )
            )
        ),
        dtype=complex_dtype,
        device=device,
    )
    weights = torch.full(
        (args.batch_size,),
        1.0 / args.batch_size,
        dtype=real_dtype,
        device=device,
    )
    log_omega = torch.zeros(args.batch_size, dtype=real_dtype, device=device)
    fixed_log_kappa = torch.zeros((), dtype=real_dtype, device=device)

    model = PositiveTensorNetworkMetric(
        np.eye(args.section_count, dtype=np.complex128),
        site_count=args.site_count,
        bond_dimension=args.bond_dimension,
        target_normalization=1.0 / (2.0 * args.site_count),
        positive_floor=1.0e-4,
        initialization_noise=1.0e-3,
        seed=args.seed,
        dtype=complex_dtype,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-4)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def native_loss(potential, metric):
        _, _, log_energy, ma, tail, _ = distillation_components(
            potential,
            metric,
            teacher_potential=None,
            teacher_inverse_cholesky=None,
            log_omega=log_omega,
            weights=weights,
            fixed_log_kappa=fixed_log_kappa,
            tail_fraction=0.02,
            tail_ratio_threshold=1.5,
            tail_smooth_temperature=0.05,
        )
        return log_energy + ma + 0.1 * tail

    # Warm both inference and update paths before resetting the measurements.
    for _ in range(args.warmup_repeats):
        with torch.no_grad():
            model.potential_and_metric(values, derivatives)
        optimizer.zero_grad(set_to_none=True)
        potential, metric = model.potential_and_metric(values, derivatives)
        native_loss(potential, metric).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    synchronize()

    forward_times: list[float] = []
    forward_loss_times: list[float] = []
    update_times: list[float] = []
    for _ in range(args.timed_repeats):
        synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            model.potential_and_metric(values, derivatives)
        synchronize()
        forward_times.append(time.perf_counter() - started)

        synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            potential, metric = model.potential_and_metric(values, derivatives)
            native_loss(potential, metric)
        synchronize()
        forward_loss_times.append(time.perf_counter() - started)

        synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        potential, metric = model.potential_and_metric(values, derivatives)
        loss = native_loss(potential, metric)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        synchronize()
        update_times.append(time.perf_counter() - started)

    device_metadata: dict[str, object] = {"kind": str(device)}
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
        "schema": "positive-tensor-network-synchronized-timing-v1",
        "configuration": {
            "site_count": args.site_count,
            "section_count": args.section_count,
            "bond_dimension": args.bond_dimension,
            "coordinate_count": args.coordinate_count,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "trainable_real_parameter_count": model.trainable_real_parameter_count,
            "seed": args.seed,
            "warmup_repeats": args.warmup_repeats,
            "timed_repeats": args.timed_repeats,
        },
        "device": device_metadata,
        "timing_scope": {
            "forward": "potential_and_metric under no_grad",
            "forward_and_loss": "forward plus native E2/log/tail loss under no_grad",
            "training_update": "zero_grad, forward, native loss, backward, clipping, Adam step",
            "cuda_synchronization": "before and after every timed region",
        },
        "forward": summarize(forward_times),
        "forward_and_loss": summarize(forward_loss_times),
        "training_update": summarize(update_times),
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
