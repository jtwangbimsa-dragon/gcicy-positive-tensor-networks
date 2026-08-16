#!/usr/bin/env python3
"""Benchmark one native-E2 tree block backward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_generic_quintic_h4_architecture_arms import (  # noqa: E402
    build_checkpoint_model,
)
from scripts.train_generic_quintic_compiled_tree import whiten_dataset  # noqa: E402
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    load_pool_arrays,
    make_dataset,
    native_e2_streaming_backward,
    normalized_native_e2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--pullbacks", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("streaming", "graph"), required=True)
    parser.add_argument("--stage", choices=("root", "internal", "all"), default="internal")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=202607472)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    configuration = payload["configuration"]
    dtype = (
        torch.complex64
        if str(configuration["precision"]) == "complex64"
        else torch.complex128
    )
    model = build_checkpoint_model(payload, device=device)
    model.set_trainable_stage_(args.stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    arrays, _ = load_pool_arrays(
        args.points.expanduser().resolve(),
        args.pullbacks.expanduser().resolve(),
        size=args.batch_size,
        seed=args.seed,
    )
    dataset = whiten_dataset(
        make_dataset(
            arrays,
            np.asarray(configuration["exponents"], dtype=np.int64),
            feature_batch_size=512,
            complex_dtype=dtype,
            device=device,
        ),
        np.asarray(configuration["whitening"], dtype=np.complex128),
    )

    def one_step() -> float:
        model.zero_grad(set_to_none=True)
        if args.mode == "streaming":
            loss, _ = native_e2_streaming_backward(
                model,
                dataset,
                chunk_size=args.chunk_size,
            )
        else:
            loss, _ = normalized_native_e2(
                model,
                dataset,
                chunk_size=args.chunk_size,
            )
            loss.backward()
        return float(loss)

    for _ in range(args.warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    losses = [one_step() for _ in range(args.repeats)]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "mode": args.mode,
                "stage": args.stage,
                "batch_size": args.batch_size,
                "chunk_size": args.chunk_size,
                "trainable_real_parameters": int(
                    2 * sum(parameter.numel() for parameter in parameters)
                ),
                "seconds_total": elapsed,
                "seconds_per_step": elapsed / args.repeats,
                "losses": losses,
                "peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if device.type == "cuda"
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
