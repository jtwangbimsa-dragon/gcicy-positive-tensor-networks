#!/usr/bin/env python3
"""Prepare, execute, verify, or inspect an immutable X21 CUDA host probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.x21_host_gpu_probe import (  # noqa: E402
    X21HostGPUProbeError,
    execute_cuda_worker,
    prepare_probe,
    probe_status,
    run_probe,
    verify_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--source-artifact", type=Path, required=True)
    prepare.add_argument("--train-common-pool", type=Path, required=True)
    prepare.add_argument("--selection-common-pool", type=Path, required=True)
    prepare.add_argument("--parent-model", type=Path, required=True)
    prepare.add_argument("--parent-model-sha256", required=True)
    prepare.add_argument("--parent-seed", type=int, required=True)
    prepare.add_argument("--probe-id", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--repository-root", type=Path, default=ROOT)
    prepare.add_argument("--workers", type=int, default=1)
    prepare.add_argument("--teacher-chunk-size", type=int, default=512)

    for name in ("run", "verify", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        value = prepare_probe(
            protocol_path=args.protocol,
            source_artifact=args.source_artifact,
            train_common_pool=args.train_common_pool,
            selection_common_pool=args.selection_common_pool,
            parent_model=args.parent_model,
            parent_model_sha256=args.parent_model_sha256,
            parent_seed=args.parent_seed,
            probe_id=args.probe_id,
            output_root=args.output_root,
            repository_root=args.repository_root,
            runtime={
                "device": "cuda",
                "workers": args.workers,
                "teacher_chunk_size": args.teacher_chunk_size,
            },
        )
        exit_code = 0
    elif args.command == "run":
        value = run_probe(args.root)
        verified = verify_probe(args.root, require_passing=False)
        exit_code = 0 if verified["normalized_receipt"]["passes"] else 3
    elif args.command == "verify":
        value = verify_probe(args.root, require_passing=True)
        exit_code = 0
    elif args.command == "status":
        value = probe_status(args.root)
        exit_code = 0
    else:
        value = execute_cuda_worker(args.root)
        exit_code = 0
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (X21HostGPUProbeError, OSError, ValueError) as error:
        print(f"x21-host-gpu-probe error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
