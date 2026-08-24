#!/usr/bin/env python3
"""Prepare, run, and verify a strict quintic CUDA host-health probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.architecture_auto_research import (  # noqa: E402
    AutoResearchError,
)
from gcicy_metric.pipeline.quintic_host_gpu_probe import (  # noqa: E402
    DEFAULT_OPTIMIZER_STEPS,
    execute_cuda_worker,
    prepare_probe,
    probe_status,
    run_probe,
    verify_probe,
)


def _runtime(args: argparse.Namespace) -> dict[str, object]:
    return {
        "device": "cuda",
        "optimizer_steps": args.optimizer_steps,
        "learning_rate": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "threads": args.threads,
        "train_chunk_size": args.train_chunk_size,
        "feature_batch_size": args.feature_batch_size,
        "eval_batch_size": args.eval_batch_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--catalog", type=Path, required=True)
    prepare.add_argument("--round1-bridge-root", type=Path, required=True)
    prepare.add_argument("--parent-checkpoint", type=Path, required=True)
    prepare.add_argument("--parent-checkpoint-sha256", required=True)
    prepare.add_argument("--parent-seed", type=int, required=True)
    prepare.add_argument("--probe-id", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--repository-root", type=Path, default=ROOT)
    prepare.add_argument("--optimizer-steps", type=int, default=DEFAULT_OPTIMIZER_STEPS)
    prepare.add_argument("--learning-rate", type=float, default=3.0e-6)
    prepare.add_argument("--gradient-clip-norm", type=float, default=1.0)
    prepare.add_argument("--threads", type=int, default=6)
    prepare.add_argument("--train-chunk-size", type=int, default=512)
    prepare.add_argument("--feature-batch-size", type=int, default=512)
    prepare.add_argument("--eval-batch-size", type=int, default=256)

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
            catalog_path=args.catalog,
            round1_bridge_root=args.round1_bridge_root,
            parent_checkpoint=args.parent_checkpoint,
            parent_checkpoint_sha256=args.parent_checkpoint_sha256,
            parent_seed=args.parent_seed,
            probe_id=args.probe_id,
            output_root=args.output_root,
            runtime=_runtime(args),
            repository_root=args.repository_root,
        )
        exit_code = 0
    elif args.command == "run":
        value = run_probe(args.root)
        result = verify_probe(args.root, require_passing=False)
        exit_code = 0 if result["normalized_receipt"]["passes"] else 3
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
    except (AutoResearchError, OSError, ValueError) as error:
        print(f"quintic-host-gpu-probe error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
