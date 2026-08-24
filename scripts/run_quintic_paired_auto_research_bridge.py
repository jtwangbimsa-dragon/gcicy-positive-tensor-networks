#!/usr/bin/env python3
"""Prepare, run, normalize, and adjudicate quintic paired Auto Research."""

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
from gcicy_metric.pipeline.quintic_paired_auto_research_bridge import (  # noqa: E402
    adjudicate_paired_bridge,
    normalize_paired_bridge,
    paired_bridge_status,
    prepare_paired_bridge,
    run_paired_bridge,
)


def _parent(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "expected SEED=/path/to/parent-checkpoint.pt"
        ) from error
    if seed <= 0 or not path_text:
        raise argparse.ArgumentTypeError("seed/path must be nonempty")
    return seed, Path(path_text)


def _runtime(args: argparse.Namespace) -> dict[str, object]:
    return {
        "device": args.device,
        "threads": args.threads,
        "train_chunk_size": args.train_chunk_size,
        "feature_batch_size": args.feature_batch_size,
        "eval_batch_size": args.eval_batch_size,
    }


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manager-root", type=Path, required=True)
    prepare.add_argument("--execution-handoff", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument(
        "--host-certificate", type=_parent, action="append", required=True
    )
    prepare.add_argument("--repository-root", type=Path, default=ROOT)
    prepare.add_argument("--device", choices=("cuda",), default="cuda")
    prepare.add_argument("--threads", type=int, default=6)
    prepare.add_argument("--train-chunk-size", type=int, default=512)
    prepare.add_argument("--feature-batch-size", type=int, default=512)
    prepare.add_argument("--eval-batch-size", type=int, default=256)

    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--host-certificate", type=_parent, action="append", required=True)
    run.add_argument("--seed", type=int)

    for name in ("normalize", "adjudicate", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        certificates = dict(args.host_certificate)
        if len(certificates) != len(args.host_certificate):
            raise AutoResearchError("host-certificate seed was provided more than once")
        value = prepare_paired_bridge(
            manager_root=args.manager_root,
            execution_handoff_path=args.execution_handoff,
            output_root=args.output_root,
            host_stability_certificates=certificates,
            runtime=_runtime(args),
            repository_root=args.repository_root,
        )
    elif args.command == "run":
        certificates = dict(args.host_certificate)
        if len(certificates) != len(args.host_certificate):
            raise AutoResearchError("host-certificate seed was provided more than once")
        value = run_paired_bridge(
            args.root,
            host_stability_certificates=certificates,
            selected_seed=args.seed,
        )
    elif args.command == "normalize":
        value = normalize_paired_bridge(args.root)
    elif args.command == "adjudicate":
        value = adjudicate_paired_bridge(args.root)
    else:
        value = paired_bridge_status(args.root)
    _json(value)


if __name__ == "__main__":
    try:
        main()
    except (AutoResearchError, OSError, ValueError) as error:
        print(f"paired-auto-research error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
