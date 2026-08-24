#!/usr/bin/env python3
"""Prepare, materialize, inspect, or validate the fresh X21 development pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.x21_fresh_development_pool import (  # noqa: E402
    X21FreshDevelopmentPoolError,
    fresh_development_pool_status,
    load_fresh_development_binding,
    materialize_fresh_development_pool,
    prepare_fresh_development_pool,
)


DEFAULT_PROTOCOL = ROOT / "experiments" / "protocols" / "x21_auto_research_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", allow_abbrev=False)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--frozen-root", type=Path, required=True)
    prepare.add_argument("--base-protocol", type=Path, default=DEFAULT_PROTOCOL)
    prepare.add_argument("--repository-root", type=Path, default=ROOT)
    prepare.add_argument(
        "--python-executable", type=Path, default=Path(sys.executable).resolve()
    )
    prepare.add_argument("--workers", type=int, default=6)
    prepare.add_argument("--backend", choices=("process", "thread"), default="process")

    for name in ("materialize", "status"):
        child = commands.add_parser(name, allow_abbrev=False)
        child.add_argument("--output-root", type=Path, required=True)

    validate = commands.add_parser("validate-binding", allow_abbrev=False)
    validate.add_argument("--binding", type=Path, required=True)
    validate.add_argument("--expected-binding-sha256")
    validate.add_argument("--expected-pool-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            output = prepare_fresh_development_pool(
                output_root=args.output_root,
                frozen_root=args.frozen_root,
                base_protocol=args.base_protocol,
                repository_root=args.repository_root,
                python_executable=args.python_executable,
                workers=args.workers,
                backend=args.backend,
            )
        elif args.command == "materialize":
            output = materialize_fresh_development_pool(args.output_root)
        elif args.command == "status":
            output = fresh_development_pool_status(args.output_root)
        elif args.command == "validate-binding":
            output = load_fresh_development_binding(
                args.binding,
                expected_binding_sha256=args.expected_binding_sha256,
                expected_pool_sha256=args.expected_pool_sha256,
            )
        else:  # pragma: no cover - argparse enforces known commands.
            raise X21FreshDevelopmentPoolError(f"unsupported command: {args.command}")
    except (OSError, ValueError, X21FreshDevelopmentPoolError) as error:
        print(f"[x21-fresh-development-pool] error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
