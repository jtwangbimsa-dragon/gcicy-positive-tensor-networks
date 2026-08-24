#!/usr/bin/env python3
"""Operate the honest v2 manager around the executable quintic Round 1.

The runner audits existing artifacts and advances only to adapters explicitly
registered as available. ``run-next`` emits a hash-bound paired-bridge handoff
and never starts numerical work itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.architecture_auto_research import (  # noqa: E402
    AutoResearchError,
)
from gcicy_metric.pipeline.quintic_architecture_multi_round import (  # noqa: E402
    MultiRoundManager,
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutoResearchError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AutoResearchError(f"JSON root must be an object: {path}")
    return value


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _add_run_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser(
        "init", help="Lock the v2 catalog and bind the existing Round 1 roots."
    )
    _add_run_root(initialize)
    initialize.add_argument("--catalog", type=Path, required=True)
    initialize.add_argument("--round1-campaign-root", type=Path, required=True)
    initialize.add_argument("--round1-bridge-root", type=Path, required=True)

    sync = commands.add_parser(
        "sync-r1",
        help=(
            "Audit Round 1 and route promotion, scientific rejection, or "
            "technical failure into the durable v2 state."
        ),
    )
    _add_run_root(sync)

    status = commands.add_parser("status", help="Audit and print manager state.")
    _add_run_root(status)

    run_next = commands.add_parser(
        "run-next",
        help="Create a handoff for the current available adapter; do not start GPU work.",
    )
    _add_run_root(run_next)

    historical = commands.add_parser(
        "sync-historical-r2",
        help="Strictly import the rejected scale-0.25 R2 without claiming manager launch.",
    )
    _add_run_root(historical)
    historical.add_argument("--bridge-root", type=Path, required=True)

    record = commands.add_parser(
        "record-round-result",
        help="Consume an adjudicated manager-launched paired R3/R4 bridge.",
    )
    _add_run_root(record)
    record.add_argument("--round", type=int, choices=(3, 4), required=True)
    record.add_argument("--bridge-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        manager = MultiRoundManager.initialize(
            args.run_root,
            _read_object(args.catalog),
            round1_campaign_root=args.round1_campaign_root,
            round1_bridge_root=args.round1_bridge_root,
        )
        result = manager.state()
    else:
        manager = MultiRoundManager(args.run_root)
        if args.command == "sync-r1":
            result = manager.sync_round1()
        elif args.command == "status":
            result = manager.state()
        elif args.command == "run-next":
            result = manager.run_next()
        elif args.command == "sync-historical-r2":
            result = manager.sync_historical_r2(args.bridge_root)
        elif args.command == "record-round-result":
            result = manager.record_round_result(args.round, args.bridge_root)
        else:  # pragma: no cover - argparse enforces the command set.
            raise AssertionError(args.command)
    _emit(result)


if __name__ == "__main__":
    try:
        main()
    except AutoResearchError as error:
        print(f"multi-round auto-research error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
