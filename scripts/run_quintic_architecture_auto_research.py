#!/usr/bin/env python3
"""Operate the constrained quintic architecture Auto Research v1 ledger.

The command accepts declarative JSON only.  It never evaluates arbitrary code
or launches an arbitrary shell command; existing numerical workers produce the
registered evidence files consumed by ``adjudicate-round``, ``record-replay``,
and ``record-shadow``.
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
    CampaignStore,
    validate_action,
)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AutoResearchError(f"JSON root must be an object: {path}")
    return value


def emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def add_run_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser(
        "init", help="Lock a protocol and materialize fixed search indices."
    )
    add_run_root(initialize)
    initialize.add_argument("--protocol", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate-action", help="Validate and canonicalize one proposed action."
    )
    validate.add_argument("--action", type=Path, required=True)
    validate.add_argument("--parent-family-sha256")
    validate.add_argument("--search-indices-sha256")

    register = subparsers.add_parser(
        "register-action", help="Register a create-only action in the open round."
    )
    add_run_root(register)
    register.add_argument("--action", type=Path, required=True)

    record = subparsers.add_parser(
        "record-evidence",
        help="Persist one candidate's three-seed evidence for crash recovery.",
    )
    add_run_root(record)
    record.add_argument("--evidence", type=Path, required=True)

    adjudicate = subparsers.add_parser(
        "adjudicate-round",
        help="Validate all candidate evidence and select at most one winner.",
    )
    add_run_root(adjudicate)
    adjudicate.add_argument("--round", type=int, required=True)
    adjudicate.add_argument("--evidence", type=Path, action="append", default=[])

    freeze = subparsers.add_parser(
        "freeze", help="Freeze the current promoted three-seed finalist."
    )
    add_run_root(freeze)

    replay = subparsers.add_parser(
        "record-replay",
        help="Record the finalist's zero-update complex128 replay exactly once.",
    )
    add_run_root(replay)
    replay.add_argument("--evidence", type=Path, required=True)

    claim = subparsers.add_parser(
        "claim-shadow",
        help="Bind the single post-freeze shadow pool before evaluation.",
    )
    add_run_root(claim)
    claim.add_argument("--claim", type=Path, required=True)

    shadow = subparsers.add_parser(
        "record-shadow",
        help="Consume and adjudicate one post-freeze shadow evaluation.",
    )
    add_run_root(shadow)
    shadow.add_argument("--evidence", type=Path, required=True)

    status = subparsers.add_parser("status", help="Print the durable ledger.")
    add_run_root(status)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        store = CampaignStore.initialize(args.run_root, read_object(args.protocol))
        emit(
            {
                "protocol": store.protocol(),
                "search_indices": store.indices(),
                "ledger": store.ledger(),
            }
        )
        return
    if args.command == "validate-action":
        emit(
            validate_action(
                read_object(args.action),
                expected_parent_family_sha256=args.parent_family_sha256,
                expected_indices_sha256=args.search_indices_sha256,
            )
        )
        return
    store = CampaignStore(args.run_root)
    if args.command == "register-action":
        emit(store.register_action(read_object(args.action)))
    elif args.command == "record-evidence":
        emit(store.record_search_evidence(read_object(args.evidence)))
    elif args.command == "adjudicate-round":
        emit(
            store.adjudicate_round(
                args.round,
                [read_object(path) for path in args.evidence],
            )
        )
    elif args.command == "freeze":
        emit(store.freeze())
    elif args.command == "record-replay":
        emit(store.record_complex128_replay(read_object(args.evidence)))
    elif args.command == "claim-shadow":
        emit(store.claim_shadow(read_object(args.claim)))
    elif args.command == "record-shadow":
        emit(store.record_shadow(read_object(args.evidence)))
    elif args.command == "status":
        emit(store.status())
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except AutoResearchError as error:
        print(f"auto-research error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
