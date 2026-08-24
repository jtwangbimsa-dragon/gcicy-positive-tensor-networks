#!/usr/bin/env python3
"""Operate the isolated X21 recovery and scientific auto-research ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.x21_auto_research import (  # noqa: E402
    RecoveryStore,
    X21AutoResearchError,
    X21CampaignStore,
)


def _read_object(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X21AutoResearchError(
            f"could not read {role} {resolved}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise X21AutoResearchError(f"{role} must be a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    init_recovery = commands.add_parser("init-recovery")
    init_recovery.add_argument("--run-root", type=Path, required=True)
    init_recovery.add_argument("--campaign-id", required=True)

    record_recovery = commands.add_parser("record-recovery")
    record_recovery.add_argument("--run-root", type=Path, required=True)
    record_recovery.add_argument("--evidence", type=Path, required=True)

    recovery_status = commands.add_parser("recovery-status")
    recovery_status.add_argument("--run-root", type=Path, required=True)

    init = commands.add_parser("init")
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--protocol", type=Path, required=True)

    baseline = commands.add_parser("register-baseline")
    baseline.add_argument("--run-root", type=Path, required=True)
    baseline.add_argument("--family", type=Path, required=True)

    action = commands.add_parser("register-action")
    action.add_argument("--run-root", type=Path, required=True)
    action.add_argument("--action", type=Path, required=True)

    evidence = commands.add_parser("record-evidence")
    evidence.add_argument("--run-root", type=Path, required=True)
    evidence.add_argument("--evidence", type=Path, required=True)

    adjudicate = commands.add_parser("adjudicate-round")
    adjudicate.add_argument("--run-root", type=Path, required=True)
    adjudicate.add_argument("--round", type=int, required=True)

    skip = commands.add_parser("skip-resource")
    skip.add_argument("--run-root", type=Path, required=True)
    skip.add_argument("--skip", type=Path, required=True)

    status = commands.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-recovery":
            store = RecoveryStore.initialize(args.run_root, args.campaign_id)
            output = store.status()
        elif args.command == "record-recovery":
            store = RecoveryStore(args.run_root)
            output = store.record(_read_object(args.evidence, role="recovery evidence"))
        elif args.command == "recovery-status":
            output = RecoveryStore(args.run_root).status()
        elif args.command == "init":
            store = X21CampaignStore.initialize(
                args.run_root, _read_object(args.protocol, role="protocol")
            )
            output = store.status()
        elif args.command == "register-baseline":
            output = X21CampaignStore(args.run_root).register_baseline(
                _read_object(args.family, role="baseline family")
            )
        elif args.command == "register-action":
            output = X21CampaignStore(args.run_root).register_action(
                _read_object(args.action, role="action")
            )
        elif args.command == "record-evidence":
            output = X21CampaignStore(args.run_root).record_evidence(
                _read_object(args.evidence, role="scientific evidence")
            )
        elif args.command == "adjudicate-round":
            output = X21CampaignStore(args.run_root).adjudicate_round(args.round)
        elif args.command == "skip-resource":
            output = X21CampaignStore(args.run_root).skip_resource(
                _read_object(args.skip, role="resource skip")
            )
        elif args.command == "status":
            output = X21CampaignStore(args.run_root).status()
        else:  # pragma: no cover - argparse enforces a registered command.
            raise X21AutoResearchError(f"unsupported command: {args.command}")
    except (OSError, ValueError, X21AutoResearchError) as error:
        raise SystemExit(f"x21 auto-research error: {error}") from error
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
