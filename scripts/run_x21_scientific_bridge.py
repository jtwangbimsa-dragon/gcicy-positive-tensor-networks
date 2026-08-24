#!/usr/bin/env python3
"""Prepare, run, inspect, and fail-closed-normalize X21 Round 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.x21_scientific_bridge import (  # noqa: E402
    X21ScientificBridgeError,
    adjudicate_optimizer_path_bridge,
    bridge_status,
    normalize_optimizer_path_bridge,
    prepare_optimizer_path_bridge,
    run_optimizer_path_bridge,
)


def _read_object(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X21ScientificBridgeError(
            f"cannot read {role} {resolved}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise X21ScientificBridgeError(f"{role} must be a JSON object")
    return value


def _host_certificate(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "expected SEED=/path/to/certificate.json"
        ) from error
    if seed <= 0 or not path_text:
        raise argparse.ArgumentTypeError("certificate seed/path must be nonempty")
    return seed, Path(path_text)


def _host_certificates(args: argparse.Namespace) -> dict[int, Path]:
    rows = dict(args.host_stability_certificate)
    if len(rows) != len(args.host_stability_certificate):
        raise X21ScientificBridgeError("host certificate seed was repeated")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", allow_abbrev=False)
    prepare.add_argument("--campaign-run-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--parent-bindings", type=Path, required=True)
    prepare.add_argument("--source-artifact", type=Path, required=True)
    prepare.add_argument("--train-common-pool", type=Path, required=True)
    prepare.add_argument("--selection-common-pool", type=Path, required=True)
    prepare.add_argument("--fresh-development-binding", type=Path)
    prepare.add_argument("--repository-root", type=Path, default=ROOT)
    prepare.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prepare.add_argument("--workers", type=int, default=1)
    run = commands.add_parser("run", allow_abbrev=False)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument(
        "--host-stability-certificate",
        action="append",
        type=_host_certificate,
        default=[],
        metavar="SEED=PATH",
    )
    for name in ("status", "normalize", "adjudicate"):
        child = commands.add_parser(name, allow_abbrev=False)
        child.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            output = prepare_optimizer_path_bridge(
                campaign_run_root=args.campaign_run_root,
                output_root=args.output_root,
                parent_bindings=_read_object(
                    args.parent_bindings, role="parent bindings"
                ),
                source_artifact=args.source_artifact,
                train_common_pool=args.train_common_pool,
                selection_common_pool=args.selection_common_pool,
                repository_root=args.repository_root,
                fresh_development_binding=(
                    None
                    if args.fresh_development_binding is None
                    else _read_object(
                        args.fresh_development_binding,
                        role="fresh development binding",
                    )
                ),
                device=args.device,
                workers=args.workers,
            )
        elif args.command == "run":
            output = run_optimizer_path_bridge(
                args.output_root,
                host_stability_certificates=_host_certificates(args),
            )
        elif args.command == "status":
            output = bridge_status(args.output_root)
        elif args.command == "normalize":
            output = normalize_optimizer_path_bridge(args.output_root)
        elif args.command == "adjudicate":
            output = adjudicate_optimizer_path_bridge(args.output_root)
        else:  # pragma: no cover - argparse enforces known commands.
            raise X21ScientificBridgeError(f"unsupported command: {args.command}")
    except (OSError, ValueError, X21ScientificBridgeError) as error:
        print(f"[x21-scientific-bridge] error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
