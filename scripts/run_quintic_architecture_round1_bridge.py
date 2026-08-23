#!/usr/bin/env python3
"""Prepare, run, and normalize the fixed quintic architecture Round-1 bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.architecture_auto_research import (
    ACTION_SCHEMA,
    AutoResearchError,
    CampaignStore,
)  # noqa: E402
from gcicy_metric.pipeline.quintic_architecture_round1_bridge import (  # noqa: E402
    ROUND1_BASELINE_SHA256,
    ROUND1_EDGE,
    ROUND1_NEW_REAL_PARAMETERS,
    ROUND1_SOURCE_DIMENSION,
    ROUND1_STRUCTURAL_MAXIMUM,
    ROUND1_TARGET_DIMENSION,
    normalize_round1_bridge,
    prepare_round1_bridge,
    run_round1_bridge,
)


DEFAULT_CANDIDATE_ID = "edge6-rank39"
_USER_UNIT = re.compile(r"[A-Za-z0-9_.@:-]+\.service")


def _baseline(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "expected SEED=/path/to/checkpoint.pt"
        ) from error
    if seed <= 0 or not path_text:
        raise argparse.ArgumentTypeError("seed and checkpoint path must be nonempty")
    return seed, Path(path_text)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AutoResearchError(f"JSON root must be an object: {path}")
    return value


def _runtime(args: argparse.Namespace) -> dict[str, object]:
    return {
        "device": args.device,
        "threads": args.threads,
        "train_chunk_size": args.train_chunk_size,
        "feature_batch_size": args.feature_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "early_stopping_evaluations": args.early_stopping_evaluations,
    }


def _baselines(args: argparse.Namespace) -> dict[int, Path]:
    baselines = dict(args.baseline_checkpoint)
    if len(baselines) != len(args.baseline_checkpoint):
        raise AutoResearchError("baseline seed was provided more than once")
    return baselines


def _ensure_registered_action(
    store: CampaignStore,
    *,
    candidate_id: str,
) -> dict[str, object]:
    ledger = store.status()
    existing_round = ledger.get("rounds", {}).get("1")
    if isinstance(existing_round, dict):
        candidate = existing_round.get("candidates", {}).get(candidate_id)
        if not isinstance(candidate, dict):
            raise AutoResearchError("Round 1 belongs to another candidate set")
        return _read_object(Path(candidate["action_path"]))
    if ledger.get("current_round") != 0 or ledger.get("champion_candidate_id") != "baseline":
        raise AutoResearchError("Round 1 cannot be bootstrapped from this campaign state")
    action = {
        "schema": ACTION_SCHEMA,
        "candidate_id": candidate_id,
        "round": 1,
        "parent_family_sha256": ledger["champion_family"]["family_sha256"],
        "search_indices_sha256": ledger["search_indices_sha256"],
        "mutation": {
            "kind": "rank",
            "target": "internal-edge",
            "edge": ROUND1_EDGE,
            "source_dimension": ROUND1_SOURCE_DIMENSION,
            "target_dimension": ROUND1_TARGET_DIMENSION,
            "structural_maximum": ROUND1_STRUCTURAL_MAXIMUM,
            "new_output_real_parameters": ROUND1_NEW_REAL_PARAMETERS,
        },
        "pipeline": [{"kind": "local-activate"}, {"kind": "matched-relax"}],
    }
    return store.register_action(action)


def wait_for_user_unit(unit: str, *, poll_seconds: int) -> dict[str, str]:
    """Wait for one registered user service to exit successfully, or fail closed."""

    if not _USER_UNIT.fullmatch(unit):
        raise AutoResearchError("wait-for-user-unit is malformed")
    if poll_seconds <= 0:
        raise AutoResearchError("wait-poll-seconds must be positive")
    properties = (
        "LoadState",
        "ActiveState",
        "SubState",
        "Result",
        "ExecMainStatus",
    )
    while True:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                *[f"--property={name}" for name in properties],
                "--no-pager",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AutoResearchError(
                f"cannot inspect prerequisite user service {unit}"
            )
        state = {}
        for line in completed.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                state[key] = value
        if set(state) != set(properties) or state["LoadState"] != "loaded":
            raise AutoResearchError(
                f"prerequisite user service is not durably inspectable: {unit}"
            )
        active = state["ActiveState"]
        if active in {"active", "activating", "reloading", "deactivating"}:
            print(
                f"waiting for prerequisite {unit}: {active}/{state['SubState']}",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        if (
            active == "inactive"
            and state["Result"] == "success"
            and state["ExecMainStatus"] == "0"
        ):
            print(f"prerequisite completed successfully: {unit}", flush=True)
            return state
        raise AutoResearchError(
            f"prerequisite user service did not succeed: {unit} ({state})"
        )


def execute_round1(args: argparse.Namespace) -> dict[str, object]:
    """Idempotently drive the fixed campaign through Round-1 adjudication."""

    store = CampaignStore.initialize(
        args.campaign_run_root,
        _read_object(args.protocol),
    )
    action = _ensure_registered_action(store, candidate_id=args.candidate_id)
    ledger = store.status()
    candidate = ledger["rounds"]["1"]["candidates"][args.candidate_id]
    if candidate.get("evidence_sha256") is None:
        prepare_round1_bridge(
            campaign_run_root=args.campaign_run_root,
            candidate_id=args.candidate_id,
            output_root=args.output_root,
            baseline_checkpoints=_baselines(args),
            runtime=_runtime(args),
            repository_root=ROOT,
        )
        if args.wait_for_user_unit is not None:
            wait_for_user_unit(
                args.wait_for_user_unit,
                poll_seconds=args.wait_poll_seconds,
            )
        run_round1_bridge(args.output_root)
        evidence = normalize_round1_bridge(args.output_root)
        store.record_search_evidence(evidence)
    adjudication = store.adjudicate_round(1)
    return {
        "action": action,
        "adjudication": adjudication,
        "ledger": store.status(),
        "baseline_sha256": ROUND1_BASELINE_SHA256,
    }


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-run-root", type=Path, required=True)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        type=_baseline,
        action="append",
        required=True,
        metavar="SEED=PATH",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--train-chunk-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-evaluations", type=int, default=6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    _add_execution_arguments(prepare)

    execute = commands.add_parser(
        "execute",
        help=(
            "Initialize/register, run all three seeds, normalize evidence, "
            "and adjudicate Round 1 idempotently."
        ),
    )
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--wait-for-user-unit")
    execute.add_argument("--wait-poll-seconds", type=int, default=30)
    _add_execution_arguments(execute)

    run = commands.add_parser("run")
    run.add_argument("--bridge-root", type=Path, required=True)
    run.add_argument("--seed", type=int)

    normalize = commands.add_parser("normalize")
    normalize.add_argument("--bridge-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare_round1_bridge(
            campaign_run_root=args.campaign_run_root,
            candidate_id=args.candidate_id,
            output_root=args.output_root,
            baseline_checkpoints=_baselines(args),
            runtime=_runtime(args),
            repository_root=ROOT,
        )
    elif args.command == "execute":
        result = execute_round1(args)
    elif args.command == "run":
        result = run_round1_bridge(args.bridge_root, selected_seed=args.seed)
    elif args.command == "normalize":
        result = normalize_round1_bridge(args.bridge_root)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except AutoResearchError as error:
        print(f"round1 bridge error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
