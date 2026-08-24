#!/usr/bin/env python3
"""Create a short-lived fail-closed host-stability certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.host_stability_gate import (  # noqa: E402
    DEFAULT_FRESH_PROCESS_PROBES,
    DEFAULT_QUIET_WINDOW_HOURS,
    HostStabilityError,
    build_certificate,
    publish_certificate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-probe", type=Path, action="append", default=[])
    parser.add_argument(
        "--journal-file",
        type=Path,
        help="use a captured kernel journal instead of invoking journalctl",
    )
    return parser.parse_args()


def kernel_log(args: argparse.Namespace) -> str:
    if args.journal_file is not None:
        return args.journal_file.read_text(encoding="utf-8")
    completed = subprocess.run(
        [
            "journalctl",
            "-k",
            "--since",
            f"{DEFAULT_QUIET_WINDOW_HOURS} hours ago",
            "--no-pager",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise HostStabilityError("cannot read the kernel journal")
    return completed.stdout


def source_commit() -> str:
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise HostStabilityError("host certificate requires a clean source checkout")
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode != 0:
        raise HostStabilityError("cannot resolve the source commit")
    return commit


def host_identity_sha256() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise HostStabilityError("cannot resolve the GPU host identity")
    payload = {
        "node": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "gpu": completed.stdout.strip().splitlines(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fresh_process_results(
    args: argparse.Namespace,
    *,
    checkpoint_sha256: str,
    host_identity: str,
    commit: str,
) -> list[dict[str, object]]:
    code = (
        "import json,sys,numpy,scipy,torch; "
        "torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "print(json.dumps({'numpy':numpy.__version__,'scipy':scipy.__version__,"
        "'torch':torch.__version__},sort_keys=True))"
    )
    rows = []
    for attempt in range(1, DEFAULT_FRESH_PROCESS_PROBES + 1):
        completed = subprocess.run(
            [args.python, "-c", code, str(args.checkpoint.expanduser().resolve())],
            check=False,
            capture_output=True,
            text=True,
        )
        rows.append(
            {
                "attempt": attempt,
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(
                    completed.stdout.encode("utf-8")
                ).hexdigest(),
                "checkpoint_sha256": checkpoint_sha256,
                "host_identity_sha256": host_identity,
                "source_commit": commit,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    commit = source_commit()
    host_identity = host_identity_sha256()
    probes = [json.loads(path.read_text(encoding="utf-8")) for path in args.gpu_probe]
    certificate = build_certificate(
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        host_identity_sha256=host_identity,
        source_commit=commit,
        kernel_log=kernel_log(args),
        fresh_process_results=fresh_process_results(
            args,
            checkpoint_sha256=args.checkpoint_sha256,
            host_identity=host_identity,
            commit=commit,
        ),
        gpu_probe_values=probes,
    )
    publish_certificate(args.output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    raise SystemExit(0 if certificate["scientific_authorized"] else 3)


if __name__ == "__main__":
    try:
        main()
    except (OSError, json.JSONDecodeError, HostStabilityError) as error:
        print(f"host-stability error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
