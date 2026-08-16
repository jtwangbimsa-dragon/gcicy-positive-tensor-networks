#!/usr/bin/env python3
"""Record immutable file hashes and runtime metadata for a completed run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": display_path(resolved),
        "bytes": int(stat.st_size),
        "sha256": sha256(resolved),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def runtime_record() -> dict[str, Any]:
    output: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
    }
    try:
        import scipy

        output["scipy"] = scipy.__version__
    except ImportError:
        output["scipy"] = None
    try:
        import torch

        output.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_runtime": torch.version.cuda,
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError:
        output.update(
            {
                "torch": None,
                "cuda_available": False,
                "cuda_runtime": None,
                "cuda_device": None,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    summary_path = args.summary.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output = {
        "schema_version": 1,
        "description": "Post-run provenance for a completed remote computation.",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "compute_policy": "scientific computation executed on the remote RTX 4090 host",
        "runtime": runtime_record(),
        "files": {
            "specification": file_record(args.spec),
            "summary": file_record(summary_path),
            "artifact": file_record(args.artifact),
        },
        "reported_success": bool(summary.get("success", False)),
        "reported_internal_gates_passed": bool(
            summary.get("passed_internal_gates", False)
        ),
        "reported_runtime_seconds": summary.get("runtime_seconds"),
    }
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
