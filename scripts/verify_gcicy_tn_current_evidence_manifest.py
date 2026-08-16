#!/usr/bin/env python3
"""Verify every source and generated-output hash in the current TN manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path(
            "gcicy paper/generated_tn/current_evidence_manifest_20260809.json"
        ),
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    root = manifest_path.parents[2]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    checked = 0
    for group in ("sources", "outputs"):
        for name, record in manifest.get(group, {}).items():
            path = root / record["path"]
            if not path.is_file():
                failures.append(f"{group}.{name}: missing {path}")
                continue
            observed = sha256(path)
            expected = record["sha256"]
            if observed != expected:
                failures.append(
                    f"{group}.{name}: expected {expected}, observed {observed}"
                )
            checked += 1

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"verified {checked} manifest-bound artifacts")


if __name__ == "__main__":
    main()
