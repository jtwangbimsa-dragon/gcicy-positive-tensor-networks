#!/usr/bin/env python3
"""Copy a positive-TN artifact with a new strictly positive reference floor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--positive-floor", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    import torch

    args = parse_args()
    if not 0.0 < args.positive_floor < 1.0:
        raise ValueError("positive floor must lie strictly between zero and one")
    source = args.model.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema") != "type11-positive-tensor-network-v1":
        raise ValueError("unrecognized positive tensor-network artifact")

    old_floor = float(payload["positive_floor"])
    output_payload = dict(payload)
    output_payload["positive_floor"] = float(args.positive_floor)
    output_payload["positive_floor_change"] = {
        "source_model": str(source),
        "source_model_sha256": sha256_file(source),
        "old_positive_floor": old_floor,
        "new_positive_floor": float(args.positive_floor),
        "purpose": (
            "common-origin degree-allocation experiment with a numerically "
            "negligible but strictly positive reference branch"
        ),
    }

    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output)
    write_json_atomic(
        args.summary.expanduser().resolve(),
        {
            "schema": "positive-tensor-network-floor-change-v1",
            "source_model": str(source),
            "source_model_sha256": sha256_file(source),
            "model": str(output),
            "model_sha256": sha256_file(output),
            "old_positive_floor": old_floor,
            "new_positive_floor": float(args.positive_floor),
            "strictly_positive": True,
        },
    )
    print(f"positive_floor={old_floor:.8g}->{args.positive_floor:.8g}", flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
