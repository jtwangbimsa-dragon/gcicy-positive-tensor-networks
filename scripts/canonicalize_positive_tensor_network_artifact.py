#!/usr/bin/env python3
"""Put a saved shared-dictionary positive TN in mixed-canonical gauge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.positive_tensor_network import (  # noqa: E402
    positive_tensor_network_from_artifact_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--center", type=int, default=-1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    import torch

    args = parse_args()
    source = args.model.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "shared_local_dictionary":
        raise ValueError("mixed canonicalization requires a shared-dictionary model")
    site_count = int(payload["site_count"])
    center = site_count // 2 if args.center < 0 else args.center
    if center < 0 or center >= site_count:
        raise ValueError("canonical center is outside the chain")
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device="cpu",
    )
    model.mixed_canonicalize_coefficient_cores_(center)

    output_payload = dict(payload)
    output_payload["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    output_payload["coefficient_chain_gauge"] = {
        "kind": "mixed_canonical",
        "center": center,
        "source_model": str(source),
        "source_model_sha256": sha256(source),
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output)
    write_json(
        args.summary.expanduser().resolve(),
        {
            "schema": "positive-tensor-network-mixed-canonicalization-v1",
            "source_model": str(source),
            "source_model_sha256": sha256(source),
            "model": str(output),
            "model_sha256": sha256(output),
            "site_count": site_count,
            "bond_dimension": int(payload["bond_dimension"]),
            "center": center,
            "trainable_real_parameter_count": int(
                model.trainable_real_parameter_count
            ),
        },
    )
    print(f"canonicalized sites={site_count} center={center}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
