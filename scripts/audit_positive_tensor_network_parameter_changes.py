#!/usr/bin/env python3
"""Audit which tensors changed between two positive TN checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--trained-model", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_expected_trainable(payload: dict[str, Any], name: str) -> bool:
    if name.startswith(("cores.", "coefficient_cores.")):
        return True
    if name == "physical_dictionary":
        return bool(payload.get("trainable_physical_dictionary", False))
    return name in {
        "two_site_core",
        "two_site_orbit_parameters",
        "three_site_core",
    } or name.startswith("blocked_two_site_orbit_parameters.")


def main() -> None:
    import torch

    args = parse_args()
    if args.absolute_tolerance < 0.0:
        raise SystemExit("absolute-tolerance cannot be negative")
    initial_path = args.initial_model.expanduser().resolve()
    trained_path = args.trained_model.expanduser().resolve()
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    trained = torch.load(trained_path, map_location="cpu", weights_only=False)
    initial_state = initial["state_dict"]
    trained_state = trained["state_dict"]
    if set(initial_state) != set(trained_state):
        raise SystemExit("checkpoint state keys differ")

    tensors = {}
    trainable_names = []
    changed_trainable_names = []
    changed_fixed_names = []
    for name, initial_value in initial_state.items():
        trained_value = trained_state[name]
        if initial_value.shape != trained_value.shape:
            raise SystemExit(f"tensor shape changed for {name}")
        difference = trained_value - initial_value
        maximum = float(torch.max(torch.abs(difference)))
        initial_norm = float(torch.linalg.vector_norm(initial_value))
        difference_norm = float(torch.linalg.vector_norm(difference))
        expected_trainable = is_expected_trainable(initial, name)
        changed = maximum > args.absolute_tolerance
        if expected_trainable:
            trainable_names.append(name)
            if changed:
                changed_trainable_names.append(name)
        elif changed:
            changed_fixed_names.append(name)
        tensors[name] = {
            "shape": list(initial_value.shape),
            "expected_trainable": expected_trainable,
            "changed": changed,
            "maximum_absolute_change": maximum,
            "frobenius_change": difference_norm,
            "relative_frobenius_change": (
                difference_norm / initial_norm if initial_norm > 0.0 else None
            ),
        }

    output = {
        "schema": "positive-tensor-network-parameter-change-audit-v1",
        "initial_model": str(initial_path),
        "initial_model_sha256": sha256(initial_path),
        "trained_model": str(trained_path),
        "trained_model_sha256": sha256(trained_path),
        "absolute_tolerance": args.absolute_tolerance,
        "expected_trainable_tensor_count": len(trainable_names),
        "changed_trainable_tensor_count": len(changed_trainable_names),
        "all_expected_trainable_tensors_changed": (
            set(trainable_names) == set(changed_trainable_names)
        ),
        "changed_fixed_tensors": changed_fixed_names,
        "fixed_tensors_unchanged": not changed_fixed_names,
        "tensors": tensors,
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
