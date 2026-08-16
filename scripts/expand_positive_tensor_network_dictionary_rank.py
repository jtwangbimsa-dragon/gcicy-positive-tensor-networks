#!/usr/bin/env python3
"""Losslessly expand a saved shared-dictionary tensor-network rank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import expand_shared_local_dictionary_rank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--maximum-relative-core-error", type=float, default=1.0e-12)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for dictionary expansion") from exc

    args = parse_args()
    source_path = args.model.expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("schema") not in {
        "type11-positive-tensor-network-v1",
        "quintic-positive-tensor-network-v1",
    }:
        raise ValueError("unrecognized positive tensor-network artifact")
    if payload.get("architecture") != "shared_local_dictionary":
        raise ValueError("dictionary expansion requires a shared dictionary model")
    if payload.get("teacher_artifact_sha256") is not None:
        raise ValueError("registered teacher-free expansion requires no saved teacher")
    state = payload["state_dict"]
    dictionary_value = state.get("physical_dictionary")
    if dictionary_value is None:
        raise ValueError("saved model has no physical dictionary")
    coefficient_keys = sorted(
        (key for key in state if key.startswith("coefficient_cores.")),
        key=lambda key: int(key.split(".")[1]),
    )
    if len(coefficient_keys) != int(payload["site_count"]):
        raise ValueError("saved coefficient-core count does not match site count")
    dictionary = dictionary_value.detach().cpu().numpy()
    coefficients = tuple(state[key].detach().cpu().numpy() for key in coefficient_keys)
    source_rank = int(dictionary.shape[0])
    expansion = expand_shared_local_dictionary_rank(
        dictionary,
        coefficients,
        args.target_rank,
        seed=args.seed,
    )
    if expansion.relative_core_error > args.maximum_relative_core_error:
        raise FloatingPointError(
            "dictionary expansion exceeded the registered core-error tolerance"
        )

    precision = str(payload["precision"])
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    output_state = {
        key: value.detach().cpu().clone() for key, value in state.items()
    }
    output_state["physical_dictionary"] = torch.tensor(
        expansion.physical_dictionary,
        dtype=dtype,
    )
    for key, coefficient in zip(
        coefficient_keys,
        expansion.coefficient_cores,
        strict=True,
    ):
        output_state[key] = torch.tensor(coefficient, dtype=dtype)

    output_payload = dict(payload)
    output_payload.update(
        {
            "state_dict": output_state,
            "physical_dictionary_rank": int(args.target_rank),
            "trainable_physical_dictionary": True,
            "physical_dictionary_gauge": "row_orthonormal",
            "training_mode": "teacher_free_dictionary_rank_expansion_initialization",
            "dictionary_rank_expansion_source_model": str(source_path),
            "dictionary_rank_expansion_source_sha256": sha256_file(source_path),
            "dictionary_rank_expansion_source_rank": source_rank,
            "dictionary_rank_expansion_seed": int(args.seed),
            "dictionary_rank_expansion_rule": (
                "exact compensated orthonormalization plus zero-coefficient "
                "orthogonal complement"
            ),
        }
    )
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(out_path)

    dictionary_parameter_count = 2 * expansion.physical_dictionary.size
    coefficient_parameter_count = 2 * sum(
        coefficient.size for coefficient in expansion.coefficient_cores
    )
    summary = {
        "schema": "positive-tensor-network-dictionary-rank-expansion-v1",
        "source_model": str(source_path),
        "source_model_sha256": sha256_file(source_path),
        "model": str(out_path),
        "model_sha256": sha256_file(out_path),
        "source_rank": source_rank,
        "target_rank": int(args.target_rank),
        "seed": int(args.seed),
        "site_count": int(payload["site_count"]),
        "bond_dimension": int(payload["bond_dimension"]),
        "dictionary_real_parameter_count": int(dictionary_parameter_count),
        "coefficient_real_parameter_count": int(coefficient_parameter_count),
        "trainable_real_parameter_count": int(
            dictionary_parameter_count + coefficient_parameter_count
        ),
        "relative_core_error": expansion.relative_core_error,
        "row_orthonormality_error": expansion.row_orthonormality_error,
        "teacher_artifact_sha256": None,
    }
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temporary.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_temporary.replace(summary_path)
    print(
        f"rank={source_rank}->{args.target_rank} "
        f"relative_core_error={expansion.relative_core_error:.3e} "
        f"parameters={summary['trainable_real_parameter_count']}",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
