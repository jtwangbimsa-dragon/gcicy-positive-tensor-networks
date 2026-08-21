#!/usr/bin/env python3
"""Create a hash-bound, update-free precision twin of a saved positive TN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SUPPORTED_SCHEMAS = {
    "type11-positive-tensor-network-v1",
    "quintic-positive-tensor-network-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _expected_dtypes(precision: str) -> tuple[torch.dtype, torch.dtype]:
    if precision == "complex64":
        return torch.complex64, torch.float32
    if precision == "complex128":
        return torch.complex128, torch.float64
    raise ValueError(f"unsupported precision: {precision}")


def validate_source_payload(payload: dict[str, Any]) -> str:
    schema = str(payload.get("schema", ""))
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported positive tensor-network schema: {schema!r}")
    source_precision = str(payload.get("precision", ""))
    complex_dtype, real_dtype = _expected_dtypes(source_precision)
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("model artifact has no non-empty state_dict")
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        if value.is_complex() and value.dtype != complex_dtype:
            raise ValueError(
                f"tensor {name!r} dtype {value.dtype} disagrees with "
                f"artifact precision {source_precision}"
            )
        if value.is_floating_point() and value.dtype != real_dtype:
            raise ValueError(
                f"tensor {name!r} dtype {value.dtype} disagrees with "
                f"artifact precision {source_precision}"
            )
    return source_precision


def cast_payload(
    payload: dict[str, Any], *, target_precision: str
) -> tuple[dict[str, Any], bool, float]:
    """Cast state tensors without changing their mathematical values or weights."""

    source_precision = validate_source_payload(payload)
    if source_precision == target_precision:
        raise ValueError("source and target precision must be different")
    complex_dtype, real_dtype = _expected_dtypes(target_precision)
    converted = dict(payload)
    converted_state: dict[str, Any] = {}
    maximum_roundtrip_difference = 0.0
    exact_promotion = (
        source_precision == "complex64" and target_precision == "complex128"
    )
    for name, value in payload["state_dict"].items():
        if not torch.is_tensor(value):
            converted_state[name] = value
            continue
        if value.is_complex():
            target = value.detach().cpu().to(dtype=complex_dtype)
            roundtrip = target.to(dtype=value.dtype)
        elif value.is_floating_point():
            target = value.detach().cpu().to(dtype=real_dtype)
            roundtrip = target.to(dtype=value.dtype)
        else:
            target = value.detach().cpu().clone()
            roundtrip = target
        if value.numel() and (value.is_complex() or value.is_floating_point()):
            difference = torch.max(torch.abs(roundtrip - value.detach().cpu())).item()
            maximum_roundtrip_difference = max(
                maximum_roundtrip_difference, float(difference)
            )
        converted_state[name] = target
    if exact_promotion and maximum_roundtrip_difference != 0.0:
        raise AssertionError("complex64 to complex128 promotion was not exact")
    converted["state_dict"] = converted_state
    converted["precision"] = target_precision
    return converted, exact_promotion, maximum_roundtrip_difference


def embedded_cast_provenance(
    *,
    source_hash: str,
    source_precision: str,
    target_precision: str,
    exact_promotion: bool,
    maximum_roundtrip_difference: float,
) -> dict[str, Any]:
    """Return location-independent lineage embedded in the model artifact."""

    return {
        "schema": "positive-tensor-network-precision-cast-v1",
        "source_model_sha256": source_hash,
        "source_precision": source_precision,
        "target_precision": target_precision,
        "optimizer_updates": 0,
        "exact_binary_promotion": exact_promotion,
        "maximum_source_dtype_roundtrip_difference": (maximum_roundtrip_difference),
    }


def main() -> None:
    args = parse_args()
    source_path = args.model.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    if len({source_path, output_path, summary_path}) != 3:
        raise ValueError("model, output, and summary paths must be distinct")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    source_precision = validate_source_payload(payload)
    converted, exact_promotion, maximum_roundtrip_difference = cast_payload(
        payload, target_precision=args.precision
    )
    source_hash = sha256_file(source_path)
    converted["precision_cast"] = embedded_cast_provenance(
        source_hash=source_hash,
        source_precision=source_precision,
        target_precision=args.precision,
        exact_promotion=exact_promotion,
        maximum_roundtrip_difference=maximum_roundtrip_difference,
    )
    _atomic_torch_save(converted, output_path)
    output_hash = sha256_file(output_path)
    summary = {
        "schema": "positive-tensor-network-precision-cast-summary-v1",
        "source_model": str(source_path),
        "source_model_sha256": source_hash,
        "source_precision": source_precision,
        "cast_model": str(output_path),
        "cast_model_sha256": output_hash,
        "target_precision": args.precision,
        "optimizer_updates": 0,
        "exact_binary_promotion": exact_promotion,
        "maximum_source_dtype_roundtrip_difference": (maximum_roundtrip_difference),
    }
    _atomic_write_json(summary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
