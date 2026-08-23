#!/usr/bin/env python3
"""Create a hash-bound certificate for a completed X21 resource preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence


SCHEMA = "x21-resource-preflight-certificate-v1"


class CertificateError(RuntimeError):
    """Raised when a prerequisite or resource preflight is not valid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise CertificateError(f"missing or empty {role}: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CertificateError(f"could not read {role} {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise CertificateError(f"{role} must be a JSON object")
    return payload, sha256_file(resolved)


def _finite_tree(value: Any, *, field: str = "summary") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CertificateError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, field=f"{field}.{index}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, field=f"{field}.{key}")
        return
    raise CertificateError(f"{field} contains unsupported JSON data")


def _require_equal(payload: dict[str, Any], field: str, expected: Any) -> None:
    current: Any = payload
    for component in field.split("."):
        if not isinstance(current, dict) or component not in current:
            raise CertificateError(f"missing summary field {field}")
        current = current[component]
    if current != expected:
        raise CertificateError(
            f"summary field {field} is {current!r}, expected {expected!r}"
        )


def _number(payload: dict[str, Any], field: str) -> float:
    current: Any = payload
    for component in field.split("."):
        if not isinstance(current, dict) or component not in current:
            raise CertificateError(f"missing summary field {field}")
        current = current[component]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise CertificateError(f"summary field {field} must be numeric")
    result = float(current)
    if not math.isfinite(result):
        raise CertificateError(f"summary field {field} must be finite")
    return result


def build_certificate(args: argparse.Namespace) -> dict[str, Any]:
    source_run_root = args.source_run_root.expanduser().resolve()
    summary_path = args.source_summary.expanduser().resolve()
    expected_manifest = args.expected_manifest.expanduser().resolve()
    if not source_run_root.is_dir():
        raise CertificateError(f"source run root does not exist: {source_run_root}")
    try:
        summary_path.relative_to(source_run_root)
    except ValueError as error:
        raise CertificateError("summary must be inside source run root") from error

    sentinel_path = source_run_root / ".gcicy-experiment-root"
    plan_lock_path = source_run_root / ".workflow" / "plan.lock.json"
    state_path = source_run_root / ".workflow" / "jobs" / f"{args.job_id}.json"
    sentinel, sentinel_sha256 = _read_json(sentinel_path, role="run-root sentinel")
    plan_lock, plan_lock_sha256 = _read_json(plan_lock_path, role="plan lock")
    state, state_sha256 = _read_json(state_path, role="preflight job state")
    summary, summary_sha256 = _read_json(summary_path, role="preflight summary")
    manifest_sha256 = sha256_file(expected_manifest)

    for payload, label in ((sentinel, "sentinel"), (plan_lock, "plan lock"), (state, "job state")):
        if payload.get("campaign_id") != args.source_campaign_id:
            raise CertificateError(f"{label} has the wrong campaign_id")
    plan_sha256 = sentinel.get("plan_sha256")
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise CertificateError("sentinel plan_sha256 is missing")
    if plan_lock.get("plan_sha256") != plan_sha256 or state.get("plan_sha256") != plan_sha256:
        raise CertificateError("sentinel, plan lock, and job state plan hashes disagree")
    if sentinel.get("manifest_sha256") != manifest_sha256:
        raise CertificateError("sentinel manifest hash does not match expected manifest")
    if plan_lock.get("manifest_sha256") != manifest_sha256:
        raise CertificateError("plan-lock manifest hash does not match expected manifest")
    if state.get("job_id") != args.job_id or state.get("status") != "succeeded":
        raise CertificateError("registered preflight job did not succeed")
    attempts = state.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise CertificateError("preflight job state has no attempts")
    exit_code = attempts[-1].get("returncode")
    if exit_code != 0:
        raise CertificateError(f"preflight exit code is {exit_code!r}, expected 0")
    gate_rows = state.get("json_gates")
    if not isinstance(gate_rows, list) or not gate_rows:
        raise CertificateError("preflight job has no recorded JSON gates")
    if not all(isinstance(row, dict) and row.get("passed") is True for row in gate_rows):
        raise CertificateError("not all recorded preflight JSON gates passed")
    output_rows = state.get("outputs")
    if not isinstance(output_rows, list):
        raise CertificateError("preflight job output records are missing")
    matching_outputs = [
        row
        for row in output_rows
        if isinstance(row, dict) and Path(str(row.get("path"))).resolve() == summary_path
    ]
    if len(matching_outputs) != 1 or matching_outputs[0].get("sha256") != summary_sha256:
        raise CertificateError("job-state summary output hash does not match the file")

    _finite_tree(summary)
    expected_fields = {
        "schema": "type11-positive-tensor-network-training-v1",
        "termination_reason": "completed_requested_epochs",
        "site_count": args.site_count,
        "bond_dimension": args.bond_dimension,
        "trainable_real_parameter_count": args.parameter_count,
        "precision": "complex64",
        "physical_dictionary_rank": 121,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "positive_floor": args.positive_floor,
        "epochs": 1,
        "batch_size": 1024,
        "train.points": 196608,
        "train.seed": 86201,
        "train.common_pool": str(args.train_pool.expanduser().resolve()),
        "train.common_pool_sha256": args.train_pool_sha256,
        "validation.points": 24576,
        "validation.seed": 86202,
        "validation.common_pool": str(args.selection_pool.expanduser().resolve()),
        "validation.common_pool_sha256": args.selection_pool_sha256,
    }
    for field, expected in expected_fields.items():
        _require_equal(summary, field, expected)

    allocated = _number(summary, "device_memory.maximum_allocated_bytes")
    reserved = _number(summary, "device_memory.maximum_reserved_bytes")
    runtime = _number(summary, "runtime_seconds")
    minimum_eigenvalue = _number(summary, "best_validation.minimum_metric_eigenvalue")
    if allocated <= 0.0 or allocated > args.allocated_max_bytes:
        raise CertificateError("allocated memory is outside the registered gate")
    if reserved < allocated or reserved > args.reserved_max_bytes:
        raise CertificateError("reserved memory is outside the registered gate")
    if runtime <= 0.0:
        raise CertificateError("runtime_seconds must be positive")
    if minimum_eigenvalue <= 0.0:
        raise CertificateError("preflight validation metric is not strictly positive")

    certifier = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "status": "passed",
        "source_campaign_id": args.source_campaign_id,
        "source_run_root": str(source_run_root),
        "source_plan_sha256": plan_sha256,
        "source_manifest": str(expected_manifest),
        "source_manifest_sha256": manifest_sha256,
        "job_id": args.job_id,
        "exit_code": exit_code,
        "all_recorded_json_gates_passed": True,
        "all_summary_numbers_finite": True,
        "strictly_positive_validation_metric": True,
        "site_count": args.site_count,
        "bond_dimension": args.bond_dimension,
        "trainable_real_parameter_count": args.parameter_count,
        "precision": "complex64",
        "physical_dictionary_rank": 121,
        "positive_floor": args.positive_floor,
        "maximum_allocated_bytes": int(allocated),
        "maximum_reserved_bytes": int(reserved),
        "runtime_seconds": runtime,
        "minimum_metric_eigenvalue": minimum_eigenvalue,
        "limits": {
            "maximum_allocated_bytes": args.allocated_max_bytes,
            "maximum_reserved_bytes": args.reserved_max_bytes,
        },
        "inputs": {
            "sentinel": {"path": str(sentinel_path), "sha256": sentinel_sha256},
            "plan_lock": {"path": str(plan_lock_path), "sha256": plan_lock_sha256},
            "job_state": {"path": str(state_path), "sha256": state_sha256},
            "summary": {"path": str(summary_path), "sha256": summary_sha256},
        },
        "certifier": {"path": str(certifier), "sha256": sha256_file(certifier)},
        "create_only": True,
    }


def write_create_only(path: Path, payload: dict[str, Any]) -> str:
    """Create an immutable certificate, or verify an identical retry."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        try:
            existing_bytes = resolved.read_bytes()
            existing = json.loads(existing_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CertificateError(
                f"existing certificate is unreadable and cannot be replaced: {resolved}"
            ) from error
        try:
            identical = canonical_json_bytes(existing) == canonical_json_bytes(payload)
        except (TypeError, ValueError) as error:
            raise CertificateError(
                f"existing certificate is not canonical finite JSON: {resolved}"
            ) from error
        if not identical:
            raise CertificateError(
                f"existing certificate differs and cannot be replaced: {resolved}"
            )
        return hashlib.sha256(existing_bytes).hexdigest()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--source-campaign-id", required=True)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--site-count", type=int, required=True)
    parser.add_argument("--bond-dimension", type=int, required=True)
    parser.add_argument("--parameter-count", type=int, required=True)
    parser.add_argument("--positive-floor", type=float, required=True)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--train-pool-sha256", required=True)
    parser.add_argument("--selection-pool", type=Path, required=True)
    parser.add_argument("--selection-pool-sha256", required=True)
    parser.add_argument("--allocated-max-bytes", type=int, required=True)
    parser.add_argument("--reserved-max-bytes", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for field in (
        "site_count",
        "bond_dimension",
        "parameter_count",
        "allocated_max_bytes",
        "reserved_max_bytes",
    ):
        if getattr(args, field) <= 0:
            build_parser().error(f"--{field.replace('_', '-')} must be positive")
    if args.reserved_max_bytes < args.allocated_max_bytes:
        build_parser().error("reserved limit must be at least the allocated limit")
    if not math.isfinite(args.positive_floor) or args.positive_floor <= 0.0:
        build_parser().error("--positive-floor must be positive and finite")
    for field in ("train_pool_sha256", "selection_pool_sha256"):
        value = getattr(args, field).lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            build_parser().error(f"--{field.replace('_', '-')} must be a SHA-256 digest")
        setattr(args, field, value)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output = args.out.expanduser().resolve()
        payload = build_certificate(args)
        digest = write_create_only(output, payload)
    except CertificateError as error:
        print(f"[x21-resource-preflight] error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"out": str(output), "sha256": digest, "status": "passed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
