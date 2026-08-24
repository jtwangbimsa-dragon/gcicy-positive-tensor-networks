"""Content-addressed host-health gate for scientific GPU experiments.

The gate is deliberately separate from model evaluation.  It converts
read-only kernel-log evidence, repeated fresh-process import/checkpoint probes,
and three short GPU probe receipts into either a short-lived scientific
authorization or a ``diagnostic_only`` certificate.  A failed gate never makes
an accuracy judgement and cannot be used as recovery or leaderboard evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CERTIFICATE_SCHEMA = "gcicy-host-stability-certificate-v1"
GPU_PROBE_SCHEMA = "gcicy-host-gpu-probe-v1"
DEFAULT_QUIET_WINDOW_HOURS = 6
DEFAULT_FRESH_PROCESS_PROBES = 20
DEFAULT_GPU_PROBE_COUNT = 3
DEFAULT_MINIMUM_GPU_STEPS = 10
DEFAULT_MAXIMUM_GPU_BYTES = 18 * 1024**3
DEFAULT_VALIDITY_SECONDS = 30 * 60

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_KERNEL_EVENT = re.compile(
    r"(?:segfault|general protection|out of memory|oom-kill|killed process|"
    r"\bmce\b|\bbert\b|\bedac\b|nvrm:.*\bxid\b)",
    re.IGNORECASE,
)


class HostStabilityError(RuntimeError):
    """Raised when a host-health artifact is malformed or has drifted."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise HostStabilityError(
            f"{context} keys are invalid; missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise HostStabilityError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise HostStabilityError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise HostStabilityError(f"{context} must be finite")
    return result


def parse_kernel_events(text: str) -> list[str]:
    """Return matching lines without retaining unrelated kernel messages."""

    return [line.strip() for line in text.splitlines() if _KERNEL_EVENT.search(line)]


def _parse_utc(value: Any, *, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise HostStabilityError(f"{context} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise HostStabilityError(f"{context} must include a timezone")
    return parsed.astimezone(timezone.utc)


def normalize_gpu_probe(
    value: Any,
    *,
    index: int,
    expected_checkpoint_sha256: str,
    expected_host_identity_sha256: str,
    expected_source_commit: str,
    issued_at: datetime,
) -> dict[str, Any]:
    context = f"gpu_probes[{index}]"
    if not isinstance(value, dict):
        raise HostStabilityError(f"{context} must be an object")
    _exact_keys(
        value,
        required={
            "schema",
            "probe_id",
            "checkpoint_sha256",
            "host_identity_sha256",
            "source_commit",
            "started_utc",
            "finished_utc",
            "report_path",
            "report_sha256",
            "optimizer_steps",
            "selection_evaluation_completed",
            "all_finite",
            "minimum_metric_eigenvalue",
            "maximum_allocated_bytes",
        },
        context=context,
    )
    if value["schema"] != GPU_PROBE_SCHEMA:
        raise HostStabilityError(f"{context} schema is invalid")
    probe_id = str(value["probe_id"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", probe_id):
        raise HostStabilityError(f"{context}.probe_id is invalid")
    steps = value["optimizer_steps"]
    memory = value["maximum_allocated_bytes"]
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or isinstance(memory, bool)
        or not isinstance(memory, int)
        or steps < 0
        or memory < 0
    ):
        raise HostStabilityError(f"{context} integer fields are invalid")
    if not isinstance(value["selection_evaluation_completed"], bool) or not isinstance(
        value["all_finite"], bool
    ):
        raise HostStabilityError(f"{context} boolean fields are invalid")
    minimum = _finite(
        value["minimum_metric_eigenvalue"],
        context=f"{context}.minimum_metric_eigenvalue",
    )
    checkpoint_sha256 = str(value["checkpoint_sha256"])
    host_identity_sha256 = str(value["host_identity_sha256"])
    source_commit = str(value["source_commit"])
    if (
        checkpoint_sha256 != expected_checkpoint_sha256
        or host_identity_sha256 != expected_host_identity_sha256
        or source_commit != expected_source_commit
    ):
        raise HostStabilityError(f"{context} identity binding is invalid")
    report_path = Path(str(value["report_path"])).expanduser().resolve()
    report_sha256 = str(value["report_sha256"])
    if (
        not report_path.is_file()
        or not _SHA256.fullmatch(report_sha256)
        or sha256_file(report_path) != report_sha256
    ):
        raise HostStabilityError(f"{context} report identity is invalid")
    started = _parse_utc(value["started_utc"], context=f"{context}.started_utc")
    finished = _parse_utc(value["finished_utc"], context=f"{context}.finished_utc")
    issued = issued_at.astimezone(timezone.utc)
    if (
        not issued - timedelta(hours=DEFAULT_QUIET_WINDOW_HOURS)
        <= started
        <= finished
        <= issued
    ):
        raise HostStabilityError(f"{context} lies outside the certified time window")
    passes = (
        steps >= DEFAULT_MINIMUM_GPU_STEPS
        and value["selection_evaluation_completed"]
        and value["all_finite"]
        and minimum > 0.0
        and memory < DEFAULT_MAXIMUM_GPU_BYTES
    )
    return {
        "schema": GPU_PROBE_SCHEMA,
        "probe_id": probe_id,
        "checkpoint_sha256": checkpoint_sha256,
        "host_identity_sha256": host_identity_sha256,
        "source_commit": source_commit,
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "finished_utc": finished.isoformat().replace("+00:00", "Z"),
        "report_path": str(report_path),
        "report_sha256": report_sha256,
        "optimizer_steps": steps,
        "selection_evaluation_completed": value["selection_evaluation_completed"],
        "all_finite": value["all_finite"],
        "minimum_metric_eigenvalue": minimum,
        "maximum_allocated_bytes": memory,
        "passes": passes,
    }


def build_certificate(
    *,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    host_identity_sha256: str,
    source_commit: str,
    kernel_log: str,
    fresh_process_results: Sequence[Mapping[str, Any]],
    gpu_probe_values: Sequence[Mapping[str, Any]],
    quiet_window_hours: int = DEFAULT_QUIET_WINDOW_HOURS,
    issued_at: datetime | None = None,
    validity_seconds: int = DEFAULT_VALIDITY_SECONDS,
) -> dict[str, Any]:
    """Build a deterministic verdict from already-collected read-only evidence."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise HostStabilityError("checkpoint is missing")
    if not _SHA256.fullmatch(expected_checkpoint_sha256):
        raise HostStabilityError("expected checkpoint hash is malformed")
    observed_checkpoint_sha256 = sha256_file(checkpoint_path)
    if observed_checkpoint_sha256 != expected_checkpoint_sha256:
        raise HostStabilityError("checkpoint hash does not match")
    if quiet_window_hours != DEFAULT_QUIET_WINDOW_HOURS:
        raise HostStabilityError("the scientific quiet window is fixed at six hours")
    if validity_seconds != DEFAULT_VALIDITY_SECONDS:
        raise HostStabilityError("certificate validity is fixed at thirty minutes")
    if not _SHA256.fullmatch(host_identity_sha256):
        raise HostStabilityError("host identity hash is malformed")
    if not _GIT_COMMIT.fullmatch(source_commit):
        raise HostStabilityError("source commit is malformed")

    issued = issued_at or datetime.now(timezone.utc)
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=timezone.utc)
    issued = issued.astimezone(timezone.utc).replace(microsecond=0)

    process_rows = []
    for index, row in enumerate(fresh_process_results):
        if not isinstance(row, Mapping):
            raise HostStabilityError("fresh-process result must be an object")
        _exact_keys(
            row,
            required={
                "attempt",
                "returncode",
                "stdout_sha256",
                "checkpoint_sha256",
                "host_identity_sha256",
                "source_commit",
            },
            context=f"fresh_process_results[{index}]",
        )
        attempt = row["attempt"]
        returncode = row["returncode"]
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt != index + 1
            or isinstance(returncode, bool)
            or not isinstance(returncode, int)
        ):
            raise HostStabilityError("fresh-process result sequence is malformed")
        stdout_sha256 = str(row["stdout_sha256"])
        if not _SHA256.fullmatch(stdout_sha256):
            raise HostStabilityError("fresh-process stdout hash is malformed")
        if (
            row["checkpoint_sha256"] != observed_checkpoint_sha256
            or row["host_identity_sha256"] != host_identity_sha256
            or row["source_commit"] != source_commit
        ):
            raise HostStabilityError("fresh-process identity binding is invalid")
        process_rows.append(
            {
                "attempt": attempt,
                "returncode": returncode,
                "stdout_sha256": stdout_sha256,
                "checkpoint_sha256": observed_checkpoint_sha256,
                "host_identity_sha256": host_identity_sha256,
                "source_commit": source_commit,
                "passes": returncode == 0,
            }
        )

    probes = [
        normalize_gpu_probe(
            value,
            index=index,
            expected_checkpoint_sha256=observed_checkpoint_sha256,
            expected_host_identity_sha256=host_identity_sha256,
            expected_source_commit=source_commit,
            issued_at=issued,
        )
        for index, value in enumerate(gpu_probe_values)
    ]
    distinct_probes = len({row["probe_id"] for row in probes}) == len(probes)
    kernel_events = parse_kernel_events(kernel_log)
    gates = {
        "checkpoint_identity": True,
        "kernel_quiet_window": not kernel_events,
        "fresh_process_probe_count": len(process_rows) == DEFAULT_FRESH_PROCESS_PROBES,
        "fresh_process_probes": bool(process_rows)
        and all(row["passes"] for row in process_rows),
        "gpu_probe_count": len(probes) == DEFAULT_GPU_PROBE_COUNT,
        "gpu_probe_identity": distinct_probes,
        "gpu_probes": bool(probes) and all(row["passes"] for row in probes),
    }
    authorized = all(gates.values())
    valid_until = issued + timedelta(seconds=validity_seconds)
    payload = {
        "schema": CERTIFICATE_SCHEMA,
        "issued_utc": issued.isoformat().replace("+00:00", "Z"),
        "valid_until_utc": valid_until.isoformat().replace("+00:00", "Z"),
        "quiet_window_hours": quiet_window_hours,
        "host_identity_sha256": host_identity_sha256,
        "source_commit": source_commit,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": observed_checkpoint_sha256,
            "bytes": checkpoint_path.stat().st_size,
        },
        "kernel_events": kernel_events,
        "kernel_log_sha256": hashlib.sha256(kernel_log.encode("utf-8")).hexdigest(),
        "fresh_process_results": process_rows,
        "gpu_probes": probes,
        "gates": gates,
        "scientific_authorized": authorized,
        "scope": "scientific" if authorized else "diagnostic_only",
    }
    return {**payload, "certificate_sha256": digest_value(payload)}


def publish_certificate(path: Path, certificate: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(certificate), indent=2, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(certificate):
            raise HostStabilityError(
                "certificate path already contains another verdict"
            )
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def validate_certificate(
    value: Mapping[str, Any],
    *,
    expected_checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    expected_host_identity_sha256: str,
    expected_source_commit: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    if (
        not _SHA256.fullmatch(expected_checkpoint_sha256)
        or not _SHA256.fullmatch(expected_host_identity_sha256)
        or not _GIT_COMMIT.fullmatch(expected_source_commit)
    ):
        raise HostStabilityError("certificate validation identity is malformed")
    _exact_keys(
        value,
        required={
            "schema",
            "issued_utc",
            "valid_until_utc",
            "quiet_window_hours",
            "host_identity_sha256",
            "source_commit",
            "checkpoint",
            "kernel_events",
            "kernel_log_sha256",
            "fresh_process_results",
            "gpu_probes",
            "gates",
            "scientific_authorized",
            "scope",
            "certificate_sha256",
        },
        context="certificate",
    )
    if value["schema"] != CERTIFICATE_SCHEMA:
        raise HostStabilityError("certificate schema is invalid")
    observed = value.get("certificate_sha256")
    payload = {key: item for key, item in value.items() if key != "certificate_sha256"}
    if not isinstance(observed, str) or observed != digest_value(payload):
        raise HostStabilityError("certificate hash is invalid")
    if value["quiet_window_hours"] != DEFAULT_QUIET_WINDOW_HOURS:
        raise HostStabilityError("certificate quiet window is invalid")
    if (
        value["host_identity_sha256"] != expected_host_identity_sha256
        or value["source_commit"] != expected_source_commit
    ):
        raise HostStabilityError("certificate host/source binding is invalid")
    issued = _parse_utc(value["issued_utc"], context="certificate.issued_utc")
    valid_until = _parse_utc(
        value["valid_until_utc"], context="certificate.valid_until_utc"
    )
    if valid_until - issued != timedelta(seconds=DEFAULT_VALIDITY_SECONDS):
        raise HostStabilityError("certificate validity interval is invalid")
    current = now or datetime.now(timezone.utc)
    current = current.astimezone(timezone.utc)
    if current < issued or current > valid_until:
        raise HostStabilityError("certificate has expired")

    checkpoint = value["checkpoint"]
    if not isinstance(checkpoint, Mapping):
        raise HostStabilityError("certificate checkpoint is malformed")
    _exact_keys(
        checkpoint,
        required={"path", "sha256", "bytes"},
        context="certificate.checkpoint",
    )
    checkpoint_path = expected_checkpoint_path.expanduser().resolve()
    if (
        str(checkpoint_path) != checkpoint["path"]
        or checkpoint["sha256"] != expected_checkpoint_sha256
        or not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size != checkpoint["bytes"]
        or sha256_file(checkpoint_path) != expected_checkpoint_sha256
    ):
        raise HostStabilityError("certificate checkpoint has drifted")

    if (
        not isinstance(value["kernel_events"], list)
        or any(not isinstance(row, str) for row in value["kernel_events"])
        or not _SHA256.fullmatch(str(value["kernel_log_sha256"]))
    ):
        raise HostStabilityError("certificate kernel events are malformed")
    process_rows = value["fresh_process_results"]
    if not isinstance(process_rows, list):
        raise HostStabilityError("certificate fresh-process evidence is malformed")
    normalized_process_rows = []
    for index, row in enumerate(process_rows):
        if not isinstance(row, Mapping):
            raise HostStabilityError("certificate fresh-process row is malformed")
        _exact_keys(
            row,
            required={
                "attempt",
                "returncode",
                "stdout_sha256",
                "checkpoint_sha256",
                "host_identity_sha256",
                "source_commit",
                "passes",
            },
            context=f"certificate.fresh_process_results[{index}]",
        )
        expected_row = {
            "attempt": index + 1,
            "returncode": row["returncode"],
            "stdout_sha256": row["stdout_sha256"],
            "checkpoint_sha256": expected_checkpoint_sha256,
            "host_identity_sha256": expected_host_identity_sha256,
            "source_commit": expected_source_commit,
            "passes": row["returncode"] == 0,
        }
        if (
            isinstance(row["attempt"], bool)
            or not isinstance(row["attempt"], int)
            or isinstance(row["returncode"], bool)
            or not isinstance(row["returncode"], int)
            or not isinstance(row["passes"], bool)
            or row != expected_row
            or not _SHA256.fullmatch(str(row["stdout_sha256"]))
        ):
            raise HostStabilityError("certificate fresh-process row is invalid")
        normalized_process_rows.append(expected_row)

    raw_probes = value["gpu_probes"]
    if not isinstance(raw_probes, list):
        raise HostStabilityError("certificate GPU evidence is malformed")
    probes = []
    for index, row in enumerate(raw_probes):
        if not isinstance(row, Mapping) or not isinstance(row.get("passes"), bool):
            raise HostStabilityError("certificate GPU evidence row is malformed")
        probes.append(
            normalize_gpu_probe(
                {key: item for key, item in row.items() if key != "passes"},
                index=index,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_host_identity_sha256=expected_host_identity_sha256,
                expected_source_commit=expected_source_commit,
                issued_at=issued,
            )
        )
    if probes != raw_probes:
        raise HostStabilityError("certificate GPU evidence was not normalized")
    gates = {
        "checkpoint_identity": True,
        "kernel_quiet_window": not value["kernel_events"],
        "fresh_process_probe_count": len(normalized_process_rows)
        == DEFAULT_FRESH_PROCESS_PROBES,
        "fresh_process_probes": bool(normalized_process_rows)
        and all(row["passes"] for row in normalized_process_rows),
        "gpu_probe_count": len(probes) == DEFAULT_GPU_PROBE_COUNT,
        "gpu_probe_identity": len({row["probe_id"] for row in probes}) == len(probes),
        "gpu_probes": bool(probes) and all(row["passes"] for row in probes),
    }
    authorized = all(gates.values())
    if value["gates"] != gates:
        raise HostStabilityError("certificate gates do not match its evidence")
    if value["scientific_authorized"] is not authorized or value["scope"] != (
        "scientific" if authorized else "diagnostic_only"
    ):
        raise HostStabilityError("certificate verdict does not match its gates")
    if not authorized:
        raise HostStabilityError("certificate authorizes diagnostic work only")
    return dict(value)


__all__ = [
    "CERTIFICATE_SCHEMA",
    "GPU_PROBE_SCHEMA",
    "HostStabilityError",
    "build_certificate",
    "digest_value",
    "normalize_gpu_probe",
    "parse_kernel_events",
    "publish_certificate",
    "validate_certificate",
]
