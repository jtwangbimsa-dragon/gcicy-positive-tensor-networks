from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

import pytest

from gcicy_metric.pipeline import host_stability_gate as gate


HOST_IDENTITY = hashlib.sha256(b"test-host").hexdigest()
SOURCE_COMMIT = "a" * 40
ISSUED = datetime(2026, 8, 24, tzinfo=timezone.utc)


def process_rows(count: int = 20):
    return [
        {
            "attempt": index,
            "returncode": 0,
            "stdout_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "checkpoint_sha256": None,
            "host_identity_sha256": HOST_IDENTITY,
            "source_commit": SOURCE_COMMIT,
        }
        for index in range(1, count + 1)
    ]


def probes(tmp_path, checkpoint_sha256: str, count: int = 3):
    rows = []
    for index in range(1, count + 1):
        report = tmp_path / f"gpu-probe-{index}.json"
        report.write_text(f"probe {index}\n", encoding="utf-8")
        rows.append(
            {
                "schema": gate.GPU_PROBE_SCHEMA,
                "probe_id": f"probe-{index}",
                "checkpoint_sha256": checkpoint_sha256,
                "host_identity_sha256": HOST_IDENTITY,
                "source_commit": SOURCE_COMMIT,
                "started_utc": (ISSUED - timedelta(minutes=2)).isoformat(),
                "finished_utc": (ISSUED - timedelta(minutes=1)).isoformat(),
                "report_path": str(report),
                "report_sha256": gate.sha256_file(report),
                "optimizer_steps": 10,
                "selection_evaluation_completed": True,
                "all_finite": True,
                "minimum_metric_eigenvalue": 0.01,
                "maximum_allocated_bytes": 10 * 1024**3,
            }
        )
    return rows


def certificate(tmp_path, **overrides):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha256 = gate.sha256_file(checkpoint)
    rows = process_rows()
    for row in rows:
        row["checkpoint_sha256"] = checkpoint_sha256
    arguments = {
        "checkpoint_path": checkpoint,
        "expected_checkpoint_sha256": checkpoint_sha256,
        "host_identity_sha256": HOST_IDENTITY,
        "source_commit": SOURCE_COMMIT,
        "kernel_log": "quiet kernel\n",
        "fresh_process_results": rows,
        "gpu_probe_values": probes(tmp_path, checkpoint_sha256),
        "issued_at": ISSUED,
    }
    arguments.update(overrides)
    return gate.build_certificate(**arguments)


def validate(tmp_path, value, *, now=None):
    checkpoint = tmp_path / "checkpoint.pt"
    return gate.validate_certificate(
        value,
        expected_checkpoint_path=checkpoint,
        expected_checkpoint_sha256=gate.sha256_file(checkpoint),
        expected_host_identity_sha256=HOST_IDENTITY,
        expected_source_commit=SOURCE_COMMIT,
        now=now,
    )


def test_clean_evidence_creates_short_lived_scientific_certificate(tmp_path):
    value = certificate(tmp_path)
    assert value["scientific_authorized"] is True
    assert value["scope"] == "scientific"
    validate(tmp_path, value, now=datetime(2026, 8, 24, 0, 20, tzinfo=timezone.utc))


def test_kernel_fault_is_diagnostic_only(tmp_path):
    value = certificate(
        tmp_path,
        kernel_log="kernel: python[1]: segfault at 0\nkernel: NVRM: Xid 31\n",
    )
    assert value["scientific_authorized"] is False
    assert value["scope"] == "diagnostic_only"
    assert len(value["kernel_events"]) == 2
    with pytest.raises(gate.HostStabilityError, match="diagnostic work only"):
        validate(tmp_path, value, now=ISSUED)


def test_missing_or_bad_probes_fail_closed(tmp_path):
    missing = certificate(tmp_path, gpu_probe_values=[])
    assert missing["gates"]["gpu_probe_count"] is False
    assert missing["scientific_authorized"] is False

    bad = probes(tmp_path, missing["checkpoint"]["sha256"])
    bad[2]["minimum_metric_eigenvalue"] = 0.0
    failed = certificate(tmp_path, gpu_probe_values=bad)
    assert failed["gates"]["gpu_probes"] is False


def test_checkpoint_drift_and_duplicate_gpu_probe_ids_are_rejected(tmp_path):
    checkpoint = tmp_path / "drift.pt"
    checkpoint.write_bytes(b"new")
    with pytest.raises(gate.HostStabilityError, match="hash does not match"):
        gate.build_certificate(
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            kernel_log="",
            host_identity_sha256=HOST_IDENTITY,
            source_commit=SOURCE_COMMIT,
            fresh_process_results=[],
            gpu_probe_values=[],
        )
    base = certificate(tmp_path)
    checkpoint_sha256 = base["checkpoint"]["sha256"]
    duplicate = probes(tmp_path, checkpoint_sha256)
    duplicate[2]["probe_id"] = duplicate[1]["probe_id"]
    value = certificate(tmp_path, gpu_probe_values=duplicate)
    assert value["gates"]["gpu_probe_identity"] is False
    assert value["scientific_authorized"] is False


def test_tampering_and_expiry_are_rejected(tmp_path):
    value = certificate(tmp_path)
    tampered = dict(value)
    tampered["scope"] = "diagnostic_only"
    with pytest.raises(gate.HostStabilityError, match="hash is invalid"):
        validate(tmp_path, tampered, now=ISSUED)
    with pytest.raises(gate.HostStabilityError, match="expired"):
        validate(
            tmp_path,
            value,
            now=datetime(2026, 8, 24, tzinfo=timezone.utc) + timedelta(hours=1),
        )


def test_certificate_rejects_short_window_and_checkpoint_or_report_drift(tmp_path):
    with pytest.raises(gate.HostStabilityError, match="fixed at six hours"):
        certificate(tmp_path, quiet_window_hours=1)

    value = certificate(tmp_path)
    (tmp_path / "checkpoint.pt").write_bytes(b"drifted")
    with pytest.raises(gate.HostStabilityError, match="checkpoint has drifted"):
        gate.validate_certificate(
            value,
            expected_checkpoint_path=tmp_path / "checkpoint.pt",
            expected_checkpoint_sha256=value["checkpoint"]["sha256"],
            expected_host_identity_sha256=HOST_IDENTITY,
            expected_source_commit=SOURCE_COMMIT,
            now=ISSUED,
        )

    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint")
    Path(value["gpu_probes"][0]["report_path"]).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(gate.HostStabilityError, match="report identity"):
        validate(tmp_path, value, now=ISSUED)


def test_minimal_self_hashed_certificate_is_rejected(tmp_path):
    payload = {
        "schema": gate.CERTIFICATE_SCHEMA,
        "scientific_authorized": True,
        "scope": "scientific",
        "valid_until_utc": "2026-08-24T00:30:00Z",
    }
    forged = {**payload, "certificate_sha256": gate.digest_value(payload)}
    with pytest.raises(gate.HostStabilityError, match="keys are invalid"):
        gate.validate_certificate(
            forged,
            expected_checkpoint_path=tmp_path / "checkpoint.pt",
            expected_checkpoint_sha256="0" * 64,
            expected_host_identity_sha256=HOST_IDENTITY,
            expected_source_commit=SOURCE_COMMIT,
            now=ISSUED,
        )
