from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cast_positive_tensor_network_precision.py"
SPEC = importlib.util.spec_from_file_location("cast_positive_tn_precision", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def artifact(precision: str = "complex64") -> dict:
    complex_dtype, real_dtype = MODULE._expected_dtypes(precision)
    return {
        "schema": "quintic-positive-tensor-network-v1",
        "precision": precision,
        "site_count": 4,
        "bond_dimension": 2,
        "state_dict": {
            "complex": torch.tensor(
                [[1.25 + 2.5j, -0.75 + 0.125j]], dtype=complex_dtype
            ),
            "real": torch.tensor([0.25, -3.5], dtype=real_dtype),
            "integer": torch.tensor([1, 2], dtype=torch.int64),
        },
    }


def test_complex64_to_complex128_is_exact_binary_promotion() -> None:
    source = artifact()
    converted, exact, maximum_difference = MODULE.cast_payload(
        source, target_precision="complex128"
    )
    assert exact is True
    assert maximum_difference == 0.0
    assert converted["precision"] == "complex128"
    assert converted["state_dict"]["complex"].dtype == torch.complex128
    assert converted["state_dict"]["real"].dtype == torch.float64
    assert torch.equal(
        converted["state_dict"]["complex"].to(torch.complex64),
        source["state_dict"]["complex"],
    )
    assert torch.equal(
        converted["state_dict"]["integer"], source["state_dict"]["integer"]
    )


def test_embedded_lineage_is_location_independent() -> None:
    provenance = MODULE.embedded_cast_provenance(
        source_hash="a" * 64,
        source_precision="complex64",
        target_precision="complex128",
        exact_promotion=True,
        maximum_roundtrip_difference=0.0,
    )
    assert provenance["source_model_sha256"] == "a" * 64
    assert "source_model" not in provenance


def test_cast_model_hash_is_independent_of_source_directory(tmp_path: Path) -> None:
    output_hashes = []
    for directory_name in ("host-a", "host-b"):
        directory = tmp_path / directory_name
        directory.mkdir()
        source_path = directory / "source.pt"
        output_path = directory / "model.pt"
        torch.save(artifact(), source_path)
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        converted, exact, difference = MODULE.cast_payload(
            payload, target_precision="complex128"
        )
        converted["precision_cast"] = MODULE.embedded_cast_provenance(
            source_hash=MODULE.sha256_file(source_path),
            source_precision="complex64",
            target_precision="complex128",
            exact_promotion=exact,
            maximum_roundtrip_difference=difference,
        )
        MODULE._atomic_torch_save(converted, output_path)
        output_hashes.append(MODULE.sha256_file(output_path))
    assert output_hashes[0] == output_hashes[1]


def test_rejects_same_precision() -> None:
    with pytest.raises(ValueError, match="must be different"):
        MODULE.cast_payload(artifact(), target_precision="complex64")


def test_rejects_tensor_dtype_metadata_mismatch() -> None:
    payload = artifact()
    payload["state_dict"]["complex"] = payload["state_dict"]["complex"].to(
        torch.complex128
    )
    with pytest.raises(ValueError, match="disagrees"):
        MODULE.validate_source_payload(payload)


def test_rejects_unknown_schema() -> None:
    payload = artifact()
    payload["schema"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        MODULE.validate_source_payload(payload)
