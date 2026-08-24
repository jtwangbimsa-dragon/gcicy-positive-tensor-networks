from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gcicy_metric.pipeline.safe_torch_load import (
    UnsafeTensorArtifactError,
    safe_torch_load,
)


class UnregisteredPayload:
    pass


def test_safe_loader_accepts_registered_numpy_metadata(tmp_path: Path) -> None:
    path = tmp_path / "scientific-artifact.pt"
    torch.save(
        {
            "indices": np.asarray([1, 2, 3], dtype=np.int64),
            "whitening": np.eye(2, dtype=np.complex128),
            "state_dict": {"core": torch.ones(2, dtype=torch.complex64)},
        },
        path,
    )
    payload = safe_torch_load(path)
    assert np.array_equal(payload["indices"], np.asarray([1, 2, 3]))
    assert payload["whitening"].dtype == np.complex128
    assert payload["state_dict"]["core"].dtype == torch.complex64


def test_safe_loader_rejects_unregistered_pickle_global(tmp_path: Path) -> None:
    path = tmp_path / "unsafe-artifact.pt"
    torch.save({"payload": UnregisteredPayload()}, path)
    with pytest.raises(UnsafeTensorArtifactError, match="UnregisteredPayload"):
        safe_torch_load(path)


def test_safe_loader_rejects_corrupt_container(tmp_path: Path) -> None:
    path = tmp_path / "corrupt-artifact.pt"
    path.write_bytes(b"not a torch artifact")
    with pytest.raises(UnsafeTensorArtifactError, match="inspected safely"):
        safe_torch_load(path)


def test_safe_loader_never_falls_back_to_general_unpickler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "plain-artifact.pt"
    torch.save({"tensor": torch.ones(1)}, path)
    observed = []
    original = torch.load

    def checked_load(*args, **kwargs):
        observed.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    safe_torch_load(path)
    assert observed == [True]
