"""Restricted loading for tensor artifacts produced by this repository.

PyTorch artifacts are pickle containers. Scientific workers must therefore
never fall back to the general-purpose unpickler merely because an older
artifact contains NumPy arrays. The historical X21 and quintic artifacts use
only NumPy arrays with int64 or complex128 dtype metadata in addition to the
types accepted by PyTorch's weights-only loader. This module statically
rejects every other external global before loading and then uses that minimal
allowlist with ``weights_only=True``.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any


class UnsafeTensorArtifactError(RuntimeError):
    """Raised when a tensor artifact requests an unregistered pickle global."""


_ALLOWED_EXTERNAL_GLOBALS = frozenset(
    {
        "numpy.core.multiarray._reconstruct",
        "numpy._core.multiarray._reconstruct",
        "numpy.dtype",
        "numpy.ndarray",
    }
)


def safe_torch_load(path: str | Path, *, map_location: Any = "cpu") -> Any:
    """Load a repository tensor artifact without enabling arbitrary pickle code."""

    import numpy as np
    import torch

    resolved = Path(path).expanduser().resolve()
    try:
        observed_globals = set(
            torch.serialization.get_unsafe_globals_in_checkpoint(resolved)
        )
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        ValueError,
    ) as error:
        raise UnsafeTensorArtifactError(
            "tensor artifact cannot be inspected safely"
        ) from error
    unexpected = observed_globals - _ALLOWED_EXTERNAL_GLOBALS
    if unexpected:
        raise UnsafeTensorArtifactError(
            "tensor artifact requests unregistered globals: "
            + ", ".join(sorted(unexpected))
        )
    allowed_types = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.int64)),
        type(np.dtype(np.complex128)),
    ]
    try:
        with torch.serialization.safe_globals(allowed_types):
            return torch.load(resolved, map_location=map_location, weights_only=True)
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        ValueError,
    ) as error:
        raise UnsafeTensorArtifactError(
            "tensor artifact failed restricted loading"
        ) from error
