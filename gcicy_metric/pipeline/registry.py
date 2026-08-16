"""Registry for built-in and downstream gCICY adapters."""

from __future__ import annotations

from collections.abc import Callable

from .adapter import GCICYAdapter
from .adapters import (
    P1P1P5Type22Adapter,
    P4P1HirzebruchM4Type11Adapter,
    P4P1HirzebruchType11Adapter,
    P4P1P1HirzebruchType21Adapter,
    P5P1K3Type21Adapter,
)


AdapterFactory = Callable[[], GCICYAdapter]


_REGISTRY: dict[str, AdapterFactory] = {
    P1P1P5Type22Adapter.key: P1P1P5Type22Adapter,
    P4P1HirzebruchM4Type11Adapter.key: P4P1HirzebruchM4Type11Adapter,
    P4P1HirzebruchType11Adapter.key: P4P1HirzebruchType11Adapter,
    P4P1P1HirzebruchType21Adapter.key: P4P1P1HirzebruchType21Adapter,
    P5P1K3Type21Adapter.key: P5P1K3Type21Adapter,
}


def register_adapter(key: str, factory: AdapterFactory) -> None:
    """Register a configuration adapter without modifying the pipeline core."""

    if not key:
        raise ValueError("adapter key must be non-empty")
    if key in _REGISTRY:
        raise ValueError(f"adapter is already registered: {key}")
    adapter = factory()
    if adapter.key != key:
        raise ValueError("registry key does not match adapter.key")
    _REGISTRY[key] = factory


def available_adapters() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_adapter(key: str) -> GCICYAdapter:
    try:
        adapter = _REGISTRY[key]()
    except KeyError as exc:
        choices = ", ".join(available_adapters())
        raise KeyError(f"unknown adapter {key!r}; available adapters: {choices}") from exc
    if adapter.configuration.key != key:
        raise ValueError("adapter and configuration keys are inconsistent")
    return adapter
