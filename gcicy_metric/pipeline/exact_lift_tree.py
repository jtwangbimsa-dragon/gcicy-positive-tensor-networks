"""Exact positive tensor-tree initializations from low-degree H metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .positive_tensor_network import (
    PositiveTensorNetworkMetric,
    positive_tensor_network_from_artifact_payload,
)


EXACT_LIFT_TREE_SCHEMA = "positive-tensor-network-exact-power-lift-v1"
NATIVE_POWER_LIFT_TREE_SCHEMA = "positive-tensor-network-native-power-lift-v1"


def _positive_hermitian(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("source H must be square")
    value = 0.5 * (value + value.conj().T)
    eigenvalues = np.linalg.eigvalsh(value)
    if (
        not np.all(np.isfinite(eigenvalues))
        or eigenvalues[0] <= 0
    ):
        raise ValueError("source H must be finite and positive definite")
    return value


def _scaled_degree(
    source_degree: int | Sequence[int],
    power: int,
) -> int | list[int]:
    if isinstance(source_degree, (int, np.integer)):
        if int(source_degree) <= 0:
            raise ValueError("source degree must be positive")
        return int(source_degree) * power
    values = [int(value) for value in source_degree]
    if not values or any(value < 0 for value in values) or not any(values):
        raise ValueError("source multidegree must be nonzero and non-negative")
    return [power * value for value in values]


def build_exact_power_lift_tree(
    source_h: np.ndarray,
    *,
    source_normalization: float,
    power: int,
    bond_dimension: int | Sequence[int] = 1,
    positive_floor: float = 1.0e-8,
    precision: str = "complex128",
    device: Any = None,
):
    """Represent ``F_source**power`` exactly as a positive tensor network.

    The zeroth virtual channel repeats the positive square root of ``source_h``
    at every site. All other bond channels are exactly zero, so increasing
    ``bond_dimension`` preserves the source metric before native enrichment.
    """

    import torch

    reference_h = _positive_hermitian(source_h)
    if not np.isfinite(source_normalization) or source_normalization <= 0:
        raise ValueError("source normalization must be finite and positive")
    if power <= 0:
        raise ValueError("power must be positive")
    if isinstance(bond_dimension, (int, np.integer)):
        resolved_bonds = (int(bond_dimension),) * max(power - 1, 0)
    else:
        resolved_bonds = tuple(int(value) for value in bond_dimension)
        if len(resolved_bonds) != max(power - 1, 0):
            raise ValueError("one bond dimension is required per internal edge")
    if any(value <= 0 for value in resolved_bonds):
        raise ValueError("bond dimensions must be positive")
    if (
        not np.isfinite(positive_floor)
        or positive_floor < 0
        or positive_floor >= 1
    ):
        raise ValueError("positive floor must lie in [0,1)")
    if precision not in {"complex64", "complex128"}:
        raise ValueError("precision must be complex64 or complex128")
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    model = PositiveTensorNetworkMetric(
        reference_h,
        site_count=int(power),
        bond_dimension=resolved_bonds,
        target_normalization=float(source_normalization) / float(power),
        output_dimension=len(reference_h),
        positive_floor=float(positive_floor),
        initialization_noise=0.0,
        physical_dictionary=None,
        transfer_implementation="vectorized",
        dtype=dtype,
        device=device,
    )
    if positive_floor:
        learned_core_scale = (1.0 - positive_floor) ** (
            1.0 / (2.0 * power)
        )
        with torch.no_grad():
            for core in model.cores:
                core.mul_(learned_core_scale)
    return model


def expand_exact_power_lift_bonds(
    model: Any,
    target_bond_dimensions: int | Sequence[int],
    *,
    relative_activation_scale: float,
    seed: int,
):
    """Embed a dense-core chain in larger selected bond spaces exactly."""

    import torch

    if model.architecture != "dense_local_cores":
        raise ValueError("selective bond expansion requires dense local cores")
    source = tuple(int(value) for value in model.bond_dimensions)
    if isinstance(target_bond_dimensions, (int, np.integer)):
        target = (int(target_bond_dimensions),) * len(source)
    else:
        target = tuple(int(value) for value in target_bond_dimensions)
    if (
        len(target) != len(source)
        or any(new < old for new, old in zip(target, source, strict=True))
        or not any(new > old for new, old in zip(target, source, strict=True))
    ):
        raise ValueError("target bonds must form a strict nested expansion")
    if (
        not np.isfinite(relative_activation_scale)
        or relative_activation_scale <= 0
    ):
        raise ValueError("activation scale must be finite and positive")

    dtype = model.reference_h.dtype
    device = model.reference_h.device
    expanded = PositiveTensorNetworkMetric(
        model.reference_h.detach().cpu().numpy(),
        site_count=model.site_count,
        bond_dimension=target,
        target_normalization=model.target_normalization,
        output_dimension=model.output_dimension,
        positive_floor=model.positive_floor,
        initialization_noise=0.0,
        transfer_implementation=model.transfer_implementation,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        for source_core, target_core in zip(
            model.cores,
            expanded.cores,
            strict=True,
        ):
            target_core.zero_()
            source_slices = tuple(
                slice(0, size) for size in source_core.shape
            )
            target_core[source_slices].copy_(source_core)

        rng = np.random.default_rng(seed)
        for bond, (old_dimension, new_dimension) in enumerate(
            zip(source, target, strict=True)
        ):
            if new_dimension == old_dimension:
                continue
            left_core = expanded.cores[bond]
            reachable_left = 1 if bond == 0 else source[bond - 1]
            active = left_core[
                :reachable_left,
                :old_dimension,
            ]
            new_slice = left_core[
                :reachable_left,
                old_dimension:new_dimension,
            ]
            entry_scale = (
                relative_activation_scale
                * float(torch.linalg.vector_norm(active))
                / np.sqrt(max(new_slice.numel(), 1))
            )
            noise = rng.normal(size=tuple(new_slice.shape)) + 1j * rng.normal(
                size=tuple(new_slice.shape)
            )
            new_slice.copy_(
                torch.tensor(
                    entry_scale * noise / np.sqrt(2.0),
                    dtype=dtype,
                    device=device,
                )
            )
    return expanded


def exact_power_lift_payload(
    model: Any,
    *,
    source_degree: int | Sequence[int],
    source_exponents: np.ndarray,
    source_normalization: float,
    source_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze a round-zero tree package with no teacher file dependency."""

    import torch

    if model.architecture != "dense_local_cores":
        raise ValueError("exact lift export currently requires dense local cores")
    power = int(model.site_count)
    target_degree = _scaled_degree(source_degree, power)
    expected_normalization = float(source_normalization) / float(power)
    if not np.isclose(
        float(model.target_normalization),
        expected_normalization,
        rtol=2.0e-13,
        atol=2.0e-15,
    ):
        raise ValueError("model normalization is inconsistent with the power lift")
    exponents = np.asarray(source_exponents, dtype=np.int64)
    if exponents.ndim != 2 or len(exponents) != model.section_count:
        raise ValueError("source exponents do not match the local section space")
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    reference_h = state.get("reference_h")
    if reference_h is None or reference_h.shape != (
        model.section_count,
        model.section_count,
    ):
        raise RuntimeError("exact lift state has no aligned reference H")
    precision = (
        "complex64"
        if reference_h.dtype == torch.complex64
        else "complex128"
    )
    return {
        "schema": EXACT_LIFT_TREE_SCHEMA,
        "geometry_role": "round_zero_initialization_only",
        "teacher_runtime_dependency": False,
        "state_dict": state,
        "architecture": model.architecture,
        "source_degree": (
            int(source_degree)
            if isinstance(source_degree, (int, np.integer))
            else [int(value) for value in source_degree]
        ),
        "target_degree": target_degree,
        "source_exponents": exponents,
        "source_normalization": float(source_normalization),
        "target_normalization": float(model.target_normalization),
        "power": power,
        "site_count": power,
        "bond_dimension": int(model.bond_dimension),
        "bond_dimensions": tuple(int(value) for value in model.bond_dimensions),
        "output_dimension": int(model.output_dimension),
        "positive_floor": float(model.positive_floor),
        "precision": precision,
        "transfer_implementation": model.transfer_implementation,
        "trainable_physical_dictionary": False,
        "source_artifact_sha256": source_artifact_sha256,
        "exact_identity": (
            "K_target=(source_normalization/power)*log(F_source**power)"
            "=source_normalization*log(F_source)"
        ),
    }


def exact_power_lift_from_payload(
    payload: Mapping[str, Any],
    *,
    device: Any = None,
):
    """Restore an exact lift using only its frozen initialization package."""

    if payload.get("schema") != EXACT_LIFT_TREE_SCHEMA:
        raise ValueError("unsupported exact-lift tree schema")
    if payload.get("teacher_runtime_dependency") is not False:
        raise ValueError("native initialization must not require a teacher runtime")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or "reference_h" not in state:
        raise ValueError("exact-lift package has no embedded source metric")
    reference = state["reference_h"]
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    model = positive_tensor_network_from_artifact_payload(
        np.asarray(reference, dtype=np.complex128),
        dict(payload),
        device=device,
        trainable_physical_dictionary=False,
    )
    return model


def power_lift_tree_from_payload(
    payload: Mapping[str, Any],
    *,
    device: Any = None,
):
    """Restore either the exact round-zero tree or a native continuation."""

    if payload.get("schema") not in {
        EXACT_LIFT_TREE_SCHEMA,
        NATIVE_POWER_LIFT_TREE_SCHEMA,
    }:
        raise ValueError("unsupported power-lift tree schema")
    if payload.get("teacher_runtime_dependency") is not False:
        raise ValueError("native tree must not require a teacher runtime")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or "reference_h" not in state:
        raise ValueError("power-lift tree package has no embedded source metric")
    reference = state["reference_h"]
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    return positive_tensor_network_from_artifact_payload(
        np.asarray(reference, dtype=np.complex128),
        dict(payload),
        device=device,
        trainable_physical_dictionary=False,
    )


def native_power_lift_payload(
    model: Any,
    parent_payload: Mapping[str, Any],
    *,
    native_round: Mapping[str, Any],
) -> dict[str, Any]:
    """Save a teacher-free native continuation of an exact-lift tree."""

    if parent_payload.get("schema") not in {
        EXACT_LIFT_TREE_SCHEMA,
        NATIVE_POWER_LIFT_TREE_SCHEMA,
    }:
        raise ValueError("native continuation requires a power-lift parent")
    if parent_payload.get("teacher_runtime_dependency") is not False:
        raise ValueError("native continuation cannot depend on a teacher runtime")
    payload = {
        key: value
        for key, value in parent_payload.items()
        if key not in {"state_dict", "exact_identity", "native_rounds"}
    }
    payload["schema"] = NATIVE_POWER_LIFT_TREE_SCHEMA
    payload["geometry_role"] = "teacher_free_native_continuation"
    payload["teacher_runtime_dependency"] = False
    payload["state_dict"] = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    payload["bond_dimension"] = int(model.bond_dimension)
    payload["bond_dimensions"] = tuple(
        int(value) for value in model.bond_dimensions
    )
    payload["positive_floor"] = float(model.positive_floor)
    rounds = list(parent_payload.get("native_rounds", ()))
    rounds.append(dict(native_round))
    payload["native_rounds"] = rounds
    payload["initial_exact_identity"] = parent_payload.get(
        "initial_exact_identity",
        parent_payload.get("exact_identity"),
    )
    return payload


def dense_h_potential_and_metric(
    source_h: Any,
    section_values: Any,
    section_derivatives: Any,
    *,
    normalization: float,
) -> tuple[Any, Any]:
    """Evaluate a dense low-degree H metric for exact-lift verification."""

    import torch

    if torch.is_tensor(source_h):
        if (
            source_h.ndim != 2
            or source_h.shape[0] != source_h.shape[1]
            or source_h.shape[0] != section_values.shape[1]
        ):
            raise ValueError("source H tensor is not aligned with sections")
        matrix = source_h.to(
            dtype=section_values.dtype,
            device=section_values.device,
        )
    else:
        matrix = torch.as_tensor(
            _positive_hermitian(source_h),
            dtype=section_values.dtype,
            device=section_values.device,
        )
    h_values = torch.einsum("ab,nb->na", matrix, section_values)
    denominator = torch.real(
        torch.einsum("na,na->n", torch.conj(section_values), h_values)
    )
    if not bool(torch.all(denominator > 0)):
        raise FloatingPointError("source H section norm is not positive")
    h_derivatives = torch.einsum(
        "ab,nbj->naj",
        matrix,
        section_derivatives,
    )
    gradient = torch.einsum(
        "na,naj->nj",
        torch.conj(section_values),
        h_derivatives,
    )
    mixed = torch.einsum(
        "nai,naj->nij",
        torch.conj(section_derivatives),
        h_derivatives,
    )
    metric = mixed / denominator[:, None, None]
    metric = metric - (
        torch.conj(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metric = float(normalization) * metric
    metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
    potential = float(normalization) * torch.log(denominator)
    return potential, metric


def exact_lift_identity_errors(
    model: Any,
    source_h: np.ndarray,
    section_values: Any,
    section_derivatives: Any,
    *,
    source_normalization: float,
) -> dict[str, float]:
    """Compare a restored tree with its low-degree dense source metric."""

    source_potential, source_metric = dense_h_potential_and_metric(
        source_h,
        section_values,
        section_derivatives,
        normalization=source_normalization,
    )
    tree_potential, tree_metric = model.potential_and_metric(
        section_values,
        section_derivatives,
    )
    potential_difference = tree_potential - source_potential
    metric_difference = tree_metric - source_metric
    return {
        "potential_maximum_absolute_error": float(
            np.max(np.abs(potential_difference.detach().cpu().numpy()))
        ),
        "metric_maximum_absolute_error": float(
            np.max(np.abs(metric_difference.detach().cpu().numpy()))
        ),
        "metric_relative_frobenius_error": float(
            np.linalg.norm(metric_difference.detach().cpu().numpy())
            / max(
                np.linalg.norm(source_metric.detach().cpu().numpy()),
                np.finfo(float).tiny,
            )
        ),
    }
