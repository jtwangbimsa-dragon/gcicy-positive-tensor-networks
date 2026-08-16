"""Positive tensor-network Bergman metrics with native section contractions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TensorNetworkMoments:
    """Feature norm and its holomorphic/mixed derivatives."""

    norm: Any
    holomorphic_gradient: Any
    mixed_hessian: Any


@dataclass(frozen=True)
class SharedLocalDictionaryCompression:
    """Truncated shared physical dictionary for a list of dense MPS cores."""

    physical_dictionary: np.ndarray
    coefficient_cores: tuple[np.ndarray, ...]
    singular_values: np.ndarray
    retained_energy_fraction: float
    relative_frobenius_error: float


@dataclass(frozen=True)
class SharedLocalDictionaryExpansion:
    """Lossless rank expansion of a shared local dictionary."""

    physical_dictionary: np.ndarray
    coefficient_cores: tuple[np.ndarray, ...]
    relative_core_error: float
    row_orthonormality_error: float


def _orthonormal_column_completion(
    initial_columns: np.ndarray,
    target_count: int,
    *,
    seed: int,
) -> np.ndarray:
    columns = np.asarray(initial_columns, dtype=np.complex128)
    if columns.ndim != 2 or target_count < columns.shape[1]:
        raise ValueError("orthonormal completion requires a valid target count")
    ambient_dimension, initial_count = columns.shape
    if target_count > ambient_dimension:
        raise ValueError("target count exceeds the ambient dimension")
    gram = columns.conjugate().T @ columns
    if not np.allclose(gram, np.eye(initial_count), rtol=1.0e-11, atol=1.0e-12):
        raise ValueError("initial columns must be orthonormal")
    additional_count = target_count - initial_count
    if additional_count == 0:
        return columns.copy()

    rng = np.random.default_rng(seed)
    candidates = rng.normal(size=(ambient_dimension, additional_count))
    candidates = candidates + 1j * rng.normal(
        size=(ambient_dimension, additional_count)
    )
    candidates -= columns @ (columns.conjugate().T @ candidates)
    additional, triangular = np.linalg.qr(candidates, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) < 1.0e-12:
        raise FloatingPointError("deterministic dictionary completion lost rank")
    completed = np.concatenate((columns, additional), axis=1)
    completed_gram = completed.conjugate().T @ completed
    if not np.allclose(
        completed_gram,
        np.eye(target_count),
        rtol=1.0e-11,
        atol=1.0e-12,
    ):
        raise FloatingPointError("completed dictionary is not orthonormal")
    return completed


def anchored_orthonormal_physical_dictionary(
    anchor: np.ndarray,
    rank: int,
    *,
    seed: int,
) -> np.ndarray:
    """Complete one nonzero matrix to deterministic orthonormal rows."""

    matrix = np.asarray(anchor, dtype=np.complex128)
    if (
        matrix.ndim != 2
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("dictionary anchor must be a finite matrix")
    flattened = matrix.reshape(-1)
    norm = np.linalg.norm(flattened)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("dictionary anchor must be nonzero")
    if rank <= 0 or rank > flattened.size:
        raise ValueError("dictionary rank must fit the physical ambient space")
    initial_column = (flattened / norm).reshape(-1, 1)
    completed = _orthonormal_column_completion(
        initial_column,
        rank,
        seed=seed,
    )
    return np.asarray(completed.T.reshape(rank, *matrix.shape), dtype=np.complex128)


def canonical_matrix_unit_dictionary(
    output_dimension: int,
    input_dimension: int,
) -> np.ndarray:
    """Return the complete fixed orthonormal basis of rectangular local maps."""

    if output_dimension <= 0 or input_dimension <= 0:
        raise ValueError("local-map dimensions must be positive")
    local_dimension = output_dimension * input_dimension
    return np.eye(local_dimension, dtype=np.complex128).reshape(
        local_dimension,
        output_dimension,
        input_dimension,
    )


def expand_shared_local_dictionary_rank(
    physical_dictionary: np.ndarray,
    coefficient_cores: list[np.ndarray] | tuple[np.ndarray, ...],
    target_rank: int,
    *,
    seed: int,
) -> SharedLocalDictionaryExpansion:
    """Increase dictionary rank while preserving every materialized dense core."""

    dictionary = np.asarray(physical_dictionary, dtype=np.complex128)
    coefficients = tuple(
        np.asarray(core, dtype=np.complex128) for core in coefficient_cores
    )
    if (
        dictionary.ndim != 3
        or not np.all(np.isfinite(dictionary))
    ):
        raise ValueError("physical dictionary must have finite matrix rows")
    source_rank = dictionary.shape[0]
    if target_rank <= source_rank:
        raise ValueError("target dictionary rank must exceed the source rank")
    flattened = dictionary.reshape(source_rank, -1)
    if target_rank > flattened.shape[1]:
        raise ValueError("target dictionary rank exceeds the physical dimension")
    if not coefficients or any(
        core.ndim != 3
        or core.shape[-1] != source_rank
        or not np.all(np.isfinite(core))
        for core in coefficients
    ):
        raise ValueError("coefficient cores must share the source dictionary rank")

    orthonormal_columns, triangular = np.linalg.qr(flattened.T, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) < 1.0e-12:
        raise ValueError("source physical dictionary is rank deficient")
    completed_columns = _orthonormal_column_completion(
        orthonormal_columns,
        target_rank,
        seed=seed,
    )
    expanded_dictionary = completed_columns.T.reshape(
        target_rank,
        *dictionary.shape[1:],
    )
    coefficient_transform = triangular.T
    expanded_coefficients = []
    maximum_relative_error = 0.0
    for core in coefficients:
        compensated = np.einsum("lrq,qp->lrp", core, coefficient_transform)
        expanded = np.zeros((*core.shape[:2], target_rank), dtype=np.complex128)
        expanded[..., :source_rank] = compensated
        expected_dense = np.einsum("lrq,qpi->lrpi", core, dictionary)
        actual_dense = np.einsum(
            "lrq,qpi->lrpi", expanded, expanded_dictionary
        )
        relative_error = float(
            np.linalg.norm(actual_dense - expected_dense)
            / max(np.linalg.norm(expected_dense), np.finfo(float).tiny)
        )
        maximum_relative_error = max(maximum_relative_error, relative_error)
        expanded_coefficients.append(expanded)
    expanded_flat = expanded_dictionary.reshape(target_rank, -1)
    row_gram = expanded_flat @ expanded_flat.conjugate().T
    orthonormality_error = float(
        np.linalg.norm(row_gram - np.eye(target_rank), ord=2)
    )
    return SharedLocalDictionaryExpansion(
        physical_dictionary=np.asarray(expanded_dictionary, dtype=np.complex128),
        coefficient_cores=tuple(expanded_coefficients),
        relative_core_error=maximum_relative_error,
        row_orthonormality_error=orthonormality_error,
    )


def compress_cores_to_shared_local_dictionary(
    cores: list[np.ndarray] | tuple[np.ndarray, ...],
    rank: int,
) -> SharedLocalDictionaryCompression:
    """Apply one complex SVD to all local section-pair matrices."""

    arrays = tuple(np.asarray(core, dtype=np.complex128) for core in cores)
    if not arrays or rank <= 0:
        raise ValueError("dense cores and a positive dictionary rank are required")
    if any(core.ndim != 4 for core in arrays):
        raise ValueError("each dense core must have four indices")
    physical_shape = arrays[0].shape[2:]
    if any(core.shape[2:] != physical_shape for core in arrays):
        raise ValueError("dense cores must share physical dimensions")
    rows_per_core = tuple(core.shape[0] * core.shape[1] for core in arrays)
    stacked = np.concatenate(
        [core.reshape(rows, -1) for core, rows in zip(arrays, rows_per_core, strict=True)],
        axis=0,
    )
    maximum_rank = min(stacked.shape)
    if rank > maximum_rank:
        raise ValueError(
            f"dictionary rank {rank} exceeds the maximum rank {maximum_rank}"
        )
    left, singular_values, right = np.linalg.svd(stacked, full_matrices=False)
    dictionary = right[:rank].reshape(rank, *physical_shape)
    coefficients = left[:, :rank] * singular_values[None, :rank]
    coefficient_cores = []
    cursor = 0
    for core, row_count in zip(arrays, rows_per_core, strict=True):
        coefficient_cores.append(
            coefficients[cursor : cursor + row_count].reshape(
                core.shape[0], core.shape[1], rank
            )
        )
        cursor += row_count
    squared = np.square(singular_values)
    retained = float(np.sum(squared[:rank]) / np.sum(squared))
    return SharedLocalDictionaryCompression(
        physical_dictionary=np.asarray(dictionary, dtype=np.complex128),
        coefficient_cores=tuple(
            np.asarray(core, dtype=np.complex128) for core in coefficient_cores
        ),
        singular_values=np.asarray(singular_values, dtype=float),
        retained_energy_fraction=retained,
        relative_frobenius_error=float(np.sqrt(max(0.0, 1.0 - retained))),
    )


def rectangular_reference_factor(
    reference_h: np.ndarray,
    output_dimension: int,
) -> np.ndarray:
    """Return a rectangular factor ``F`` satisfying ``F^* F = H``."""

    square_root = _positive_hermitian_square_root(reference_h)
    section_count = square_root.shape[0]
    if output_dimension < section_count:
        raise ValueError(
            "output dimension must be at least the full-rank section dimension"
        )
    factor = np.zeros(
        (output_dimension, section_count), dtype=np.complex128
    )
    factor[:section_count] = square_root
    return factor


def symmetric_square_reference_h(reference_h: np.ndarray) -> np.ndarray:
    """Induce ``H tensor H`` on the normalized symmetric-square basis."""

    matrix = np.asarray(reference_h, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("reference H must be square")
    section_count = matrix.shape[0]
    symmetric_count = section_count * (section_count + 1) // 2
    embedding = np.zeros(
        (section_count**2, symmetric_count), dtype=np.complex128
    )
    column = 0
    for left in range(section_count):
        for right in range(left, section_count):
            if left == right:
                embedding[left * section_count + right, column] = 1.0
            else:
                value = 1.0 / np.sqrt(2.0)
                embedding[left * section_count + right, column] = value
                embedding[right * section_count + left, column] = value
            column += 1
    induced = embedding.conjugate().T @ np.kron(matrix, matrix) @ embedding
    return 0.5 * (induced + induced.conjugate().T)


def pair_adjacent_dense_cores_to_symmetric_square(
    cores: list[np.ndarray] | tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Pair adjacent MPS maps while restricting inputs to ``Sym^2``.

    An input dimension ``d`` becomes ``d(d+1)/2`` while the purification
    output dimension remains the full tensor product of the two source output
    spaces.  Keeping that rectangular output is required for exact pairing.
    """

    arrays = tuple(np.asarray(core, dtype=np.complex128) for core in cores)
    if not arrays or len(arrays) % 2:
        raise ValueError("pairing requires a nonempty even number of cores")
    if any(core.ndim != 4 for core in arrays):
        raise ValueError("each dense core must have four indices")
    input_dimension = arrays[0].shape[3]
    if any(core.shape[3] != input_dimension for core in arrays):
        raise ValueError("paired cores must share one input dimension")

    paired: list[np.ndarray] = []
    for first, second in zip(arrays[::2], arrays[1::2], strict=True):
        if first.shape[1] != second.shape[0]:
            raise ValueError("adjacent cores have incompatible bond dimensions")
        product = np.einsum(
            "lbpi,brqj->lrpqij",
            first,
            second,
        )
        symmetric_inputs = []
        for left in range(input_dimension):
            for right in range(left, input_dimension):
                if left == right:
                    symmetric_inputs.append(product[..., left, right])
                else:
                    symmetric_inputs.append(
                        (product[..., left, right] + product[..., right, left])
                        / np.sqrt(2.0)
                    )
        paired_core = np.stack(symmetric_inputs, axis=-1)
        paired.append(
            paired_core.reshape(
                first.shape[0],
                second.shape[1],
                first.shape[2] * second.shape[2],
                input_dimension * (input_dimension + 1) // 2,
            )
        )
    return tuple(paired)


def resize_shared_dictionary_coefficient_cores(
    coefficient_cores: list[np.ndarray] | tuple[np.ndarray, ...],
    target_site_count: int,
) -> tuple[np.ndarray, ...]:
    """Interpolate interior coefficient cores while preserving boundaries."""

    arrays = tuple(np.asarray(core, dtype=np.complex128) for core in coefficient_cores)
    if len(arrays) < 3 or target_site_count < 3:
        raise ValueError("site transfer requires source and target interior cores")
    rank = arrays[0].shape[-1]
    bond_dimension = arrays[0].shape[1]
    if arrays[0].shape != (1, bond_dimension, rank):
        raise ValueError("left coefficient boundary has an invalid shape")
    if arrays[-1].shape != (bond_dimension, 1, rank):
        raise ValueError("right coefficient boundary has an invalid shape")
    if any(
        core.shape != (bond_dimension, bond_dimension, rank)
        for core in arrays[1:-1]
    ):
        raise ValueError("interior coefficient cores have inconsistent shapes")
    source_interiors = arrays[1:-1]
    target_interior_count = target_site_count - 2
    if len(source_interiors) == 1:
        target_interiors = [source_interiors[0].copy() for _ in range(target_interior_count)]
    else:
        source_positions = np.linspace(0.0, 1.0, len(source_interiors))
        target_positions = np.linspace(0.0, 1.0, target_interior_count)
        target_interiors = []
        for position in target_positions:
            right = int(np.searchsorted(source_positions, position, side="right"))
            right = min(max(1, right), len(source_positions) - 1)
            left = right - 1
            denominator = source_positions[right] - source_positions[left]
            fraction = float((position - source_positions[left]) / denominator)
            target_interiors.append(
                (1.0 - fraction) * source_interiors[left]
                + fraction * source_interiors[right]
            )
    return (
        arrays[0].copy(),
        *(np.asarray(core, dtype=np.complex128) for core in target_interiors),
        arrays[-1].copy(),
    )


def repeat_shared_dictionary_coefficient_cores(
    coefficient_cores: list[np.ndarray] | tuple[np.ndarray, ...],
    repeat_count: int,
) -> tuple[np.ndarray, ...]:
    """Concatenate identical open-boundary MPS blocks without approximation.

    The scalar bridge between consecutive copies is embedded in the zeroth
    bond channel.  Consequently, the learned feature norm of the repeated
    network is the source learned norm raised to ``repeat_count``.  With the
    degree normalization divided by the same factor, its Kahler potential and
    metric are unchanged before adding the separate positive reference floor.
    """

    arrays = tuple(np.asarray(core, dtype=np.complex128) for core in coefficient_cores)
    if len(arrays) < 2 or repeat_count < 2:
        raise ValueError("MPS repetition requires at least two sites and two copies")
    rank = arrays[0].shape[-1]
    bond_dimension = arrays[0].shape[1]
    if arrays[0].shape != (1, bond_dimension, rank):
        raise ValueError("left coefficient boundary has an invalid shape")
    if arrays[-1].shape != (bond_dimension, 1, rank):
        raise ValueError("right coefficient boundary has an invalid shape")
    if any(
        core.shape != (bond_dimension, bond_dimension, rank)
        for core in arrays[1:-1]
    ):
        raise ValueError("interior coefficient cores have inconsistent shapes")

    repeated: list[np.ndarray] = []
    for copy_index in range(repeat_count):
        if copy_index == 0:
            repeated.append(arrays[0].copy())
        else:
            embedded_left = np.zeros(
                (bond_dimension, bond_dimension, rank), dtype=np.complex128
            )
            embedded_left[0, :, :] = arrays[0][0, :, :]
            repeated.append(embedded_left)
        repeated.extend(core.copy() for core in arrays[1:-1])
        if copy_index == repeat_count - 1:
            repeated.append(arrays[-1].copy())
        else:
            embedded_right = np.zeros(
                (bond_dimension, bond_dimension, rank), dtype=np.complex128
            )
            embedded_right[:, 0, :] = arrays[-1][:, 0, :]
            repeated.append(embedded_right)
    return tuple(repeated)


def activate_repeated_shared_dictionary_bridges(
    coefficient_cores: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    source_site_count: int,
    relative_noise: float,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Activate dormant bond channels at exact-repetition block boundaries.

    Repetition embeds each scalar bridge in channel zero.  Adding matched
    small entries to the inactive outgoing and incoming blocks changes the
    represented feature norm only at second order in ``relative_noise`` while
    making the cross-block channels visible to first-order optimization.
    """

    arrays = tuple(np.asarray(core, dtype=np.complex128) for core in coefficient_cores)
    if source_site_count < 2 or len(arrays) % source_site_count:
        raise ValueError("repeated cores are not aligned with the source block size")
    repeat_count = len(arrays) // source_site_count
    if repeat_count < 2 or relative_noise <= 0 or not np.isfinite(relative_noise):
        raise ValueError("bridge activation requires repeated blocks and positive noise")
    bond_dimension = arrays[0].shape[1]
    rank = arrays[0].shape[-1]
    if bond_dimension < 2:
        raise ValueError("bridge activation requires at least two bond channels")
    if any(core.shape[-1] != rank for core in arrays):
        raise ValueError("repeated cores have inconsistent dictionary ranks")

    squared_norm = sum(float(np.vdot(core, core).real) for core in arrays)
    entry_count = sum(core.size for core in arrays)
    root_mean_square = np.sqrt(squared_norm / entry_count)
    if not np.isfinite(root_mean_square) or root_mean_square <= 0:
        raise ValueError("repeated cores have no finite activation scale")

    rng = np.random.default_rng(seed)
    result = [core.copy() for core in arrays]
    standard_deviation = relative_noise * root_mean_square / np.sqrt(2.0)
    for boundary in range(1, repeat_count):
        right_index = boundary * source_site_count - 1
        left_index = boundary * source_site_count
        right = result[right_index]
        left = result[left_index]
        expected_shape = (bond_dimension, bond_dimension, rank)
        if right.shape != expected_shape or left.shape != expected_shape:
            raise ValueError("repeated scalar bridge cores have invalid shapes")
        if np.any(right[:, 1:, :]) or np.any(left[1:, :, :]):
            raise ValueError("repeated scalar bridge inactive channels are not zero")
        right_noise = rng.normal(size=right[:, 1:, :].shape) + 1j * rng.normal(
            size=right[:, 1:, :].shape
        )
        left_noise = rng.normal(size=left[1:, :, :].shape) + 1j * rng.normal(
            size=left[1:, :, :].shape
        )
        right[:, 1:, :] = standard_deviation * right_noise
        left[1:, :, :] = standard_deviation * left_noise
    return tuple(result)


def activate_one_sided_repeated_shared_dictionary_bridges(
    coefficient_cores: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    source_site_count: int,
    relative_scale: float,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Expose repeated-block channels without changing the represented function.

    Exact repetition connects adjacent copies only through bond channel zero.
    This initializer fills the dormant outgoing channels on the left side of
    each repeated-block boundary while leaving the corresponding incoming
    channels identically zero.  The contraction is therefore unchanged, but
    gradients with respect to the zero incoming block are nonzero.
    """

    arrays = tuple(np.asarray(core, dtype=np.complex128) for core in coefficient_cores)
    if source_site_count < 2 or len(arrays) % source_site_count:
        raise ValueError("repeated cores are not aligned with the source block size")
    repeat_count = len(arrays) // source_site_count
    if repeat_count < 2 or relative_scale <= 0 or not np.isfinite(relative_scale):
        raise ValueError(
            "one-sided bridge activation requires repeated blocks and positive scale"
        )
    bond_dimension = arrays[0].shape[1]
    rank = arrays[0].shape[-1]
    if bond_dimension < 2:
        raise ValueError("bridge activation requires at least two bond channels")
    if any(core.shape[-1] != rank for core in arrays):
        raise ValueError("repeated cores have inconsistent dictionary ranks")

    squared_norm = sum(float(np.vdot(core, core).real) for core in arrays)
    entry_count = sum(core.size for core in arrays)
    root_mean_square = np.sqrt(squared_norm / entry_count)
    if not np.isfinite(root_mean_square) or root_mean_square <= 0:
        raise ValueError("repeated cores have no finite activation scale")

    rng = np.random.default_rng(seed)
    result = [core.copy() for core in arrays]
    standard_deviation = relative_scale * root_mean_square / np.sqrt(2.0)
    for boundary in range(1, repeat_count):
        right_index = boundary * source_site_count - 1
        left_index = boundary * source_site_count
        right = result[right_index]
        left = result[left_index]
        expected_shape = (bond_dimension, bond_dimension, rank)
        if right.shape != expected_shape or left.shape != expected_shape:
            raise ValueError("repeated scalar bridge cores have invalid shapes")
        if np.any(right[:, 1:, :]) or np.any(left[1:, :, :]):
            raise ValueError("repeated scalar bridge inactive channels are not zero")
        right_noise = rng.normal(size=right[:, 1:, :].shape) + 1j * rng.normal(
            size=right[:, 1:, :].shape
        )
        right[:, 1:, :] = standard_deviation * right_noise
    return tuple(result)


def _positive_hermitian_square_root(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("reference H must be square")
    value = 0.5 * (value + value.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues[0] <= 0 or not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("reference H must be positive definite")
    square_root = (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conjugate().T
    return 0.5 * (square_root + square_root.conjugate().T)


class PositiveTensorNetworkMetric:
    """Purified open-boundary MPS map evaluated without a dense target H.

    Each core has indices ``(left bond, right bond, output section, input
    section)``.  Contracting every input index with a low-degree section vector
    produces an output MPS ``B_theta s^tensor(m)``.  Its squared norm is
    ``s^* B_theta^* B_theta s``.  A positive tensor-power reference floor keeps
    the potential globally non-vanishing.
    """

    def __new__(cls, *args, **kwargs):
        import torch

        class _TorchPositiveTensorNetworkMetric(torch.nn.Module):
            def __init__(
                self,
                reference_h,
                *,
                site_count: int,
                bond_dimension: int | Sequence[int],
                target_normalization: float,
                output_dimension: int | None = None,
                positive_floor: float = 1.0e-4,
                initialization_noise: float = 1.0e-3,
                physical_dictionary=None,
                trainable_physical_dictionary: bool = False,
                transfer_implementation: str = "vectorized",
                seed: int = 20260717,
                dtype=None,
                device=None,
            ) -> None:
                super().__init__()
                if site_count <= 0:
                    raise ValueError("site_count must be positive")
                if isinstance(bond_dimension, (int, np.integer)):
                    resolved_bonds = (int(bond_dimension),) * max(
                        site_count - 1,
                        0,
                    )
                else:
                    resolved_bonds = tuple(int(value) for value in bond_dimension)
                    if len(resolved_bonds) != max(site_count - 1, 0):
                        raise ValueError(
                            "one bond dimension is required per internal edge"
                        )
                if any(value <= 0 for value in resolved_bonds):
                    raise ValueError("bond dimensions must be positive")
                if target_normalization <= 0 or not np.isfinite(target_normalization):
                    raise ValueError("target_normalization must be finite and positive")
                if positive_floor < 0 or initialization_noise < 0:
                    raise ValueError("floor and initialization noise must be non-negative")
                if transfer_implementation not in {"scalar", "vectorized"}:
                    raise ValueError(
                        "transfer_implementation must be 'scalar' or 'vectorized'"
                    )
                complex_dtype = dtype or torch.complex128
                if complex_dtype not in {torch.complex64, torch.complex128}:
                    raise ValueError("dtype must be torch.complex64 or torch.complex128")
                reference = np.asarray(reference_h, dtype=np.complex128)
                section_count = len(reference)
                dictionary = None
                if physical_dictionary is not None:
                    dictionary = np.asarray(
                        physical_dictionary,
                        dtype=np.complex128,
                    )
                    if dictionary.ndim != 3 or not np.all(np.isfinite(dictionary)):
                        raise ValueError(
                            "physical dictionary must have shape "
                            "(rank, outputs, sections)"
                        )
                if output_dimension is None:
                    output_dimension = (
                        int(dictionary.shape[1])
                        if dictionary is not None
                        else section_count
                    )
                if output_dimension < section_count:
                    raise ValueError(
                        "output dimension must be at least the section dimension"
                    )
                if dictionary is not None and dictionary.shape[1:] != (
                    output_dimension,
                    section_count,
                ):
                    raise ValueError(
                        "physical dictionary must have shape "
                        "(rank, output_dimension, sections)"
                    )
                reference_factor = rectangular_reference_factor(
                    reference,
                    output_dimension,
                )
                if trainable_physical_dictionary and physical_dictionary is None:
                    raise ValueError(
                        "a trainable physical dictionary requires dictionary values"
                    )
                self.site_count = int(site_count)
                self.bond_dimensions = resolved_bonds
                self.bond_dimension = max(resolved_bonds, default=1)
                self.section_count = int(section_count)
                self.output_dimension = int(output_dimension)
                self.target_normalization = float(target_normalization)
                self.positive_floor = float(positive_floor)
                self.transfer_implementation = transfer_implementation
                self.register_buffer(
                    "reference_h",
                    torch.tensor(reference, dtype=complex_dtype, device=device),
                )

                rng = np.random.default_rng(seed)
                dense_initializers = []
                for site in range(site_count):
                    left = 1 if site == 0 else resolved_bonds[site - 1]
                    right = (
                        1
                        if site == site_count - 1
                        else resolved_bonds[site]
                    )
                    shape = (
                        left,
                        right,
                        self.output_dimension,
                        section_count,
                    )
                    noise = rng.normal(size=shape) + 1j * rng.normal(size=shape)
                    core = initialization_noise * noise / np.sqrt(2.0 * section_count)
                    core[0, 0] += reference_factor
                    dense_initializers.append(core)

                self.cores = torch.nn.ParameterList()
                self.coefficient_cores = torch.nn.ParameterList()
                self.register_parameter("two_site_core", None)
                self.register_parameter("two_site_orbit_parameters", None)
                self.register_buffer("two_site_orbit_labels", None)
                self.two_site_start = None
                self.use_two_site_core = False
                self.register_parameter("three_site_core", None)
                self.three_site_start = None
                self.use_three_site_core = False
                self.blocked_two_site_orbit_parameters = torch.nn.ParameterList()
                self.blocked_two_site_starts = ()
                self.blocked_two_site_sparse_metadata = ()
                self.force_indexed_block_contraction = False
                if physical_dictionary is None:
                    self.architecture = "dense_local_cores"
                    self.physical_dictionary_rank = None
                    self.trainable_physical_dictionary = False
                    self.fixed_canonical_matrix_units = False
                    self.cores.extend(
                        torch.nn.Parameter(
                            torch.tensor(core, dtype=complex_dtype, device=device)
                        )
                        for core in dense_initializers
                    )
                else:
                    if (
                        dictionary is None
                        or dictionary.shape[0] <= 0
                    ):
                        raise ValueError(
                            "physical dictionary must contain at least one row"
                        )
                    self.architecture = "shared_local_dictionary"
                    self.physical_dictionary_rank = int(dictionary.shape[0])
                    self.trainable_physical_dictionary = bool(
                        trainable_physical_dictionary
                    )
                    self.fixed_canonical_matrix_units = bool(
                        not self.trainable_physical_dictionary
                        and len(dictionary) == dictionary.shape[1] * dictionary.shape[2]
                        and np.array_equal(
                            dictionary.reshape(len(dictionary), -1),
                            np.eye(len(dictionary), dtype=np.complex128),
                        )
                    )
                    dictionary_tensor = torch.tensor(
                        dictionary,
                        dtype=complex_dtype,
                        device=device,
                    )
                    if self.trainable_physical_dictionary:
                        self.physical_dictionary = torch.nn.Parameter(dictionary_tensor)
                    else:
                        self.register_buffer(
                            "physical_dictionary",
                            dictionary_tensor,
                        )
                    dictionary_flat = dictionary.reshape(len(dictionary), -1)
                    dictionary_pseudoinverse = np.linalg.pinv(dictionary_flat)
                    for core in dense_initializers:
                        coefficient = core.reshape(
                            -1, self.output_dimension * section_count
                        )
                        coefficient = coefficient @ dictionary_pseudoinverse
                        coefficient = coefficient.reshape(
                            core.shape[0], core.shape[1], len(dictionary)
                        )
                        self.coefficient_cores.append(
                            torch.nn.Parameter(
                                torch.tensor(
                                    coefficient,
                                    dtype=complex_dtype,
                                    device=device,
                                )
                            )
                        )

            @property
            def trainable_real_parameter_count(self) -> int:
                return 2 * sum(parameter.numel() for parameter in self.parameters())

            def materialized_cores(self):
                if self.architecture == "dense_local_cores":
                    return tuple(self.cores)
                if self.fixed_canonical_matrix_units:
                    return tuple(
                        coefficient.reshape(
                            *coefficient.shape[:2],
                            self.output_dimension,
                            self.section_count,
                        )
                        for coefficient in self.coefficient_cores
                    )
                return tuple(
                    torch.einsum(
                        "lrq,qpi->lrpi",
                        coefficient,
                        self.physical_dictionary,
                    )
                    for coefficient in self.coefficient_cores
                )

            def _contract_site(self, site_index, values):
                if self.architecture == "dense_local_cores":
                    return torch.einsum(
                        "lrpi,ni->nlrp",
                        self.cores[site_index],
                        values,
                    )
                transformed = self._transform_physical_inputs(values)
                return self._contract_dictionary_site(site_index, transformed)

            def _transform_physical_inputs(self, values):
                return torch.einsum(
                    "qpi,ni->nqp",
                    self.physical_dictionary,
                    values,
                )

            def _contract_dictionary_site(self, site_index, transformed):
                return torch.einsum(
                    "lrq,nqp->nlrp",
                    self.coefficient_cores[site_index],
                    transformed,
                )

            def orthonormalize_physical_dictionary_(self):
                """Fix the shared-dictionary gauge without changing dense cores."""

                if self.architecture != "shared_local_dictionary":
                    raise ValueError("dense local cores have no physical dictionary")
                with torch.no_grad():
                    flattened = self.physical_dictionary.reshape(
                        self.physical_dictionary_rank,
                        -1,
                    )
                    orthonormal_columns, triangular = torch.linalg.qr(
                        torch.transpose(flattened, 0, 1),
                        mode="reduced",
                    )
                    coefficient_transform = torch.transpose(triangular, 0, 1)
                    for coefficient in self.coefficient_cores:
                        coefficient.copy_(
                            torch.einsum(
                                "lrq,qp->lrp",
                                coefficient,
                                coefficient_transform,
                            )
                        )
                    self.physical_dictionary.copy_(
                        torch.transpose(orthonormal_columns, 0, 1).reshape_as(
                            self.physical_dictionary
                        )
                    )

            def left_canonicalize_coefficient_cores_(self):
                """Put a shared-dictionary coefficient chain in left MPS gauge.

                The QR factors are absorbed into the next site, so the represented
                purification map, feature norm, potential, and metric are unchanged.
                """

                if self.architecture != "shared_local_dictionary":
                    raise ValueError("dense local cores have no coefficient-chain gauge")
                with torch.no_grad():
                    for site_index in range(self.site_count - 1):
                        core = self.coefficient_cores[site_index]
                        left, right, physical = core.shape
                        matrix = core.permute(0, 2, 1).reshape(
                            left * physical, right
                        )
                        if matrix.shape[0] < matrix.shape[1]:
                            raise ValueError(
                                "left canonicalization requires left*physical >= right"
                            )
                        orthonormal, triangular = torch.linalg.qr(
                            matrix, mode="reduced"
                        )
                        core.copy_(
                            orthonormal.reshape(left, physical, right).permute(
                                0, 2, 1
                            )
                        )
                        following = self.coefficient_cores[site_index + 1]
                        following.copy_(
                            torch.einsum("ab,brq->arq", triangular, following)
                        )

            def mixed_canonicalize_coefficient_cores_(self, center: int):
                """Put the coefficient MPS in mixed gauge around one site."""

                if self.architecture != "shared_local_dictionary":
                    raise ValueError("dense local cores have no coefficient-chain gauge")
                if center < 0 or center >= self.site_count:
                    raise ValueError("mixed-canonical center is outside the chain")
                with torch.no_grad():
                    for site_index in range(center):
                        core = self.coefficient_cores[site_index]
                        left, right, physical = core.shape
                        matrix = core.permute(0, 2, 1).reshape(
                            left * physical, right
                        )
                        if matrix.shape[0] < matrix.shape[1]:
                            raise ValueError(
                                "left canonicalization requires left*physical >= right"
                            )
                        orthonormal, triangular = torch.linalg.qr(
                            matrix, mode="reduced"
                        )
                        core.copy_(
                            orthonormal.reshape(left, physical, right).permute(
                                0, 2, 1
                            )
                        )
                        following = self.coefficient_cores[site_index + 1]
                        following.copy_(
                            torch.einsum("ab,brq->arq", triangular, following)
                        )

                    for site_index in range(self.site_count - 1, center, -1):
                        core = self.coefficient_cores[site_index]
                        left, right, physical = core.shape
                        matrix = core.permute(0, 2, 1).reshape(
                            left, physical * right
                        )
                        if matrix.shape[1] < matrix.shape[0]:
                            raise ValueError(
                                "right canonicalization requires physical*right >= left"
                            )
                        orthonormal_columns, triangular = torch.linalg.qr(
                            torch.transpose(matrix, 0, 1), mode="reduced"
                        )
                        core.copy_(
                            torch.transpose(orthonormal_columns, 0, 1)
                            .reshape(left, physical, right)
                            .permute(0, 2, 1)
                        )
                        preceding = self.coefficient_cores[site_index - 1]
                        preceding.copy_(
                            torch.einsum(
                                "loq,oa->laq",
                                preceding,
                                torch.transpose(triangular, 0, 1),
                            )
                        )

            def mixed_canonicalize_coefficient_pair_(self, start: int):
                """Put the coefficient MPS in mixed gauge around two sites."""

                if self.architecture != "shared_local_dictionary":
                    raise ValueError("dense local cores have no coefficient-chain gauge")
                if start < 0 or start + 1 >= self.site_count:
                    raise ValueError("two-site center is outside the chain")
                with torch.no_grad():
                    for site_index in range(start):
                        core = self.coefficient_cores[site_index]
                        left, right, physical = core.shape
                        matrix = core.permute(0, 2, 1).reshape(
                            left * physical, right
                        )
                        if matrix.shape[0] < matrix.shape[1]:
                            raise ValueError(
                                "left canonicalization requires left*physical >= right"
                            )
                        orthonormal, triangular = torch.linalg.qr(
                            matrix, mode="reduced"
                        )
                        core.copy_(
                            orthonormal.reshape(left, physical, right).permute(
                                0, 2, 1
                            )
                        )
                        following = self.coefficient_cores[site_index + 1]
                        following.copy_(
                            torch.einsum("ab,brq->arq", triangular, following)
                        )

                    for site_index in range(self.site_count - 1, start + 1, -1):
                        core = self.coefficient_cores[site_index]
                        left, right, physical = core.shape
                        matrix = core.permute(0, 2, 1).reshape(
                            left, physical * right
                        )
                        if matrix.shape[1] < matrix.shape[0]:
                            raise ValueError(
                                "right canonicalization requires physical*right >= left"
                            )
                        orthonormal_columns, triangular = torch.linalg.qr(
                            torch.transpose(matrix, 0, 1), mode="reduced"
                        )
                        core.copy_(
                            torch.transpose(orthonormal_columns, 0, 1)
                            .reshape(left, physical, right)
                            .permute(0, 2, 1)
                        )
                        preceding = self.coefficient_cores[site_index - 1]
                        preceding.copy_(
                            torch.einsum(
                                "loq,oa->laq",
                                preceding,
                                torch.transpose(triangular, 0, 1),
                            )
                        )

            def mixed_canonicalize_coefficient_block_(
                self,
                start: int,
                length: int,
            ):
                """Put the coefficient MPS in mixed gauge around one site block."""

                if self.architecture != "shared_local_dictionary":
                    raise ValueError("dense local cores have no coefficient-chain gauge")
                if length <= 0 or start < 0 or start + length > self.site_count:
                    raise ValueError("coefficient block is outside the chain")
                final = start + length - 1
                with torch.no_grad():
                    for site_index in range(start):
                        core = self.coefficient_cores[site_index]
                        left, right, physical = core.shape
                        matrix = core.permute(0, 2, 1).reshape(
                            left * physical,
                            right,
                        )
                        if matrix.shape[0] < matrix.shape[1]:
                            raise ValueError(
                                "left canonicalization requires left*physical >= right"
                            )
                        orthonormal, triangular = torch.linalg.qr(
                            matrix,
                            mode="reduced",
                        )
                        core.copy_(
                            orthonormal.reshape(left, physical, right).permute(
                                0,
                                2,
                                1,
                            )
                        )
                        following = self.coefficient_cores[site_index + 1]
                        following.copy_(
                            torch.einsum("ab,brq->arq", triangular, following)
                        )

                    for site_index in range(
                        self.site_count - 1,
                        final,
                        -1,
                    ):
                        core = self.coefficient_cores[site_index]
                        left, right, physical = core.shape
                        matrix = core.permute(0, 2, 1).reshape(
                            left,
                            physical * right,
                        )
                        if matrix.shape[1] < matrix.shape[0]:
                            raise ValueError(
                                "right canonicalization requires physical*right >= left"
                            )
                        orthonormal_columns, triangular = torch.linalg.qr(
                            torch.transpose(matrix, 0, 1),
                            mode="reduced",
                        )
                        core.copy_(
                            torch.transpose(orthonormal_columns, 0, 1)
                            .reshape(left, physical, right)
                            .permute(0, 2, 1)
                        )
                        preceding = self.coefficient_cores[site_index - 1]
                        preceding.copy_(
                            torch.einsum(
                                "loq,oa->laq",
                                preceding,
                                torch.transpose(triangular, 0, 1),
                            )
                        )

            def activate_two_site_coefficient_core_(self, start: int):
                """Replace two adjacent coefficient cores by their linear supercore."""

                if self.architecture != "shared_local_dictionary":
                    raise ValueError("two-site coefficient updates require a dictionary")
                if (
                    self.two_site_core is not None
                    or self.two_site_orbit_parameters is not None
                    or self.use_two_site_core
                    or self.three_site_core is not None
                    or self.use_three_site_core
                ):
                    raise RuntimeError("a coefficient supercore is already active")
                if start < 0 or start + 1 >= self.site_count:
                    raise ValueError("two-site center is outside the chain")
                left = self.coefficient_cores[start]
                right = self.coefficient_cores[start + 1]
                if left.shape[1] != right.shape[0]:
                    raise ValueError("adjacent coefficient-core bonds do not match")
                merged = torch.einsum("lmq,mrs->lrqs", left, right)
                self.two_site_core = torch.nn.Parameter(merged.detach().clone())
                self.two_site_start = int(start)
                self.use_two_site_core = True
                return self.two_site_core

            def activate_three_site_coefficient_core_(self, start: int):
                """Replace three adjacent coefficient cores by one linear supercore."""

                if self.architecture != "shared_local_dictionary":
                    raise ValueError("three-site coefficient updates require a dictionary")
                if (
                    self.two_site_core is not None
                    or self.two_site_orbit_parameters is not None
                    or self.use_two_site_core
                    or self.three_site_core is not None
                    or self.use_three_site_core
                ):
                    raise RuntimeError("a coefficient supercore is already active")
                if start < 0 or start + 2 >= self.site_count:
                    raise ValueError("three-site center is outside the chain")
                covered_sites = {int(start), int(start) + 1, int(start) + 2}
                for blocked_start in self.blocked_two_site_starts:
                    if covered_sites.intersection(
                        {int(blocked_start), int(blocked_start) + 1}
                    ):
                        raise ValueError(
                            "three-site updates cannot overlap a permanent two-site block"
                        )
                left = self.coefficient_cores[start]
                middle = self.coefficient_cores[start + 1]
                right = self.coefficient_cores[start + 2]
                if left.shape[1] != middle.shape[0] or middle.shape[1] != right.shape[0]:
                    raise ValueError("adjacent coefficient-core bonds do not match")
                merged = torch.einsum(
                    "lmx,mny,nrz->lrxyz",
                    left,
                    middle,
                    right,
                )
                self.three_site_core = torch.nn.Parameter(merged.detach().clone())
                self.three_site_start = int(start)
                self.use_three_site_core = True
                return self.three_site_core

            def has_three_site_coefficient_core_at(self, start: int) -> bool:
                """Return whether a transient three-site core starts here."""

                return bool(
                    self.use_three_site_core
                    and self.three_site_core is not None
                    and self.three_site_start == int(start)
                )

            def clear_three_site_coefficient_core_(self) -> None:
                """Remove a transient three-site core without changing stored cores."""

                self.use_three_site_core = False
                self.three_site_start = None
                self.three_site_core = None

            @staticmethod
            def _materialize_orbit_tensor(parameters, labels):
                flat_labels = labels.reshape(-1)
                valid = flat_labels >= 0
                if parameters.ndim != 1:
                    raise ValueError("orbit parameters must form one vector")
                if bool(torch.any(flat_labels[valid] >= parameters.numel())):
                    raise ValueError("orbit label exceeds the parameter vector")
                flat = torch.zeros(
                    flat_labels.shape,
                    dtype=parameters.dtype,
                    device=parameters.device,
                )
                flat[valid] = parameters[flat_labels[valid]]
                return flat.reshape(labels.shape)

            def activate_two_site_coefficient_orbits_(self, start: int, orbit_labels):
                """Activate a minimal parameter vector for a tied two-site core."""

                if self.architecture != "shared_local_dictionary":
                    raise ValueError("two-site coefficient updates require a dictionary")
                if (
                    self.two_site_core is not None
                    or self.two_site_orbit_parameters is not None
                    or self.use_two_site_core
                ):
                    raise RuntimeError("a two-site coefficient core is already active")
                if start < 0 or start + 1 >= self.site_count:
                    raise ValueError("two-site center is outside the chain")
                left = self.coefficient_cores[start]
                right = self.coefficient_cores[start + 1]
                merged = torch.einsum("lmq,mrs->lrqs", left, right)
                labels = torch.as_tensor(
                    orbit_labels,
                    dtype=torch.int64,
                    device=merged.device,
                )
                if labels.shape != merged.shape:
                    raise ValueError("two-site orbit labels have the wrong shape")
                valid = labels >= 0
                if not bool(torch.any(valid)):
                    raise ValueError("two-site orbit labels contain no allowed entries")
                orbit_count = int(torch.max(labels[valid]).item()) + 1
                selected_labels = labels[valid]
                sums = torch.zeros(
                    orbit_count,
                    dtype=merged.dtype,
                    device=merged.device,
                )
                sums = sums.index_add(0, selected_labels, merged[valid])
                counts = torch.bincount(
                    selected_labels,
                    minlength=orbit_count,
                ).to(dtype=merged.real.dtype)
                parameters = sums / counts
                reconstructed = self._materialize_orbit_tensor(parameters, labels)
                relative_error = torch.linalg.vector_norm(merged - reconstructed) / torch.clamp(
                    torch.linalg.vector_norm(merged),
                    min=torch.finfo(merged.real.dtype).tiny,
                )
                self.two_site_orbit_parameters = torch.nn.Parameter(
                    parameters.detach().clone()
                )
                self.two_site_orbit_labels = labels
                self.two_site_start = int(start)
                self.use_two_site_core = True
                return self.two_site_orbit_parameters, float(relative_error)

            def active_two_site_coefficient_core(self):
                """Materialize either dense or orbit-parameterized active core."""

                if self.two_site_core is not None:
                    return self.two_site_core
                if (
                    self.two_site_orbit_parameters is not None
                    and self.two_site_orbit_labels is not None
                ):
                    return self._materialize_orbit_tensor(
                        self.two_site_orbit_parameters,
                        self.two_site_orbit_labels,
                    )
                raise RuntimeError("no active two-site coefficient core")

            def block_two_site_coefficient_orbits_(
                self,
                starts,
                orbit_labels,
                *,
                drop_covered_coefficient_cores: bool = False,
            ):
                """Replace disjoint site pairs by permanent tied supercores."""

                resolved_starts = tuple(int(start) for start in starts)
                label_iterator = iter(orbit_labels)
                if self.blocked_two_site_starts:
                    raise RuntimeError("two-site blocks are already installed")
                occupied = set()
                parameters = []
                sparse_metadata = []
                projection_errors = []
                for block_index, start in enumerate(resolved_starts):
                    try:
                        labels_value = next(label_iterator)
                    except StopIteration as error:
                        raise ValueError(
                            "blocked starts and orbit labels disagree"
                        ) from error
                    if start < 0 or start + 1 >= self.site_count:
                        raise ValueError("blocked pair lies outside the chain")
                    if start in occupied or start + 1 in occupied:
                        raise ValueError("blocked two-site pairs must be disjoint")
                    occupied.update((start, start + 1))
                    left = self.coefficient_cores[start]
                    right = self.coefficient_cores[start + 1]
                    merged = torch.einsum("lmq,mrs->lrqs", left, right)
                    labels = torch.as_tensor(
                        labels_value,
                        dtype=torch.int64,
                        device=merged.device,
                    )
                    if labels.shape != merged.shape:
                        raise ValueError("blocked orbit labels have the wrong shape")
                    valid = labels >= 0
                    orbit_count = int(torch.max(labels[valid]).item()) + 1
                    selected_labels = labels[valid]
                    sums = torch.zeros(
                        orbit_count,
                        dtype=merged.dtype,
                        device=merged.device,
                    )
                    sums = sums.index_add(0, selected_labels, merged[valid])
                    counts = torch.bincount(
                        selected_labels,
                        minlength=orbit_count,
                    ).to(dtype=merged.real.dtype)
                    values = sums / counts
                    reconstructed = self._materialize_orbit_tensor(values, labels)
                    relative_error = torch.linalg.vector_norm(
                        merged - reconstructed
                    ) / torch.clamp(
                        torch.linalg.vector_norm(merged),
                        min=torch.finfo(merged.real.dtype).tiny,
                    )
                    parameters.append(torch.nn.Parameter(values.detach().clone()))
                    edge_indices_name = (
                        f"blocked_two_site_edge_indices_{block_index}"
                    )
                    edge_labels_name = f"blocked_two_site_edge_labels_{block_index}"
                    edge_indices = torch.nonzero(valid, as_tuple=False)
                    edge_labels = labels[valid]
                    self.register_buffer(edge_indices_name, edge_indices)
                    self.register_buffer(edge_labels_name, edge_labels)
                    sparse_metadata.append(
                        (
                            edge_indices_name,
                            edge_labels_name,
                            tuple(int(value) for value in labels.shape),
                        )
                    )
                    projection_errors.append(float(relative_error))
                    left.requires_grad_(False)
                    right.requires_grad_(False)
                try:
                    next(label_iterator)
                except StopIteration:
                    pass
                else:
                    raise ValueError("blocked starts and orbit labels disagree")
                self.blocked_two_site_orbit_parameters.extend(parameters)
                self.blocked_two_site_starts = resolved_starts
                self.blocked_two_site_sparse_metadata = tuple(sparse_metadata)
                if drop_covered_coefficient_cores and occupied == set(
                    range(self.site_count)
                ):
                    self.coefficient_cores = torch.nn.ParameterList()
                return tuple(projection_errors)

            def blocked_two_site_coefficient_core(self, start: int):
                """Materialize one permanent two-site block at a chain position."""

                try:
                    block_index = self.blocked_two_site_starts.index(int(start))
                except ValueError:
                    return None
                edge_indices_name, edge_labels_name, shape = (
                    self.blocked_two_site_sparse_metadata[block_index]
                )
                edge_indices = getattr(self, edge_indices_name)
                edge_labels = getattr(self, edge_labels_name)
                parameters = self.blocked_two_site_orbit_parameters[block_index]
                result = torch.zeros(
                    shape,
                    dtype=parameters.dtype,
                    device=parameters.device,
                )
                result[
                    edge_indices[:, 0],
                    edge_indices[:, 1],
                    edge_indices[:, 2],
                    edge_indices[:, 3],
                ] = parameters[edge_labels]
                return result

            def two_site_coefficient_core_at(self, start: int):
                """Resolve a transient optimizer core or a permanent block."""

                if self.use_two_site_core and self.two_site_start == int(start):
                    return self.active_two_site_coefficient_core()
                return self.blocked_two_site_coefficient_core(start)

            def has_two_site_coefficient_core_at(self, start: int) -> bool:
                """Return whether a transient or permanent pair starts here."""

                resolved_start = int(start)
                return bool(
                    (self.use_two_site_core and self.two_site_start == resolved_start)
                    or resolved_start in self.blocked_two_site_starts
                )

            def split_two_site_coefficient_core(
                self,
                core=None,
                *,
                maximum_bond_dimension: int | None = None,
                relative_singular_value_cutoff: float = 0.0,
                direction: str = "right",
            ):
                """SVD-split an active supercore into the preallocated bond space."""

                if (
                    self.two_site_core is None
                    and self.two_site_orbit_parameters is None
                ) or self.two_site_start is None:
                    raise RuntimeError("no two-site coefficient core is active")
                if direction not in {"left", "right", "symmetric"}:
                    raise ValueError("split direction must be left, right, or symmetric")
                if relative_singular_value_cutoff < 0:
                    raise ValueError("relative singular-value cutoff must be non-negative")
                start = self.two_site_start
                value = self.active_two_site_coefficient_core() if core is None else core
                left_bond, right_bond, first_physical, second_physical = value.shape
                current_capacity = int(self.coefficient_cores[start].shape[1])
                if maximum_bond_dimension is None:
                    capacity = current_capacity
                else:
                    capacity = int(maximum_bond_dimension)
                    if capacity <= 0 or capacity > current_capacity:
                        raise ValueError(
                            "maximum bond dimension must lie in the preallocated bond space"
                        )
                matrix = value.permute(0, 2, 3, 1).reshape(
                    left_bond * first_physical,
                    second_physical * right_bond,
                )
                left_vectors, singular_values, right_vectors = torch.linalg.svd(
                    matrix, full_matrices=False
                )
                if singular_values.numel() == 0 or not bool(
                    torch.all(torch.isfinite(singular_values))
                ):
                    raise FloatingPointError("two-site SVD returned invalid singular values")
                threshold = relative_singular_value_cutoff * float(singular_values[0])
                numerical_rank = int(torch.count_nonzero(singular_values > threshold))
                retained_rank = max(1, min(capacity, numerical_rank))
                retained_singular_values = singular_values[:retained_rank]
                left_matrix = left_vectors[:, :retained_rank]
                right_matrix = right_vectors[:retained_rank]
                if direction == "right":
                    right_matrix = retained_singular_values[:, None] * right_matrix
                elif direction == "left":
                    left_matrix = left_matrix * retained_singular_values[None, :]
                else:
                    square_root = torch.sqrt(retained_singular_values)
                    left_matrix = left_matrix * square_root[None, :]
                    right_matrix = square_root[:, None] * right_matrix

                left_factor = value.new_zeros(
                    left_bond, current_capacity, first_physical
                )
                right_factor = value.new_zeros(
                    current_capacity, right_bond, second_physical
                )
                left_factor[:, :retained_rank].copy_(
                    left_matrix.reshape(
                        left_bond, first_physical, retained_rank
                    ).permute(0, 2, 1)
                )
                right_factor[:retained_rank].copy_(
                    right_matrix.reshape(
                        retained_rank, second_physical, right_bond
                    ).permute(0, 2, 1)
                )
                squared = torch.square(singular_values)
                total_weight = float(torch.sum(squared))
                discarded_weight = float(torch.sum(squared[retained_rank:]))
                return left_factor, right_factor, {
                    "bond_capacity": capacity,
                    "storage_bond_dimension": current_capacity,
                    "numerical_rank": numerical_rank,
                    "retained_rank": retained_rank,
                    "relative_singular_value_cutoff": float(
                        relative_singular_value_cutoff
                    ),
                    "discarded_relative_weight": (
                        0.0 if total_weight == 0.0 else discarded_weight / total_weight
                    ),
                    "largest_singular_value": float(singular_values[0]),
                    "smallest_retained_singular_value": float(
                        singular_values[retained_rank - 1]
                    ),
                }

            def commit_two_site_coefficient_split_(
                self, left_factor, right_factor
            ) -> None:
                """Commit factorized two-site values and remove the transient core."""

                if self.two_site_start is None:
                    raise RuntimeError("no two-site coefficient core is active")
                start = self.two_site_start
                if left_factor.shape != self.coefficient_cores[start].shape:
                    raise ValueError("left split factor has the wrong shape")
                if right_factor.shape != self.coefficient_cores[start + 1].shape:
                    raise ValueError("right split factor has the wrong shape")
                with torch.no_grad():
                    self.coefficient_cores[start].copy_(left_factor)
                    self.coefficient_cores[start + 1].copy_(right_factor)
                self.use_two_site_core = False
                self.two_site_start = None
                self.two_site_core = None
                self.two_site_orbit_parameters = None
                self.two_site_orbit_labels = None

            def _contract_dictionary_two_site_inputs(
                self,
                transformed_inputs,
                core=None,
            ):
                if core is None:
                    core = self.active_two_site_coefficient_core()
                values = transformed_inputs[:, 0]
                derivatives = transformed_inputs[:, 1:]
                site = torch.einsum(
                    "lrxy,nxp,nyq->nlrpq",
                    core,
                    values,
                    values,
                )
                derivative_sites = torch.einsum(
                    "lrxy,nixp,nyq->nilrpq",
                    core,
                    derivatives,
                    values,
                )
                derivative_sites = derivative_sites + torch.einsum(
                    "lrxy,nxp,niyq->nilrpq",
                    core,
                    values,
                    derivatives,
                )
                combined = torch.cat((site[:, None], derivative_sites), dim=1)
                return combined.flatten(start_dim=-2)

            def _contract_sparse_blocked_two_site_inputs(
                self,
                start: int,
                transformed_inputs,
            ):
                """Contract one permanent block without materializing its dense core."""

                try:
                    block_index = self.blocked_two_site_starts.index(int(start))
                except ValueError as error:
                    raise ValueError("no permanent two-site block starts here") from error
                edge_indices_name, edge_labels_name, shape = (
                    self.blocked_two_site_sparse_metadata[block_index]
                )
                edge_indices = getattr(self, edge_indices_name)
                edge_labels = getattr(self, edge_labels_name)
                parameters = self.blocked_two_site_orbit_parameters[block_index]
                left_bond, right_bond, first_rank, second_rank = shape
                if transformed_inputs.ndim != 4:
                    raise ValueError(
                        "transformed two-site inputs must have shape "
                        "(batch, channels, dictionary rank, outputs)"
                    )
                if transformed_inputs.shape[2] != first_rank or first_rank != second_rank:
                    raise ValueError("blocked core and transformed inputs disagree")

                values = transformed_inputs[:, 0]
                pair_features = torch.einsum(
                    "ncxp,nyq->ncxypq",
                    transformed_inputs,
                    values,
                )
                pair_features = pair_features + torch.einsum(
                    "nxp,ncyq->ncxypq",
                    values,
                    transformed_inputs,
                )
                channel_weights = torch.ones(
                    transformed_inputs.shape[1],
                    dtype=transformed_inputs.real.dtype,
                    device=transformed_inputs.device,
                )
                channel_weights[0] = 0.5
                pair_features = pair_features * channel_weights[
                    None, :, None, None, None, None
                ]
                batch_size, channel_count, _, _, first_output, second_output = (
                    pair_features.shape
                )
                feature_matrix = pair_features.permute(2, 3, 0, 1, 4, 5).reshape(
                    first_rank * second_rank,
                    batch_size * channel_count * first_output * second_output,
                )

                row_indices = edge_indices[:, 0] * right_bond + edge_indices[:, 1]
                column_indices = (
                    edge_indices[:, 2] * second_rank + edge_indices[:, 3]
                )
                if self.force_indexed_block_contraction:
                    weighted_features = parameters[edge_labels, None] * torch.index_select(
                        feature_matrix,
                        0,
                        column_indices,
                    )
                    contracted = torch.zeros(
                        (
                            left_bond * right_bond,
                            feature_matrix.shape[1],
                        ),
                        dtype=parameters.dtype,
                        device=parameters.device,
                    ).index_add(0, row_indices, weighted_features)
                else:
                    sparse_indices = torch.stack((row_indices, column_indices))
                    sparse_core = torch.sparse_coo_tensor(
                        sparse_indices,
                        parameters[edge_labels],
                        size=(left_bond * right_bond, first_rank * second_rank),
                        dtype=parameters.dtype,
                        device=parameters.device,
                    ).coalesce()
                    contracted = torch.sparse.mm(sparse_core, feature_matrix)
                return (
                    contracted.reshape(
                        left_bond,
                        right_bond,
                        batch_size,
                        channel_count,
                        first_output,
                        second_output,
                    )
                    .permute(2, 3, 0, 1, 4, 5)
                    .flatten(start_dim=-2)
                )

            def _contract_dictionary_two_site_inputs_at(
                self,
                start: int,
                transformed_inputs,
            ):
                """Use sparse contraction for permanent blocks and dense for updates."""

                if int(start) in self.blocked_two_site_starts:
                    return self._contract_sparse_blocked_two_site_inputs(
                        start,
                        transformed_inputs,
                    )
                return self._contract_dictionary_two_site_inputs(
                    transformed_inputs,
                    core=self.two_site_coefficient_core_at(start),
                )

            def _contract_dictionary_three_site_inputs(
                self,
                transformed_inputs,
                core=None,
            ):
                """Contract one transient three-site core and its first derivatives."""

                if core is None:
                    if self.three_site_core is None:
                        raise RuntimeError("no active three-site coefficient core")
                    core = self.three_site_core
                values = transformed_inputs[:, 0]
                derivatives = transformed_inputs[:, 1:]
                site = torch.einsum(
                    "lrxyz,nxp,nyq,nzu->nlrpqu",
                    core,
                    values,
                    values,
                    values,
                )
                derivative_sites = torch.einsum(
                    "lrxyz,nixp,nyq,nzu->nilrpqu",
                    core,
                    derivatives,
                    values,
                    values,
                )
                derivative_sites = derivative_sites + torch.einsum(
                    "lrxyz,nxp,niyq,nzu->nilrpqu",
                    core,
                    values,
                    derivatives,
                    values,
                )
                derivative_sites = derivative_sites + torch.einsum(
                    "lrxyz,nxp,nyq,nizu->nilrpqu",
                    core,
                    values,
                    values,
                    derivatives,
                )
                combined = torch.cat((site[:, None], derivative_sites), dim=1)
                return combined.flatten(start_dim=-3)

            def normalize_coefficient_chain_scale_(
                self, target_norm: float = 1.0, site_index: int = -1
            ):
                """Remove the metric-invariant overall purification scale.

                The final coefficient core is rescaled and the positive reference
                floor is rescaled by the squared factor.  The complete feature norm
                therefore changes only by one global constant, preserving the metric.
                """

                if self.architecture != "shared_local_dictionary":
                    raise ValueError("dense local cores have no coefficient-chain scale")
                if target_norm <= 0 or not np.isfinite(target_norm):
                    raise ValueError("target norm must be finite and positive")
                resolved_site = site_index % self.site_count
                with torch.no_grad():
                    selected = self.coefficient_cores[resolved_site]
                    observed = torch.linalg.vector_norm(selected)
                    if not bool(torch.isfinite(observed)) or float(observed) <= 0:
                        raise FloatingPointError(
                            "selected coefficient core has no finite norm"
                        )
                    scale = target_norm / float(observed)
                    selected.mul_(scale)
                    self.positive_floor *= scale**2
                return scale

            def _contracted_sites(self, site_inputs):
                return [
                    self._contract_site(site_index, values)
                    for site_index, values in enumerate(site_inputs)
                ]

            @staticmethod
            def _mps_overlap(bra_sites, ket_sites):
                if len(bra_sites) != len(ket_sites) or not bra_sites:
                    raise ValueError("MPS overlap requires aligned non-empty sites")
                batch = bra_sites[0].shape[0]
                environment = torch.ones(
                    (batch, 1, 1),
                    dtype=bra_sites[0].dtype,
                    device=bra_sites[0].device,
                )
                for bra, ket in zip(bra_sites, ket_sites, strict=True):
                    environment = torch.einsum(
                        "nab,narp,nbsp->nrs",
                        environment,
                        torch.conj(bra),
                        ket,
                    )
                return environment[:, 0, 0]

            @staticmethod
            def _overlap_transfer(environment, bra_site, ket_site):
                """Advance one overlap environment without materializing a state."""
                return torch.einsum(
                    "nab,narp,nbsp->nrs",
                    environment,
                    torch.conj(bra_site),
                    ket_site,
                )

            def _state_sites(self, section_values, replacement=None):
                inputs = [section_values] * self.site_count
                if replacement is not None:
                    site, values = replacement
                    inputs = list(inputs)
                    inputs[site] = values
                return self._contracted_sites(inputs)

            def _feature_moments_scalar(self, section_values, section_derivatives):
                values = section_values
                derivatives = section_derivatives
                if values.ndim != 2:
                    raise ValueError("section values must have shape (batch, sections)")
                if (
                    derivatives.ndim != 3
                    or derivatives.shape[:2] != values.shape
                    or values.shape[1] != self.section_count
                ):
                    raise ValueError("section derivatives are not aligned with values")
                coordinate_count = derivatives.shape[2]
                environment = torch.ones(
                    (len(values), 1, 1),
                    dtype=values.dtype,
                    device=values.device,
                )
                holomorphic_environments = [
                    torch.zeros_like(environment) for _ in range(coordinate_count)
                ]
                antiholomorphic_environments = [
                    torch.zeros_like(environment) for _ in range(coordinate_count)
                ]
                mixed_environments = [
                    [
                        torch.zeros_like(environment)
                        for _ in range(coordinate_count)
                    ]
                    for _ in range(coordinate_count)
                ]
                dictionary_values = None
                dictionary_derivatives = None
                if self.architecture == "shared_local_dictionary":
                    dictionary_values = self._transform_physical_inputs(values)
                    dictionary_derivatives = [
                        self._transform_physical_inputs(
                            derivatives[:, :, coordinate]
                        )
                        for coordinate in range(coordinate_count)
                    ]

                # Propagate the overlap and all first/mixed derivatives together.
                # For fixed complex dimension and bond dimension this is one
                # linear pass over the tensor-network sites.
                site_index = 0
                while site_index < self.site_count:
                    active_triple = self.has_three_site_coefficient_core_at(site_index)
                    active_pair = (
                        False
                        if active_triple
                        else self.has_two_site_coefficient_core_at(site_index)
                    )
                    if dictionary_values is None:
                        if active_pair or active_triple:
                            raise ValueError(
                                "coefficient supercores require a shared local dictionary"
                            )
                        site = self._contract_site(site_index, values)
                        derivative_sites = [
                            self._contract_site(
                                site_index,
                                derivatives[:, :, coordinate],
                            )
                            for coordinate in range(coordinate_count)
                        ]
                    elif active_pair or active_triple:
                        block_inputs = torch.cat(
                            (
                                dictionary_values[:, None],
                                torch.stack(dictionary_derivatives, dim=1),
                            ),
                            dim=1,
                        )
                        contracted_block = (
                            self._contract_dictionary_three_site_inputs(block_inputs)
                            if active_triple
                            else self._contract_dictionary_two_site_inputs_at(
                                site_index,
                                block_inputs,
                            )
                        )
                        site = contracted_block[:, 0]
                        derivative_sites = [
                            contracted_block[:, coordinate + 1]
                            for coordinate in range(coordinate_count)
                        ]
                    else:
                        site = self._contract_dictionary_site(
                            site_index,
                            dictionary_values,
                        )
                        derivative_sites = [
                            self._contract_dictionary_site(
                                site_index,
                                dictionary_derivatives[coordinate],
                            )
                            for coordinate in range(coordinate_count)
                        ]
                    previous_environment = environment
                    previous_holomorphic = holomorphic_environments
                    previous_antiholomorphic = antiholomorphic_environments
                    previous_mixed = mixed_environments
                    environment = self._overlap_transfer(
                        previous_environment, site, site
                    )
                    holomorphic_environments = [
                        self._overlap_transfer(previous, site, site)
                        + self._overlap_transfer(
                            previous_environment,
                            site,
                            derivative_sites[coordinate],
                        )
                        for coordinate, previous in enumerate(previous_holomorphic)
                    ]
                    antiholomorphic_environments = [
                        self._overlap_transfer(previous, site, site)
                        + self._overlap_transfer(
                            previous_environment,
                            derivative_sites[coordinate],
                            site,
                        )
                        for coordinate, previous in enumerate(
                            previous_antiholomorphic
                        )
                    ]
                    mixed_environments = [
                        [
                            self._overlap_transfer(
                                previous_mixed[antiholomorphic][holomorphic],
                                site,
                                site,
                            )
                            + self._overlap_transfer(
                                previous_holomorphic[holomorphic],
                                derivative_sites[antiholomorphic],
                                site,
                            )
                            + self._overlap_transfer(
                                previous_antiholomorphic[antiholomorphic],
                                site,
                                derivative_sites[holomorphic],
                            )
                            + self._overlap_transfer(
                                previous_environment,
                                derivative_sites[antiholomorphic],
                                derivative_sites[holomorphic],
                            )
                            for holomorphic in range(coordinate_count)
                        ]
                        for antiholomorphic in range(coordinate_count)
                    ]
                    site_index += 3 if active_triple else (2 if active_pair else 1)

                learned_norm = torch.real(environment[:, 0, 0])
                learned_gradient = torch.stack(
                    [value[:, 0, 0] for value in holomorphic_environments],
                    dim=1,
                )
                learned_mixed = torch.stack(
                    [
                        torch.stack(
                            [value[:, 0, 0] for value in row],
                            dim=1,
                        )
                        for row in mixed_environments
                    ],
                    dim=1,
                )

                h_values = torch.einsum("ab,nb->na", self.reference_h, values)
                reference_norm = torch.real(
                    torch.einsum("na,na->n", torch.conj(values), h_values)
                )
                h_derivatives = torch.einsum(
                    "ab,nbj->naj",
                    self.reference_h,
                    derivatives,
                )
                reference_gradient = torch.einsum(
                    "na,naj->nj",
                    torch.conj(values),
                    h_derivatives,
                )
                reference_mixed = torch.einsum(
                    "nai,naj->nij",
                    torch.conj(derivatives),
                    h_derivatives,
                )
                power = self.site_count
                reference_power = reference_norm**power
                reference_power_gradient = (
                    power
                    * reference_norm[:, None] ** (power - 1)
                    * reference_gradient
                )
                reference_power_mixed = (
                    power
                    * reference_norm[:, None, None] ** (power - 1)
                    * reference_mixed
                )
                if power > 1:
                    reference_power_mixed += (
                        power
                        * (power - 1)
                        * reference_norm[:, None, None] ** (power - 2)
                        * torch.conj(reference_gradient)[:, :, None]
                        * reference_gradient[:, None, :]
                    )

                norm = learned_norm + self.positive_floor * reference_power
                gradient = (
                    learned_gradient
                    + self.positive_floor * reference_power_gradient
                )
                mixed = learned_mixed + self.positive_floor * reference_power_mixed
                return TensorNetworkMoments(
                    norm=norm,
                    holomorphic_gradient=gradient,
                    mixed_hessian=mixed,
                )

            def _feature_moments_vectorized(self, section_values, section_derivatives):
                """Propagate all value/first/mixed jets in four transfers per site."""

                values = section_values
                derivatives = section_derivatives
                if values.ndim != 2:
                    raise ValueError("section values must have shape (batch, sections)")
                if (
                    derivatives.ndim != 3
                    or derivatives.shape[:2] != values.shape
                    or values.shape[1] != self.section_count
                ):
                    raise ValueError("section derivatives are not aligned with values")
                coordinate_count = derivatives.shape[2]
                channel_count = coordinate_count + 1

                # Channel zero carries the value; the remaining channels carry
                # one holomorphic derivative each.  Transforming all channels
                # together avoids one small GPU kernel per coordinate and site.
                source_inputs = torch.cat(
                    (values.unsqueeze(-1), derivatives), dim=2
                ).transpose(1, 2)
                use_canonical_dense_cores = bool(
                    self.fixed_canonical_matrix_units
                    and not self.use_two_site_core
                    and not self.use_three_site_core
                    and not self.blocked_two_site_starts
                )
                canonical_dense_cores = (
                    self.materialized_cores() if use_canonical_dense_cores else None
                )
                if (
                    self.architecture == "shared_local_dictionary"
                    and not use_canonical_dense_cores
                ):
                    transformed_inputs = torch.einsum(
                        "qpi,nci->ncqp",
                        self.physical_dictionary,
                        source_inputs,
                    )
                else:
                    transformed_inputs = None

                jet = torch.zeros(
                    (len(values), channel_count, channel_count, 1, 1),
                    dtype=values.dtype,
                    device=values.device,
                )
                jet[:, 0, 0, 0, 0] = 1.0
                site_index = 0
                while site_index < self.site_count:
                    active_triple = self.has_three_site_coefficient_core_at(site_index)
                    active_pair = (
                        False
                        if active_triple
                        else self.has_two_site_coefficient_core_at(site_index)
                    )
                    if transformed_inputs is None:
                        if active_pair or active_triple:
                            raise ValueError(
                                "coefficient supercores require a shared local dictionary"
                            )
                        core = (
                            canonical_dense_cores[site_index]
                            if canonical_dense_cores is not None
                            else self.cores[site_index]
                        )
                        site_inputs = torch.einsum(
                            "lrpi,nci->nclrp",
                            core,
                            source_inputs,
                        )
                    elif active_triple:
                        site_inputs = self._contract_dictionary_three_site_inputs(
                            transformed_inputs,
                        )
                    elif active_pair:
                        site_inputs = self._contract_dictionary_two_site_inputs_at(
                            site_index,
                            transformed_inputs,
                        )
                    else:
                        site_inputs = torch.einsum(
                            "lrq,ncqp->nclrp",
                            self.coefficient_cores[site_index],
                            transformed_inputs,
                        )
                    site = site_inputs[:, 0]
                    derivative_sites = site_inputs[:, 1:]

                    transported = torch.einsum(
                        "nuvab,narp,nbsp->nuvrs",
                        jet,
                        torch.conj(site),
                        site,
                    )
                    bra_updates = torch.einsum(
                        "nvab,niarp,nbsp->nivrs",
                        jet[:, 0],
                        torch.conj(derivative_sites),
                        site,
                    )
                    ket_updates = torch.einsum(
                        "nuab,narp,njbsp->nujrs",
                        jet[:, :, 0],
                        torch.conj(site),
                        derivative_sites,
                    )
                    mixed_updates = torch.einsum(
                        "nab,niarp,njbsp->nijrs",
                        jet[:, 0, 0],
                        torch.conj(derivative_sites),
                        derivative_sites,
                    )

                    right_bond = site.shape[2]
                    bra_updates = torch.cat(
                        (
                            bra_updates.new_zeros(
                                len(values), 1, channel_count, right_bond, right_bond
                            ),
                            bra_updates,
                        ),
                        dim=1,
                    )
                    ket_updates = torch.cat(
                        (
                            ket_updates.new_zeros(
                                len(values), channel_count, 1, right_bond, right_bond
                            ),
                            ket_updates,
                        ),
                        dim=2,
                    )
                    mixed_updates = torch.cat(
                        (
                            mixed_updates.new_zeros(
                                len(values), 1, coordinate_count, right_bond, right_bond
                            ),
                            mixed_updates,
                        ),
                        dim=1,
                    )
                    mixed_updates = torch.cat(
                        (
                            mixed_updates.new_zeros(
                                len(values), channel_count, 1, right_bond, right_bond
                            ),
                            mixed_updates,
                        ),
                        dim=2,
                    )
                    jet = transported + bra_updates + ket_updates + mixed_updates
                    site_index += 3 if active_triple else (2 if active_pair else 1)

                learned_norm = torch.real(jet[:, 0, 0, 0, 0])
                learned_gradient = jet[:, 0, 1:, 0, 0]
                learned_mixed = jet[:, 1:, 1:, 0, 0]

                h_values = torch.einsum("ab,nb->na", self.reference_h, values)
                reference_norm = torch.real(
                    torch.einsum("na,na->n", torch.conj(values), h_values)
                )
                h_derivatives = torch.einsum(
                    "ab,nbj->naj",
                    self.reference_h,
                    derivatives,
                )
                reference_gradient = torch.einsum(
                    "na,naj->nj",
                    torch.conj(values),
                    h_derivatives,
                )
                reference_mixed = torch.einsum(
                    "nai,naj->nij",
                    torch.conj(derivatives),
                    h_derivatives,
                )
                power = self.site_count
                reference_power = reference_norm**power
                reference_power_gradient = (
                    power
                    * reference_norm[:, None] ** (power - 1)
                    * reference_gradient
                )
                reference_power_mixed = (
                    power
                    * reference_norm[:, None, None] ** (power - 1)
                    * reference_mixed
                )
                if power > 1:
                    reference_power_mixed += (
                        power
                        * (power - 1)
                        * reference_norm[:, None, None] ** (power - 2)
                        * torch.conj(reference_gradient)[:, :, None]
                        * reference_gradient[:, None, :]
                    )

                return TensorNetworkMoments(
                    norm=learned_norm + self.positive_floor * reference_power,
                    holomorphic_gradient=(
                        learned_gradient
                        + self.positive_floor * reference_power_gradient
                    ),
                    mixed_hessian=(
                        learned_mixed
                        + self.positive_floor * reference_power_mixed
                    ),
                )

            def feature_moments(self, section_values, section_derivatives):
                if self.transfer_implementation == "scalar":
                    return self._feature_moments_scalar(
                        section_values, section_derivatives
                    )
                return self._feature_moments_vectorized(
                    section_values, section_derivatives
                )

            def _homogeneously_normalized_feature_moments(
                self, section_values, section_derivatives
            ):
                """Evaluate moments after a pointwise degree-zero rescaling.

                Scaling both section values and their supplied first derivatives
                by the same scalar multiplies all three feature moments by the
                same degree factor, so the metric is unchanged.  Normalizing by
                the reference norm prevents overflow and cancellation at high
                tensor degree while retaining the original potential through a
                logarithmic correction.
                """

                h_values = self.reference_h @ section_values.unsqueeze(-1)
                reference_norm = torch.real(
                    torch.sum(
                        torch.conj(section_values) * h_values.squeeze(-1),
                        dim=1,
                    )
                )
                if not bool(torch.all(reference_norm > 0)):
                    raise FloatingPointError("reference section norm is not positive")
                inverse_scale = torch.rsqrt(reference_norm)
                normalized_values = section_values * inverse_scale[:, None]
                normalized_derivatives = (
                    section_derivatives * inverse_scale[:, None, None]
                )
                moments = self.feature_moments(
                    normalized_values,
                    normalized_derivatives,
                )
                log_norm_correction = self.site_count * torch.log(reference_norm)
                return moments, log_norm_correction

            def log_feature_norm(self, section_values):
                """Evaluate log F without constructing first or mixed metric jets."""

                if section_values.ndim != 2 or section_values.shape[1] != self.section_count:
                    raise ValueError(
                        "section values must have shape (batch, sections)"
                    )
                h_values = self.reference_h @ section_values.unsqueeze(-1)
                reference_norm = torch.real(
                    torch.sum(
                        torch.conj(section_values) * h_values.squeeze(-1),
                        dim=1,
                    )
                )
                if not bool(torch.all(reference_norm > 0)):
                    raise FloatingPointError("reference section norm is not positive")
                normalized_values = section_values * torch.rsqrt(reference_norm)[:, None]
                transformed_values = None
                if self.architecture == "shared_local_dictionary":
                    transformed_values = self._transform_physical_inputs(
                        normalized_values
                    )

                environment = torch.ones(
                    (len(section_values), 1, 1),
                    dtype=section_values.dtype,
                    device=section_values.device,
                )
                site_index = 0
                while site_index < self.site_count:
                    active_triple = self.has_three_site_coefficient_core_at(site_index)
                    active_pair = (
                        False
                        if active_triple
                        else self.has_two_site_coefficient_core_at(site_index)
                    )
                    if transformed_values is None:
                        if active_pair or active_triple:
                            raise ValueError(
                                "coefficient supercores require a shared local dictionary"
                            )
                        site = self._contract_site(site_index, normalized_values)
                    elif active_triple:
                        site = self._contract_dictionary_three_site_inputs(
                            transformed_values[:, None],
                        )[:, 0]
                    elif active_pair:
                        site = self._contract_dictionary_two_site_inputs_at(
                            site_index,
                            transformed_values[:, None],
                        )[:, 0]
                    else:
                        site = self._contract_dictionary_site(
                            site_index,
                            transformed_values,
                        )
                    environment = self._overlap_transfer(environment, site, site)
                    site_index += 3 if active_triple else (2 if active_pair else 1)

                learned_norm = torch.real(environment[:, 0, 0])
                normalized_h_values = torch.einsum(
                    "ab,nb->na",
                    self.reference_h,
                    normalized_values,
                )
                normalized_reference_norm = torch.real(
                    torch.einsum(
                        "na,na->n",
                        torch.conj(normalized_values),
                        normalized_h_values,
                    )
                )
                norm = learned_norm + self.positive_floor * (
                    normalized_reference_norm**self.site_count
                )
                if not bool(torch.all(norm > 0)):
                    raise FloatingPointError("tensor-network feature norm is not positive")
                return torch.log(norm) + self.site_count * torch.log(reference_norm)

            def potential(self, section_values, section_derivatives=None):
                if section_derivatives is None:
                    return self.target_normalization * self.log_feature_norm(
                        section_values
                    )
                moments, log_norm_correction = (
                    self._homogeneously_normalized_feature_moments(
                        section_values, section_derivatives
                    )
                )
                if not bool(torch.all(moments.norm > 0)):
                    raise FloatingPointError("tensor-network feature norm is not positive")
                return self.target_normalization * (
                    torch.log(moments.norm) + log_norm_correction
                )

            def _metric_from_moments(self, moments):
                if not bool(torch.all(moments.norm > 0)):
                    raise FloatingPointError("tensor-network feature norm is not positive")
                metric = moments.mixed_hessian / moments.norm[:, None, None]
                metric -= (
                    torch.conj(moments.holomorphic_gradient)[:, :, None]
                    * moments.holomorphic_gradient[:, None, :]
                    / moments.norm[:, None, None] ** 2
                )
                metric *= self.target_normalization
                return 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))

            def potential_and_metric(self, section_values, section_derivatives):
                moments, log_norm_correction = (
                    self._homogeneously_normalized_feature_moments(
                        section_values, section_derivatives
                    )
                )
                potential = self.target_normalization * (
                    torch.log(moments.norm) + log_norm_correction
                )
                return potential, self._metric_from_moments(moments)

            def forward(self, section_values, section_derivatives):
                moments, _ = self._homogeneously_normalized_feature_moments(
                    section_values, section_derivatives
                )
                return self._metric_from_moments(moments)

            def materialize_m2_h(self, *, maximum_section_count: int = 16):
                """Materialize the dense m=2 form only for equivalence tests."""

                if self.site_count != 2:
                    raise ValueError("materialize_m2_h requires site_count=2")
                if self.section_count > maximum_section_count:
                    raise ValueError("refusing to materialize a production-size dense H")
                cores = self.materialized_cores()
                first = cores[0][0]
                second = cores[1][:, 0]
                dense_map = torch.einsum("api,aqj->pqij", first, second).reshape(
                    self.section_count**2,
                    self.section_count**2,
                )
                dense_h = torch.conj(dense_map.T) @ dense_map
                reference_power = torch.kron(self.reference_h, self.reference_h)
                dense_h = dense_h + self.positive_floor * reference_power
                return 0.5 * (dense_h + torch.conj(dense_h.T))

        return _TorchPositiveTensorNetworkMetric(*args, **kwargs)


class PositiveTensorNetworkCoherentSumMetric:
    """Coherent amplitude sum of equal-degree positive tensor networks.

    For branch amplitudes ``Psi_a`` this represents

    ``F = ||Psi_0 + sum_{a>0} gamma_a Psi_a||^2 + epsilon F_ref``.

    The cross terms are retained, unlike :class:`PositiveTensorNetworkDirectSumMetric`.
    The leading branch has fixed gate one, removing a redundant common complex scale.
    """

    def __new__(
        cls,
        branches,
        *,
        positive_floor: float,
        initial_new_gates=None,
    ):
        import torch

        class _TorchPositiveTensorNetworkCoherentSumMetric(torch.nn.Module):
            def __init__(
                self,
                branch_modules,
                *,
                floor: float,
                new_gates,
            ) -> None:
                super().__init__()
                modules = tuple(branch_modules)
                if len(modules) < 2:
                    raise ValueError("a coherent sum requires at least two branches")
                if not np.isfinite(floor) or floor < 0:
                    raise ValueError("coherent reference floor must be finite and non-negative")
                first = modules[0]
                metadata = (
                    first.site_count,
                    first.section_count,
                    first.output_dimension,
                    float(first.target_normalization),
                )
                for branch in modules[1:]:
                    observed = (
                        branch.site_count,
                        branch.section_count,
                        branch.output_dimension,
                        float(branch.target_normalization),
                    )
                    if observed[:3] != metadata[:3] or not np.isclose(
                        observed[3], metadata[3], rtol=1.0e-12, atol=1.0e-14
                    ):
                        raise ValueError(
                            "coherent branches must share degree, sections, outputs, "
                            "and target normalization"
                        )
                    if not torch.allclose(
                        branch.reference_h,
                        first.reference_h,
                        rtol=1.0e-12,
                        atol=1.0e-14,
                    ):
                        raise ValueError("coherent branches must share reference H")
                self.branches = torch.nn.ModuleList(modules)
                self.site_count = int(metadata[0])
                self.section_count = int(metadata[1])
                self.output_dimension = int(metadata[2])
                self.target_normalization = float(metadata[3])
                self.positive_floor = float(floor)
                self.register_buffer("reference_h", first.reference_h.detach().clone())
                if new_gates is None:
                    gate_values = torch.zeros(
                        len(modules) - 1,
                        dtype=first.reference_h.dtype,
                        device=first.reference_h.device,
                    )
                else:
                    gate_values = torch.as_tensor(
                        new_gates,
                        dtype=first.reference_h.dtype,
                        device=first.reference_h.device,
                    )
                    if gate_values.shape != (len(modules) - 1,):
                        raise ValueError("new coherent gates do not match branch count")
                self.new_gates = torch.nn.Parameter(gate_values.clone())

            @property
            def trainable_real_parameter_count(self) -> int:
                return sum(
                    parameter.numel() * (2 if parameter.is_complex() else 1)
                    for parameter in self.parameters()
                )

            def coherent_gates(self):
                leading = torch.ones(
                    1,
                    dtype=self.new_gates.dtype,
                    device=self.new_gates.device,
                )
                return torch.cat((leading, self.new_gates))

            def set_new_gates_(self, values) -> None:
                gates = torch.as_tensor(
                    values,
                    dtype=self.new_gates.dtype,
                    device=self.new_gates.device,
                )
                if gates.shape != self.new_gates.shape or not bool(
                    torch.all(torch.isfinite(gates))
                ):
                    raise ValueError("coherent gate values must be aligned and finite")
                with torch.no_grad():
                    self.new_gates.copy_(gates)

            @staticmethod
            def _site_input_jets(branch, source_inputs):
                if branch.architecture == "shared_local_dictionary":
                    transformed = torch.einsum(
                        "qpi,nci->ncqp",
                        branch.physical_dictionary,
                        source_inputs,
                    )
                    return tuple(
                        torch.einsum(
                            "lrq,ncqp->nclrp",
                            branch.coefficient_cores[site_index],
                            transformed,
                        )
                        for site_index in range(branch.site_count)
                    )
                return tuple(
                    torch.einsum(
                        "lrpi,nci->nclrp",
                        branch.cores[site_index],
                        source_inputs,
                    )
                    for site_index in range(branch.site_count)
                )

            @staticmethod
            def _cross_moments(bra_sites, ket_sites):
                if len(bra_sites) != len(ket_sites) or not bra_sites:
                    raise ValueError("coherent cross moments require aligned branches")
                batch = bra_sites[0].shape[0]
                channel_count = bra_sites[0].shape[1]
                coordinate_count = channel_count - 1
                jet = torch.zeros(
                    (batch, channel_count, channel_count, 1, 1),
                    dtype=bra_sites[0].dtype,
                    device=bra_sites[0].device,
                )
                jet[:, 0, 0, 0, 0] = 1.0
                for bra_inputs, ket_inputs in zip(
                    bra_sites, ket_sites, strict=True
                ):
                    bra = bra_inputs[:, 0]
                    ket = ket_inputs[:, 0]
                    bra_derivatives = bra_inputs[:, 1:]
                    ket_derivatives = ket_inputs[:, 1:]
                    transported = torch.einsum(
                        "nuvab,narp,nbsp->nuvrs",
                        jet,
                        torch.conj(bra),
                        ket,
                    )
                    bra_updates = torch.einsum(
                        "nvab,niarp,nbsp->nivrs",
                        jet[:, 0],
                        torch.conj(bra_derivatives),
                        ket,
                    )
                    ket_updates = torch.einsum(
                        "nuab,narp,njbsp->nujrs",
                        jet[:, :, 0],
                        torch.conj(bra),
                        ket_derivatives,
                    )
                    mixed_updates = torch.einsum(
                        "nab,niarp,njbsp->nijrs",
                        jet[:, 0, 0],
                        torch.conj(bra_derivatives),
                        ket_derivatives,
                    )
                    right_bra = bra.shape[2]
                    right_ket = ket.shape[2]
                    bra_updates = torch.cat(
                        (
                            bra_updates.new_zeros(
                                batch,
                                1,
                                channel_count,
                                right_bra,
                                right_ket,
                            ),
                            bra_updates,
                        ),
                        dim=1,
                    )
                    ket_updates = torch.cat(
                        (
                            ket_updates.new_zeros(
                                batch,
                                channel_count,
                                1,
                                right_bra,
                                right_ket,
                            ),
                            ket_updates,
                        ),
                        dim=2,
                    )
                    mixed_updates = torch.cat(
                        (
                            mixed_updates.new_zeros(
                                batch,
                                1,
                                coordinate_count,
                                right_bra,
                                right_ket,
                            ),
                            mixed_updates,
                        ),
                        dim=1,
                    )
                    mixed_updates = torch.cat(
                        (
                            mixed_updates.new_zeros(
                                batch,
                                channel_count,
                                1,
                                right_bra,
                                right_ket,
                            ),
                            mixed_updates,
                        ),
                        dim=2,
                    )
                    jet = transported + bra_updates + ket_updates + mixed_updates
                return TensorNetworkMoments(
                    norm=jet[:, 0, 0, 0, 0],
                    holomorphic_gradient=jet[:, 0, 1:, 0, 0],
                    mixed_hessian=jet[:, 1:, 1:, 0, 0],
                )

            def _reference_power_moments(self, values, derivatives):
                h_values = torch.einsum("ab,nb->na", self.reference_h, values)
                reference_norm = torch.real(
                    torch.einsum("na,na->n", torch.conj(values), h_values)
                )
                h_derivatives = torch.einsum(
                    "ab,nbj->naj", self.reference_h, derivatives
                )
                reference_gradient = torch.einsum(
                    "na,naj->nj", torch.conj(values), h_derivatives
                )
                reference_mixed = torch.einsum(
                    "nai,naj->nij", torch.conj(derivatives), h_derivatives
                )
                power = self.site_count
                powered_norm = reference_norm**power
                powered_gradient = (
                    power
                    * reference_norm[:, None] ** (power - 1)
                    * reference_gradient
                )
                powered_mixed = (
                    power
                    * reference_norm[:, None, None] ** (power - 1)
                    * reference_mixed
                )
                if power > 1:
                    powered_mixed = powered_mixed + (
                        power
                        * (power - 1)
                        * reference_norm[:, None, None] ** (power - 2)
                        * torch.conj(reference_gradient)[:, :, None]
                        * reference_gradient[:, None, :]
                    )
                return TensorNetworkMoments(
                    norm=powered_norm,
                    holomorphic_gradient=powered_gradient,
                    mixed_hessian=powered_mixed,
                )

            def feature_moments(self, section_values, section_derivatives):
                if section_values.ndim != 2:
                    raise ValueError("section values must have shape (batch, sections)")
                if (
                    section_derivatives.ndim != 3
                    or section_derivatives.shape[:2] != section_values.shape
                    or section_values.shape[1] != self.section_count
                ):
                    raise ValueError("section derivatives are not aligned with values")
                source_inputs = torch.cat(
                    (section_values.unsqueeze(-1), section_derivatives), dim=2
                ).transpose(1, 2)
                branch_sites = tuple(
                    self._site_input_jets(branch, source_inputs)
                    for branch in self.branches
                )
                gates = self.coherent_gates()
                norm = None
                gradient = None
                mixed = None
                for bra_index, bra_sites in enumerate(branch_sites):
                    for ket_index, ket_sites in enumerate(branch_sites):
                        cross = self._cross_moments(bra_sites, ket_sites)
                        weight = torch.conj(gates[bra_index]) * gates[ket_index]
                        weighted_norm = weight * cross.norm
                        weighted_gradient = weight * cross.holomorphic_gradient
                        weighted_mixed = weight * cross.mixed_hessian
                        norm = weighted_norm if norm is None else norm + weighted_norm
                        gradient = (
                            weighted_gradient
                            if gradient is None
                            else gradient + weighted_gradient
                        )
                        mixed = weighted_mixed if mixed is None else mixed + weighted_mixed
                reference = self._reference_power_moments(
                    section_values, section_derivatives
                )
                return TensorNetworkMoments(
                    norm=torch.real(norm) + self.positive_floor * reference.norm,
                    holomorphic_gradient=(
                        gradient
                        + self.positive_floor * reference.holomorphic_gradient
                    ),
                    mixed_hessian=(
                        mixed + self.positive_floor * reference.mixed_hessian
                    ),
                )

            def _homogeneously_normalized_feature_moments(
                self, section_values, section_derivatives
            ):
                h_values = self.reference_h @ section_values.unsqueeze(-1)
                reference_norm = torch.real(
                    torch.sum(
                        torch.conj(section_values) * h_values.squeeze(-1), dim=1
                    )
                )
                if not bool(torch.all(reference_norm > 0)):
                    raise FloatingPointError("reference section norm is not positive")
                inverse_scale = torch.rsqrt(reference_norm)
                moments = self.feature_moments(
                    section_values * inverse_scale[:, None],
                    section_derivatives * inverse_scale[:, None, None],
                )
                return moments, self.site_count * torch.log(reference_norm)

            def _metric_from_moments(self, moments):
                if not bool(torch.all(moments.norm > 0)):
                    raise FloatingPointError("coherent feature norm is not positive")
                metric = moments.mixed_hessian / moments.norm[:, None, None]
                metric = metric - (
                    torch.conj(moments.holomorphic_gradient)[:, :, None]
                    * moments.holomorphic_gradient[:, None, :]
                    / moments.norm[:, None, None] ** 2
                )
                metric = self.target_normalization * metric
                return 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))

            def potential_and_metric(self, section_values, section_derivatives):
                moments, correction = self._homogeneously_normalized_feature_moments(
                    section_values, section_derivatives
                )
                potential = self.target_normalization * (
                    torch.log(moments.norm) + correction
                )
                return potential, self._metric_from_moments(moments)

            def forward(self, section_values, section_derivatives):
                moments, _ = self._homogeneously_normalized_feature_moments(
                    section_values, section_derivatives
                )
                return self._metric_from_moments(moments)

        return _TorchPositiveTensorNetworkCoherentSumMetric(
            branches,
            floor=positive_floor,
            new_gates=initial_new_gates,
        )


class PositiveTensorNetworkPositiveSumMetric:
    """Incoherent positive sum of equal-degree purification norms.

    For branch amplitudes ``Psi_a`` this represents

    ``F = ||Psi_0||^2 + sum_{a>0} |gamma_a|^2 ||Psi_a||^2 + epsilon F_ref``.

    It is the positive-branch control for
    :class:`PositiveTensorNetworkCoherentSumMetric`: the same branches and
    complex gates are used, but all amplitude cross terms are removed.
    """

    def __new__(
        cls,
        branches,
        *,
        positive_floor: float,
        initial_new_gates=None,
    ):
        import torch

        class _TorchPositiveTensorNetworkPositiveSumMetric(torch.nn.Module):
            def __init__(self, branch_modules, *, floor: float, new_gates) -> None:
                super().__init__()
                modules = tuple(branch_modules)
                if len(modules) < 2:
                    raise ValueError("a positive sum requires at least two branches")
                if not np.isfinite(floor) or floor < 0:
                    raise ValueError("positive reference floor must be finite and non-negative")
                first = modules[0]
                metadata = (
                    first.site_count,
                    first.section_count,
                    first.output_dimension,
                    float(first.target_normalization),
                )
                for branch in modules[1:]:
                    observed = (
                        branch.site_count,
                        branch.section_count,
                        branch.output_dimension,
                        float(branch.target_normalization),
                    )
                    if observed[:3] != metadata[:3] or not np.isclose(
                        observed[3], metadata[3], rtol=1.0e-12, atol=1.0e-14
                    ):
                        raise ValueError(
                            "positive-sum branches must share degree, sections, outputs, "
                            "and target normalization"
                        )
                    if not torch.allclose(
                        branch.reference_h,
                        first.reference_h,
                        rtol=1.0e-12,
                        atol=1.0e-14,
                    ):
                        raise ValueError("positive-sum branches must share reference H")
                self.branches = torch.nn.ModuleList(modules)
                self.site_count = int(metadata[0])
                self.section_count = int(metadata[1])
                self.output_dimension = int(metadata[2])
                self.target_normalization = float(metadata[3])
                self.positive_floor = float(floor)
                self.register_buffer("reference_h", first.reference_h.detach().clone())
                for branch in self.branches:
                    branch.positive_floor = 0.0
                if new_gates is None:
                    gate_values = torch.zeros(
                        len(modules) - 1,
                        dtype=first.reference_h.dtype,
                        device=first.reference_h.device,
                    )
                else:
                    gate_values = torch.as_tensor(
                        new_gates,
                        dtype=first.reference_h.dtype,
                        device=first.reference_h.device,
                    )
                    if gate_values.shape != (len(modules) - 1,):
                        raise ValueError("new positive gates do not match branch count")
                self.new_gates = torch.nn.Parameter(gate_values.clone())

            @property
            def trainable_real_parameter_count(self) -> int:
                return sum(
                    parameter.numel() * (2 if parameter.is_complex() else 1)
                    for parameter in self.parameters()
                )

            def branch_weights(self):
                leading = torch.ones(
                    1,
                    dtype=self.new_gates.real.dtype,
                    device=self.new_gates.device,
                )
                return torch.cat((leading, torch.abs(self.new_gates) ** 2))

            def set_new_gates_(self, values) -> None:
                gates = torch.as_tensor(
                    values,
                    dtype=self.new_gates.dtype,
                    device=self.new_gates.device,
                )
                if gates.shape != self.new_gates.shape or not bool(
                    torch.all(torch.isfinite(gates))
                ):
                    raise ValueError("positive-sum gate values must be aligned and finite")
                with torch.no_grad():
                    self.new_gates.copy_(gates)

            def _reference_power_moments(self, values, derivatives):
                h_values = torch.einsum("ab,nb->na", self.reference_h, values)
                reference_norm = torch.real(
                    torch.einsum("na,na->n", torch.conj(values), h_values)
                )
                h_derivatives = torch.einsum(
                    "ab,nbj->naj", self.reference_h, derivatives
                )
                reference_gradient = torch.einsum(
                    "na,naj->nj", torch.conj(values), h_derivatives
                )
                reference_mixed = torch.einsum(
                    "nai,naj->nij", torch.conj(derivatives), h_derivatives
                )
                power = self.site_count
                powered_norm = reference_norm**power
                powered_gradient = (
                    power
                    * reference_norm[:, None] ** (power - 1)
                    * reference_gradient
                )
                powered_mixed = (
                    power
                    * reference_norm[:, None, None] ** (power - 1)
                    * reference_mixed
                )
                if power > 1:
                    powered_mixed = powered_mixed + (
                        power
                        * (power - 1)
                        * reference_norm[:, None, None] ** (power - 2)
                        * torch.conj(reference_gradient)[:, :, None]
                        * reference_gradient[:, None, :]
                    )
                return TensorNetworkMoments(
                    norm=powered_norm,
                    holomorphic_gradient=powered_gradient,
                    mixed_hessian=powered_mixed,
                )

            def feature_moments(self, section_values, section_derivatives):
                if section_values.ndim != 2:
                    raise ValueError("section values must have shape (batch, sections)")
                if (
                    section_derivatives.ndim != 3
                    or section_derivatives.shape[:2] != section_values.shape
                    or section_values.shape[1] != self.section_count
                ):
                    raise ValueError("section derivatives are not aligned with values")
                weights = self.branch_weights()
                branch_moments = tuple(
                    branch.feature_moments(section_values, section_derivatives)
                    for branch in self.branches
                )
                reference = self._reference_power_moments(
                    section_values, section_derivatives
                )
                return TensorNetworkMoments(
                    norm=(
                        sum(
                            weights[index] * moments.norm
                            for index, moments in enumerate(branch_moments)
                        )
                        + self.positive_floor * reference.norm
                    ),
                    holomorphic_gradient=(
                        sum(
                            weights[index] * moments.holomorphic_gradient
                            for index, moments in enumerate(branch_moments)
                        )
                        + self.positive_floor * reference.holomorphic_gradient
                    ),
                    mixed_hessian=(
                        sum(
                            weights[index] * moments.mixed_hessian
                            for index, moments in enumerate(branch_moments)
                        )
                        + self.positive_floor * reference.mixed_hessian
                    ),
                )

            def _homogeneously_normalized_feature_moments(
                self, section_values, section_derivatives
            ):
                h_values = self.reference_h @ section_values.unsqueeze(-1)
                reference_norm = torch.real(
                    torch.sum(
                        torch.conj(section_values) * h_values.squeeze(-1), dim=1
                    )
                )
                if not bool(torch.all(reference_norm > 0)):
                    raise FloatingPointError("reference section norm is not positive")
                inverse_scale = torch.rsqrt(reference_norm)
                moments = self.feature_moments(
                    section_values * inverse_scale[:, None],
                    section_derivatives * inverse_scale[:, None, None],
                )
                return moments, self.site_count * torch.log(reference_norm)

            def _metric_from_moments(self, moments):
                if not bool(torch.all(moments.norm > 0)):
                    raise FloatingPointError("positive-sum feature norm is not positive")
                metric = moments.mixed_hessian / moments.norm[:, None, None]
                metric = metric - (
                    torch.conj(moments.holomorphic_gradient)[:, :, None]
                    * moments.holomorphic_gradient[:, None, :]
                    / moments.norm[:, None, None] ** 2
                )
                metric = self.target_normalization * metric
                return 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))

            def potential_and_metric(self, section_values, section_derivatives):
                moments, correction = self._homogeneously_normalized_feature_moments(
                    section_values, section_derivatives
                )
                potential = self.target_normalization * (
                    torch.log(moments.norm) + correction
                )
                return potential, self._metric_from_moments(moments)

            def forward(self, section_values, section_derivatives):
                moments, _ = self._homogeneously_normalized_feature_moments(
                    section_values, section_derivatives
                )
                return self._metric_from_moments(moments)

        return _TorchPositiveTensorNetworkPositiveSumMetric(
            branches,
            floor=positive_floor,
            new_gates=initial_new_gates,
        )


class PositiveTensorNetworkDirectSumMetric:
    """Positive direct sum of equal-degree tensor-network feature maps.

    Each branch may use a different low-degree source section vector, provided
    that the tensor powers have the same total line-bundle degree and target
    normalization. The represented feature norm is a non-negative weighted
    sum, so a branch with zero weight recovers the preceding model exactly.
    """

    def __new__(
        cls,
        branches,
        *,
        branch_total_degrees,
        branch_weights,
    ):
        import torch

        class _TorchPositiveTensorNetworkDirectSumMetric(torch.nn.Module):
            def __init__(
                self,
                branch_modules,
                *,
                total_degrees,
                weights,
            ) -> None:
                super().__init__()
                modules = tuple(branch_modules)
                degrees = tuple(int(value) for value in total_degrees)
                numeric_weights = tuple(float(value) for value in weights)
                if len(modules) < 2 or len(degrees) != len(modules):
                    raise ValueError("a direct sum requires aligned branch metadata")
                if len(numeric_weights) != len(modules):
                    raise ValueError("branch weights do not match the branch count")
                if min(degrees) <= 0 or len(set(degrees)) != 1:
                    raise ValueError("all direct-sum branches must have equal degree")
                if any(
                    not np.isfinite(value) or value < 0 for value in numeric_weights
                ) or not any(value > 0 for value in numeric_weights):
                    raise ValueError("branch weights must be finite and non-negative")
                target_normalizations = tuple(
                    float(module.target_normalization) for module in modules
                )
                if not np.allclose(
                    target_normalizations,
                    target_normalizations[0],
                    rtol=1.0e-12,
                    atol=1.0e-14,
                ):
                    raise ValueError("direct-sum target normalizations do not agree")
                reference = modules[0].reference_h
                self.branches = torch.nn.ModuleList(modules)
                self.branch_total_degrees = degrees
                self.total_degree = degrees[0]
                self.target_normalization = target_normalizations[0]
                self.register_buffer(
                    "branch_weights",
                    torch.tensor(
                        numeric_weights,
                        dtype=reference.real.dtype,
                        device=reference.device,
                    ),
                )

            @property
            def trainable_real_parameter_count(self) -> int:
                return sum(
                    parameter.numel() * (2 if parameter.is_complex() else 1)
                    for parameter in self.parameters()
                )

            def set_branch_weights_(self, weights) -> None:
                values = torch.as_tensor(
                    weights,
                    dtype=self.branch_weights.dtype,
                    device=self.branch_weights.device,
                )
                if values.shape != self.branch_weights.shape:
                    raise ValueError("branch weight shape mismatch")
                if not bool(torch.all(torch.isfinite(values) & (values >= 0))):
                    raise ValueError("branch weights must be finite and non-negative")
                if not bool(torch.any(values > 0)):
                    raise ValueError("at least one branch weight must be positive")
                with torch.no_grad():
                    self.branch_weights.copy_(values)

            def orthonormalize_physical_dictionaries_(self) -> None:
                for branch in self.branches:
                    dictionary = getattr(branch, "physical_dictionary", None)
                    if dictionary is not None and dictionary.requires_grad:
                        branch.orthonormalize_physical_dictionary_()

            def _aligned_branch_moments(self, branch_inputs):
                inputs = tuple(branch_inputs)
                if len(inputs) != len(self.branches):
                    raise ValueError("direct-sum inputs do not match branch count")
                moments = []
                corrections = []
                for branch, pair in zip(self.branches, inputs, strict=True):
                    if len(pair) != 2:
                        raise ValueError("each branch input must contain values and jets")
                    branch_moments, correction = (
                        branch._homogeneously_normalized_feature_moments(*pair)
                    )
                    moments.append(branch_moments)
                    corrections.append(correction)

                reference_correction = corrections[0]
                aligned = []
                for index, (branch_moments, correction) in enumerate(
                    zip(moments, corrections, strict=True)
                ):
                    scale = self.branch_weights[index] * torch.exp(
                        correction - reference_correction
                    )
                    aligned.append(
                        TensorNetworkMoments(
                            norm=scale * branch_moments.norm,
                            holomorphic_gradient=(
                                scale[:, None]
                                * branch_moments.holomorphic_gradient
                            ),
                            mixed_hessian=(
                                scale[:, None, None]
                                * branch_moments.mixed_hessian
                            ),
                        )
                    )
                combined = TensorNetworkMoments(
                    norm=sum(value.norm for value in aligned),
                    holomorphic_gradient=sum(
                        value.holomorphic_gradient for value in aligned
                    ),
                    mixed_hessian=sum(value.mixed_hessian for value in aligned),
                )
                return combined, reference_correction, tuple(aligned)

            def branch_norm_fractions(self, branch_inputs):
                combined, _, aligned = self._aligned_branch_moments(branch_inputs)
                return torch.stack(
                    [value.norm / combined.norm for value in aligned],
                    dim=1,
                )

            def _metric_from_moments(self, moments):
                if not bool(torch.all(moments.norm > 0)):
                    raise FloatingPointError("direct-sum feature norm is not positive")
                metric = moments.mixed_hessian / moments.norm[:, None, None]
                metric -= (
                    torch.conj(moments.holomorphic_gradient)[:, :, None]
                    * moments.holomorphic_gradient[:, None, :]
                    / moments.norm[:, None, None] ** 2
                )
                metric *= self.target_normalization
                return 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))

            def potential_and_metric(self, branch_inputs):
                moments, correction, _ = self._aligned_branch_moments(branch_inputs)
                potential = self.target_normalization * (
                    torch.log(moments.norm) + correction
                )
                return potential, self._metric_from_moments(moments)

            def forward(self, branch_inputs):
                moments, _, _ = self._aligned_branch_moments(branch_inputs)
                return self._metric_from_moments(moments)

        return _TorchPositiveTensorNetworkDirectSumMetric(
            branches,
            total_degrees=branch_total_degrees,
            weights=branch_weights,
        )


def positive_tensor_network_from_artifact_payload(
    reference_h: np.ndarray,
    payload: dict[str, Any],
    *,
    device=None,
    trainable_physical_dictionary: bool | None = None,
):
    """Construct and restore either dense or shared-dictionary TN artifacts."""

    import torch

    precision = str(payload["precision"])
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    state = payload["state_dict"]
    architecture = payload.get("architecture", "dense_local_cores")
    if architecture == "dense_local_cores":
        physical_dictionary = None
        dictionary_trainable = False
    elif architecture == "shared_local_dictionary":
        if "physical_dictionary" not in state:
            raise ValueError("shared-dictionary artifact has no physical dictionary")
        dictionary_value = state["physical_dictionary"]
        if hasattr(dictionary_value, "detach"):
            dictionary_value = dictionary_value.detach().cpu().numpy()
        physical_dictionary = np.asarray(dictionary_value, dtype=np.complex128)
        dictionary_trainable = bool(
            payload.get("trainable_physical_dictionary", False)
        )
        if trainable_physical_dictionary is not None:
            dictionary_trainable = bool(trainable_physical_dictionary)
    else:
        raise ValueError(f"unrecognized tensor-network architecture: {architecture}")
    model = PositiveTensorNetworkMetric(
        reference_h,
        site_count=int(payload["site_count"]),
        bond_dimension=payload.get(
            "bond_dimensions",
            int(payload["bond_dimension"]),
        ),
        target_normalization=float(payload["target_normalization"]),
        output_dimension=int(
            payload.get(
                "output_dimension",
                (
                    physical_dictionary.shape[1]
                    if physical_dictionary is not None
                    else len(reference_h)
                ),
            )
        ),
        positive_floor=float(payload["positive_floor"]),
        initialization_noise=0.0,
        physical_dictionary=physical_dictionary,
        trainable_physical_dictionary=dictionary_trainable,
        dtype=complex_dtype,
        device=device,
    )
    blocked_starts = tuple(
        int(value) for value in payload.get("fermat_two_site_block_starts", ())
    )
    if blocked_starts:
        from .fermat_symmetry import phase_s5_two_site_orbit_labels

        multiplicity = int(payload.get("fermat_phase_charge_multiplicity", 0))
        if multiplicity <= 0 or not payload.get("fermat_s5_orbit_tying", False):
            raise ValueError("blocked Fermat artifact lacks hard-symmetry metadata")
        labels = tuple(
            phase_s5_two_site_orbit_labels(
                left_boundary=start == 0,
                right_boundary=start + 1 == model.site_count - 1,
                multiplicity=multiplicity,
            )
            for start in blocked_starts
        )
        model.block_two_site_coefficient_orbits_(
            blocked_starts,
            labels,
            drop_covered_coefficient_cores=bool(
                payload.get("fermat_block_compact_storage", False)
            ),
        )
    model.load_state_dict(state)
    return model
