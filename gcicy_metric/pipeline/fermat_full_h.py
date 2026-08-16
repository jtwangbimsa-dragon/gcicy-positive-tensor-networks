"""Positive full-H metrics in the exact Fermat symmetry commutant."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from gcicy_metric.fermat_quintic import (
    FERMAT_COORDINATE_COUNT,
    fermat_permutation_sector_block,
    fermat_phase_sector_indices,
)


@dataclass(frozen=True)
class FermatOrbitBlueprint:
    representative_charge: tuple[int, ...]
    target_charges: tuple[tuple[int, ...], ...]
    target_indices: np.ndarray
    reynolds_maps: np.ndarray
    representative_stabilizer_blocks: np.ndarray
    coordinate_start: int
    coordinate_stop: int
    effective_coordinate_start: int
    effective_coordinate_stop: int
    stabilizer_order: int
    effective_real_dimension: int

    @property
    def block_dimension(self) -> int:
        return int(self.target_indices.shape[1])

    @property
    def orbit_size(self) -> int:
        return int(self.target_indices.shape[0])


@dataclass(frozen=True)
class FermatFullHBlueprint:
    section_count: int
    phase_sector_count: int
    phase_block_real_dimension: int
    representative_real_parameter_count: int
    effective_invariant_real_dimension: int
    orbits: tuple[FermatOrbitBlueprint, ...]


def _hermitian_coordinate_basis(dimension: int) -> np.ndarray:
    rows, columns = np.tril_indices(dimension, k=-1)
    basis = []
    for index in range(dimension):
        value = np.zeros((dimension, dimension), dtype=np.complex128)
        value[index, index] = 1.0
        basis.append(value)
    for row, column in zip(rows, columns):
        value = np.zeros((dimension, dimension), dtype=np.complex128)
        value[row, column] = 1.0
        value[column, row] = 1.0
        basis.append(value)
    for row, column in zip(rows, columns):
        value = np.zeros((dimension, dimension), dtype=np.complex128)
        value[row, column] = 1j
        value[column, row] = -1j
        basis.append(value)
    result = np.asarray(basis)
    if result.shape != (dimension**2, dimension, dimension):
        raise RuntimeError("Hermitian coordinate basis has the wrong dimension")
    return result


def _real_symmetric_coordinate_basis(dimension: int) -> np.ndarray:
    return _hermitian_coordinate_basis(dimension)[: dimension * (dimension + 1) // 2]


def _congruence_map(matrix: np.ndarray) -> np.ndarray:
    """Map row-major ``vec(A)`` to ``vec(T^dagger A T)``."""

    value = np.asarray(matrix, dtype=np.complex128)
    return np.einsum(
        "au,bv->uvab", np.conj(value), value, optimize=True
    ).reshape(value.shape[1] ** 2, value.shape[0] ** 2)


def build_fermat_full_h_blueprint(exponents: np.ndarray) -> FermatFullHBlueprint:
    values = np.asarray(exponents, dtype=np.int64)
    sectors = fermat_phase_sector_indices(values)
    orbit_charges: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for charge in sectors:
        orbit_charges.setdefault(tuple(sorted(charge)), []).append(charge)

    permutations = tuple(itertools.permutations(range(FERMAT_COORDINATE_COUNT)))
    offset = 0
    effective_offset = 0
    orbit_rows = []
    for orbit_key, charges_unsorted in sorted(orbit_charges.items()):
        charges = tuple(sorted(charges_unsorted))
        representative = charges[0]
        source = sectors[representative]
        transforms: dict[tuple[int, ...], list[np.ndarray]] = {
            charge: [] for charge in charges
        }
        target_indices: dict[tuple[int, ...], np.ndarray] = {}
        for permutation in permutations:
            target_charge, target, block = fermat_permutation_sector_block(
                values, source, permutation
            )
            transforms[target_charge].append(block)
            target_indices[target_charge] = target
        expected_stabilizer = len(permutations) // len(charges)
        if any(len(rows) != expected_stabilizer for rows in transforms.values()):
            raise RuntimeError("S5 orbit-stabilizer accounting failed")
        maps = []
        indices = []
        for charge in charges:
            maps.append(
                np.mean(
                    [_congruence_map(block) for block in transforms[charge]],
                    axis=0,
                )
            )
            indices.append(target_indices[charge])
        maps_array = np.asarray(maps, dtype=np.complex128)
        indices_array = np.asarray(indices, dtype=np.int64)
        dimension = len(source)
        hermitian_basis = _hermitian_coordinate_basis(dimension)
        representative_map = maps_array[charges.index(representative)]
        projected = np.einsum(
            "ab,kb->ka",
            representative_map,
            hermitian_basis.reshape(dimension**2, dimension**2),
            optimize=True,
        )
        real_projected = np.concatenate((projected.real, projected.imag), axis=1)
        effective_dimension = int(
            np.linalg.matrix_rank(real_projected, tol=1.0e-10)
        )
        orbit_rows.append(
            FermatOrbitBlueprint(
                representative_charge=representative,
                target_charges=charges,
                target_indices=indices_array,
                reynolds_maps=maps_array,
                representative_stabilizer_blocks=np.asarray(
                    transforms[representative], dtype=np.complex128
                ),
                coordinate_start=offset,
                coordinate_stop=offset + dimension**2,
                effective_coordinate_start=effective_offset,
                effective_coordinate_stop=effective_offset + effective_dimension,
                stabilizer_order=expected_stabilizer,
                effective_real_dimension=effective_dimension,
            )
        )
        offset += dimension**2
        effective_offset += effective_dimension

    covered = np.concatenate(
        [orbit.target_indices.reshape(-1) for orbit in orbit_rows]
    )
    if not np.array_equal(np.sort(covered), np.arange(len(values))):
        raise RuntimeError("Fermat symmetry blueprint did not cover every section")
    return FermatFullHBlueprint(
        section_count=len(values),
        phase_sector_count=len(sectors),
        phase_block_real_dimension=int(
            sum(len(indices) ** 2 for indices in sectors.values())
        ),
        representative_real_parameter_count=offset,
        effective_invariant_real_dimension=int(
            sum(orbit.effective_real_dimension for orbit in orbit_rows)
        ),
        orbits=tuple(orbit_rows),
    )


def fermat_reynolds_project_h(
    exponents: np.ndarray,
    matrix: np.ndarray,
    *,
    conjugation_invariant: bool = False,
) -> np.ndarray:
    """Project a Hermitian matrix to the exact Fermat symmetry commutant.

    The diagonal phase subgroup is averaged first by deleting cross-sector
    blocks.  The remaining matrix is then averaged over all coordinate
    permutations by exact quotient-basis congruences.  Both operations preserve
    positive semidefiniteness.
    """

    values = np.asarray(exponents, dtype=np.int64)
    candidate = np.asarray(matrix, dtype=np.complex128)
    if candidate.shape != (len(values), len(values)):
        raise ValueError("candidate H and section basis dimensions differ")
    candidate = 0.5 * (candidate + candidate.conjugate().T)
    sectors = fermat_phase_sector_indices(values)
    phase_projected = np.zeros_like(candidate)
    for indices in sectors.values():
        phase_projected[np.ix_(indices, indices)] = candidate[
            np.ix_(indices, indices)
        ]

    accumulated = np.zeros_like(candidate)
    permutations = tuple(itertools.permutations(range(FERMAT_COORDINATE_COUNT)))
    for permutation in permutations:
        for source_indices in sectors.values():
            _, target_indices, block = fermat_permutation_sector_block(
                values,
                source_indices,
                permutation,
            )
            source_block = phase_projected[np.ix_(source_indices, source_indices)]
            transformed = block.conjugate().T @ source_block @ block
            accumulated[np.ix_(target_indices, target_indices)] += transformed
    result = accumulated / len(permutations)
    if conjugation_invariant:
        result = 0.5 * (result + np.conj(result))
    result = 0.5 * (result + result.conjugate().T)
    source_trace = float(np.trace(candidate).real)
    result_trace = float(np.trace(result).real)
    if not np.isfinite(source_trace) or not np.isfinite(result_trace) or result_trace <= 0:
        raise FloatingPointError("Fermat Reynolds projection has invalid trace")
    return result * (source_trace / result_trace)


class FermatReynoldsBlockProjector:
    """Precompute the exact Fermat Reynolds map and return invariant H blocks.

    ``fermat_reynolds_project_h`` is intentionally simple, but it repeats the
    full permutation/phase-sector loop for every candidate.  Native tree
    optimization evaluates hundreds of nearby candidates, so the group action
    is compiled once here.  Runtime projection then only applies one small
    linear map per charge orbit.
    """

    def __init__(
        self,
        exponents: np.ndarray,
        *,
        conjugation_invariant: bool = False,
    ) -> None:
        values = np.asarray(exponents, dtype=np.int64)
        self.exponents = values
        self.size = len(values)
        self.conjugation_invariant = bool(conjugation_invariant)
        self.blueprint = build_fermat_full_h_blueprint(values)
        sectors = fermat_phase_sector_indices(values)
        permutations = tuple(itertools.permutations(range(FERMAT_COORDINATE_COUNT)))

        rows = []
        for orbit in self.blueprint.orbits:
            representative = orbit.representative_charge
            source_maps = []
            for charge, source_indices in zip(
                orbit.target_charges,
                orbit.target_indices,
            ):
                transforms = []
                for permutation in permutations:
                    target_charge, _, block = fermat_permutation_sector_block(
                        values,
                        source_indices,
                        permutation,
                    )
                    if target_charge == representative:
                        transforms.append(_congruence_map(block))
                if len(transforms) != orbit.stabilizer_order:
                    raise RuntimeError(
                        "source-to-representative Reynolds map has the wrong "
                        "orbit-stabilizer count"
                    )
                source_maps.append(np.mean(transforms, axis=0))
            rows.append(
                (
                    np.asarray(orbit.target_indices, dtype=np.int64),
                    np.asarray(source_maps, dtype=np.complex128),
                    np.asarray(orbit.reynolds_maps, dtype=np.complex128),
                )
            )
        covered = np.concatenate([indices.reshape(-1) for indices, _, _ in rows])
        if not np.array_equal(np.sort(covered), np.arange(self.size)):
            raise RuntimeError("compiled Reynolds blocks do not cover the basis")
        if set(sectors) != {
            charge
            for orbit in self.blueprint.orbits
            for charge in orbit.target_charges
        }:
            raise RuntimeError("compiled Reynolds charge inventory is incomplete")
        self._orbits = tuple(rows)

    def project_blocks(
        self,
        matrix: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Project ``matrix`` and return grouped invariant blocks.

        Each row contains an ``(orbit, block_dimension)`` index array and the
        corresponding ``(orbit, block_dimension, block_dimension)`` matrices.
        """

        candidate = np.asarray(matrix, dtype=np.complex128)
        if candidate.shape != (self.size, self.size):
            raise ValueError("candidate H and section basis dimensions differ")
        candidate = 0.5 * (candidate + candidate.conjugate().T)
        source_trace = float(np.trace(candidate).real)
        if not np.isfinite(source_trace) or source_trace <= 0:
            raise FloatingPointError("candidate H has an invalid trace")

        projected_rows = []
        result_trace = 0.0
        for indices, source_maps, target_maps in self._orbits:
            source_blocks = np.asarray(
                [
                    candidate[np.ix_(source_indices, source_indices)]
                    for source_indices in indices
                ],
                dtype=np.complex128,
            )
            representative = np.mean(
                np.einsum(
                    "oab,ob->oa",
                    source_maps,
                    source_blocks.reshape(len(indices), -1),
                    optimize=True,
                ),
                axis=0,
            )
            blocks = np.einsum(
                "oab,b->oa",
                target_maps,
                representative,
                optimize=True,
            ).reshape(len(indices), indices.shape[1], indices.shape[1])
            if self.conjugation_invariant:
                blocks = np.real(blocks).astype(np.complex128)
            blocks = 0.5 * (
                blocks + np.conj(np.transpose(blocks, (0, 2, 1)))
            )
            result_trace += float(
                np.real(np.trace(blocks, axis1=1, axis2=2)).sum()
            )
            projected_rows.append((indices, blocks))
        if not np.isfinite(result_trace) or result_trace <= 0:
            raise FloatingPointError("compiled Reynolds projection has invalid trace")
        scale = source_trace / result_trace
        return tuple(
            (indices, blocks * scale) for indices, blocks in projected_rows
        )

    def project_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Materialize the projected matrix for artifact and regression checks."""

        result = np.zeros(
            (self.size, self.size),
            dtype=np.complex128,
        )
        for indices, blocks in self.project_blocks(matrix):
            for block_indices, block in zip(indices, blocks):
                result[np.ix_(block_indices, block_indices)] = block
        return result


def _positive_square_root(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value = 0.5 * (matrix + matrix.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if np.any(eigenvalues <= 0):
        raise ValueError("initial representative H block must be positive definite")
    square_root = (
        eigenvectors * np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.conjugate().T
    inverse_square_root = (
        eigenvectors * np.reciprocal(np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.conjugate().T
    return square_root, inverse_square_root


def _unitary_commutant_basis(
    stabilizer_blocks: np.ndarray,
    reference_h: np.ndarray,
    *,
    conjugation_invariant: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = len(reference_h)
    square_root, inverse_square_root = _positive_square_root(reference_h)
    unitary_blocks = np.einsum(
        "ab,hbc,cd->had",
        square_root,
        stabilizer_blocks,
        inverse_square_root,
        optimize=True,
    )
    identities = np.einsum(
        "hba,hbc->hac",
        np.conj(unitary_blocks),
        unitary_blocks,
        optimize=True,
    )
    unitary_error = np.max(
        np.linalg.norm(
            identities - np.eye(dimension, dtype=np.complex128)[None, :, :],
            axis=(1, 2),
        )
    )
    if unitary_error > 2.0e-9:
        raise RuntimeError(
            f"FS-whitened stabilizer is not unitary: error={unitary_error:.3e}"
        )

    raw_basis = (
        _real_symmetric_coordinate_basis(dimension)
        if conjugation_invariant
        else _hermitian_coordinate_basis(dimension)
    )
    projected = np.mean(
        np.einsum(
            "hba,kbc,hcd->hkad",
            np.conj(unitary_blocks),
            raw_basis,
            unitary_blocks,
            optimize=True,
        ),
        axis=0,
    )
    embedded = np.concatenate(
        (
            projected.reshape(len(raw_basis), dimension**2).real,
            projected.reshape(len(raw_basis), dimension**2).imag,
        ),
        axis=1,
    )
    _, singular_values, right_adjoint = np.linalg.svd(
        embedded, full_matrices=False
    )
    tolerance = (
        np.finfo(np.float64).eps
        * max(embedded.shape)
        * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    vectors = right_adjoint[:rank]
    basis = (
        vectors[:, : dimension**2]
        + 1j * vectors[:, dimension**2 :]
    ).reshape(rank, dimension, dimension)
    basis = 0.5 * (basis + np.conj(np.transpose(basis, (0, 2, 1))))
    for unitary in unitary_blocks:
        transformed = np.einsum(
            "ba,kbc,cd->kad",
            np.conj(unitary),
            basis,
            unitary,
            optimize=True,
        )
        if np.linalg.norm(transformed - basis) > 2.0e-9:
            raise RuntimeError("computed commutant basis is not stabilizer invariant")
    return square_root, basis


class FermatSymmetricFullH(torch.nn.Module):
    """The complete positive Hermitian cone invariant under ``(Z5)^4 semidirect S5``."""

    def __init__(
        self,
        exponents: np.ndarray,
        initial_h: np.ndarray,
        *,
        normalization: float,
        device: torch.device,
        conjugation_invariant: bool = False,
    ) -> None:
        super().__init__()
        values = np.asarray(exponents, dtype=np.int64)
        matrix = np.asarray(initial_h, dtype=np.complex128)
        if matrix.shape != (len(values), len(values)):
            raise ValueError("initial H and section basis dimensions differ")
        if normalization <= 0 or not np.isfinite(normalization):
            raise ValueError("normalization must be finite and positive")
        self.blueprint = build_fermat_full_h_blueprint(values)
        self.size = len(values)
        self.normalization = float(normalization)
        self.conjugation_invariant = bool(conjugation_invariant)

        sectors = fermat_phase_sector_indices(values)
        phase_mask = np.zeros_like(matrix, dtype=bool)
        for indices in sectors.values():
            phase_mask[np.ix_(indices, indices)] = True
        off_sector_norm = np.linalg.norm(matrix[~phase_mask])
        if off_sector_norm > 1.0e-10 * max(np.linalg.norm(matrix), 1.0):
            raise ValueError("initial H is not Fermat phase invariant")

        effective_count = 0
        coordinate_slices = []
        for orbit_index, orbit in enumerate(self.blueprint.orbits):
            representative_indices = sectors[orbit.representative_charge]
            block = matrix[np.ix_(representative_indices, representative_indices)]
            square_root, commutant_basis = _unitary_commutant_basis(
                orbit.representative_stabilizer_blocks,
                block,
                conjugation_invariant=self.conjugation_invariant,
            )
            if (
                not self.conjugation_invariant
                and len(commutant_basis) != orbit.effective_real_dimension
            ):
                raise RuntimeError("whitened and direct commutant dimensions disagree")
            coordinate_slices.append(
                slice(effective_count, effective_count + len(commutant_basis))
            )
            effective_count += len(commutant_basis)
            self.register_buffer(
                f"orbit_{orbit_index}_indices",
                torch.tensor(orbit.target_indices, dtype=torch.long, device=device),
            )
            self.register_buffer(
                f"orbit_{orbit_index}_maps",
                torch.tensor(
                    orbit.reynolds_maps, dtype=torch.complex128, device=device
                ),
            )
            self.register_buffer(
                f"orbit_{orbit_index}_square_root",
                torch.tensor(square_root, dtype=torch.complex128, device=device),
            )
            self.register_buffer(
                f"orbit_{orbit_index}_commutant_basis",
                torch.tensor(
                    commutant_basis, dtype=torch.complex128, device=device
                ),
            )
            reference_blocks = np.asarray(
                [
                    matrix[np.ix_(indices, indices)]
                    for indices in orbit.target_indices
                ]
            )
            self.register_buffer(
                f"orbit_{orbit_index}_reference",
                torch.tensor(reference_blocks, dtype=torch.complex128, device=device),
            )
        if (
            not self.conjugation_invariant
            and effective_count != self.blueprint.effective_invariant_real_dimension
        ):
            raise RuntimeError("effective Fermat coordinate count is inconsistent")
        self._coordinate_slices = tuple(coordinate_slices)
        self.coordinates = torch.nn.Parameter(
            torch.zeros(effective_count, dtype=torch.float64, device=device)
        )

    @property
    def trainable_real_parameter_count(self) -> int:
        return int(self.coordinates.numel())

    @property
    def effective_invariant_real_dimension(self) -> int:
        return self.trainable_real_parameter_count

    def h_blocks(
        self,
        coordinates: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        coordinate_values = self.coordinates if coordinates is None else coordinates
        if coordinate_values.shape != self.coordinates.shape:
            raise ValueError("Fermat coordinate vector has the wrong shape")
        if coordinate_values.is_complex():
            raise ValueError("Fermat coordinates must be real")
        rows = []
        trace = torch.zeros(
            (), dtype=torch.float64, device=coordinate_values.device
        )
        for orbit_index, orbit in enumerate(self.blueprint.orbits):
            dimension = orbit.block_dimension
            coordinate_slice = self._coordinate_slices[orbit_index]
            basis = getattr(self, f"orbit_{orbit_index}_commutant_basis")
            generator = torch.sum(
                coordinate_values[coordinate_slice].to(torch.complex128)[
                    :, None, None
                ]
                * basis,
                dim=0,
            )
            square_root = getattr(self, f"orbit_{orbit_index}_square_root")
            seed = (
                square_root
                @ torch.linalg.matrix_exp(generator)
                @ square_root
            )
            maps = getattr(self, f"orbit_{orbit_index}_maps")
            blocks = torch.matmul(maps, seed.reshape(-1)).reshape(
                orbit.orbit_size, dimension, dimension
            )
            blocks = 0.5 * (blocks + torch.conj(torch.transpose(blocks, 1, 2)))
            trace = trace + torch.real(
                torch.diagonal(blocks, dim1=1, dim2=2)
            ).sum()
            rows.append((getattr(self, f"orbit_{orbit_index}_indices"), blocks))
        scale = self.size / trace
        return tuple((indices, blocks * scale) for indices, blocks in rows)

    def coordinates_from_h(self, matrix: np.ndarray) -> np.ndarray:
        """Recover exact commutant log-coordinates from an invariant positive H."""

        value = np.asarray(matrix, dtype=np.complex128)
        if value.shape != (self.size, self.size):
            raise ValueError("candidate H has the wrong shape")
        if self.conjugation_invariant:
            value = 0.5 * (value + np.conj(value))
        value = 0.5 * (value + value.conjugate().T)
        if np.linalg.eigvalsh(value)[0] <= 0:
            raise ValueError("candidate H must be positive definite")
        rows = []
        for orbit_index, orbit in enumerate(self.blueprint.orbits):
            indices = orbit.target_indices[
                orbit.target_charges.index(orbit.representative_charge)
            ]
            block = value[np.ix_(indices, indices)]
            square_root = getattr(
                self, f"orbit_{orbit_index}_square_root"
            ).detach().cpu().numpy()
            inverse_square_root = np.linalg.inv(square_root)
            whitened = inverse_square_root @ block @ inverse_square_root
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (whitened + whitened.conjugate().T)
            )
            if np.any(eigenvalues <= 0):
                raise ValueError("candidate representative block is not positive")
            generator = (
                eigenvectors * np.log(eigenvalues)[None, :]
            ) @ eigenvectors.conjugate().T
            basis = getattr(
                self, f"orbit_{orbit_index}_commutant_basis"
            ).detach().cpu().numpy()
            coefficients = np.real(
                np.einsum("kab,ab->k", np.conj(basis), generator)
            )
            reconstructed = np.einsum("k,kab->ab", coefficients, basis)
            if np.linalg.norm(reconstructed - generator) > 2.0e-8:
                raise ValueError("candidate H is outside the selected Fermat commutant")
            rows.append(coefficients)
        return np.concatenate(rows)

    def set_coordinates_from_h_(self, matrix: np.ndarray) -> None:
        coordinates = self.coordinates_from_h(matrix)
        with torch.no_grad():
            self.coordinates.copy_(
                torch.tensor(
                    coordinates,
                    dtype=self.coordinates.dtype,
                    device=self.coordinates.device,
                )
            )

    def reference_blocks(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (
                getattr(self, f"orbit_{orbit_index}_indices"),
                getattr(self, f"orbit_{orbit_index}_reference"),
            )
            for orbit_index in range(len(self.blueprint.orbits))
        )

    @staticmethod
    def _quadratic_norm(
        section_values: torch.Tensor,
        blocks: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        result = torch.zeros(
            len(section_values), dtype=torch.float64, device=section_values.device
        )
        for indices, matrices in blocks:
            dimension = indices.shape[1]
            values = section_values[:, indices.reshape(-1)].reshape(
                len(section_values), len(indices), dimension
            )
            transformed = torch.matmul(
                matrices.unsqueeze(0), values.unsqueeze(-1)
            ).squeeze(-1)
            result = result + torch.real(
                torch.sum(torch.conj(values) * transformed, dim=(1, 2))
            )
        return result

    @staticmethod
    def _quadratic_jets(
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
        blocks: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = len(section_values)
        tangent_dimension = section_derivatives.shape[-1]
        denominator = torch.zeros(
            count, dtype=torch.float64, device=section_values.device
        )
        first = torch.zeros(
            (count, tangent_dimension, tangent_dimension),
            dtype=torch.complex128,
            device=section_values.device,
        )
        gradient = torch.zeros(
            (count, tangent_dimension),
            dtype=torch.complex128,
            device=section_values.device,
        )
        for indices, matrices in blocks:
            dimension = indices.shape[1]
            values = section_values[:, indices.reshape(-1)].reshape(
                count, len(indices), dimension
            )
            derivatives = section_derivatives[:, indices.reshape(-1), :].reshape(
                count, len(indices), dimension, tangent_dimension
            )
            transformed_values = torch.matmul(
                matrices.unsqueeze(0), values.unsqueeze(-1)
            ).squeeze(-1)
            transformed_derivatives = torch.matmul(
                matrices.unsqueeze(0), derivatives
            )
            denominator = denominator + torch.real(
                torch.sum(
                    torch.conj(values) * transformed_values,
                    dim=(1, 2),
                )
            )
            flattened_derivatives = derivatives.reshape(
                count, -1, tangent_dimension
            )
            flattened_transformed_derivatives = transformed_derivatives.reshape(
                count, -1, tangent_dimension
            )
            first = first + torch.matmul(
                torch.conj(flattened_derivatives).transpose(1, 2),
                flattened_transformed_derivatives,
            )
            gradient = gradient + torch.matmul(
                torch.conj(values).reshape(count, 1, -1),
                flattened_transformed_derivatives,
            ).squeeze(1)
        return denominator, first, gradient

    def h_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(
            (self.size, self.size),
            dtype=torch.complex128,
            device=self.coordinates.device,
        )
        for indices, blocks in self.h_blocks():
            for block_index in range(len(indices)):
                rows = indices[block_index]
                matrix[rows[:, None], rows[None, :]] = blocks[block_index]
        return matrix

    def potential_and_metric_from_blocks(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
        blocks: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference_norm = self._quadratic_norm(
            section_values, self.reference_blocks()
        )
        if not bool(torch.all(reference_norm > 0)):
            raise FloatingPointError("reference section norm is not positive")
        inverse_scale = torch.rsqrt(reference_norm)
        values = section_values * inverse_scale[:, None]
        derivatives = section_derivatives * inverse_scale[:, None, None]
        denominator, first, gradient = self._quadratic_jets(
            values, derivatives, blocks
        )
        if not bool(torch.all(denominator > 0)):
            raise FloatingPointError("Fermat-symmetric section norm is not positive")
        metric = first / denominator[:, None, None]
        metric = metric - (
            torch.conj(gradient)[:, :, None]
            * gradient[:, None, :]
            / denominator[:, None, None] ** 2
        )
        metric = self.normalization * metric
        metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        potential = self.normalization * (
            torch.log(denominator) + torch.log(reference_norm)
        )
        return potential, metric

    def potential_and_metric(
        self, section_values: torch.Tensor, section_derivatives: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.potential_and_metric_from_blocks(
            section_values,
            section_derivatives,
            self.h_blocks(),
        )

    def forward(
        self, section_values: torch.Tensor, section_derivatives: torch.Tensor
    ) -> torch.Tensor:
        return self.potential_and_metric(section_values, section_derivatives)[1]
