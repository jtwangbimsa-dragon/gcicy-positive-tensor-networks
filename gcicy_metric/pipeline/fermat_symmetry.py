"""Exact finite-group structure for tensor networks on the Fermat quintic."""

from __future__ import annotations

import itertools

import numpy as np


FERMAT_COORDINATE_COUNT = 5
FERMAT_PHASE_MODULUS = 5


def projective_coordinate_charges(
    *,
    coordinate_count: int = FERMAT_COORDINATE_COUNT,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> np.ndarray:
    """Represent diagonal projective phases in the gauge with a_last = 0."""

    if coordinate_count < 2 or modulus <= 1:
        raise ValueError("invalid projective phase-group dimensions")
    charges = np.zeros((coordinate_count, coordinate_count - 1), dtype=np.int64)
    charges[:-1] = np.eye(coordinate_count - 1, dtype=np.int64)
    return charges % modulus


def matrix_unit_phase_charges(
    *,
    coordinate_count: int = FERMAT_COORDINATE_COUNT,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> np.ndarray:
    """Return charges of E_(output,input) in canonical dictionary order."""

    coordinates = projective_coordinate_charges(
        coordinate_count=coordinate_count,
        modulus=modulus,
    )
    charges = (
        coordinates[:, None, :] - coordinates[None, :, :]
    ) % modulus
    return charges.reshape(coordinate_count * coordinate_count, -1)


def fermat_root_virtual_charges(
    multiplicity: int = 1,
    *,
    coordinate_count: int = FERMAT_COORDINATE_COUNT,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> np.ndarray:
    """Return zero plus all ordered root charges, with sector multiplicity."""

    if multiplicity <= 0:
        raise ValueError("charge-sector multiplicity must be positive")
    coordinate_charges = projective_coordinate_charges(
        coordinate_count=coordinate_count,
        modulus=modulus,
    )
    sectors = [np.zeros(coordinate_count - 1, dtype=np.int64)]
    for output_index, input_index in fermat_root_sector_pairs(
        coordinate_count=coordinate_count
    )[1:]:
        sectors.append(
            (
                coordinate_charges[output_index]
                - coordinate_charges[input_index]
            )
            % modulus
        )
    base = np.asarray(sectors, dtype=np.int64)
    if len(np.unique(base, axis=0)) != 1 + coordinate_count * (
        coordinate_count - 1
    ):
        raise RuntimeError("Fermat root-charge construction lost uniqueness")
    return np.repeat(base, multiplicity, axis=0)


def fermat_root_sector_pairs(
    *,
    coordinate_count: int = FERMAT_COORDINATE_COUNT,
) -> tuple[tuple[int, int], ...]:
    """Label the zero sector and ordered roots by coordinate pairs."""

    if coordinate_count < 2:
        raise ValueError("coordinate count must be at least two")
    roots = tuple(
        (output_index, input_index)
        for output_index in range(coordinate_count)
        for input_index in range(coordinate_count)
        if output_index != input_index
    )
    return ((-1, -1), *roots)


def phase_charge_core_masks(
    site_count: int,
    multiplicity: int = 1,
    *,
    coordinate_count: int = FERMAT_COORDINATE_COUNT,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> tuple[np.ndarray, ...]:
    """Build masks imposing c_right = c_left + q_local modulo five."""

    if site_count <= 0:
        raise ValueError("site count must be positive")
    virtual_charges = fermat_root_virtual_charges(
        multiplicity,
        coordinate_count=coordinate_count,
        modulus=modulus,
    )
    return phase_charge_core_masks_for_bonds(
        tuple(virtual_charges for _ in range(max(0, site_count - 1))),
        coordinate_count=coordinate_count,
        modulus=modulus,
    )


def phase_charge_core_masks_for_bonds(
    bond_charges: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    coordinate_count: int = FERMAT_COORDINATE_COUNT,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> tuple[np.ndarray, ...]:
    """Build exact charge masks for arbitrary, possibly nonuniform bond sectors."""

    local_charges = matrix_unit_phase_charges(
        coordinate_count=coordinate_count,
        modulus=modulus,
    )
    normalized_bonds = tuple(
        np.asarray(charges, dtype=np.int64) % modulus for charges in bond_charges
    )
    expected_charge_dimension = coordinate_count - 1
    for charges in normalized_bonds:
        if charges.ndim != 2 or charges.shape[1] != expected_charge_dimension:
            raise ValueError("each bond-charge table has the wrong shape")
        if len(charges) == 0:
            raise ValueError("bond-charge tables cannot be empty")
    site_count = len(normalized_bonds) + 1
    boundary_charge = np.zeros((1, coordinate_count - 1), dtype=np.int64)
    masks = []
    for site_index in range(site_count):
        left = (
            boundary_charge
            if site_index == 0
            else normalized_bonds[site_index - 1]
        )
        right = (
            boundary_charge
            if site_index == site_count - 1
            else normalized_bonds[site_index]
        )
        mismatch = (
            left[:, None, None, :]
            + local_charges[None, None, :, :]
            - right[None, :, None, :]
        ) % modulus
        masks.append(np.all(mismatch == 0, axis=-1))
    return tuple(masks)


def phase_charge_sum_zero_lift(
    charge: np.ndarray | tuple[int, ...],
    *,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> tuple[int, ...]:
    """Lift a four-component projective character to five entries summing to zero."""

    value = np.asarray(charge, dtype=np.int64)
    if value.shape != (FERMAT_COORDINATE_COUNT - 1,):
        raise ValueError("a Fermat phase charge must have four components")
    lifted = [int(component % modulus) for component in value]
    lifted.append(int((-sum(lifted)) % modulus))
    return tuple(lifted)


def phase_charge_s5_orbit_key(
    charge: np.ndarray | tuple[int, ...],
    *,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> tuple[int, ...]:
    """Return a canonical key for the S5 orbit of one phase character."""

    return tuple(sorted(phase_charge_sum_zero_lift(charge, modulus=modulus)))


def phase_charge_s5_orbit_partition(
    charges: np.ndarray,
    *,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> dict[tuple[int, ...], np.ndarray]:
    """Partition a unique charge table into exact S5 coordinate orbits."""

    values = np.asarray(charges, dtype=np.int64) % modulus
    if values.ndim != 2 or values.shape[1] != FERMAT_COORDINATE_COUNT - 1:
        raise ValueError("charge table must have shape (count, 4)")
    if len(np.unique(values, axis=0)) != len(values):
        raise ValueError("charge orbit partition requires unique charges")
    groups: dict[tuple[int, ...], list[np.ndarray]] = {}
    for charge in values:
        groups.setdefault(
            phase_charge_s5_orbit_key(charge, modulus=modulus), []
        ).append(charge)
    return {
        key: np.asarray(rows, dtype=np.int64)
        for key, rows in sorted(groups.items())
    }


def expand_reachable_phase_charges(
    charges: np.ndarray,
    *,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> np.ndarray:
    """Add one local matrix-unit charge and deduplicate the reachable sectors."""

    values = np.asarray(charges, dtype=np.int64) % modulus
    if values.ndim != 2 or values.shape[1] != FERMAT_COORDINATE_COUNT - 1:
        raise ValueError("reachable charge table must have shape (count, 4)")
    local = np.unique(matrix_unit_phase_charges(modulus=modulus), axis=0)
    expanded = (values[:, None, :] + local[None, :, :]) % modulus
    return np.unique(expanded.reshape(-1, values.shape[1]), axis=0)


def reachable_phase_charges(
    local_steps: int,
    *,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> np.ndarray:
    """Enumerate all phase charges reachable after a fixed number of O(1) sites."""

    if local_steps < 0:
        raise ValueError("local step count must be non-negative")
    charges = np.zeros((1, FERMAT_COORDINATE_COUNT - 1), dtype=np.int64)
    for _ in range(local_steps):
        charges = expand_reachable_phase_charges(charges, modulus=modulus)
    return charges


def phase_charge_active_complex_parameter_count(
    site_count: int,
    multiplicity: int = 1,
) -> int:
    """Count independent complex coefficients allowed by the charge masks."""

    return int(
        sum(
            np.count_nonzero(mask)
            for mask in phase_charge_core_masks(site_count, multiplicity)
        )
    )


def phase_s5_core_orbit_labels(
    site_count: int,
    multiplicity: int = 1,
    *,
    coordinate_count: int = FERMAT_COORDINATE_COUNT,
) -> tuple[np.ndarray, ...]:
    """Label S_coordinate orbits of all phase-allowed core coefficients."""

    masks = phase_charge_core_masks(
        site_count,
        multiplicity,
        coordinate_count=coordinate_count,
    )
    if multiplicity > 1:
        base_labels = phase_s5_core_orbit_labels(
            site_count,
            1,
            coordinate_count=coordinate_count,
        )
        lifted = []
        for site_index, mask in enumerate(masks):
            keys = {}
            for left, right, local in np.argwhere(mask):
                base_left, left_copy = (
                    (0, 0)
                    if site_index == 0
                    else divmod(int(left), multiplicity)
                )
                base_right, right_copy = (
                    (0, 0)
                    if site_index == site_count - 1
                    else divmod(int(right), multiplicity)
                )
                base_orbit = int(
                    base_labels[site_index][base_left, base_right, int(local)]
                )
                keys[(int(left), int(right), int(local))] = (
                    base_orbit,
                    left_copy,
                    right_copy,
                )
            key_to_label = {
                key: label for label, key in enumerate(sorted(set(keys.values())))
            }
            labels = np.full(mask.shape, -1, dtype=np.int64)
            for index, key in keys.items():
                labels[index] = key_to_label[key]
            lifted.append(labels)
        return tuple(lifted)

    sector_pairs = fermat_root_sector_pairs(coordinate_count=coordinate_count)
    pair_to_sector = {pair: index for index, pair in enumerate(sector_pairs)}
    permutations = tuple(itertools.permutations(range(coordinate_count)))

    def map_virtual(index: int, permutation: tuple[int, ...]) -> int:
        sector, copy_index = divmod(index, multiplicity)
        pair = sector_pairs[sector]
        mapped_sector = (
            0
            if sector == 0
            else pair_to_sector[(permutation[pair[0]], permutation[pair[1]])]
        )
        return mapped_sector * multiplicity + copy_index

    result = []
    for site_index, mask in enumerate(masks):
        canonical_flat_indices: dict[tuple[int, int, int], int] = {}
        for left, right, local in np.argwhere(mask):
            output_index, input_index = divmod(int(local), coordinate_count)
            images = []
            for permutation in permutations:
                mapped_left = (
                    0
                    if site_index == 0
                    else map_virtual(int(left), permutation)
                )
                mapped_right = (
                    0
                    if site_index == site_count - 1
                    else map_virtual(int(right), permutation)
                )
                mapped_local = (
                    permutation[output_index] * coordinate_count
                    + permutation[input_index]
                )
                images.append(
                    np.ravel_multi_index(
                        (mapped_left, mapped_right, mapped_local),
                        mask.shape,
                    )
                )
            canonical_flat_indices[(int(left), int(right), int(local))] = min(
                images
            )
        canonical_values = {
            value: label
            for label, value in enumerate(
                sorted(set(canonical_flat_indices.values()))
            )
        }
        labels = np.full(mask.shape, -1, dtype=np.int64)
        for index, canonical in canonical_flat_indices.items():
            labels[index] = canonical_values[canonical]
        result.append(labels)
    return tuple(result)


def phase_s5_active_complex_parameter_count(
    site_count: int,
    multiplicity: int = 1,
) -> int:
    """Count independent complex coefficients after phase and S5 constraints."""

    labels = phase_s5_core_orbit_labels(site_count, multiplicity)
    return int(
        sum(int(np.max(core_labels)) + 1 for core_labels in labels)
    )


def phase_charge_two_site_mask(
    left_charges: np.ndarray,
    right_charges: np.ndarray,
    *,
    coordinate_count: int = FERMAT_COORDINATE_COUNT,
    modulus: int = FERMAT_PHASE_MODULUS,
) -> np.ndarray:
    """Mask a two-site supercore by total phase-charge conservation."""

    left = np.asarray(left_charges, dtype=np.int64) % modulus
    right = np.asarray(right_charges, dtype=np.int64) % modulus
    expected_dimension = coordinate_count - 1
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != expected_dimension
        or right.shape[1] != expected_dimension
    ):
        raise ValueError("two-site boundary charges have the wrong shape")
    local = matrix_unit_phase_charges(
        coordinate_count=coordinate_count,
        modulus=modulus,
    )
    mismatch = (
        left[:, None, None, None, :]
        + local[None, None, :, None, :]
        + local[None, None, None, :, :]
        - right[None, :, None, None, :]
    ) % modulus
    return np.all(mismatch == 0, axis=-1)


def phase_s5_two_site_orbit_labels(
    *,
    left_boundary: bool = False,
    right_boundary: bool = False,
    multiplicity: int = 1,
    coordinate_count: int = FERMAT_COORDINATE_COUNT,
) -> np.ndarray:
    """Label S5 orbits in a root-sector two-site invariant supercore."""

    if multiplicity <= 0:
        raise ValueError("charge-sector multiplicity must be positive")
    boundary = np.zeros((1, coordinate_count - 1), dtype=np.int64)
    virtual = fermat_root_virtual_charges(
        multiplicity,
        coordinate_count=coordinate_count,
    )
    left_charges = boundary if left_boundary else virtual
    right_charges = boundary if right_boundary else virtual
    mask = phase_charge_two_site_mask(
        left_charges,
        right_charges,
        coordinate_count=coordinate_count,
    )
    if multiplicity > 1:
        base_labels = phase_s5_two_site_orbit_labels(
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            multiplicity=1,
            coordinate_count=coordinate_count,
        )
        keys = {}
        for left, right, first, second in np.argwhere(mask):
            base_left, left_copy = (
                (0, 0)
                if left_boundary
                else divmod(int(left), multiplicity)
            )
            base_right, right_copy = (
                (0, 0)
                if right_boundary
                else divmod(int(right), multiplicity)
            )
            keys[(int(left), int(right), int(first), int(second))] = (
                int(base_labels[base_left, base_right, first, second]),
                left_copy,
                right_copy,
            )
        key_to_label = {
            key: label for label, key in enumerate(sorted(set(keys.values())))
        }
        labels = np.full(mask.shape, -1, dtype=np.int64)
        for index, key in keys.items():
            labels[index] = key_to_label[key]
        return labels

    sector_pairs = fermat_root_sector_pairs(coordinate_count=coordinate_count)
    pair_to_sector = {pair: index for index, pair in enumerate(sector_pairs)}
    permutations = tuple(itertools.permutations(range(coordinate_count)))

    def map_virtual(index: int, permutation: tuple[int, ...]) -> int:
        pair = sector_pairs[index]
        return (
            0
            if index == 0
            else pair_to_sector[(permutation[pair[0]], permutation[pair[1]])]
        )

    canonical = {}
    for left, right, first, second in np.argwhere(mask):
        first_output, first_input = divmod(int(first), coordinate_count)
        second_output, second_input = divmod(int(second), coordinate_count)
        images = []
        for permutation in permutations:
            mapped = (
                0 if left_boundary else map_virtual(int(left), permutation),
                0 if right_boundary else map_virtual(int(right), permutation),
                permutation[first_output] * coordinate_count
                + permutation[first_input],
                permutation[second_output] * coordinate_count
                + permutation[second_input],
            )
            images.append(np.ravel_multi_index(mapped, mask.shape))
        canonical[(int(left), int(right), int(first), int(second))] = min(
            images
        )
    canonical_to_label = {
        value: label
        for label, value in enumerate(sorted(set(canonical.values())))
    }
    labels = np.full(mask.shape, -1, dtype=np.int64)
    for index, value in canonical.items():
        labels[index] = canonical_to_label[value]
    return labels
