#!/usr/bin/env python3
"""Initialize a degree-30 positive TN from a degree-10 full-H oracle.

The source purification is the principal square root ``B = H^(1/2)`` embedded
into ten ordered degree-one sites by uniform monomial symmetrization.  A
matrix-free occupation-count TT-SVD truncates this exact tensor to a requested
bond dimension.  Three identical ten-site factors are then concatenated, so
their squared norm is an approximation to ``F_10^3`` at degree 30.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import PositiveTensorNetworkMetric
from scripts.audit_quintic_full_h_oracle_purification_schmidt import (
    compressed_purification_cut,
    multinomial_count,
    principal_positive_square_root,
    weak_compositions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-h", type=Path, required=True)
    parser.add_argument("--template-model", type=Path, required=True)
    parser.add_argument("--bond-dimension", type=int, required=True)
    parser.add_argument("--power", type=int, default=3)
    parser.add_argument("--relative-entry-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _leading_left_singular_vectors(
    matrix: sparse.spmatrix,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    smaller = min(matrix.shape)
    kept = min(rank, smaller)
    if kept == smaller or smaller <= 64:
        dense = matrix.toarray()
        left, singular_values, _ = np.linalg.svd(dense, full_matrices=False)
        return left[:, :kept], singular_values[:kept]
    try:
        left, singular_values, _ = svds(
            matrix,
            k=kept,
            which="LM",
            solver="propack",
            tol=1.0e-10,
        )
    except (TypeError, ValueError):
        left, singular_values, _ = svds(
            matrix,
            k=kept,
            which="LM",
            tol=1.0e-10,
        )
    order = np.argsort(singular_values)[::-1]
    return left[:, order], singular_values[order]


def _extension_projection(
    previous_overlaps: np.ndarray,
    *,
    previous_degree: int,
    coordinate_count: int,
    used_current_rows: np.ndarray,
) -> sparse.csr_matrix:
    """Return ``<left_bond, physical | current_count_class>``."""

    current_degree = previous_degree + 1
    previous_compositions = weak_compositions(previous_degree, coordinate_count)
    current_compositions = weak_compositions(current_degree, coordinate_count)
    previous_index = {
        value: index for index, value in enumerate(previous_compositions)
    }
    previous_count = len(previous_compositions)
    current_count = len(current_compositions)
    if previous_overlaps.shape[0] != previous_count**2:
        raise ValueError("previous count-overlap table has the wrong size")
    previous_rank = previous_overlaps.shape[1]
    physical_dimension = coordinate_count**2
    previous_multiplicities = {
        value: multinomial_count(value) for value in previous_compositions
    }
    current_multiplicities = {
        value: multinomial_count(value) for value in current_compositions
    }

    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[complex] = []
    for restricted_column, full_row in enumerate(used_current_rows):
        output_current = current_compositions[int(full_row) // current_count]
        input_current = current_compositions[int(full_row) % current_count]
        for output_symbol, output_occupation in enumerate(output_current):
            if output_occupation == 0:
                continue
            output_previous = list(output_current)
            output_previous[output_symbol] -= 1
            output_previous_tuple = tuple(output_previous)
            output_scale = math.sqrt(
                previous_multiplicities[output_previous_tuple]
                / current_multiplicities[output_current]
            )
            for input_symbol, input_occupation in enumerate(input_current):
                if input_occupation == 0:
                    continue
                input_previous = list(input_current)
                input_previous[input_symbol] -= 1
                input_previous_tuple = tuple(input_previous)
                input_scale = math.sqrt(
                    previous_multiplicities[input_previous_tuple]
                    / current_multiplicities[input_current]
                )
                previous_pair = (
                    previous_index[output_previous_tuple] * previous_count
                    + previous_index[input_previous_tuple]
                )
                physical = output_symbol * coordinate_count + input_symbol
                values = (
                    np.conj(previous_overlaps[previous_pair])
                    * output_scale
                    * input_scale
                )
                for bond_index, value in enumerate(values):
                    if value == 0:
                        continue
                    row_indices.append(
                        bond_index * physical_dimension + physical
                    )
                    column_indices.append(restricted_column)
                    data.append(value)
    return sparse.coo_matrix(
        (
            np.asarray(data, dtype=np.complex128),
            (row_indices, column_indices),
        ),
        shape=(previous_rank * physical_dimension, len(used_current_rows)),
    ).tocsr()


def _full_count_vector(
    exponents: np.ndarray,
    factor: np.ndarray,
    *,
    relative_entry_tolerance: float,
) -> sparse.csr_matrix:
    degree = int(np.sum(exponents[0]))
    coordinate_count = int(exponents.shape[1])
    compositions = weak_compositions(degree, coordinate_count)
    index = {value: position for position, value in enumerate(compositions)}
    count = len(compositions)
    threshold = relative_entry_tolerance * float(np.max(np.abs(factor)))
    rows: list[int] = []
    data: list[complex] = []
    for output_index, output_exponent_array in enumerate(exponents):
        output_exponent = tuple(int(value) for value in output_exponent_array)
        output_row = index[output_exponent]
        for input_index, input_exponent_array in enumerate(exponents):
            value = factor[output_index, input_index]
            if abs(value) <= threshold:
                continue
            input_exponent = tuple(int(entry) for entry in input_exponent_array)
            input_row = index[input_exponent]
            rows.append(output_row * count + input_row)
            data.append(value / math.sqrt(multinomial_count(input_exponent)))
    return sparse.coo_matrix(
        (
            np.asarray(data, dtype=np.complex128),
            (rows, np.zeros(len(rows), dtype=np.int64)),
        ),
        shape=(count**2, 1),
    ).tocsr()


def truncated_symmetric_purification_cores(
    exponents: np.ndarray,
    h_matrix: np.ndarray,
    *,
    bond_dimension: int,
    relative_entry_tolerance: float = 1.0e-12,
) -> tuple[tuple[np.ndarray, ...], list[dict[str, Any]]]:
    if bond_dimension <= 0:
        raise ValueError("bond dimension must be positive")
    degree = int(np.sum(exponents[0]))
    coordinate_count = int(exponents.shape[1])
    physical_dimension = coordinate_count**2
    previous_overlaps = np.ones((1, 1), dtype=np.complex128)
    cores: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []

    for site in range(1, degree):
        cut_matrix, cut_diagnostics = compressed_purification_cut(
            exponents,
            h_matrix,
            cut=site,
            relative_entry_tolerance=relative_entry_tolerance,
        )
        used_rows = np.flatnonzero(np.diff(cut_matrix.indptr) > 0)
        projection = _extension_projection(
            previous_overlaps,
            previous_degree=site - 1,
            coordinate_count=coordinate_count,
            used_current_rows=used_rows,
        )
        residual = (projection @ cut_matrix[used_rows]).tocsr()
        left, singular_values = _leading_left_singular_vectors(
            residual,
            bond_dimension,
        )
        previous_rank = previous_overlaps.shape[1]
        kept_rank = left.shape[1]
        core = left.reshape(
            previous_rank,
            physical_dimension,
            kept_rank,
        ).transpose(0, 2, 1)
        cores.append(core)

        restricted_overlaps = projection.conj().T @ left
        current_count = len(weak_compositions(site, coordinate_count))
        previous_overlaps = np.zeros(
            (current_count**2, kept_rank),
            dtype=np.complex128,
        )
        previous_overlaps[used_rows] = restricted_overlaps
        total_squared = float(np.sum(np.abs(residual.data) ** 2))
        captured_squared = float(np.sum(singular_values**2))
        diagnostics.append(
            {
                "site": site,
                "input_bond_dimension": previous_rank,
                "output_bond_dimension": kept_rank,
                "residual_shape": [
                    int(residual.shape[0]),
                    int(residual.shape[1]),
                ],
                "residual_nonzero_entries": int(residual.nnz),
                "squared_weight": total_squared,
                "retained_squared_weight": captured_squared,
                "local_retained_fraction": captured_squared / total_squared,
                "cut_matrix": cut_diagnostics,
            }
        )

    factor = principal_positive_square_root(h_matrix)
    full_vector = _full_count_vector(
        exponents,
        factor,
        relative_entry_tolerance=relative_entry_tolerance,
    )
    used_rows = full_vector.nonzero()[0]
    projection = _extension_projection(
        previous_overlaps,
        previous_degree=degree - 1,
        coordinate_count=coordinate_count,
        used_current_rows=used_rows,
    )
    final = projection @ full_vector[used_rows]
    final_core = np.asarray(final.toarray()).reshape(
        previous_overlaps.shape[1],
        physical_dimension,
        1,
    ).transpose(0, 2, 1)
    cores.append(final_core)
    diagnostics.append(
        {
            "site": degree,
            "input_bond_dimension": int(previous_overlaps.shape[1]),
            "output_bond_dimension": 1,
            "residual_shape": [int(final.shape[0]), 1],
            "residual_nonzero_entries": int(final.nnz),
        }
    )
    return tuple(cores), diagnostics


def uniform_power_cores(
    source_cores: tuple[np.ndarray, ...],
    *,
    power: int,
    bond_dimension: int,
) -> tuple[np.ndarray, ...]:
    if power <= 0:
        raise ValueError("power must be positive")
    physical_dimension = source_cores[0].shape[2]
    rows = []
    total_sites = len(source_cores) * power
    global_site = 0
    for _ in range(power):
        for source_core in source_cores:
            left = 1 if global_site == 0 else bond_dimension
            right = 1 if global_site == total_sites - 1 else bond_dimension
            target = np.zeros(
                (left, right, physical_dimension),
                dtype=np.complex128,
            )
            source_left, source_right, _ = source_core.shape
            if source_left > left or source_right > right:
                raise ValueError("source core does not fit the uniform target bond")
            target[:source_left, :source_right] = source_core
            rows.append(target)
            global_site += 1
    return tuple(rows)


def _copy_cores_into_model(
    model: Any,
    cores: tuple[np.ndarray, ...],
) -> None:
    import torch

    if len(model.coefficient_cores) != len(cores):
        raise ValueError("model and initialization have different site counts")
    with torch.no_grad():
        for parameter, value in zip(model.coefficient_cores, cores):
            if tuple(parameter.shape) != value.shape:
                raise ValueError(
                    f"core shape mismatch: {tuple(parameter.shape)} versus {value.shape}"
                )
            parameter.copy_(
                torch.tensor(value, dtype=parameter.dtype, device=parameter.device)
            )


def main() -> None:
    import torch

    args = parse_args()
    if args.bond_dimension <= 0 or args.power <= 0:
        raise SystemExit("bond dimension and power must be positive")
    full_h_path = args.full_h.expanduser().resolve()
    template_path = args.template_model.expanduser().resolve()
    artifact = np.load(full_h_path)
    source_degree = int(artifact["degree"])
    exponents = np.asarray(artifact["exponents"], dtype=np.int64)
    h_matrix = np.asarray(artifact["global_h_matrix"], dtype=np.complex128)
    if int(np.sum(exponents[0])) != source_degree:
        raise SystemExit("full-H exponent degree is inconsistent")

    source_cores, truncation = truncated_symmetric_purification_cores(
        exponents,
        h_matrix,
        bond_dimension=args.bond_dimension,
        relative_entry_tolerance=args.relative_entry_tolerance,
    )
    target_degree = source_degree * args.power
    cores = uniform_power_cores(
        source_cores,
        power=args.power,
        bond_dimension=args.bond_dimension,
    )
    if len(cores) != target_degree:
        raise RuntimeError("power lift produced the wrong site count")

    reference_h = np.eye(exponents.shape[1], dtype=np.complex128)
    physical_dictionary = np.eye(
        exponents.shape[1] ** 2,
        dtype=np.complex128,
    ).reshape(
        exponents.shape[1] ** 2,
        exponents.shape[1],
        exponents.shape[1],
    )
    model = PositiveTensorNetworkMetric(
        reference_h,
        site_count=target_degree,
        bond_dimension=args.bond_dimension,
        target_normalization=1.0 / (math.pi * target_degree),
        output_dimension=exponents.shape[1],
        positive_floor=0.0,
        initialization_noise=0.0,
        physical_dictionary=physical_dictionary,
        trainable_physical_dictionary=False,
        dtype=torch.complex128,
        device="cpu",
    )
    _copy_cores_into_model(model, cores)

    template = torch.load(template_path, map_location="cpu", weights_only=False)
    payload = dict(template)
    payload.update(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": "shared_local_dictionary",
            "trainable_physical_dictionary": False,
            "physical_dictionary_rank": int(exponents.shape[1] ** 2),
            "fixed_canonical_matrix_units": True,
            "site_count": target_degree,
            "bond_dimension": args.bond_dimension,
            "source_degree": 1,
            "total_degree": target_degree,
            "target_normalization": 1.0 / (math.pi * target_degree),
            "output_dimension": int(exponents.shape[1]),
            "positive_floor": 0.0,
            "precision": "complex128",
            "oracle_ttsvd_initialization": {
                "full_h": str(full_h_path),
                "full_h_sha256": sha256_file(full_h_path),
                "source_degree": source_degree,
                "power": args.power,
                "bond_dimension": args.bond_dimension,
                "purification": "principal-positive-square-root",
                "monomial_embedding": "uniform-symmetric",
                "relative_entry_tolerance": args.relative_entry_tolerance,
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.out)
    report = {
        "schema": "quintic-full-h-oracle-ttsvd-initialization-v1",
        "source": {
            "full_h": str(full_h_path),
            "full_h_sha256": sha256_file(full_h_path),
            "template_model": str(template_path),
            "template_model_sha256": sha256_file(template_path),
        },
        "configuration": {
            "source_degree": source_degree,
            "power": args.power,
            "target_degree": target_degree,
            "bond_dimension": args.bond_dimension,
            "relative_entry_tolerance": args.relative_entry_tolerance,
        },
        "model": {
            "path": str(args.out.expanduser().resolve()),
            "sha256": sha256_file(args.out),
            "trainable_real_parameter_count": int(
                model.trainable_real_parameter_count
            ),
            "positive_floor": 0.0,
        },
        "truncation": truncation,
        "scope": (
            "This is a TT-SVD of one symmetry-natural principal-square-root "
            "purification. It is a constructive initialization, not a proof of "
            "the minimum rank over all function-equivalent polynomial lifts."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)
    print(f"Wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
