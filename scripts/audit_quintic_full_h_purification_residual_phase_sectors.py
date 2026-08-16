#!/usr/bin/env python3
"""Audit phase-sector parameter counts of a purification-residual tree root."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from scripts.audit_quintic_full_h_oracle_multiplication_tree import (
    exact_product_right_inverse,
    quotient_multiplication_map,
)
from scripts.build_quintic_full_h_multiplication_tree_truncations import (
    parse_split,
)
from scripts.build_quintic_full_h_purification_residual_tree_truncations import (
    align_factor_by_left_unitary,
    lifted_purification_schmidt_matrix,
    principal_positive_factor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-h", type=Path, required=True)
    parser.add_argument("--baseline-h", type=Path, required=True)
    parser.add_argument("--baseline-h-key", default="global_h_matrix")
    parser.add_argument("--split", default="4+6")
    parser.add_argument("--rank", type=int, nargs="+", default=(96, 128, 144, 160))
    parser.add_argument("--relative-entry-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def operator_phase_charges(exponents: np.ndarray) -> np.ndarray:
    """Return the Fermat phase charge of each row-major matrix unit."""

    values = np.asarray(exponents, dtype=np.int64)
    return (
        values[:, None, :] - values[None, :, :]
    ).reshape(len(values) ** 2, values.shape[1]) % 5


def phase_sector_singular_values(
    matrix: sparse.csr_matrix,
    *,
    row_exponents: np.ndarray,
    column_exponents: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    """Compute exact singular values in every conserved phase-charge block."""

    row_charges = operator_phase_charges(row_exponents)
    column_charges = operator_phase_charges(column_exponents)
    row_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    column_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, charge in enumerate(row_charges):
        row_groups[tuple(map(int, charge))].append(index)
    for index, charge in enumerate(column_charges):
        column_groups[tuple(map(int, charge))].append(index)

    coo = matrix.tocoo()
    violation = (
        row_charges[coo.row] + column_charges[coo.col]
    ) % 5
    maximum_violation_entry = float(
        np.max(
            np.abs(coo.data[np.any(violation != 0, axis=1)]),
            initial=0.0,
        )
    )
    rows = []
    for charge, row_indices_list in sorted(row_groups.items()):
        partner = tuple(
            int(value)
            for value in ((-np.asarray(charge, dtype=np.int64)) % 5)
        )
        column_indices_list = column_groups.get(partner)
        if not column_indices_list:
            continue
        row_indices = np.asarray(row_indices_list, dtype=np.int64)
        column_indices = np.asarray(column_indices_list, dtype=np.int64)
        block = matrix[row_indices][:, column_indices]
        if block.nnz == 0:
            continue
        singular_values = np.linalg.svd(
            block.toarray(),
            compute_uv=False,
            full_matrices=False,
        )
        tolerance = (
            np.finfo(np.float64).eps
            * max(block.shape)
            * singular_values[0]
        )
        singular_values = singular_values[singular_values > tolerance]
        rows.append(
            {
                "charge": list(charge),
                "partner_charge": list(partner),
                "row_dimension": int(len(row_indices)),
                "column_dimension": int(len(column_indices)),
                "nonzero_entries": int(block.nnz),
                "singular_values": singular_values.tolist(),
            }
        )
    return rows, maximum_violation_entry


def selected_sector_parameter_count(
    sectors: list[dict[str, Any]],
    *,
    rank: int,
) -> dict[str, Any]:
    entries = []
    for sector_index, sector in enumerate(sectors):
        entries.extend(
            (float(value), sector_index, local_index)
            for local_index, value in enumerate(sector["singular_values"])
        )
    entries.sort(key=lambda item: item[0], reverse=True)
    if not 0 < rank <= len(entries):
        raise ValueError("requested rank exceeds exact phase-sector rank")
    selected = entries[:rank]
    multiplicities: dict[int, int] = defaultdict(int)
    for _, sector_index, _ in selected:
        multiplicities[sector_index] += 1
    complex_factor_parameters = 0
    rows = []
    for sector_index, multiplicity in sorted(multiplicities.items()):
        sector = sectors[sector_index]
        complex_count = multiplicity * (
            sector["row_dimension"] + sector["column_dimension"]
        )
        complex_factor_parameters += complex_count
        rows.append(
            {
                "charge": sector["charge"],
                "multiplicity": int(multiplicity),
                "row_dimension": int(sector["row_dimension"]),
                "column_dimension": int(sector["column_dimension"]),
                "complex_factor_parameters": int(complex_count),
            }
        )
    all_squared = np.asarray(
        [entry[0] ** 2 for entry in entries],
        dtype=np.float64,
    )
    return {
        "rank": int(rank),
        "captured_squared_weight": float(
            np.sum(all_squared[:rank]) / np.sum(all_squared)
        ),
        "active_sector_count": int(len(rows)),
        "complex_factor_parameters": int(complex_factor_parameters),
        "real_parameter_count_including_singular_values": int(
            2 * complex_factor_parameters + rank
        ),
        "sector_multiplicities": rows,
    }


def main() -> None:
    args = parse_args()
    teacher_payload = np.load(args.full_h.expanduser().resolve())
    baseline_payload = np.load(args.baseline_h.expanduser().resolve())
    degree = int(teacher_payload["degree"])
    output_exponents = np.asarray(
        teacher_payload["exponents"],
        dtype=np.int64,
    )
    if int(baseline_payload["degree"]) != degree:
        raise ValueError("baseline and teacher degrees differ")
    teacher_h = np.asarray(
        teacher_payload["global_h_matrix"],
        dtype=np.complex128,
    )
    baseline_h = np.asarray(
        baseline_payload[args.baseline_h_key],
        dtype=np.complex128,
    )
    teacher_factor = principal_positive_factor(teacher_h)
    baseline_factor_unscaled = principal_positive_factor(baseline_h)
    aligned_teacher, _ = align_factor_by_left_unitary(
        teacher_factor,
        baseline_factor_unscaled,
    )
    baseline_scale = float(
        np.real(np.vdot(baseline_factor_unscaled, aligned_teacher))
        / np.real(
            np.vdot(baseline_factor_unscaled, baseline_factor_unscaled)
        )
    )
    residual = aligned_teacher - baseline_scale * baseline_factor_unscaled

    left_degree, right_degree = parse_split(args.split)
    left, right, observed_output, _ = quotient_multiplication_map(
        left_degree,
        right_degree,
    )
    if not np.array_equal(observed_output, output_exponents):
        raise RuntimeError("tree output basis differs from teacher basis")
    right_inverse = exact_product_right_inverse(
        left,
        right,
        observed_output,
        mode="canonical-exact",
    )
    schmidt, diagnostics = lifted_purification_schmidt_matrix(
        residual,
        right_inverse,
        left_count=len(left),
        right_count=len(right),
        relative_entry_tolerance=args.relative_entry_tolerance,
    )
    sectors, maximum_violation_entry = phase_sector_singular_values(
        schmidt,
        row_exponents=left,
        column_exponents=right,
    )
    total_exact_rank = int(
        sum(len(sector["singular_values"]) for sector in sectors)
    )
    requested = sorted(set(args.rank))
    rows = [
        selected_sector_parameter_count(sectors, rank=rank)
        for rank in requested
    ]
    dense_rows = [
        {
            "rank": rank,
            "real_parameter_count_including_singular_values": int(
                2
                * rank
                * (diagnostics["shape"][0] + diagnostics["shape"][1])
                + rank
            ),
        }
        for rank in requested
    ]
    report = {
        "schema": "quintic-purification-residual-phase-sector-root-audit-v1",
        "configuration": {
            "split": [left_degree, right_degree],
            "ranks": requested,
            "relative_entry_tolerance": args.relative_entry_tolerance,
        },
        "schmidt_matrix": diagnostics,
        "phase_conservation": {
            "maximum_violating_entry": maximum_violation_entry,
            "active_sector_count": int(len(sectors)),
            "exact_rank": total_exact_rank,
        },
        "phase_sector_parameter_counts": rows,
        "dense_parameter_counts": dense_rows,
        "sectors": sectors,
        "scope": (
            "Counts exploit exact phase-charge blocks only. S5 orbit tying "
            "and recursive child compression are not included."
        ),
    }
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"rank={row['rank']} sectors={row['active_sector_count']} "
            f"real_parameters="
            f"{row['real_parameter_count_including_singular_values']} "
            f"captured={row['captured_squared_weight']:.9f}",
            flush=True,
        )
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
