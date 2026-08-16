#!/usr/bin/env python3
"""Pack and verify a compact phase-sector purification-residual root."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from gcicy_metric.pipeline.fermat_full_h import fermat_reynolds_project_h
from scripts.audit_quintic_full_h_oracle_multiplication_tree import (
    exact_product_right_inverse,
    quotient_multiplication_map,
)
from scripts.audit_quintic_full_h_purification_residual_phase_sectors import (
    operator_phase_charges,
)
from scripts.build_quintic_full_h_multiplication_tree_truncations import (
    canonical_selected_pairs,
    matrix_summary,
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
    parser.add_argument("--reference-h", type=Path)
    parser.add_argument("--split", default="4+6")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--relative-entry-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-h", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_sector_svd_blocks(
    matrix: sparse.csr_matrix,
    *,
    row_exponents: np.ndarray,
    column_exponents: np.ndarray,
) -> list[dict[str, Any]]:
    """Return exact SVD factors for every conserved phase-charge block."""

    row_charges = operator_phase_charges(row_exponents)
    column_charges = operator_phase_charges(column_exponents)
    row_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    column_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, charge in enumerate(row_charges):
        row_groups[tuple(map(int, charge))].append(index)
    for index, charge in enumerate(column_charges):
        column_groups[tuple(map(int, charge))].append(index)

    blocks = []
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
        left, singular, right = np.linalg.svd(
            block.toarray(),
            full_matrices=False,
        )
        tolerance = (
            np.finfo(np.float64).eps
            * max(block.shape)
            * singular[0]
        )
        keep = singular > tolerance
        blocks.append(
            {
                "charge": charge,
                "partner_charge": partner,
                "row_indices": row_indices,
                "column_indices": column_indices,
                "left": left[:, keep],
                "singular": singular[keep],
                "right": right[keep],
            }
        )
    return blocks


def truncate_phase_sector_blocks(
    blocks: list[dict[str, Any]],
    *,
    rank: int,
) -> tuple[list[dict[str, Any]], float]:
    entries = []
    for block_index, block in enumerate(blocks):
        entries.extend(
            (float(value), block_index, local_index)
            for local_index, value in enumerate(block["singular"])
        )
    entries.sort(key=lambda item: item[0], reverse=True)
    if not 0 < rank <= len(entries):
        raise ValueError("requested rank exceeds exact phase-sector rank")
    counts: dict[int, int] = defaultdict(int)
    for _, block_index, _ in entries[:rank]:
        counts[block_index] += 1
    selected = []
    for block_index, count in sorted(counts.items()):
        block = blocks[block_index]
        selected.append(
            {
                "charge": block["charge"],
                "partner_charge": block["partner_charge"],
                "row_indices": block["row_indices"],
                "column_indices": block["column_indices"],
                "left": block["left"][:, :count],
                "singular": block["singular"][:count],
                "right": block["right"][:count],
            }
        )
    all_squared = np.asarray(
        [entry[0] ** 2 for entry in entries],
        dtype=np.float64,
    )
    captured = float(np.sum(all_squared[:rank]) / np.sum(all_squared))
    return selected, captured


def reconstruct_schmidt_from_blocks(
    blocks: list[dict[str, Any]],
    *,
    shape: tuple[int, int],
) -> sparse.csr_matrix:
    rows = []
    columns = []
    data = []
    for block in blocks:
        dense = (
            block["left"] * block["singular"][None, :]
        ) @ block["right"]
        row_grid, column_grid = np.meshgrid(
            block["row_indices"],
            block["column_indices"],
            indexing="ij",
        )
        rows.append(row_grid.reshape(-1))
        columns.append(column_grid.reshape(-1))
        data.append(dense.reshape(-1))
    result = sparse.coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=shape,
    ).tocsr()
    result.eliminate_zeros()
    return result


def reconstruct_section_factor_from_schmidt(
    schmidt: sparse.csr_matrix,
    right_inverse: sparse.csr_matrix,
    *,
    left_count: int,
    right_count: int,
) -> np.ndarray:
    selected_left, selected_right, selected_coefficient = (
        canonical_selected_pairs(
            right_inverse,
            right_count=right_count,
        )
    )
    output_index = np.repeat(
        np.arange(len(selected_left)),
        len(selected_left),
    )
    input_index = np.tile(
        np.arange(len(selected_left)),
        len(selected_left),
    )
    rows = (
        selected_left[output_index] * left_count
        + selected_left[input_index]
    )
    columns = (
        selected_right[output_index] * right_count
        + selected_right[input_index]
    )
    lifted = np.asarray(schmidt[rows, columns]).reshape(-1)
    return (
        lifted.reshape(len(selected_left), len(selected_left))
        / np.conj(selected_coefficient[None, :])
    )


def compact_real_parameter_count(blocks: list[dict[str, Any]]) -> int:
    complex_count = sum(
        block["left"].size + block["right"].size
        for block in blocks
    )
    singular_count = sum(block["singular"].size for block in blocks)
    return int(2 * complex_count + singular_count)


def compact_payload(
    blocks: list[dict[str, Any]],
    *,
    metadata: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    payload = dict(metadata)
    payload["sector_count"] = np.asarray(len(blocks), dtype=np.int64)
    for index, block in enumerate(blocks):
        prefix = f"sector_{index:03d}"
        payload[f"{prefix}_charge"] = np.asarray(block["charge"], dtype=np.int64)
        payload[f"{prefix}_partner_charge"] = np.asarray(
            block["partner_charge"],
            dtype=np.int64,
        )
        payload[f"{prefix}_row_indices"] = block["row_indices"]
        payload[f"{prefix}_column_indices"] = block["column_indices"]
        payload[f"{prefix}_left"] = block["left"]
        payload[f"{prefix}_singular"] = block["singular"]
        payload[f"{prefix}_right"] = block["right"]
    return payload


def main() -> None:
    args = parse_args()
    teacher_path = args.full_h.expanduser().resolve()
    baseline_path = args.baseline_h.expanduser().resolve()
    teacher_payload = np.load(teacher_path)
    baseline_payload = np.load(baseline_path)
    degree = int(teacher_payload["degree"])
    exponents = np.asarray(teacher_payload["exponents"], dtype=np.int64)
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
    aligned_teacher, unitary = align_factor_by_left_unitary(
        teacher_factor,
        baseline_factor_unscaled,
    )
    baseline_scale = float(
        np.real(np.vdot(baseline_factor_unscaled, aligned_teacher))
        / np.real(
            np.vdot(baseline_factor_unscaled, baseline_factor_unscaled)
        )
    )
    baseline_factor = baseline_scale * baseline_factor_unscaled
    residual = aligned_teacher - baseline_factor

    left_degree, right_degree = parse_split(args.split)
    left, right, output, _ = quotient_multiplication_map(
        left_degree,
        right_degree,
    )
    if not np.array_equal(output, exponents):
        raise RuntimeError("tree output basis differs from teacher basis")
    right_inverse = exact_product_right_inverse(
        left,
        right,
        output,
        mode="canonical-exact",
    )
    schmidt, diagnostics = lifted_purification_schmidt_matrix(
        residual,
        right_inverse,
        left_count=len(left),
        right_count=len(right),
        relative_entry_tolerance=args.relative_entry_tolerance,
    )
    blocks = phase_sector_svd_blocks(
        schmidt,
        row_exponents=left,
        column_exponents=right,
    )
    selected, captured = truncate_phase_sector_blocks(
        blocks,
        rank=args.rank,
    )
    reconstructed_schmidt = reconstruct_schmidt_from_blocks(
        selected,
        shape=schmidt.shape,
    )
    correction = reconstruct_section_factor_from_schmidt(
        reconstructed_schmidt,
        right_inverse,
        left_count=len(left),
        right_count=len(right),
    )
    candidate_factor = baseline_factor + correction
    raw_h = candidate_factor.conj().T @ candidate_factor
    raw_h *= np.trace(teacher_h).real / np.trace(raw_h).real
    reynolds_h = fermat_reynolds_project_h(
        exponents,
        raw_h,
        conjugation_invariant=True,
    )

    reference = None
    if args.reference_h is not None:
        reference_payload = np.load(args.reference_h.expanduser().resolve())
        reference_h = np.asarray(
            reference_payload["global_h_matrix"],
            dtype=np.complex128,
        )
        reference = {
            "path": str(args.reference_h.expanduser().resolve()),
            "sha256": sha256_file(args.reference_h.expanduser().resolve()),
            "relative_frobenius_error": float(
                np.linalg.norm(reynolds_h - reference_h)
                / np.linalg.norm(reference_h)
            ),
        }

    output_model = args.output_model.expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "degree": np.asarray(degree, dtype=np.int64),
        "exponents": exponents,
        "split": np.asarray((left_degree, right_degree), dtype=np.int64),
        "rank": np.asarray(args.rank, dtype=np.int64),
        "baseline_factor": baseline_factor,
        "baseline_amplitude_scale": np.asarray(baseline_scale),
        "purification_unitary": unitary,
    }
    np.savez_compressed(
        output_model,
        **compact_payload(selected, metadata=metadata),
    )
    output_h = args.output_h.expanduser().resolve()
    output_h.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_h,
        degree=np.asarray(degree, dtype=np.int64),
        exponents=exponents,
        global_h_matrix=reynolds_h,
        compact_model_sha256=np.asarray(sha256_file(output_model)),
    )
    report = {
        "schema": "quintic-compact-phase-sector-purification-residual-v1",
        "configuration": {
            "split": [left_degree, right_degree],
            "rank": int(args.rank),
            "relative_entry_tolerance": args.relative_entry_tolerance,
        },
        "schmidt_matrix": diagnostics,
        "captured_squared_weight": captured,
        "active_sector_count": int(len(selected)),
        "trainable_real_parameter_count": compact_real_parameter_count(selected),
        "baseline_trainable_parameter_count": 7,
        "total_trainable_real_parameter_count": int(
            compact_real_parameter_count(selected) + 7
        ),
        "raw_h": matrix_summary(raw_h, teacher_h),
        "positive_fermat_reynolds": matrix_summary(reynolds_h, teacher_h),
        "reference_reproduction": reference,
        "artifacts": {
            "compact_model": {
                "path": str(output_model),
                "sha256": sha256_file(output_model),
                "bytes": output_model.stat().st_size,
            },
            "reconstructed_h": {
                "path": str(output_h),
                "sha256": sha256_file(output_h),
                "bytes": output_h.stat().st_size,
            },
        },
        "positivity": (
            "The compact factors define B=B0+DeltaB and H=B^*B. Exact "
            "S5/conjugation Reynolds averaging preserves positivity."
        ),
    }
    output_report = args.out.expanduser().resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
