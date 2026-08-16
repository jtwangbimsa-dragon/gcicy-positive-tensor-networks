#!/usr/bin/env python3
"""Audit the first internal nodes below a purification-residual tree root."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from scripts.audit_quintic_full_h_oracle_multiplication_tree import (
    exact_product_right_inverse,
    quotient_multiplication_map,
)
from scripts.audit_quintic_full_h_oracle_purification_schmidt import (
    leading_schmidt_spectrum,
)
from scripts.build_quintic_full_h_multiplication_tree_truncations import (
    canonical_selected_pairs,
    parse_split,
    sorted_sparse_svd,
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
    parser.add_argument("--root-split", default="4+6")
    parser.add_argument("--root-rank", type=int, default=144)
    parser.add_argument("--maximum-singular-values", type=int, default=256)
    parser.add_argument("--exact-row-dimension", type=int, default=512)
    parser.add_argument("--relative-entry-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def child_family_unfolding(
    family: np.ndarray,
    *,
    parent_degree: int,
    left_degree: int,
    right_degree: int,
    side: str,
    relative_entry_tolerance: float,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Lift an operator family and unfold one child against all other axes."""

    if left_degree + right_degree != parent_degree:
        raise ValueError("child split does not sum to the parent degree")
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    left, right, parent, _ = quotient_multiplication_map(
        left_degree,
        right_degree,
    )
    right_inverse = exact_product_right_inverse(
        left,
        right,
        parent,
        mode="canonical-exact",
    )
    selected_left, selected_right, selected_coefficient = (
        canonical_selected_pairs(
            right_inverse,
            right_count=len(right),
        )
    )
    values = np.asarray(family, dtype=np.complex128)
    parent_operator_count = len(parent) ** 2
    if values.ndim != 2 or values.shape[0] != parent_operator_count:
        raise ValueError("family has the wrong parent operator dimension")
    channel_count = values.shape[1]
    output_index = np.repeat(np.arange(len(parent)), len(parent))
    input_index = np.tile(np.arange(len(parent)), len(parent))
    left_index = (
        selected_left[output_index] * len(left)
        + selected_left[input_index]
    )
    right_index = (
        selected_right[output_index] * len(right)
        + selected_right[input_index]
    )
    scaled = values * np.conj(selected_coefficient[input_index])[:, None]
    largest = float(np.max(np.abs(scaled)))
    keep = np.abs(scaled) > relative_entry_tolerance * largest
    parent_flat, channel = np.nonzero(keep)
    data = scaled[parent_flat, channel]
    if side == "left":
        rows = left_index[parent_flat]
        columns = right_index[parent_flat] * channel_count + channel
        shape = (len(left) ** 2, len(right) ** 2 * channel_count)
    else:
        rows = right_index[parent_flat]
        columns = left_index[parent_flat] * channel_count + channel
        shape = (len(right) ** 2, len(left) ** 2 * channel_count)
    result = sparse.coo_matrix(
        (data, (rows, columns)),
        shape=shape,
    ).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return result, {
        "parent_degree": int(parent_degree),
        "split": [int(left_degree), int(right_degree)],
        "side": side,
        "parent_channel_count": int(channel_count),
        "shape": [int(item) for item in result.shape],
        "nonzero_entries": int(result.nnz),
        "density": float(result.nnz / (result.shape[0] * result.shape[1])),
    }


def spectrum_for_child(
    family: np.ndarray,
    *,
    parent_degree: int,
    split: tuple[int, int],
    side: str,
    maximum_singular_values: int,
    exact_row_dimension: int,
    relative_entry_tolerance: float,
) -> dict[str, Any]:
    unfolding, diagnostics = child_family_unfolding(
        family,
        parent_degree=parent_degree,
        left_degree=split[0],
        right_degree=split[1],
        side=side,
        relative_entry_tolerance=relative_entry_tolerance,
    )
    spectrum = leading_schmidt_spectrum(
        unfolding,
        maximum_singular_values=maximum_singular_values,
        exact_row_dimension=exact_row_dimension,
    )
    del unfolding
    gc.collect()
    return {"diagnostics": diagnostics, "spectrum": spectrum}


def main() -> None:
    args = parse_args()
    teacher_payload = np.load(args.full_h.expanduser().resolve())
    baseline_payload = np.load(args.baseline_h.expanduser().resolve())
    degree = int(teacher_payload["degree"])
    exponents = np.asarray(teacher_payload["exponents"], dtype=np.int64)
    if int(baseline_payload["degree"]) != degree:
        raise ValueError("baseline and teacher degrees differ")
    if not np.array_equal(
        exponents,
        np.asarray(baseline_payload["exponents"], dtype=np.int64),
    ):
        raise ValueError("baseline and teacher quotient bases differ")
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

    root_left_degree, root_right_degree = parse_split(args.root_split)
    if root_left_degree + root_right_degree != degree:
        raise ValueError("root split does not sum to the teacher degree")
    root_left, root_right, root_output, _ = quotient_multiplication_map(
        root_left_degree,
        root_right_degree,
    )
    if not np.array_equal(root_output, exponents):
        raise RuntimeError("root output basis differs from teacher basis")
    root_inverse = exact_product_right_inverse(
        root_left,
        root_right,
        root_output,
        mode="canonical-exact",
    )
    root_matrix, root_diagnostics = lifted_purification_schmidt_matrix(
        residual,
        root_inverse,
        left_count=len(root_left),
        right_count=len(root_right),
        relative_entry_tolerance=args.relative_entry_tolerance,
    )
    left_vectors, singular_values, right_vectors = sorted_sparse_svd(
        root_matrix,
        rank=args.root_rank,
    )
    total_squared = float(np.sum(np.abs(root_matrix.data) ** 2))
    root_captured = float(
        np.sum(singular_values**2) / total_squared
    )
    weighted_left_family = left_vectors * singular_values[None, :]
    weighted_right_family = right_vectors.T * singular_values[None, :]
    del root_matrix, left_vectors, right_vectors
    gc.collect()

    if root_left_degree != 4 or root_right_degree != 6:
        raise NotImplementedError(
            "the first audit is frozen to the 4+6 tree topology"
        )
    nodes = {
        "degree4_left_child_degree2": spectrum_for_child(
            weighted_left_family,
            parent_degree=4,
            split=(2, 2),
            side="left",
            maximum_singular_values=args.maximum_singular_values,
            exact_row_dimension=args.exact_row_dimension,
            relative_entry_tolerance=args.relative_entry_tolerance,
        ),
        "degree4_right_child_degree2": spectrum_for_child(
            weighted_left_family,
            parent_degree=4,
            split=(2, 2),
            side="right",
            maximum_singular_values=args.maximum_singular_values,
            exact_row_dimension=args.exact_row_dimension,
            relative_entry_tolerance=args.relative_entry_tolerance,
        ),
        "degree6_left_child_degree2": spectrum_for_child(
            weighted_right_family,
            parent_degree=6,
            split=(2, 4),
            side="left",
            maximum_singular_values=args.maximum_singular_values,
            exact_row_dimension=args.exact_row_dimension,
            relative_entry_tolerance=args.relative_entry_tolerance,
        ),
        "degree6_right_child_degree4": spectrum_for_child(
            weighted_right_family,
            parent_degree=6,
            split=(2, 4),
            side="right",
            maximum_singular_values=args.maximum_singular_values,
            exact_row_dimension=args.exact_row_dimension,
            relative_entry_tolerance=args.relative_entry_tolerance,
        ),
    }
    for name, row in nodes.items():
        print(
            name,
            row["spectrum"]["rank_for_cumulative_squared_weight"],
            flush=True,
        )

    root_direct_real_parameters = int(
        2
        * args.root_rank
        * (len(root_left) ** 2 + len(root_right) ** 2)
        + args.root_rank
    )
    report = {
        "schema": "quintic-purification-residual-hierarchical-node-audit-v1",
        "configuration": {
            "root_split": [root_left_degree, root_right_degree],
            "root_rank": int(args.root_rank),
            "maximum_singular_values": int(args.maximum_singular_values),
            "exact_row_dimension": int(args.exact_row_dimension),
            "relative_entry_tolerance": float(args.relative_entry_tolerance),
        },
        "root": {
            "diagnostics": root_diagnostics,
            "captured_squared_weight": root_captured,
            "dense_factor_real_parameter_count": root_direct_real_parameters,
        },
        "nodes": nodes,
        "scope": (
            "This audits only the first internal level. It does not yet "
            "construct or metric-test a fully truncated hierarchical tree."
        ),
    }
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
