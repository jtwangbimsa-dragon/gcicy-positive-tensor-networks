#!/usr/bin/env python3
"""Initialize symmetry-allowed random root channels above a positive skeleton."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.fermat_full_h import (  # noqa: E402
    FermatReynoldsBlockProjector,
)
from scripts.audit_quintic_full_h_oracle_multiplication_tree import (  # noqa: E402
    exact_product_right_inverse,
    quotient_multiplication_map,
)
from scripts.audit_quintic_full_h_purification_residual_phase_sectors import (  # noqa: E402
    operator_phase_charges,
)
from scripts.build_quintic_compact_phase_sector_purification_residual import (  # noqa: E402
    reconstruct_schmidt_from_blocks,
    reconstruct_section_factor_from_schmidt,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton-template", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--seed", type=int, default=202607356)
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


def orthonormal_columns(
    rng: np.random.Generator,
    rows: int,
    columns: int,
) -> np.ndarray:
    value = rng.normal(size=(rows, columns)) + 1j * rng.normal(
        size=(rows, columns)
    )
    q, _ = np.linalg.qr(value)
    return np.asarray(q[:, :columns], dtype=np.complex128)


def all_phase_blocks(
    row_exponents: np.ndarray,
    column_exponents: np.ndarray,
) -> list[dict[str, Any]]:
    row_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    column_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, charge in enumerate(operator_phase_charges(row_exponents)):
        row_groups[tuple(map(int, charge))].append(index)
    for index, charge in enumerate(operator_phase_charges(column_exponents)):
        column_groups[tuple(map(int, charge))].append(index)
    rows = []
    for charge, row_indices in row_groups.items():
        partner = tuple(
            int(value)
            for value in ((-np.asarray(charge, dtype=np.int64)) % 5)
        )
        column_indices = column_groups.get(partner)
        if column_indices:
            rows.append(
                {
                    "charge": charge,
                    "partner_charge": partner,
                    "row_indices": np.asarray(row_indices, dtype=np.int64),
                    "column_indices": np.asarray(column_indices, dtype=np.int64),
                    "structural_size": len(row_indices) * len(column_indices),
                }
            )
    rows.sort(
        key=lambda row: (
            -row["structural_size"],
            row["charge"],
        )
    )
    return rows


def materialize_blocks(
    blocks: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    size: int,
) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.complex128)
    for indices, values in blocks:
        for block_indices, block in zip(indices, values):
            result[np.ix_(block_indices, block_indices)] = block
    return result


def main() -> None:
    args = parse_args()
    if args.rank <= 0:
        raise ValueError("rank must be positive")
    template_path = args.skeleton_template.expanduser().resolve()
    template = np.load(template_path, allow_pickle=False)
    degree = int(template["degree"])
    exponents = np.asarray(template["exponents"], dtype=np.int64)
    split = tuple(int(value) for value in template["split"])
    baseline_factor = np.asarray(
        template["baseline_factor"],
        dtype=np.complex128,
    )
    left, right, output, _ = quotient_multiplication_map(*split)
    if not np.array_equal(output, exponents):
        raise RuntimeError("skeleton template uses an inconsistent quotient basis")
    right_inverse = exact_product_right_inverse(
        left,
        right,
        output,
        mode="canonical-exact",
    )
    candidates = all_phase_blocks(left, right)
    rng = np.random.default_rng(args.seed)
    blocks = []
    direction_norms = []
    skipped_unreachable = []
    target_direction_norm = float(
        np.linalg.norm(baseline_factor) / math.sqrt(args.rank)
    )
    for row in candidates:
        left_vector = orthonormal_columns(
            rng,
            len(row["row_indices"]),
            1,
        )
        right_vector = np.conj(
            orthonormal_columns(
                rng,
                len(row["column_indices"]),
                1,
            ).T
        )
        probe_block = {
            "row_indices": row["row_indices"],
            "column_indices": row["column_indices"],
            "left": left_vector,
            "singular": np.ones(1, dtype=np.float64),
            "right": right_vector,
        }
        schmidt = reconstruct_schmidt_from_blocks(
            [probe_block],
            shape=(len(left) ** 2, len(right) ** 2),
        )
        direction = reconstruct_section_factor_from_schmidt(
            schmidt,
            right_inverse,
            left_count=len(left),
            right_count=len(right),
        )
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm):
            raise RuntimeError("random root channel has nonfinite norm")
        if direction_norm <= np.finfo(float).tiny:
            skipped_unreachable.append(row["charge"])
            continue
        direction_norms.append(direction_norm)
        blocks.append(
            {
                **row,
                "left": left_vector,
                "singular": np.zeros(1, dtype=np.float64),
                "right": right_vector,
                "coordinate_scale": np.asarray(
                    [target_direction_norm / direction_norm],
                    dtype=np.float64,
                ),
            }
        )
        if len(blocks) == args.rank:
            break
    if len(blocks) != args.rank:
        raise RuntimeError(
            f"only {len(blocks)} multiplication-map-reachable sectors were "
            f"found for requested rank {args.rank}"
        )

    output_model = args.output_model.expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "degree": np.asarray(degree, dtype=np.int64),
        "exponents": exponents,
        "split": np.asarray(split, dtype=np.int64),
        "rank": np.asarray(args.rank, dtype=np.int64),
        "baseline_factor": baseline_factor,
        "baseline_amplitude_scale": np.asarray(
            float(template["baseline_amplitude_scale"])
        ),
        "purification_unitary": np.asarray(
            template["purification_unitary"],
            dtype=np.complex128,
        ),
        "sector_count": np.asarray(len(blocks), dtype=np.int64),
    }
    for index, block in enumerate(blocks):
        prefix = f"sector_{index:03d}"
        payload[f"{prefix}_charge"] = np.asarray(
            block["charge"],
            dtype=np.int64,
        )
        payload[f"{prefix}_partner_charge"] = np.asarray(
            block["partner_charge"],
            dtype=np.int64,
        )
        for key in (
            "row_indices",
            "column_indices",
            "left",
            "singular",
            "right",
            "coordinate_scale",
        ):
            payload[f"{prefix}_{key}"] = np.asarray(block[key])
    np.savez_compressed(output_model, **payload)

    raw_h = baseline_factor.conj().T @ baseline_factor
    projector = FermatReynoldsBlockProjector(
        exponents,
        conjugation_invariant=True,
    )
    projected_blocks = projector.project_blocks(raw_h)
    skeleton_h = materialize_blocks(projected_blocks, size=len(exponents))
    skeleton_h *= len(exponents) / np.trace(skeleton_h).real
    output_h = args.output_h.expanduser().resolve()
    output_h.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_h,
        degree=np.asarray(degree, dtype=np.int64),
        exponents=exponents,
        global_h_matrix=skeleton_h,
        compact_model_sha256=np.asarray(sha256_file(output_model)),
    )

    report = {
        "schema": "quintic-random-symmetry-allowed-compact-root-v1",
        "scientific_scope": {
            "baseline": "fixed positive lifted k=5 skeleton",
            "channel_selection": (
                "48 largest symmetry-allowed phase blocks by structural "
                "row_count*column_count; no metric proposal is read"
            ),
            "channel_vectors": "seeded independent complex Gaussian QR vectors",
            "initial_correction": "exactly zero",
            "coordinate_whitening": (
                "each additive coefficient has equal section-factor Frobenius "
                "norm at unit coordinate"
            ),
        },
        "configuration": {
            "degree": degree,
            "split": list(split),
            "rank": args.rank,
            "seed": args.seed,
            "unreachable_structural_sectors_skipped": len(skipped_unreachable),
        },
        "normalization": {
            "baseline_factor_frobenius_norm": float(
                np.linalg.norm(baseline_factor)
            ),
            "target_unit_coordinate_direction_norm": target_direction_norm,
            "raw_direction_norm_minimum": float(np.min(direction_norms)),
            "raw_direction_norm_median": float(np.median(direction_norms)),
            "raw_direction_norm_maximum": float(np.max(direction_norms)),
        },
        "selected_sectors": [
            {
                "charge": list(block["charge"]),
                "partner_charge": list(block["partner_charge"]),
                "row_dimension": int(len(block["row_indices"])),
                "column_dimension": int(len(block["column_indices"])),
                "structural_size": int(block["structural_size"]),
            }
            for block in blocks
        ],
        "artifacts": {
            "compact_model": {
                "path": str(output_model),
                "sha256": sha256_file(output_model),
            },
            "skeleton_h": {
                "path": str(output_h),
                "sha256": sha256_file(output_h),
            },
        },
    }
    write_json(args.out.expanduser().resolve(), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
