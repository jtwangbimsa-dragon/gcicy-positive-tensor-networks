#!/usr/bin/env python3
"""Append zero-initialized random symmetry-allowed channels to a compact root."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_quintic_full_h_oracle_multiplication_tree import (  # noqa: E402
    exact_product_right_inverse,
    quotient_multiplication_map,
)
from scripts.build_quintic_compact_phase_sector_purification_residual import (  # noqa: E402
    reconstruct_schmidt_from_blocks,
    reconstruct_section_factor_from_schmidt,
)
from scripts.initialize_quintic_random_compact_root import (  # noqa: E402
    all_phase_blocks,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-compact", type=Path, required=True)
    parser.add_argument("--base-h", type=Path, required=True)
    parser.add_argument("--add-rank", type=int, default=48)
    parser.add_argument("--seed", type=int, default=202607360)
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


def orthogonal_vector(
    rng: np.random.Generator,
    dimension: int,
    existing: np.ndarray,
) -> np.ndarray | None:
    basis = np.asarray(existing, dtype=np.complex128)
    if basis.size:
        basis, _ = np.linalg.qr(basis)
    else:
        basis = np.empty((dimension, 0), dtype=np.complex128)
    if basis.shape[1] >= dimension:
        return None
    for _ in range(16):
        value = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        if basis.shape[1]:
            value = value - basis @ (np.conj(basis.T) @ value)
        norm = float(np.linalg.norm(value))
        if norm > 1.0e-10:
            return np.asarray(value / norm, dtype=np.complex128)
    raise RuntimeError("failed to sample a vector outside the existing subspace")


def existing_sector_vectors(
    blocks: list[dict[str, Any]],
    charge: tuple[int, ...],
    *,
    row_dimension: int,
    column_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    left_rows = []
    right_rows = []
    for block in blocks:
        if block["charge"] == charge:
            left_rows.append(block["left"])
            right_rows.append(np.conj(block["right"].T))
    left = (
        np.concatenate(left_rows, axis=1)
        if left_rows
        else np.empty((row_dimension, 0), dtype=np.complex128)
    )
    right = (
        np.concatenate(right_rows, axis=1)
        if right_rows
        else np.empty((column_dimension, 0), dtype=np.complex128)
    )
    return left, right


def load_blocks(payload: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for index in range(int(payload["sector_count"])):
        prefix = f"sector_{index:03d}"
        singular = np.asarray(payload[f"{prefix}_singular"], dtype=np.float64)
        rows.append(
            {
                "charge": tuple(
                    map(int, np.asarray(payload[f"{prefix}_charge"]))
                ),
                "partner_charge": tuple(
                    map(
                        int,
                        np.asarray(payload[f"{prefix}_partner_charge"]),
                    )
                ),
                "row_indices": np.asarray(
                    payload[f"{prefix}_row_indices"],
                    dtype=np.int64,
                ),
                "column_indices": np.asarray(
                    payload[f"{prefix}_column_indices"],
                    dtype=np.int64,
                ),
                "left": np.asarray(
                    payload[f"{prefix}_left"],
                    dtype=np.complex128,
                ),
                "singular": singular,
                "right": np.asarray(
                    payload[f"{prefix}_right"],
                    dtype=np.complex128,
                ),
                "coordinate_scale": np.asarray(
                    payload.get(
                        f"{prefix}_coordinate_scale",
                        np.ones_like(singular),
                    ),
                    dtype=np.float64,
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.add_rank <= 0:
        raise ValueError("added rank must be positive")
    base_path = args.base_compact.expanduser().resolve()
    base_h_path = args.base_h.expanduser().resolve()
    loaded = np.load(base_path, allow_pickle=False)
    payload = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    degree = int(payload["degree"])
    exponents = np.asarray(payload["exponents"], dtype=np.int64)
    split = tuple(int(value) for value in payload["split"])
    baseline_factor = np.asarray(payload["baseline_factor"], dtype=np.complex128)
    blocks = load_blocks(payload)
    old_rank = sum(len(block["singular"]) for block in blocks)
    if old_rank != int(payload["rank"]):
        raise RuntimeError("base compact rank is inconsistent")

    left, right, output, _ = quotient_multiplication_map(*split)
    if not np.array_equal(output, exponents):
        raise RuntimeError("base compact quotient basis is inconsistent")
    right_inverse = exact_product_right_inverse(
        left,
        right,
        output,
        mode="canonical-exact",
    )
    candidates = all_phase_blocks(left, right)
    represented = {block["charge"] for block in blocks}
    candidates.sort(
        key=lambda row: (
            row["charge"] in represented,
            -row["structural_size"],
            row["charge"],
        )
    )

    rng = np.random.default_rng(args.seed)
    target_direction_norm = float(
        np.linalg.norm(baseline_factor) / math.sqrt(args.add_rank)
    )
    added = []
    skipped_unreachable = 0
    for row in candidates:
        existing_left, existing_right = existing_sector_vectors(
            blocks + added,
            row["charge"],
            row_dimension=len(row["row_indices"]),
            column_dimension=len(row["column_indices"]),
        )
        left_vector = orthogonal_vector(
            rng,
            len(row["row_indices"]),
            existing_left,
        )
        right_vector = orthogonal_vector(
            rng,
            len(row["column_indices"]),
            existing_right,
        )
        if left_vector is None or right_vector is None:
            continue
        probe = {
            "row_indices": row["row_indices"],
            "column_indices": row["column_indices"],
            "left": left_vector[:, None],
            "singular": np.ones(1, dtype=np.float64),
            "right": np.conj(right_vector[None, :]),
        }
        schmidt = reconstruct_schmidt_from_blocks(
            [probe],
            shape=(len(left) ** 2, len(right) ** 2),
        )
        direction = reconstruct_section_factor_from_schmidt(
            schmidt,
            right_inverse,
            left_count=len(left),
            right_count=len(right),
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= np.finfo(float).tiny:
            skipped_unreachable += 1
            continue
        added.append(
            {
                **row,
                "left": left_vector[:, None],
                "singular": np.zeros(1, dtype=np.float64),
                "right": np.conj(right_vector[None, :]),
                "coordinate_scale": np.asarray(
                    [target_direction_norm / direction_norm],
                    dtype=np.float64,
                ),
                "raw_direction_norm": direction_norm,
            }
        )
        if len(added) == args.add_rank:
            break
    if len(added) != args.add_rank:
        raise RuntimeError(
            f"only {len(added)} new reachable channels were found for "
            f"requested added rank {args.add_rank}"
        )

    old_sector_count = int(payload["sector_count"])
    payload["rank"] = np.asarray(old_rank + args.add_rank, dtype=np.int64)
    payload["sector_count"] = np.asarray(
        old_sector_count + args.add_rank,
        dtype=np.int64,
    )
    for offset, block in enumerate(added):
        prefix = f"sector_{old_sector_count + offset:03d}"
        for key in (
            "charge",
            "partner_charge",
            "row_indices",
            "column_indices",
            "left",
            "singular",
            "right",
            "coordinate_scale",
        ):
            dtype = (
                np.int64
                if key
                in {
                    "charge",
                    "partner_charge",
                    "row_indices",
                    "column_indices",
                }
                else None
            )
            payload[f"{prefix}_{key}"] = np.asarray(block[key], dtype=dtype)

    output_model = args.output_model.expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_model, **payload)
    base_h_payload = np.load(base_h_path, allow_pickle=False)
    base_h = np.asarray(
        base_h_payload["global_h_matrix"],
        dtype=np.complex128,
    )
    output_h = args.output_h.expanduser().resolve()
    output_h.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_h,
        degree=np.asarray(degree, dtype=np.int64),
        exponents=exponents,
        global_h_matrix=base_h,
        compact_model_sha256=np.asarray(sha256_file(output_model)),
    )
    report = {
        "schema": "quintic-random-compact-root-expansion-v1",
        "scientific_scope": {
            "nesting": (
                "all existing channels are copied exactly; every new channel "
                "has zero initial coefficient"
            ),
            "selection": (
                "previously unrepresented reachable phase sectors are "
                "prioritized by structural size"
            ),
            "vectors": (
                "new left/right vectors are Gaussian QR directions orthogonal "
                "to existing vectors in the same sector"
            ),
            "proposal_usage": "no full-H or teacher proposal is read",
        },
        "configuration": {
            "degree": degree,
            "split": list(split),
            "old_rank": old_rank,
            "added_rank": args.add_rank,
            "new_rank": old_rank + args.add_rank,
            "seed": args.seed,
            "unreachable_sectors_skipped": skipped_unreachable,
        },
        "normalization": {
            "target_unit_coordinate_direction_norm": target_direction_norm,
            "raw_direction_norm_minimum": float(
                min(block["raw_direction_norm"] for block in added)
            ),
            "raw_direction_norm_median": float(
                np.median(
                    [block["raw_direction_norm"] for block in added]
                )
            ),
            "raw_direction_norm_maximum": float(
                max(block["raw_direction_norm"] for block in added)
            ),
        },
        "artifacts": {
            "base_compact": {
                "path": str(base_path),
                "sha256": sha256_file(base_path),
            },
            "base_h": {
                "path": str(base_h_path),
                "sha256": sha256_file(base_h_path),
            },
            "expanded_compact": {
                "path": str(output_model),
                "sha256": sha256_file(output_model),
            },
            "epoch0_h": {
                "path": str(output_h),
                "sha256": sha256_file(output_h),
            },
        },
    }
    write_json(args.out.expanduser().resolve(), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
