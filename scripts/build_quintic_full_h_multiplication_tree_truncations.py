#!/usr/bin/env python3
"""Build positive Fermat-symmetric H artifacts from a tree-root truncation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import svds

from gcicy_metric.pipeline.fermat_full_h import fermat_reynolds_project_h
from scripts.audit_quintic_full_h_oracle_multiplication_tree import (
    exact_product_right_inverse,
    lifted_operator_schmidt_matrix,
    quotient_multiplication_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-h", type=Path, required=True)
    parser.add_argument(
        "--baseline-h",
        type=Path,
        help="optional exact low-degree power lift; SVD is then applied to H-H_baseline",
    )
    parser.add_argument("--baseline-h-key", default="global_h_matrix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="4+6")
    parser.add_argument("--rank", type=int, nargs="+", default=(12, 32, 64, 107, 128))
    parser.add_argument("--svd-rank", type=int, default=160)
    parser.add_argument("--reconstruction-row-batch", type=int, default=32)
    parser.add_argument("--positive-floor-relative", type=float, default=1.0e-10)
    parser.add_argument("--relative-entry-tolerance", type=float, default=1.0e-12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_split(value: str) -> tuple[int, int]:
    fields = value.split("+")
    if len(fields) != 2:
        raise ValueError(f"invalid split: {value}")
    left, right = map(int, fields)
    if left <= 0 or right <= 0:
        raise ValueError("split degrees must be positive")
    return left, right


def sorted_sparse_svd(
    matrix: Any,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < rank < min(matrix.shape):
        raise ValueError("SVD rank must lie below the smaller matrix dimension")
    try:
        left, singular, right = svds(
            matrix,
            k=rank,
            which="LM",
            return_singular_vectors=True,
            solver="propack",
            tol=1.0e-11,
        )
    except (TypeError, ValueError):
        left, singular, right = svds(
            matrix,
            k=rank,
            which="LM",
            return_singular_vectors=True,
            tol=1.0e-11,
        )
    order = np.argsort(singular)[::-1]
    return left[:, order], singular[order], right[order]


def canonical_selected_pairs(
    right_inverse: Any,
    *,
    right_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = right_inverse.tocsc()
    counts = np.diff(value.indptr)
    if not np.all(counts == 1):
        raise ValueError("canonical right inverse must select one pair per section")
    pair = value.indices[value.indptr[:-1]]
    coefficient = value.data[value.indptr[:-1]]
    return pair // right_count, pair % right_count, coefficient


def reconstruct_section_h(
    left_vectors: np.ndarray,
    singular_values: np.ndarray,
    right_vectors: np.ndarray,
    *,
    rank: int,
    selected_left: np.ndarray,
    selected_right: np.ndarray,
    selected_coefficient: np.ndarray,
    left_count: int,
    right_count: int,
    row_batch: int,
) -> np.ndarray:
    section_count = len(selected_left)
    result = np.empty((section_count, section_count), dtype=np.complex128)
    scaled_left = left_vectors[:, :rank] * singular_values[:rank][None, :]
    selected_coefficient = np.asarray(selected_coefficient, dtype=np.complex128)
    for start in range(0, section_count, row_batch):
        stop = min(start + row_batch, section_count)
        row_indices = (
            selected_left[start:stop, None] * left_count
            + selected_left[None, :]
        )
        column_indices = (
            selected_right[start:stop, None] * right_count
            + selected_right[None, :]
        )
        left_rows = scaled_left[row_indices]
        right_columns = np.transpose(
            right_vectors[:rank, column_indices],
            (1, 2, 0),
        )
        lifted_entries = np.sum(left_rows * right_columns, axis=-1)
        denominator = (
            selected_coefficient[start:stop, None]
            * np.conj(selected_coefficient[None, :])
        )
        result[start:stop] = lifted_entries / denominator
    return result


def positive_projection(
    matrix: np.ndarray,
    *,
    trace: float,
    floor_relative: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    maximum = float(max(eigenvalues[-1], np.finfo(np.float64).tiny))
    floor = floor_relative * maximum
    clipped = np.maximum(eigenvalues, floor)
    positive = (eigenvectors * clipped[None, :]) @ eigenvectors.conj().T
    positive *= trace / np.trace(positive).real
    return positive, {
        "raw_minimum_eigenvalue": float(eigenvalues[0]),
        "raw_maximum_eigenvalue": float(eigenvalues[-1]),
        "raw_nonpositive_eigenvalue_count": int(np.count_nonzero(eigenvalues <= 0)),
        "positive_floor": float(floor),
        "relative_eigenvalue_clip_frobenius": float(
            np.linalg.norm(clipped - eigenvalues) / np.linalg.norm(eigenvalues)
        ),
    }


def matrix_summary(matrix: np.ndarray, teacher: np.ndarray) -> dict[str, Any]:
    value = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(value)
    optimal_scale = float(
        np.real(np.vdot(value, teacher)) / np.real(np.vdot(value, value))
    )
    return {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "relative_frobenius_error": float(
            np.linalg.norm(value - teacher) / np.linalg.norm(teacher)
        ),
        "scale_optimized_relative_frobenius_error": float(
            np.linalg.norm(optimal_scale * value - teacher)
            / np.linalg.norm(teacher)
        ),
        "trace": float(np.trace(value).real),
    }


def main() -> None:
    args = parse_args()
    source_path = args.full_h.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(source_path)
    degree = int(payload["degree"])
    teacher_exponents = np.asarray(payload["exponents"], dtype=np.int64)
    teacher = np.asarray(payload["global_h_matrix"], dtype=np.complex128)
    teacher = 0.5 * (teacher + teacher.conj().T)
    baseline_path = (
        None if args.baseline_h is None else args.baseline_h.expanduser().resolve()
    )
    baseline = None
    baseline_scale = None
    target = teacher
    if baseline_path is not None:
        baseline_payload = np.load(baseline_path)
        if args.baseline_h_key not in baseline_payload.files:
            raise KeyError(
                f"{args.baseline_h_key!r} is absent from {baseline_path}"
            )
        if int(baseline_payload["degree"]) != degree:
            raise ValueError("baseline and teacher degrees differ")
        if not np.array_equal(
            np.asarray(baseline_payload["exponents"], dtype=np.int64),
            teacher_exponents,
        ):
            raise ValueError("baseline and teacher quotient bases differ")
        baseline = np.asarray(
            baseline_payload[args.baseline_h_key],
            dtype=np.complex128,
        )
        baseline = 0.5 * (baseline + baseline.conj().T)
        baseline_scale = float(
            np.real(np.vdot(baseline, teacher))
            / np.real(np.vdot(baseline, baseline))
        )
        if not np.isfinite(baseline_scale) or baseline_scale <= 0:
            raise FloatingPointError("baseline alignment scale is invalid")
        baseline = baseline_scale * baseline
        target = teacher - baseline
    left_degree, right_degree = parse_split(args.split)
    if left_degree + right_degree != degree:
        raise SystemExit("split does not sum to the teacher degree")
    left, right, output, multiplication = quotient_multiplication_map(
        left_degree,
        right_degree,
    )
    if not np.array_equal(output, teacher_exponents):
        raise RuntimeError("tree output basis differs from teacher basis")
    right_inverse = exact_product_right_inverse(
        left,
        right,
        output,
        mode="canonical-exact",
    )
    schmidt, schmidt_diagnostics = lifted_operator_schmidt_matrix(
        target,
        right_inverse,
        left_count=len(left),
        right_count=len(right),
        relative_entry_tolerance=args.relative_entry_tolerance,
    )
    started = time.perf_counter()
    left_vectors, singular_values, right_vectors = sorted_sparse_svd(
        schmidt,
        rank=args.svd_rank,
    )
    svd_seconds = time.perf_counter() - started
    selected_left, selected_right, selected_coefficient = canonical_selected_pairs(
        right_inverse,
        right_count=len(right),
    )
    total_squared = float(np.sum(np.abs(schmidt.data) ** 2))
    trace = float(np.trace(teacher).real)
    rows = []
    for rank in sorted(set(args.rank)):
        if not 0 < rank <= args.svd_rank:
            raise ValueError("requested truncation rank exceeds the computed SVD")
        started = time.perf_counter()
        raw_correction = reconstruct_section_h(
            left_vectors,
            singular_values,
            right_vectors,
            rank=rank,
            selected_left=selected_left,
            selected_right=selected_right,
            selected_coefficient=selected_coefficient,
            left_count=len(left),
            right_count=len(right),
            row_batch=args.reconstruction_row_batch,
        )
        raw = (
            raw_correction
            if baseline is None
            else baseline + raw_correction
        )
        raw = 0.5 * (raw + raw.conj().T)
        raw *= trace / np.trace(raw).real
        positive, positivity = positive_projection(
            raw,
            trace=trace,
            floor_relative=args.positive_floor_relative,
        )
        reynolds = fermat_reynolds_project_h(
            teacher_exponents,
            positive,
            conjugation_invariant=True,
        )
        artifact_prefix = "canonical_residual" if baseline is not None else "canonical"
        artifact_path = (
            output_dir
            / f"{artifact_prefix}_{left_degree}_{right_degree}_r{rank}.npz"
        )
        np.savez_compressed(
            artifact_path,
            degree=np.asarray(degree),
            exponents=teacher_exponents,
            global_h_matrix=reynolds,
            source_full_h_sha256=np.asarray(sha256_file(source_path)),
            baseline_h_sha256=np.asarray(
                "" if baseline_path is None else sha256_file(baseline_path)
            ),
            split=np.asarray((left_degree, right_degree)),
            operator_schmidt_rank=np.asarray(rank),
        )
        row = {
            "rank": int(rank),
            "captured_squared_weight": float(
                np.sum(singular_values[:rank] ** 2) / total_squared
            ),
            "raw": matrix_summary(raw, teacher),
            "positivity_projection": positivity,
            "positive": matrix_summary(positive, teacher),
            "positive_fermat_reynolds": matrix_summary(reynolds, teacher),
            "artifact": {
                "path": str(artifact_path),
                "sha256": sha256_file(artifact_path),
            },
            "wall_seconds": float(time.perf_counter() - started),
        }
        rows.append(row)
        print(
            f"rank={rank} captured={row['captured_squared_weight']:.9f} "
            f"raw_error={row['raw']['relative_frobenius_error']:.6e} "
            f"sym_error={row['positive_fermat_reynolds']['relative_frobenius_error']:.6e} "
            f"min_eig={row['positive_fermat_reynolds']['minimum_eigenvalue']:.6e}",
            flush=True,
        )

    report = {
        "schema": "quintic-full-h-multiplication-tree-truncations-v1",
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "degree": degree,
            "section_count": int(len(teacher_exponents)),
        },
        "baseline": (
            None
            if baseline_path is None
            else {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
                "alignment_scale": baseline_scale,
                "aligned_matrix": matrix_summary(baseline, teacher),
                "residual_relative_frobenius_norm": float(
                    np.linalg.norm(target) / np.linalg.norm(teacher)
                ),
            }
        ),
        "configuration": {
            "split": [left_degree, right_degree],
            "lift": "canonical-exact",
            "ranks": sorted(set(args.rank)),
            "svd_rank": int(args.svd_rank),
            "reconstruction_row_batch": int(args.reconstruction_row_batch),
            "positive_floor_relative": float(args.positive_floor_relative),
        },
        "schmidt_matrix": schmidt_diagnostics,
        "svd": {
            "wall_seconds": float(svd_seconds),
            "computed_rank": int(len(singular_values)),
            "captured_squared_weight": float(
                np.sum(singular_values**2) / total_squared
            ),
            "leading_singular_values": singular_values.tolist(),
        },
        "truncations": rows,
        "projection_order": (
            "Hermitian truncation -> eigenvalue floor -> exact phase/S5/"
            "conjugation Reynolds projection"
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
