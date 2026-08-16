#!/usr/bin/env python3
"""Build positive full-H artifacts from low-rank purification residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from gcicy_metric.pipeline.fermat_full_h import fermat_reynolds_project_h
from scripts.audit_quintic_full_h_oracle_multiplication_tree import (
    exact_product_right_inverse,
    quotient_multiplication_map,
)
from scripts.build_quintic_full_h_multiplication_tree_truncations import (
    canonical_selected_pairs,
    matrix_summary,
    parse_split,
    reconstruct_section_h,
    sorted_sparse_svd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-h", type=Path, required=True)
    parser.add_argument("--baseline-h", type=Path, required=True)
    parser.add_argument("--baseline-h-key", default="global_h_matrix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="4+6")
    parser.add_argument(
        "--rank",
        type=int,
        nargs="+",
        default=(4, 8, 16, 32, 64, 96, 128),
    )
    parser.add_argument("--svd-rank", type=int, default=160)
    parser.add_argument("--reconstruction-row-batch", type=int, default=32)
    parser.add_argument("--relative-entry-tolerance", type=float, default=1.0e-12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def principal_positive_factor(matrix: np.ndarray) -> np.ndarray:
    """Return the Hermitian positive factor ``B`` satisfying ``B^* B = H``."""

    hermitian = 0.5 * (
        np.asarray(matrix, dtype=np.complex128)
        + np.asarray(matrix, dtype=np.complex128).conj().T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    tolerance = (
        np.finfo(np.float64).eps
        * len(eigenvalues)
        * max(float(eigenvalues[-1]), 1.0)
    )
    if eigenvalues[0] < -tolerance:
        raise ValueError("H is not positive semidefinite")
    roots = np.sqrt(np.maximum(eigenvalues, 0.0))
    return (eigenvectors * roots[None, :]) @ eigenvectors.conj().T


def align_factor_by_left_unitary(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ``source`` to ``target`` by the optimal left unitary."""

    source_value = np.asarray(source, dtype=np.complex128)
    target_value = np.asarray(target, dtype=np.complex128)
    if source_value.shape != target_value.shape:
        raise ValueError("source and target factors must have equal shape")
    left, _, right_h = np.linalg.svd(
        target_value @ source_value.conj().T,
        full_matrices=False,
    )
    unitary = left @ right_h
    return unitary @ source_value, unitary


def lifted_purification_schmidt_matrix(
    factor: np.ndarray,
    right_inverse: sparse.csr_matrix,
    *,
    left_count: int,
    right_count: int,
    relative_entry_tolerance: float,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Lift ``B`` as ``Q B R^*`` and reshape its two child operator pairs.

    ``R`` is the canonical exact input right inverse. ``Q`` uses the same
    selected product pair for each output section, but with unit coefficient,
    so ``Q^* Q = I``. Therefore ``Q B R^*`` represents the same amplitude and
    preserves its norm exactly.
    """

    if not 0 <= relative_entry_tolerance < 1:
        raise ValueError("relative entry tolerance must lie in [0,1)")
    value = np.asarray(factor, dtype=np.complex128)
    selected_left, selected_right, selected_coefficient = (
        canonical_selected_pairs(
            right_inverse,
            right_count=right_count,
        )
    )
    largest = float(np.max(np.abs(value)))
    threshold = relative_entry_tolerance * largest
    nonzero = np.argwhere(np.abs(value) > threshold)
    rows = (
        selected_left[nonzero[:, 0]] * left_count
        + selected_left[nonzero[:, 1]]
    )
    columns = (
        selected_right[nonzero[:, 0]] * right_count
        + selected_right[nonzero[:, 1]]
    )
    data = value[nonzero[:, 0], nonzero[:, 1]] * np.conj(
        selected_coefficient[nonzero[:, 1]]
    )
    result = sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(left_count**2, right_count**2),
    ).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return result, {
        "shape": [int(item) for item in result.shape],
        "nonzero_entries": int(result.nnz),
        "density": float(result.nnz / (result.shape[0] * result.shape[1])),
        "source_factor_entries": int(len(nonzero)),
        "compressed_frobenius_squared": float(
            np.sum(np.abs(result.data) ** 2)
        ),
        "output_embedding": "canonical one-hot isometry",
        "input_embedding": "canonical exact right inverse",
    }


def reconstruct_section_factor(
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
    """Recover a section-space factor from a truncated lifted amplitude."""

    lifted = reconstruct_section_h(
        left_vectors,
        singular_values,
        right_vectors,
        rank=rank,
        selected_left=selected_left,
        selected_right=selected_right,
        selected_coefficient=np.ones_like(selected_coefficient),
        left_count=left_count,
        right_count=right_count,
        row_batch=row_batch,
    )
    return lifted / np.conj(selected_coefficient[None, :])


def factor_summary(
    value: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    difference = np.asarray(value) - np.asarray(target)
    return {
        "relative_frobenius_error": float(
            np.linalg.norm(difference) / np.linalg.norm(target)
        ),
        "frobenius_norm": float(np.linalg.norm(value)),
    }


def main() -> None:
    args = parse_args()
    teacher_path = args.full_h.expanduser().resolve()
    baseline_path = args.baseline_h.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_payload = np.load(teacher_path)
    baseline_payload = np.load(baseline_path)
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
    teacher_h = 0.5 * (teacher_h + teacher_h.conj().T)
    baseline_h = 0.5 * (baseline_h + baseline_h.conj().T)

    teacher_factor = principal_positive_factor(teacher_h)
    baseline_factor_unscaled = principal_positive_factor(baseline_h)
    aligned_teacher_factor, unitary = align_factor_by_left_unitary(
        teacher_factor,
        baseline_factor_unscaled,
    )
    baseline_amplitude_scale = float(
        np.real(np.vdot(baseline_factor_unscaled, aligned_teacher_factor))
        / np.real(np.vdot(baseline_factor_unscaled, baseline_factor_unscaled))
    )
    if not np.isfinite(baseline_amplitude_scale) or baseline_amplitude_scale <= 0:
        raise FloatingPointError("baseline amplitude alignment scale is invalid")
    baseline_factor = baseline_amplitude_scale * baseline_factor_unscaled
    residual_factor = aligned_teacher_factor - baseline_factor

    left_degree, right_degree = parse_split(args.split)
    if left_degree + right_degree != degree:
        raise ValueError("split does not sum to the teacher degree")
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
    schmidt, schmidt_diagnostics = lifted_purification_schmidt_matrix(
        residual_factor,
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
    selected_left, selected_right, selected_coefficient = (
        canonical_selected_pairs(
            right_inverse,
            right_count=len(right),
        )
    )
    total_squared = float(np.sum(np.abs(schmidt.data) ** 2))
    teacher_trace = float(np.trace(teacher_h).real)
    rows = []
    for rank in sorted(set(args.rank)):
        if not 0 < rank <= args.svd_rank:
            raise ValueError("requested truncation rank exceeds computed SVD")
        started = time.perf_counter()
        correction = reconstruct_section_factor(
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
        candidate_factor = baseline_factor + correction
        raw_h = candidate_factor.conj().T @ candidate_factor
        raw_h *= teacher_trace / np.trace(raw_h).real
        reynolds_h = fermat_reynolds_project_h(
            exponents,
            raw_h,
            conjugation_invariant=True,
        )
        artifact_path = (
            output_dir
            / f"purification_residual_{left_degree}_{right_degree}_r{rank}.npz"
        )
        np.savez_compressed(
            artifact_path,
            degree=np.asarray(degree),
            exponents=exponents,
            global_h_matrix=reynolds_h,
            source_full_h_sha256=np.asarray(sha256_file(teacher_path)),
            baseline_h_sha256=np.asarray(sha256_file(baseline_path)),
            split=np.asarray((left_degree, right_degree)),
            purification_schmidt_rank=np.asarray(rank),
        )
        row = {
            "rank": int(rank),
            "captured_squared_weight": float(
                np.sum(singular_values[:rank] ** 2) / total_squared
            ),
            "factor": factor_summary(candidate_factor, aligned_teacher_factor),
            "raw_h": matrix_summary(raw_h, teacher_h),
            "positive_fermat_reynolds": matrix_summary(reynolds_h, teacher_h),
            "artifact": {
                "path": str(artifact_path),
                "sha256": sha256_file(artifact_path),
            },
            "wall_seconds": float(time.perf_counter() - started),
        }
        rows.append(row)
        print(
            f"rank={rank} captured={row['captured_squared_weight']:.9f} "
            f"factor_error={row['factor']['relative_frobenius_error']:.6e} "
            f"sym_h_error="
            f"{row['positive_fermat_reynolds']['relative_frobenius_error']:.6e} "
            f"min_eig="
            f"{row['positive_fermat_reynolds']['minimum_eigenvalue']:.6e}",
            flush=True,
        )

    report = {
        "schema": "quintic-full-h-purification-residual-tree-truncations-v1",
        "teacher": {
            "path": str(teacher_path),
            "sha256": sha256_file(teacher_path),
            "degree": degree,
            "section_count": int(len(exponents)),
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "amplitude_alignment_scale": baseline_amplitude_scale,
            "aligned_h": matrix_summary(
                baseline_factor.conj().T @ baseline_factor,
                teacher_h,
            ),
        },
        "purification_alignment": {
            "unitarity_relative_error": float(
                np.linalg.norm(unitary.conj().T @ unitary - np.eye(len(unitary)))
                / np.sqrt(len(unitary))
            ),
            "residual_relative_frobenius_norm": float(
                np.linalg.norm(residual_factor)
                / np.linalg.norm(aligned_teacher_factor)
            ),
            "exact_teacher_reconstruction_relative_error": float(
                np.linalg.norm(
                    aligned_teacher_factor.conj().T @ aligned_teacher_factor
                    - teacher_h
                )
                / np.linalg.norm(teacher_h)
            ),
        },
        "configuration": {
            "split": [left_degree, right_degree],
            "lift": "canonical exact input plus canonical one-hot output",
            "ranks": sorted(set(args.rank)),
            "svd_rank": int(args.svd_rank),
            "reconstruction_row_batch": int(args.reconstruction_row_batch),
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
        "positivity": (
            "Each raw candidate is B^*B; exact Fermat Reynolds averaging "
            "preserves positive semidefiniteness."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
