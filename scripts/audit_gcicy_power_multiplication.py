#!/usr/bin/env python3
"""Audit a symmetric power multiplication map between section spaces.

For a source basis of H^0(X,L^b) and a target basis of H^0(X,L^(mb)),
this script evaluates all unordered m-fold source products, reconstructs
them in the target basis on one point set, and verifies the reconstruction
on an independent point set.  The coefficient-map rank is computed from a
small target-dimension Gram matrix, so the full product coefficient matrix
does not need to be retained in memory.
"""

from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, load_common_point_pool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--common-pool", type=Path, required=True)
    parser.add_argument("--common-pool-split", default="train")
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--target-artifact", type=Path, required=True)
    parser.add_argument("--power", type=int, required=True)
    parser.add_argument("--fit-points", type=int, default=2048)
    parser.add_argument("--holdout-points", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[86401, 86402])
    parser.add_argument(
        "--rank-tolerances",
        type=float,
        nargs="+",
        default=[1e-8, 1e-10, 1e-12],
    )
    parser.add_argument("--least-squares-rcond", type=float, default=1e-12)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--product-chunk-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def artifact_basis(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as payload:
        degree = tuple(
            int(value)
            for value in np.asarray(payload["global_section_degree"]).tolist()
        )
        exponents = np.asarray(payload["global_section_exponents"], dtype=np.int64)
    if not degree or exponents.ndim != 2 or len(exponents) == 0:
        raise ValueError(f"{path} does not contain a valid section basis")
    return degree, exponents


def evaluate_values(
    adapter: object,
    points: list[object],
    exponents: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        values, _ = adapter.section_values_and_jacobian_batch(  # type: ignore[attr-defined]
            points[start:stop],
            exponents,
        )
        rows.append(np.asarray(values, dtype=np.complex128))
    return np.concatenate(rows, axis=0)


def rank_sweep(singular_values: np.ndarray, tolerances: list[float]) -> dict[str, int]:
    if not len(singular_values) or singular_values[0] <= 0:
        return {f"{value:.1e}": 0 for value in tolerances}
    return {
        f"{value:.1e}": int(np.sum(singular_values > value * singular_values[0]))
        for value in tolerances
    }


def product_values(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.prod(values[:, indices], axis=2)


def audit_seed(
    *,
    source_fit: np.ndarray,
    source_holdout: np.ndarray,
    target_fit: np.ndarray,
    target_holdout: np.ndarray,
    product_indices: np.ndarray,
    tolerances: list[float],
    rcond: float,
    chunk_size: int,
) -> dict[str, object]:
    target_singular_values = np.linalg.svd(target_fit, compute_uv=False)
    target_pseudoinverse = np.linalg.pinv(target_fit, rcond=rcond)
    coefficient_gram = np.zeros(
        (target_fit.shape[1], target_fit.shape[1]),
        dtype=np.complex128,
    )
    fit_numerator = 0.0
    fit_denominator = 0.0
    holdout_numerator = 0.0
    holdout_denominator = 0.0
    minimum_coefficient_norm = np.inf
    maximum_coefficient_norm = 0.0
    inactive_columns = 0

    for start in range(0, len(product_indices), chunk_size):
        stop = min(start + chunk_size, len(product_indices))
        index_chunk = product_indices[start:stop]
        products_fit = product_values(source_fit, index_chunk)
        products_holdout = product_values(source_holdout, index_chunk)
        coefficients = target_pseudoinverse @ products_fit

        coefficient_norms = np.linalg.norm(coefficients, axis=0)
        active = coefficient_norms > np.finfo(np.float64).eps
        inactive_columns += int(np.sum(~active))
        if np.any(active):
            active_norms = coefficient_norms[active]
            minimum_coefficient_norm = min(
                minimum_coefficient_norm,
                float(np.min(active_norms)),
            )
            maximum_coefficient_norm = max(
                maximum_coefficient_norm,
                float(np.max(active_norms)),
            )
            normalized = coefficients[:, active] / active_norms[None, :]
            coefficient_gram += normalized @ normalized.conj().T

        fit_difference = target_fit @ coefficients - products_fit
        holdout_difference = target_holdout @ coefficients - products_holdout
        fit_numerator += float(np.linalg.norm(fit_difference) ** 2)
        fit_denominator += float(np.linalg.norm(products_fit) ** 2)
        holdout_numerator += float(np.linalg.norm(holdout_difference) ** 2)
        holdout_denominator += float(np.linalg.norm(products_holdout) ** 2)

    eigenvalues = np.linalg.eigvalsh(coefficient_gram)
    coefficient_singular_values = np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
    return {
        "target_evaluation_rank_by_relative_tolerance": rank_sweep(
            target_singular_values,
            tolerances,
        ),
        "target_evaluation_condition_number": float(
            target_singular_values[0] / target_singular_values[-1]
        ),
        "coefficient_rank_by_relative_tolerance": rank_sweep(
            coefficient_singular_values,
            tolerances,
        ),
        "leading_coefficient_singular_values": [
            float(value) for value in coefficient_singular_values[:32]
        ],
        "smallest_coefficient_singular_value": float(
            coefficient_singular_values[-1]
        ),
        "inactive_product_columns": inactive_columns,
        "coefficient_column_norm_min": (
            float(minimum_coefficient_norm)
            if np.isfinite(minimum_coefficient_norm)
            else 0.0
        ),
        "coefficient_column_norm_max": float(maximum_coefficient_norm),
        "fit_relative_reconstruction_error": float(
            np.sqrt(fit_numerator / max(fit_denominator, np.finfo(float).tiny))
        ),
        "holdout_relative_reconstruction_error": float(
            np.sqrt(
                holdout_numerator
                / max(holdout_denominator, np.finfo(float).tiny)
            )
        ),
    }


def main() -> None:
    args = parse_args()
    if args.power < 2:
        raise SystemExit("power must be at least two")
    if args.fit_points <= 0 or args.holdout_points <= 0:
        raise SystemExit("fit and holdout point counts must be positive")
    if args.product_chunk_size <= 0 or args.evaluation_batch_size <= 0:
        raise SystemExit("batch and chunk sizes must be positive")

    source_degree, source_exponents = artifact_basis(args.source_artifact)
    target_degree, target_exponents = artifact_basis(args.target_artifact)
    expected_target_degree = tuple(args.power * value for value in source_degree)
    if target_degree != expected_target_degree:
        raise SystemExit(
            f"target degree {target_degree} does not equal "
            f"{args.power} times source degree {source_degree}"
        )
    product_indices = np.asarray(
        list(combinations_with_replacement(range(len(source_exponents)), args.power)),
        dtype=np.int64,
    )
    tolerances = sorted({float(value) for value in args.rank_tolerances}, reverse=True)
    if any(value <= 0 for value in tolerances):
        raise SystemExit("rank tolerances must be positive")

    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=True)
    pool = load_common_point_pool(
        args.common_pool,
        adapter,
        model,
        expected_model_seed=args.model_seed,
        expected_exact_model=True,
        expected_split=args.common_pool_split,
    )
    sample_count = args.fit_points + args.holdout_points
    if sample_count > len(pool.points):
        raise SystemExit("the common point pool is too small for this audit")

    reports: list[dict[str, object]] = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(len(pool.points), size=sample_count, replace=False)
        selected_points = [pool.points[int(index)] for index in selected_indices]
        source_values = evaluate_values(
            adapter,
            selected_points,
            source_exponents,
            batch_size=args.evaluation_batch_size,
        )
        target_values = evaluate_values(
            adapter,
            selected_points,
            target_exponents,
            batch_size=args.evaluation_batch_size,
        )
        scale = np.maximum(
            np.linalg.norm(target_values, axis=1),
            np.finfo(np.float64).tiny,
        )
        target_values = target_values / scale[:, None]
        source_values = source_values / scale[:, None] ** (1.0 / args.power)

        report = audit_seed(
            source_fit=source_values[: args.fit_points],
            source_holdout=source_values[args.fit_points :],
            target_fit=target_values[: args.fit_points],
            target_holdout=target_values[args.fit_points :],
            product_indices=product_indices,
            tolerances=tolerances,
            rcond=args.least_squares_rcond,
            chunk_size=args.product_chunk_size,
        )
        report.update(
            {
                "seed": int(seed),
                "fit_point_count": int(args.fit_points),
                "holdout_point_count": int(args.holdout_points),
                "selected_index_sha256": __import__("hashlib")
                .sha256(np.asarray(selected_indices, dtype=np.int64).tobytes())
                .hexdigest(),
            }
        )
        reports.append(report)
        print(
            f"seed={seed} "
            f"rank={report['coefficient_rank_by_relative_tolerance']} "
            f"fit_error={report['fit_relative_reconstruction_error']:.3e} "
            f"holdout_error={report['holdout_relative_reconstruction_error']:.3e}",
            flush=True,
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "gcicy-power-multiplication-audit-v1",
        "adapter": adapter.key,
        "model_seed": int(args.model_seed),
        "common_pool": str(args.common_pool.expanduser().resolve()),
        "common_pool_split": args.common_pool_split,
        "source_artifact": str(args.source_artifact.expanduser().resolve()),
        "target_artifact": str(args.target_artifact.expanduser().resolve()),
        "source_degree": list(source_degree),
        "target_degree": list(target_degree),
        "power": int(args.power),
        "source_section_count": int(len(source_exponents)),
        "target_section_count": int(len(target_exponents)),
        "symmetric_product_count": int(len(product_indices)),
        "rank_tolerances": tolerances,
        "least_squares_rcond": float(args.least_squares_rcond),
        "seeds": reports,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
