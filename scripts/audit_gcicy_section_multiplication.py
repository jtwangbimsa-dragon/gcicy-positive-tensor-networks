#!/usr/bin/env python3
"""Numerically audit section-ring multiplication on a frozen gCICY point pool.

The audit uses fixed section bases from validated H-metric artifacts.  It fits
the coefficient maps of

    Sym^4 H^0(L) -> H^0(L^4),
    H^0(L) tensor H^0(L^3) -> H^0(L^4),
    Sym^2 H^0(L^2) -> H^0(L^4),

on one subset of a frozen training pool and verifies the resulting identities
on an independent subset.  Numerical ranks are reported across a tolerance
sweep so that conclusions do not depend on a single SVD cutoff.
"""

from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys
from typing import Callable

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
    parser.add_argument("--artifact-k1", type=Path, required=True)
    parser.add_argument("--artifact-k2", type=Path, required=True)
    parser.add_argument("--artifact-k3", type=Path, required=True)
    parser.add_argument("--artifact-k4", type=Path, required=True)
    parser.add_argument("--fit-points", type=int, default=1024)
    parser.add_argument("--holdout-points", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[86211, 86212])
    parser.add_argument(
        "--rank-tolerances",
        type=float,
        nargs="+",
        default=[1e-8, 1e-10, 1e-12],
    )
    parser.add_argument("--least-squares-rcond", type=float, default=1e-12)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--product-chunk-size", type=int, default=256)
    parser.add_argument("--coefficient-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def artifact_exponents(path: Path, expected_power: int) -> np.ndarray:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as payload:
        degree = tuple(
            int(value)
            for value in np.asarray(payload["global_section_degree"]).tolist()
        )
        exponents = np.asarray(
            payload["global_section_exponents"],
            dtype=np.int64,
        )
    if not degree or any(value != expected_power for value in degree):
        raise ValueError(
            f"{path} has section degree {degree}, expected power {expected_power}"
        )
    if exponents.ndim != 2 or len(exponents) == 0:
        raise ValueError(f"{path} has an invalid section basis")
    return exponents


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


def degree_one_fourfold_products(values: np.ndarray) -> np.ndarray:
    indices = np.asarray(
        list(combinations_with_replacement(range(values.shape[1]), 4)),
        dtype=np.int64,
    )
    return np.prod(values[:, indices], axis=2)


def mixed_products(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nj->nij", left, right, optimize=True).reshape(
        len(left),
        left.shape[1] * right.shape[1],
    )


def symmetric_square_products(values: np.ndarray) -> np.ndarray:
    rows, columns = np.triu_indices(values.shape[1])
    return values[:, rows] * values[:, columns]


def normalized_singular_values(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    column_norms = np.linalg.norm(matrix, axis=0)
    active = column_norms > np.finfo(np.float64).eps
    if not np.any(active):
        return np.empty(0, dtype=np.float64), column_norms
    normalized = matrix[:, active] / column_norms[active][None, :]
    return np.linalg.svd(normalized, compute_uv=False), column_norms


def rank_sweep(
    singular_values: np.ndarray,
    tolerances: list[float],
) -> dict[str, int]:
    if not len(singular_values) or singular_values[0] <= 0:
        return {f"{value:.1e}": 0 for value in tolerances}
    return {
        f"{value:.1e}": int(
            np.sum(singular_values > value * singular_values[0])
        )
        for value in tolerances
    }


def fit_product_coefficients(
    section_fit: np.ndarray,
    product_fit: np.ndarray,
    *,
    rcond: float,
    chunk_size: int,
) -> np.ndarray:
    pseudoinverse = np.linalg.pinv(section_fit, rcond=rcond)
    chunks = [
        pseudoinverse @ product_fit[:, start : start + chunk_size]
        for start in range(0, product_fit.shape[1], chunk_size)
    ]
    return np.concatenate(chunks, axis=1)


def reconstruction_error(
    sections: np.ndarray,
    coefficients: np.ndarray,
    products: np.ndarray,
    *,
    chunk_size: int,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for start in range(0, products.shape[1], chunk_size):
        stop = min(start + chunk_size, products.shape[1])
        difference = sections @ coefficients[:, start:stop] - products[:, start:stop]
        numerator += float(np.linalg.norm(difference) ** 2)
        denominator += float(np.linalg.norm(products[:, start:stop]) ** 2)
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))


def audit_map(
    name: str,
    product_builder: Callable[[dict[int, np.ndarray]], np.ndarray],
    fit_values: dict[int, np.ndarray],
    holdout_values: dict[int, np.ndarray],
    *,
    tolerances: list[float],
    rcond: float,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    fit_products = product_builder(fit_values)
    holdout_products = product_builder(holdout_values)
    coefficients = fit_product_coefficients(
        fit_values[4],
        fit_products,
        rcond=rcond,
        chunk_size=chunk_size,
    )
    singular_values, column_norms = normalized_singular_values(coefficients)
    report = {
        "name": name,
        "domain_column_count": int(coefficients.shape[1]),
        "coefficient_shape": [int(value) for value in coefficients.shape],
        "rank_by_relative_tolerance": rank_sweep(singular_values, tolerances),
        "leading_normalized_singular_values": [
            float(value) for value in singular_values[:32]
        ],
        "smallest_normalized_singular_value": (
            float(singular_values[-1]) if len(singular_values) else 0.0
        ),
        "coefficient_column_norm_min": float(np.min(column_norms)),
        "coefficient_column_norm_max": float(np.max(column_norms)),
        "fit_relative_reconstruction_error": reconstruction_error(
            fit_values[4],
            coefficients,
            fit_products,
            chunk_size=chunk_size,
        ),
        "holdout_relative_reconstruction_error": reconstruction_error(
            holdout_values[4],
            coefficients,
            holdout_products,
            chunk_size=chunk_size,
        ),
    }
    return coefficients, report


def main() -> None:
    args = parse_args()
    if args.fit_points <= 0 or args.holdout_points <= 0:
        raise SystemExit("fit and holdout point counts must be positive")
    if args.evaluation_batch_size <= 0 or args.product_chunk_size <= 0:
        raise SystemExit("batch and chunk sizes must be positive")
    if not args.seeds:
        raise SystemExit("at least one audit seed is required")
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

    artifacts = {
        1: args.artifact_k1,
        2: args.artifact_k2,
        3: args.artifact_k3,
        4: args.artifact_k4,
    }
    exponents = {
        power: artifact_exponents(path, power)
        for power, path in artifacts.items()
    }
    seed_reports: list[dict[str, object]] = []
    coefficient_payload: dict[str, np.ndarray] = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(pool.points), size=sample_count, replace=False)
        fit_indices = indices[: args.fit_points]
        holdout_indices = indices[args.fit_points :]
        selected_points = [
            pool.points[int(index)]
            for index in np.concatenate((fit_indices, holdout_indices))
        ]
        values = {
            power: evaluate_values(
                adapter,
                selected_points,
                powers,
                batch_size=args.evaluation_batch_size,
            )
            for power, powers in exponents.items()
        }
        fit_values = {
            power: matrix[: args.fit_points].copy()
            for power, matrix in values.items()
        }
        holdout_values = {
            power: matrix[args.fit_points :].copy()
            for power, matrix in values.items()
        }
        for split_values in (fit_values, holdout_values):
            scale = np.maximum(
                np.linalg.norm(split_values[4], axis=1),
                np.finfo(np.float64).tiny,
            )
            for power in split_values:
                split_values[power] /= scale[:, None] ** (power / 4.0)

        section_singular_values = np.linalg.svd(
            fit_values[4],
            compute_uv=False,
        )
        maps: dict[str, dict[str, object]] = {}
        coefficients: dict[str, np.ndarray] = {}
        builders: dict[str, Callable[[dict[int, np.ndarray]], np.ndarray]] = {
            "mu_1111": lambda data: degree_one_fourfold_products(data[1]),
            "mu_13": lambda data: mixed_products(data[1], data[3]),
            "mu_22": lambda data: symmetric_square_products(data[2]),
        }
        for name, builder in builders.items():
            coefficient, report = audit_map(
                name,
                builder,
                fit_values,
                holdout_values,
                tolerances=tolerances,
                rcond=args.least_squares_rcond,
                chunk_size=args.product_chunk_size,
            )
            coefficients[name] = coefficient
            maps[name] = report
        if args.coefficient_output is not None:
            coefficient_payload[f"mu1111_coefficients_seed_{seed}"] = (
                coefficients["mu_1111"]
            )

        union_coefficients = np.concatenate(
            (coefficients["mu_13"], coefficients["mu_22"]),
            axis=1,
        )
        union_singular_values, union_column_norms = normalized_singular_values(
            union_coefficients
        )
        union_ranks = rank_sweep(union_singular_values, tolerances)
        maps["union_mu_13_mu_22"] = {
            "domain_column_count": int(union_coefficients.shape[1]),
            "coefficient_shape": [
                int(value) for value in union_coefficients.shape
            ],
            "rank_by_relative_tolerance": union_ranks,
            "primitive_complement_dimension_by_relative_tolerance": {
                key: int(len(exponents[4]) - rank)
                for key, rank in union_ranks.items()
            },
            "intersection_dimension_by_relative_tolerance": {
                key: int(
                    maps["mu_13"]["rank_by_relative_tolerance"][key]  # type: ignore[index]
                    + maps["mu_22"]["rank_by_relative_tolerance"][key]  # type: ignore[index]
                    - rank
                )
                for key, rank in union_ranks.items()
            },
            "leading_normalized_singular_values": [
                float(value) for value in union_singular_values[:32]
            ],
            "smallest_normalized_singular_value": (
                float(union_singular_values[-1])
                if len(union_singular_values)
                else 0.0
            ),
            "coefficient_column_norm_min": float(np.min(union_column_norms)),
            "coefficient_column_norm_max": float(np.max(union_column_norms)),
        }
        seed_reports.append(
            {
                "seed": int(seed),
                "fit_point_count": int(args.fit_points),
                "holdout_point_count": int(args.holdout_points),
                "fit_index_sha256": __import__("hashlib")
                .sha256(np.asarray(fit_indices, dtype=np.int64).tobytes())
                .hexdigest(),
                "holdout_index_sha256": __import__("hashlib")
                .sha256(np.asarray(holdout_indices, dtype=np.int64).tobytes())
                .hexdigest(),
                "h0_l4_evaluation_rank_by_relative_tolerance": rank_sweep(
                    section_singular_values,
                    tolerances,
                ),
                "h0_l4_evaluation_condition_number": float(
                    section_singular_values[0] / section_singular_values[-1]
                ),
                "maps": maps,
            }
        )
        print(
            f"seed={seed} "
            f"rank1111={maps['mu_1111']['rank_by_relative_tolerance']} "
            f"rank13={maps['mu_13']['rank_by_relative_tolerance']} "
            f"rank22={maps['mu_22']['rank_by_relative_tolerance']} "
            f"union={union_ranks}",
            flush=True,
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "gcicy-section-multiplication-audit-v1",
        "adapter": adapter.key,
        "model_seed": int(args.model_seed),
        "common_pool": str(args.common_pool.expanduser().resolve()),
        "common_pool_split": args.common_pool_split,
        "section_counts": {
            f"k{power}": int(len(powers))
            for power, powers in exponents.items()
        },
        "artifact_paths": {
            f"k{power}": str(path.expanduser().resolve())
            for power, path in artifacts.items()
        },
        "rank_tolerances": tolerances,
        "least_squares_rcond": float(args.least_squares_rcond),
        "seeds": seed_reports,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"wrote {output}", flush=True)
    if args.coefficient_output is not None:
        coefficient_output = args.coefficient_output.expanduser().resolve()
        coefficient_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_coefficients = coefficient_output.with_suffix(
            coefficient_output.suffix + ".tmp"
        )
        with temporary_coefficients.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray(
                    "gcicy-section-multiplication-coefficients-v1"
                ),
                adapter=np.asarray(adapter.key),
                model_seed=np.asarray(args.model_seed, dtype=np.int64),
                degree_one_fourfold_indices=np.asarray(
                    list(
                        combinations_with_replacement(
                            range(len(exponents[1])),
                            4,
                        )
                    ),
                    dtype=np.int64,
                ),
                **coefficient_payload,
            )
        temporary_coefficients.replace(coefficient_output)
        print(f"wrote {coefficient_output}", flush=True)


if __name__ == "__main__":
    main()
