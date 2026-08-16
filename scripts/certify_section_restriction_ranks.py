#!/usr/bin/env python3
"""Build finite-field full-rank witnesses for ambient section restrictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACTS: dict[tuple[str, tuple[int, ...]], Path] = {
    ("X11_m3", (2, 2, 0)): ROOT / "outputs/pipeline/p4p1_type11_hirzebruch_k2_gpu.npz",
    ("X11_m3", (4, 4, 0)): ROOT
    / "outputs/pipeline/p4p1_type11_hirzebruch_k4_tail_refined_gpu.npz",
    ("X21", (1, 1)): ROOT / "outputs/pipeline/p5p1_type21_k3_1223_k1_refined.npz",
    ("X21", (2, 2)): ROOT / "outputs/pipeline/p5p1_type21_k3_1223_k2_gpu.npz",
    ("X21", (3, 3)): ROOT / "outputs/pipeline/p5p1_type21_k3_1223_k3_gpu_large.npz",
    ("X21", (4, 4)): ROOT / "outputs/pipeline/p5p1_type21_k3_1223_k4_gpu_large.npz",
    ("X22", (1, 1, 1)): ROOT / "outputs/gcicy_generic_global_h_metric_weighted.npz",
    ("X22", (2, 2, 2)): ROOT / "outputs/gcicy_generic_global_h_metric_k2_rank40_weighted.npz",
    ("X22", (3, 3, 3)): ROOT
    / "outputs/gcicy_generic_global_h_metric_k3_rank64_whitened_gpu.npz",
    ("X22_seed20260712", (3, 3, 3)): ROOT
    / "outputs/pipeline/gcicy_model_20260712_k3_rank64_whitened_gpu.npz",
}

from gcicy_metric.generic_model import (  # noqa: E402
    _q_polynomial_data,
    make_exact_generic_model,
)
from gcicy_metric.global_sections import ambient_monomial_exponents  # noqa: E402
from gcicy_metric.product_projective import (  # noqa: E402
    product_monomial_exponents,
)
from gcicy_metric.type21_candidate_p5p1_1223 import (  # noqa: E402
    local_polynomial_data as p5p1_local_polynomial_data,
    make_p5p1_type21_candidate_1223_model,
)
from gcicy_metric.type21_hirzebruch_x3 import (  # noqa: E402
    local_polynomial_data as hirzebruch_local_polynomial_data,
    make_hirzebruch_type21_model,
)


Array = np.ndarray


@dataclass(frozen=True)
class RankCase:
    key: str
    model_seed: int
    ambient_dimensions: tuple[int, ...]
    chart: tuple[int, ...]
    degrees_and_targets: tuple[tuple[tuple[int, ...], int], ...]
    primes: tuple[int, ...]
    polynomial_factory: Callable[[], tuple[list[tuple[Array, Array]], list[Array]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "section_restriction_rank_certificates.json",
    )
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--maximum-batches", type=int, default=200)
    return parser.parse_args()


def integer_coefficients(values: Array) -> Array:
    data = np.asarray(values, dtype=np.complex128)
    if np.max(np.abs(data.imag), initial=0.0) > 1e-12:
        raise ValueError("finite-field certificates require real integer coefficients")
    rounded = np.rint(data.real).astype(np.int64)
    if np.max(np.abs(data.real - rounded), initial=0.0) > 1e-12:
        raise ValueError("finite-field certificates require integer coefficients")
    return rounded


def affine_random_points(
    rng: np.random.Generator,
    count: int,
    dimensions: tuple[int, ...],
    chart: tuple[int, ...],
    prime: int,
) -> Array:
    total = sum(value + 1 for value in dimensions)
    points = rng.integers(0, prime, size=(count, total), dtype=np.int64)
    offset = 0
    for dimension, selected in zip(dimensions, chart, strict=True):
        points[:, offset + selected] = 1
        offset += dimension + 1
    return points


def evaluate_monomials(points: Array, exponents: Array, prime: int) -> Array:
    output = np.ones((len(points), len(exponents)), dtype=np.int64)
    for coordinate in range(points.shape[1]):
        powers = exponents[:, coordinate]
        for power in np.unique(powers):
            if power:
                columns = np.flatnonzero(powers == power)
                output[:, columns] *= np.power(
                    points[:, coordinate : coordinate + 1],
                    int(power),
                    dtype=np.int64,
                )
                output[:, columns] %= prime
    return output


def polynomial_zero_mask(
    points: Array,
    polynomials: list[tuple[Array, Array]],
    prime: int,
) -> Array:
    mask = np.ones(len(points), dtype=bool)
    for exponents, coefficients in polynomials:
        values = evaluate_monomials(points, exponents, prime)
        residues = values @ (integer_coefficients(coefficients) % prime)
        mask &= residues % prime == 0
    return mask


def rank_witness(matrix: Array, prime: int) -> tuple[int, list[int], list[int]]:
    work = np.asarray(matrix % prime, dtype=np.int64).copy()
    row_order = np.arange(work.shape[0], dtype=np.int64)
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    rank = 0
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[rank:, column] % prime)
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
            row_order[[rank, pivot]] = row_order[[pivot, rank]]
        inverse = pow(int(work[rank, column]), -1, prime)
        work[rank] = work[rank] * inverse % prime
        if rank + 1 < work.shape[0]:
            factors = work[rank + 1 :, column].copy()
            nonzero = np.flatnonzero(factors)
            if len(nonzero):
                selected = rank + 1 + nonzero
                work[selected] -= factors[nonzero, None] * work[rank]
                work[selected] %= prime
        pivot_rows.append(int(row_order[rank]))
        pivot_columns.append(column)
        rank += 1
        if rank == work.shape[0]:
            break
    return rank, pivot_rows, pivot_columns


def determinant_mod_prime(matrix: Array, prime: int) -> int:
    work = np.asarray(matrix % prime, dtype=np.int64).copy()
    if work.shape[0] != work.shape[1]:
        raise ValueError("determinant witness must be square")
    determinant = 1
    for column in range(work.shape[0]):
        candidates = np.flatnonzero(work[column:, column] % prime)
        if not len(candidates):
            return 0
        pivot = column + int(candidates[0])
        if pivot != column:
            work[[column, pivot]] = work[[pivot, column]]
            determinant = -determinant
        pivot_value = int(work[column, column] % prime)
        determinant = determinant * pivot_value % prime
        inverse = pow(pivot_value, -1, prime)
        if column + 1 < work.shape[0]:
            factors = work[column + 1 :, column] * inverse % prime
            work[column + 1 :] -= factors[:, None] * work[column]
            work[column + 1 :] %= prime
    return int(determinant % prime)


def collect_points(
    case: RankCase,
    polynomials: list[tuple[Array, Array]],
    prime: int,
    required: int,
    *,
    batch_size: int,
    maximum_batches: int,
) -> Array:
    rng = np.random.default_rng(10_000_019 + prime + case.model_seed)
    accepted = np.empty((0, sum(value + 1 for value in case.ambient_dimensions)), dtype=np.int64)
    for _ in range(maximum_batches):
        candidates = affine_random_points(
            rng,
            batch_size,
            case.ambient_dimensions,
            case.chart,
            prime,
        )
        selected = candidates[polynomial_zero_mask(candidates, polynomials, prime)]
        if len(selected):
            accepted = np.unique(np.concatenate((accepted, selected)), axis=0)
        if len(accepted) >= required:
            return accepted
    raise RuntimeError(
        f"{case.key}: found only {len(accepted)} finite-field points modulo {prime}"
    )


def coefficient_fingerprint(arrays: list[Array]) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        data = integer_coefficients(values).astype("<i8", copy=False)
        digest.update(np.asarray(data.shape, dtype="<i8").tobytes())
        digest.update(data.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hirzebruch_factory(base_degree: int) -> Callable[[], tuple[list[tuple[Array, Array]], list[Array]]]:
    def build() -> tuple[list[tuple[Array, Array]], list[Array]]:
        model = make_hirzebruch_type21_model(
            20260731 if base_degree == 3 else 20260831,
            exact=True,
            base_degree=base_degree,
        )
        polynomials = list(hirzebruch_local_polynomial_data(model, (0, 0, 0)))
        arrays = [
            model.p_tensor,
            model.q_cubic_coefficients,
            model.point_equation,
            model.point_section,
        ]
        return polynomials, arrays

    return build


def p5p1_factory() -> tuple[list[tuple[Array, Array]], list[Array]]:
    model = make_p5p1_type21_candidate_1223_model(20260802, exact=True)
    return list(p5p1_local_polynomial_data(model, (0, 0))), [
        model.a_tensor,
        model.b_tensor,
        model.r_a_coefficients,
        model.r_b_coefficients,
    ]


def type22_factory(
    model_seed: int,
) -> Callable[[], tuple[list[tuple[Array, Array]], list[Array]]]:
    def build() -> tuple[list[tuple[Array, Array]], list[Array]]:
        model = make_exact_generic_model(model_seed)
        polynomials = [
            (model.p1_exponents, model.p1_coefficients),
            (model.p2_exponents, model.p2_coefficients),
            _q_polynomial_data(model, (0, 0, 0), 1),
            _q_polynomial_data(model, (0, 0, 0), 2),
        ]
        return polynomials, [model.p1_coefficients, model.p2_tensor]

    return build


def cases() -> tuple[RankCase, ...]:
    return (
        RankCase(
            "X11_m3",
            20260731,
            (4, 1, 1),
            (0, 0, 0),
            (((2, 2, 0), 45), ((4, 4, 0), 274)),
            (11, 13),
            hirzebruch_factory(3),
        ),
        RankCase(
            "X21",
            20260802,
            (5, 1),
            (0, 0),
            (((1, 1), 11), ((2, 2), 50), ((3, 3), 145), ((4, 4), 324)),
            (11, 13),
            p5p1_factory,
        ),
        RankCase(
            "X22",
            20260711,
            (1, 1, 5),
            (0, 0, 0),
            (((1, 1, 1), 17), ((2, 2, 2), 84), ((3, 3, 3), 251)),
            (7, 11, 13, 17, 19),
            type22_factory(20260711),
        ),
        RankCase(
            "X22_seed20260712",
            20260712,
            (1, 1, 5),
            (0, 0, 0),
            (((3, 3, 3), 251),),
            (7, 11, 13, 17, 19),
            type22_factory(20260712),
        ),
        RankCase(
            "X11_m4",
            20260831,
            (4, 1, 1),
            (0, 0, 0),
            (((3, 3, 0), 140),),
            (11, 13),
            hirzebruch_factory(4),
        ),
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.maximum_batches <= 0:
        raise SystemExit("batch-size and maximum-batches must be positive")
    certificate_rows = []
    for case in cases():
        polynomials, coefficient_arrays = case.polynomial_factory()
        required = max(target for _, target in case.degrees_and_targets) + 16
        case_succeeded = False
        for prime in case.primes:
            points = collect_points(
                case,
                polynomials,
                prime,
                required,
                batch_size=args.batch_size,
                maximum_batches=args.maximum_batches,
            )
            degree_rows = []
            all_targets_met = True
            for degree, target in case.degrees_and_targets:
                exponents = product_monomial_exponents(
                    degree,
                    case.ambient_dimensions,
                )
                matrix = evaluate_monomials(points, exponents, prime)
                rank, pivot_rows, pivot_columns = rank_witness(matrix, prime)
                all_targets_met &= rank == target
                selected_rows = pivot_rows[:target]
                selected_columns = pivot_columns[:target]
                minor = matrix[np.ix_(selected_rows, selected_columns)]
                determinant = determinant_mod_prime(minor, prime)
                if rank == target and determinant == 0:
                    raise RuntimeError("rank witness determinant unexpectedly vanished")
                degree_row = {
                        "degree": list(degree),
                        "ambient_monomial_count": int(len(exponents)),
                        "riemann_roch_target": target,
                        "finite_field_rank": rank,
                        "pivot_minor_determinant_mod_prime": determinant,
                        "pivot_section_exponents": exponents[selected_columns].tolist(),
                        "pivot_point_coordinates": points[selected_rows].tolist(),
                        "evaluation_matrix_sha256": hashlib.sha256(
                            np.asarray(matrix, dtype="<i8").tobytes()
                        ).hexdigest(),
                    }
                artifact_path = ARTIFACTS.get((case.key, degree))
                if artifact_path is not None:
                    with np.load(artifact_path, allow_pickle=False) as artifact:
                        artifact_exponents = np.asarray(
                            artifact["global_section_exponents"], dtype=np.int64
                        )
                    artifact_matrix = evaluate_monomials(
                        points, artifact_exponents, prime
                    )
                    (
                        artifact_rank,
                        artifact_pivot_rows,
                        artifact_pivot_columns,
                    ) = rank_witness(artifact_matrix, prime)
                    artifact_minor = artifact_matrix[
                        np.ix_(
                            artifact_pivot_rows[:target],
                            artifact_pivot_columns[:target],
                        )
                    ]
                    artifact_determinant = determinant_mod_prime(
                        artifact_minor, prime
                    )
                    all_targets_met &= bool(
                        artifact_rank == target and artifact_determinant != 0
                    )
                    degree_row["reported_artifact"] = artifact_path.relative_to(
                        ROOT
                    ).as_posix()
                    degree_row["reported_artifact_sha256"] = file_sha256(
                        artifact_path
                    )
                    degree_row["reported_artifact_section_count"] = int(
                        len(artifact_exponents)
                    )
                    degree_row["reported_artifact_finite_field_rank"] = int(
                        artifact_rank
                    )
                    degree_row[
                        "reported_artifact_pivot_minor_determinant_mod_prime"
                    ] = int(artifact_determinant)
                    print(
                        f"  artifact={artifact_path.name} rank={artifact_rank} "
                        f"det={artifact_determinant}"
                    )
                degree_rows.append(degree_row)
                print(
                    f"{case.key} p={prime} degree={degree}: "
                    f"ambient={len(exponents)} rank={rank} target={target}"
                )
            if all_targets_met:
                certificate_rows.append(
                    {
                        "geometry": case.key,
                        "model_seed": case.model_seed,
                        "chart": list(case.chart),
                        "prime": prime,
                        "accepted_finite_field_points": int(len(points)),
                        "coefficient_sha256": coefficient_fingerprint(coefficient_arrays),
                        "degrees": degree_rows,
                        "all_targets_met": True,
                    }
                )
                case_succeeded = True
                break
        if not case_succeeded:
            raise SystemExit(f"no finite-field rank witness reached all targets for {case.key}")

    output = {
        "schema_version": 1,
        "description": (
            "Finite-field nonzero-minor certificates for the exact ambient-section "
            "restriction ranks used in the manuscript."
        ),
        "method": (
            "A target-size evaluation minor that is nonzero modulo a prime proves "
            "linear independence over Q after coefficient reduction; the exact "
            "Riemann--Roch dimension supplies the matching upper bound."
        ),
        "certificates": certificate_rows,
        "all_certificates_passed": all(
            row["all_targets_met"] for row in certificate_rows
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
