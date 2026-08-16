#!/usr/bin/env python3
"""Compare operator and purification Schmidt spectra for a full-H oracle.

The same quotient-basis matrix is lifted to ordered degree-one sites in four
ways:

* the Hermitian operator ``H`` representing ``s^dagger H s``;
* the principal positive square root ``H^(1/2)``;
* a Cholesky purification;
* a spectral purification that only rotates inside exact sparsity blocks.

All four lifts reproduce the same scalar section metric.  Their different
Schmidt spectra diagnose whether the observed chain-rank growth comes mainly
from the chosen purification gauge or is already present in the positive
operator itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_quintic_full_h_oracle_purification_schmidt import (
    _validate_basis,
    block_spectral_purification,
    cholesky_purification,
    compressed_operator_cut,
    compressed_purification_factor_cut,
    leading_schmidt_spectrum,
    principal_positive_square_root,
)


OBJECTS = (
    "operator-h",
    "purification-principal",
    "purification-cholesky",
    "purification-block-spectral",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-h", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-singular-values", type=int, default=256)
    parser.add_argument("--exact-row-dimension", type=int, default=1500)
    parser.add_argument("--relative-entry-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--relative-block-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--cuts",
        type=int,
        nargs="*",
        help="Cuts to audit. By default, audit 1,...,floor(k/2).",
    )
    parser.add_argument(
        "--objects",
        choices=OBJECTS,
        nargs="*",
        default=list(OBJECTS),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_factorization_error(factor: np.ndarray, matrix: np.ndarray) -> float:
    return float(
        np.linalg.norm(factor.conj().T @ factor - matrix)
        / np.linalg.norm(matrix)
    )


def _factor_summary(
    factor: np.ndarray,
    matrix: np.ndarray,
    *,
    elapsed_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    largest = float(np.max(np.abs(factor)))
    threshold = 1.0e-12 * largest
    result: dict[str, Any] = {
        "shape": [int(value) for value in factor.shape],
        "factorization_relative_frobenius_error": relative_factorization_error(
            factor,
            matrix,
        ),
        "condition_number": float(np.linalg.cond(factor)),
        "maximum_absolute_entry": largest,
        "entries_above_relative_1e-12": int(np.count_nonzero(np.abs(factor) > threshold)),
        "construction_wall_seconds": float(elapsed_seconds),
    }
    if extra:
        result.update(extra)
    return result


def build_objects(
    matrix: np.ndarray,
    *,
    requested: tuple[str, ...],
    relative_block_tolerance: float,
) -> tuple[dict[str, np.ndarray | None], dict[str, dict[str, Any]]]:
    coefficients: dict[str, np.ndarray | None] = {}
    summaries: dict[str, dict[str, Any]] = {}
    if "operator-h" in requested:
        coefficients["operator-h"] = None
        summaries["operator-h"] = {
            "shape": [int(value) for value in matrix.shape],
            "hermiticity_relative_frobenius_error": float(
                np.linalg.norm(matrix - matrix.conj().T) / np.linalg.norm(matrix)
            ),
            "condition_number": float(np.linalg.cond(matrix)),
            "maximum_absolute_entry": float(np.max(np.abs(matrix))),
            "entries_above_relative_1e-12": int(
                np.count_nonzero(
                    np.abs(matrix) > 1.0e-12 * np.max(np.abs(matrix))
                )
            ),
        }
    if "purification-principal" in requested:
        started = time.perf_counter()
        factor = principal_positive_square_root(matrix)
        elapsed = time.perf_counter() - started
        coefficients["purification-principal"] = factor
        summaries["purification-principal"] = _factor_summary(
            factor,
            matrix,
            elapsed_seconds=elapsed,
        )
    if "purification-cholesky" in requested:
        started = time.perf_counter()
        factor = cholesky_purification(matrix)
        elapsed = time.perf_counter() - started
        coefficients["purification-cholesky"] = factor
        summaries["purification-cholesky"] = _factor_summary(
            factor,
            matrix,
            elapsed_seconds=elapsed,
        )
    if "purification-block-spectral" in requested:
        started = time.perf_counter()
        factor, blocks = block_spectral_purification(
            matrix,
            relative_block_tolerance=relative_block_tolerance,
        )
        elapsed = time.perf_counter() - started
        coefficients["purification-block-spectral"] = factor
        summaries["purification-block-spectral"] = _factor_summary(
            factor,
            matrix,
            elapsed_seconds=elapsed,
            extra={
                "block_count": int(len(blocks)),
                "block_sizes": [int(len(block)) for block in blocks],
                "largest_block_size": int(max(map(len, blocks))),
            },
        )
    return coefficients, summaries


def main() -> None:
    args = parse_args()
    source_path = args.full_h.expanduser().resolve()
    payload = np.load(source_path)
    degree = int(payload["degree"])
    exponents = np.asarray(payload["exponents"], dtype=np.int64)
    matrix = np.asarray(payload["global_h_matrix"], dtype=np.complex128)
    matrix = 0.5 * (matrix + matrix.conj().T)
    basis_degree, coordinate_count = _validate_basis(exponents)
    if basis_degree != degree:
        raise SystemExit("artifact degree and exponent degree disagree")
    cuts = (
        tuple(args.cuts)
        if args.cuts
        else tuple(range(1, degree // 2 + 1))
    )
    if any(not 0 < cut <= degree // 2 for cut in cuts):
        raise SystemExit("cuts must lie in 1,...,floor(k/2)")
    requested = tuple(dict.fromkeys(args.objects))

    coefficients, object_summaries = build_objects(
        matrix,
        requested=requested,
        relative_block_tolerance=args.relative_block_tolerance,
    )
    object_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in requested
    }
    for cut in cuts:
        for name in requested:
            started = time.perf_counter()
            factor = coefficients[name]
            if name == "operator-h":
                compressed, diagnostics = compressed_operator_cut(
                    exponents,
                    matrix,
                    cut=cut,
                    relative_entry_tolerance=args.relative_entry_tolerance,
                )
            else:
                if factor is None:
                    raise RuntimeError("purification factor is missing")
                compressed, diagnostics = compressed_purification_factor_cut(
                    exponents,
                    factor,
                    cut=cut,
                    relative_entry_tolerance=args.relative_entry_tolerance,
                )
                diagnostics["factorization_relative_frobenius_error"] = (
                    relative_factorization_error(factor, matrix)
                )
            build_seconds = time.perf_counter() - started
            started = time.perf_counter()
            spectrum = leading_schmidt_spectrum(
                compressed,
                maximum_singular_values=args.maximum_singular_values,
                exact_row_dimension=args.exact_row_dimension,
            )
            spectrum_seconds = time.perf_counter() - started
            object_rows[name].append(
                {
                    "cut": int(cut),
                    "matrix": diagnostics,
                    "spectrum": spectrum,
                    "timing": {
                        "compressed_matrix_wall_seconds": float(build_seconds),
                        "spectrum_wall_seconds": float(spectrum_seconds),
                    },
                }
            )
            print(
                f"object={name} cut={cut} shape={compressed.shape} "
                f"nnz={compressed.nnz} "
                f"r999={spectrum['rank_for_cumulative_squared_weight']['0.999']} "
                f"captured={spectrum['captured_squared_weight']:.9f}",
                flush=True,
            )

    eigenvalues = np.linalg.eigvalsh(matrix)
    report = {
        "schema": "quintic-full-h-operator-purification-schmidt-v1",
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "degree": degree,
            "section_count": int(len(exponents)),
            "coordinate_count": coordinate_count,
            "minimum_eigenvalue": float(eigenvalues[0]),
            "maximum_eigenvalue": float(eigenvalues[-1]),
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        },
        "configuration": {
            "objects": list(requested),
            "cuts": list(cuts),
            "maximum_singular_values": int(args.maximum_singular_values),
            "exact_row_dimension": int(args.exact_row_dimension),
            "relative_entry_tolerance": float(args.relative_entry_tolerance),
            "relative_block_tolerance": float(args.relative_block_tolerance),
        },
        "objects": {
            name: {
                "summary": object_summaries[name],
                "cuts": object_rows[name],
            }
            for name in requested
        },
        "interpretation_scope": (
            "All lifts reproduce the same quotient-basis scalar metric. "
            "Schmidt spectra depend on the chosen ordered-site lift and, for "
            "purifications, on the output gauge. They diagnose this registered "
            "tensorization but are not gauge-independent lower bounds."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
