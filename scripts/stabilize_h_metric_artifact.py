#!/usr/bin/env python3
"""Create a provenance-recorded positive-definite copy of an H artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.adapter import (  # noqa: E402
    H_NEGATIVE_RELATIVE_TOLERANCE,
    H_POSITIVE_RELATIVE_FLOOR,
    positive_hermitian_projection,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    provenance = args.provenance.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"missing source artifact: {source}")
    for path in (output, provenance):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

    with np.load(source, allow_pickle=False) as archive:
        if "global_h_matrix" not in archive.files:
            raise SystemExit("source artifact has no global_h_matrix")
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}

    source_h = np.asarray(payload["global_h_matrix"], dtype=np.complex128)
    source_h = 0.5 * (source_h + source_h.conjugate().T)
    source_eigenvalues = np.linalg.eigvalsh(source_h)
    projected_h = positive_hermitian_projection(source_h)
    projected_h *= float(np.trace(source_h).real / np.trace(projected_h).real)
    projected_eigenvalues = np.linalg.eigvalsh(projected_h)
    np.linalg.cholesky(projected_h)

    source_hash = file_sha256(source)
    relative_change = float(
        np.linalg.norm(projected_h - source_h) / np.linalg.norm(source_h)
    )
    payload["global_h_matrix"] = projected_h
    payload["global_h_positive_relative_floor"] = np.asarray(
        H_POSITIVE_RELATIVE_FLOOR,
        dtype=np.float64,
    )
    payload["global_h_stabilized_from_sha256"] = np.asarray(source_hash)
    payload["global_h_stabilization_relative_change"] = np.asarray(
        relative_change,
        dtype=np.float64,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    record = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": source_hash,
        "output": str(output),
        "output_sha256": file_sha256(output),
        "relative_floor": H_POSITIVE_RELATIVE_FLOOR,
        "negative_relative_tolerance": H_NEGATIVE_RELATIVE_TOLERANCE,
        "source_minimum_eigenvalue": float(source_eigenvalues[0]),
        "source_maximum_eigenvalue": float(source_eigenvalues[-1]),
        "output_minimum_eigenvalue": float(projected_eigenvalues[0]),
        "output_maximum_eigenvalue": float(projected_eigenvalues[-1]),
        "relative_frobenius_change": relative_change,
        "trace_before": float(np.trace(source_h).real),
        "trace_after": float(np.trace(projected_h).real),
        "cholesky_verified": True,
    }
    provenance.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
