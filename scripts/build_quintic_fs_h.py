#!/usr/bin/env python3
"""Build a deterministic Fubini-Study full-H initialization artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.fermat_quintic import (
    exponent_compositions,
    fubini_study_h,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.degree <= 0:
        raise ValueError("degree must be positive")
    exponents = exponent_compositions(args.degree, 5)
    matrix = np.asarray(fubini_study_h(exponents), dtype=np.complex128)
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray("quintic-fubini-study-h-v1"),
        degree=np.asarray(args.degree, dtype=np.int64),
        exponents=np.asarray(exponents, dtype=np.int64),
        global_h_matrix=matrix,
        training_points=np.asarray(0, dtype=np.int64),
        teacher_used=np.asarray(False),
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    report = {
        "schema": "quintic-fubini-study-h-v1",
        "artifact": str(output_path),
        "artifact_sha256": sha256_file(output_path),
        "degree": args.degree,
        "section_count": len(exponents),
        "trainable_real_parameter_count": 0,
        "training_points": 0,
        "teacher_used": False,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
