#!/usr/bin/env python3
"""Build an exact quotient-basis full-H artifact for an integer power lift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gcicy_metric.pipeline.fermat_h_power_lift import fermat_full_h_power_lift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h", type=Path, required=True)
    parser.add_argument("--source-h-key", default="global_h_matrix")
    parser.add_argument("--power", type=int, required=True)
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
    source_path = args.source_h.expanduser().resolve()
    payload = np.load(source_path)
    source_degree = int(payload["degree"])
    if args.source_h_key not in payload.files:
        raise KeyError(f"{args.source_h_key!r} is absent from {source_path}")
    source_h = np.asarray(payload[args.source_h_key], dtype=np.complex128)
    output_exponents, output_h = fermat_full_h_power_lift(
        source_h,
        source_degree=source_degree,
        power=args.power,
    )
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        degree=np.asarray(source_degree * args.power),
        exponents=output_exponents,
        global_h_matrix=output_h,
        source_degree=np.asarray(source_degree),
        source_power=np.asarray(args.power),
        source_h_sha256=np.asarray(sha256_file(source_path)),
    )
    eigenvalues = np.linalg.eigvalsh(output_h)
    report = {
        "schema": "quintic-full-h-power-lift-v1",
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "degree": source_degree,
            "section_count": int(len(source_h)),
        },
        "power": int(args.power),
        "target": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "degree": int(source_degree * args.power),
            "section_count": int(len(output_h)),
            "minimum_eigenvalue": float(eigenvalues[0]),
            "maximum_eigenvalue": float(eigenvalues[-1]),
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        },
        "exact_identity": (
            f"(1/{source_degree * args.power}) log F_target = "
            f"(1/{source_degree}) log F_source"
        ),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
