#!/usr/bin/env python3
"""Generate one fresh generic-quintic blind pool from a frozen cymetric basis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from cymetric.models.fubinistudy import FSModel
from cymetric.models.tfhelper import prepare_tf_basis


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_generic_quintic_kill_test_data import (  # noqa: E402
    json_value,
    sha256_file,
    write_json,
    write_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=202607412)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points <= 0 or args.batch_size <= 0:
        raise ValueError("point and batch counts must be positive")
    basis_path = args.basis.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a frozen blind pool")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading_basis"})
    started = time.perf_counter()
    raw_basis = np.load(basis_path, allow_pickle=True)
    model = FSModel(prepare_tf_basis(raw_basis))
    write_json(status_path, {"state": "running", "phase": "blind_pool"})
    pool = write_pool(
        model,
        output_dir / "blind",
        count=args.points,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    report = {
        "schema": "generic-quintic-fresh-blind-pool-v1",
        "configuration": {
            "points": args.points,
            "seed": args.seed,
            "batch_size": args.batch_size,
        },
        "basis": str(basis_path),
        "basis_sha256": sha256_file(basis_path),
        "blind": pool,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "report.json", report)
    write_json(
        status_path,
        {
            "state": "complete",
            "phase": "complete",
            "report_sha256": sha256_file(output_dir / "report.json"),
            "wall_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps(json_value(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
