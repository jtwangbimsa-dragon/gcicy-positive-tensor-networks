#!/usr/bin/env python3
"""Bootstrap the frozen X22 q=22 versus q=60 dictionary comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.tail import paired_cluster_bootstrap  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "log_eta": np.asarray(payload["model_log_eta"], dtype=np.float64),
            "weights": np.asarray(payload["importance_weights"], dtype=np.float64),
            "clusters": np.asarray(payload["sampling_cluster_ids"]),
        }


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    baseline = load_arrays(baseline_path)
    candidate = load_arrays(candidate_path)
    if not np.array_equal(baseline["weights"], candidate["weights"]):
        raise ValueError("baseline and candidate weights differ")
    if not np.array_equal(baseline["clusters"], candidate["clusters"]):
        raise ValueError("baseline and candidate fibre labels differ")

    report = paired_cluster_bootstrap(
        baseline["log_eta"],
        candidate["log_eta"],
        baseline["weights"],
        baseline["clusters"],
        replicates=args.replicates,
        seed=args.seed,
    )
    report["comparison"] = "X22 q=22 k=4 versus q=60 k=4"
    report["baseline_artifact"] = {
        "path": str(baseline_path),
        "sha256": sha256(baseline_path),
    }
    report["candidate_artifact"] = {
        "path": str(candidate_path),
        "sha256": sha256(candidate_path),
    }

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
