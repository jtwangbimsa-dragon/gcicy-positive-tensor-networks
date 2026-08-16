#!/usr/bin/env python3
"""Compare two point-aligned gCICY metric audits by fibre-cluster bootstrap."""

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
    parser.add_argument("--baseline-arrays", type=Path, required=True)
    parser.add_argument("--candidate-arrays", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    resolved = path.expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as data:
        required = {
            "model_log_eta",
            "importance_weights",
            "sampling_cluster_ids",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{resolved} is missing arrays: {sorted(missing)}")
        adapter = str(data["adapter"].item()) if "adapter" in data.files else None
        return (
            np.asarray(data["model_log_eta"], dtype=np.float64),
            np.asarray(data["importance_weights"], dtype=np.float64),
            np.asarray(data["sampling_cluster_ids"]),
            adapter,
        )


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline_arrays.expanduser().resolve()
    candidate_path = args.candidate_arrays.expanduser().resolve()
    baseline, baseline_weights, baseline_clusters, baseline_adapter = load_arrays(
        baseline_path
    )
    candidate, candidate_weights, candidate_clusters, candidate_adapter = load_arrays(
        candidate_path
    )
    if baseline_adapter != candidate_adapter:
        raise SystemExit("baseline and candidate adapter labels differ")
    if not np.array_equal(baseline_clusters, candidate_clusters):
        raise SystemExit("baseline and candidate cluster IDs are not point-aligned")
    baseline_weight_sum = float(np.sum(baseline_weights))
    candidate_weight_sum = float(np.sum(candidate_weights))
    if (
        not np.isfinite(baseline_weight_sum)
        or not np.isfinite(candidate_weight_sum)
        or baseline_weight_sum <= 0.0
        or candidate_weight_sum <= 0.0
    ):
        raise SystemExit("baseline and candidate importance weights must have positive sums")
    baseline_weights = baseline_weights / baseline_weight_sum
    candidate_weights = candidate_weights / candidate_weight_sum
    if not np.allclose(
        baseline_weights,
        candidate_weights,
        rtol=1.0e-13,
        atol=1.0e-15,
    ):
        raise SystemExit("baseline and candidate importance weights differ")

    output = paired_cluster_bootstrap(
        baseline,
        candidate,
        baseline_weights,
        baseline_clusters,
        replicates=args.replicates,
        seed=args.seed,
    )
    output.update(
        {
            "adapter": baseline_adapter,
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "baseline_artifact": str(baseline_path),
            "candidate_artifact": str(candidate_path),
            "baseline_artifact_sha256": sha256(baseline_path),
            "candidate_artifact_sha256": sha256(candidate_path),
        }
    )
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
