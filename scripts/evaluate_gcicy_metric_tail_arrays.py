#!/usr/bin/env python3
"""Evaluate saved gCICY log-volume arrays with bilateral tail diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.tail import (  # noqa: E402
    add_metric_geometry_evidence,
    bilateral_tail_metrics,
)


EIGENVALUE_KEYS = (
    "metric_minimum_eigenvalues",
    "minimum_metric_eigenvalues",
    "metric_min_eigenvalues",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--cluster-size", type=int)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_string(array: np.ndarray, default: str | None = None) -> str | None:
    if array.shape != ():
        raise ValueError("expected a scalar string array")
    value = str(array.item())
    return value if value else default


def evaluate(path: Path, cluster_size: int | None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as data:
        required = {"model_log_eta", "importance_weights"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{resolved} is missing arrays: {sorted(missing)}")
        log_eta = np.asarray(data["model_log_eta"], dtype=np.float64)
        weights = np.asarray(data["importance_weights"], dtype=np.float64)
        if "sampling_cluster_ids" in data.files:
            cluster_ids = np.asarray(data["sampling_cluster_ids"])
        elif cluster_size is not None:
            if cluster_size <= 0 or log_eta.size % cluster_size:
                raise ValueError("cluster-size must divide the point count")
            cluster_ids = np.repeat(
                np.arange(log_eta.size // cluster_size),
                cluster_size,
            )
        else:
            cluster_ids = None

        eigenvalues = None
        eigenvalue_key = None
        for key in EIGENVALUE_KEYS:
            if key in data.files:
                eigenvalue_key = key
                eigenvalues = np.asarray(data[key], dtype=np.float64)
                break
        metrics = bilateral_tail_metrics(log_eta, weights, cluster_ids)
        metrics = add_metric_geometry_evidence(metrics, eigenvalues)
        return {
            "array_artifact": str(resolved),
            "array_artifact_sha256": sha256(resolved),
            "source_schema": scalar_string(data["schema"])
            if "schema" in data.files
            else None,
            "adapter": scalar_string(data["adapter"])
            if "adapter" in data.files
            else None,
            "minimum_eigenvalue_array_key": eigenvalue_key,
            "metrics": metrics,
        }


def main() -> None:
    args = parse_args()
    if args.label is not None and len(args.label) != len(args.arrays):
        raise SystemExit("--label must be supplied once per --arrays input")
    labels = args.label or [path.stem for path in args.arrays]
    if len(set(labels)) != len(labels):
        raise SystemExit("labels must be unique")

    models = {
        label: evaluate(path, args.cluster_size)
        for label, path in zip(labels, args.arrays, strict=True)
    }
    output = {
        "schema": "gcicy-bilateral-tail-array-evaluation-v1",
        "tail_definition": {
            "absolute": "abs(log r)",
            "upper": "max(log r, 0)",
            "lower": "max(-log r, 0)",
            "quantile_probability": 0.999,
            "cvar_tail_fraction": 0.01,
        },
        "models": models,
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
