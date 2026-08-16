#!/usr/bin/env python3
"""Convert saved TN audit points into an immutable common point pool."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.common_point_pool import (  # noqa: E402
    save_common_point_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--point-arrays", type=Path, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--cluster-size", type=int, required=True)
    parser.add_argument("--split", default="blind")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    arrays_path = args.point_arrays.expanduser().resolve()
    adapter = get_adapter(args.adapter)
    geometry = adapter.make_model(args.model_seed, exact=True)

    with np.load(arrays_path, allow_pickle=False) as payload:
        schema = str(np.asarray(payload["schema"]).item())
        if schema != "type11-positive-tensor-network-blind-arrays-v1":
            raise ValueError("unrecognized TN point-array schema")
        saved_adapter = str(np.asarray(payload["adapter"]).item())
        if saved_adapter != adapter.key:
            raise ValueError("point-array adapter does not match")
        metadata_keys = {
            "schema",
            "adapter",
            "model_log_eta",
            "metric_minimum_eigenvalues",
            "importance_weights",
            "sampling_cluster_ids",
            "teacher_log_eta",
        }
        point_payload = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key not in metadata_keys
        }
        saved_weights = np.asarray(payload["importance_weights"], dtype=np.float64)
        saved_clusters = np.asarray(payload["sampling_cluster_ids"], dtype=np.int64)

    points = adapter.points_from_storage_payload(geometry, point_payload)
    if len(points) % args.cluster_size:
        raise ValueError("saved point count is not divisible by the cluster size")
    recomputed_weights = np.asarray(adapter.importance_weights(points), dtype=np.float64)
    recomputed_weights /= np.sum(recomputed_weights)
    if not np.allclose(recomputed_weights, saved_weights, rtol=5e-12, atol=5e-14):
        raise ValueError("replayed importance weights do not match the saved arrays")
    recomputed_clusters = np.asarray(
        adapter.sampling_cluster_ids(points), dtype=np.int64
    )
    if not np.array_equal(recomputed_clusters, saved_clusters):
        raise ValueError("replayed fibre clusters do not match the saved arrays")

    manifest = save_common_point_pool(
        args.out,
        args.manifest,
        adapter,
        points,
        model_seed=args.model_seed,
        exact_model=True,
        split=args.split,
        sampling_seed=args.sampling_seed,
        cluster_size=args.cluster_size,
        sampling_metadata={
            "kind": "replayed_positive_tn_audit_arrays",
            "source_path": str(arrays_path),
            "source_sha256": sha256_file(arrays_path),
        },
    )
    print(f"wrote {manifest['pool_path']}")


if __name__ == "__main__":
    main()
