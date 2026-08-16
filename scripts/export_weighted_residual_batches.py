#!/usr/bin/env python3
"""Export residuals and importance weights for named frozen training batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from scripts.train_gcicy_pipeline import load_specification, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "checkpoint"), required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = args.spec.expanduser().resolve()
    data = load_specification(spec_path)
    adapter = get_adapter(str(data["adapter"]))
    model_data = data["model"]
    model_seed = int(model_data["seed"])
    exact_model = bool(model_data.get("exact", True))
    model = adapter.make_model(model_seed, exact=exact_model)
    training = data["training"]
    batch_key = "train_batches" if args.role == "train" else "checkpoint_batches"
    batches = training.get(batch_key, [])
    if not batches:
        raise ValueError(f"specification has no {batch_key}")
    artifact_path = (
        args.artifact.expanduser().resolve()
        if args.artifact is not None
        else resolve_path(training["initial_artifact"], spec_path.parent)
    )
    artifact = adapter.load_h_artifact(artifact_path, model)
    workers = int(training.get("sampling_workers", 1))
    cluster_size = int(training.get("sampling_cluster_size", 1))
    backend = str(training.get("sampling_backend", "process"))
    output = args.out.expanduser().resolve()
    summary_output = (
        args.summary_out.expanduser().resolve()
        if args.summary_out is not None
        else output.with_name(output.stem + "_summary.json")
    )
    if output.exists() or summary_output.exists():
        raise FileExistsError("residual export outputs already exist")

    started = time.perf_counter()
    raw_parts = []
    weight_parts = []
    seed_parts = []
    local_index_parts = []
    cluster_parts = []
    cluster_offset = 0
    rows = []
    for row in batches:
        if not isinstance(row, dict) or set(row) != {"seed", "points"}:
            raise ValueError(f"{batch_key} entries must contain seed and points")
        seed = int(row["seed"])
        count = int(row["points"])
        batch_started = time.perf_counter()
        if workers > 1:
            points, shards = sample_points_parallel(
                adapter,
                model_seed=model_seed,
                exact_model=exact_model,
                count=count,
                seed=seed,
                workers=workers,
                cluster_size=cluster_size,
                backend=backend,
            )
        else:
            points = adapter.sample_points(model, count, seed=seed)
            shards = []
        metrics = adapter.h_metrics(points, artifact)
        raw = np.asarray(adapter.residual_values(points, metrics), dtype=np.float64)
        weights = np.asarray(adapter.importance_weights(points), dtype=np.float64)
        clusters = np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
        unique_clusters, inverse = np.unique(clusters, return_inverse=True)
        global_clusters = inverse.astype(np.int64) + cluster_offset
        cluster_offset += len(unique_clusters)
        raw_parts.append(raw)
        weight_parts.append(weights)
        seed_parts.append(np.full(len(points), seed, dtype=np.int64))
        local_index_parts.append(np.arange(len(points), dtype=np.int64))
        cluster_parts.append(global_clusters)
        rows.append(
            {
                "seed": seed,
                "point_count": len(points),
                "independent_cluster_count": len(unique_clusters),
                "minimum_raw": float(np.min(raw)),
                "maximum_raw": float(np.max(raw)),
                "weight_sum": float(np.sum(weights)),
                "sampling_shards": shards,
                "runtime_seconds": float(time.perf_counter() - batch_started),
            }
        )
        print(
            f"role={args.role}, seed={seed}, points={len(points)}, "
            f"clusters={len(unique_clusters)}, seconds="
            f"{time.perf_counter() - batch_started:.1f}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        corrected_log_ma=np.concatenate(raw_parts),
        importance_weights=np.concatenate(weight_parts),
        source_seed=np.concatenate(seed_parts),
        source_local_index=np.concatenate(local_index_parts),
        sampling_cluster_id=np.concatenate(cluster_parts),
        adapter=np.asarray(adapter.key),
        model_seed=np.asarray(model_seed, dtype=np.int64),
        exact_model=np.asarray(exact_model),
        source_artifact=np.asarray(str(artifact_path)),
        source_specification=np.asarray(str(spec_path)),
        role=np.asarray(args.role),
    )
    summary = {
        "schema_version": 1,
        "adapter": adapter.key,
        "model_seed": model_seed,
        "exact_model": exact_model,
        "role": args.role,
        "source_specification": str(spec_path),
        "source_artifact": str(artifact_path),
        "output": str(output),
        "point_count": int(sum(len(part) for part in raw_parts)),
        "independent_cluster_count": int(cluster_offset),
        "sampling": {
            "workers": workers,
            "cluster_size": cluster_size,
            "backend": backend,
        },
        "batches": rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    summary_output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)
    print(f"wrote {summary_output}", flush=True)


if __name__ == "__main__":
    main()
