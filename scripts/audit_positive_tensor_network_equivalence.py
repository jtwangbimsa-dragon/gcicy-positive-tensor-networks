#!/usr/bin/env python3
"""Compare two saved positive tensor-network metrics on common blind points."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    get_adapter,
    positive_tensor_network_from_artifact_payload,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    section_values_and_jacobian_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="p4p1_type11_hirzebruch_x3",
        help="registered gCICY adapter key",
    )
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument(
        "--sampling-cluster-size",
        type=int,
        default=4,
        help="number of complete fibre roots returned by one sampling cluster",
    )
    parser.add_argument("--seed", type=int, default=72961)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--relative-metric-tolerance",
        type=float,
        help=(
            "optional scale-invariant metric gate; when set, potential uses the "
            "absolute tolerance and metric equivalence uses this relative tolerance"
        ),
    )
    parser.add_argument(
        "--allow-different-site-counts",
        action="store_true",
        help=(
            "compare function-preserving degree transfers whose saved models "
            "have different site counts"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path, source_h, device):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "type11-positive-tensor-network-v1":
        raise ValueError(f"unrecognized model artifact {path}")
    precision = str(payload["precision"])
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    model = positive_tensor_network_from_artifact_payload(
        source_h,
        device=device,
        payload=payload,
    )
    model.eval()
    return payload, model, complex_dtype


def main() -> None:
    import torch

    args = parse_args()
    if (
        args.points <= 0
        or args.sampling_cluster_size <= 0
        or args.points % args.sampling_cluster_size
        or args.chunk_size <= 0
    ):
        raise SystemExit(
            "points must be positive and divisible by the sampling cluster size"
        )
    started = time.perf_counter()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    model_a_path = args.model_a.expanduser().resolve()
    model_b_path = args.model_b.expanduser().resolve()
    source_path = args.source_artifact.expanduser().resolve()
    adapter = get_adapter(args.adapter)
    geometry = adapter.make_model(args.model_seed, exact=True)
    source = adapter.load_h_artifact(source_path, geometry)
    payload_a, model_a, complex_dtype_a = load_model(
        model_a_path, source.h_matrix, device
    )
    payload_b, model_b, complex_dtype_b = load_model(
        model_b_path, source.h_matrix, device
    )
    source_hash = sha256_file(source_path)
    for payload in (payload_a, payload_b):
        if payload["source_artifact_sha256"] != source_hash:
            raise ValueError("model source artifact hash mismatch")
    if complex_dtype_a != complex_dtype_b:
        raise ValueError("models use different precisions")
    if (
        payload_a["site_count"] != payload_b["site_count"]
        and not args.allow_different_site_counts
    ):
        raise ValueError("models use different site counts")

    points, shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.points,
        seed=args.seed,
        workers=args.workers,
        cluster_size=args.sampling_cluster_size,
        backend="thread" if args.workers == 1 else "process",
    )
    maximum_potential_difference = 0.0
    maximum_metric_entry_difference = 0.0
    maximum_metric_frobenius_difference = 0.0
    maximum_relative_metric_frobenius_difference = 0.0
    for start in range(0, args.points, args.chunk_size):
        stop = min(start + args.chunk_size, args.points)
        values_numpy, derivatives_numpy = section_values_and_jacobian_arrays(
            adapter, points[start:stop], source.section_exponents
        )
        values = torch.tensor(values_numpy, dtype=complex_dtype_a, device=device)
        derivatives = torch.tensor(
            derivatives_numpy, dtype=complex_dtype_a, device=device
        )
        with torch.no_grad():
            potential_a, metric_a = model_a.potential_and_metric(values, derivatives)
            potential_b, metric_b = model_b.potential_and_metric(values, derivatives)
        potential_difference = torch.abs(potential_a - potential_b)
        metric_difference = metric_a - metric_b
        metric_entry_difference = torch.abs(metric_difference)
        metric_frobenius_difference = torch.linalg.matrix_norm(
            metric_difference, ord="fro"
        )
        metric_scale = torch.maximum(
            torch.linalg.matrix_norm(metric_a, ord="fro"),
            torch.linalg.matrix_norm(metric_b, ord="fro"),
        )
        relative_metric_frobenius_difference = metric_frobenius_difference / torch.clamp(
            metric_scale, min=torch.finfo(metric_scale.dtype).tiny
        )
        maximum_potential_difference = max(
            maximum_potential_difference, float(torch.max(potential_difference).cpu())
        )
        maximum_metric_entry_difference = max(
            maximum_metric_entry_difference,
            float(torch.max(metric_entry_difference).cpu()),
        )
        maximum_metric_frobenius_difference = max(
            maximum_metric_frobenius_difference,
            float(torch.max(metric_frobenius_difference).cpu()),
        )
        maximum_relative_metric_frobenius_difference = max(
            maximum_relative_metric_frobenius_difference,
            float(torch.max(relative_metric_frobenius_difference).cpu()),
        )

    if args.relative_metric_tolerance is None:
        success = max(
            maximum_potential_difference,
            maximum_metric_entry_difference,
            maximum_metric_frobenius_difference,
        ) <= args.absolute_tolerance
        metric_gate = "absolute"
    else:
        if args.relative_metric_tolerance <= 0:
            raise ValueError("relative metric tolerance must be positive")
        success = (
            maximum_potential_difference <= args.absolute_tolerance
            and maximum_relative_metric_frobenius_difference
            <= args.relative_metric_tolerance
        )
        metric_gate = "relative_frobenius"
    report = {
        "schema": "positive-tensor-network-common-point-equivalence-v1",
        "model_a": str(model_a_path),
        "model_a_sha256": sha256_file(model_a_path),
        "model_a_site_count": int(payload_a["site_count"]),
        "model_a_bond_dimension": int(payload_a["bond_dimension"]),
        "model_b": str(model_b_path),
        "model_b_sha256": sha256_file(model_b_path),
        "model_b_site_count": int(payload_b["site_count"]),
        "model_b_bond_dimension": int(payload_b["bond_dimension"]),
        "source_artifact": str(source_path),
        "source_artifact_sha256": source_hash,
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "seed": args.seed,
        "points": args.points,
        "sampling_cluster_size": args.sampling_cluster_size,
        "point_shards": shards,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_metric_tolerance": args.relative_metric_tolerance,
        "metric_gate": metric_gate,
        "maximum_absolute_potential_difference": maximum_potential_difference,
        "maximum_absolute_metric_entry_difference": maximum_metric_entry_difference,
        "maximum_metric_frobenius_difference": maximum_metric_frobenius_difference,
        "maximum_relative_metric_frobenius_difference": (
            maximum_relative_metric_frobenius_difference
        ),
        "success": success,
        "runtime_seconds": time.perf_counter() - started,
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(json.dumps(report, indent=2), flush=True)
    if not success:
        raise SystemExit("models failed the registered common-point equivalence gate")


if __name__ == "__main__":
    main()
