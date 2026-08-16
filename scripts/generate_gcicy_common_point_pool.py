#!/usr/bin/env python3
"""Generate one immutable common point pool for matched gCICY experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.common_point_pool import (  # noqa: E402
    save_common_point_pool,
)
from gcicy_metric.pipeline.parallel_sampling import (  # noqa: E402
    sample_points_parallel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cluster-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--backend",
        choices=("process", "thread"),
        default="process",
    )
    parser.add_argument(
        "--root-separation-tolerance",
        type=float,
        help=(
            "projective chordal-distance threshold for samplers that support "
            "duplicate-root rejection; otherwise the adapter default is used"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def git_value(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if min(args.points, args.cluster_size, args.workers) <= 0:
        raise SystemExit("points, cluster size, and workers must be positive")
    if args.points % args.cluster_size:
        raise SystemExit("point count must be divisible by cluster size")
    adapter = get_adapter(args.adapter)
    adapter.make_model(args.model_seed, exact=True)
    sampling_options = adapter.default_sampling_options()
    if args.root_separation_tolerance is not None:
        if "root_separation_tolerance" not in sampling_options:
            raise SystemExit(
                f"adapter {adapter.key} does not support "
                "--root-separation-tolerance"
            )
        sampling_options["root_separation_tolerance"] = (
            args.root_separation_tolerance
        )
    started = time.perf_counter()
    points, shards = sample_points_parallel(
        adapter,
        model_seed=args.model_seed,
        exact_model=True,
        count=args.points,
        seed=args.seed,
        workers=args.workers,
        cluster_size=args.cluster_size,
        backend=args.backend,
        sampling_options=sampling_options,
    )
    elapsed = time.perf_counter() - started
    manifest = save_common_point_pool(
        args.out,
        args.manifest,
        adapter,
        points,
        model_seed=args.model_seed,
        exact_model=True,
        split=args.split,
        sampling_seed=args.seed,
        cluster_size=args.cluster_size,
        sampling_metadata={
            "workers": args.workers,
            "backend": args.backend,
            "sampling_options": sampling_options,
            "shards": shards,
            "seconds": elapsed,
        },
        extra_metadata={
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status_porcelain": git_value("status", "--short"),
            "generator": str(Path(__file__).resolve()),
        },
    )
    print(f"wrote {args.out.expanduser().resolve()}")
    print(f"sha256={manifest['pool_sha256']}")
    print(f"points={manifest['point_count']}")
    print(f"clusters={manifest['cluster_count']}")


if __name__ == "__main__":
    main()
