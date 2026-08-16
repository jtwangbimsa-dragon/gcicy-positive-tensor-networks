"""Deterministic complete-cluster parallel sampling for pipeline workflows."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from typing import Any, Mapping

import numpy as np

from .adapter import GCICYAdapter


def parallel_sampling_jobs(
    count: int,
    seed: int,
    workers: int,
    cluster_size: int,
) -> list[tuple[int, int]]:
    """Split a base seed into deterministic, complete-cluster sampling jobs."""

    if min(count, workers, cluster_size) <= 0:
        raise ValueError("count, workers, and cluster_size must be positive")
    if count % cluster_size:
        raise ValueError("parallel sample count must be divisible by cluster_size")
    clusters = count // cluster_size
    active_workers = min(workers, clusters)
    quotient, remainder = divmod(clusters, active_workers)
    child_sequences = np.random.SeedSequence(seed).spawn(active_workers)
    child_seeds = [
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in child_sequences
    ]
    return [
        ((quotient + (index < remainder)) * cluster_size, child_seeds[index])
        for index in range(active_workers)
    ]


def _sample_points_worker(
    adapter_key: str,
    model_seed: int,
    exact_model: bool,
    count: int,
    seed: int,
    sampling_options: dict[str, Any],
) -> list[Any]:
    from .registry import get_adapter

    adapter = get_adapter(adapter_key)
    model = adapter.make_model(model_seed, exact=exact_model)
    return adapter.sample_points_configured(
        model,
        count,
        seed=seed,
        sampling_options=sampling_options,
    )


def sample_points_parallel(
    adapter: GCICYAdapter,
    *,
    model_seed: int,
    exact_model: bool,
    count: int,
    seed: int,
    workers: int,
    cluster_size: int,
    backend: str,
    sampling_options: Mapping[str, Any] | None = None,
) -> tuple[list[Any], list[dict[str, int]]]:
    """Sample deterministic shards and preserve their derivation metadata."""

    if backend not in {"process", "thread"}:
        raise ValueError("parallel sampling backend must be process or thread")
    jobs = parallel_sampling_jobs(count, seed, workers, cluster_size)
    resolved_sampling_options = dict(
        adapter.default_sampling_options()
        if sampling_options is None
        else sampling_options
    )
    if backend == "process":
        executor_context = ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=multiprocessing.get_context("spawn"),
        )
    else:
        executor_context = ThreadPoolExecutor(max_workers=len(jobs))
    with executor_context as executor:
        shards = list(
            executor.map(
                _sample_points_worker,
                [adapter.key] * len(jobs),
                [model_seed] * len(jobs),
                [exact_model] * len(jobs),
                [job[0] for job in jobs],
                [job[1] for job in jobs],
                [resolved_sampling_options] * len(jobs),
            )
        )
    points = [point for shard in shards for point in shard]
    if len(points) != count:
        raise RuntimeError(
            f"parallel sampler returned {len(points)} points, expected {count}"
        )
    metadata = [
        {"worker": index, "points": job[0], "seed": job[1]}
        for index, job in enumerate(jobs)
    ]
    return points, metadata
