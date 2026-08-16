#!/usr/bin/env python3
"""Train or continue a positive tensor-network metric on a type-(1,1) gCICY."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import PositiveTensorNetworkMetric, get_adapter  # noqa: E402
from gcicy_metric.pipeline.audit import standard_errors  # noqa: E402
from gcicy_metric.pipeline.common_point_pool import (  # noqa: E402
    load_common_point_pool,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from gcicy_metric.pipeline.risk import (  # noqa: E402
    smooth_upper_log_ratio_excess_torch,
    weighted_cvar_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="p4p1_type11_hirzebruch_x3",
        help="registered gCICY adapter key",
    )
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument(
        "--teacher-artifact",
        type=Path,
        help="optional same-degree full-H teacher; omit for geometric training",
    )
    parser.add_argument(
        "--site-count",
        type=int,
        help="tensor-power site count; inferred from the teacher when present",
    )
    parser.add_argument(
        "--kappa-source",
        choices=("auto", "initial_model", "saved_model", "teacher"),
        default="auto",
        help="fixed volume normalization source; saved_model preserves continuation kappa",
    )
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--bond-dimension", type=int, required=True)
    parser.add_argument("--positive-floor", type=float, default=1.0e-4)
    parser.add_argument("--initialization-noise", type=float, default=1.0e-3)
    parser.add_argument("--train-points", type=int, default=8192)
    parser.add_argument("--train-seed", type=int, default=72201)
    parser.add_argument(
        "--train-common-pool",
        type=Path,
        help="immutable common train pool; count and seed must match its manifest",
    )
    parser.add_argument("--validation-points", type=int, default=4096)
    parser.add_argument("--validation-seed", type=int, default=72202)
    parser.add_argument(
        "--validation-common-pool",
        type=Path,
        help="immutable common selection pool; count and seed must match its manifest",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--sampling-cluster-size",
        type=int,
        default=4,
        help="number of complete fibre roots returned by one sampling cluster",
    )
    parser.add_argument(
        "--teacher-chunk-size",
        type=int,
        default=512,
        help="teacher sections processed at once while preparing each dataset",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--potential-loss-weight", type=float, default=1.0)
    parser.add_argument("--metric-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--ma-loss-weight", type=float, default=0.1)
    parser.add_argument("--tail-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--tail-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--tail-smooth-temperature", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "stop after this many validation evaluations without a material "
            "selection-score improvement; zero disables early stopping"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-relative-improvement",
        type=float,
        default=0.0,
        help=(
            "minimum relative selection-score decrease that resets early-stopping "
            "patience"
        ),
    )
    parser.add_argument("--torch-seed", type=int, default=20260719)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        help="optional same-architecture model used to initialize continuation training",
    )
    parser.add_argument(
        "--allow-bond-expansion",
        action="store_true",
        help="embed a smaller-bond initial model into the requested bond dimension",
    )
    parser.add_argument(
        "--train-physical-dictionary",
        action="store_true",
        help="also optimize the shared dictionary stored in an initial model",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def validate_initial_model_compatibility(
    payload: dict,
    *,
    adapter_key: str,
    site_count: int,
    precision: str,
    source_artifact_sha256: str,
    teacher_artifact_sha256: str | None,
    target_degree: tuple[int, ...],
    target_normalization: float,
) -> str:
    """Validate a continuation checkpoint and classify its teacher transition."""

    expected = {
        "adapter": adapter_key,
        "site_count": site_count,
        "precision": precision,
        "source_artifact_sha256": source_artifact_sha256,
        "target_degree": list(target_degree),
    }
    observed = {
        "adapter": payload.get("adapter", "p4p1_type11_hirzebruch_x3"),
        "site_count": payload.get("site_count"),
        "precision": payload.get("precision"),
        "source_artifact_sha256": payload.get("source_artifact_sha256"),
        "target_degree": payload.get("target_degree"),
    }
    if observed != expected:
        raise ValueError(
            f"initial model metadata mismatch: expected {expected}, got {observed}"
        )
    saved_normalization = payload.get("target_normalization")
    if saved_normalization is None or not np.isclose(
        float(saved_normalization),
        target_normalization,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError(
            "initial model target normalization does not match the requested model"
        )

    saved_teacher_sha256 = payload.get("teacher_artifact_sha256")
    if (
        saved_teacher_sha256 is not None
        and teacher_artifact_sha256 is not None
        and saved_teacher_sha256 != teacher_artifact_sha256
    ):
        raise ValueError(
            "initial model and requested training use different non-null teachers"
        )
    if saved_teacher_sha256 is None and teacher_artifact_sha256 is not None:
        return "teacher_introduced"
    if saved_teacher_sha256 is not None and teacher_artifact_sha256 is None:
        return "teacher_removed"
    return "teacher_unchanged"


def infer_tensor_site_count(
    source_degree: tuple[int, ...],
    target_degree: tuple[int, ...],
) -> int:
    """Infer a common positive tensor-power ratio from multidegrees."""

    source = tuple(int(value) for value in source_degree)
    target = tuple(int(value) for value in target_degree)
    if len(source) != len(target) or not source:
        raise ValueError("source and teacher degrees must have equal nonzero length")
    ratios = []
    for source_value, target_value in zip(source, target, strict=True):
        if source_value <= 0 or target_value <= 0 or target_value % source_value:
            raise ValueError("teacher degree must be a positive tensor power of source")
        ratios.append(target_value // source_value)
    if len(set(ratios)) != 1:
        raise ValueError("teacher/source degree ratios must agree in every factor")
    return ratios[0]


def expand_tensor_network_state_dict(target_state: dict, source_state: dict) -> dict:
    """Embed smaller MPS bond axes while preserving the existing subnetwork."""

    if set(target_state) != set(source_state):
        raise ValueError("source and target tensor-network state keys do not match")
    expanded_state = {}
    for key, target_value in target_state.items():
        source_value = source_state[key].to(
            dtype=target_value.dtype,
            device=target_value.device,
        )
        if source_value.shape == target_value.shape:
            expanded_state[key] = source_value.clone()
            continue
        if (
            not key.startswith(("cores.", "coefficient_cores."))
            or source_value.ndim != target_value.ndim
        ):
            raise ValueError(f"state entry {key} cannot be bond-expanded")
        if any(
            source_size > target_size
            for source_size, target_size in zip(
                source_value.shape, target_value.shape, strict=True
            )
        ):
            raise ValueError(f"state entry {key} would require bond contraction")
        expanded_value = target_value.clone()
        source_slices = tuple(slice(0, size) for size in source_value.shape)
        expanded_value[source_slices] = source_value
        expanded_state[key] = expanded_value
    return expanded_state


def activate_one_sided_expanded_bond_channels(
    expanded_state: dict,
    source_state: dict,
    *,
    scale: float,
    seed: int,
) -> dict:
    """Activate new bond columns without changing the represented map.

    A path entering a new channel meets an exactly zero incoming row in the
    following core. The initial function is unchanged, while those zero rows
    generally acquire a nonzero first-order gradient.
    """

    import torch

    if scale < 0.0 or not np.isfinite(scale):
        raise ValueError("one-sided activation scale must be finite and non-negative")
    if set(expanded_state) != set(source_state):
        raise ValueError("source and expanded state keys do not match")
    activated = {key: value.clone() for key, value in expanded_state.items()}
    if scale == 0.0:
        return activated
    core_prefix = (
        "coefficient_cores."
        if any(key.startswith("coefficient_cores.") for key in source_state)
        else "cores."
    )
    core_keys = sorted(
        (key for key in source_state if key.startswith(core_prefix)),
        key=lambda key: int(key.rsplit(".", 1)[1]),
    )
    if len(core_keys) < 2:
        raise ValueError("bond activation requires at least two tensor-network cores")
    rng = np.random.default_rng(seed)
    for key in core_keys[:-1]:
        source = source_state[key]
        target = activated[key]
        if source.ndim < 3 or target.ndim != source.ndim:
            raise ValueError(f"unsupported core shape for {key}")
        old_left, old_right = source.shape[:2]
        new_right = target.shape[1]
        if target.shape[0] < old_left or new_right < old_right:
            raise ValueError(f"expanded core {key} is smaller than its source")
        if new_right == old_right:
            continue
        block_shape = (old_left, new_right - old_right, *target.shape[2:])
        real = rng.normal(size=block_shape)
        imaginary = rng.normal(size=block_shape)
        physical_size = int(np.prod(target.shape[2:]))
        values = scale * (real + 1j * imaginary) / np.sqrt(2.0 * physical_size)
        target[:old_left, old_right:new_right, ...] = torch.as_tensor(
            values,
            dtype=target.dtype,
            device=target.device,
        )
    return activated


def dense_h_potential_and_metric(values, derivatives, h_matrix, normalization: float):
    import torch

    h_values = torch.einsum("ab,nb->na", h_matrix, values)
    denominator = torch.real(
        torch.einsum("na,na->n", torch.conj(values), h_values)
    )
    if not bool(torch.all(denominator > 0)):
        raise FloatingPointError("teacher section norm is not positive")
    h_derivatives = torch.einsum("ab,nbj->naj", h_matrix, derivatives)
    first = torch.einsum(
        "nmi,nmj->nij", torch.conj(derivatives), h_derivatives
    )
    gradient = torch.einsum("nm,nmj->nj", torch.conj(values), h_derivatives)
    metric = first / denominator[:, None, None]
    metric -= (
        torch.conj(gradient)[:, :, None]
        * gradient[:, None, :]
        / denominator[:, None, None] ** 2
    )
    metric = normalization * 0.5 * (metric + torch.conj(metric.transpose(1, 2)))
    potential = normalization * torch.log(denominator)
    return potential, metric


def section_values_and_jacobian_arrays(adapter, points, section_exponents):
    """Evaluate section data without retaining one Python object per point."""

    if len(points) == 0:
        raise ValueError("section evaluation requires at least one point")
    batch_evaluator = getattr(adapter, "section_values_and_jacobian_batch", None)
    if batch_evaluator is not None:
        return batch_evaluator(points, section_exponents)
    rows = [
        adapter.section_values_and_jacobian(point, section_exponents)
        for point in points
    ]
    return (
        np.asarray([row[0] for row in rows], dtype=np.complex128),
        np.asarray([row[1] for row in rows], dtype=np.complex128),
    )


def prepare_dataset(
    adapter,
    model,
    source,
    teacher,
    *,
    count: int,
    seed: int,
    workers: int,
    device,
    real_dtype,
    complex_dtype,
    teacher_chunk_size: int = 512,
    sampling_cluster_size: int = 4,
    common_pool_path: Path | None = None,
    common_pool_split: str | None = None,
) -> dict:
    import torch

    if teacher_chunk_size <= 0:
        raise ValueError("teacher_chunk_size must be positive")
    common_pool = None
    common_pool_sha256 = None
    if common_pool_path is None:
        backend = "thread" if workers == 1 else "process"
        points, shards = sample_points_parallel(
            adapter,
            model_seed=int(model.seed),
            exact_model=True,
            count=count,
            seed=seed,
            workers=workers,
            cluster_size=sampling_cluster_size,
            backend=backend,
        )
        print(
            f"dataset seed={seed}: sampled {len(points)} points in "
            f"{len(shards)} shards",
            flush=True,
        )
    else:
        resolved_pool = common_pool_path.expanduser().resolve()
        common_pool = load_common_point_pool(
            resolved_pool,
            adapter,
            model,
            expected_model_seed=int(model.seed),
            expected_exact_model=True,
            expected_split=common_pool_split,
        )
        if int(common_pool.metadata["point_count"]) != count:
            raise ValueError("common point pool count does not match training request")
        if int(common_pool.metadata["sampling_seed"]) != seed:
            raise ValueError("common point pool seed does not match training request")
        if int(common_pool.metadata["cluster_size"]) != sampling_cluster_size:
            raise ValueError(
                "common point pool cluster size does not match training request"
            )
        points = common_pool.points
        shards = list(common_pool.metadata["sampling"].get("shards", []))
        common_pool_sha256 = sha256_file(resolved_pool)
        print(
            f"dataset seed={seed}: loaded frozen {common_pool_split} pool "
            f"{resolved_pool}",
            flush=True,
        )
    source_values_numpy, source_derivatives_numpy = section_values_and_jacobian_arrays(
        adapter, points, source.section_exponents
    )
    print(
        f"dataset seed={seed}: evaluated {source.section_count} source sections",
        flush=True,
    )
    source_values = torch.tensor(
        source_values_numpy,
        dtype=complex_dtype,
        device=device,
    )
    source_derivatives = torch.tensor(
        source_derivatives_numpy,
        dtype=complex_dtype,
        device=device,
    )
    weights_numpy = np.asarray(
        adapter.importance_weights(points)
        if common_pool is None
        else common_pool.importance_weights,
        dtype=np.float64,
    )
    weights_numpy /= np.sum(weights_numpy)
    log_omega_numpy = np.asarray(
        [adapter.holomorphic_volume_log_density(point) for point in points]
        if common_pool is None
        else common_pool.holomorphic_volume_log_density,
        dtype=np.float64,
    )
    weights = torch.tensor(weights_numpy, dtype=real_dtype, device=device)
    log_omega = torch.tensor(log_omega_numpy, dtype=real_dtype, device=device)
    teacher_potential = None
    teacher_metric = None
    teacher_inverse_cholesky = None
    teacher_log_eta = None
    if teacher is not None:
        teacher_h = torch.tensor(teacher.h_matrix, dtype=complex_dtype, device=device)
        teacher_potential_chunks = []
        teacher_metric_chunks = []
        teacher_inverse_cholesky_chunks = []
        teacher_log_eta_chunks = []
        teacher_chunk_count = (count + teacher_chunk_size - 1) // teacher_chunk_size
        teacher_progress_interval = max(1, teacher_chunk_count // 10)
        with torch.no_grad():
            for chunk_index, start in enumerate(
                range(0, count, teacher_chunk_size), start=1
            ):
                stop = min(start + teacher_chunk_size, count)
                teacher_values_numpy, teacher_derivatives_numpy = (
                    section_values_and_jacobian_arrays(
                        adapter,
                        points[start:stop],
                        teacher.section_exponents,
                    )
                )
                teacher_values = torch.tensor(
                    teacher_values_numpy,
                    dtype=complex_dtype,
                    device=device,
                )
                teacher_derivatives = torch.tensor(
                    teacher_derivatives_numpy,
                    dtype=complex_dtype,
                    device=device,
                )
                chunk_potential, chunk_metric = dense_h_potential_and_metric(
                    teacher_values,
                    teacher_derivatives,
                    teacher_h,
                    float(teacher.normalization),
                )
                chunk_eigenvalues = torch.linalg.eigvalsh(chunk_metric)
                if not bool(torch.all(chunk_eigenvalues > 0)):
                    raise FloatingPointError(
                        "teacher metric is not positive on the dataset"
                    )
                chunk_cholesky = torch.linalg.cholesky(chunk_metric)
                teacher_potential_chunks.append(chunk_potential)
                teacher_metric_chunks.append(chunk_metric)
                teacher_inverse_cholesky_chunks.append(
                    torch.linalg.inv(chunk_cholesky)
                )
                teacher_log_eta_chunks.append(
                    torch.sum(torch.log(chunk_eigenvalues), dim=1)
                    - log_omega[start:stop]
                )
                if (
                    chunk_index == teacher_chunk_count
                    or chunk_index % teacher_progress_interval == 0
                ):
                    print(
                        f"dataset seed={seed}: compressed teacher chunks "
                        f"{chunk_index}/{teacher_chunk_count}",
                        flush=True,
                    )
            teacher_potential = torch.cat(teacher_potential_chunks)
            teacher_metric = torch.cat(teacher_metric_chunks)
            teacher_inverse_cholesky = torch.cat(
                teacher_inverse_cholesky_chunks
            )
            teacher_log_eta = torch.cat(teacher_log_eta_chunks)
    try:
        point_storage_payload = adapter.point_storage_payload(points)
        point_storage_available = True
    except NotImplementedError:
        point_storage_payload = {}
        point_storage_available = False
    return {
        "count": count,
        "seed": seed,
        "shards": shards,
        "source_values": source_values,
        "source_derivatives": source_derivatives,
        "weights": weights,
        "weights_numpy": weights_numpy,
        "log_omega": log_omega,
        "teacher_potential": teacher_potential,
        "teacher_metric": teacher_metric,
        "teacher_inverse_cholesky": teacher_inverse_cholesky,
        "teacher_log_eta": teacher_log_eta,
        "sampling_cluster_ids": np.asarray(
            adapter.sampling_cluster_ids(points), dtype=np.int64
        ),
        "point_storage_payload": point_storage_payload,
        "point_storage_available": point_storage_available,
        "importance_effective_sample_size": float(1.0 / np.sum(weights_numpy**2)),
        "common_pool": (
            None
            if common_pool_path is None
            else str(common_pool_path.expanduser().resolve())
        ),
        "common_pool_sha256": common_pool_sha256,
    }


def weighted_log_mean_exp(values, weights):
    import torch

    maximum = torch.max(values)
    return maximum + torch.log(torch.sum(weights * torch.exp(values - maximum)))


def distillation_components(
    model_potential,
    model_metric,
    *,
    teacher_potential,
    teacher_inverse_cholesky,
    log_omega,
    weights,
    fixed_log_kappa,
    tail_fraction: float,
    tail_ratio_threshold: float,
    tail_smooth_temperature: float,
):
    import torch

    if (teacher_potential is None) != (teacher_inverse_cholesky is None):
        raise ValueError("teacher potential and metric data must be provided together")
    if teacher_potential is None:
        potential_loss = torch.zeros((), dtype=weights.dtype, device=weights.device)
        metric_loss = torch.zeros_like(potential_loss)
    else:
        delta_potential = model_potential - teacher_potential
        delta_potential -= torch.sum(weights * delta_potential)
        potential_loss = torch.sum(weights * delta_potential**2)

        relative = teacher_inverse_cholesky @ model_metric @ torch.conj(
            teacher_inverse_cholesky.transpose(1, 2)
        )
        relative = 0.5 * (relative + torch.conj(relative.transpose(1, 2)))
        relative_eigenvalues = torch.linalg.eigvalsh(relative)
        if not bool(torch.all(relative_eigenvalues > 0)):
            raise FloatingPointError(
                "compressed metric is not positive relative to teacher"
            )
        log_relative = torch.log(relative_eigenvalues)
        metric_loss = torch.sum(weights * torch.mean(log_relative**2, dim=1))

    model_eigenvalues = torch.linalg.eigvalsh(model_metric)
    if not bool(torch.all(model_eigenvalues > 0)):
        raise FloatingPointError("compressed metric is not positive")
    model_log_eta = torch.sum(torch.log(model_eigenvalues), dim=1) - log_omega
    log_ratio = model_log_eta - fixed_log_kappa
    log_energy_loss = torch.sum(weights * log_ratio**2)
    ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
    ma_loss = torch.sum(weights * (ratio - 1.0) ** 2)
    upper_excess = smooth_upper_log_ratio_excess_torch(
        log_ratio,
        ratio_threshold=tail_ratio_threshold,
        smooth_temperature=tail_smooth_temperature,
    )
    tail_loss = weighted_cvar_torch(
        upper_excess**2,
        weights,
        tail_fraction=tail_fraction,
    )
    return (
        potential_loss,
        metric_loss,
        log_energy_loss,
        ma_loss,
        tail_loss,
        model_log_eta,
    )


def evaluate_model(
    model,
    dataset: dict,
    *,
    fixed_log_kappa,
    tail_fraction: float = 0.01,
    tail_ratio_threshold: float = 1.5,
    tail_smooth_temperature: float = 0.05,
    chunk_size: int = 512,
) -> dict:
    import torch

    model_log_eta = []
    model_potential = [] if dataset["teacher_potential"] is not None else None
    model_metric_loss = (
        [] if dataset["teacher_inverse_cholesky"] is not None else None
    )
    minimum_metric_eigenvalue = float("inf")
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            slc = slice(start, stop)
            potential, metric = model.potential_and_metric(
                dataset["source_values"][slc],
                dataset["source_derivatives"][slc],
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            minimum_metric_eigenvalue = min(
                minimum_metric_eigenvalue,
                float(torch.min(eigenvalues).detach().cpu()),
            )
            if not bool(torch.all(eigenvalues > 0)):
                raise FloatingPointError("compressed validation metric is not positive")
            log_eta = torch.sum(torch.log(eigenvalues), dim=1) - dataset["log_omega"][slc]
            if model_metric_loss is not None:
                relative = dataset["teacher_inverse_cholesky"][slc] @ metric @ torch.conj(
                    dataset["teacher_inverse_cholesky"][slc].transpose(1, 2)
                )
                relative = 0.5 * (
                    relative + torch.conj(relative.transpose(1, 2))
                )
                relative_eigenvalues = torch.linalg.eigvalsh(relative)
                model_metric_loss.append(
                    torch.mean(torch.log(relative_eigenvalues) ** 2, dim=1)
                )
            model_log_eta.append(log_eta)
            if model_potential is not None:
                model_potential.append(potential)

        log_eta = torch.cat(model_log_eta)
        weights = dataset["weights"]
        potential_squared = None
        metric_squared = None
        if model_potential is not None:
            potential = torch.cat(model_potential)
            delta = potential - dataset["teacher_potential"]
            delta -= torch.sum(weights * delta)
            potential_squared = torch.sum(weights * delta**2)
        if model_metric_loss is not None:
            point_metric_loss = torch.cat(model_metric_loss)
            metric_squared = torch.sum(weights * point_metric_loss)
        log_ratio = log_eta - fixed_log_kappa
        fixed_kappa_log_energy = torch.sum(weights * log_ratio**2)
        ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
        fixed_kappa_ma = torch.sum(weights * (ratio - 1.0) ** 2)
        upper_excess = smooth_upper_log_ratio_excess_torch(
            log_ratio,
            ratio_threshold=tail_ratio_threshold,
            smooth_temperature=tail_smooth_temperature,
        )
        fixed_kappa_tail = weighted_cvar_torch(
            upper_excess**2,
            weights,
            tail_fraction=tail_fraction,
        )
        errors = standard_errors(
            log_eta.detach().cpu().numpy(),
            dataset["weights_numpy"],
        )
        teacher_errors = None
        if dataset["teacher_log_eta"] is not None:
            teacher_errors = standard_errors(
                dataset["teacher_log_eta"].detach().cpu().numpy(),
                dataset["weights_numpy"],
            )
    return {
        "potential_rms_to_teacher": (
            None
            if potential_squared is None
            else float(torch.sqrt(potential_squared).cpu())
        ),
        "affine_metric_rms_to_teacher": (
            None if metric_squared is None else float(torch.sqrt(metric_squared).cpu())
        ),
        "fixed_kappa_log_energy_rms": float(
            torch.sqrt(fixed_kappa_log_energy).cpu()
        ),
        "fixed_teacher_kappa_sqrt_ma_energy": float(torch.sqrt(fixed_kappa_ma).cpu()),
        "fixed_teacher_kappa_upper_tail_cvar": float(fixed_kappa_tail.cpu()),
        "minimum_metric_eigenvalue": minimum_metric_eigenvalue,
        "compressed_ma_errors": errors,
        "teacher_ma_errors": teacher_errors,
    }


def evaluate_log_eta_and_minimum_eigenvalue_arrays(
    model,
    dataset: dict,
    *,
    chunk_size: int = 512,
):
    import torch

    log_eta_rows = []
    minimum_eigenvalue_rows = []
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            _, metric = model.potential_and_metric(
                dataset["source_values"][start:stop],
                dataset["source_derivatives"][start:stop],
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(eigenvalues > 0)):
                raise FloatingPointError("compressed metric is not positive")
            log_eta_rows.append(
                torch.sum(torch.log(eigenvalues), dim=1)
                - dataset["log_omega"][start:stop]
            )
            minimum_eigenvalue_rows.append(eigenvalues[:, 0])
    return (
        torch.cat(log_eta_rows).detach().cpu().numpy(),
        torch.cat(minimum_eigenvalue_rows).detach().cpu().numpy(),
    )


def evaluate_log_eta_arrays(model, dataset: dict, *, chunk_size: int = 512):
    return evaluate_log_eta_and_minimum_eigenvalue_arrays(
        model,
        dataset,
        chunk_size=chunk_size,
    )[0]


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for tensor-network training") from exc

    args = parse_args()
    if (
        args.bond_dimension <= 0
        or args.train_points <= 0
        or args.validation_points <= 0
        or args.sampling_cluster_size <= 0
        or args.train_points % args.sampling_cluster_size
        or args.validation_points % args.sampling_cluster_size
        or args.workers <= 0
        or args.epochs < 0
        or args.batch_size <= 0
        or args.eval_every <= 0
        or args.early_stopping_patience < 0
    ):
        raise SystemExit("invalid positive integer training argument")
    if args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        raise SystemExit("learning rate and gradient clip norm must be positive")
    if (
        not 0 < args.tail_fraction <= 1
        or args.tail_ratio_threshold <= 1
        or args.tail_smooth_temperature <= 0
        or not 0 <= args.early_stopping_min_relative_improvement < 1
    ):
        raise SystemExit("invalid upper-tail loss configuration")
    loss_weights = (
        args.potential_loss_weight,
        args.metric_loss_weight,
        args.log_energy_loss_weight,
        args.ma_loss_weight,
        args.tail_loss_weight,
    )
    if min(loss_weights) < 0 or not any(loss_weights):
        raise SystemExit("loss weights must be non-negative with one enabled")

    started = time.perf_counter()
    torch.manual_seed(args.torch_seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    real_dtype = torch.float32 if args.precision == "complex64" else torch.float64
    complex_dtype = torch.complex64 if args.precision == "complex64" else torch.complex128

    adapter = get_adapter(args.adapter)
    model_geometry = adapter.make_model(args.model_seed, exact=True)
    source_path = args.source_artifact.expanduser().resolve()
    source = adapter.load_h_artifact(source_path, model_geometry)
    teacher_path = (
        None
        if args.teacher_artifact is None
        else args.teacher_artifact.expanduser().resolve()
    )
    teacher = (
        None
        if teacher_path is None
        else adapter.load_h_artifact(teacher_path, model_geometry)
    )
    source_artifact_sha256 = sha256_file(source_path)
    teacher_artifact_sha256 = (
        None if teacher_path is None else sha256_file(teacher_path)
    )
    if teacher is None:
        if args.site_count is None or args.site_count <= 0:
            raise ValueError("teacher-free training requires a positive --site-count")
        if args.potential_loss_weight or args.metric_loss_weight:
            raise ValueError(
                "teacher-free training requires zero potential and metric loss weights"
            )
        site_count = int(args.site_count)
        target_normalization = float(source.normalization) / site_count
        target_degree = tuple(int(value) * site_count for value in source.degree)
        training_mode = "teacher_free_geometric"
    else:
        inferred_site_count = infer_tensor_site_count(source.degree, teacher.degree)
        if args.site_count is not None and args.site_count != inferred_site_count:
            raise ValueError("--site-count disagrees with the teacher/source degree ratio")
        site_count = inferred_site_count
        target_normalization = float(teacher.normalization)
        target_degree = tuple(int(value) for value in teacher.degree)
        if not np.isclose(
            target_normalization,
            float(source.normalization) / site_count,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError(
                "teacher normalization is inconsistent with its tensor power"
            )
        training_mode = "full_h_teacher_distillation"
    print(f"preparing {training_mode} datasets", flush=True)
    train = prepare_dataset(
        adapter,
        model_geometry,
        source,
        teacher,
        count=args.train_points,
        seed=args.train_seed,
        workers=args.workers,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        teacher_chunk_size=args.teacher_chunk_size,
        sampling_cluster_size=args.sampling_cluster_size,
        common_pool_path=args.train_common_pool,
        common_pool_split="train",
    )
    validation = prepare_dataset(
        adapter,
        model_geometry,
        source,
        teacher,
        count=args.validation_points,
        seed=args.validation_seed,
        workers=args.workers,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        teacher_chunk_size=args.teacher_chunk_size,
        sampling_cluster_size=args.sampling_cluster_size,
        common_pool_path=args.validation_common_pool,
        common_pool_split="selection",
    )
    initial_model_payload = None
    initial_model_path = None
    initial_teacher_transition = None
    physical_dictionary = None
    trainable_physical_dictionary = False
    if args.initial_model is not None:
        initial_model_path = args.initial_model.expanduser().resolve()
        initial_model_payload = torch.load(
            initial_model_path,
            map_location="cpu",
            weights_only=False,
        )
        if initial_model_payload.get("schema") != "type11-positive-tensor-network-v1":
            raise ValueError("unrecognized initial tensor-network artifact")
        initial_teacher_transition = validate_initial_model_compatibility(
            initial_model_payload,
            adapter_key=adapter.key,
            site_count=site_count,
            precision=args.precision,
            source_artifact_sha256=source_artifact_sha256,
            teacher_artifact_sha256=teacher_artifact_sha256,
            target_degree=target_degree,
            target_normalization=target_normalization,
        )
        architecture = initial_model_payload.get(
            "architecture", "dense_local_cores"
        )
        if architecture == "shared_local_dictionary":
            dictionary_value = initial_model_payload["state_dict"].get(
                "physical_dictionary"
            )
            if dictionary_value is None:
                raise ValueError("initial dictionary model has no dictionary values")
            physical_dictionary = dictionary_value.detach().cpu().numpy()
            trainable_physical_dictionary = bool(
                initial_model_payload.get("trainable_physical_dictionary", False)
                or args.train_physical_dictionary
            )
        elif architecture != "dense_local_cores":
            raise ValueError(f"unrecognized initial architecture: {architecture}")
        elif args.train_physical_dictionary:
            raise ValueError(
                "--train-physical-dictionary requires a dictionary initial model"
            )
    elif args.train_physical_dictionary:
        raise ValueError("--train-physical-dictionary requires --initial-model")

    model = PositiveTensorNetworkMetric(
        source.h_matrix,
        site_count=site_count,
        bond_dimension=args.bond_dimension,
        target_normalization=target_normalization,
        positive_floor=args.positive_floor,
        initialization_noise=args.initialization_noise,
        physical_dictionary=physical_dictionary,
        trainable_physical_dictionary=trainable_physical_dictionary,
        seed=args.torch_seed,
        dtype=complex_dtype,
        device=device,
    )
    initialization = {
        "kind": "reference_tensor_power_plus_noise",
        "model": None,
        "model_sha256": None,
    }
    if initial_model_payload is not None:
        initial_bond_dimension = int(initial_model_payload.get("bond_dimension", 0))
        if initial_bond_dimension == model.bond_dimension:
            model.load_state_dict(initial_model_payload["state_dict"])
            initialization_kind = "continued_saved_model"
        else:
            if not args.allow_bond_expansion:
                raise ValueError(
                    "initial model bond dimension differs; pass "
                    "--allow-bond-expansion to embed a smaller model"
                )
            if initial_bond_dimension <= 0 or initial_bond_dimension > model.bond_dimension:
                raise ValueError("initial model must have a smaller positive bond dimension")
            expanded_state = expand_tensor_network_state_dict(
                model.state_dict(), initial_model_payload["state_dict"]
            )
            model.load_state_dict(expanded_state)
            initialization_kind = "bond_expanded_saved_model"
        initialization = {
            "kind": initialization_kind,
            "model": str(initial_model_path),
            "model_sha256": sha256_file(initial_model_path),
            "source_bond_dimension": initial_bond_dimension,
            "teacher_transition": initial_teacher_transition,
        }
    use_teacher_kappa = args.kappa_source == "teacher" or (
        args.kappa_source == "auto" and teacher is not None
    )
    if args.kappa_source == "saved_model":
        if (
            initial_model_payload is None
            or initial_model_payload.get("fixed_log_kappa") is None
        ):
            raise ValueError(
                "saved_model kappa source requires an initial model with saved kappa"
            )
        fixed_log_kappa = torch.tensor(
            float(initial_model_payload["fixed_log_kappa"]),
            dtype=real_dtype,
            device=device,
        )
        fixed_log_kappa_source = "continued_saved_model"
    elif use_teacher_kappa:
        if teacher is None:
            raise ValueError("teacher kappa source requires --teacher-artifact")
        fixed_log_kappa = weighted_log_mean_exp(
            train["teacher_log_eta"],
            train["weights"],
        ).detach()
        fixed_log_kappa_source = "full_h_teacher_training_pool"
    else:
        initial_log_eta = torch.tensor(
            evaluate_log_eta_arrays(model, train),
            dtype=real_dtype,
            device=device,
        )
        fixed_log_kappa = weighted_log_mean_exp(
            initial_log_eta,
            train["weights"],
        ).detach()
        fixed_log_kappa_source = "initial_model_training_pool"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.torch_seed + 1)
    history = []
    best_score = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    material_best_score = float("inf")
    evaluations_since_material_improvement = 0
    termination = "completed_requested_epochs"

    for epoch in range(args.epochs + 1):
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation_row = evaluate_model(
                model,
                validation,
                fixed_log_kappa=fixed_log_kappa,
                tail_fraction=args.tail_fraction,
                tail_ratio_threshold=args.tail_ratio_threshold,
                tail_smooth_temperature=args.tail_smooth_temperature,
            )
            score = (
                args.log_energy_loss_weight
                * validation_row["fixed_kappa_log_energy_rms"] ** 2
                + args.ma_loss_weight
                * validation_row["fixed_teacher_kappa_sqrt_ma_energy"] ** 2
                + args.tail_loss_weight
                * validation_row["fixed_teacher_kappa_upper_tail_cvar"]
            )
            if teacher is not None:
                score += (
                    args.potential_loss_weight
                    * validation_row["potential_rms_to_teacher"] ** 2
                    + args.metric_loss_weight
                    * validation_row["affine_metric_rms_to_teacher"] ** 2
                )
            row = {"epoch": epoch, "validation_selection_score": score, **validation_row}
            history.append(row)
            anchor_text = ""
            if teacher is not None:
                anchor_text = (
                    f" potential={validation_row['potential_rms_to_teacher']:.4e}"
                    f" metric={validation_row['affine_metric_rms_to_teacher']:.4e}"
                )
            print(
                f"epoch={epoch} score={score:.6e}{anchor_text} "
                f"logE={validation_row['fixed_kappa_log_energy_rms']:.4e} "
                f"chi={validation_row['compressed_ma_errors']['sqrt_squared_energy']:.4e} "
                f"max_r={validation_row['compressed_ma_errors']['normalized_ratio_max']:.4e} "
                f"tail={validation_row['fixed_teacher_kappa_upper_tail_cvar']:.4e}",
                flush=True,
            )
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
            if np.isfinite(score):
                material_threshold = material_best_score * (
                    1.0 - args.early_stopping_min_relative_improvement
                )
                if not np.isfinite(material_best_score) or score < material_threshold:
                    material_best_score = score
                    evaluations_since_material_improvement = 0
                elif epoch > 0:
                    evaluations_since_material_improvement += 1
            if (
                args.early_stopping_patience > 0
                and evaluations_since_material_improvement
                >= args.early_stopping_patience
            ):
                termination = "validation_plateau"
                print(
                    "early stopping: no material validation improvement for "
                    f"{evaluations_since_material_improvement} evaluations",
                    flush=True,
                )
                break
        if epoch == args.epochs:
            break

        model.train()
        permutation = torch.randperm(train["count"], generator=generator, device=device)
        for start in range(0, train["count"], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch_weights = train["weights"][indices]
            batch_weights = batch_weights / torch.sum(batch_weights)
            optimizer.zero_grad(set_to_none=True)
            potential, metric = model.potential_and_metric(
                train["source_values"][indices],
                train["source_derivatives"][indices],
            )
            teacher_potential = (
                None
                if train["teacher_potential"] is None
                else train["teacher_potential"][indices]
            )
            teacher_inverse_cholesky = (
                None
                if train["teacher_inverse_cholesky"] is None
                else train["teacher_inverse_cholesky"][indices]
            )
            (
                potential_loss,
                metric_loss,
                log_energy_loss,
                ma_loss,
                tail_loss,
                _,
            ) = distillation_components(
                potential,
                metric,
                teacher_potential=teacher_potential,
                teacher_inverse_cholesky=teacher_inverse_cholesky,
                log_omega=train["log_omega"][indices],
                weights=batch_weights,
                fixed_log_kappa=fixed_log_kappa,
                tail_fraction=args.tail_fraction,
                tail_ratio_threshold=args.tail_ratio_threshold,
                tail_smooth_temperature=args.tail_smooth_temperature,
            )
            loss = (
                args.potential_loss_weight * potential_loss
                + args.metric_loss_weight * metric_loss
                + args.log_energy_loss_weight * log_energy_loss
                + args.ma_loss_weight * ma_loss
                + args.tail_loss_weight * tail_loss
            )
            if not bool(torch.isfinite(loss)):
                termination = "nonfinite_training_loss"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            if model.trainable_physical_dictionary:
                model.orthonormalize_physical_dictionary_()
        if termination != "completed_requested_epochs":
            break

    model.load_state_dict(best_state)
    final_validation = evaluate_model(
        model,
        validation,
        fixed_log_kappa=fixed_log_kappa,
        tail_fraction=args.tail_fraction,
        tail_ratio_threshold=args.tail_ratio_threshold,
        tail_smooth_temperature=args.tail_smooth_temperature,
    )
    output_path = args.out.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(
        {
            "schema": "type11-positive-tensor-network-v1",
            "adapter": adapter.key,
            "model_seed": args.model_seed,
            "sampling_cluster_size": args.sampling_cluster_size,
            "state_dict": best_state,
            "source_artifact": str(source_path),
            "source_artifact_sha256": source_artifact_sha256,
            "teacher_artifact": None if teacher_path is None else str(teacher_path),
            "teacher_artifact_sha256": (
                teacher_artifact_sha256
            ),
            "training_mode": training_mode,
            "source_degree": list(source.degree),
            "target_degree": list(target_degree),
            "source_section_exponents": source.section_exponents,
            "site_count": site_count,
            "bond_dimension": args.bond_dimension,
            "architecture": model.architecture,
            "physical_dictionary_rank": model.physical_dictionary_rank,
            "trainable_physical_dictionary": model.trainable_physical_dictionary,
            "physical_dictionary_gauge": (
                "row_orthonormalized_after_each_optimizer_step"
                if model.trainable_physical_dictionary
                else "fixed"
            ),
            "target_normalization": target_normalization,
            "positive_floor": args.positive_floor,
            "precision": args.precision,
            "fixed_log_kappa": float(fixed_log_kappa.cpu()),
            "fixed_log_kappa_source": fixed_log_kappa_source,
            "train_common_pool": train["common_pool"],
            "train_common_pool_sha256": train["common_pool_sha256"],
            "validation_common_pool": validation["common_pool"],
            "validation_common_pool_sha256": validation["common_pool_sha256"],
        },
        temporary,
    )
    temporary.replace(output_path)
    device_memory = None
    if device.type == "cuda":
        device_memory = {
            "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    summary = {
        "schema": "type11-positive-tensor-network-training-v1",
        "adapter": adapter.key,
        "model": str(output_path),
        "model_sha256": sha256_file(output_path),
        "source_artifact": str(source_path),
        "source_artifact_sha256": source_artifact_sha256,
        "teacher_artifact": None if teacher_path is None else str(teacher_path),
        "teacher_artifact_sha256": (
            teacher_artifact_sha256
        ),
        "training_mode": training_mode,
        "model_seed": args.model_seed,
        "torch_seed": args.torch_seed,
        "site_count": site_count,
        "bond_dimension": args.bond_dimension,
        "architecture": model.architecture,
        "physical_dictionary_rank": model.physical_dictionary_rank,
        "trainable_physical_dictionary": model.trainable_physical_dictionary,
        "physical_dictionary_gauge": (
            "row_orthonormalized_after_each_optimizer_step"
            if model.trainable_physical_dictionary
            else "fixed"
        ),
        "trainable_real_parameter_count": model.trainable_real_parameter_count,
        "positive_floor": args.positive_floor,
        "initialization_noise": args.initialization_noise,
        "initialization": initialization,
        "device": str(device),
        "precision": args.precision,
        "teacher_chunk_size": args.teacher_chunk_size,
        "sampling_cluster_size": args.sampling_cluster_size,
        "device_memory": device_memory,
        "train": {
            "points": train["count"],
            "seed": train["seed"],
            "shards": train["shards"],
            "common_pool": train["common_pool"],
            "common_pool_sha256": train["common_pool_sha256"],
            "importance_effective_sample_size": train["importance_effective_sample_size"],
        },
        "validation": {
            "points": validation["count"],
            "seed": validation["seed"],
            "shards": validation["shards"],
            "common_pool": validation["common_pool"],
            "common_pool_sha256": validation["common_pool_sha256"],
            "importance_effective_sample_size": validation[
                "importance_effective_sample_size"
            ],
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "loss_weights": {
            "potential": args.potential_loss_weight,
            "metric": args.metric_loss_weight,
            "log_energy": args.log_energy_loss_weight,
            "ma": args.ma_loss_weight,
            "tail": args.tail_loss_weight,
        },
        "tail_loss": {
            "tail_fraction": args.tail_fraction,
            "ratio_threshold": args.tail_ratio_threshold,
            "smooth_temperature": args.tail_smooth_temperature,
            "point_loss": "squared smooth positive log-ratio excess",
        },
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "minimum_relative_improvement": (
                args.early_stopping_min_relative_improvement
            ),
            "evaluations_since_material_improvement": (
                evaluations_since_material_improvement
            ),
            "material_best_selection_score": material_best_score,
        },
        "fixed_log_kappa": float(fixed_log_kappa.cpu()),
        "fixed_log_kappa_source": fixed_log_kappa_source,
        "fixed_teacher_log_kappa": (
            None if teacher is None else float(fixed_log_kappa.cpu())
        ),
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "best_validation": final_validation,
        "history": history,
        "termination_reason": termination,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_text_atomic(summary_path, json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
