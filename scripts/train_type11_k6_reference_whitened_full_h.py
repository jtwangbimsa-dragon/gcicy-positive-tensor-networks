#!/usr/bin/env python3
"""Train a complete X11 H matrix in coordinates whitened by a reference H."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, load_common_point_pool  # noqa: E402
from gcicy_metric.pipeline.common_point_pool import file_sha256  # noqa: E402
from gcicy_metric.pipeline.risk import (  # noqa: E402
    smooth_upper_log_ratio_excess_torch,
    weighted_cvar_numpy,
    weighted_cvar_torch,
)
from gcicy_metric.pipeline.tail import (  # noqa: E402
    add_metric_geometry_evidence,
    bilateral_tail_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--initial-artifact", type=Path)
    parser.add_argument("--train-common-pool", type=Path, required=True)
    parser.add_argument("--selection-common-pool", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", default="p4p1_type11_hirzebruch_x3")
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument(
        "--fixed-log-kappa",
        type=float,
        help=(
            "fixed training normalization; when omitted, compute it from the "
            "reference metric on the complete training set"
        ),
    )
    parser.add_argument("--maximum-steps", type=int, default=30000)
    parser.add_argument("--minimum-steps", type=int, default=2000)
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--metric-chunk-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--patience-evaluations", type=int, default=8)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.003)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--learning-rate-factor", type=float, default=0.3)
    parser.add_argument("--maximum-learning-rate-reductions", type=int, default=3)
    parser.add_argument("--gradient-clip-norm", type=float, default=2.0)
    parser.add_argument("--log-energy-weight", type=float, default=1.0)
    parser.add_argument("--ma-weight", type=float, default=1.0)
    parser.add_argument("--tail-weight", type=float, default=0.1)
    parser.add_argument("--tail-fraction", type=float, default=0.02)
    parser.add_argument("--tail-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--tail-smooth-temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261901)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def trace_normalize(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.complex128)
    result = 0.5 * (result + result.conj().T)
    return result * (len(result) / np.trace(result).real)


def weighted_log_mean_exp(values: np.ndarray, weights: np.ndarray) -> float:
    """Return log(sum(weights * exp(values)) / sum(weights)) stably."""

    rows = np.asarray(values, dtype=np.float64)
    masses = np.asarray(weights, dtype=np.float64)
    if rows.ndim != 1 or masses.shape != rows.shape:
        raise ValueError("values and weights must be one-dimensional and aligned")
    if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(masses)):
        raise ValueError("values and weights must be finite")
    if np.any(masses < 0) or not float(np.sum(masses)) > 0:
        raise ValueError("weights must be nonnegative with positive total mass")
    maximum = float(np.max(rows))
    return maximum + float(
        np.log(np.sum(masses * np.exp(rows - maximum)) / np.sum(masses))
    )


class CompactPositiveMatrix(torch.nn.Module):
    def __init__(self, initial: np.ndarray, *, dtype: torch.dtype, device: torch.device):
        super().__init__()
        matrix = trace_normalize(initial)
        factor = np.linalg.cholesky(matrix)
        count = len(matrix)
        rows, columns = np.tril_indices(count, k=-1)
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.diagonal_log = torch.nn.Parameter(
            torch.log(torch.tensor(np.diag(factor).real, dtype=real_dtype, device=device))
        )
        strict = factor[rows, columns]
        self.strict_real = torch.nn.Parameter(
            torch.tensor(strict.real, dtype=real_dtype, device=device)
        )
        self.strict_imag = torch.nn.Parameter(
            torch.tensor(strict.imag, dtype=real_dtype, device=device)
        )
        self.register_buffer("rows", torch.tensor(rows, dtype=torch.int64, device=device))
        self.register_buffer(
            "columns", torch.tensor(columns, dtype=torch.int64, device=device)
        )
        self.count = count
        self.complex_dtype = dtype

    def forward(self) -> torch.Tensor:
        strict = torch.complex(self.strict_real, self.strict_imag).to(
            self.complex_dtype
        )
        factor = torch.zeros(
            (self.count, self.count), dtype=self.complex_dtype, device=strict.device
        )
        factor = factor.index_put((self.rows, self.columns), strict)
        factor = factor + torch.diag(torch.exp(self.diagonal_log)).to(
            self.complex_dtype
        )
        matrix = factor @ torch.conj(factor.T)
        return matrix * (self.count / torch.real(torch.trace(matrix)))


def relative_matrix(reference_factor: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(reference_factor)
    return trace_normalize(inverse @ matrix @ inverse.conj().T)


def physical_matrix(reference_factor: np.ndarray, relative: np.ndarray) -> np.ndarray:
    return trace_normalize(reference_factor @ relative @ reference_factor.conj().T)


def cache_manifest_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"{split}_manifest.json"


def prepare_feature_cache(
    *,
    adapter: object,
    pool: object,
    exponents: np.ndarray,
    reference_factor: np.ndarray,
    cache_dir: Path,
    split: str,
    batch_size: int,
    numpy_dtype: np.dtype,
    torch_dtype: torch.dtype,
    device: torch.device,
    reference_sha256: str,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    values_path = cache_dir / f"{split}_values.npy"
    derivatives_path = cache_dir / f"{split}_derivatives.npy"
    manifest_path = cache_manifest_path(cache_dir, split)
    expected = {
        "schema": "type11-reference-whitened-feature-cache-v1",
        "split": split,
        "common_pool": str(pool.path),
        "common_pool_sha256": file_sha256(pool.path),
        "reference_artifact_sha256": reference_sha256,
        "exponents_sha256": array_sha256(exponents),
        "point_count": len(pool.points),
        "section_count": len(exponents),
        "dtype": np.dtype(numpy_dtype).name,
    }
    reusable = False
    if manifest_path.exists() and values_path.exists() and derivatives_path.exists():
        stored = json.loads(manifest_path.read_text())
        reusable = all(stored.get(key) == value for key, value in expected.items())
    if not reusable:
        temporary_values = values_path.with_name(f".{values_path.name}.tmp")
        temporary_derivatives = derivatives_path.with_name(
            f".{derivatives_path.name}.tmp"
        )
        values_cache = np.lib.format.open_memmap(
            temporary_values,
            mode="w+",
            dtype=numpy_dtype,
            shape=(len(pool.points), len(exponents)),
        )
        derivatives_cache = np.lib.format.open_memmap(
            temporary_derivatives,
            mode="w+",
            dtype=numpy_dtype,
            shape=(len(pool.points), len(exponents), 3),
        )
        factor_t = torch.as_tensor(
            reference_factor.conj(), dtype=torch.complex128, device=device
        )
        for start in range(0, len(pool.points), batch_size):
            stop = min(start + batch_size, len(pool.points))
            values, derivatives = adapter.section_values_and_jacobian_batch(
                pool.points[start:stop], exponents
            )
            values_t = torch.as_tensor(values, dtype=torch.complex128, device=device)
            derivatives_t = torch.as_tensor(
                derivatives, dtype=torch.complex128, device=device
            )
            whitened_values = values_t @ factor_t
            whitened_derivatives = torch.einsum(
                "nmj,ma->naj", derivatives_t, factor_t
            )
            values_cache[start:stop] = (
                whitened_values.to(torch_dtype).cpu().numpy()
            )
            derivatives_cache[start:stop] = (
                whitened_derivatives.to(torch_dtype).cpu().numpy()
            )
            if stop == len(pool.points) or stop % (32 * batch_size) == 0:
                print(f"feature_cache_{split}={stop}/{len(pool.points)}", flush=True)
        values_cache.flush()
        derivatives_cache.flush()
        del values_cache, derivatives_cache
        temporary_values.replace(values_path)
        temporary_derivatives.replace(derivatives_path)
        write_json(manifest_path, expected)
    return {
        "count": len(pool.points),
        "values": np.load(values_path, mmap_mode="r"),
        "derivatives": np.load(derivatives_path, mmap_mode="r"),
        "weights": np.asarray(pool.importance_weights, dtype=np.float64),
        "cluster_ids": np.asarray(pool.sampling_cluster_ids, dtype=np.int64),
        "log_omega": np.asarray(
            pool.holomorphic_volume_log_density, dtype=np.float64
        ),
        "pool_path": str(pool.path),
        "pool_sha256": file_sha256(pool.path),
    }


def torch_batch(
    dataset: dict[str, Any],
    indices: np.ndarray,
    *,
    dtype: torch.dtype,
    real_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "values": torch.as_tensor(
            np.asarray(dataset["values"][indices]), dtype=dtype, device=device
        ),
        "derivatives": torch.as_tensor(
            np.asarray(dataset["derivatives"][indices]), dtype=dtype, device=device
        ),
        "weights": torch.as_tensor(
            dataset["weights"][indices], dtype=real_dtype, device=device
        ),
        "log_omega": torch.as_tensor(
            dataset["log_omega"][indices], dtype=real_dtype, device=device
        ),
    }


def raw_log_eta(
    matrix: torch.Tensor,
    values: torch.Tensor,
    derivatives: torch.Tensor,
    log_omega: torch.Tensor,
    *,
    normalization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_values = torch.einsum("ab,nb->na", matrix, values)
    denominator = torch.real(
        torch.einsum("na,na->n", torch.conj(values), h_values)
    )
    h_derivatives = torch.einsum("ab,nbj->naj", matrix, derivatives)
    first = torch.einsum(
        "nmi,nmj->nij", torch.conj(derivatives), h_derivatives
    )
    gradient = torch.einsum("nm,nmj->nj", torch.conj(values), h_derivatives)
    metric = first / denominator[:, None, None]
    metric = metric - (
        torch.conj(gradient)[:, :, None]
        * gradient[:, None, :]
        / torch.square(denominator[:, None, None])
    )
    metric = normalization * 0.5 * (
        metric + torch.conj(torch.transpose(metric, 1, 2))
    )
    eigenvalues = torch.linalg.eigvalsh(metric)
    if not bool(torch.all(torch.isfinite(eigenvalues))) or not bool(
        torch.all(eigenvalues > 0)
    ):
        raise FloatingPointError("the full-H metric is not positive and finite")
    return torch.sum(torch.log(eigenvalues), dim=1) - log_omega, eigenvalues


def training_loss(
    matrix: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    normalization: float,
    fixed_log_kappa: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw, _ = raw_log_eta(
        matrix,
        batch["values"],
        batch["derivatives"],
        batch["log_omega"],
        normalization=normalization,
    )
    weights = batch["weights"] / torch.sum(batch["weights"])
    log_ratio = raw - fixed_log_kappa
    log_energy = torch.sum(weights * torch.square(log_ratio))
    ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
    ma = torch.sum(weights * torch.square(ratio - 1.0))
    upper_excess = smooth_upper_log_ratio_excess_torch(
        log_ratio,
        ratio_threshold=args.tail_ratio_threshold,
        smooth_temperature=args.tail_smooth_temperature,
    )
    tail = weighted_cvar_torch(
        torch.square(upper_excess),
        weights,
        tail_fraction=args.tail_fraction,
    )
    loss = (
        args.log_energy_weight * log_energy
        + args.ma_weight * ma
        + args.tail_weight * tail
    )
    return loss, {
        "log_energy": float(log_energy.detach().cpu()),
        "ma": float(ma.detach().cpu()),
        "tail": float(tail.detach().cpu()),
    }


def evaluate(
    matrix: torch.Tensor,
    dataset: dict[str, Any],
    *,
    dtype: torch.dtype,
    real_dtype: torch.dtype,
    device: torch.device,
    normalization: float,
    fixed_log_kappa: float,
    chunk_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_rows = []
    minimum_rows = []
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            indices = np.arange(start, min(start + chunk_size, dataset["count"]))
            batch = torch_batch(
                dataset, indices, dtype=dtype, real_dtype=real_dtype, device=device
            )
            raw, eigenvalues = raw_log_eta(
                matrix,
                batch["values"],
                batch["derivatives"],
                batch["log_omega"],
                normalization=normalization,
            )
            raw_rows.append(raw.cpu().numpy().astype(np.float64))
            minimum_rows.append(
                eigenvalues[:, 0].cpu().numpy().astype(np.float64)
            )
    raw = np.concatenate(raw_rows)
    minimum = np.concatenate(minimum_rows)
    weights = dataset["weights"]
    probabilities = weights / np.sum(weights)
    log_ratio = raw - fixed_log_kappa
    ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))
    log_energy = float(np.sum(probabilities * np.square(log_ratio)))
    ma = float(np.sum(probabilities * np.square(ratio - 1.0)))
    log_threshold = np.log(args.tail_ratio_threshold)
    upper_excess = args.tail_smooth_temperature * np.logaddexp(
        0.0,
        (log_ratio - log_threshold) / args.tail_smooth_temperature,
    )
    tail = weighted_cvar_numpy(
        np.square(upper_excess), weights, tail_fraction=args.tail_fraction
    ).value
    score = (
        args.log_energy_weight * log_energy
        + args.ma_weight * ma
        + args.tail_weight * tail
    )
    official = add_metric_geometry_evidence(
        bilateral_tail_metrics(raw, weights, dataset["cluster_ids"]), minimum
    )
    return {
        "selection_score": score,
        "fixed_kappa_log_energy": log_energy,
        "fixed_kappa_ma_energy": ma,
        "fixed_kappa_upper_tail_cvar": tail,
        "metrics": official,
    }


def reference_log_kappa(
    matrix: torch.Tensor,
    dataset: dict[str, Any],
    *,
    dtype: torch.dtype,
    real_dtype: torch.dtype,
    device: torch.device,
    normalization: float,
    chunk_size: int,
) -> float:
    """Compute the fixed volume normalization from one complete data set."""

    raw_rows = []
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            indices = np.arange(start, min(start + chunk_size, dataset["count"]))
            batch = torch_batch(
                dataset, indices, dtype=dtype, real_dtype=real_dtype, device=device
            )
            raw, _ = raw_log_eta(
                matrix,
                batch["values"],
                batch["derivatives"],
                batch["log_omega"],
                normalization=normalization,
            )
            raw_rows.append(raw.cpu().numpy().astype(np.float64))
    return weighted_log_mean_exp(np.concatenate(raw_rows), dataset["weights"])


def main() -> None:
    args = parse_args()
    positive = (
        args.maximum_steps,
        args.minimum_steps,
        args.training_batch_size,
        args.feature_batch_size,
        args.metric_chunk_size,
        args.eval_every,
        args.patience_evaluations,
        args.learning_rate,
        args.minimum_learning_rate,
        args.gradient_clip_norm,
        args.threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("training sizes and rates must be positive")
    if args.minimum_steps > args.maximum_steps:
        raise ValueError("minimum steps cannot exceed maximum steps")
    if not 0 <= args.minimum_relative_improvement < 1:
        raise ValueError("minimum relative improvement must lie in [0,1)")
    if not 0 < args.learning_rate_factor < 1:
        raise ValueError("learning-rate factor must lie in (0,1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    device = torch.device(args.device)
    dtype = torch.complex64 if args.precision == "complex64" else torch.complex128
    numpy_dtype = np.complex64 if dtype == torch.complex64 else np.complex128
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a full-H run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})
    started = time.perf_counter()

    adapter = get_adapter(args.adapter)
    geometric_model = adapter.make_model(args.model_seed, exact=True)
    reference_path = args.reference_artifact.expanduser().resolve()
    reference = adapter.load_h_artifact(reference_path, geometric_model)
    section_count = reference.section_count
    if section_count <= 0:
        raise ValueError("the reference artifact has no sections")
    reference_h = trace_normalize(reference.h_matrix)
    reference_factor = np.linalg.cholesky(reference_h)
    initial_path = (
        reference_path
        if args.initial_artifact is None
        else args.initial_artifact.expanduser().resolve()
    )
    initial = adapter.load_h_artifact(initial_path, geometric_model)
    if tuple(initial.degree) != tuple(reference.degree) or not np.array_equal(
        initial.section_exponents, reference.section_exponents
    ):
        raise ValueError("initial and reference artifacts use different section bases")
    initial_relative = relative_matrix(reference_factor, initial.h_matrix)

    train_pool = load_common_point_pool(
        args.train_common_pool,
        adapter,
        geometric_model,
        expected_model_seed=args.model_seed,
        expected_exact_model=True,
        expected_split="train",
    )
    selection_pool = load_common_point_pool(
        args.selection_common_pool,
        adapter,
        geometric_model,
        expected_model_seed=args.model_seed,
        expected_exact_model=True,
        expected_split="selection",
    )
    cache_dir = args.cache_dir.expanduser().resolve()
    write_json(status_path, {"state": "running", "phase": "features"})
    train = prepare_feature_cache(
        adapter=adapter,
        pool=train_pool,
        exponents=reference.section_exponents,
        reference_factor=reference_factor,
        cache_dir=cache_dir,
        split="train",
        batch_size=args.feature_batch_size,
        numpy_dtype=numpy_dtype,
        torch_dtype=dtype,
        device=device,
        reference_sha256=file_sha256(reference_path),
    )
    selection = prepare_feature_cache(
        adapter=adapter,
        pool=selection_pool,
        exponents=reference.section_exponents,
        reference_factor=reference_factor,
        cache_dir=cache_dir,
        split="selection",
        batch_size=args.feature_batch_size,
        numpy_dtype=numpy_dtype,
        torch_dtype=dtype,
        device=device,
        reference_sha256=file_sha256(reference_path),
    )
    del train_pool, selection_pool

    model = CompactPositiveMatrix(initial_relative, dtype=dtype, device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != section_count**2:
        raise RuntimeError("the compact full-H parameter count is incorrect")
    normalization = float(reference.normalization)
    fixed_log_kappa = args.fixed_log_kappa
    if fixed_log_kappa is None:
        fixed_log_kappa = reference_log_kappa(
            torch.eye(section_count, dtype=dtype, device=device),
            train,
            dtype=dtype,
            real_dtype=real_dtype,
            device=device,
            normalization=normalization,
            chunk_size=args.metric_chunk_size,
        )
        print(
            f"computed_reference_fixed_log_kappa={fixed_log_kappa:.16e}",
            flush=True,
        )
    initial_evaluation = evaluate(
        model(),
        selection,
        dtype=dtype,
        real_dtype=real_dtype,
        device=device,
        normalization=normalization,
        fixed_log_kappa=fixed_log_kappa,
        chunk_size=args.metric_chunk_size,
        args=args,
    )
    best_score = float(initial_evaluation["selection_score"])
    best_state = copy.deepcopy(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    best_step = 0
    material_reference = best_score
    stale = 0
    reductions = 0
    current_learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate)
    history = [{"step": 0, "learning_rate": current_learning_rate, **initial_evaluation}]
    write_json(output_dir / "history.json", {"rows": history})
    print(
        f"degree={tuple(reference.degree)} sections={section_count} "
        f"real_parameters={parameter_count} "
        f"initial_sigma={initial_evaluation['metrics']['sigma']:.8e} "
        f"initial_chi={initial_evaluation['metrics']['chi']:.8e} "
        f"initial_score={best_score:.8e}",
        flush=True,
    )

    permutation = rng.permutation(train["count"])
    cursor = 0
    termination = "maximum_steps"
    write_json(status_path, {"state": "running", "phase": "training", "step": 0})
    completed_steps = 0
    for step in range(1, args.maximum_steps + 1):
        completed_steps = step
        if cursor + args.training_batch_size > train["count"]:
            permutation = rng.permutation(train["count"])
            cursor = 0
        indices = permutation[cursor : cursor + args.training_batch_size]
        cursor += len(indices)
        batch = torch_batch(
            train, indices, dtype=dtype, real_dtype=real_dtype, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        relative_h = model()
        try:
            loss, components = training_loss(
                relative_h,
                batch,
                normalization=normalization,
                fixed_log_kappa=fixed_log_kappa,
                args=args,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("training produced a non-finite loss")
        except FloatingPointError as error:
            next_learning_rate = current_learning_rate * args.learning_rate_factor
            history.append(
                {
                    "step": step,
                    "event": "nonpositive_or_nonfinite_training_metric",
                    "message": str(error),
                    "learning_rate": current_learning_rate,
                    "best_step": best_step,
                    "best_score": best_score,
                }
            )
            write_json(output_dir / "history.json", {"rows": history})
            if (
                reductions < args.maximum_learning_rate_reductions
                and next_learning_rate >= args.minimum_learning_rate
            ):
                model.load_state_dict(best_state)
                current_learning_rate = next_learning_rate
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=current_learning_rate
                )
                reductions += 1
                stale = 0
                material_reference = best_score
                print(
                    "recovered_from_nonpositive_training_metric "
                    f"lr={current_learning_rate:.3e} "
                    f"restored_best_step={best_step}",
                    flush=True,
                )
                continue
            termination = "numerical_stability_plateau"
            break
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip_norm
        )
        optimizer.step()
        if step % args.eval_every and step != args.maximum_steps:
            continue

        try:
            candidate = evaluate(
                model(),
                selection,
                dtype=dtype,
                real_dtype=real_dtype,
                device=device,
                normalization=normalization,
                fixed_log_kappa=fixed_log_kappa,
                chunk_size=args.metric_chunk_size,
                args=args,
            )
        except FloatingPointError as error:
            next_learning_rate = current_learning_rate * args.learning_rate_factor
            history.append(
                {
                    "step": step,
                    "event": "nonpositive_or_nonfinite_selection_metric",
                    "message": str(error),
                    "learning_rate": current_learning_rate,
                    "best_step": best_step,
                    "best_score": best_score,
                }
            )
            write_json(output_dir / "history.json", {"rows": history})
            if (
                reductions < args.maximum_learning_rate_reductions
                and next_learning_rate >= args.minimum_learning_rate
            ):
                model.load_state_dict(best_state)
                current_learning_rate = next_learning_rate
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=current_learning_rate
                )
                reductions += 1
                stale = 0
                material_reference = best_score
                print(
                    "recovered_from_nonpositive_selection_metric "
                    f"lr={current_learning_rate:.3e} "
                    f"restored_best_step={best_step}",
                    flush=True,
                )
                continue
            termination = "numerical_stability_plateau"
            break
        score = float(candidate["selection_score"])
        if score < best_score:
            best_score = score
            best_step = step
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
        if score <= material_reference * (1.0 - args.minimum_relative_improvement):
            material_reference = score
            stale = 0
        else:
            stale += 1
        row = {
            "step": step,
            "learning_rate": current_learning_rate,
            "training_loss": float(loss.detach().cpu()),
            "training_components": components,
            "gradient_norm": float(gradient_norm),
            "best_step": best_step,
            "best_score": best_score,
            "stale_evaluations": stale,
            "learning_rate_reductions": reductions,
            **candidate,
        }
        history.append(row)
        write_json(output_dir / "history.json", {"rows": history})
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "training",
                "step": step,
                "best_step": best_step,
                "best_score": best_score,
                "selection_sigma": candidate["metrics"]["sigma"],
                "selection_chi": candidate["metrics"]["chi"],
                "learning_rate": current_learning_rate,
                "stale_evaluations": stale,
                "learning_rate_reductions": reductions,
            },
        )
        print(
            f"step={step} lr={current_learning_rate:.3e} "
            f"sigma={candidate['metrics']['sigma']:.8e} "
            f"chi={candidate['metrics']['chi']:.8e} "
            f"score={score:.8e} best={best_score:.8e} stale={stale}",
            flush=True,
        )
        if step < args.minimum_steps or stale < args.patience_evaluations:
            continue
        next_learning_rate = current_learning_rate * args.learning_rate_factor
        if (
            reductions < args.maximum_learning_rate_reductions
            and next_learning_rate >= args.minimum_learning_rate
        ):
            model.load_state_dict(best_state)
            current_learning_rate = next_learning_rate
            optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate)
            reductions += 1
            stale = 0
            material_reference = best_score
            print(
                f"reduced_learning_rate={current_learning_rate:.3e} "
                f"restored_best_step={best_step}",
                flush=True,
            )
        else:
            termination = "validation_plateau"
            break

    model.load_state_dict(best_state)
    best_relative = model().detach().cpu().numpy().astype(np.complex128)
    best_h = physical_matrix(reference_factor, best_relative)
    best_eigenvalues = np.linalg.eigvalsh(best_h)
    if best_eigenvalues[0] <= 0:
        raise FloatingPointError("the exported full H is not positive definite")
    best_evaluation = evaluate(
        model(),
        selection,
        dtype=dtype,
        real_dtype=real_dtype,
        device=device,
        normalization=normalization,
        fixed_log_kappa=fixed_log_kappa,
        chunk_size=args.metric_chunk_size,
        args=args,
    )

    if len(set(reference.degree)) == 1:
        degree_label = f"k{reference.degree[0]}"
    else:
        degree_label = "degree_" + "_".join(str(value) for value in reference.degree)
    artifact_path = output_dir / f"full_h_{degree_label}.npz"
    artifact_payload = {
        **adapter.artifact_model_payload(geometric_model, exact=True),
        "pipeline_schema_version": np.asarray(1, dtype=np.int64),
        "pipeline_adapter": np.asarray(adapter.key),
        "importance_weighted_training": np.asarray(True),
        "global_section_degree": np.asarray(reference.degree, dtype=np.int64),
        "global_section_exponents": reference.section_exponents,
        "global_h_matrix": best_h,
        "global_h_positive_relative_floor": np.asarray(
            best_eigenvalues[0] / best_eigenvalues[-1], dtype=np.float64
        ),
        "global_section_normalization": np.asarray(normalization),
        "basis_selected_indices": np.arange(section_count, dtype=np.int64),
        "basis_relation_error": np.asarray(0.0, dtype=np.float64),
        "h_parameterization": np.asarray("reference_whitened_full_cholesky"),
        "reference_artifact_sha256": np.asarray(file_sha256(reference_path)),
        "initial_artifact_sha256": np.asarray(file_sha256(initial_path)),
        "fixed_group_log_kappa": np.asarray(fixed_log_kappa),
        "optimizer_step_count": np.asarray(completed_steps, dtype=np.int64),
        "optimization_termination_reason": np.asarray(termination),
    }
    with artifact_path.open("wb") as handle:
        np.savez_compressed(handle, **artifact_payload)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "schema": "type11-reference-whitened-full-h-v2",
            "state_dict": best_state,
            "best_step": best_step,
            "reference_artifact": str(reference_path),
            "initial_artifact": str(initial_path),
            "fixed_log_kappa": fixed_log_kappa,
            "configuration": vars(args),
        },
        checkpoint_path,
    )
    report = {
        "schema": "type11-reference-whitened-full-h-v2",
        "configuration": vars(args),
        "model": {
            "section_count": section_count,
            "trainable_real_parameter_count": parameter_count,
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": file_sha256(reference_path),
            "initial_artifact": str(initial_path),
            "initial_artifact_sha256": file_sha256(initial_path),
            "artifact": str(artifact_path),
            "artifact_sha256": file_sha256(artifact_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "best_step": best_step,
            "completed_steps": completed_steps,
            "termination_reason": termination,
            "learning_rate_reductions": reductions,
        },
        "data": {
            "training_points": train["count"],
            "selection_points": selection["count"],
            "training_pool": train["pool_path"],
            "training_pool_sha256": train["pool_sha256"],
            "selection_pool": selection["pool_path"],
            "selection_pool_sha256": selection["pool_sha256"],
        },
        "initial_selection": initial_evaluation,
        "best_selection": best_evaluation,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    write_json(output_dir / "report.json", report)
    write_json(
        status_path,
        {
            "state": "complete",
            "phase": "complete",
            "best_step": best_step,
            "completed_steps": report["model"]["completed_steps"],
            "termination_reason": termination,
            "selection_sigma": best_evaluation["metrics"]["sigma"],
            "selection_chi": best_evaluation["metrics"]["chi"],
            "wall_seconds": report["wall_seconds"],
        },
    )
    print(json.dumps(json.loads(status_path.read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
