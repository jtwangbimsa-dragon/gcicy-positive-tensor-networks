#!/usr/bin/env python3
"""Teacher-free native-E2 continuation of a frozen exact power-lift tree."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.exact_lift_tree import (  # noqa: E402
    native_power_lift_payload,
    power_lift_tree_from_payload,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    prepare_features,
    ratio_statistics,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--pool-points", type=Path, required=True)
    parser.add_argument("--pool-pullbacks", type=Path, required=True)
    parser.add_argument("--selection-pool-points", type=Path)
    parser.add_argument("--selection-pool-pullbacks", type=Path)
    parser.add_argument("--confirmation-pool-points", type=Path)
    parser.add_argument("--confirmation-pool-pullbacks", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--selection-size", type=int, default=20000)
    parser.add_argument("--confirmation-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=202607388)
    parser.add_argument("--active-sites", type=int, nargs="+", default=(0,))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--stochastic-batch-size",
        type=int,
        default=0,
        help="draw this many native-training points per epoch; zero uses all",
    )
    parser.add_argument("--selection-eval-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--maximum-relative-step", type=float, default=5.0e-3)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--train-chunk-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--maximum-confirmation-tail-relative-degradation",
        type=float,
        default=0.0,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tail_summary(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "q999": float(
            statistics["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99": float(
            statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum": float(
            statistics["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
    }


def tail_guard(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    relative_degradation: float,
) -> bool:
    limit = 1.0 + relative_degradation
    return bool(
        candidate["q999"] <= limit * baseline["q999"]
        and candidate["cvar99"] <= limit * baseline["cvar99"]
        and candidate["maximum"] <= limit * baseline["maximum"]
    )


def paired_improvement(
    baseline_ratio: np.ndarray,
    candidate_ratio: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    mass = np.asarray(weights, dtype=np.float64)
    mass = mass / np.sum(mass)
    baseline = np.asarray(baseline_ratio, dtype=np.float64) - 1.0
    candidate = np.asarray(candidate_ratio, dtype=np.float64) - 1.0

    def summary(values: np.ndarray) -> dict[str, float]:
        mean = float(np.sum(mass * values))
        variance = float(np.sum(mass * np.square(values - mean)))
        effective_count = float(1.0 / np.sum(np.square(mass)))
        standard_error = math.sqrt(max(variance, 0.0) / effective_count)
        return {
            "mean": mean,
            "standard_error": standard_error,
            "ci95_low": mean - 1.96 * standard_error,
            "ci95_high": mean + 1.96 * standard_error,
            "effective_count": effective_count,
        }

    return {
        "e2": summary(np.square(baseline) - np.square(candidate)),
        "sigma": summary(np.abs(baseline) - np.abs(candidate)),
    }


def load_split_arrays(
    points_path: Path,
    pullbacks_path: Path,
    *,
    sizes: tuple[int, int, int],
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[np.ndarray]]:
    points = np.load(points_path, allow_pickle=False)
    x_values = np.asarray(points["X"], dtype=np.float32)
    labels = np.column_stack(
        (
            np.asarray(points["weights"], dtype=np.float64),
            np.asarray(points["omega_squared"], dtype=np.float64),
        )
    )
    pullbacks = np.load(pullbacks_path, mmap_mode="r")
    if len(x_values) != len(labels) or len(x_values) != len(pullbacks):
        raise RuntimeError("point-pool arrays are not aligned")
    requested = sum(sizes)
    if requested > len(x_values):
        raise ValueError("requested splits exceed the point pool")
    indices = np.random.default_rng(seed).permutation(len(x_values))[:requested]
    rows = []
    index_rows = []
    start = 0
    for size in sizes:
        selected = np.asarray(indices[start : start + size], dtype=np.int64)
        rows.append(
            (
                np.asarray(x_values[selected]),
                np.asarray(labels[selected]),
                np.asarray(pullbacks[selected]),
            )
        )
        index_rows.append(selected)
        start += size
    return rows, index_rows


def load_pool_arrays(
    points_path: Path,
    pullbacks_path: Path,
    *,
    size: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    points = np.load(points_path, allow_pickle=False)
    x_values = np.asarray(points["X"], dtype=np.float32)
    labels = np.column_stack(
        (
            np.asarray(points["weights"], dtype=np.float64),
            np.asarray(points["omega_squared"], dtype=np.float64),
        )
    )
    pullbacks = np.load(pullbacks_path, mmap_mode="r")
    if len(x_values) != len(labels) or len(x_values) != len(pullbacks):
        raise RuntimeError("point-pool arrays are not aligned")
    if size > len(x_values):
        raise ValueError("requested split exceeds its point pool")
    indices = np.random.default_rng(seed).permutation(len(x_values))[:size]
    selected = np.asarray(indices, dtype=np.int64)
    return (
        np.asarray(x_values[selected]),
        np.asarray(labels[selected]),
        np.asarray(pullbacks[selected]),
    ), selected


def dataset_subset(
    dataset: dict[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    selected = torch.as_tensor(
        indices,
        dtype=torch.int64,
        device=dataset["values"].device,
    )
    result = {
        "count": len(indices),
        "values": dataset["values"].index_select(0, selected),
        "derivatives": dataset["derivatives"].index_select(0, selected),
        "weights": dataset["weights"].index_select(0, selected),
        "log_omega": dataset["log_omega"].index_select(0, selected),
    }
    for key in ("real_coordinates", "pullbacks"):
        if key in dataset:
            result[key] = dataset[key].index_select(0, selected)
    return result


def make_dataset(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    exponents: np.ndarray,
    *,
    feature_batch_size: int,
    complex_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    x_values, labels, pullbacks = arrays
    values, derivatives = prepare_features(
        x_values,
        pullbacks,
        exponents,
        feature_batch_size,
        complex_dtype=(
            np.dtype(np.complex64)
            if complex_dtype == torch.complex64
            else np.dtype(np.complex128)
        ),
    )
    weights_numpy = np.asarray(labels[:, 0], dtype=np.float64)
    weights_numpy = weights_numpy / np.sum(weights_numpy)
    omega = np.asarray(labels[:, 1], dtype=np.float64)
    real_dtype = (
        torch.float32
        if complex_dtype == torch.complex64
        else torch.float64
    )
    return {
        "count": len(values),
        "values": torch.tensor(values, dtype=complex_dtype, device=device),
        "derivatives": torch.tensor(
            derivatives,
            dtype=complex_dtype,
            device=device,
        ),
        "weights": torch.tensor(
            weights_numpy,
            dtype=real_dtype,
            device=device,
        ),
        "weights_numpy": weights_numpy,
        "log_omega": torch.log(
            torch.tensor(omega, dtype=real_dtype, device=device)
        ),
    }


def raw_log_volume(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_rows = []
    minimum_rows = []
    for start in range(0, dataset["count"], chunk_size):
        stop = min(start + chunk_size, dataset["count"])
        arguments = [
            dataset["values"][start:stop],
            dataset["derivatives"][start:stop],
        ]
        if "real_coordinates" in dataset or "pullbacks" in dataset:
            if "real_coordinates" not in dataset or "pullbacks" not in dataset:
                raise ValueError(
                    "projective residual models require coordinates and pullbacks"
                )
            arguments.extend(
                (
                    dataset["real_coordinates"][start:stop],
                    dataset["pullbacks"][start:stop],
                )
            )
        _, metric = model.potential_and_metric(*arguments)
        eigenvalues = torch.linalg.eigvalsh(metric)
        if not bool(torch.all(torch.isfinite(eigenvalues))):
            raise FloatingPointError("tree metric has nonfinite eigenvalues")
        if not bool(torch.all(eigenvalues > 0)):
            raise FloatingPointError("tree metric is not positive")
        raw_rows.append(
            torch.sum(torch.log(eigenvalues), dim=1)
            - dataset["log_omega"][start:stop]
        )
        minimum_rows.append(torch.min(eigenvalues, dim=1).values)
    return torch.cat(raw_rows), torch.cat(minimum_rows)


def normalized_native_e2(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw, _ = raw_log_volume(model, dataset, chunk_size=chunk_size)
    return normalized_ratio_and_e2(raw, dataset["weights"])


def normalized_native_e2_raw_gradient(
    raw: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return E2, normalized ratio, and the exact derivative with respect to raw."""

    if raw.ndim != 1 or weights.shape != raw.shape:
        raise ValueError("raw ratios and weights must be aligned vectors")
    normalized_weights = weights / torch.sum(weights)
    log_mean = torch.logsumexp(
        torch.log(normalized_weights) + raw,
        dim=0,
    )
    ratio = torch.exp(raw - log_mean)
    residual = ratio - 1.0
    energy = torch.sum(normalized_weights * torch.square(residual))
    normalization_coupling = torch.sum(
        normalized_weights * residual * ratio
    )
    raw_gradient = (
        2.0
        * normalized_weights
        * ratio
        * (residual - normalization_coupling)
    )
    return energy, ratio, raw_gradient


def native_e2_streaming_backward(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backpropagate exact normalized E2 without retaining every point graph."""

    with torch.no_grad():
        detached_raw, _ = raw_log_volume(
            model,
            dataset,
            chunk_size=chunk_size,
        )
        energy, ratio, raw_gradient = normalized_native_e2_raw_gradient(
            detached_raw,
            dataset["weights"],
        )
    for start in range(0, dataset["count"], chunk_size):
        stop = min(start + chunk_size, dataset["count"])
        chunk = {
            "count": stop - start,
            "values": dataset["values"][start:stop],
            "derivatives": dataset["derivatives"][start:stop],
            "weights": dataset["weights"][start:stop],
            "log_omega": dataset["log_omega"][start:stop],
        }
        for key in ("real_coordinates", "pullbacks"):
            if key in dataset:
                chunk[key] = dataset[key][start:stop]
        chunk_raw, _ = raw_log_volume(
            model,
            chunk,
            chunk_size=stop - start,
        )
        torch.sum(raw_gradient[start:stop] * chunk_raw).backward()
    return energy, ratio


def normalized_ratio_and_e2(
    raw: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native E2 while retaining the model-volume normalization graph."""

    if raw.ndim != 1 or weights.shape != raw.shape:
        raise ValueError("raw ratios and weights must be aligned vectors")
    if not bool(torch.all(torch.isfinite(raw))):
        raise FloatingPointError("raw log volume ratio is nonfinite")
    if not bool(torch.all(torch.isfinite(weights) & (weights > 0))):
        raise ValueError("native E2 weights must be finite and positive")
    normalized_weights = weights / torch.sum(weights)
    log_mean = torch.logsumexp(
        torch.log(normalized_weights) + raw,
        dim=0,
    )
    ratio = torch.exp(raw - log_mean)
    residual = ratio - 1.0
    return torch.sum(normalized_weights * torch.square(residual)), ratio


def statistics(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    with torch.no_grad():
        raw, minimum = raw_log_volume(
            model,
            dataset,
            chunk_size=chunk_size,
        )
    return ratio_statistics(
        raw.cpu().numpy(),
        dataset["weights_numpy"],
        minimum.cpu().numpy(),
    )


def main() -> None:
    args = parse_args()
    if (
        min(
            args.train_size,
            args.selection_size,
            args.confirmation_size,
            args.feature_batch_size,
            args.train_chunk_size,
            args.eval_batch_size,
            args.selection_eval_every,
        )
        <= 0
        or args.epochs <= 0
        or args.learning_rate <= 0
        or args.gradient_clip_norm <= 0
        or args.maximum_relative_step <= 0
        or args.stochastic_batch_size < 0
    ):
        raise ValueError("training, sample, batch, and trust values must be positive")
    if (
        args.maximum_selection_tail_relative_degradation < 0
        or args.maximum_confirmation_tail_relative_degradation < 0
    ):
        raise ValueError("tail degradation allowances must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    started = time.perf_counter()
    initialization_path = args.initialization.expanduser().resolve()
    points_path = args.pool_points.expanduser().resolve()
    pullbacks_path = args.pool_pullbacks.expanduser().resolve()
    separate_pool_paths = (
        args.selection_pool_points,
        args.selection_pool_pullbacks,
        args.confirmation_pool_points,
        args.confirmation_pool_pullbacks,
    )
    if any(value is not None for value in separate_pool_paths) and not all(
        value is not None for value in separate_pool_paths
    ):
        raise ValueError(
            "selection and confirmation point/pullback paths must all be supplied"
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    try:
        payload = torch.load(
            initialization_path,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("teacher_runtime_dependency") is not False:
            raise RuntimeError("native continuation received a teacher-dependent package")
        model = power_lift_tree_from_payload(payload, device=device)
        source_exponents = np.asarray(payload["source_exponents"], dtype=np.int64)
        if len(source_exponents) != model.section_count:
            raise RuntimeError("initialization section basis is inconsistent")
        active_sites = tuple(sorted(set(args.active_sites)))
        if (
            not active_sites
            or active_sites[0] < 0
            or active_sites[-1] >= model.site_count
        ):
            raise ValueError("active site index is outside the tree")
        if model.architecture != "dense_local_cores":
            raise ValueError("first native stage requires dense exact-lift cores")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        active_parameters = []
        for site in active_sites:
            model.cores[site].requires_grad_(True)
            active_parameters.append(model.cores[site])
        active_real_parameter_count = 2 * sum(
            parameter.numel() for parameter in active_parameters
        )

        if all(value is not None for value in separate_pool_paths):
            train_arrays, train_indices = load_pool_arrays(
                points_path,
                pullbacks_path,
                size=args.train_size,
                seed=args.seed,
            )
            selection_arrays, selection_indices = load_pool_arrays(
                args.selection_pool_points.expanduser().resolve(),
                args.selection_pool_pullbacks.expanduser().resolve(),
                size=args.selection_size,
                seed=args.seed + 1,
            )
            confirmation_arrays, confirmation_indices = load_pool_arrays(
                args.confirmation_pool_points.expanduser().resolve(),
                args.confirmation_pool_pullbacks.expanduser().resolve(),
                size=args.confirmation_size,
                seed=args.seed + 2,
            )
            split_arrays = [
                train_arrays,
                selection_arrays,
                confirmation_arrays,
            ]
            split_indices = [
                train_indices,
                selection_indices,
                confirmation_indices,
            ]
        else:
            split_arrays, split_indices = load_split_arrays(
                points_path,
                pullbacks_path,
                sizes=(
                    args.train_size,
                    args.selection_size,
                    args.confirmation_size,
                ),
                seed=args.seed,
            )
        complex_dtype = next(model.parameters()).dtype
        datasets = [
            make_dataset(
                arrays,
                source_exponents,
                feature_batch_size=args.feature_batch_size,
                complex_dtype=complex_dtype,
                device=device,
            )
            for arrays in split_arrays
        ]
        train, selection, confirmation = datasets
        np.savez_compressed(
            output_dir / "optimization_indices.npz",
            train_indices=split_indices[0],
            selection_indices=split_indices[1],
            confirmation_indices=split_indices[2],
        )

        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        baseline_selection, _ = statistics(
            model,
            selection,
            chunk_size=args.eval_batch_size,
        )
        baseline_selection_tail = tail_summary(baseline_selection)
        best_state = copy.deepcopy(initial_state)
        best_epoch = 0
        best_selection = baseline_selection
        best_selection_tail = baseline_selection_tail
        history = [
            {
                "epoch": 0,
                "train_e2": None,
                "selection": baseline_selection,
                "selection_tail": baseline_selection_tail,
                "accepted": True,
                "relative_step": 0.0,
            }
        ]

        optimizer = torch.optim.Adam(
            active_parameters,
            lr=args.learning_rate,
        )
        stochastic_rng = np.random.default_rng(args.seed + 3)
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_train = train
            if args.stochastic_batch_size:
                batch_size = min(args.stochastic_batch_size, train["count"])
                batch_indices = stochastic_rng.choice(
                    train["count"],
                    size=batch_size,
                    replace=False,
                )
                epoch_train = dataset_subset(train, batch_indices)
            train_e2, _ = normalized_native_e2(
                model,
                epoch_train,
                chunk_size=args.train_chunk_size,
            )
            if not bool(torch.isfinite(train_e2)):
                raise FloatingPointError("native E2 is nonfinite")
            train_e2.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    active_parameters,
                    args.gradient_clip_norm,
                )
            )
            previous = [
                parameter.detach().clone()
                for parameter in active_parameters
            ]
            optimizer.step()
            with torch.no_grad():
                step_squared = sum(
                    float(torch.sum(torch.abs(parameter - old) ** 2))
                    for parameter, old in zip(
                        active_parameters,
                        previous,
                        strict=True,
                    )
                )
                baseline_squared = sum(
                    float(torch.sum(torch.abs(old) ** 2))
                    for old in previous
                )
                relative_step = math.sqrt(
                    step_squared / max(baseline_squared, np.finfo(float).tiny)
                )
                if relative_step > args.maximum_relative_step:
                    scale = args.maximum_relative_step / relative_step
                    for parameter, old in zip(
                        active_parameters,
                        previous,
                        strict=True,
                    ):
                        parameter.copy_(old + scale * (parameter - old))
                    relative_step = args.maximum_relative_step

            evaluate_selection = bool(
                epoch % args.selection_eval_every == 0
                or epoch == args.epochs
            )
            try:
                if evaluate_selection:
                    selection_result, _ = statistics(
                        model,
                        selection,
                        chunk_size=args.eval_batch_size,
                    )
                    selection_tail = tail_summary(selection_result)
                    eligible = bool(
                        selection_result["weighted_rms_abs_residual"]
                        < best_selection["weighted_rms_abs_residual"]
                        and tail_guard(
                            selection_tail,
                            baseline_selection_tail,
                            relative_degradation=(
                                args.maximum_selection_tail_relative_degradation
                            ),
                        )
                    )
                else:
                    selection_result = None
                    selection_tail = None
                    eligible = False
            except FloatingPointError:
                with torch.no_grad():
                    for parameter, old in zip(
                        active_parameters,
                        previous,
                        strict=True,
                    ):
                        parameter.copy_(old)
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
                selection_result = None
                selection_tail = None
                eligible = False

            if eligible:
                best_epoch = epoch
                best_selection = selection_result
                best_selection_tail = selection_tail
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            row = {
                "epoch": epoch,
                "train_e2": float(train_e2.detach().cpu()),
                "gradient_norm": gradient_norm,
                "relative_step": relative_step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "selection": selection_result,
                "selection_tail": selection_tail,
                "accepted": eligible,
            }
            history.append(row)
            write_json(
                output_dir / "training_history.json",
                {
                    "best_epoch": best_epoch,
                    "rows": history,
                },
            )
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "native_training",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "latest": row,
                },
            )
            print(
                f"epoch={epoch} train_e2={row['train_e2']:.6e} "
                f"grad={gradient_norm:.3e} step={relative_step:.3e} "
                f"selection_sigma="
                f"{selection_result['sigma_official_formula'] if selection_result else float('nan'):.6e} "
                f"selection_chi="
                f"{selection_result['weighted_rms_abs_residual'] if selection_result else float('nan'):.6e} "
                f"accepted={eligible}",
                flush=True,
            )

        model.load_state_dict(initial_state)
        baseline_confirmation, baseline_ratio = statistics(
            model,
            confirmation,
            chunk_size=args.eval_batch_size,
        )
        model.load_state_dict(best_state)
        candidate_confirmation, candidate_ratio = statistics(
            model,
            confirmation,
            chunk_size=args.eval_batch_size,
        )
        paired = paired_improvement(
            baseline_ratio,
            candidate_ratio,
            confirmation["weights_numpy"],
        )
        baseline_confirmation_tail = tail_summary(baseline_confirmation)
        candidate_confirmation_tail = tail_summary(candidate_confirmation)
        confirmation_tail_passed = tail_guard(
            candidate_confirmation_tail,
            baseline_confirmation_tail,
            relative_degradation=(
                args.maximum_confirmation_tail_relative_degradation
            ),
        )
        accepted = bool(
            best_epoch > 0
            and paired["e2"]["ci95_low"] > 0
            and paired["sigma"]["ci95_low"] > 0
            and confirmation_tail_passed
            and candidate_confirmation["nonpositive_min_eigenvalue"]["count"] == 0
        )
        round_metadata = {
            "round": len(payload.get("native_rounds", ())) + 1,
            "objective": "model-normalized native E2",
            "normalization_derivative": "included by autograd",
            "active_sites": list(active_sites),
            "best_epoch": best_epoch,
            "accepted": accepted,
        }
        candidate_payload = native_power_lift_payload(
            model,
            payload,
            native_round=round_metadata,
        )
        candidate_path = output_dir / "candidate_native_tree.pt"
        torch.save(candidate_payload, candidate_path)
        accepted_path = None
        if accepted:
            accepted_path = output_dir / "accepted_native_tree.pt"
            torch.save(candidate_payload, accepted_path)

        report = {
            "schema": "quintic-native-power-lift-tree-training-v1",
            "accepted": accepted,
            "runtime_contract": {
                "initialization": str(initialization_path),
                "initialization_sha256": sha256_file(initialization_path),
                "teacher_runtime_dependency": False,
                "teacher_files_opened_after_initialization": 0,
                "separate_selection_and_confirmation_pools": all(
                    value is not None for value in separate_pool_paths
                ),
            },
            "configuration": {
                key: value
                for key, value in vars(args).items()
                if not isinstance(value, Path)
            },
            "tree": {
                "source_degree": payload["source_degree"],
                "target_degree": payload["target_degree"],
                "site_count": model.site_count,
                "bond_dimension": model.bond_dimension,
                "total_trainable_real_parameter_count": (
                    model.trainable_real_parameter_count
                ),
                "active_real_parameter_count": active_real_parameter_count,
                "positive_floor": model.positive_floor,
            },
            "objective": {
                "formula": "sum_i w_i (exp(ell_i-logsumexp_w(ell))-1)^2",
                "normalization_derivative": "included",
                "teacher_loss": None,
                "stochastic_batch_size": args.stochastic_batch_size,
            },
            "selection": {
                "baseline": baseline_selection,
                "baseline_tail": baseline_selection_tail,
                "best_epoch": best_epoch,
                "candidate": best_selection,
                "candidate_tail": best_selection_tail,
            },
            "confirmation": {
                "baseline": baseline_confirmation,
                "baseline_tail": baseline_confirmation_tail,
                "candidate": candidate_confirmation,
                "candidate_tail": candidate_confirmation_tail,
                "paired_improvement": paired,
                "tail_guard_passed": confirmation_tail_passed,
            },
            "artifacts": {
                "candidate": str(candidate_path),
                "candidate_sha256": sha256_file(candidate_path),
                "accepted": (
                    None if accepted_path is None else str(accepted_path)
                ),
                "accepted_sha256": (
                    None
                    if accepted_path is None
                    else sha256_file(accepted_path)
                ),
                "optimization_indices": str(
                    output_dir / "optimization_indices.npz"
                ),
            },
            "history": history,
            "timing_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "accepted": accepted,
                "best_epoch": best_epoch,
                "timing_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps(report, indent=2, default=str), flush=True)
    except Exception as exc:
        write_json(
            status_path,
            {
                "state": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
