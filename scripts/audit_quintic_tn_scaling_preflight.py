#!/usr/bin/env python3
"""Audit numerical and optimizer floors of a trained quintic TN checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    positive_tensor_network_from_artifact_payload,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    ratio_statistics,
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    evaluate_raw,
    tensor_split,
    training_log_volume,
    weighted_log_mean_exp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--run-report",
        type=Path,
        help=(
            "explicit training report; defaults to RUN_DIR/report.json and is "
            "useful for atomically published study-arm reports"
        ),
    )
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--blind-reference-run-dir", type=Path)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("blind", "validation"), default="blind")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--scales", type=float, nargs="+", default=(1.0, 3.0, 10.0))
    parser.add_argument("--gradient-probe-size", type=int, default=4096)
    parser.add_argument(
        "--gradient-batch-sizes", type=int, nargs="+", default=(1024, 8192)
    )
    parser.add_argument("--gradient-microbatch-size", type=int, default=1024)
    parser.add_argument("--gradient-batches", type=int, default=6)
    parser.add_argument("--seed", type=int, default=202607201)
    parser.add_argument("--skip-gradients", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if (
        args.eval_batch_size <= 0
        or args.gradient_probe_size <= 0
        or args.gradient_microbatch_size <= 0
    ):
        raise ValueError("batch and probe sizes must be positive")
    if args.gradient_batches <= 1 or min(args.gradient_batch_sizes) <= 0:
        raise ValueError("gradient audit needs at least two positive batches")
    if not args.scales or any(
        not np.isfinite(value) or value <= 0 for value in args.scales
    ):
        raise ValueError("purification scales must be finite and positive")


def cast_artifact_precision(payload: dict[str, Any], precision: str) -> dict[str, Any]:
    """Copy an artifact while consistently casting all floating tensors."""

    if precision not in {"complex64", "complex128"}:
        raise ValueError("unsupported artifact precision")
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    real_dtype = torch.float32 if precision == "complex64" else torch.float64
    converted = copy.deepcopy(payload)
    state = {}
    for name, value in payload["state_dict"].items():
        if not torch.is_tensor(value):
            state[name] = copy.deepcopy(value)
        elif value.is_complex():
            state[name] = value.to(dtype=complex_dtype)
        elif value.is_floating_point():
            state[name] = value.to(dtype=real_dtype)
        else:
            state[name] = value.clone()
    converted["precision"] = precision
    converted["state_dict"] = state
    return converted


def scale_learned_purification_(model: torch.nn.Module, scale: float) -> None:
    """Multiply the learned amplitude by scaling exactly one boundary core."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("purification scale must be finite and positive")
    if model.architecture == "dense_local_cores":
        boundary = model.cores[0]
    elif model.architecture == "shared_local_dictionary":
        boundary = model.coefficient_cores[0]
    else:
        raise ValueError(f"unsupported architecture: {model.architecture}")
    with torch.no_grad():
        boundary.mul_(scale)


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: Iterable[float]
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    requested = tuple(float(value) for value in quantiles)
    if (
        array.ndim != 1
        or mass.shape != array.shape
        or not len(array)
        or not np.all(np.isfinite(array))
        or not np.all(np.isfinite(mass))
        or np.any(mass < 0)
        or np.sum(mass) <= 0
        or any(value < 0 or value > 1 for value in requested)
    ):
        raise ValueError("invalid weighted quantile inputs")
    order = np.argsort(array, kind="mergesort")
    ordered = array[order]
    cumulative = np.cumsum(mass[order])
    cumulative /= cumulative[-1]
    result = {}
    for quantile in requested:
        index = min(
            int(np.searchsorted(cumulative, quantile, side="left")), len(array) - 1
        )
        result[f"q{quantile:.4f}"] = float(ordered[index])
    return result


def _resolved_path(override: Path | None, configured: str) -> Path:
    return (
        (override if override is not None else Path(configured)).expanduser().resolve()
    )


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    run_dir = args.run_dir.expanduser().resolve()
    report_path = (
        args.run_report.expanduser().resolve()
        if getattr(args, "run_report", None) is not None
        else run_dir / "report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    configuration = report["configuration"]
    model = (
        args.model.expanduser().resolve()
        if args.model
        else run_dir / "best_tensor_network.pt"
    )
    source = _resolved_path(args.source_run_dir, configuration["source_run_dir"])
    configured_blind = configuration.get("blind_reference_run_dir")
    if configured_blind is None:
        configured_blind = configuration["source_run_dir"]
    blind = _resolved_path(args.blind_reference_run_dir, configured_blind)
    pullbacks = _resolved_path(args.pullbacks_dir, configuration["pullbacks_dir"])
    output_override = getattr(args, "output", None)
    output = (
        output_override.expanduser().resolve()
        if output_override
        else run_dir / "scaling_preflight_audit.json"
    )
    return {
        "run_dir": run_dir,
        "report": report_path,
        "model": model,
        "source": source,
        "blind": blind,
        "pullbacks": pullbacks,
        "output": output,
    }


def audited_input_hashes(paths: dict[str, Path], split: str) -> dict[str, str]:
    """Hash exactly the numerical inputs read for the selected audit split."""

    hashes = {
        "dataset": sha256_file(paths["source"] / "training_data" / "dataset.npz"),
        "pullback_report": sha256_file(paths["pullbacks"] / "report.json"),
    }
    if split == "validation":
        hashes["validation_pullbacks"] = sha256_file(
            paths["pullbacks"] / "validation_pullbacks.npy"
        )
    else:
        hashes["blind_points"] = sha256_file(paths["blind"] / "blind_points.npz")
        hashes["blind_pullbacks"] = sha256_file(
            paths["pullbacks"] / "blind_pullbacks.npy"
        )
    return hashes


def load_numpy_split(
    paths: dict[str, Path], split: str, limit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split == "blind":
        points = np.load(paths["blind"] / "blind_points.npz", allow_pickle=False)
        x_values = np.asarray(points["X"], dtype=np.float32)
        labels = np.column_stack(
            (
                np.asarray(points["weights"], dtype=np.float64),
                np.asarray(points["omega_squared"], dtype=np.float64),
            )
        )
        pullbacks = np.load(paths["pullbacks"] / "blind_pullbacks.npy", mmap_mode="r")
    else:
        data = np.load(
            paths["source"] / "training_data" / "dataset.npz", allow_pickle=False
        )
        x_values = np.asarray(data["X_val"], dtype=np.float32)
        labels = np.asarray(data["y_val"], dtype=np.float64)
        pullbacks = np.load(
            paths["pullbacks"] / "validation_pullbacks.npy", mmap_mode="r"
        )
    if len(x_values) != len(labels) or len(x_values) != len(pullbacks):
        raise RuntimeError("point, label, and pullback counts disagree")
    if limit:
        x_values = x_values[:limit]
        labels = labels[:limit]
        pullbacks = pullbacks[:limit]
    return x_values, labels, np.asarray(pullbacks)


def make_tensor_split(
    x_values: np.ndarray,
    labels: np.ndarray,
    pullbacks: np.ndarray,
    *,
    source_degree: int,
    precision: str,
    device: torch.device,
) -> dict[str, Any]:
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    real_dtype = torch.float32 if precision == "complex64" else torch.float64
    return tensor_split(
        x_values,
        pullbacks,
        labels,
        source_degree=source_degree,
        complex_dtype=complex_dtype,
        real_dtype=real_dtype,
        device=device,
    )


def build_model(payload: dict[str, Any], device: torch.device) -> torch.nn.Module:
    reference = payload["state_dict"]["reference_h"]
    reference_h = np.asarray(reference.detach().cpu(), dtype=np.complex128)
    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    )
    model.eval()
    return model


def floor_fraction_statistics(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> dict[str, Any]:
    fractions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            values = dataset["values"][start:stop]
            derivatives = dataset["derivatives"][start:stop]
            h_values = model.reference_h @ values.unsqueeze(-1)
            reference_norm = torch.real(
                torch.sum(torch.conj(values) * h_values.squeeze(-1), dim=1)
            )
            inverse_scale = torch.rsqrt(reference_norm)
            normalized_values = values * inverse_scale[:, None]
            normalized_derivatives = derivatives * inverse_scale[:, None, None]
            moments = model.feature_moments(normalized_values, normalized_derivatives)
            normalized_h_values = model.reference_h @ normalized_values.unsqueeze(-1)
            normalized_reference_norm = torch.real(
                torch.sum(
                    torch.conj(normalized_values) * normalized_h_values.squeeze(-1),
                    dim=1,
                )
            )
            floor_norm = (
                model.positive_floor * normalized_reference_norm**model.site_count
            )
            fractions.append((floor_norm / moments.norm).detach().cpu().numpy())
    values = np.concatenate(fractions).astype(np.float64)
    weights = dataset["weights_numpy"]
    return {
        "weighted_mean": float(np.sum(weights * values)),
        "weighted_quantiles": weighted_quantiles(
            values, weights, (0.5, 0.99, 0.999, 1.0)
        ),
        "maximum": float(np.max(values)),
    }


def fixed_normalization_statistics(
    raw: np.ndarray, weights: np.ndarray, fixed_log_kappa: float
) -> dict[str, float]:
    log_ratio = raw - fixed_log_kappa
    ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))
    residual = ratio - 1.0
    return {
        "log_kappa": float(fixed_log_kappa),
        "weighted_mean_ratio": float(np.sum(weights * ratio)),
        "sigma": float(np.sum(weights * np.abs(residual))),
        "chi": float(np.sqrt(np.sum(weights * np.square(residual)))),
        "log_energy": float(np.sum(weights * np.square(log_ratio))),
        "ma_energy": float(np.sum(weights * np.square(residual))),
    }


def evaluate_configuration(
    payload: dict[str, Any],
    dataset: dict[str, Any],
    *,
    scale: float,
    chunk_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    model = build_model(payload, device)
    scale_learned_purification_(model, scale)
    raw, minimum = evaluate_raw(model, dataset, chunk_size=chunk_size)
    normalized, _ = ratio_statistics(raw, dataset["weights_numpy"], minimum)
    empirical_log_kappa = weighted_log_mean_exp(raw, dataset["weights_numpy"])
    fixed_log_kappa = float(payload["fixed_log_kappa"])
    result = {
        "scale": float(scale),
        "precision": payload["precision"],
        "empirical_log_kappa": empirical_log_kappa,
        "empirical_minus_fixed_log_kappa": empirical_log_kappa - fixed_log_kappa,
        "renormalized": normalized,
        "fixed_normalization": fixed_normalization_statistics(
            raw, dataset["weights_numpy"], fixed_log_kappa
        ),
        "floor_fraction": floor_fraction_statistics(
            model, dataset, chunk_size=chunk_size
        ),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, raw


def _gradient_vector(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    indices: torch.Tensor,
    *,
    fixed_log_kappa: float,
    normalization: str,
) -> tuple[float, torch.Tensor]:
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    batch_weights = dataset["weights"][indices]
    batch_weights = batch_weights / torch.sum(batch_weights)
    _, metric = model.potential_and_metric(
        dataset["values"][indices], dataset["derivatives"][indices]
    )
    raw = training_log_volume(metric, "cholesky") - dataset["log_omega"][indices]
    if normalization == "fixed":
        log_kappa = raw.new_tensor(fixed_log_kappa)
    elif normalization == "empirical":
        log_kappa = torch.logsumexp(torch.log(batch_weights) + raw, dim=0)
    else:
        raise ValueError("unknown residual normalization")
    ratio = torch.exp(torch.clamp(raw - log_kappa, -20.0, 20.0))
    loss = torch.sum(batch_weights * torch.square(ratio - 1.0))
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    vectors = []
    for parameter, gradient in zip(parameters, gradients, strict=True):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        if value.is_complex():
            value = torch.view_as_real(value)
        vectors.append(value.reshape(-1).detach().to(device="cpu", dtype=torch.float64))
    return float(loss.detach().cpu()), torch.cat(vectors)


def _fixed_gradient_vector_chunked(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    indices: torch.Tensor,
    *,
    fixed_log_kappa: float,
    microbatch_size: int,
) -> tuple[float, torch.Tensor]:
    """Accumulate the exact fixed-normalization batch gradient in microbatches."""

    total_weight = torch.sum(dataset["weights"][indices])
    loss = 0.0
    gradient = None
    for start in range(0, len(indices), microbatch_size):
        chunk = indices[start : start + microbatch_size]
        chunk_mass = float(torch.sum(dataset["weights"][chunk]) / total_weight)
        chunk_loss, chunk_gradient = _gradient_vector(
            model,
            dataset,
            chunk,
            fixed_log_kappa=fixed_log_kappa,
            normalization="fixed",
        )
        loss += chunk_mass * chunk_loss
        if gradient is None:
            gradient = chunk_mass * chunk_gradient
        else:
            gradient.add_(chunk_gradient, alpha=chunk_mass)
    if gradient is None:
        raise ValueError("gradient batch must be nonempty")
    return loss, gradient


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def gradient_audit(
    payload: dict[str, Any],
    dataset: dict[str, Any],
    *,
    probe_size: int,
    batch_sizes: Iterable[int],
    batch_count: int,
    microbatch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model = build_model(payload, device)
    model.train()
    fixed_log_kappa = float(payload["fixed_log_kappa"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    probe_count = min(probe_size, dataset["count"])
    probe_indices = torch.randperm(
        dataset["count"], generator=generator, device=device
    )[:probe_count]
    fixed_loss, fixed_gradient = _gradient_vector(
        model,
        dataset,
        probe_indices,
        fixed_log_kappa=fixed_log_kappa,
        normalization="fixed",
    )
    empirical_loss, empirical_gradient = _gradient_vector(
        model,
        dataset,
        probe_indices,
        fixed_log_kappa=fixed_log_kappa,
        normalization="empirical",
    )
    result: dict[str, Any] = {
        "probe_points": probe_count,
        "normalization_gradient": {
            "fixed_loss": fixed_loss,
            "empirical_loss": empirical_loss,
            "fixed_norm": float(torch.linalg.vector_norm(fixed_gradient)),
            "empirical_norm": float(torch.linalg.vector_norm(empirical_gradient)),
            "cosine": gradient_cosine(fixed_gradient, empirical_gradient),
            "relative_difference_over_fixed": float(
                torch.linalg.vector_norm(empirical_gradient - fixed_gradient)
                / max(
                    float(torch.linalg.vector_norm(fixed_gradient)),
                    np.finfo(float).tiny,
                )
            ),
        },
        "euclidean_batch_gradient_noise": {},
    }
    for batch_size_value in batch_sizes:
        batch_size = min(int(batch_size_value), dataset["count"])
        gradients = []
        losses = []
        for _ in range(batch_count):
            indices = torch.randperm(
                dataset["count"], generator=generator, device=device
            )[:batch_size]
            loss, gradient = _fixed_gradient_vector_chunked(
                model,
                dataset,
                indices,
                fixed_log_kappa=fixed_log_kappa,
                microbatch_size=microbatch_size,
            )
            losses.append(loss)
            gradients.append(gradient)
        stacked = torch.stack(gradients)
        mean_gradient = torch.mean(stacked, dim=0)
        signal = torch.linalg.vector_norm(mean_gradient)
        noise = torch.sqrt(
            torch.mean(torch.sum(torch.square(stacked - mean_gradient), dim=1))
        )
        pairwise_cosines = [
            gradient_cosine(stacked[left], stacked[right])
            for left in range(batch_count)
            for right in range(left + 1, batch_count)
        ]
        result["euclidean_batch_gradient_noise"][str(batch_size)] = {
            "batches": batch_count,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses, ddof=1)),
            "mean_gradient_norm": float(signal),
            "rms_noise_norm": float(noise),
            "noise_to_signal_ratio": float(
                noise / max(float(signal), np.finfo(float).tiny)
            ),
            "pairwise_cosine_median": float(np.nanmedian(pairwise_cosines)),
            "pairwise_cosine_minimum": float(np.nanmin(pairwise_cosines)),
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = resolve_inputs(args)
    started = time.perf_counter()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    payload = torch.load(paths["model"], map_location="cpu", weights_only=False)
    source_degree = int(payload.get("source_degree", 1))
    x_values, labels, pullbacks = load_numpy_split(paths, args.split, args.limit)
    datasets = {
        precision: make_tensor_split(
            x_values,
            labels,
            pullbacks,
            source_degree=source_degree,
            precision=precision,
            device=device,
        )
        for precision in ("complex64", "complex128")
    }
    payloads = {
        precision: cast_artifact_precision(payload, precision)
        for precision in ("complex64", "complex128")
    }

    scale_results = []
    baseline_raw = {}
    for precision in ("complex64", "complex128"):
        scales = args.scales if precision == "complex64" else (1.0,)
        for scale in scales:
            result, raw = evaluate_configuration(
                payloads[precision],
                datasets[precision],
                scale=float(scale),
                chunk_size=args.eval_batch_size,
                device=device,
            )
            scale_results.append(result)
            if float(scale) == 1.0:
                baseline_raw[precision] = raw

    raw_difference = baseline_raw["complex128"] - baseline_raw["complex64"]
    precision_comparison = {
        "raw_log_ratio_difference_weighted_rms": float(
            np.sqrt(np.sum(datasets["complex64"]["weights_numpy"] * raw_difference**2))
        ),
        "raw_log_ratio_difference_maximum_absolute": float(
            np.max(np.abs(raw_difference))
        ),
    }
    scale_one = {
        row["precision"]: row for row in scale_results if float(row["scale"]) == 1.0
    }
    for metric_name in ("sigma_official_formula", "weighted_rms_abs_residual"):
        left = float(scale_one["complex64"]["renormalized"][metric_name])
        right = float(scale_one["complex128"]["renormalized"][metric_name])
        precision_comparison[f"{metric_name}_complex128_minus_complex64"] = right - left
    gradients = None
    if not args.skip_gradients:
        gradients = gradient_audit(
            payloads["complex64"],
            datasets["complex64"],
            probe_size=args.gradient_probe_size,
            batch_sizes=args.gradient_batch_sizes,
            batch_count=args.gradient_batches,
            microbatch_size=args.gradient_microbatch_size,
            seed=args.seed,
            device=device,
        )

    report = {
        "schema": "quintic-tn-scaling-preflight-v1",
        "scientific_scope": {
            "split": args.split,
            "point_count": int(len(x_values)),
            "pool_status": "reused architecture benchmark, not sealed final test",
            "gradient_metric": (
                "Euclidean preflight only; Fisher/GN-normalized reachability is a "
                "separate registered diagnostic"
            ),
            "precision_limit": (
                "both paths intentionally use coordinates quantized to float32 "
                "by this audit, matching the complex64 training input path; "
                "complex128 therefore audits arithmetic/model precision rather "
                "than recovering discarded coordinate bits"
            ),
        },
        "configuration": {
            "device": str(device),
            "eval_batch_size": args.eval_batch_size,
            "scales": [float(value) for value in args.scales],
            "gradient_probe_size": args.gradient_probe_size,
            "gradient_batch_sizes": list(args.gradient_batch_sizes),
            "gradient_microbatch_size": args.gradient_microbatch_size,
            "gradient_batches": args.gradient_batches,
            "seed": args.seed,
        },
        "source": {
            "run_report": str(paths["report"]),
            "run_report_sha256": sha256_file(paths["report"]),
            "model": str(paths["model"]),
            "model_sha256": sha256_file(paths["model"]),
            "audited_input_sha256": audited_input_hashes(paths, args.split),
            "source_run_dir": str(paths["source"]),
            "blind_reference_run_dir": (
                str(paths["blind"]) if args.split == "blind" else None
            ),
            "pullbacks_dir": str(paths["pullbacks"]),
        },
        "scale_and_precision": scale_results,
        "precision_comparison": precision_comparison,
        "gradient_audit": gradients,
        "timing_seconds": time.perf_counter() - started,
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    write_json(paths["output"], report)
    if args.quiet:
        print(
            json.dumps(
                {
                    "output": str(paths["output"]),
                    "points": len(x_values),
                    "timing_seconds": report["timing_seconds"],
                    "precision_comparison": precision_comparison,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
