#!/usr/bin/env python3
"""Train our full-H algebraic metric on the exact cymetric quintic points."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional

from gcicy_metric.fermat_quintic import (
    exponent_compositions,
    fermat_quintic_quotient_basis,
    fermat_quotient_fubini_study_h,
    fubini_study_h,
)


PROBABILITIES = np.array(
    [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 1.0],
    dtype=np.float64,
)
UPPER_THRESHOLDS = (1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0)
LOWER_THRESHOLDS = (0.5, 0.2, 0.1, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-groups", type=int, default=3)
    parser.add_argument("--sigma-weight", type=float, default=0.5)
    parser.add_argument("--l2-weight", type=float, default=0.05)
    parser.add_argument("--barrier-weight", type=float, default=200.0)
    parser.add_argument("--checkpoint-l2-weight", type=float, default=0.25)
    parser.add_argument("--checkpoint-cvar-weight", type=float, default=0.05)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--test-batch-size", type=int, default=8192)
    parser.add_argument("--torch-seed", type=int, default=202607161)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
        return json_value(array.item() if array.ndim == 0 else array.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monomial_values_and_ambient_derivatives(
    points: np.ndarray,
    exponents: np.ndarray,
    *,
    complex_dtype: np.dtype = np.dtype(np.complex64),
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, n_coordinates = points.shape
    n_sections = len(exponents)
    powers = np.empty((batch_size, n_sections, n_coordinates), dtype=complex_dtype)
    for coordinate in range(n_coordinates):
        powers[:, :, coordinate] = np.power(
            points[:, coordinate, None], exponents[None, :, coordinate]
        )
    values = np.prod(powers, axis=-1)
    derivatives = np.empty(
        (batch_size, n_sections, n_coordinates), dtype=complex_dtype
    )
    for coordinate in range(n_coordinates):
        coordinate_factor = np.zeros((batch_size, n_sections), dtype=complex_dtype)
        positive = exponents[:, coordinate] > 0
        positive_exponents = exponents[positive, coordinate]
        coordinate_factor[:, positive] = (
            positive_exponents[None, :]
            * np.power(
                points[:, coordinate, None],
                positive_exponents[None, :] - 1,
            )
        )
        other_factor = np.ones((batch_size, n_sections), dtype=complex_dtype)
        for other in range(n_coordinates):
            if other != coordinate:
                other_factor *= powers[:, :, other]
        derivatives[:, :, coordinate] = coordinate_factor * other_factor
    return values, derivatives


def prepare_features(
    x_values: np.ndarray,
    pullback_values: np.ndarray,
    exponents: np.ndarray,
    batch_size: int,
    *,
    complex_dtype: np.dtype = np.dtype(np.complex64),
) -> tuple[np.ndarray, np.ndarray]:
    if len(x_values) != len(pullback_values):
        raise ValueError("point and pullback counts differ")
    values = np.empty((len(x_values), len(exponents)), dtype=complex_dtype)
    derivatives = np.empty(
        (len(x_values), len(exponents), pullback_values.shape[1]), dtype=complex_dtype
    )
    n_coordinates = x_values.shape[1] // 2
    for start in range(0, len(x_values), batch_size):
        stop = min(start + batch_size, len(x_values))
        block = x_values[start:stop]
        complex_points = block[:, :n_coordinates] + 1j * block[:, n_coordinates:]
        block_values, ambient_derivatives = monomial_values_and_ambient_derivatives(
            complex_points.astype(complex_dtype),
            exponents,
            complex_dtype=complex_dtype,
        )
        block_pullbacks = np.asarray(pullback_values[start:stop], dtype=complex_dtype)
        values[start:stop] = block_values
        derivatives[start:stop] = np.einsum(
            "bmi,bai->bma",
            ambient_derivatives,
            block_pullbacks,
            optimize=True,
        )
    return values, derivatives


def numpy_h_metrics(
    values: np.ndarray,
    derivatives: np.ndarray,
    h_matrix: np.ndarray,
    normalization: float,
) -> np.ndarray:
    h_values = np.einsum("ab,nb->na", h_matrix, values, optimize=True)
    denominator = np.real(np.einsum("na,na->n", np.conj(values), h_values))
    h_derivatives = np.einsum("ab,nbj->naj", h_matrix, derivatives, optimize=True)
    first = np.einsum(
        "nmi,nmj->nji", np.conj(derivatives), h_derivatives, optimize=True
    )
    gradient = np.einsum(
        "nm,nmj->nj", np.conj(values), h_derivatives, optimize=True
    )
    metric = first / denominator[:, None, None]
    metric -= (
        gradient[:, :, None]
        * np.conj(gradient)[:, None, :]
        / np.square(denominator[:, None, None])
    )
    return normalization * 0.5 * (metric + np.conj(np.swapaxes(metric, 1, 2)))


def fs_consistency_check(
    x_values: np.ndarray,
    pullback_values: np.ndarray,
    official_metrics: np.ndarray,
    exponents: np.ndarray,
    initial_h: np.ndarray,
    normalization: float,
) -> dict[str, float]:
    sample_count = min(512, len(x_values), len(official_metrics))
    sample = x_values[:sample_count]
    values, derivatives = prepare_features(
        sample,
        pullback_values[:sample_count],
        exponents,
        len(sample),
    )
    algebraic = numpy_h_metrics(values, derivatives, initial_h, normalization)
    official = np.asarray(official_metrics[:sample_count], dtype=np.complex64)
    numerator = np.real(np.vdot(official, algebraic))
    denominator = np.real(np.vdot(official, official))
    fitted_scale = float(numerator / denominator)
    relative = np.linalg.norm(
        algebraic - fitted_scale * official, axis=(1, 2)
    ) / np.maximum(np.linalg.norm(algebraic, axis=(1, 2)), 1.0e-15)
    result = {
        "fitted_algebraic_over_official_scale": fitted_scale,
        "median_relative_tensor_error_after_scale": float(np.median(relative)),
        "maximum_relative_tensor_error_after_scale": float(np.max(relative)),
    }
    if abs(fitted_scale - 1.0) > 5.0e-5 or np.max(relative) > 5.0e-5:
        raise RuntimeError(f"algebraic/official Fubini--Study mismatch: {result}")
    return result


def positive_projection(matrix: np.ndarray, relative_floor: float = 1.0e-10) -> np.ndarray:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    floor = relative_floor * max(float(np.max(eigenvalues)), 1.0)
    projected = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.conj().T
    return projected * (len(projected) / np.trace(projected).real)


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray = PROBABILITIES,
) -> np.ndarray:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, values, left=values[0], right=values[-1])


def cvar(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    threshold = weighted_quantiles(values, weights, np.array([probability]))[0]
    mask = values >= threshold
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def tail_entry(mask: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(np.count_nonzero(mask)),
        "point_fraction": float(np.mean(mask)),
        "weighted_mass": float(np.sum(weights[mask]) / np.sum(weights)),
    }


def ratio_statistics(
    raw: np.ndarray,
    weights: np.ndarray,
    min_eigenvalues: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    finite = np.isfinite(raw) & np.isfinite(weights) & (weights > 0)
    if not np.all(finite):
        raise RuntimeError("nonfinite raw metric-volume values in audit")
    maximum = float(np.max(raw))
    scaled = np.exp(raw - maximum)
    log_mean = maximum + math.log(float(np.sum(weights * scaled) / np.sum(weights)))
    ratio = np.exp(raw - log_mean)
    residual = np.abs(1.0 - ratio)
    sigma = float(np.sum(weights * residual) / np.sum(weights))
    rms = float(np.sqrt(np.sum(weights * np.square(residual)) / np.sum(weights)))
    ratio_quantiles = weighted_quantiles(ratio, weights)
    residual_quantiles = weighted_quantiles(residual, weights)
    statistics = {
        "n_points": int(len(raw)),
        "sigma_official_formula": sigma,
        "weighted_rms_abs_residual": rms,
        "ratio_weighted_quantiles": {
            f"q{probability:.4f}": float(value)
            for probability, value in zip(PROBABILITIES, ratio_quantiles)
        },
        "ratio_unweighted_quantiles": {
            f"q{probability:.4f}": float(value)
            for probability, value in zip(PROBABILITIES, np.quantile(ratio, PROBABILITIES))
        },
        "abs_residual_weighted_quantiles": {
            f"q{probability:.4f}": float(value)
            for probability, value in zip(PROBABILITIES, residual_quantiles)
        },
        "abs_residual_weighted_cvar": {
            "cvar_0.9900": cvar(residual, weights, 0.99),
            "cvar_0.9990": cvar(residual, weights, 0.999),
            "cvar_0.9999": cvar(residual, weights, 0.9999),
        },
        "ratio_upper_tails": {
            str(threshold): tail_entry(ratio > threshold, weights)
            for threshold in UPPER_THRESHOLDS
        },
        "ratio_lower_tails": {
            str(threshold): tail_entry(ratio < threshold, weights)
            for threshold in LOWER_THRESHOLDS
        },
        "nonpositive_min_eigenvalue": tail_entry(min_eigenvalues <= 0, weights),
        "min_eigenvalue_weighted_quantiles": {
            f"q{probability:.4f}": float(value)
            for probability, value in zip(
                PROBABILITIES, weighted_quantiles(min_eigenvalues, weights)
            )
        },
        "log_mean_unnormalized_ratio": log_mean,
    }
    return statistics, ratio


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source_dir = args.source_run_dir.resolve()
    pullbacks_dir = (
        args.pullbacks_dir.resolve()
        if args.pullbacks_dir is not None
        else source_dir / "full_h_common_geometry"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    configuration = vars(args).copy()
    configuration["source_run_dir"] = str(source_dir)
    configuration["pullbacks_dir"] = str(pullbacks_dir)
    configuration["output_dir"] = str(output_dir)
    write_json(status_path, {"state": "running", "phase": "initializing"})

    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU")
        device = torch.device(args.device)
        torch.manual_seed(args.torch_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.torch_seed)
            torch.cuda.reset_peak_memory_stats(device)

        dataset_path = source_dir / "training_data" / "dataset.npz"
        basis_path = source_dir / "training_data" / "basis.pickle"
        blind_points_path = source_dir / "blind_points.npz"
        cymetric_tail_path = source_dir / "blind_test_tail_arrays.npz"
        pullback_report_path = pullbacks_dir / "report.json"
        train_pullbacks_path = pullbacks_dir / "train_pullbacks.npy"
        validation_pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
        blind_pullbacks_path = pullbacks_dir / "blind_pullbacks.npy"
        official_fs_metrics_path = (
            pullbacks_dir / "validation_official_fs_metrics.npy"
        )
        for path in (
            dataset_path,
            basis_path,
            blind_points_path,
            cymetric_tail_path,
            pullback_report_path,
            train_pullbacks_path,
            validation_pullbacks_path,
            blind_pullbacks_path,
            official_fs_metrics_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        pullback_report = json.loads(pullback_report_path.read_text())
        current_source_hashes = {
            "dataset": sha256_file(dataset_path),
            "basis": sha256_file(basis_path),
            "blind_points": sha256_file(blind_points_path),
            "cymetric_tail": sha256_file(cymetric_tail_path),
        }
        if pullback_report.get("source_sha256") != current_source_hashes:
            raise RuntimeError("pullbacks were not generated from the current source data")
        pullback_paths = {
            "train": train_pullbacks_path,
            "validation": validation_pullbacks_path,
            "blind": blind_pullbacks_path,
            "validation_official_fs_metrics": official_fs_metrics_path,
        }
        current_pullback_hashes = {
            name: sha256_file(path) for name, path in pullback_paths.items()
        }
        if pullback_report.get("output_sha256") != current_pullback_hashes:
            raise RuntimeError("pullback artifact hashes do not match their report")

        data = np.load(dataset_path, allow_pickle=False)
        train_x = np.asarray(data["X_train"], dtype=np.float32)
        train_y = np.asarray(data["y_train"], dtype=np.float64)
        validation_x = np.asarray(data["X_val"], dtype=np.float32)
        validation_y = np.asarray(data["y_val"], dtype=np.float64)
        train_pullbacks = np.load(train_pullbacks_path, mmap_mode="r")
        validation_pullbacks = np.load(validation_pullbacks_path, mmap_mode="r")
        if len(train_pullbacks) != len(train_x):
            raise RuntimeError("training pullback count does not match dataset")
        if len(validation_pullbacks) != len(validation_x):
            raise RuntimeError("validation pullback count does not match dataset")
        if args.train_limit:
            train_x, train_y = train_x[: args.train_limit], train_y[: args.train_limit]
            train_pullbacks = train_pullbacks[: args.train_limit]
        if args.validation_limit:
            validation_x = validation_x[: args.validation_limit]
            validation_y = validation_y[: args.validation_limit]
            validation_pullbacks = validation_pullbacks[: args.validation_limit]

        if train_x.shape[1] // 2 != 5:
            raise ValueError("the Fermat quotient basis requires five homogeneous coordinates")
        ambient_exponents, exponents, quotient_lift = fermat_quintic_quotient_basis(
            args.degree
        )
        initial_h = fermat_quotient_fubini_study_h(
            ambient_exponents, quotient_lift
        )
        normalization = 1.0 / (math.pi * args.degree)
        official_fs_metrics = np.load(official_fs_metrics_path, mmap_mode="r")
        fs_check = fs_consistency_check(
            validation_x,
            validation_pullbacks,
            official_fs_metrics,
            exponents,
            initial_h,
            normalization,
        )

        write_json(status_path, {"state": "running", "phase": "preparing_features"})
        feature_started = time.perf_counter()
        train_values, train_derivatives = prepare_features(
            train_x, train_pullbacks, exponents, args.feature_batch_size
        )
        validation_values, validation_derivatives = prepare_features(
            validation_x,
            validation_pullbacks,
            exponents,
            args.feature_batch_size,
        )
        feature_seconds = time.perf_counter() - feature_started

        complex_dtype = torch.complex64
        real_dtype = torch.float32

        def tensor_dataset(
            values: np.ndarray,
            derivatives: np.ndarray,
            labels: np.ndarray,
        ) -> dict[str, torch.Tensor]:
            return {
                "values": torch.tensor(values, dtype=complex_dtype, device=device),
                "derivatives": torch.tensor(
                    derivatives, dtype=complex_dtype, device=device
                ),
                "weights": torch.tensor(labels[:, 0], dtype=real_dtype, device=device),
                "log_omega": torch.log(
                    torch.tensor(labels[:, 1], dtype=real_dtype, device=device)
                ),
            }

        training_groups = []
        for indices in np.array_split(np.arange(len(train_x)), args.train_groups):
            training_groups.append(
                tensor_dataset(
                    train_values[indices], train_derivatives[indices], train_y[indices]
                )
            )
        validation = tensor_dataset(
            validation_values, validation_derivatives, validation_y
        )
        del train_values, train_derivatives, validation_values, validation_derivatives

        initial_cholesky = np.linalg.cholesky(initial_h)
        diagonal_log = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    np.diag(initial_cholesky).real,
                    dtype=real_dtype,
                    device=device,
                )
            )
        )
        lower_real = torch.nn.Parameter(
            torch.tensor(initial_cholesky.real, dtype=real_dtype, device=device)
        )
        lower_imag = torch.nn.Parameter(
            torch.tensor(initial_cholesky.imag, dtype=real_dtype, device=device)
        )
        lower_mask = torch.tril(torch.ones_like(lower_real), diagonal=-1)

        def current_h() -> torch.Tensor:
            lower = lower_mask * lower_real + 1j * lower_mask * lower_imag
            cholesky = lower.to(complex_dtype) + torch.diag(
                torch.exp(diagonal_log)
            ).to(complex_dtype)
            matrix = cholesky @ torch.conj(cholesky.T)
            return matrix * (len(initial_h) / torch.real(torch.trace(matrix)))

        def components(
            dataset: dict[str, torch.Tensor],
            matrix: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            values = dataset["values"]
            derivatives = dataset["derivatives"]
            weights = dataset["weights"]
            h_values = torch.einsum("ab,nb->na", matrix, values)
            denominator = torch.real(
                torch.einsum("na,na->n", torch.conj(values), h_values)
            )
            h_derivatives = torch.einsum("ab,nbj->naj", matrix, derivatives)
            first = torch.einsum(
                "nmi,nmj->nji", torch.conj(derivatives), h_derivatives
            )
            gradient = torch.einsum(
                "nm,nmj->nj", torch.conj(values), h_derivatives
            )
            metric = first / denominator[:, None, None]
            metric = metric - (
                gradient[:, :, None]
                * torch.conj(gradient)[:, None, :]
                / denominator[:, None, None] ** 2
            )
            metric = normalization * 0.5 * (
                metric + torch.conj(torch.transpose(metric, 1, 2))
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            raw = torch.log(torch.clamp(eigenvalues, min=1.0e-12)).sum(dim=1)
            raw = raw - dataset["log_omega"]
            positive_weights = torch.clamp(
                weights, min=torch.finfo(weights.dtype).tiny
            )
            log_mean = torch.logsumexp(torch.log(positive_weights) + raw, dim=0)
            log_mean = log_mean - torch.log(torch.sum(positive_weights))
            centered = raw - log_mean
            ratio = torch.exp(centered)
            weighted_mean = lambda value: torch.sum(weights * value) / torch.sum(weights)
            sigma = weighted_mean(torch.abs(1.0 - ratio))
            l2 = torch.sqrt(
                weighted_mean(torch.square(1.0 - ratio))
                + torch.finfo(weights.dtype).eps
            )
            log_variance = weighted_mean(torch.square(centered))
            scale = torch.clamp(
                torch.mean(torch.abs(eigenvalues), dim=1, keepdim=True), min=1.0e-14
            )
            relative_eigenvalues = eigenvalues / scale
            barrier = torch_functional.softplus(
                (1.0e-8 - relative_eigenvalues) * 80.0
            ).mean() / 80.0
            return {
                "raw": raw,
                "ratio": ratio,
                "eigenvalues": eigenvalues,
                "sigma": sigma,
                "l2": l2,
                "log_variance": log_variance,
                "barrier": barrier,
            }

        def validation_row(matrix: np.ndarray, epoch: int) -> dict[str, Any]:
            matrix_t = torch.tensor(matrix, dtype=complex_dtype, device=device)
            with torch.no_grad():
                result = components(validation, matrix_t)
            ratio = result["ratio"].detach().cpu().numpy().astype(np.float64)
            weights = validation["weights"].detach().cpu().numpy().astype(np.float64)
            residual = np.abs(1.0 - ratio)
            cvar_99 = cvar(residual, weights, 0.99)
            sigma = float(result["sigma"].detach().cpu())
            l2 = float(result["l2"].detach().cpu())
            return {
                "epoch": epoch,
                "sigma": sigma,
                "l2": l2,
                "log_variance": float(result["log_variance"].detach().cpu()),
                "cvar_0.99_abs_residual": cvar_99,
                "maximum_ratio": float(np.max(ratio)),
                "minimum_ratio": float(np.min(ratio)),
                "minimum_metric_eigenvalue": float(
                    torch.min(result["eigenvalues"]).detach().cpu()
                ),
                "score": sigma
                + args.checkpoint_l2_weight * l2
                + args.checkpoint_cvar_weight * cvar_99,
            }

        optimizer = torch.optim.Adam(
            [diagonal_log, lower_real, lower_imag], lr=args.learning_rate
        )
        best_h = initial_h.copy()
        history = [validation_row(best_h, 0)]
        best_score = history[0]["score"]
        best_epoch = 0
        write_json(output_dir / "training_history.json", {"rows": history})
        print(
            f"FS check max relative error={fs_check['maximum_relative_tensor_error_after_scale']:.3e}; "
            f"sections={len(exponents)}; initial validation sigma={history[0]['sigma']:.6e}",
            flush=True,
        )

        write_json(status_path, {"state": "running", "phase": "training", "epoch": 0})
        optimization_started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            group_losses = []
            for dataset in training_groups:
                result = components(dataset, current_h())
                loss = (
                    result["log_variance"]
                    + args.sigma_weight * result["sigma"]
                    + args.l2_weight * result["l2"]
                    + args.barrier_weight * result["barrier"]
                ) / len(training_groups)
                loss.backward()
                group_losses.append(float(loss.detach().cpu()))
            optimizer.step()

            if epoch % args.eval_every != 0 and epoch != args.epochs:
                continue
            candidate = positive_projection(current_h().detach().cpu().numpy())
            row = validation_row(candidate, epoch)
            row["training_loss_sum"] = float(sum(group_losses))
            history.append(row)
            accepted = row["score"] < best_score
            if accepted:
                best_h = candidate
                best_score = row["score"]
                best_epoch = epoch
                np.savez_compressed(
                    output_dir / "best_h_metric.npz",
                    degree=np.asarray(args.degree),
                    exponents=exponents,
                    ambient_exponents=ambient_exponents,
                    quotient_lift=quotient_lift,
                    initial_h_matrix=initial_h,
                    global_h_matrix=best_h,
                    best_epoch=np.asarray(best_epoch),
                    normalization=np.asarray(normalization),
                )
            write_json(output_dir / "training_history.json", {"rows": history})
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "training",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "latest": row,
                },
            )
            print(
                f"epoch={epoch} sigma={row['sigma']:.6e} l2={row['l2']:.6e} "
                f"cvar99={row['cvar_0.99_abs_residual']:.6e} "
                f"ratio=[{row['minimum_ratio']:.6e},{row['maximum_ratio']:.6e}] "
                f"accepted={accepted}",
                flush=True,
            )
        optimization_seconds = time.perf_counter() - optimization_started

        write_json(status_path, {"state": "running", "phase": "blind_audit"})
        blind = np.load(blind_points_path, allow_pickle=False)
        test_x = np.asarray(blind["X"], dtype=np.float32)
        test_weights = np.asarray(blind["weights"], dtype=np.float64)
        test_omega = np.asarray(blind["omega_squared"], dtype=np.float64)
        test_pullbacks = np.load(blind_pullbacks_path, mmap_mode="r")
        if len(test_pullbacks) != len(test_x):
            raise RuntimeError("blind pullback count does not match blind points")
        if args.test_limit:
            test_x = test_x[: args.test_limit]
            test_weights = test_weights[: args.test_limit]
            test_omega = test_omega[: args.test_limit]
            test_pullbacks = test_pullbacks[: args.test_limit]

        def audit_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            raw_values = np.empty(len(test_x), dtype=np.float64)
            minimum_eigenvalues = np.empty(len(test_x), dtype=np.float64)
            matrix_t = torch.tensor(matrix, dtype=complex_dtype, device=device)
            for start in range(0, len(test_x), args.test_batch_size):
                stop = min(start + args.test_batch_size, len(test_x))
                block_values, block_derivatives = prepare_features(
                    test_x[start:stop],
                    test_pullbacks[start:stop],
                    exponents,
                    args.feature_batch_size,
                )
                block = tensor_dataset(
                    block_values,
                    block_derivatives,
                    np.column_stack((test_weights[start:stop], test_omega[start:stop])),
                )
                with torch.no_grad():
                    result = components(block, matrix_t)
                raw_values[start:stop] = result["raw"].detach().cpu().numpy()
                minimum_eigenvalues[start:stop] = (
                    torch.min(result["eigenvalues"], dim=1).values.detach().cpu().numpy()
                )
            return raw_values, minimum_eigenvalues

        blind_started = time.perf_counter()
        best_raw, best_minimum_eigenvalues = audit_matrix(best_h)
        fs_raw, fs_minimum_eigenvalues = audit_matrix(initial_h)
        blind_seconds = time.perf_counter() - blind_started
        best_statistics, best_ratio = ratio_statistics(
            best_raw, test_weights, best_minimum_eigenvalues
        )
        fs_statistics, fs_ratio = ratio_statistics(
            fs_raw, test_weights, fs_minimum_eigenvalues
        )

        cymetric_arrays = np.load(cymetric_tail_path, allow_pickle=False)
        stored_source_fs_ratio = np.asarray(
            cymetric_arrays["fs_normalized_ratio"][: len(test_x)], dtype=np.float64
        )
        stored_source_phi_ratio = np.asarray(
            cymetric_arrays["normalized_ratio"][: len(test_x)], dtype=np.float64
        )
        source_fs_determinant = np.asarray(
            cymetric_arrays["fs_determinant"][: len(test_x)], dtype=np.float64
        )
        source_fs_minimum_eigenvalue = np.asarray(
            cymetric_arrays["fs_min_eigenvalue"][: len(test_x)], dtype=np.float64
        )
        source_phi_determinant = np.asarray(
            cymetric_arrays["determinant"][: len(test_x)], dtype=np.float64
        )
        source_phi_minimum_eigenvalue = np.asarray(
            cymetric_arrays["min_eigenvalue"][: len(test_x)], dtype=np.float64
        )
        _, source_fs_ratio = ratio_statistics(
            np.log(source_fs_determinant) - np.log(test_omega),
            test_weights,
            source_fs_minimum_eigenvalue,
        )
        _, source_phi_ratio = ratio_statistics(
            np.log(source_phi_determinant) - np.log(test_omega),
            test_weights,
            source_phi_minimum_eigenvalue,
        )
        our_fs_determinant = np.exp(fs_raw + np.log(test_omega))
        fs_determinant_difference = np.abs(
            our_fs_determinant - source_fs_determinant
        )
        fs_determinant_relative_difference = fs_determinant_difference / np.maximum(
            np.abs(source_fs_determinant), np.finfo(np.float64).tiny
        )
        fs_ratio_difference = np.abs(fs_ratio - source_fs_ratio)
        fs_reproduction = {
            "determinant_maximum_absolute_difference": float(
                np.max(fs_determinant_difference)
            ),
            "determinant_median_relative_difference": float(
                np.median(fs_determinant_relative_difference)
            ),
            "determinant_q99_relative_difference": float(
                np.quantile(fs_determinant_relative_difference, 0.99)
            ),
            "determinant_maximum_relative_difference": float(
                np.max(fs_determinant_relative_difference)
            ),
            "normalized_ratio_median_absolute_difference": float(
                np.median(fs_ratio_difference)
            ),
            "normalized_ratio_q99_absolute_difference": float(
                np.quantile(fs_ratio_difference, 0.99)
            ),
            "normalized_ratio_maximum_absolute_difference": float(
                np.max(fs_ratio_difference)
            ),
            "source_stored_ratio_maximum_absolute_difference_after_common_subset_renormalization": float(
                np.max(np.abs(source_fs_ratio - stored_source_fs_ratio))
            ),
            "comparison_normalization_count": int(len(test_x)),
        }
        print(f"FS blind reproduction: {fs_reproduction}", flush=True)
        if float(np.max(fs_ratio_difference)) > 2.0e-5:
            raise RuntimeError(
                "our FS baseline does not reproduce the source cymetric blind-point "
                f"ratio within tolerance: {fs_reproduction}"
            )

        worst_indices = np.argsort(np.abs(1.0 - best_ratio))[-64:][::-1]
        np.savez_compressed(
            output_dir / "blind_test_tail_arrays.npz",
            weights=test_weights,
            omega_squared=test_omega,
            raw_log_ratio=best_raw,
            normalized_ratio=best_ratio,
            min_eigenvalue=best_minimum_eigenvalues,
            fs_normalized_ratio=fs_ratio,
            cymetric_phi_normalized_ratio=source_phi_ratio,
            cymetric_phi_stored_normalized_ratio=stored_source_phi_ratio,
        )
        np.savez_compressed(
            output_dir / "worst_points.npz",
            indices=worst_indices,
            X=test_x[worst_indices],
            weights=test_weights[worst_indices],
            normalized_ratio=best_ratio[worst_indices],
            min_eigenvalue=best_minimum_eigenvalues[worst_indices],
        )
        artifact_path = output_dir / "best_h_metric.npz"
        if not artifact_path.exists():
            np.savez_compressed(
                artifact_path,
                degree=np.asarray(args.degree),
                exponents=exponents,
                ambient_exponents=ambient_exponents,
                quotient_lift=quotient_lift,
                initial_h_matrix=initial_h,
                global_h_matrix=best_h,
                best_epoch=np.asarray(best_epoch),
                normalization=np.asarray(normalization),
            )

        source_report = json.loads((source_dir / "report.json").read_text())
        report = {
            "schema": "quintic-full-h-same-points-v1",
            "configuration": configuration,
            "scientific_scope": {
                "geometry": "Fermat quintic X_5 in P^4",
                "ansatz": "K=(1/k) log(s^dagger H s), full positive Hermitian H",
                "claim_limit": (
                    "The blind audit detects observed tails on a finite common point set; "
                    "it is not a global sup-norm certificate."
                ),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            "data": {
                "train_count": int(len(train_x)),
                "validation_count": int(len(validation_x)),
                "blind_test_count": int(len(test_x)),
                "dataset_sha256": sha256_file(dataset_path),
                "basis_sha256": sha256_file(basis_path),
                "blind_points_sha256": sha256_file(blind_points_path),
                "pullback_artifact_sha256": current_pullback_hashes,
                "pullback_precomputation_report_sha256": sha256_file(
                    pullback_report_path
                ),
                "source_cymetric_report_sha256": sha256_file(source_dir / "report.json"),
            },
            "basis": {
                "degree": args.degree,
                "ambient_monomial_count": int(len(ambient_exponents)),
                "section_count": int(len(exponents)),
                "real_hermitian_parameter_count": int(len(exponents) ** 2),
                "normalization": normalization,
                "quotient_convention": (
                    "standard monomials with exponent(z0)<5; "
                    "z0^5=-(z1^5+z2^5+z3^5+z4^5)"
                ),
                "fs_consistency": fs_check,
                "blind_fs_reproduction_from_cymetric": fs_reproduction,
            },
            "training": {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "initial_validation": history[0],
                "best_validation": min(history, key=lambda row: row["score"]),
                "history_rows": history,
            },
            "blind": {
                "our_full_h": best_statistics,
                "our_fubini_study_control": fs_statistics,
                "cymetric_phi_from_source_report": source_report["trained_phi_model"],
            },
            "timing_seconds": {
                "feature_preparation": feature_seconds,
                "optimization": optimization_seconds,
                "blind_audit_two_h_matrices": blind_seconds,
                "wall_total": time.perf_counter() - started,
            },
            "artifacts": {
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "h_metric_sha256": sha256_file(artifact_path),
            },
            "hardware": {
                "peak_cuda_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                )
            },
        }
        report_path = output_dir / "report.json"
        write_json(report_path, report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "best_epoch": best_epoch,
                "report_sha256": sha256_file(report_path),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps(json_value(report), indent=2, sort_keys=True, allow_nan=False))
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "exception",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
