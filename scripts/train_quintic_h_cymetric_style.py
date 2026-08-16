#!/usr/bin/env python3
"""Train the quintic algebraic H metric with cymetric-style mini-batch updates.

This is a mechanism experiment, not a new geometry benchmark.  It reuses the
exact Fermat-quintic points, pullbacks, holomorphic-volume labels, and fixed
KAPPA produced by the official cymetric control.  No symmetry projection or
symmetry-dependent parameterization is used.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_quintic_full_h_same_points import (  # noqa: E402
    cvar,
    fermat_quintic_quotient_basis,
    fermat_quotient_fubini_study_h,
    fs_consistency_check,
    positive_projection,
    prepare_features,
    ratio_statistics,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument(
        "--initial-h-file",
        type=Path,
        help="Warm-start from a registered best/last H npz; Adam state is reset.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--target-normalization",
        choices=("fixed_kappa", "batch_weighted_mean"),
        default="fixed_kappa",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=("uniform", "importance"),
        default="uniform",
    )
    parser.add_argument(
        "--importance-normalization",
        choices=("global_mean", "batch_mean"),
        default="global_mean",
        help=(
            "Normalize importance weights by the fixed training-set mean for an "
            "unbiased sampled energy, or by each batch mean for the legacy ratio estimator."
        ),
    )
    parser.add_argument(
        "--batch-reduction",
        choices=("sum", "mean"),
        default="sum",
        help="cymetric differentiates a per-point loss vector, equivalent to sum.",
    )
    parser.add_argument(
        "--point-loss",
        choices=("ma_l1", "energy_l2"),
        default="ma_l1",
        help="Use cymetric's absolute MA loss or the Headrick--Nassar squared energy.",
    )
    parser.add_argument("--barrier-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--torch-seed", type=int, default=202607161)
    parser.add_argument("--shuffle-seed", type=int, default=202607162)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-blind-audit", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "degree",
        "epochs",
        "batch_size",
        "eval_every",
        "feature_batch_size",
        "eval_batch_size",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.gradient_clip < 0:
        raise ValueError("gradient-clip cannot be negative")
    if args.barrier_weight < 0:
        raise ValueError("barrier-weight cannot be negative")
    if args.degree >= 10:
        raise ValueError("the registered Fermat quotient convention requires degree < 10")
    for name in ("train_limit", "validation_limit", "test_limit"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} cannot be negative")


def weighted_log_mean_exp(raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    positive_weights = torch.clamp(weights, min=torch.finfo(weights.dtype).tiny)
    return torch.logsumexp(torch.log(positive_weights) + raw, dim=0) - torch.log(
        torch.sum(positive_weights)
    )


def weighted_numpy_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def compact_ratio_statistics(
    ratio: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    ratio = np.asarray(ratio, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(ratio) & np.isfinite(weights) & (weights > 0)
    if not np.all(finite) or np.any(ratio <= 0):
        raise RuntimeError("ratio statistics received invalid values")
    residual = np.abs(1.0 - ratio)
    return {
        "sigma_unweighted": float(np.mean(residual)),
        "sigma_importance_weighted": weighted_numpy_mean(residual, weights),
        "l2_importance_weighted": math.sqrt(
            weighted_numpy_mean(np.square(residual), weights)
        ),
        "cvar_0.99_abs_residual_importance_weighted": cvar(
            residual, weights, 0.99
        ),
        "ratio_unweighted_quantiles": {
            label: float(value)
            for label, value in zip(
                ("minimum", "q0.001", "median", "q0.999", "maximum"),
                np.quantile(ratio, [0.0, 0.001, 0.5, 0.999, 1.0]),
                strict=True,
            )
        },
        "abs_residual_unweighted_quantiles": {
            label: float(value)
            for label, value in zip(
                ("median", "q0.99", "q0.999", "maximum"),
                np.quantile(residual, [0.5, 0.99, 0.999, 1.0]),
                strict=True,
            )
        },
        "count_ratio_above_1.5": int(np.count_nonzero(ratio > 1.5)),
        "count_ratio_below_0.5": int(np.count_nonzero(ratio < 0.5)),
    }


def normalized_ratio_from_raw(raw: np.ndarray, weights: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    maximum = float(np.max(raw))
    log_mean = maximum + math.log(
        float(np.sum(weights * np.exp(raw - maximum)) / np.sum(weights))
    )
    return np.exp(raw - log_mean)


def gradient_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return torch.zeros((), dtype=torch.float32)
    return torch.linalg.vector_norm(torch.stack(norms))


def quantile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, [0.0, 0.5, 0.9, 0.99, 1.0])
    return {
        label: float(value)
        for label, value in zip(
            ("minimum", "median", "q0.9", "q0.99", "maximum"),
            quantiles,
            strict=True,
        )
    }


def load_registered_initial_h(
    path: Path,
    *,
    degree: int,
    exponents: np.ndarray,
    quotient_lift: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        if "global_h_matrix" not in payload:
            raise ValueError("initial H artifact has no global_h_matrix")
        matrix = np.asarray(payload["global_h_matrix"], dtype=np.complex128)
        if "degree" in payload and int(np.asarray(payload["degree"]).item()) != degree:
            raise ValueError("initial H artifact uses another degree")
        if "exponents" in payload and not np.array_equal(payload["exponents"], exponents):
            raise ValueError("initial H artifact uses another quotient basis")
        if "quotient_lift" in payload and not np.allclose(
            payload["quotient_lift"], quotient_lift, rtol=0.0, atol=0.0
        ):
            raise ValueError("initial H artifact uses another quotient lift")
        source_epoch = None
        for key in ("best_epoch", "epoch"):
            if key in payload:
                source_epoch = int(np.asarray(payload[key]).item())
                break
    if matrix.shape != (len(exponents), len(exponents)):
        raise ValueError("initial H matrix has the wrong section dimension")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("initial H matrix is not finite")
    hermitian = 0.5 * (matrix + matrix.conjugate().T)
    relative_antihermitian = float(
        np.linalg.norm(matrix - matrix.conjugate().T)
        / max(np.linalg.norm(hermitian), np.finfo(float).tiny)
    )
    if relative_antihermitian > 1.0e-6:
        raise ValueError("initial H matrix is not Hermitian within tolerance")
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if float(np.min(eigenvalues)) <= 0:
        raise ValueError("initial H matrix is not positive definite")
    hermitian *= len(exponents) / float(np.real(np.trace(hermitian)))
    return hermitian, {
        "kind": "registered_H_warm_start_with_reset_Adam",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "source_epoch": source_epoch,
        "relative_antihermitian_norm": relative_antihermitian,
        "condition_number": float(np.max(eigenvalues) / np.min(eigenvalues)),
    }


def metric_components_from_cholesky(
    values: torch.Tensor,
    derivatives: torch.Tensor,
    log_omega: torch.Tensor,
    cholesky: torch.Tensor,
    normalization: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the algebraic metric without materializing H=L L^dagger."""
    transformed_values = values @ torch.conj(cholesky)
    transformed_derivatives = torch.einsum(
        "mc,nmj->ncj", torch.conj(cholesky), derivatives
    )
    denominator = torch.sum(
        torch.real(torch.conj(transformed_values) * transformed_values), dim=1
    )
    first = torch.einsum(
        "nci,ncj->nji",
        torch.conj(transformed_derivatives),
        transformed_derivatives,
    )
    gradient = torch.einsum(
        "nc,ncj->nj", torch.conj(transformed_values), transformed_derivatives
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
    raw = raw - log_omega
    eigenvalue_scale = torch.clamp(
        torch.mean(torch.abs(eigenvalues), dim=1, keepdim=True), min=1.0e-14
    )
    relative_eigenvalues = eigenvalues / eigenvalue_scale
    barrier = torch_functional.softplus(
        (1.0e-8 - relative_eigenvalues) * 80.0
    ).mean() / 80.0
    return raw, eigenvalues, barrier


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = (
        args.pullbacks_dir.expanduser().resolve()
        if args.pullbacks_dir is not None
        else source_dir / "full_h_common_geometry"
    )
    probe_path = (
        args.probe_file.expanduser().resolve()
        if args.probe_file is not None
        else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    configuration = vars(args).copy()
    for key in (
        "source_run_dir",
        "pullbacks_dir",
        "probe_file",
        "initial_h_file",
        "output_dir",
    ):
        value = configuration[key]
        configuration[key] = str(value) if value is not None else None
    write_json(
        status_path,
        {"state": "running", "phase": "initializing", "configuration": configuration},
    )

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
        source_report_path = source_dir / "report.json"
        pullback_report_path = pullbacks_dir / "report.json"
        train_pullbacks_path = pullbacks_dir / "train_pullbacks.npy"
        validation_pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
        blind_pullbacks_path = pullbacks_dir / "blind_pullbacks.npy"
        official_fs_metrics_path = (
            pullbacks_dir / "validation_official_fs_metrics.npy"
        )
        required_paths = (
            dataset_path,
            basis_path,
            blind_points_path,
            cymetric_tail_path,
            source_report_path,
            pullback_report_path,
            train_pullbacks_path,
            validation_pullbacks_path,
            blind_pullbacks_path,
            official_fs_metrics_path,
        )
        if probe_path is not None:
            required_paths += (probe_path,)
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(path)

        pullback_report = json.loads(pullback_report_path.read_text(encoding="utf-8"))
        source_hashes = {
            "dataset": sha256_file(dataset_path),
            "basis": sha256_file(basis_path),
            "blind_points": sha256_file(blind_points_path),
            "cymetric_tail": sha256_file(cymetric_tail_path),
        }
        registered_source_hashes = pullback_report.get("source_sha256", {})
        geometry_source_keys = ("dataset", "basis", "blind_points")
        if any(
            registered_source_hashes.get(key) != source_hashes[key]
            for key in geometry_source_keys
        ):
            raise RuntimeError("pullbacks do not match the source geometry artifacts")
        cymetric_tail_hash_matches_pullback_registration = (
            registered_source_hashes.get("cymetric_tail")
            == source_hashes["cymetric_tail"]
        )
        pullback_paths = {
            "train": train_pullbacks_path,
            "validation": validation_pullbacks_path,
            "blind": blind_pullbacks_path,
            "validation_official_fs_metrics": official_fs_metrics_path,
        }
        pullback_hashes = {
            name: sha256_file(path) for name, path in pullback_paths.items()
        }
        if pullback_report.get("output_sha256") != pullback_hashes:
            raise RuntimeError("pullback artifact hashes do not match their report")

        with basis_path.open("rb") as stream:
            raw_basis = pickle.load(stream)  # noqa: S301 - trusted generated artifact
        fixed_kappa = float(np.real(np.asarray(raw_basis["KAPPA"])).item())
        if not np.isfinite(fixed_kappa) or fixed_kappa <= 0:
            raise RuntimeError(f"invalid fixed KAPPA in basis: {fixed_kappa}")
        log_fixed_kappa = math.log(fixed_kappa)

        with np.load(dataset_path, allow_pickle=False) as data:
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
            train_x = train_x[: args.train_limit]
            train_y = train_y[: args.train_limit]
            train_pullbacks = train_pullbacks[: args.train_limit]
        if args.validation_limit:
            validation_x = validation_x[: args.validation_limit]
            validation_y = validation_y[: args.validation_limit]
            validation_pullbacks = validation_pullbacks[: args.validation_limit]
        if len(train_x) < args.batch_size:
            raise ValueError("batch-size exceeds the retained training-point count")

        if train_x.shape[1] // 2 != 5:
            raise ValueError("the Fermat quotient basis requires five coordinates")
        ambient_exponents, exponents, quotient_lift = fermat_quintic_quotient_basis(
            args.degree
        )
        fubini_study_h = fermat_quotient_fubini_study_h(
            ambient_exponents, quotient_lift
        )
        if args.initial_h_file is None:
            initial_h = fubini_study_h
            initial_h_source = {
                "kind": "fermat_quotient_fubini_study",
                "path": None,
                "sha256": None,
                "source_epoch": None,
            }
        else:
            initial_h, initial_h_source = load_registered_initial_h(
                args.initial_h_file,
                degree=args.degree,
                exponents=exponents,
                quotient_lift=quotient_lift,
            )
        normalization = 1.0 / (math.pi * args.degree)
        official_fs_metrics = np.load(official_fs_metrics_path, mmap_mode="r")
        fs_check = fs_consistency_check(
            validation_x,
            validation_pullbacks,
            official_fs_metrics,
            exponents,
            fubini_study_h,
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

        probe_metadata: dict[str, Any] | None = None
        probe_values: np.ndarray | None = None
        probe_derivatives: np.ndarray | None = None
        probe_y: np.ndarray | None = None
        if probe_path is not None:
            with np.load(probe_path, allow_pickle=False) as probes:
                probe_indices = np.asarray(probes["blind_indices"], dtype=np.int64)
                center_indices = np.asarray(probes["center_indices"], dtype=np.int64)
                center_labels = np.asarray(probes["center_source_labels"], dtype=str)
                stored_blind_hash = str(probes["source_blind_points_sha256"].item())
                stored_dataset_hash = str(
                    probes["source_training_dataset_sha256"].item()
                )
            if stored_blind_hash != source_hashes["blind_points"]:
                raise RuntimeError("probe file does not match the common blind points")
            if stored_dataset_hash != source_hashes["dataset"]:
                raise RuntimeError("probe file does not match the training dataset")
            with np.load(blind_points_path, allow_pickle=False) as blind:
                all_probe_x = np.asarray(blind["X"][probe_indices], dtype=np.float32)
                all_probe_weights = np.asarray(
                    blind["weights"][probe_indices], dtype=np.float64
                )
                all_probe_omega = np.asarray(
                    blind["omega_squared"][probe_indices], dtype=np.float64
                )
            blind_pullbacks = np.load(blind_pullbacks_path, mmap_mode="r")
            probe_values, probe_derivatives = prepare_features(
                all_probe_x,
                blind_pullbacks[probe_indices],
                exponents,
                args.feature_batch_size,
            )
            probe_y = np.column_stack((all_probe_weights, all_probe_omega))
            global_to_local = {
                int(global_index): local_index
                for local_index, global_index in enumerate(probe_indices)
            }
            center_local = np.asarray(
                [global_to_local[int(index)] for index in center_indices],
                dtype=np.int64,
            )
            center_groups = {"all_selected_centers": center_local}
            for source_name in ("raw_full_h", "cymetric_phi"):
                selected = [
                    center_local[index]
                    for index, label in enumerate(center_labels)
                    if source_name in label.split("+")
                ]
                center_groups[source_name] = np.asarray(selected, dtype=np.int64)
            probe_metadata = {
                "probe_count": int(len(probe_indices)),
                "center_count": int(len(center_indices)),
                "center_groups": center_groups,
            }
        feature_seconds = time.perf_counter() - feature_started

        complex_dtype = torch.complex64
        real_dtype = torch.float32

        def tensor_dataset(
            values: np.ndarray,
            derivatives: np.ndarray,
            labels: np.ndarray,
        ) -> dict[str, torch.Tensor]:
            return {
                "values": torch.as_tensor(values, dtype=complex_dtype, device=device),
                "derivatives": torch.as_tensor(
                    derivatives, dtype=complex_dtype, device=device
                ),
                "weights": torch.as_tensor(
                    labels[:, 0], dtype=real_dtype, device=device
                ),
                "log_omega": torch.log(
                    torch.as_tensor(labels[:, 1], dtype=real_dtype, device=device)
                ),
            }

        training = tensor_dataset(train_values, train_derivatives, train_y)
        validation = tensor_dataset(
            validation_values, validation_derivatives, validation_y
        )
        probe = (
            tensor_dataset(probe_values, probe_derivatives, probe_y)
            if probe_values is not None
            and probe_derivatives is not None
            and probe_y is not None
            else None
        )
        del train_values, train_derivatives, validation_values, validation_derivatives
        del probe_values, probe_derivatives, probe_y
        global_training_weight_mean = torch.mean(training["weights"]).detach()

        initial_cholesky = np.linalg.cholesky(initial_h)
        diagonal_log = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    np.diag(initial_cholesky).real.copy(),
                    dtype=real_dtype,
                    device=device,
                )
            )
        )
        lower_rows, lower_columns = torch.tril_indices(
            len(initial_h), len(initial_h), offset=-1, device=device
        )
        lower_real = torch.nn.Parameter(
            torch.tensor(
                initial_cholesky.real[
                    lower_rows.cpu().numpy(), lower_columns.cpu().numpy()
                ],
                dtype=real_dtype,
                device=device,
            )
        )
        lower_imag = torch.nn.Parameter(
            torch.tensor(
                initial_cholesky.imag[
                    lower_rows.cpu().numpy(), lower_columns.cpu().numpy()
                ],
                dtype=real_dtype,
                device=device,
            )
        )
        parameters = [diagonal_log, lower_real, lower_imag]
        registered_real_parameter_count = sum(
            parameter.numel() for parameter in parameters
        )
        if registered_real_parameter_count != len(initial_h) ** 2:
            raise RuntimeError("packed Cholesky parameter count is inconsistent")

        def current_cholesky() -> torch.Tensor:
            cholesky = torch.diag(torch.exp(diagonal_log)).to(complex_dtype)
            cholesky = cholesky.index_put(
                (lower_rows, lower_columns),
                torch.complex(lower_real, lower_imag),
            )
            trace = torch.sum(torch.real(torch.conj(cholesky) * cholesky))
            return cholesky * torch.sqrt(len(initial_h) / trace)

        def current_h() -> torch.Tensor:
            cholesky = current_cholesky()
            return cholesky @ torch.conj(cholesky.T)

        def raw_components(
            dataset: dict[str, torch.Tensor], cholesky: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return metric_components_from_cholesky(
                dataset["values"],
                dataset["derivatives"],
                dataset["log_omega"],
                cholesky,
                normalization,
            )

        def training_loss(
            dataset: dict[str, torch.Tensor],
            cholesky: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            raw, _, barrier = raw_components(dataset, cholesky)
            if args.target_normalization == "fixed_kappa":
                centered = raw - log_fixed_kappa
            else:
                centered = raw - weighted_log_mean_exp(raw, dataset["weights"])
            ratio = torch.exp(centered)
            residual = 1.0 - ratio
            per_point = (
                torch.abs(residual)
                if args.point_loss == "ma_l1"
                else torch.square(residual)
            )
            if args.loss_weighting == "importance":
                denominator = (
                    global_training_weight_mean
                    if args.importance_normalization == "global_mean"
                    else torch.mean(dataset["weights"])
                )
                normalized_weights = dataset["weights"] / denominator
                per_point = per_point * normalized_weights
            loss = (
                torch.sum(per_point)
                if args.batch_reduction == "sum"
                else torch.mean(per_point)
            )
            loss = loss + args.barrier_weight * barrier
            return loss, torch.mean(torch.abs(1.0 - ratio))

        def sliced_dataset(
            dataset: dict[str, torch.Tensor], indices: torch.Tensor | slice
        ) -> dict[str, torch.Tensor]:
            return {name: value[indices] for name, value in dataset.items()}

        def evaluate_raw(
            dataset: dict[str, torch.Tensor], matrix: torch.Tensor
        ) -> tuple[np.ndarray, np.ndarray]:
            count = len(dataset["values"])
            raw_blocks = []
            minimum_blocks = []
            cholesky = torch.linalg.cholesky(matrix)
            with torch.no_grad():
                for start in range(0, count, args.eval_batch_size):
                    stop = min(start + args.eval_batch_size, count)
                    raw, eigenvalues, _ = raw_components(
                        sliced_dataset(dataset, slice(start, stop)), cholesky
                    )
                    raw_blocks.append(raw.detach().cpu().numpy().astype(np.float64))
                    minimum_blocks.append(
                        torch.min(eigenvalues, dim=1)
                        .values.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
            return np.concatenate(raw_blocks), np.concatenate(minimum_blocks)

        def matrix_condition(matrix: np.ndarray) -> dict[str, float]:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return {
                "minimum_eigenvalue": float(np.min(eigenvalues)),
                "maximum_eigenvalue": float(np.max(eigenvalues)),
                "condition_number": float(np.max(eigenvalues) / np.min(eigenvalues)),
            }

        def evaluation_summary(
            dataset: dict[str, torch.Tensor], matrix: np.ndarray
        ) -> dict[str, Any]:
            matrix_t = torch.as_tensor(matrix, dtype=complex_dtype, device=device)
            raw, minimum_eigenvalues = evaluate_raw(dataset, matrix_t)
            weights = dataset["weights"].detach().cpu().numpy().astype(np.float64)
            fixed_ratio = np.exp(raw - log_fixed_kappa)
            normalized_ratio = normalized_ratio_from_raw(raw, weights)
            return {
                "fixed_kappa": compact_ratio_statistics(fixed_ratio, weights),
                "audit_normalized": compact_ratio_statistics(
                    normalized_ratio, weights
                ),
                "minimum_metric_eigenvalue": float(np.min(minimum_eigenvalues)),
                "h_matrix": matrix_condition(matrix),
                "_raw": raw,
                "_fixed_ratio": fixed_ratio,
            }

        def public_summary(summary: dict[str, Any]) -> dict[str, Any]:
            return {key: value for key, value in summary.items() if not key.startswith("_")}

        def probe_summary(matrix: np.ndarray) -> dict[str, Any] | None:
            if probe is None or probe_metadata is None:
                return None
            result = evaluation_summary(probe, matrix)
            fixed_ratio = result["_fixed_ratio"]
            groups = {}
            for name, local_indices in probe_metadata["center_groups"].items():
                selected_ratio = fixed_ratio[local_indices]
                groups[name] = {
                    "count": int(len(selected_ratio)),
                    "fixed_ratio": compact_ratio_statistics(
                        selected_ratio,
                        np.ones(len(selected_ratio), dtype=np.float64),
                    ),
                }
            public = public_summary(result)
            public["selected_center_groups"] = groups
            public["warning"] = (
                "These probes were selected from earlier model tails and are diagnostic; "
                "they are never used for optimization or checkpoint selection."
            )
            return public

        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        shuffle_rng = np.random.default_rng(args.shuffle_seed)
        initial_validation = evaluation_summary(validation, initial_h)
        initial_audit = initial_validation["audit_normalized"]
        initial_score = (
            initial_audit["sigma_importance_weighted"]
            + 0.25 * initial_audit["l2_importance_weighted"]
            + 0.05
            * initial_audit["cvar_0.99_abs_residual_importance_weighted"]
        )
        history = [
            {
                "epoch": 0,
                "updates": 0,
                "validation": public_summary(initial_validation),
                "diagnostic_probe": probe_summary(initial_h),
                "selection_score": initial_score,
            }
        ]
        best_h = initial_h.copy()
        best_epoch = 0
        best_score = initial_score
        total_updates = 0
        np.savez_compressed(
            output_dir / "best_h_metric.npz",
            degree=np.asarray(args.degree),
            ambient_exponents=ambient_exponents,
            exponents=exponents,
            quotient_lift=quotient_lift,
            initial_h_matrix=initial_h,
            global_h_matrix=best_h,
            best_epoch=np.asarray(best_epoch),
            normalization=np.asarray(normalization),
            fixed_kappa=np.asarray(fixed_kappa),
        )
        write_json(output_dir / "training_history.json", {"rows": history})
        print(
            f"sections={len(exponents)} train={len(train_x)} batch={args.batch_size} "
            f"KAPPA={fixed_kappa:.9e} initial_sigma="
            f"{initial_audit['sigma_importance_weighted']:.6e}",
            flush=True,
        )

        write_json(status_path, {"state": "running", "phase": "training", "epoch": 0})
        optimization_started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            permutation = shuffle_rng.permutation(len(train_x))
            epoch_loss_per_point = []
            epoch_sigma = []
            epoch_gradient_norm = []
            for start in range(0, len(permutation), args.batch_size):
                batch_indices_np = permutation[start : start + args.batch_size]
                batch_indices = torch.as_tensor(
                    batch_indices_np, dtype=torch.long, device=device
                )
                batch = sliced_dataset(training, batch_indices)
                optimizer.zero_grad(set_to_none=True)
                loss, unweighted_sigma = training_loss(batch, current_cholesky())
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"nonfinite training loss at epoch {epoch}, batch start {start}"
                    )
                loss.backward()
                if args.gradient_clip > 0:
                    raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        parameters,
                        max_norm=args.gradient_clip,
                        error_if_nonfinite=True,
                    )
                else:
                    raw_gradient_norm = gradient_norm(parameters)
                    if not torch.isfinite(raw_gradient_norm):
                        raise RuntimeError("nonfinite gradient norm")
                optimizer.step()
                total_updates += 1
                epoch_loss_per_point.append(float(loss.detach().cpu()) / len(batch_indices))
                epoch_sigma.append(float(unweighted_sigma.detach().cpu()))
                epoch_gradient_norm.append(float(raw_gradient_norm.detach().cpu()))

            if epoch % args.eval_every != 0 and epoch != args.epochs:
                continue
            candidate = positive_projection(current_h().detach().cpu().numpy())
            validation_result = evaluation_summary(validation, candidate)
            validation_audit = validation_result["audit_normalized"]
            selection_score = (
                validation_audit["sigma_importance_weighted"]
                + 0.25 * validation_audit["l2_importance_weighted"]
                + 0.05
                * validation_audit[
                    "cvar_0.99_abs_residual_importance_weighted"
                ]
            )
            accepted = selection_score < best_score
            if accepted:
                best_h = candidate
                best_score = selection_score
                best_epoch = epoch
                np.savez_compressed(
                    output_dir / "best_h_metric.npz",
                    degree=np.asarray(args.degree),
                    ambient_exponents=ambient_exponents,
                    exponents=exponents,
                    quotient_lift=quotient_lift,
                    initial_h_matrix=initial_h,
                    global_h_matrix=best_h,
                    best_epoch=np.asarray(best_epoch),
                    normalization=np.asarray(normalization),
                    fixed_kappa=np.asarray(fixed_kappa),
                )
            row = {
                "epoch": epoch,
                "updates": total_updates,
                "training": {
                    "loss_per_point": quantile_summary(epoch_loss_per_point),
                    "unweighted_batch_sigma": quantile_summary(epoch_sigma),
                    "preclip_gradient_norm": quantile_summary(epoch_gradient_norm),
                    "fraction_gradient_norm_above_clip": float(
                        np.mean(
                            np.asarray(epoch_gradient_norm) > args.gradient_clip
                        )
                        if args.gradient_clip > 0
                        else 0.0
                    ),
                },
                "validation": public_summary(validation_result),
                "diagnostic_probe": probe_summary(candidate),
                "selection_score": selection_score,
                "accepted_as_best": accepted,
            }
            history.append(row)
            write_json(output_dir / "training_history.json", {"rows": history})
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "training",
                    "epoch": epoch,
                    "updates": total_updates,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "latest": row,
                },
            )
            ratio_quantiles = validation_result["audit_normalized"][
                "ratio_unweighted_quantiles"
            ]
            print(
                f"epoch={epoch} updates={total_updates} "
                f"sigma={validation_audit['sigma_importance_weighted']:.6e} "
                f"ratio=[{ratio_quantiles['minimum']:.4f},"
                f"{ratio_quantiles['maximum']:.4f}] condH="
                f"{validation_result['h_matrix']['condition_number']:.3e} "
                f"best={accepted}",
                flush=True,
            )
        optimization_seconds = time.perf_counter() - optimization_started
        last_h = positive_projection(current_h().detach().cpu().numpy())
        np.savez_compressed(
            output_dir / "last_h_metric.npz",
            degree=np.asarray(args.degree),
            ambient_exponents=ambient_exponents,
            exponents=exponents,
            quotient_lift=quotient_lift,
            initial_h_matrix=initial_h,
            global_h_matrix=last_h,
            epoch=np.asarray(args.epochs),
            normalization=np.asarray(normalization),
            fixed_kappa=np.asarray(fixed_kappa),
        )

        blind_report = None
        blind_seconds = 0.0
        if not args.skip_blind_audit:
            write_json(status_path, {"state": "running", "phase": "blind_audit"})
            with np.load(blind_points_path, allow_pickle=False) as blind:
                test_x = np.asarray(blind["X"], dtype=np.float32)
                test_weights = np.asarray(blind["weights"], dtype=np.float64)
                test_omega = np.asarray(blind["omega_squared"], dtype=np.float64)
            test_pullbacks = np.load(blind_pullbacks_path, mmap_mode="r")
            if args.test_limit:
                test_x = test_x[: args.test_limit]
                test_weights = test_weights[: args.test_limit]
                test_omega = test_omega[: args.test_limit]
                test_pullbacks = test_pullbacks[: args.test_limit]

            def audit_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                raw_values = np.empty(len(test_x), dtype=np.float64)
                minimum_values = np.empty(len(test_x), dtype=np.float64)
                matrix_t = torch.as_tensor(matrix, dtype=complex_dtype, device=device)
                cholesky_t = torch.linalg.cholesky(matrix_t)
                for start in range(0, len(test_x), args.eval_batch_size):
                    stop = min(start + args.eval_batch_size, len(test_x))
                    block_values, block_derivatives = prepare_features(
                        test_x[start:stop],
                        test_pullbacks[start:stop],
                        exponents,
                        args.feature_batch_size,
                    )
                    block = tensor_dataset(
                        block_values,
                        block_derivatives,
                        np.column_stack(
                            (test_weights[start:stop], test_omega[start:stop])
                        ),
                    )
                    with torch.no_grad():
                        raw, eigenvalues, _ = raw_components(block, cholesky_t)
                    raw_values[start:stop] = raw.detach().cpu().numpy()
                    minimum_values[start:stop] = (
                        torch.min(eigenvalues, dim=1).values.detach().cpu().numpy()
                    )
                return raw_values, minimum_values

            blind_started = time.perf_counter()
            best_raw, best_minimum = audit_matrix(best_h)
            blind_seconds = time.perf_counter() - blind_started
            best_statistics, best_ratio = ratio_statistics(
                best_raw, test_weights, best_minimum
            )
            fixed_ratio = np.exp(best_raw - log_fixed_kappa)
            cymetric_arrays = np.load(cymetric_tail_path, allow_pickle=False)
            cymetric_ratio = np.asarray(
                cymetric_arrays["normalized_ratio"][: len(test_x)], dtype=np.float64
            )
            np.savez_compressed(
                output_dir / "blind_test_tail_arrays.npz",
                weights=test_weights,
                omega_squared=test_omega,
                raw_log_ratio=best_raw,
                fixed_kappa_ratio=fixed_ratio,
                normalized_ratio=best_ratio,
                minimum_metric_eigenvalue=best_minimum,
                cymetric_phi_normalized_ratio=cymetric_ratio,
            )
            worst_indices = np.argsort(np.abs(1.0 - best_ratio))[-64:][::-1]
            np.savez_compressed(
                output_dir / "worst_points.npz",
                indices=worst_indices,
                X=test_x[worst_indices],
                weights=test_weights[worst_indices],
                fixed_kappa_ratio=fixed_ratio[worst_indices],
                normalized_ratio=best_ratio[worst_indices],
                minimum_metric_eigenvalue=best_minimum[worst_indices],
            )
            blind_report = {
                "point_count": int(len(test_x)),
                "selected_best_epoch": best_epoch,
                "audit_normalized": best_statistics,
                "fixed_kappa": compact_ratio_statistics(fixed_ratio, test_weights),
                "cymetric_phi_common_points": compact_ratio_statistics(
                    cymetric_ratio, test_weights
                ),
            }

        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        report = {
            "schema": "quintic-h-cymetric-style-mechanism-v1",
            "configuration": configuration,
            "scientific_scope": {
                "geometry": "Fermat quintic X_5 in P^4",
                "ansatz": "K=(1/k) log(s^dagger H s), full positive Hermitian H",
                "symmetry_usage": "none",
                "purpose": (
                    "Isolate optimization-schedule differences from geometry and "
                    "model-capacity differences on exact common points."
                ),
                "loss_correspondence": (
                    "ma_l1 reproduces cymetric's pointwise absolute MA objective; "
                    "energy_l2 is the sampled Headrick--Nassar squared MA energy. "
                    "H is already globally projective and Kahler, so no transition "
                    "or Kahler penalty is needed."
                ),
                "probe_warning": (
                    "The common tail probes are diagnostic and excluded from training "
                    "and checkpoint selection."
                ),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
            },
            "data": {
                "train_count": int(len(train_x)),
                "validation_count": int(len(validation_x)),
                "fixed_kappa": fixed_kappa,
                "global_training_weight_mean": float(
                    global_training_weight_mean.detach().cpu()
                ),
                "source_sha256": source_hashes,
                "pullback_registered_source_sha256": registered_source_hashes,
                "pullback_geometry_source_keys_verified": list(
                    geometry_source_keys
                ),
                "cymetric_tail_hash_matches_pullback_registration": (
                    cymetric_tail_hash_matches_pullback_registration
                ),
                "pullback_sha256": pullback_hashes,
                "probe_file": str(probe_path) if probe_path is not None else None,
                "probe_sha256": sha256_file(probe_path)
                if probe_path is not None
                else None,
            },
            "basis": {
                "degree": args.degree,
                "ambient_monomial_count": int(len(ambient_exponents)),
                "section_count": int(len(exponents)),
                "real_hermitian_parameter_count": int(len(exponents) ** 2),
                "registered_real_parameter_count": (
                    registered_real_parameter_count
                ),
                "quotient_convention": (
                    "standard monomials exponent(z0)<5 with the Fermat relation"
                ),
                "normalization": normalization,
                "fs_consistency": fs_check,
                "initial_h_source": initial_h_source,
            },
            "training": {
                "total_updates": total_updates,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "history": history,
            },
            "blind": blind_report,
            "official_cymetric_reference": source_report["trained_phi_model"],
            "timing_seconds": {
                "feature_preparation": feature_seconds,
                "optimization": optimization_seconds,
                "blind_audit": blind_seconds,
                "wall_total": time.perf_counter() - started,
            },
            "hardware": {
                "peak_cuda_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                )
            },
            "artifacts": {
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "best_h_sha256": sha256_file(output_dir / "best_h_metric.npz"),
                "last_h_sha256": sha256_file(output_dir / "last_h_metric.npz"),
            },
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "best_epoch": best_epoch,
                "best_score": best_score,
                "total_updates": total_updates,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(f"complete: {output_dir}", flush=True)
    except Exception as exc:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
