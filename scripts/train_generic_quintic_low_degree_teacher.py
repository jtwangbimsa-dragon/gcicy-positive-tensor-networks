#!/usr/bin/env python3
"""Train the fixed low-degree full-H initializer on the generic quintic."""

from __future__ import annotations

import argparse
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

from gcicy_metric.fermat_quintic import (  # noqa: E402
    exponent_compositions,
    fubini_study_h,
)
from gcicy_metric.generic_quintic import experiment_manifest  # noqa: E402
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    prepare_features,
    ratio_statistics,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--sigma-weight", type=float, default=0.05)
    parser.add_argument("--condition-weight", type=float, default=1.0e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--feature-batch-size", type=int, default=2048)
    parser.add_argument("--metric-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=0,
        help="random native-E2 points per optimizer step; zero uses all points",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=1,
        help="optimizer steps between validation epochs",
    )
    parser.add_argument("--seed", type=int, default=202607267)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--precision",
        choices=("complex64", "complex128"),
        default="complex64",
    )
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


class PositiveFullH(torch.nn.Module):
    """Cholesky coordinates for one dense positive Hermitian matrix."""

    def __init__(
        self,
        initial_h: np.ndarray,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        initial = np.asarray(initial_h, dtype=np.complex128)
        cholesky = np.linalg.cholesky(initial)
        real_dtype = (
            torch.float32 if dtype == torch.complex64 else torch.float64
        )
        self.diagonal_log = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    np.diag(cholesky).real,
                    dtype=real_dtype,
                    device=device,
                )
            )
        )
        self.lower_real = torch.nn.Parameter(
            torch.tensor(cholesky.real, dtype=real_dtype, device=device)
        )
        self.lower_imag = torch.nn.Parameter(
            torch.tensor(cholesky.imag, dtype=real_dtype, device=device)
        )
        self.register_buffer(
            "lower_mask",
            torch.tril(
                torch.ones(
                    initial.shape,
                    dtype=real_dtype,
                    device=device,
                ),
                diagonal=-1,
            ),
        )
        self.complex_dtype = dtype

    def forward(self) -> torch.Tensor:
        lower = self.lower_mask * self.lower_real
        lower = lower + 1j * self.lower_mask * self.lower_imag
        factor = lower.to(self.complex_dtype)
        factor = factor + torch.diag(torch.exp(self.diagonal_log)).to(
            self.complex_dtype
        )
        matrix = factor @ torch.conj(factor.T)
        return matrix * (len(matrix) / torch.real(torch.trace(matrix)))


def make_dataset(
    x_values: np.ndarray,
    labels: np.ndarray,
    pullbacks: np.ndarray,
    exponents: np.ndarray,
    *,
    feature_batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    values, derivatives = prepare_features(
        x_values,
        pullbacks,
        exponents,
        feature_batch_size,
        complex_dtype=(
            np.dtype(np.complex64)
            if dtype == torch.complex64
            else np.dtype(np.complex128)
        ),
    )
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    weights_numpy = np.asarray(labels[:, 0], dtype=np.float64)
    weights_numpy /= np.sum(weights_numpy)
    omega_squared = np.asarray(labels[:, 1], dtype=np.float64)
    return {
        "count": len(values),
        "values": torch.tensor(values, dtype=dtype, device=device),
        "derivatives": torch.tensor(derivatives, dtype=dtype, device=device),
        "weights": torch.tensor(
            weights_numpy,
            dtype=real_dtype,
            device=device,
        ),
        "weights_numpy": weights_numpy,
        "log_omega": torch.log(
            torch.tensor(
                omega_squared,
                dtype=real_dtype,
                device=device,
            )
        ),
    }


def dataset_subset(
    dataset: dict[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    selected = torch.as_tensor(
        indices,
        dtype=torch.int64,
        device=dataset["values"].device,
    )
    return {
        "count": len(indices),
        "values": dataset["values"].index_select(0, selected),
        "derivatives": dataset["derivatives"].index_select(0, selected),
        "weights": dataset["weights"].index_select(0, selected),
        "weights_numpy": dataset["weights_numpy"][indices],
        "log_omega": dataset["log_omega"].index_select(0, selected),
    }


def raw_log_volume(
    matrix: torch.Tensor,
    dataset: dict[str, Any],
    *,
    normalization: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_rows = []
    eigenvalue_rows = []
    for start in range(0, dataset["count"], chunk_size):
        stop = min(start + chunk_size, dataset["count"])
        values = dataset["values"][start:stop]
        derivatives = dataset["derivatives"][start:stop]
        h_values = torch.einsum("ab,nb->na", matrix, values)
        denominator = torch.real(
            torch.einsum("na,na->n", torch.conj(values), h_values)
        )
        h_derivatives = torch.einsum(
            "ab,nbj->naj",
            matrix,
            derivatives,
        )
        first = torch.einsum(
            "nmi,nmj->nji",
            torch.conj(derivatives),
            h_derivatives,
        )
        gradient = torch.einsum(
            "nm,nmj->nj",
            torch.conj(values),
            h_derivatives,
        )
        metric = first / denominator[:, None, None]
        metric = metric - (
            gradient[:, :, None]
            * torch.conj(gradient)[:, None, :]
            / torch.square(denominator[:, None, None])
        )
        metric = normalization * 0.5 * (
            metric + torch.conj(torch.transpose(metric, 1, 2))
        )
        eigenvalues = torch.linalg.eigvalsh(metric)
        if not bool(torch.all(torch.isfinite(eigenvalues))):
            raise FloatingPointError("teacher metric has nonfinite eigenvalues")
        if not bool(torch.all(eigenvalues > 0)):
            raise FloatingPointError("teacher metric is not positive")
        raw_rows.append(
            torch.sum(torch.log(eigenvalues), dim=1)
            - dataset["log_omega"][start:stop]
        )
        eigenvalue_rows.append(eigenvalues)
    return torch.cat(raw_rows), torch.cat(eigenvalue_rows)


def native_components(
    matrix: torch.Tensor,
    dataset: dict[str, Any],
    *,
    normalization: float,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    raw, eigenvalues = raw_log_volume(
        matrix,
        dataset,
        normalization=normalization,
        chunk_size=chunk_size,
    )
    weights = dataset["weights"] / torch.sum(dataset["weights"])
    log_mean = torch.logsumexp(torch.log(weights) + raw, dim=0)
    ratio = torch.exp(raw - log_mean)
    residual = ratio - 1.0
    return {
        "raw": raw,
        "ratio": ratio,
        "eigenvalues": eigenvalues,
        "e2": torch.sum(weights * torch.square(residual)),
        "sigma": torch.sum(weights * torch.abs(residual)),
    }


def evaluate(
    matrix: np.ndarray,
    dataset: dict[str, Any],
    *,
    normalization: float,
    chunk_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        components = native_components(
            torch.tensor(matrix, dtype=dtype, device=device),
            dataset,
            normalization=normalization,
            chunk_size=chunk_size,
        )
    raw = components["raw"].cpu().numpy().astype(np.float64)
    minimum = (
        torch.min(components["eigenvalues"], dim=1)
        .values.cpu().numpy().astype(np.float64)
    )
    statistics, _ = ratio_statistics(
        raw,
        dataset["weights_numpy"],
        minimum,
    )
    return {
        "statistics": statistics,
        "tail": tail_summary(statistics),
    }


def main() -> None:
    args = parse_args()
    if (
        args.degree <= 0
        or args.degree >= 5
        or args.epochs <= 0
        or args.learning_rate <= 0
        or args.minimum_learning_rate <= 0
        or args.eval_every <= 0
        or args.feature_batch_size <= 0
        or args.metric_chunk_size <= 0
        or args.training_batch_size < 0
        or args.steps_per_epoch <= 0
        or args.threads <= 0
    ):
        raise ValueError("the low-degree teacher configuration is invalid")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a teacher run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    try:
        dataset_path = data_dir / "teacher_training_data" / "dataset.npz"
        train_pullbacks_path = data_dir / "teacher_train" / "pullbacks.npy"
        validation_pullbacks_path = (
            data_dir / "teacher_validation" / "pullbacks.npy"
        )
        for path in (
            dataset_path,
            train_pullbacks_path,
            validation_pullbacks_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        raw = np.load(dataset_path, allow_pickle=False)
        train_x = np.asarray(raw["X_train"], dtype=np.float32)
        train_y = np.asarray(raw["y_train"], dtype=np.float64)
        validation_x = np.asarray(raw["X_val"], dtype=np.float32)
        validation_y = np.asarray(raw["y_val"], dtype=np.float64)
        train_pullbacks = np.load(train_pullbacks_path, mmap_mode="r")
        validation_pullbacks = np.load(
            validation_pullbacks_path,
            mmap_mode="r",
        )
        exponents = exponent_compositions(args.degree, 5)
        initial_h = fubini_study_h(exponents)
        normalization = 1.0 / (math.pi * args.degree)
        dtype = (
            torch.complex64
            if args.precision == "complex64"
            else torch.complex128
        )
        device = torch.device(args.device)

        write_json(status_path, {"state": "running", "phase": "features"})
        training = make_dataset(
            train_x,
            train_y,
            np.asarray(train_pullbacks),
            exponents,
            feature_batch_size=args.feature_batch_size,
            dtype=dtype,
            device=device,
        )
        validation = make_dataset(
            validation_x,
            validation_y,
            np.asarray(validation_pullbacks),
            exponents,
            feature_batch_size=args.feature_batch_size,
            dtype=dtype,
            device=device,
        )
        model = PositiveFullH(initial_h, dtype=dtype, device=device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * args.steps_per_epoch,
            eta_min=args.minimum_learning_rate,
        )
        training_rng = np.random.default_rng(args.seed + 1)
        initial_validation = evaluate(
            initial_h,
            validation,
            normalization=normalization,
            chunk_size=args.metric_chunk_size,
            dtype=dtype,
            device=device,
        )
        best_h = initial_h.copy()
        best_epoch = 0
        best_score = float(
            initial_validation["statistics"]["weighted_rms_abs_residual"]
        )
        history = [
            {
                "epoch": 0,
                "learning_rate": args.learning_rate,
                "sigma": initial_validation["statistics"][
                    "sigma_official_formula"
                ],
                "chi": best_score,
                **initial_validation["tail"],
            }
        ]
        write_json(output_dir / "history.json", {"rows": history})
        print(
            f"sections={len(exponents)} real_parameters={len(exponents) ** 2} "
            f"initial_sigma={history[0]['sigma']:.6e} "
            f"initial_chi={history[0]['chi']:.6e}",
            flush=True,
        )

        write_json(status_path, {"state": "running", "phase": "training"})
        for epoch in range(1, args.epochs + 1):
            model.train()
            step_e2 = []
            step_sigma = []
            for _ in range(args.steps_per_epoch):
                if (
                    args.training_batch_size
                    and args.training_batch_size < training["count"]
                ):
                    batch_indices = training_rng.choice(
                        training["count"],
                        size=args.training_batch_size,
                        replace=False,
                    )
                    active_training = dataset_subset(training, batch_indices)
                else:
                    active_training = training
                optimizer.zero_grad(set_to_none=True)
                matrix = model()
                components = native_components(
                    matrix,
                    active_training,
                    normalization=normalization,
                    chunk_size=args.metric_chunk_size,
                )
                eigenvalues = torch.linalg.eigvalsh(matrix)
                log_condition = torch.log(eigenvalues[-1] / eigenvalues[0])
                loss = (
                    components["e2"]
                    + args.sigma_weight * components["sigma"]
                    + args.condition_weight * torch.square(log_condition)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clip_norm,
                )
                optimizer.step()
                scheduler.step()
                step_e2.append(float(components["e2"].detach().cpu()))
                step_sigma.append(float(components["sigma"].detach().cpu()))

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                candidate_h = model().detach().cpu().numpy()
                row_result = evaluate(
                    candidate_h,
                    validation,
                    normalization=normalization,
                    chunk_size=args.metric_chunk_size,
                    dtype=dtype,
                    device=device,
                )
                statistics = row_result["statistics"]
                score = float(statistics["weighted_rms_abs_residual"])
                row = {
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_e2": float(np.mean(step_e2)),
                    "train_sigma": float(np.mean(step_sigma)),
                    "sigma": statistics["sigma_official_formula"],
                    "chi": score,
                    **row_result["tail"],
                }
                history.append(row)
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    best_h = candidate_h.copy()
                write_json(output_dir / "history.json", {"rows": history})
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "training",
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_chi": best_score,
                    },
                )
                print(
                    f"epoch={epoch} val_sigma={row['sigma']:.6e} "
                    f"val_chi={row['chi']:.6e} best={best_score:.6e}",
                    flush=True,
                )

        best_validation = evaluate(
            best_h,
            validation,
            normalization=normalization,
            chunk_size=args.metric_chunk_size,
            dtype=dtype,
            device=device,
        )
        artifact_path = output_dir / "teacher_h.npz"
        np.savez_compressed(
            artifact_path,
            global_h_matrix=np.asarray(best_h, dtype=np.complex128),
            degree=np.asarray(args.degree, dtype=np.int64),
            exponents=exponents,
            normalization=np.asarray(normalization, dtype=np.float64),
        )
        report = {
            "schema": "generic-quintic-low-degree-teacher-v1",
            "role": "round_zero_initialization_only",
            "teacher_runtime_dependency_after_initialization": False,
            "geometry": experiment_manifest(
                teacher_degree=args.degree,
                target_degree=20,
            ),
            "configuration": {
                **vars(args),
                "data_dir": str(data_dir),
                "output_dir": str(output_dir),
            },
            "counts": {
                "training": training["count"],
                "validation": validation["count"],
            },
            "model": {
                "degree": args.degree,
                "section_count": len(exponents),
                "trainable_real_parameter_count": len(exponents) ** 2,
                "best_epoch": best_epoch,
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
            },
            "validation": best_validation,
            "source": {
                "dataset": str(dataset_path),
                "dataset_sha256": sha256_file(dataset_path),
                "train_pullbacks_sha256": sha256_file(
                    train_pullbacks_path
                ),
                "validation_pullbacks_sha256": sha256_file(
                    validation_pullbacks_path
                ),
            },
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "best_epoch": best_epoch,
                "best_chi": best_score,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
