#!/usr/bin/env python3
"""Small global Kahler-potential correction pilot for the accepted gCICY H metric."""

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

from gcicy_metric import (  # noqa: E402
    all_projective_charts,
    generic_global_h_metrics,
    generic_holomorphic_volume_log_density,
    generic_importance_weights,
    generic_point_in_chart,
    make_exact_generic_model,
    sample_generic_gcicy_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=ROOT / "outputs" / "gcicy_generic_global_h_metric_k3_rank64_whitened_gpu.npz",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--load-model",
        type=Path,
        help="Load a saved potential checkpoint; with --steps 0 this performs evaluation only.",
    )
    parser.add_argument("--train-points", type=int, default=4096)
    parser.add_argument("--validation-points", type=int, default=1024)
    parser.add_argument("--test-points", type=int, default=512)
    parser.add_argument("--test-seeds", default="13201,13202,13203,13204")
    parser.add_argument("--train-seed", type=int, default=13101)
    parser.add_argument("--validation-seed", type=int, default=13102)
    parser.add_argument("--torch-seed", type=int, default=13103)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--patience-evals", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--potential-scale", type=float, default=0.05)
    parser.add_argument("--ratio-energy-weight", type=float, default=0.25)
    parser.add_argument("--barrier-weight", type=float, default=20.0)
    parser.add_argument("--relative-eigenvalue-floor", type=float, default=0.01)
    parser.add_argument("--correction-weight", type=float, default=1e-4)
    parser.add_argument("--min-relative-sigma-improvement", type=float, default=0.01)
    parser.add_argument("--chart-points", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "gcicy_global_phi_pilot_summary.json",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "outputs" / "gcicy_global_phi_pilot.pt",
    )
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chart_id(chart: tuple[int, int, int]) -> int:
    return int(chart[0] * 12 + chart[1] * 6 + chart[2])


def id_chart(identifier: int) -> tuple[int, int, int]:
    return identifier // 12, (identifier % 12) // 6, identifier % 6


def confidence_interval(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return [mean, mean]
    half_width = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(len(array))
    return [mean - half_width, mean + half_width]


def weighted_error_stats(raw: np.ndarray, weights: np.ndarray, min_eigenvalue: float) -> dict[str, float]:
    values = np.asarray(raw, dtype=float)
    importance = np.asarray(weights, dtype=float)
    normalized_weights = importance / np.sum(importance)
    centered = values - float(np.sum(normalized_weights * values))
    shifted_ratio = np.exp(values - float(np.max(values)))
    normalized_ratio = shifted_ratio / float(np.sum(normalized_weights * shifted_ratio))
    ratio_error = 1.0 - normalized_ratio
    inverse_ratio_error = 1.0 - 1.0 / normalized_ratio
    energy = float(np.sum(normalized_weights * ratio_error**2))
    absolute_centered = np.abs(centered)
    return {
        "sigma": float(np.sum(normalized_weights * np.abs(ratio_error))),
        "inverse_sigma": float(np.sum(normalized_weights * np.abs(inverse_ratio_error))),
        "squared_energy": energy,
        "sqrt_squared_energy": float(np.sqrt(energy)),
        "weighted_centered_log_ma_rms": float(
            np.sqrt(np.sum(normalized_weights * centered**2))
        ),
        "absolute_centered_log_ma_p95": float(np.quantile(absolute_centered, 0.95)),
        "absolute_centered_log_ma_p99": float(np.quantile(absolute_centered, 0.99)),
        "max_absolute_centered_log_ma": float(np.max(absolute_centered)),
        "min_normalized_ratio": float(np.min(normalized_ratio)),
        "max_normalized_ratio": float(np.max(normalized_ratio)),
        "min_metric_eigenvalue": float(min_eigenvalue),
        "importance_effective_sample_size": float(
            np.sum(importance) ** 2 / np.sum(importance**2)
        ),
    }


def aggregate_seed_rows(rows: list[dict]) -> dict:
    output = {"seeds": rows}
    for key in (
        "sigma",
        "inverse_sigma",
        "squared_energy",
        "sqrt_squared_energy",
        "weighted_centered_log_ma_rms",
    ):
        values = [float(row[key]) for row in rows]
        output[f"mean_{key}"] = float(np.mean(values))
        output[f"{key}_95_percent_ci"] = confidence_interval(values)
    output["min_metric_eigenvalue"] = float(
        np.min([row["min_metric_eigenvalue"] for row in rows])
    )
    output["max_seed_absolute_centered_log_ma"] = float(
        np.max([row["max_absolute_centered_log_ma"] for row in rows])
    )
    return output


def paired_improvement_summary(
    baseline_rows: list[dict], corrected_rows: list[dict], key: str
) -> dict:
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    corrected_by_seed = {int(row["seed"]): row for row in corrected_rows}
    if baseline_by_seed.keys() != corrected_by_seed.keys():
        raise ValueError("baseline and corrected rows must contain the same seeds")
    rows = []
    absolute_improvements = []
    baseline_values = []
    for seed in sorted(baseline_by_seed):
        baseline = float(baseline_by_seed[seed][key])
        corrected = float(corrected_by_seed[seed][key])
        improvement = baseline - corrected
        baseline_values.append(baseline)
        absolute_improvements.append(improvement)
        rows.append(
            {
                "seed": seed,
                "baseline": baseline,
                "corrected": corrected,
                "absolute_improvement": improvement,
                "relative_improvement": improvement / baseline,
            }
        )
    mean_baseline = float(np.mean(baseline_values))
    absolute_interval = confidence_interval(absolute_improvements)
    return {
        "metric": key,
        "seeds": rows,
        "mean_absolute_improvement": float(np.mean(absolute_improvements)),
        "absolute_improvement_95_percent_ci": absolute_interval,
        "relative_improvement_against_mean_baseline": float(
            np.mean(absolute_improvements) / mean_baseline
        ),
        "relative_improvement_95_percent_ci": [
            absolute_interval[0] / mean_baseline,
            absolute_interval[1] / mean_baseline,
        ],
    }


def main() -> None:
    try:
        import torch
        from torch.func import hessian, vmap
    except ImportError as exc:
        raise SystemExit("PyTorch with torch.func is required for this pilot") from exc

    args = parse_args()
    total_start = time.perf_counter()
    if min(args.train_points, args.validation_points, args.test_points) <= 0:
        raise SystemExit("point counts must be positive")
    if args.hidden <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0 or args.steps < 0:
        raise SystemExit("hidden, batch sizes, and steps must be positive (steps may be zero)")
    if args.eval_every <= 0 or args.patience_evals < 0:
        raise SystemExit("--eval-every must be positive and --patience-evals non-negative")
    if not args.base_artifact.exists():
        raise SystemExit(f"missing base artifact: {args.base_artifact}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    torch.manual_seed(args.torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.torch_seed)
    if args.dtype == "float64":
        real_dtype = torch.float64
        complex_dtype = torch.complex128
        numpy_real_dtype = np.float64
        numpy_complex_dtype = np.complex128
    else:
        real_dtype = torch.float32
        complex_dtype = torch.complex64
        numpy_real_dtype = np.float32
        numpy_complex_dtype = np.complex64

    artifact = np.load(args.base_artifact)
    model_seed = int(artifact["generic_model_seed"])
    model = make_exact_generic_model(model_seed)
    if not np.allclose(artifact["p1_coefficients"], model.p1_coefficients) or not np.allclose(
        artifact["p2_tensor"], model.p2_tensor
    ):
        raise SystemExit("base artifact does not match its exact gCICY model")
    exponents = artifact["global_section_exponents"]
    h_matrix = artifact["global_h_matrix"]
    normalization = float(artifact["global_section_normalization"])

    def prepare_points(points, seed: int | None = None):
        base_metrics = generic_global_h_metrics(
            points, exponents, h_matrix, normalization=normalization
        )
        return {
            "seed": seed,
            "points": points,
            "coords": np.asarray(
                [
                    np.concatenate([point.affine_coordinates.real, point.affine_coordinates.imag])
                    for point in points
                ],
                dtype=numpy_real_dtype,
            ),
            "charts": np.asarray([chart_id(point.projective_chart) for point in points], dtype=np.int64),
            "tangent": np.asarray([point.tangent_basis for point in points], dtype=numpy_complex_dtype),
            "base_metrics": np.asarray(base_metrics, dtype=numpy_complex_dtype),
            "log_omega": np.asarray(
                [generic_holomorphic_volume_log_density(point) for point in points],
                dtype=numpy_real_dtype,
            ),
            "weights": np.asarray(generic_importance_weights(points), dtype=numpy_real_dtype),
        }

    def sample_dataset(n_points: int, seed: int):
        return prepare_points(sample_generic_gcicy_points(model, n_points, seed=seed), seed=seed)

    print("sampling pilot datasets", flush=True)
    sampling_start = time.perf_counter()
    train_numpy = sample_dataset(args.train_points, args.train_seed)
    validation_numpy = sample_dataset(args.validation_points, args.validation_seed)
    test_numpy = [sample_dataset(args.test_points, seed) for seed in parse_ints(args.test_seeds)]
    sampling_seconds = time.perf_counter() - sampling_start

    def to_torch(dataset):
        return {
            "seed": dataset["seed"],
            "coords": torch.tensor(dataset["coords"], dtype=real_dtype, device=device),
            "charts": torch.tensor(dataset["charts"], dtype=torch.int64, device=device),
            "tangent": torch.tensor(dataset["tangent"], dtype=complex_dtype, device=device),
            "base_metrics": torch.tensor(
                dataset["base_metrics"], dtype=complex_dtype, device=device
            ),
            "log_omega": torch.tensor(dataset["log_omega"], dtype=real_dtype, device=device),
            "weights": torch.tensor(dataset["weights"], dtype=real_dtype, device=device),
        }

    train = to_torch(train_numpy)
    validation = to_torch(validation_numpy)
    tests = [to_torch(dataset) for dataset in test_numpy]

    class GlobalPotentialNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(44, args.hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(args.hidden, args.hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(args.hidden, 1, bias=False),
            )
            torch.nn.init.zeros_(self.net[-1].weight)

        def forward(self, features):
            return self.net(features).squeeze(-1)

    potential = GlobalPotentialNet().to(device=device, dtype=real_dtype)
    loaded_checkpoint = None
    if args.load_model is not None:
        if not args.load_model.exists():
            raise SystemExit(f"missing potential checkpoint: {args.load_model}")
        loaded_checkpoint = torch.load(args.load_model, map_location="cpu")
        if int(loaded_checkpoint["hidden"]) != args.hidden:
            raise SystemExit(
                f"checkpoint hidden width {loaded_checkpoint['hidden']} does not match --hidden {args.hidden}"
            )
        if not np.isclose(float(loaded_checkpoint["potential_scale"]), args.potential_scale):
            raise SystemExit("checkpoint potential scale does not match --potential-scale")
        potential.load_state_dict(loaded_checkpoint["state_dict"])

    def homogeneous_block(
        active_real: torch.Tensor,
        active_imag: torch.Tensor,
        size: int,
        fixed_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_parts = []
        imag_parts = []
        cursor = 0
        for index in range(size):
            if index == fixed_index:
                real_parts.append(torch.ones_like(active_real[0]))
                imag_parts.append(torch.zeros_like(active_imag[0]))
            else:
                real_parts.append(active_real[cursor])
                imag_parts.append(active_imag[cursor])
                cursor += 1
        return torch.stack(real_parts), torch.stack(imag_parts)

    def density_features(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        denominator = torch.clamp(torch.sum(real**2 + imag**2), min=1e-12)
        features = []
        for left in range(len(real)):
            features.append((real[left] ** 2 + imag[left] ** 2) / denominator)
        for left in range(len(real)):
            for right in range(left + 1, len(real)):
                features.append(
                    (real[left] * real[right] + imag[left] * imag[right]) / denominator
                )
                features.append(
                    (imag[left] * real[right] - real[left] * imag[right]) / denominator
                )
        return torch.stack(features)

    def projective_features(coords: torch.Tensor, chart: tuple[int, int, int]) -> torch.Tensor:
        real = coords[:7]
        imag = coords[7:]
        x_real, x_imag = homogeneous_block(real[:1], imag[:1], 2, chart[0])
        y_real, y_imag = homogeneous_block(real[1:2], imag[1:2], 2, chart[1])
        z_real, z_imag = homogeneous_block(real[2:], imag[2:], 6, chart[2])
        return torch.cat(
            [
                density_features(x_real, x_imag),
                density_features(y_real, y_imag),
                density_features(z_real, z_imag),
            ]
        )

    def make_single_potential(chart: tuple[int, int, int]):
        def single(coords: torch.Tensor) -> torch.Tensor:
            return args.potential_scale * potential(projective_features(coords, chart))

        return single

    hessian_functions = [
        vmap(hessian(make_single_potential(id_chart(identifier)))) for identifier in range(24)
    ]

    def correction_batch(
        coords: torch.Tensor, charts: torch.Tensor, tangent: torch.Tensor
    ) -> torch.Tensor:
        ambient_correction = torch.zeros(
            (len(coords), 7, 7), dtype=complex_dtype, device=device
        )
        for identifier, batched_hessian in enumerate(hessian_functions):
            indices = torch.nonzero(charts == identifier, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            real_hessian = batched_hessian(coords[indices])
            h_xx = real_hessian[:, :7, :7]
            h_xy = real_hessian[:, :7, 7:]
            h_yx = real_hessian[:, 7:, :7]
            h_yy = real_hessian[:, 7:, 7:]
            complex_hessian = 0.25 * (h_xx + h_yy).to(complex_dtype)
            # Matrix rows follow the repository's v^dagger g v convention:
            # g[i,j] = partial_bar_i partial_j phi.
            complex_hessian = complex_hessian - 0.25j * (h_xy - h_yx).to(complex_dtype)
            ambient_correction[indices] = complex_hessian
        pulled_back = torch.einsum(
            "nai,nab,nbj->nij", torch.conj(tangent), ambient_correction, tangent
        )
        return 0.5 * (pulled_back + torch.conj(torch.transpose(pulled_back, 1, 2)))

    def batch_metrics(dataset, indices: torch.Tensor):
        correction = correction_batch(
            dataset["coords"][indices], dataset["charts"][indices], dataset["tangent"][indices]
        )
        metric = dataset["base_metrics"][indices] + correction
        metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        eigenvalues = torch.linalg.eigvalsh(metric)
        raw = torch.log(torch.clamp(eigenvalues, min=1e-12)).sum(dim=1)
        raw = raw - dataset["log_omega"][indices]
        return metric, correction, eigenvalues, raw

    def training_loss(indices: torch.Tensor):
        metric, correction, eigenvalues, raw = batch_metrics(train, indices)
        weights = train["weights"][indices]
        weight_sum = torch.sum(weights)
        mean = torch.sum(weights * raw) / weight_sum
        centered = raw - mean
        log_loss = torch.sum(weights * centered**2) / weight_sum

        shifted_ratio = torch.exp(torch.clamp(raw - torch.max(raw), min=-20.0, max=0.0))
        normalized_ratio = shifted_ratio / (torch.sum(weights * shifted_ratio) / weight_sum)
        ratio_energy = torch.sum(weights * (1.0 - normalized_ratio) ** 2) / weight_sum

        eigen_scale = torch.clamp(torch.mean(torch.abs(eigenvalues), dim=1), min=1e-12)
        relative_eigenvalues = eigenvalues / eigen_scale[:, None]
        barrier = torch.nn.functional.softplus(
            (args.relative_eigenvalue_floor - relative_eigenvalues) * 50.0
        ).mean() / 50.0
        correction_size = torch.mean(torch.abs(correction) ** 2) / torch.clamp(
            torch.mean(torch.abs(metric) ** 2), min=1e-12
        )
        loss = (
            log_loss
            + args.ratio_energy_weight * ratio_energy
            + args.barrier_weight * barrier
            + args.correction_weight * correction_size
        )
        return loss, {
            "log_loss": log_loss,
            "ratio_energy": ratio_energy,
            "barrier": barrier,
            "correction_size": correction_size,
        }

    def evaluate(dataset, *, corrected: bool) -> dict[str, float]:
        raw_rows = []
        eigenvalue_rows = []
        count = len(dataset["coords"])
        for start in range(0, count, args.eval_batch_size):
            indices = torch.arange(
                start, min(start + args.eval_batch_size, count), device=device
            )
            if corrected:
                with torch.enable_grad():
                    _, _, eigenvalues, raw = batch_metrics(dataset, indices)
                raw_rows.append(raw.detach().cpu().numpy())
                eigenvalue_rows.append(eigenvalues.detach().cpu().numpy())
            else:
                metrics = dataset["base_metrics"][indices]
                eigenvalues = torch.linalg.eigvalsh(metrics)
                raw = torch.log(torch.clamp(eigenvalues, min=1e-12)).sum(dim=1)
                raw = raw - dataset["log_omega"][indices]
                raw_rows.append(raw.detach().cpu().numpy())
                eigenvalue_rows.append(eigenvalues.detach().cpu().numpy())
        raw_array = np.concatenate(raw_rows)
        eigenvalue_array = np.concatenate(eigenvalue_rows)
        return weighted_error_stats(
            raw_array,
            dataset["weights"].detach().cpu().numpy(),
            float(np.min(eigenvalue_array)),
        )

    baseline_validation = evaluate(validation, corrected=False)
    evaluation_only = args.load_model is not None and args.steps == 0
    best_validation = baseline_validation
    best_step = -1
    best_state = copy.deepcopy(potential.state_dict())
    stale_evaluations = 0
    history = [
        {
            "step": -1,
            "accepted": True,
            "validation_sigma": baseline_validation["sigma"],
            "validation_inverse_sigma": baseline_validation["inverse_sigma"],
            "validation_log_rms": baseline_validation["weighted_centered_log_ma_rms"],
            "validation_min_eigenvalue": baseline_validation["min_metric_eigenvalue"],
        }
    ]
    if args.load_model is not None:
        loaded_validation = evaluate(validation, corrected=True)
        history.append(
            {
                "step": -2,
                "label": "loaded_checkpoint",
                "accepted": loaded_validation["sigma"] < baseline_validation["sigma"],
                "validation_sigma": loaded_validation["sigma"],
                "validation_inverse_sigma": loaded_validation["inverse_sigma"],
                "validation_log_rms": loaded_validation["weighted_centered_log_ma_rms"],
                "validation_min_eigenvalue": loaded_validation["min_metric_eigenvalue"],
            }
        )
        if evaluation_only or loaded_validation["sigma"] < best_validation["sigma"]:
            best_validation = loaded_validation

    optimizer = torch.optim.Adam(potential.parameters(), lr=args.lr)
    print(
        f"device={device}, train={args.train_points}, validation={args.validation_points}, "
        f"hidden={args.hidden}, steps={args.steps}, baseline sigma={baseline_validation['sigma']:.6e}",
        flush=True,
    )
    optimization_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(0, args.train_points, (args.batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = training_loss(indices)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(potential.parameters(), max_norm=10.0)
        optimizer.step()

        if step % args.eval_every != 0:
            continue
        validation_stats = evaluate(validation, corrected=True)
        accepted = bool(
            np.isfinite(validation_stats["sigma"])
            and validation_stats["min_metric_eigenvalue"] > 0
            and validation_stats["sigma"] < best_validation["sigma"]
        )
        if accepted:
            best_step = step
            best_validation = validation_stats
            best_state = copy.deepcopy(potential.state_dict())
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        row = {
            "step": step,
            "accepted": accepted,
            "loss": float(loss.detach().cpu()),
            "log_loss": float(components["log_loss"].detach().cpu()),
            "ratio_energy": float(components["ratio_energy"].detach().cpu()),
            "barrier": float(components["barrier"].detach().cpu()),
            "correction_size": float(components["correction_size"].detach().cpu()),
            "validation_sigma": validation_stats["sigma"],
            "validation_inverse_sigma": validation_stats["inverse_sigma"],
            "validation_log_rms": validation_stats["weighted_centered_log_ma_rms"],
            "validation_min_eigenvalue": validation_stats["min_metric_eigenvalue"],
        }
        history.append(row)
        print(
            f"step {step}: loss={row['loss']:.6e}, sigma={row['validation_sigma']:.6e}, "
            f"log_rms={row['validation_log_rms']:.6e}, min_eig={row['validation_min_eigenvalue']:.3e}, "
            f"accepted={accepted}",
            flush=True,
        )
        if args.patience_evals > 0 and stale_evaluations >= args.patience_evals:
            print(f"early stopping after {stale_evaluations} stale evaluations", flush=True)
            break
    optimization_seconds = time.perf_counter() - optimization_start

    potential.load_state_dict(best_state)
    evaluation_start = time.perf_counter()
    baseline_test_rows = []
    corrected_test_rows = []
    for dataset in tests:
        baseline_row = {"seed": dataset["seed"], **evaluate(dataset, corrected=False)}
        corrected_row = {"seed": dataset["seed"], **evaluate(dataset, corrected=True)}
        baseline_test_rows.append(baseline_row)
        corrected_test_rows.append(corrected_row)
        print(
            f"test seed {dataset['seed']}: sigma {baseline_row['sigma']:.6e} -> "
            f"{corrected_row['sigma']:.6e}, log_rms "
            f"{baseline_row['weighted_centered_log_ma_rms']:.6e} -> "
            f"{corrected_row['weighted_centered_log_ma_rms']:.6e}, "
            f"min_eig={corrected_row['min_metric_eigenvalue']:.3e}",
            flush=True,
        )

    baseline_test = aggregate_seed_rows(baseline_test_rows)
    corrected_test = aggregate_seed_rows(corrected_test_rows)
    paired_sigma = paired_improvement_summary(
        baseline_test_rows, corrected_test_rows, "sigma"
    )
    paired_log_rms = paired_improvement_summary(
        baseline_test_rows, corrected_test_rows, "weighted_centered_log_ma_rms"
    )

    chart_points = sample_generic_gcicy_points(model, max(1, args.chart_points), seed=13301)
    max_chart_ma_error = 0.0
    charts_seen: set[tuple[int, int, int]] = set()
    for point in chart_points:
        candidates = []
        for chart in all_projective_charts():
            if min(abs(point.x[chart[0]]), abs(point.y[chart[1]]), abs(point.z[chart[2]])) <= 1e-4:
                continue
            candidates.append(generic_point_in_chart(model, point.x, point.y, point.z, chart))
            charts_seen.add(chart)
        chart_dataset = to_torch(prepare_points(candidates))
        raw_values = []
        for index in range(len(candidates)):
            indices = torch.tensor([index], device=device)
            with torch.enable_grad():
                _, _, _, raw = batch_metrics(chart_dataset, indices)
            raw_values.append(float(raw.detach().cpu()[0]))
        if raw_values:
            max_chart_ma_error = max(max_chart_ma_error, float(np.max(raw_values) - np.min(raw_values)))

    relative_sigma_improvement = float(
        (baseline_test["mean_sigma"] - corrected_test["mean_sigma"])
        / baseline_test["mean_sigma"]
    )
    gates = {
        "validation_checkpoint_improved": best_validation["sigma"] < baseline_validation["sigma"],
        "test_sigma_improved": corrected_test["mean_sigma"] < baseline_test["mean_sigma"],
        "test_log_rms_improved": (
            corrected_test["mean_weighted_centered_log_ma_rms"]
            < baseline_test["mean_weighted_centered_log_ma_rms"]
        ),
        "minimum_requested_sigma_improvement": (
            relative_sigma_improvement >= args.min_relative_sigma_improvement
        ),
        "positive_on_all_test_points": corrected_test["min_metric_eigenvalue"] > 0,
        "complete_projective_atlas": len(charts_seen) == 24,
        "projective_consistency": max_chart_ma_error <= 1e-3,
    }
    evaluation_and_atlas_seconds = time.perf_counter() - evaluation_start
    summary = {
        "description": "Small global projective-invariant Kahler-potential correction pilot.",
        "base_artifact": str(args.base_artifact.resolve()),
        "base_artifact_sha256": sha256(args.base_artifact),
        "model_seed": model_seed,
        "device": str(device),
        "dtype": args.dtype,
        "loaded_model": str(args.load_model.resolve()) if args.load_model is not None else None,
        "evaluation_only": evaluation_only,
        "method": {
            "metric": "g_phi = g_H + i partial partialbar phi_theta",
            "features": "44 real entries of the three normalized projective density matrices",
            "network": f"44-{args.hidden}-{args.hidden}-1 tanh MLP",
            "potential_scale": args.potential_scale,
            "training_objective": "weighted centered log-MA plus normalized ratio energy and positivity barrier",
        },
        "sample_sizes": {
            "train": args.train_points,
            "validation": args.validation_points,
            "test_per_seed": args.test_points,
            "test_seeds": parse_ints(args.test_seeds),
        },
        "optimization": {
            "requested_steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "best_step": best_step,
        },
        "baseline_validation": baseline_validation,
        "best_validation": best_validation,
        "baseline_test": baseline_test,
        "corrected_test": corrected_test,
        "paired_sigma_improvement": paired_sigma,
        "paired_log_rms_improvement": paired_log_rms,
        "relative_test_sigma_improvement": relative_sigma_improvement,
        "atlas": {
            "projective_charts_seen": len(charts_seen),
            "max_corrected_ma_error": max_chart_ma_error,
        },
        "runtime_seconds": {
            "sampling": sampling_seconds,
            "optimization": optimization_seconds,
            "evaluation_and_atlas": evaluation_and_atlas_seconds,
            "total_before_serialization": time.perf_counter() - total_start,
        },
        "gates": gates,
        "pilot_success": bool(all(gates.values())),
        "history": history,
        "model_file": str(args.model_out.resolve()),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "hidden": args.hidden,
            "potential_scale": args.potential_scale,
            "base_artifact": str(args.base_artifact.resolve()),
            "model_seed": model_seed,
            "feature_definition": "projective_density_matrices_v1",
            "best_step": best_step,
        },
        args.model_out,
    )
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"pilot success={summary['pilot_success']}, relative sigma improvement="
        f"{100.0 * relative_sigma_improvement:.2f}%, chart error={max_chart_ma_error:.3e}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.model_out}", flush=True)


if __name__ == "__main__":
    main()
