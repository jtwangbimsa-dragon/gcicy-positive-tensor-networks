#!/usr/bin/env python3
"""Audit Fermat-quintic phase and permutation invariance of a TN metric."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    positive_tensor_network_from_artifact_payload,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    veronese_source_features_numpy,
)


@dataclass(frozen=True)
class FermatAction:
    label: str
    permutation: tuple[int, int, int, int, int]
    phase_exponents: tuple[int, int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--pullbacks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--random-elements", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=202607221)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator_actions() -> tuple[FermatAction, ...]:
    identity = (0, 1, 2, 3, 4)
    zero = (0, 0, 0, 0, 0)
    rows = [
        FermatAction("perm_swap_01", (1, 0, 2, 3, 4), zero),
        FermatAction("perm_cycle_01234", (1, 2, 3, 4, 0), zero),
    ]
    for coordinate in range(4):
        exponents = [0] * 5
        exponents[coordinate] = 1
        rows.append(
            FermatAction(
                f"phase_coordinate_{coordinate}",
                identity,
                tuple(exponents),
            )
        )
    return tuple(rows)


def random_actions(count: int, *, seed: int) -> tuple[FermatAction, ...]:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        permutation = tuple(int(value) for value in rng.permutation(5))
        exponents = rng.integers(0, 5, size=5)
        exponents = (exponents - exponents[-1]) % 5
        rows.append(
            FermatAction(
                f"random_{index:03d}",
                permutation,
                tuple(int(value) for value in exponents),
            )
        )
    return tuple(rows)


def exact_phase_actions() -> tuple[FermatAction, ...]:
    identity = (0, 1, 2, 3, 4)
    return tuple(
        FermatAction(
            "phase_" + "".join(str(value) for value in exponents),
            identity,
            tuple(exponents) + (0,),
        )
        for exponents in itertools.product(range(5), repeat=4)
    )


def exact_permutation_actions() -> tuple[FermatAction, ...]:
    zero = (0, 0, 0, 0, 0)
    return tuple(
        FermatAction(
            "permutation_" + "".join(str(value) for value in permutation),
            tuple(permutation),
            zero,
        )
        for permutation in itertools.permutations(range(5))
    )


def action_factors(action: FermatAction) -> np.ndarray:
    omega = np.exp(2j * np.pi / 5.0)
    return np.power(omega, np.asarray(action.phase_exponents, dtype=np.int64))


def apply_action(
    points: np.ndarray,
    ambient_derivatives: np.ndarray,
    action: FermatAction,
) -> tuple[np.ndarray, np.ndarray]:
    """Transport homogeneous points and their ambient tangent vectors."""

    point_array = np.asarray(points, dtype=np.complex128)
    derivative_array = np.asarray(ambient_derivatives, dtype=np.complex128)
    if point_array.ndim != 2 or point_array.shape[1] != 5:
        raise ValueError("points must have shape (n,5)")
    if derivative_array.shape[:2] != point_array.shape:
        raise ValueError("ambient derivatives must have shape (n,5,d)")
    factors = action_factors(action)
    permutation = np.asarray(action.permutation, dtype=np.int64)
    transformed_points = point_array[:, permutation] * factors[None, :]
    transformed_derivatives = (
        derivative_array[:, permutation, :] * factors[None, :, None]
    )
    return transformed_points, transformed_derivatives


def fermat_polynomial(points: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(points) ** 5, axis=1)


def fermat_tangent_residual(
    points: np.ndarray,
    ambient_derivatives: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "na,naj->nj",
        5.0 * np.asarray(points) ** 4,
        np.asarray(ambient_derivatives),
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)
    return float(values[order[index]])


def residual_metrics(raw: np.ndarray, weights: np.ndarray, log_kappa: float) -> dict[str, float]:
    ratio = np.exp(np.clip(raw - log_kappa, -20.0, 20.0))
    absolute = np.abs(ratio - 1.0)
    return {
        "sigma": float(np.sum(weights * absolute)),
        "chi": float(np.sqrt(np.sum(weights * absolute**2))),
        "q99_9": weighted_quantile(absolute, weights, 0.999),
        "maximum": float(np.max(absolute)),
    }


def delta_metrics(delta: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    absolute = np.abs(delta)
    return {
        "weighted_mean": float(np.sum(weights * delta)),
        "weighted_rms": float(np.sqrt(np.sum(weights * delta**2))),
        "weighted_mean_absolute": float(np.sum(weights * absolute)),
        "weighted_q99": weighted_quantile(absolute, weights, 0.99),
        "weighted_q99_9": weighted_quantile(absolute, weights, 0.999),
        "maximum_absolute": float(np.max(absolute)),
    }


def evaluate(
    model: torch.nn.Module,
    points: np.ndarray,
    ambient_derivatives: np.ndarray,
    omega_squared: np.ndarray,
    *,
    source_degree: int,
    batch_size: int,
    complex_dtype: torch.dtype,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, derivatives = veronese_source_features_numpy(
        points,
        ambient_derivatives,
        source_degree=source_degree,
    )
    raw_rows = []
    potential_rows = []
    minimum_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            value_block = torch.as_tensor(
                values[start:stop], dtype=complex_dtype, device=device
            )
            derivative_block = torch.as_tensor(
                derivatives[start:stop], dtype=complex_dtype, device=device
            )
            potential, metric = model.potential_and_metric(
                value_block,
                derivative_block,
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(torch.isfinite(eigenvalues))):
                raise FloatingPointError("nonfinite metric eigenvalue")
            raw = torch.sum(torch.log(eigenvalues), dim=1)
            raw -= torch.log(
                torch.as_tensor(
                    omega_squared[start:stop],
                    dtype=eigenvalues.dtype,
                    device=device,
                )
            )
            raw_rows.append(raw.detach().cpu().numpy().astype(np.float64))
            potential_rows.append(
                potential.detach().cpu().numpy().astype(np.float64)
            )
            minimum_rows.append(
                torch.min(eigenvalues, dim=1).values.detach().cpu().numpy().astype(np.float64)
            )
    return (
        np.concatenate(raw_rows),
        np.concatenate(potential_rows),
        np.concatenate(minimum_rows),
    )


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.batch_size <= 0 or args.random_elements < 0:
        raise SystemExit("limits and batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model_path = args.model.expanduser().resolve()
    points_path = args.points.expanduser().resolve()
    pullbacks_path = args.pullbacks.expanduser().resolve()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    precision = str(payload["precision"])
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    reference = payload["state_dict"]["reference_h"]
    reference_h = np.asarray(reference.detach().cpu(), dtype=np.complex128)
    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    )
    point_data = np.load(points_path, allow_pickle=False)
    x_values = np.asarray(point_data["X"][: args.limit], dtype=np.float64)
    points = x_values[:, :5] + 1j * x_values[:, 5:]
    weights = np.asarray(point_data["weights"][: args.limit], dtype=np.float64)
    weights /= np.sum(weights)
    omega_squared = np.asarray(
        point_data["omega_squared"][: args.limit], dtype=np.float64
    )
    pullbacks = np.asarray(
        np.load(pullbacks_path, mmap_mode="r")[: args.limit],
        dtype=np.complex128,
    )
    ambient_derivatives = np.transpose(pullbacks, (0, 2, 1))
    fixed_log_kappa = float(payload["fixed_log_kappa"])
    source_degree = int(payload.get("source_degree", 1))
    baseline_raw, baseline_potential, baseline_minimum = evaluate(
        model,
        points,
        ambient_derivatives,
        omega_squared,
        source_degree=source_degree,
        batch_size=args.batch_size,
        complex_dtype=complex_dtype,
        device=device,
    )
    actions = generator_actions() + random_actions(
        args.random_elements,
        seed=args.seed,
    )
    action_rows = []
    for action in actions:
        transformed_points, transformed_derivatives = apply_action(
            points,
            ambient_derivatives,
            action,
        )
        transformed_raw, transformed_potential, transformed_minimum = evaluate(
            model,
            transformed_points,
            transformed_derivatives,
            omega_squared,
            source_degree=source_degree,
            batch_size=args.batch_size,
            complex_dtype=complex_dtype,
            device=device,
        )
        action_rows.append(
            {
                "label": action.label,
                "permutation": list(action.permutation),
                "phase_exponents_mod_5": list(action.phase_exponents),
                "dimensionless_log_feature_delta": delta_metrics(
                    (transformed_potential - baseline_potential)
                    / float(model.target_normalization),
                    weights,
                ),
                "log_volume_ratio_delta": delta_metrics(
                    transformed_raw - baseline_raw,
                    weights,
                ),
                "transformed_residual": residual_metrics(
                    transformed_raw,
                    weights,
                    fixed_log_kappa,
                ),
                "minimum_metric_eigenvalue": float(np.min(transformed_minimum)),
                "maximum_fermat_equation_transport_error": float(
                    np.max(
                        np.abs(
                            fermat_polynomial(transformed_points)
                            - fermat_polynomial(points)
                        )
                    )
                ),
                "maximum_tangent_transport_error": float(
                    np.max(
                        np.abs(
                            fermat_tangent_residual(
                                transformed_points,
                                transformed_derivatives,
                            )
                            - fermat_tangent_residual(points, ambient_derivatives)
                        )
                    )
                ),
            }
        )
    report = {
        "schema": "quintic-tn-fermat-symmetry-audit-v1",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "points": str(points_path),
        "points_sha256": sha256_file(points_path),
        "pullbacks": str(pullbacks_path),
        "pullbacks_sha256": sha256_file(pullbacks_path),
        "configuration": {
            "point_count": int(len(points)),
            "generator_action_count": int(len(generator_actions())),
            "random_action_count": int(args.random_elements),
            "seed": int(args.seed),
            "precision": precision,
            "device": str(device),
            "fixed_log_kappa": fixed_log_kappa,
        },
        "baseline": {
            "residual": residual_metrics(
                baseline_raw,
                weights,
                fixed_log_kappa,
            ),
            "minimum_metric_eigenvalue": float(np.min(baseline_minimum)),
            "maximum_fermat_equation_residual": float(
                np.max(np.abs(fermat_polynomial(points)))
            ),
            "maximum_tangent_residual": float(
                np.max(np.abs(fermat_tangent_residual(points, ambient_derivatives)))
            ),
        },
        "actions": action_rows,
        "aggregate": {
            "maximum_log_feature_rms": float(
                max(
                    row["dimensionless_log_feature_delta"]["weighted_rms"]
                    for row in action_rows
                )
            ),
            "maximum_log_volume_ratio_rms": float(
                max(
                    row["log_volume_ratio_delta"]["weighted_rms"]
                    for row in action_rows
                )
            ),
            "maximum_log_volume_ratio_absolute": float(
                max(
                    row["log_volume_ratio_delta"]["maximum_absolute"]
                    for row in action_rows
                )
            ),
            "transformed_sigma_range": [
                float(min(row["transformed_residual"]["sigma"] for row in action_rows)),
                float(max(row["transformed_residual"]["sigma"] for row in action_rows)),
            ],
        },
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
