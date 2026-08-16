#!/usr/bin/env python3
"""Estimate the F-level Fermat Reynolds projection of a trained quintic TN."""

from __future__ import annotations

import argparse
import json
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
from scripts.audit_quintic_tn_fermat_symmetry import (  # noqa: E402
    evaluate,
    exact_permutation_actions,
    exact_phase_actions,
    random_actions,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    ratio_statistics,
    sha256_file,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    apply_fermat_action_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--pullbacks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument("--action-counts", type=int, nargs="+", default=(1, 4, 16, 64))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=202607225)
    parser.add_argument(
        "--action-mode",
        choices=("random_full_group", "exact_phase", "exact_permutation"),
        default="random_full_group",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_action_counts(counts: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(int(value) for value in counts)
    if not values or any(value <= 0 for value in values):
        raise ValueError("Reynolds action counts must be positive")
    if tuple(sorted(set(values))) != values:
        raise ValueError("Reynolds action counts must be strictly increasing")
    return values


def fixed_kappa_metrics(
    raw: np.ndarray,
    weights: np.ndarray,
    fixed_log_kappa: float,
) -> dict[str, float]:
    ratio = np.exp(np.clip(raw - fixed_log_kappa, -20.0, 20.0))
    residual = ratio - 1.0
    return {
        "weighted_mean_ratio": float(np.sum(weights * ratio)),
        "sigma": float(np.sum(weights * np.abs(residual))),
        "chi": float(np.sqrt(np.sum(weights * residual**2))),
        "log_energy": float(
            np.sum(weights * np.square(raw - fixed_log_kappa))
        ),
        "ma_energy": float(np.sum(weights * residual**2)),
    }


def compact_statistics(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "sigma": float(statistics["sigma_official_formula"]),
        "chi": float(statistics["weighted_rms_abs_residual"]),
        "q99_9": float(
            statistics["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99": float(
            statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum": float(
            statistics["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
    }


def projected_raw_ratios(
    model: torch.nn.Module,
    points: np.ndarray,
    ambient_derivatives: np.ndarray,
    omega_squared: np.ndarray,
    actions,
    action_counts: tuple[int, ...],
    *,
    batch_size: int,
    complex_dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, float]]:
    """Average F and its jets before constructing each projected metric."""

    raw = {count: np.empty(len(points), dtype=np.float64) for count in action_counts}
    minimum = {
        count: np.empty(len(points), dtype=np.float64) for count in action_counts
    }
    maximum_correction_delta = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            values = torch.as_tensor(
                points[start:stop],
                dtype=complex_dtype,
                device=device,
            )
            derivatives = torch.as_tensor(
                ambient_derivatives[start:stop],
                dtype=complex_dtype,
                device=device,
            )
            cumulative_norm = None
            cumulative_gradient = None
            cumulative_mixed = None
            baseline_correction = None
            for action_index, action in enumerate(actions, start=1):
                transformed_values, transformed_derivatives = apply_fermat_action_torch(
                    values,
                    derivatives,
                    torch.as_tensor(action.permutation, device=device),
                    torch.as_tensor(action.phase_exponents, device=device),
                )
                moments, correction = model._homogeneously_normalized_feature_moments(
                    transformed_values,
                    transformed_derivatives,
                )
                if baseline_correction is None:
                    baseline_correction = correction
                else:
                    maximum_correction_delta = max(
                        maximum_correction_delta,
                        float(torch.max(torch.abs(correction - baseline_correction))),
                    )
                cumulative_norm = (
                    moments.norm
                    if cumulative_norm is None
                    else cumulative_norm + moments.norm
                )
                cumulative_gradient = (
                    moments.holomorphic_gradient
                    if cumulative_gradient is None
                    else cumulative_gradient + moments.holomorphic_gradient
                )
                cumulative_mixed = (
                    moments.mixed_hessian
                    if cumulative_mixed is None
                    else cumulative_mixed + moments.mixed_hessian
                )
                if action_index not in raw:
                    continue
                averaged = type(moments)(
                    norm=cumulative_norm / action_index,
                    holomorphic_gradient=cumulative_gradient / action_index,
                    mixed_hessian=cumulative_mixed / action_index,
                )
                metric = model._metric_from_moments(averaged)
                eigenvalues = torch.linalg.eigvalsh(metric)
                if not bool(torch.all(torch.isfinite(eigenvalues))):
                    raise FloatingPointError("projected metric has nonfinite eigenvalues")
                block_raw = torch.sum(torch.log(eigenvalues), dim=1)
                block_raw -= torch.log(
                    torch.as_tensor(
                        omega_squared[start:stop],
                        dtype=eigenvalues.dtype,
                        device=device,
                    )
                )
                raw[action_index][start:stop] = (
                    block_raw.detach().cpu().numpy().astype(np.float64)
                )
                minimum[action_index][start:stop] = (
                    torch.min(eigenvalues, dim=1)
                    .values.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
    diagnostics = {
        "maximum_log_norm_correction_delta": maximum_correction_delta,
    }
    return raw, minimum, diagnostics


def main() -> None:
    args = parse_args()
    action_counts = validate_action_counts(args.action_counts)
    if args.limit <= 0 or args.batch_size <= 0:
        raise SystemExit("limit and batch size must be positive")
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
    if int(payload.get("source_degree", 1)) != 1:
        raise ValueError("the current Reynolds evaluator requires an O(1) source")
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

    baseline_raw, _, baseline_minimum = evaluate(
        model,
        points,
        ambient_derivatives,
        omega_squared,
        source_degree=1,
        batch_size=args.batch_size,
        complex_dtype=complex_dtype,
        device=device,
    )
    baseline_statistics, _ = ratio_statistics(
        baseline_raw,
        weights,
        baseline_minimum,
    )
    if args.action_mode == "random_full_group":
        actions = random_actions(max(action_counts), seed=args.seed)
    elif args.action_mode == "exact_phase":
        available_actions = exact_phase_actions()
        if max(action_counts) > len(available_actions):
            raise ValueError("requested more than 625 exact phase actions")
        actions = available_actions[: max(action_counts)]
    else:
        available_actions = exact_permutation_actions()
        if max(action_counts) > len(available_actions):
            raise ValueError("requested more than 120 exact permutation actions")
        actions = available_actions[: max(action_counts)]
    projected_raw, projected_minimum, diagnostics = projected_raw_ratios(
        model,
        points,
        ambient_derivatives,
        omega_squared,
        actions,
        action_counts,
        batch_size=args.batch_size,
        complex_dtype=complex_dtype,
        device=device,
    )
    rows = []
    for count in action_counts:
        statistics, _ = ratio_statistics(
            projected_raw[count],
            weights,
            projected_minimum[count],
        )
        rows.append(
            {
                "action_count": count,
                "normalized_volume": compact_statistics(statistics),
                "fixed_kappa": fixed_kappa_metrics(
                    projected_raw[count],
                    weights,
                    fixed_log_kappa,
                ),
                "nonpositive_metric_count": int(
                    np.sum(projected_minimum[count] <= 0)
                ),
                "minimum_metric_eigenvalue": float(
                    np.min(projected_minimum[count])
                ),
            }
        )
    report = {
        "schema": "quintic-tn-stochastic-reynolds-projection-v1",
        "scientific_scope": {
            "projection_level": "F and its first/mixed-second jets before log and metric",
            "group": "(Z5)^4 semidirect S5",
            "estimator": "nested uniform Monte Carlo group average",
            "claim_limit": (
                "finite action counts estimate the Reynolds projection but do not "
                "provide machine-exact invariance"
            ),
        },
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "points": str(points_path),
        "points_sha256": sha256_file(points_path),
        "pullbacks": str(pullbacks_path),
        "pullbacks_sha256": sha256_file(pullbacks_path),
        "configuration": {
            "point_count": int(len(points)),
            "action_counts": list(action_counts),
            "seed": int(args.seed),
            "action_mode": args.action_mode,
            "batch_size": int(args.batch_size),
            "precision": precision,
            "device": str(device),
            "fixed_log_kappa": fixed_log_kappa,
        },
        "baseline": {
            "normalized_volume": compact_statistics(baseline_statistics),
            "fixed_kappa": fixed_kappa_metrics(
                baseline_raw,
                weights,
                fixed_log_kappa,
            ),
            "minimum_metric_eigenvalue": float(np.min(baseline_minimum)),
        },
        "projection_rows": rows,
        "diagnostics": diagnostics,
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(
        json.dumps(
            {
                "baseline": report["baseline"]["normalized_volume"],
                "projection_rows": rows,
                "diagnostics": diagnostics,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
