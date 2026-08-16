#!/usr/bin/env python3
"""Audit weighted normalization and CVaR values and directional derivatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.risk import (  # noqa: E402
    normalized_positive_weights_numpy,
    weighted_cvar_numpy,
    weighted_cvar_selected_linearization_torch,
    weighted_cvar_torch,
    weighted_log_mean_exp_numpy,
    weighted_log_mean_exp_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw-key", default="corrected_log_ma")
    parser.add_argument("--weights-key", default="importance_weights")
    parser.add_argument(
        "--tail-fractions",
        type=float,
        nargs="+",
        default=(1.0e-2, 1.0e-3, 1.0e-4),
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--finite-difference-step", type=float, default=1.0e-6)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def relative_error(value: float, reference: float) -> float:
    return float(abs(value - reference) / max(1.0, abs(reference)))


def numpy_losses(
    log_z: np.ndarray,
    weights: np.ndarray,
    *,
    kind: str,
) -> tuple[np.ndarray, float]:
    log_kappa = weighted_log_mean_exp_numpy(log_z, weights)
    centered = log_z - log_kappa
    if kind == "positive_log_ratio":
        return np.maximum(centered, 0.0), log_kappa
    if kind == "absolute_volume_ratio_error":
        return np.abs(1.0 - np.exp(centered)), log_kappa
    raise ValueError(f"unsupported loss kind {kind}")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as payload:
        log_z = np.asarray(payload[args.raw_key], dtype=np.float64)
        weights = np.asarray(payload[args.weights_key], dtype=np.float64)
    if log_z.ndim != 1 or weights.shape != log_z.shape:
        raise ValueError("raw residuals and weights must be equal-length vectors")
    probabilities = normalized_positive_weights_numpy(weights)
    log_kappa = weighted_log_mean_exp_numpy(log_z, weights)
    normalized_ratio = np.exp(log_z - log_kappa)
    rng = np.random.default_rng(args.seed)
    direction = rng.normal(size=len(log_z))
    direction /= np.sqrt(np.mean(direction**2))
    analytic_log_kappa_jvp = float(np.sum(probabilities * normalized_ratio * direction))

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the JVP audit") from exc
    log_z_t = torch.tensor(log_z, dtype=torch.float64, requires_grad=True)
    weights_t = torch.tensor(weights, dtype=torch.float64)
    direction_t = torch.tensor(direction, dtype=torch.float64)
    log_kappa_t = weighted_log_mean_exp_torch(log_z_t, weights_t)
    log_kappa_gradient = torch.autograd.grad(
        log_kappa_t,
        log_z_t,
        retain_graph=True,
    )[0]
    torch_log_kappa_jvp = float(torch.sum(log_kappa_gradient * direction_t))
    epsilon = float(args.finite_difference_step)
    finite_log_kappa_jvp = (
        weighted_log_mean_exp_numpy(log_z + epsilon * direction, weights)
        - weighted_log_mean_exp_numpy(log_z - epsilon * direction, weights)
    ) / (2.0 * epsilon)

    rows = []
    for kind in ("positive_log_ratio", "absolute_volume_ratio_error"):
        losses, _ = numpy_losses(log_z, weights, kind=kind)
        centered_t = log_z_t - log_kappa_t
        torch_losses = (
            torch.relu(centered_t)
            if kind == "positive_log_ratio"
            else torch.abs(1.0 - torch.exp(centered_t))
        )
        for tail_fraction in args.tail_fractions:
            audit = weighted_cvar_numpy(
                losses,
                weights,
                tail_fraction=tail_fraction,
            )
            ru_value = float(
                audit.var_threshold
                + np.sum(probabilities * np.maximum(losses - audit.var_threshold, 0.0))
                / tail_fraction
            )
            torch_value = weighted_cvar_torch(
                torch_losses,
                weights_t,
                tail_fraction=tail_fraction,
            )
            gradient = torch.autograd.grad(
                torch_value,
                log_z_t,
                retain_graph=True,
            )[0]
            torch_jvp = float(torch.sum(gradient * direction_t))
            boundary_selection_fraction = (
                audit.selected_boundary_mass / audit.boundary_mass
            )
            selected_fractions = (losses > audit.var_threshold).astype(
                np.float64
            ) + boundary_selection_fraction * (losses == audit.var_threshold).astype(
                np.float64
            )
            selected_linearization = weighted_cvar_selected_linearization_torch(
                torch_losses,
                weights_t,
                torch.tensor(selected_fractions, dtype=torch.float64),
                tail_fraction=tail_fraction,
            )
            selected_gradient = torch.autograd.grad(
                selected_linearization,
                log_z_t,
                retain_graph=True,
            )[0]
            selected_jvp = float(torch.sum(selected_gradient * direction_t))

            def objective(candidate: np.ndarray) -> float:
                candidate_losses, _ = numpy_losses(candidate, weights, kind=kind)
                return weighted_cvar_numpy(
                    candidate_losses,
                    weights,
                    tail_fraction=tail_fraction,
                ).value

            finite_jvp = (
                objective(log_z + epsilon * direction)
                - objective(log_z - epsilon * direction)
            ) / (2.0 * epsilon)
            rows.append(
                {
                    "kind": kind,
                    "tail_fraction": tail_fraction,
                    "numpy_weighted_sort_value": audit.value,
                    "rockafellar_uryasev_value": ru_value,
                    "torch_value": float(torch_value.detach()),
                    "frozen_selection_linearization_value": float(
                        selected_linearization.detach()
                    ),
                    "value_relative_error_torch_vs_numpy": relative_error(
                        float(torch_value.detach()), audit.value
                    ),
                    "value_relative_error_ru_vs_numpy": relative_error(
                        ru_value, audit.value
                    ),
                    "value_relative_error_frozen_selection_vs_numpy": (
                        relative_error(
                            float(selected_linearization.detach()),
                            audit.value,
                        )
                    ),
                    "var_threshold": audit.var_threshold,
                    "selected_tail_mass": audit.selected_tail_mass,
                    "strict_tail_mass": audit.strict_tail_mass,
                    "boundary_mass": audit.boundary_mass,
                    "selected_boundary_mass": audit.selected_boundary_mass,
                    "torch_directional_derivative": torch_jvp,
                    "frozen_selection_directional_derivative": selected_jvp,
                    "finite_difference_directional_derivative": finite_jvp,
                    "directional_derivative_relative_error": relative_error(
                        torch_jvp, finite_jvp
                    ),
                    "frozen_selection_directional_derivative_relative_error": (
                        relative_error(selected_jvp, finite_jvp)
                    ),
                }
            )
    result = {
        "schema_version": 1,
        "input": str(input_path),
        "raw_key": args.raw_key,
        "weights_key": args.weights_key,
        "point_count": len(log_z),
        "log_kappa": log_kappa,
        "sum_normalized_importance_weights": float(np.sum(probabilities)),
        "sum_p_times_r": float(np.sum(probabilities * normalized_ratio)),
        "log_kappa_directional_derivative": {
            "analytic": analytic_log_kappa_jvp,
            "torch": torch_log_kappa_jvp,
            "finite_difference": finite_log_kappa_jvp,
            "torch_vs_analytic_relative_error": relative_error(
                torch_log_kappa_jvp, analytic_log_kappa_jvp
            ),
            "torch_vs_finite_difference_relative_error": relative_error(
                torch_log_kappa_jvp, finite_log_kappa_jvp
            ),
        },
        "cvar": rows,
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        output = args.out.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
