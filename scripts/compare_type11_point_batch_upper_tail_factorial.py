#!/usr/bin/env python3
"""Build the matched 2x2 point-batching by upper-tail-loss comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    (
        "cluster_baseline",
        "Complete fibres, no tail loss",
        "complete_fibres",
        False,
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_specreg_w5_baseline_e3",
    ),
    (
        "cluster_upper",
        "Complete fibres, upper-r loss",
        "complete_fibres",
        True,
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_specreg_w5_upperlogcvar_scaled_smoke",
    ),
    (
        "point_baseline",
        "Global point-64, no tail loss",
        "global_point_64",
        False,
        "p4p1_type11_hirzebruch_k4_pointb64_baseline_e3",
    ),
    (
        "point_upper",
        "Global point-64, upper-r loss",
        "global_point_64",
        True,
        "p4p1_type11_hirzebruch_k4_pointb64_upperlogcvar_e3",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/pipeline/type11_point_batch_upper_tail_factorial_20260717"
        ),
    )
    return parser.parse_args()


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "sigma": float(metrics["sigma"]),
        "volume_ratio_l2": float(metrics["sqrt_squared_energy"]),
        "ratio_max": float(metrics["normalized_ratio_max"]),
        "positive_log_ratio_q999": float(metrics["positive_log_ratio_q999"]),
        "positive_log_ratio_cvar_1pct": float(
            metrics["positive_log_ratio_cvar_1pct"]
        ),
        "ratio_above_3_weighted_mass": float(
            metrics["normalized_ratio_above_3_weighted_mass"]
        ),
    }


def load_arm(
    identifier: str,
    label: str,
    batching: str,
    upper_tail_loss: bool,
    stem: str,
) -> dict[str, Any]:
    path = ROOT / "outputs" / "pipeline" / f"{stem}_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = [
        compact_metrics(row["candidate"])
        for row in summary["checkpoint_validation"].values()
    ]
    final_history = summary["history"][-1]
    log_spectrum_span = float(
        summary["optimization"]["selected_relative_h_update"][
            "relative_to_initial_log_eigenvalue_span"
        ]
    )
    return {
        "id": identifier,
        "label": label,
        "batching": batching,
        "upper_tail_loss": upper_tail_loss,
        "source": str(path.relative_to(ROOT)),
        "request": summary["request"],
        "optimization": summary["optimization"],
        "best_epoch": int(summary["best_epoch"]),
        "checkpoints": {
            "mean_sigma": sum(row["sigma"] for row in checkpoints)
            / len(checkpoints),
            "maximum_volume_ratio_l2": max(
                row["volume_ratio_l2"] for row in checkpoints
            ),
            "maximum_ratio": max(row["ratio_max"] for row in checkpoints),
            "maximum_positive_log_ratio_q999": max(
                row["positive_log_ratio_q999"] for row in checkpoints
            ),
            "maximum_positive_log_ratio_cvar_1pct": max(
                row["positive_log_ratio_cvar_1pct"] for row in checkpoints
            ),
            "maximum_ratio_above_3_weighted_mass": max(
                row["ratio_above_3_weighted_mass"] for row in checkpoints
            ),
        },
        "export": compact_metrics(summary["export_candidate"]),
        "optimizer_diagnostics": {
            "relative_h_log_spectrum_span": log_spectrum_span,
            "mean_preclip_gradient_norm": float(
                final_history["mean_preclip_gradient_norm_since_previous_evaluation"]
            ),
            "maximum_preclip_gradient_norm": float(
                final_history[
                    "maximum_preclip_gradient_norm_since_previous_evaluation"
                ]
            ),
            "clipped_optimizer_steps": int(
                final_history["clipped_optimizer_steps_since_previous_evaluation"]
            ),
            "optimizer_steps": int(
                final_history["optimizer_steps_since_previous_evaluation"]
            ),
        },
    }


def validate_matched(arms: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "model_seed",
        "exact_model",
        "degree",
        "precision",
        "h_parameterization",
        "basis_points",
        "train_batches",
        "checkpoint_batches",
        "export_seed",
        "epochs",
        "learning_rate",
        "relative_log_spectrum_loss_weight",
        "torch_seed",
        "group_shuffle_seed",
    )
    reference = {key: arms[0]["request"][key] for key in keys}
    for arm in arms[1:]:
        observed = {key: arm["request"][key] for key in keys}
        if observed != reference:
            raise ValueError(f"arm {arm['id']} is not matched: {observed}")
    return {
        **reference,
        "training_points": sum(
            int(row["points"]) for row in reference["train_batches"]
        ),
        "points_per_optimizer_step": 64,
        "optimizer_steps_per_epoch": 512,
    }


def relative_percent(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def effect(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        key: relative_percent(candidate[key], baseline[key]) for key in baseline
    }


def factorial_effects(arms: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {arm["id"]: arm for arm in arms}
    metrics = {
        arm_id: {
            "check_mean_sigma": arm["checkpoints"]["mean_sigma"],
            "check_max_l2": arm["checkpoints"]["maximum_volume_ratio_l2"],
            "check_max_r": arm["checkpoints"]["maximum_ratio"],
            "check_q999": arm["checkpoints"][
                "maximum_positive_log_ratio_q999"
            ],
            "check_mass_r_gt_3": arm["checkpoints"][
                "maximum_ratio_above_3_weighted_mass"
            ],
            "export_sigma": arm["export"]["sigma"],
            "export_max_r": arm["export"]["ratio_max"],
        }
        for arm_id, arm in by_id.items()
    }
    return {
        "point_batch_effect_without_tail_percent": effect(
            metrics["point_baseline"], metrics["cluster_baseline"]
        ),
        "point_batch_effect_with_tail_percent": effect(
            metrics["point_upper"], metrics["cluster_upper"]
        ),
        "upper_tail_loss_effect_with_fibre_batches_percent": effect(
            metrics["cluster_upper"], metrics["cluster_baseline"]
        ),
        "upper_tail_loss_effect_with_point_batches_percent": effect(
            metrics["point_upper"], metrics["point_baseline"]
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    rows = [
        "| Arm | Check sigma | Check max L2 | Check max r | Check q99.9 | "
        "Check mass(r>3) | Export sigma | Export max r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        check = arm["checkpoints"]
        export = arm["export"]
        rows.append(
            f"| {arm['label']} | {check['mean_sigma']:.4f} | "
            f"{check['maximum_volume_ratio_l2']:.3f} | "
            f"{check['maximum_ratio']:.2f} | "
            f"{check['maximum_positive_log_ratio_q999']:.3f} | "
            f"{check['maximum_ratio_above_3_weighted_mass']:.3%} | "
            f"{export['sigma']:.4f} | {export['ratio_max']:.2f} |"
        )
    effect_rows = [
        "| Contrast | Check sigma | Check max L2 | Check max r | Check q99.9 | "
        "Check mass(r>3) | Export sigma | Export max r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "point_batch_effect_without_tail_percent": "Point batching, no tail loss",
        "point_batch_effect_with_tail_percent": "Point batching, with tail loss",
        "upper_tail_loss_effect_with_fibre_batches_percent": (
            "Upper-r loss, fibre batches"
        ),
        "upper_tail_loss_effect_with_point_batches_percent": (
            "Upper-r loss, point batches"
        ),
    }
    keys = (
        "check_mean_sigma",
        "check_max_l2",
        "check_max_r",
        "check_q999",
        "check_mass_r_gt_3",
        "export_sigma",
        "export_max_r",
    )
    for effect_id, label in labels.items():
        values = payload["effects"][effect_id]
        effect_rows.append(
            f"| {label} | "
            + " | ".join(f"{values[key]:+.2f}%" for key in keys)
            + " |"
        )
    optimizer_rows = [
        "| Arm | H log-spectrum span | Mean preclip grad | Max preclip grad | "
        "Clipped steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        diagnostic = arm["optimizer_diagnostics"]
        optimizer_rows.append(
            f"| {arm['label']} | "
            f"{diagnostic['relative_h_log_spectrum_span']:.3f} | "
            f"{diagnostic['mean_preclip_gradient_norm']:.2f} | "
            f"{diagnostic['maximum_preclip_gradient_norm']:.2f} | "
            f"{diagnostic['clipped_optimizer_steps']} / "
            f"{diagnostic['optimizer_steps']} |"
        )
    return f"""# Type-(1,1) global point-64 by upper-r loss factorial

## Matched design

The four arms use identical section basis, 32,768 training points, 16,384
checkpoint points, 8,192 export points, initialization, model and shuffle
seeds, 64 points per optimizer step, 512 updates per epoch, and three epochs.
Only fibre grouping and the one-sided upper-r objective change.

## Absolute results

{"\n".join(rows)}

## Relative effects

Negative values are reductions; every listed diagnostic is lower-is-better.

{"\n".join(effect_rows)}

## Optimizer diagnostics

{"\n".join(optimizer_rows)}

The H spectral spans remain nearly identical, so a simple condition-number
change does not explain the factorial result. The upper-r objective roughly
doubles the mean pre-clipping gradient and raises the sampled maximum gradient
from order `1e2` to order `4e3`. Every step is clipped. With tail fractions
`0.001` and `0.0001`, only about 33 and 3 of 32,768 training points carry each
tail linearization. A random batch therefore usually receives no tail gradient
and occasionally receives a rescaled burst. The estimator is unbiased before
clipping, but clipping makes these sparse bursts nonlinear and biased.

## Interpretation

Global point-64 batching modestly improves bulk sigma, q99.9, and weighted
mass above three, but worsens the sampled maximum and maximum L2. It moves the
error toward a narrower tail rather than eliminating it. Combining point-64
with the current sparse upper-r CVaR is counterproductive for maxima. The next
tail-aware point-batch experiment should distribute tail information across
steps, for example by stratified tail replay or a separate full-tail gradient
step, instead of increasing the present loss weight.

This is a disclosed-seed development experiment. Sampled maxima are not a
global sup-norm certificate.
"""


def main() -> None:
    args = parse_args()
    arms = [load_arm(*row) for row in ARMS]
    payload = {
        "schema": "type11-point-batch-upper-tail-factorial-v1",
        "matched_configuration": validate_matched(arms),
        "arms": arms,
        "effects": factorial_effects(arms),
        "claim_limit": (
            "Disclosed-seed development ablation; not a fresh blind result or "
            "global sup-norm certificate."
        ),
    }
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {out_dir / 'report.json'}")
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
