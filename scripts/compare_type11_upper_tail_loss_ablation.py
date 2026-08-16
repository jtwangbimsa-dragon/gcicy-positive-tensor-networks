#!/usr/bin/env python3
"""Build the matched three-arm type-(1,1) upper-tail loss comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    (
        "baseline",
        "No tail loss",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_specreg_w5_baseline_e3",
    ),
    (
        "symmetric_cvar",
        "Corrected symmetric CVaR",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_specreg_w5_globalcvar_scaled_smoke",
    ),
    (
        "upper_log_cvar",
        "Upper-log CVaR (r > 3)",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_specreg_w5_upperlogcvar_scaled_smoke",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/pipeline/type11_upper_tail_loss_ablation_20260717"),
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


def load_arm(identifier: str, label: str, stem: str) -> dict[str, Any]:
    path = ROOT / "outputs" / "pipeline" / f"{stem}_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = [
        compact_metrics(row["candidate"])
        for row in summary["checkpoint_validation"].values()
    ]
    span = float(
        summary["optimization"]["selected_relative_h_update"][
            "relative_to_initial_log_eigenvalue_span"
        ]
    )
    return {
        "id": identifier,
        "label": label,
        "source": str(path.relative_to(ROOT)),
        "request": summary["request"],
        "optimization": summary["optimization"],
        "best_epoch": int(summary["best_epoch"]),
        "final_history": summary["history"][-1],
        "export": compact_metrics(summary["export_candidate"]),
        "checkpoints": {
            "count": len(checkpoints),
            "mean_sigma": sum(row["sigma"] for row in checkpoints) / len(checkpoints),
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
        "relative_h": {
            "centered_log_eigenvalue_span": span,
            "condition_number": math.exp(span),
        },
    }


def matched_configuration(arms: list[dict[str, Any]]) -> dict[str, Any]:
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
        "group_optimizer_step_mode",
        "clusters_per_optimizer_step",
        "cluster_minibatch_loss_reduction",
        "cluster_minibatch_uniform_loss",
        "relative_log_spectrum_loss_weight",
        "torch_seed",
        "group_shuffle_seed",
    )
    reference = {key: arms[0]["request"][key] for key in keys}
    for arm in arms[1:]:
        observed = {key: arm["request"][key] for key in keys}
        if observed != reference:
            raise ValueError(f"arm {arm['id']} is not matched: {observed}")
    training_points = sum(int(row["points"]) for row in reference["train_batches"])
    points_per_step = int(reference["clusters_per_optimizer_step"]) * 4
    batches_per_sweep = training_points // points_per_step
    return {
        **reference,
        "training_points": training_points,
        "points_per_step": points_per_step,
        "batches_per_sweep": batches_per_sweep,
        "old_global_cvar_gradient_suppression_factor": (
            training_points
            if reference["cluster_minibatch_loss_reduction"] == "sum"
            else batches_per_sweep
        ),
    }


def relative_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def add_baseline_deltas(arms: list[dict[str, Any]]) -> None:
    baseline = arms[0]
    for arm in arms:
        check = arm["checkpoints"]
        export = arm["export"]
        baseline_check = baseline["checkpoints"]
        baseline_export = baseline["export"]
        arm["relative_to_baseline_percent"] = {
            "check_mean_sigma": relative_change(
                check["mean_sigma"], baseline_check["mean_sigma"]
            ),
            "check_maximum_volume_ratio_l2": relative_change(
                check["maximum_volume_ratio_l2"],
                baseline_check["maximum_volume_ratio_l2"],
            ),
            "check_maximum_ratio": relative_change(
                check["maximum_ratio"], baseline_check["maximum_ratio"]
            ),
            "check_maximum_positive_log_ratio_q999": relative_change(
                check["maximum_positive_log_ratio_q999"],
                baseline_check["maximum_positive_log_ratio_q999"],
            ),
            "check_maximum_ratio_above_3_weighted_mass": relative_change(
                check["maximum_ratio_above_3_weighted_mass"],
                baseline_check["maximum_ratio_above_3_weighted_mass"],
            ),
            "export_sigma": relative_change(
                export["sigma"], baseline_export["sigma"]
            ),
            "export_maximum_ratio": relative_change(
                export["ratio_max"], baseline_export["ratio_max"]
            ),
        }


def render_markdown(payload: dict[str, Any]) -> str:
    rows = [
        "| Arm | Check sigma | Check max L2 | Check max r | Check q99.9(log r+) | "
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
    table = "\n".join(rows)
    delta_rows = [
        "| Arm | Check sigma | Check max L2 | Check max r | "
        "Check q99.9(log r+) | Check mass(r>3) | Export sigma | Export max r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"][1:]:
        delta = arm["relative_to_baseline_percent"]
        delta_rows.append(
            f"| {arm['label']} | {delta['check_mean_sigma']:+.2f}% | "
            f"{delta['check_maximum_volume_ratio_l2']:+.2f}% | "
            f"{delta['check_maximum_ratio']:+.2f}% | "
            f"{delta['check_maximum_positive_log_ratio_q999']:+.2f}% | "
            f"{delta['check_maximum_ratio_above_3_weighted_mass']:+.2f}% | "
            f"{delta['export_sigma']:+.2f}% | "
            f"{delta['export_maximum_ratio']:+.2f}% |"
        )
    delta_table = "\n".join(delta_rows)
    factor = payload["matched_configuration"][
        "old_global_cvar_gradient_suppression_factor"
    ]
    return f"""# Type-(1,1) upper-tail loss ablation

## Matched design

All three arms use the same 32,768 training points, 16,384 checkpoint points,
8,192 export points, cold Fubini--Study initialization, 75,075-dimensional
scale-free full H ansatz, 64-point complete-fibre mini-batches, three epochs,
and identical RNG seeds. Only the tail objective changes.

The previous global-CVaR mini-batch implementation omitted the sweep/batch
rescaling. With sum reduction its requested coefficient was suppressed by a
factor of `{factor:,}` relative to the primary loss. The corrected arms freeze
the full-pool tail selection once per sweep and use the unbiased batch-scale
linearization.

## Results

{table}

`sigma` and L2 measure bulk accuracy; maximum ratio, positive-log q99.9, CVaR,
and weighted mass above three measure distinct aspects of the upper tail. No
single sampled maximum is treated as a global sup-norm certificate.

## Relative to no-tail baseline

Negative values are reductions; all listed diagnostics are lower-is-better.

{delta_table}

## Interpretation

The corrected objectives are active: both reduce checkpoint L2, sampled
maximum ratio, and q99.9. The upper-log objective gives the largest checkpoint
L2 and q99.9 reductions. Neither objective solves the tail problem: both make
bulk sigma and the weighted mass above three slightly worse, and the upper-log
arm does not reduce the independent export maximum. This three-epoch,
disclosed-seed ablation supports further tail-objective work, but it is not a
fresh blind validation or a global sup-norm certificate.
"""


def main() -> None:
    args = parse_args()
    arms = [load_arm(*row) for row in ARMS]
    add_baseline_deltas(arms)
    payload = {
        "schema": "type11-upper-tail-loss-ablation-v1",
        "matched_configuration": matched_configuration(arms),
        "arms": arms,
        "claim_limit": (
            "This is a disclosed-seed development ablation, not a fresh blind result "
            "or a global sup-norm certificate."
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
