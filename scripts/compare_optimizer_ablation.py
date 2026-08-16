#!/usr/bin/env python3
"""Compare fixed-budget endpoints and selected optimizer-ablation checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRAIN_KEYS = (
    "loss",
    "mean_training_group_volume_ratio_l2",
    "maximum_training_group_volume_ratio_l2",
    "mean_training_group_volume_ratio_cvar",
    "maximum_training_group_volume_ratio_cvar",
)
CHECKPOINT_KEYS = (
    "mean_check_log_rms",
    "mean_check_sigma",
    "mean_check_volume_ratio_l2",
    "maximum_check_volume_ratio_l2",
    "maximum_check_positive_log_ratio_q999",
    "maximum_check_positive_log_ratio_cvar",
    "maximum_check_ratio_above_3_weighted_mass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=SUMMARY.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-train-improvement", type=float, default=0.02)
    parser.add_argument("--maximum-checkpoint-regression", type=float, default=0.02)
    return parser.parse_args()


def relative_change(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return float((value - baseline) / abs(baseline))


def history_at_epoch(summary: dict[str, Any], epoch: int) -> dict[str, Any]:
    matches = [row for row in summary["history"] if int(row["epoch"]) == epoch]
    if len(matches) != 1:
        raise ValueError(f"history does not contain exactly one row for epoch {epoch}")
    return matches[0]


def metric_block(
    initial: dict[str, Any],
    candidate: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    return {
        key: {
            "initial": initial.get(key),
            "candidate": candidate.get(key),
            "relative_change": relative_change(candidate.get(key), initial.get(key)),
        }
        for key in keys
    }


def optimizer_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "mean_preclip_gradient_norm_since_previous_evaluation",
            "maximum_preclip_gradient_norm_since_previous_evaluation",
            "clipped_optimizer_steps_since_previous_evaluation",
            "recorded_spd_steps_since_previous_evaluation",
            "mean_spd_log_step_span_since_previous_evaluation",
            "maximum_spd_log_step_span_since_previous_evaluation",
            "maximum_spd_log_step_radius_since_previous_evaluation",
            "maximum_unprojected_spd_log_step_radius_since_previous_evaluation",
            "projected_spd_steps_since_previous_evaluation",
            "minimum_spd_step_projection_scale_since_previous_evaluation",
            "relative_to_initial_log_eigenvalue_span",
            "relative_to_initial_max_abs_centered_log_eigenvalue",
        )
    }


def main() -> None:
    args = parse_args()
    arms = []
    for value in args.arm:
        if "=" not in value:
            raise ValueError("--arm must use LABEL=SUMMARY.json")
        label, path_value = value.split("=", 1)
        path = Path(path_value).expanduser().resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        initial = history_at_epoch(summary, 0)
        selected_epoch = int(summary["best_epoch"])
        selected = history_at_epoch(summary, selected_epoch)
        endpoint = summary["history"][-1]
        endpoint_train = metric_block(initial, endpoint, TRAIN_KEYS)
        endpoint_checkpoint = metric_block(initial, endpoint, CHECKPOINT_KEYS)
        train_loss_change = endpoint_train["loss"]["relative_change"]
        checkpoint_regressions = [
            row["relative_change"]
            for row in endpoint_checkpoint.values()
            if row["relative_change"] is not None
        ]
        train_gate = bool(
            train_loss_change is not None
            and train_loss_change <= -args.minimum_train_improvement
        )
        checkpoint_gate = bool(
            checkpoint_regressions
            and max(checkpoint_regressions) <= args.maximum_checkpoint_regression
        )
        arms.append(
            {
                "label": label,
                "summary": str(path),
                "learning_rate": summary["request"]["learning_rate"],
                "gradient_clip_norm": summary["optimization"]["gradient_clip_norm"],
                "fixed_budget_endpoint": {
                    "epoch": int(endpoint["epoch"]),
                    "optimizer_step_count": endpoint.get("optimizer_step_count"),
                    "train": endpoint_train,
                    "checkpoint": endpoint_checkpoint,
                    "optimizer_diagnostics": optimizer_diagnostics(endpoint),
                },
                "selected_checkpoint": {
                    "epoch": selected_epoch,
                    "optimizer_step_count": selected.get("optimizer_step_count"),
                    "train": metric_block(initial, selected, TRAIN_KEYS),
                    "checkpoint": metric_block(initial, selected, CHECKPOINT_KEYS),
                    "optimizer_diagnostics": optimizer_diagnostics(selected),
                },
                "preregistered_endpoint_gate": {
                    "train_loss_improved_by_required_fraction": train_gate,
                    "no_checkpoint_metric_regressed_beyond_tolerance": checkpoint_gate,
                    "passed": bool(train_gate and checkpoint_gate),
                },
            }
        )
    result = {
        "schema_version": 2,
        "minimum_train_improvement": args.minimum_train_improvement,
        "maximum_checkpoint_regression": args.maximum_checkpoint_regression,
        "arms": arms,
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
