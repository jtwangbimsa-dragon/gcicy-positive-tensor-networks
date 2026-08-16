#!/usr/bin/env python3
"""Compare the matched k=4 optimizer-step and gradient-clipping arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRIC_REDUCTIONS = {
    "sigma": ("mean", "max"),
    "sqrt_squared_energy": ("mean", "max"),
    "positive_log_ratio_q999": ("max",),
    "positive_log_ratio_cvar_1pct": ("max",),
    "normalized_ratio_above_3_weighted_mass": ("max",),
    "normalized_ratio_max": ("max",),
    "min_metric_eigenvalue": ("min",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--unclipped-summary", type=Path, required=True)
    parser.add_argument("--clipped-summary", type=Path, required=True)
    parser.add_argument("--control-artifact", type=Path)
    parser.add_argument("--unclipped-artifact", type=Path)
    parser.add_argument("--clipped-artifact", type=Path)
    parser.add_argument("--matched-epoch", type=int, default=10)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def aggregate_checkpoint_metrics(summary: dict[str, Any]) -> dict[str, float]:
    checkpoint_rows = summary["checkpoint_validation"]
    if not checkpoint_rows:
        raise ValueError("summary has no checkpoint validation rows")
    rows = [entry["candidate"] for entry in checkpoint_rows.values()]
    result: dict[str, float] = {}
    for metric, reductions in METRIC_REDUCTIONS.items():
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        for reduction in reductions:
            result[f"{reduction}_{metric}"] = float(getattr(np, reduction)(values))
    return result


def history_at_epoch(summary: dict[str, Any], epoch: int) -> dict[str, Any]:
    rows = [row for row in summary["history"] if int(row["epoch"]) == epoch]
    if len(rows) != 1:
        raise ValueError(f"summary does not contain exactly one epoch-{epoch} row")
    return rows[0]


def summarize_arm(summary: dict[str, Any], matched_epoch: int) -> dict[str, Any]:
    request = summary["request"]
    optimization = summary.get("optimization", {})
    mode = optimization.get(
        "group_optimizer_step_mode",
        request.get("group_optimizer_step_mode", "legacy_accumulate_all_groups"),
    )
    selected_steps = int(summary["best_epoch"])
    if mode == "shuffled_step_per_group":
        selected_steps *= len(request["train_batches"])
    return {
        "best_epoch": int(summary["best_epoch"]),
        "selected_artifact_optimizer_steps": selected_steps,
        "optimizer": {
            "group_optimizer_step_mode": mode,
            "group_shuffle_seed": optimization.get(
                "group_shuffle_seed", request.get("group_shuffle_seed")
            ),
            "gradient_clip_norm": optimization.get(
                "gradient_clip_norm", request.get("gradient_clip_norm")
            ),
            "total_optimizer_steps_executed": optimization.get("optimizer_step_count"),
        },
        "selected_checkpoint_metrics": aggregate_checkpoint_metrics(summary),
        "matched_epoch_history": history_at_epoch(summary, matched_epoch),
        "runtime_seconds": summary["runtime_seconds"],
    }


def relative_changes(
    control: dict[str, float], treatment: dict[str, float]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, control_value in control.items():
        treatment_value = treatment[name]
        result[name] = (
            float((treatment_value - control_value) / abs(control_value))
            if control_value != 0
            else None
        )
    return result


def load_h(path: Path) -> np.ndarray:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as payload:
        return np.asarray(payload["global_h_matrix"], dtype=np.complex128)


def main() -> None:
    args = parse_args()
    summaries = {
        "accumulated_control": load_json(args.control_summary),
        "shuffled_unclipped": load_json(args.unclipped_summary),
        "shuffled_clip5": load_json(args.clipped_summary),
    }
    arms = {
        name: summarize_arm(summary, args.matched_epoch)
        for name, summary in summaries.items()
    }
    control_metrics = arms["accumulated_control"]["selected_checkpoint_metrics"]
    comparisons = {
        name: {
            "relative_change_from_accumulated_control": relative_changes(
                control_metrics,
                arm["selected_checkpoint_metrics"],
            )
        }
        for name, arm in arms.items()
        if name != "accumulated_control"
    }

    artifact_paths = {
        "accumulated_control": args.control_artifact,
        "shuffled_unclipped": args.unclipped_artifact,
        "shuffled_clip5": args.clipped_artifact,
    }
    if all(path is not None for path in artifact_paths.values()):
        matrices = {name: load_h(path) for name, path in artifact_paths.items()}
        control_h = matrices["accumulated_control"]
        control_norm = float(np.linalg.norm(control_h))
        for name in ("shuffled_unclipped", "shuffled_clip5"):
            comparisons[name]["relative_h_frobenius_distance_from_control"] = float(
                np.linalg.norm(matrices[name] - control_h) / control_norm
            )

    result = {
        "schema_version": 1,
        "description": (
            "Matched k=4 optimizer-granularity and gradient-clipping ablation "
            "on common frozen training and checkpoint fibres."
        ),
        "matched_epoch": args.matched_epoch,
        "arms": arms,
        "comparisons": comparisons,
        "interpretation": (
            "Negative relative changes improve error metrics; for "
            "min_metric_eigenvalue a positive change is favorable."
        ),
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        output = args.out.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
