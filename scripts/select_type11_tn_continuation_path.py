#!/usr/bin/env python3
"""Select one X11 TN continuation path before the final independent evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PATHS = (
    "direct_d13",
    "direct_gentle_d13",
    "nested_d8_d10_d13",
    "nested_fine_d7_d9_d11_d13",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path_name in DEFAULT_PATHS:
        directory = args.run_root / path_name / f"replicate_{args.replicate}"
        summary_path = directory / "plateau_summary.json"
        confirmation_path = directory / "development_confirmation.json"
        marker = directory / "PATH_FINISHED"
        if not summary_path.is_file() or not confirmation_path.is_file() or not marker.is_file():
            raise FileNotFoundError(f"incomplete continuation path: {directory}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        validation = summary["best_validation"]
        validation_ma = validation["compressed_ma_errors"]
        confirmation_metrics = confirmation["metrics"]
        rows.append(
            {
                "path": path_name,
                "model": str(directory / "plateau.pt"),
                "trainable_real_parameter_count": int(
                    summary["trainable_real_parameter_count"]
                ),
                "validation_selection_score": float(
                    summary["best_validation_selection_score"]
                ),
                "validation_sigma": float(validation_ma["sigma"]),
                "validation_chi": float(validation_ma["sqrt_squared_energy"]),
                "validation_q999": float(
                    validation_ma["positive_log_ratio_q999"]
                ),
                "validation_cvar_1pct": float(
                    validation_ma["positive_log_ratio_cvar_1pct"]
                ),
                "development_confirmation": confirmation_metrics,
            }
        )

    # The path is selected only with the training-time validation score. The
    # repeatedly inspected 49,152-point set is retained as a development audit
    # and does not enter this choice.
    selected = min(rows, key=lambda row: row["validation_selection_score"])
    result = {
        "schema": "type11-tn-continuation-path-selection-v1",
        "status": "development_choice_frozen_before_new_200k_evaluation",
        "selection_rule": "minimum best_validation_selection_score on the common 24576-point validation sample",
        "development_confirmation_used_for_selection": False,
        "replicate_used_for_path_development": int(args.replicate),
        "selected_path": selected["path"],
        "selected_model": selected["model"],
        "candidates": rows,
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(selected["path"])


if __name__ == "__main__":
    main()
