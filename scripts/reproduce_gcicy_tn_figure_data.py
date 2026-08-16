#!/usr/bin/env python3
"""Reconstruct the numerical data plotted in the two manuscript figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent if (SCRIPT.parent / "outputs").is_dir() else SCRIPT.parents[1]
X11 = ROOT / "outputs/pipeline/type11_x11_equal_time_final_20260807"
X21 = ROOT / "outputs/pipeline/type21_kd_plateau_paper_summary_20260802.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_log_ratio(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as arrays:
        log_eta = np.asarray(arrays["model_log_eta"], dtype=np.float64)
        weights = np.asarray(arrays["importance_weights"], dtype=np.float64)
    weights = weights / weights.sum()
    shift = float(np.max(log_eta))
    log_mean = shift + np.log(np.sum(weights * np.exp(log_eta - shift)))
    return log_eta - log_mean, weights


def survival(values: np.ndarray, weights: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    suffix = np.cumsum(weights[order][::-1])[::-1]
    indices = np.searchsorted(sorted_values, grid, side="left")
    result = np.zeros_like(grid)
    valid = indices < len(sorted_values)
    result[valid] = suffix[indices[valid]]
    return result


def x11_figure_data() -> dict[str, Any]:
    paths = {
        "positive_TN": [
            X11 / f"replicate_{replicate}/tn_arrays.npz"
            for replicate in (1, 2, 3)
        ],
        "source_density_residual_phi": [
            X11
            / f"replicate_{replicate}/source_density_phi/blind_tail_arrays.npz"
            for replicate in (1, 2, 3)
        ],
    }
    loaded = {
        model: [normalized_log_ratio(path) for path in model_paths]
        for model, model_paths in paths.items()
    }
    largest = max(
        float(np.max(np.abs(log_ratio)))
        for model_records in loaded.values()
        for log_ratio, _ in model_records
    )
    grid = np.linspace(0.0, max(0.9, largest * 1.03), 360)
    panels: dict[str, Any] = {}
    for side in ("upper", "lower"):
        panels[side] = {}
        for model, model_records in loaded.items():
            curves = []
            for log_ratio, weights in model_records:
                values = np.maximum(log_ratio if side == "upper" else -log_ratio, 0.0)
                curves.append(survival(values, weights, grid))
            stacked = np.stack(curves)
            panels[side][model] = {
                "replicates": stacked.tolist(),
                "mean": stacked.mean(axis=0).tolist(),
                "minimum": stacked.min(axis=0).tolist(),
                "maximum": stacked.max(axis=0).tolist(),
            }
    return {
        "grid": grid.tolist(),
        "panels": panels,
        "sources": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for model_paths in paths.values()
            for path in model_paths
        },
    }


def x21_figure_data() -> dict[str, Any]:
    report = json.loads(X21.read_text(encoding="utf-8"))
    rows = report["rows"]
    plotted = {
        key: {
            "k": int(row["k"]),
            "D": int(row["D"]),
            "trainable_real_parameters": int(row["trainable_real_parameters"]),
            "development_1000_sigma": 1000.0 * float(row["metrics"]["sigma"]),
        }
        for key, row in sorted(rows.items())
    }
    arrows = [
        ["direct_k8_d10", "nested_d_k8"],
        ["direct_k16_d10", "nested_d_k16"],
        ["direct_k8_d10", "nested_k_d10"],
    ]
    return {
        "rows": plotted,
        "arrows": arrows,
        "source": X21.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(X21),
    }


def payload() -> dict[str, Any]:
    return {
        "schema": "gcicy-tn-manuscript-figure-data-v1",
        "figure_1_x11_weighted_survival": x11_figure_data(),
        "figure_2_x21_path_dependence": x21_figure_data(),
    }


def compare(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise SystemExit(f"figure-data key mismatch at {path}")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise SystemExit(f"figure-data length mismatch at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected), rel_tol=5e-13, abs_tol=5e-15
        ):
            raise SystemExit(f"figure-data numerical mismatch at {path}")
    elif actual != expected:
        raise SystemExit(f"figure-data value mismatch at {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = payload()
    if args.check is not None:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        compare(result, expected)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "reconstructed Figure 1 (2 panels, 2 models, 3 runs, 360 thresholds) "
        "and Figure 2 (11 points, 3 continuation arrows)"
    )
    if args.check is not None:
        print("figure data agree with the frozen reference")


if __name__ == "__main__":
    main()
