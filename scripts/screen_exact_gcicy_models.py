#!/usr/bin/env python3
"""Screen several exact integer gCICY models by complete finite-field atlases."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from check_exact_smoothness import check_chart  # noqa: E402
from gcicy_metric import all_projective_charts, make_exact_generic_model  # noqa: E402


def parse_ints(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-seeds",
        type=parse_ints,
        default=parse_ints("20260711,20260712,20260713,20260714,20260715"),
    )
    parser.add_argument(
        "--characteristics",
        type=parse_ints,
        default=parse_ints("31991,32003,65521"),
    )
    parser.add_argument("--coefficient-bound", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--algorithm", choices=("std", "slimgb"), default="slimgb")
    parser.add_argument("--singular", default="/opt/homebrew/bin/Singular")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "gcicy_exact_model_screen.json",
    )
    return parser.parse_args()


def coefficient_fingerprint(model) -> str:
    coefficients = np.concatenate(
        [
            np.rint(model.p1_coefficients.real).astype("<i8"),
            np.rint(model.p2_tensor.real).astype("<i8").ravel(),
        ]
    )
    return hashlib.sha256(coefficients.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    singular = Path(args.singular)
    if not singular.exists():
        raise SystemExit(f"Singular executable not found: {singular}")
    if args.coefficient_bound < 1:
        raise SystemExit("coefficient bound must be positive")
    if len(set(args.model_seeds)) != len(args.model_seeds):
        raise SystemExit("model seeds must be distinct")
    if len(set(args.characteristics)) != len(args.characteristics):
        raise SystemExit("characteristics must be distinct")

    models = {
        seed: make_exact_generic_model(seed, coefficient_bound=args.coefficient_bound)
        for seed in args.model_seeds
    }
    charts = all_projective_charts()
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {}
        for seed, model in models.items():
            for characteristic in args.characteristics:
                for chart in charts:
                    future = executor.submit(
                        check_chart,
                        str(singular),
                        model,
                        chart,
                        args.timeout,
                        characteristic,
                        args.algorithm,
                    )
                    futures[future] = (seed, characteristic, chart)
        for future in as_completed(futures):
            seed, characteristic, _ = futures[future]
            row = future.result()
            row["model_seed"] = seed
            row["characteristic"] = characteristic
            rows.append(row)

    rows.sort(key=lambda row: (row["model_seed"], row["characteristic"], row["chart"]))
    model_rows = []
    for seed in args.model_seeds:
        prime_rows = []
        for characteristic in args.characteristics:
            selected = [
                row
                for row in rows
                if row["model_seed"] == seed and row["characteristic"] == characteristic
            ]
            dimensions = [row["variety_dimension"] for row in selected]
            unexpected_dimensions = sorted(
                {dimension for dimension in dimensions if dimension not in (None, -1, 3)}
            )
            accepted = (
                len(selected) == len(charts)
                and all(row["smooth"] for row in selected)
                and not any(row["timed_out"] for row in selected)
                and any(dimension == 3 for dimension in dimensions)
                and not unexpected_dimensions
            )
            prime_rows.append(
                {
                    "characteristic": characteristic,
                    "accepted": accepted,
                    "charts_proved_smooth": sum(row["smooth"] for row in selected),
                    "charts_timed_out": sum(row["timed_out"] for row in selected),
                    "dimension_three_charts": sum(dimension == 3 for dimension in dimensions),
                    "empty_charts": sum(dimension == -1 for dimension in dimensions),
                    "unresolved_dimensions": sum(dimension is None for dimension in dimensions),
                    "unexpected_dimensions": unexpected_dimensions,
                    "total_elapsed_seconds": float(sum(row["elapsed_seconds"] for row in selected)),
                    "max_chart_elapsed_seconds": float(
                        max((row["elapsed_seconds"] for row in selected), default=0.0)
                    ),
                }
            )
        model = models[seed]
        model_rows.append(
            {
                "model_seed": seed,
                "coefficient_sha256": coefficient_fingerprint(model),
                "coefficient_bound": args.coefficient_bound,
                "p1_term_count": int(len(model.p1_coefficients)),
                "p2_term_count": int(model.p2_tensor.size),
                "accepted_all_characteristics": all(row["accepted"] for row in prime_rows),
                "characteristics_passed": sum(row["accepted"] for row in prime_rows),
                "prime_rows": prime_rows,
            }
        )
        status = "accepted" if model_rows[-1]["accepted_all_characteristics"] else "rejected"
        print(
            f"seed {seed}: {status}, "
            f"primes={model_rows[-1]['characteristics_passed']}/{len(args.characteristics)}, "
            f"sha256={model_rows[-1]['coefficient_sha256'][:12]}"
        )

    summary = {
        "description": (
            "Complete 24-chart finite-field smoothness and dimension screen for exact integer gCICY models."
        ),
        "configuration": [[1, 1, -1, 1], [1, 1, 1, -1], [3, 1, 1, 1]],
        "model_seeds": args.model_seeds,
        "characteristics": args.characteristics,
        "coefficient_bound": args.coefficient_bound,
        "algorithm": args.algorithm,
        "charts_per_model_prime": len(charts),
        "models_requested": len(args.model_seeds),
        "models_accepted_all_characteristics": sum(
            row["accepted_all_characteristics"] for row in model_rows
        ),
        "accepted_model_seeds": [
            row["model_seed"] for row in model_rows if row["accepted_all_characteristics"]
        ],
        "model_rows": model_rows,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
