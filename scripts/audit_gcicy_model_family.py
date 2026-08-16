#!/usr/bin/env python3
"""Aggregate exact smoothness and fresh-seed k=1 metric evidence across models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DEFAULT_ARTIFACTS = ",".join(
    [
        str(ROOT / "outputs" / "gcicy_generic_global_h_metric_weighted.npz"),
        *[
            str(ROOT / "outputs" / f"gcicy_model_{seed}_k1_weighted.npz")
            for seed in range(20260712, 20260716)
        ],
    ]
)
DEFAULT_AUDITS = ",".join(
    [
        str(ROOT / "outputs" / "gcicy_generic_global_h_metric_weighted_audit.json"),
        *[
            str(ROOT / "outputs" / f"gcicy_model_{seed}_k1_weighted_audit.json")
            for seed in range(20260712, 20260716)
        ],
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=ROOT / "outputs" / "gcicy_exact_model_screen.json")
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--audits", default=DEFAULT_AUDITS)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "gcicy_model_family_k1_audit.json",
    )
    return parser.parse_args()


def parse_paths(text: str) -> list[Path]:
    return [Path(value.strip()).expanduser().resolve() for value in text.split(",") if value.strip()]


def coefficient_fingerprint(p1_coefficients: np.ndarray, p2_tensor: np.ndarray) -> str:
    coefficients = np.concatenate(
        [
            np.rint(np.asarray(p1_coefficients).real).astype("<i8"),
            np.rint(np.asarray(p2_tensor).real).astype("<i8").ravel(),
        ]
    )
    return hashlib.sha256(coefficients.tobytes()).hexdigest()


def aggregate(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        "mean": mean,
        "standard_error_across_models": standard_error,
        "normal_95_percent_ci": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    if not args.screen.exists():
        raise SystemExit(f"missing exact-model screen: {args.screen}")
    artifact_paths = parse_paths(args.artifacts)
    audit_paths = parse_paths(args.audits)
    if len(artifact_paths) != len(audit_paths) or not artifact_paths:
        raise SystemExit("artifact and audit lists must have the same nonzero length")

    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    screen_by_seed = {row["model_seed"]: row for row in screen["model_rows"]}
    model_rows = []
    seen_seeds = set()
    for artifact_path, audit_path in zip(artifact_paths, audit_paths, strict=True):
        if not artifact_path.exists() or not audit_path.exists():
            raise SystemExit(f"missing artifact/audit pair: {artifact_path}, {audit_path}")
        artifact = np.load(artifact_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        seed = int(artifact["generic_model_seed"])
        if seed in seen_seeds:
            raise SystemExit(f"duplicate model seed: {seed}")
        seen_seeds.add(seed)
        if audit["model_seed"] != seed:
            raise SystemExit(f"artifact/audit model mismatch for seed {seed}")
        if tuple(int(value) for value in artifact["global_section_degree"]) != (1, 1, 1):
            raise SystemExit(f"family artifact is not k=1: {artifact_path}")
        if "exact_integer_coefficients" not in artifact.files or not bool(
            artifact["exact_integer_coefficients"]
        ):
            raise SystemExit(f"family artifact is not marked exact: {artifact_path}")
        if seed not in screen_by_seed:
            raise SystemExit(f"model seed is absent from exact screen: {seed}")
        screen_row = screen_by_seed[seed]
        fingerprint = coefficient_fingerprint(artifact["p1_coefficients"], artifact["p2_tensor"])
        if fingerprint != screen_row["coefficient_sha256"]:
            raise SystemExit(f"coefficient fingerprint mismatch for seed {seed}")
        if not screen_row["accepted_all_characteristics"]:
            raise SystemExit(f"model seed did not pass the exact screen: {seed}")

        audit_rows = audit["rows"]
        if not audit["all_passed"] or not all(row["passed"] for row in audit_rows):
            raise SystemExit(f"fresh-seed metric audit failed for seed {seed}")
        if audit["projective_charts_seen"] != 24 or audit["max_trained_h_chart_ma_error"] > 1e-8:
            raise SystemExit(f"atlas metric audit failed for seed {seed}")
        if audit["min_sampled_jacobian_singular_value"] <= 1e-7:
            raise SystemExit(f"sampled Jacobian gate failed for seed {seed}")

        baseline = [row["baseline_rms"] for row in audit_rows]
        candidate = [row["candidate_rms"] for row in audit_rows]
        baseline_weighted = [row["baseline_weighted_rms"] for row in audit_rows]
        candidate_weighted = [row["candidate_weighted_rms"] for row in audit_rows]
        model_rows.append(
            {
                "model_seed": seed,
                "coefficient_sha256": fingerprint,
                "artifact": str(artifact_path),
                "audit": str(audit_path),
                "characteristics_passed": screen_row["characteristics_passed"],
                "fresh_seed_sets": len(audit_rows),
                "mean_baseline_rms": float(np.mean(baseline)),
                "mean_candidate_rms": float(np.mean(candidate)),
                "mean_unweighted_reduction_fraction": float(1.0 - np.mean(candidate) / np.mean(baseline)),
                "mean_baseline_weighted_rms": float(np.mean(baseline_weighted)),
                "mean_candidate_weighted_rms": float(np.mean(candidate_weighted)),
                "mean_weighted_reduction_fraction": float(
                    1.0 - np.mean(candidate_weighted) / np.mean(baseline_weighted)
                ),
                "max_candidate_rms": float(np.max(candidate)),
                "max_candidate_weighted_rms": float(np.max(candidate_weighted)),
                "min_metric_eigenvalue": audit["min_candidate_eigenvalue"],
                "min_sampled_jacobian_singular_value": audit[
                    "min_sampled_jacobian_singular_value"
                ],
                "max_chart_ma_error": audit["max_trained_h_chart_ma_error"],
                "all_fresh_seeds_passed": True,
            }
        )

    model_rows.sort(key=lambda row: row["model_seed"])
    expected_screen_seeds = set(screen["accepted_model_seeds"])
    if seen_seeds != expected_screen_seeds:
        raise SystemExit(
            "family artifacts do not exactly cover the accepted exact-model screen: "
            f"artifacts={sorted(seen_seeds)}, screen={sorted(expected_screen_seeds)}"
        )

    summary = {
        "description": "Cross-model k=1 exact-smoothness, fresh-seed, positivity, and atlas audit.",
        "configuration": screen["configuration"],
        "model_count": len(model_rows),
        "model_seeds": [row["model_seed"] for row in model_rows],
        "characteristics_per_model": len(screen["characteristics"]),
        "charts_per_model_characteristic": screen["charts_per_model_prime"],
        "exact_chart_certificates": (
            len(model_rows) * len(screen["characteristics"]) * screen["charts_per_model_prime"]
        ),
        "fresh_seed_sets_per_model": sorted({row["fresh_seed_sets"] for row in model_rows}),
        "total_fresh_seed_sets": sum(row["fresh_seed_sets"] for row in model_rows),
        "all_models_passed": all(row["all_fresh_seeds_passed"] for row in model_rows),
        "candidate_rms_across_models": aggregate(
            [row["mean_candidate_rms"] for row in model_rows]
        ),
        "candidate_weighted_rms_across_models": aggregate(
            [row["mean_candidate_weighted_rms"] for row in model_rows]
        ),
        "unweighted_reduction_fraction_across_models": aggregate(
            [row["mean_unweighted_reduction_fraction"] for row in model_rows]
        ),
        "weighted_reduction_fraction_across_models": aggregate(
            [row["mean_weighted_reduction_fraction"] for row in model_rows]
        ),
        "minimum_metric_eigenvalue": float(
            min(row["min_metric_eigenvalue"] for row in model_rows)
        ),
        "minimum_sampled_jacobian_singular_value": float(
            min(row["min_sampled_jacobian_singular_value"] for row in model_rows)
        ),
        "maximum_chart_ma_error": float(max(row["max_chart_ma_error"] for row in model_rows)),
        "model_rows": model_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in model_rows:
        print(
            f"seed {row['model_seed']}: rms={row['mean_candidate_rms']:.6f}, "
            f"weighted={row['mean_candidate_weighted_rms']:.6f}, "
            f"reductions={row['mean_unweighted_reduction_fraction']:.1%}/"
            f"{row['mean_weighted_reduction_fraction']:.1%}"
        )
    print(
        f"accepted models={summary['model_count']}, exact charts={summary['exact_chart_certificates']}, "
        f"fresh seed sets={summary['total_fresh_seed_sets']}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
