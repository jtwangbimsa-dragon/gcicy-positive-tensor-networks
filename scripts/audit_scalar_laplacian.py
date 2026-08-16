#!/usr/bin/env python3
"""Audit low scalar-Laplacian Ritz spectra for saved gCICY metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, run_scalar_laplacian_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Write output and return success even if a scientific gate fails",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def main() -> None:
    args = parse_args()
    spec_path = args.spec.expanduser().resolve()
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError("scalar spectrum specification must use schema_version 1")
    base = spec_path.parent
    model = data.get("model", {})
    spectrum = data.get("spectrum", {})
    adapter = get_adapter(str(data["adapter"]))
    summary = run_scalar_laplacian_audit(
        adapter,
        model_seed=int(model["seed"]),
        exact_model=bool(model.get("exact", True)),
        artifact_paths=[resolve_path(path, base) for path in data["artifacts"]],
        seeds=[int(seed) for seed in spectrum["seeds"]],
        points_per_seed=int(spectrum["points_per_seed"]),
        trial_levels=[int(level) for level in spectrum.get("trial_levels", [1, 2])],
        eigenvalue_count=int(spectrum.get("eigenvalue_count", 12)),
        mass_relative_threshold=float(
            spectrum.get("mass_relative_threshold", 1e-10)
        ),
        mass_threshold_sweep=[
            float(value) for value in spectrum.get("mass_threshold_sweep", [])
        ],
        minimum_effective_sample_size=float(
            spectrum.get("minimum_effective_sample_size", 1.0)
        ),
        integration_measure=str(
            spectrum.get("integration_measure", "metric_volume")
        ),
        cubic_feature_count=int(spectrum.get("cubic_feature_count", 256)),
        cubic_feature_seed=int(spectrum.get("cubic_feature_seed", 314159)),
        minimum_ess_per_retained_rank=float(
            spectrum.get("minimum_ess_per_retained_rank", 1.0)
        ),
        minimum_cluster_ess_per_retained_rank=float(
            spectrum.get("minimum_cluster_ess_per_retained_rank", 1.0)
        ),
        cluster_jackknife_groups=int(
            spectrum.get("cluster_jackknife_groups", 0)
        ),
        cluster_jackknife_level=(
            int(spectrum["cluster_jackknife_level"])
            if "cluster_jackknife_level" in spectrum
            else None
        ),
        cluster_jackknife_eigenvalue_count=int(
            spectrum.get("cluster_jackknife_eigenvalue_count", 3)
        ),
        maximum_cluster_jackknife_relative_standard_error=(
            float(spectrum["maximum_cluster_jackknife_relative_standard_error"])
            if "maximum_cluster_jackknife_relative_standard_error" in spectrum
            else None
        ),
        progress=True,
    )
    summary["specification"] = str(spec_path)
    configured_output = data.get(
        "output",
        f"../outputs/pipeline/{spec_path.stem}.json",
    )
    output = (
        args.out.expanduser().resolve()
        if args.out is not None
        else resolve_path(configured_output, base)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for artifact in summary["artifacts"]:
        print(f"{artifact['key']} degree={artifact['degree']}")
        for level in artifact["trial_levels"]:
            gap = level["mean_spectral_gap"]
            interval = level["spectral_gap_95_percent_ci"]
            print(
                f"  level={level['level']} raw={level['raw_feature_count']} "
                f"rank={level['retained_trial_ranks']} lambda1={gap:.8g} "
                f"CI=[{interval[0]:.8g}, {interval[1]:.8g}]"
            )
    print(f"success={summary['success']}")
    print(f"wrote {output}")
    if not summary["success"] and not args.allow_failed:
        failed = ", ".join(key for key, value in summary["gates"].items() if not value)
        raise SystemExit(f"scalar spectrum audit failed gates: {failed}")


if __name__ == "__main__":
    main()
