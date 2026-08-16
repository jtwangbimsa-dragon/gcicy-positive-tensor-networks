#!/usr/bin/env python3
"""Run a configuration-driven gCICY metric audit from one JSON specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    AuditRequest,
    AuditThresholds,
    available_adapters,
    get_adapter,
    run_pipeline_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, help="Pipeline JSON specification")
    parser.add_argument("--out", type=Path, help="Override the output path in the specification")
    parser.add_argument("--list-adapters", action="store_true")
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Write the audit but return success even when a scientific gate fails",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_request(spec_path: Path) -> tuple[object, AuditRequest, Path, dict]:
    resolved_spec = spec_path.expanduser().resolve()
    data = json.loads(resolved_spec.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError("pipeline specification must use schema_version 1")
    adapter = get_adapter(str(data["adapter"]))
    model_data = data.get("model", {})
    audit_data = data.get("audit", {})
    threshold_data = data.get("thresholds", {})
    base = resolved_spec.parent
    artifacts = tuple(resolve_path(path, base) for path in data.get("artifacts", []))
    request = AuditRequest(
        model_seed=int(model_data["seed"]),
        exact_model=bool(model_data.get("exact", True)),
        artifact_paths=artifacts,
        seeds=tuple(int(seed) for seed in audit_data["seeds"]),
        points_per_seed=int(audit_data["points_per_seed"]),
        atlas_points_per_seed=int(audit_data.get("atlas_points_per_seed", 1)),
        minimum_selected_coordinate=float(
            audit_data.get("minimum_selected_coordinate", 1e-4)
        ),
        sampling_workers=int(audit_data.get("sampling_workers", 1)),
        sampling_cluster_size=int(audit_data.get("sampling_cluster_size", 1)),
        sampling_backend=str(audit_data.get("sampling_backend", "process")),
        thresholds=AuditThresholds(**threshold_data),
    )
    configured_output = data.get("output", f"../outputs/pipeline/{resolved_spec.stem}.json")
    output = resolve_path(configured_output, base)
    return adapter, request, output, data


def main() -> None:
    args = parse_args()
    if args.list_adapters:
        for key in available_adapters():
            adapter = get_adapter(key)
            print(f"{key}: type {adapter.configuration.type_label} - {adapter.configuration.name}")
        return
    if args.spec is None:
        raise SystemExit("--spec is required unless --list-adapters is used")

    adapter, request, configured_output, _ = load_request(args.spec)
    output = args.out.expanduser().resolve() if args.out is not None else configured_output
    summary = run_pipeline_audit(adapter, request)
    summary["specification"] = str(args.spec.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(
        f"adapter={adapter.key}, type={adapter.configuration.type_label}, "
        f"baseline sigma={summary['baseline']['mean_sigma']:.6e}"
    )
    for artifact in summary["artifacts"]:
        relative = artifact["paired_sigma_improvement"][
            "relative_improvement_against_mean_baseline"
        ]
        print(
            f"{Path(artifact['path']).name}: sigma={artifact['mean_sigma']:.6e}, "
            f"paired improvement={100.0 * relative:.3f}%, "
            f"min eig={artifact['min_metric_eigenvalue']:.3e}"
        )
    for comparison in summary["artifact_pairwise_sigma_comparisons"]:
        relative = comparison["relative_improvement_against_mean_baseline"]
        print(
            f"{comparison['source_artifact']} -> {comparison['target_artifact']}: "
            f"paired sigma improvement={100.0 * relative:.3f}%, "
            f"improved seeds={comparison['improved_seed_count']}/"
            f"{comparison['seed_count']}"
        )
    atlas = summary["atlas"]
    print(
        f"atlas: projective {atlas['projective_charts_seen']}/"
        f"{atlas['projective_charts_expected']}, implicit "
        f"{atlas['implicit_coordinate_choices_seen']}/"
        f"{atlas['implicit_coordinate_choices_expected']}"
    )
    print(f"success={summary['success']}")
    print(f"wrote {output}")
    if not summary["success"] and not args.allow_failed:
        failed = ", ".join(key for key, value in summary["gates"].items() if not value)
        raise SystemExit(f"pipeline audit failed gates: {failed}")


if __name__ == "__main__":
    main()
