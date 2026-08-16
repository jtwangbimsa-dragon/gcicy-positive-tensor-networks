#!/usr/bin/env python3
"""Train a registered gCICY H-metric from one JSON specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    TrainingRequest,
    get_adapter,
    train_h_metric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--artifact-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--allow-failed", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def parse_batches(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of seed/points objects")
    batches = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"seed", "points"}:
            raise ValueError(f"{name} entries must contain exactly seed and points")
        batches.append((int(row["seed"]), int(row["points"])))
    return tuple(batches)


def load_specification(
    path: Path,
    _resolving: tuple[Path, ...] = (),
) -> dict:
    resolved = path.expanduser().resolve()
    if resolved in _resolving:
        cycle = " -> ".join(str(item) for item in (*_resolving, resolved))
        raise ValueError(f"training specification base_spec cycle: {cycle}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    base_value = data.get("base_spec")
    if base_value is None:
        return data
    base_path = resolve_path(base_value, resolved.parent)
    base = load_specification(base_path, (*_resolving, resolved))
    merged = dict(base)
    nested_keys = {"model", "training", "output", "adaptation_provenance"}
    for key, value in data.items():
        if key not in nested_keys and key != "base_spec":
            merged[key] = value
    for key in nested_keys:
        if key in base or key in data:
            merged[key] = {**base.get(key, {}), **data.get(key, {})}
    merged["resolved_base_spec"] = str(base_path)
    return merged


def main() -> None:
    args = parse_args()
    spec_path = args.spec.expanduser().resolve()
    data = load_specification(spec_path)
    if int(data.get("schema_version", -1)) != 1:
        raise SystemExit("training specification must use schema_version 1")
    adapter = get_adapter(str(data["adapter"]))
    model_data = data["model"]
    training = dict(data["training"])
    training["degree"] = tuple(int(value) for value in training["degree"])
    training["check_seeds"] = tuple(int(value) for value in training["check_seeds"])
    training["global_volume_ratio_cvar_tail_fractions"] = tuple(
        float(value)
        for value in training.get("global_volume_ratio_cvar_tail_fractions", ())
    )
    training["global_volume_ratio_cvar_tail_weights"] = tuple(
        float(value)
        for value in training.get("global_volume_ratio_cvar_tail_weights", ())
    )
    training["global_upper_log_ratio_cvar_tail_fractions"] = tuple(
        float(value)
        for value in training.get(
            "global_upper_log_ratio_cvar_tail_fractions", ()
        )
    )
    training["global_upper_log_ratio_cvar_tail_weights"] = tuple(
        float(value)
        for value in training.get("global_upper_log_ratio_cvar_tail_weights", ())
    )
    training["train_batches"] = parse_batches(
        training.get("train_batches"), "train_batches"
    )
    training["validation_batches"] = parse_batches(
        training.get("validation_batches"), "validation_batches"
    )
    training["checkpoint_batches"] = parse_batches(
        training.get("checkpoint_batches"), "checkpoint_batches"
    )
    training["check_batches"] = parse_batches(
        training.get("check_batches"), "check_batches"
    )
    if training.get("initial_artifact") is not None:
        training["initial_artifact"] = resolve_path(
            training["initial_artifact"], spec_path.parent
        )
    if training.get("active_set_path") is not None:
        training["active_set_path"] = resolve_path(
            training["active_set_path"], spec_path.parent
        )
    for key in ("train_common_pool", "selection_common_pool"):
        if training.get(key) is not None:
            training[key] = resolve_path(training[key], spec_path.parent)
    request = TrainingRequest(
        model_seed=int(model_data["seed"]),
        exact_model=bool(model_data.get("exact", True)),
        **training,
    )

    output_data = data.get("output", {})
    configured_artifact = resolve_path(
        output_data.get("artifact", f"../outputs/pipeline/{spec_path.stem}.npz"),
        spec_path.parent,
    )
    configured_summary = resolve_path(
        output_data.get(
            "summary", f"../outputs/pipeline/{spec_path.stem}_summary.json"
        ),
        spec_path.parent,
    )
    artifact_path = (
        args.artifact_out.expanduser().resolve()
        if args.artifact_out is not None
        else configured_artifact
    )
    summary_path = (
        args.summary_out.expanduser().resolve()
        if args.summary_out is not None
        else configured_summary
    )
    summary = train_h_metric(
        adapter,
        request,
        artifact_path=artifact_path,
        summary_path=summary_path,
    )
    print(
        f"adapter={adapter.key}, type={adapter.configuration.type_label}, "
        f"degree={request.degree}, best_epoch={summary['best_epoch']}, "
        f"export sigma={summary['export_candidate']['sigma']:.6e}"
    )
    if not summary["success"] and not args.allow_failed:
        raise SystemExit("training completed but failed its scientific gates")


if __name__ == "__main__":
    main()
