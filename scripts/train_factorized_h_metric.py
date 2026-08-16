#!/usr/bin/env python3
"""Train a two-factor product H metric from one JSON specification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.factorized_h import (  # noqa: E402
    FactorizedHTrainingRequest,
    train_factorized_h_metric,
)
from gcicy_metric.pipeline.registry import get_adapter  # noqa: E402
from scripts.train_gcicy_pipeline import (  # noqa: E402
    load_specification,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--artifact-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--allow-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = args.spec.expanduser().resolve()
    data = load_specification(spec_path)
    if int(data.get("schema_version", -1)) != 1:
        raise SystemExit("factorized training specification must use schema_version 1")
    adapter = get_adapter(str(data["adapter"]))
    model_data = data["model"]
    training = dict(data["training"])
    training["source_artifact"] = resolve_path(
        training["source_artifact"], spec_path.parent
    )
    if training.get("target_degree") is not None:
        training["target_degree"] = tuple(
            int(value) for value in training["target_degree"]
        )
    request = FactorizedHTrainingRequest(
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
    summary = train_factorized_h_metric(
        adapter,
        request,
        artifact_path=artifact_path,
        summary_path=summary_path,
    )
    export = summary["export"]["selected_factorized"]
    print(
        f"adapter={adapter.key}, source_degree={summary['source']['degree']}, "
        f"target_degree={summary['target']['degree']}, "
        f"best_epoch={summary['training']['best_epoch']}, "
        f"export_sigma={export['sigma']:.6e}, "
        f"implementation_gate={summary['gates']['implementation_passed']}"
    )
    if not summary["success"] and not args.allow_failed:
        raise SystemExit(
            "factorized training completed but failed its configured gates"
        )


if __name__ == "__main__":
    main()
