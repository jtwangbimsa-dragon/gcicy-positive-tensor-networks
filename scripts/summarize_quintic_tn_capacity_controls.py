#!/usr/bin/env python3
"""Build a checked comparison of the registered quintic TN capacity controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


METRIC_LABELS = (
    "sigma",
    "chi",
    "q999_abs_residual",
    "cvar99_abs_residual",
    "maximum_abs_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leading-benchmark-report", type=Path, required=True)
    parser.add_argument("--coherent-report", type=Path, required=True)
    parser.add_argument("--positive-report", type=Path, required=True)
    parser.add_argument("--d12-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if "benchmark" not in report:
        raise ValueError(f"report has no benchmark block: {resolved}")
    return resolved, report


def metrics(statistics: dict[str, Any]) -> dict[str, float | int]:
    if statistics is None:
        raise ValueError("capacity-control report omitted its common benchmark")
    return {
        "n_points": int(statistics["n_points"]),
        "sigma": float(statistics["sigma_official_formula"]),
        "chi": float(statistics["weighted_rms_abs_residual"]),
        "q999_abs_residual": float(
            statistics["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99_abs_residual": float(
            statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum_abs_residual": float(
            statistics["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
        "nonpositive_metric_count": int(
            statistics["nonpositive_min_eigenvalue"]["count"]
        ),
    }


def relative_changes(
    reference: dict[str, float | int], candidate: dict[str, float | int]
) -> dict[str, float]:
    return {
        label: float(candidate[label]) / float(reference[label]) - 1.0
        for label in METRIC_LABELS
    }


def accepted_centers(report: dict[str, Any]) -> list[int]:
    rows = report.get("steps")
    if rows is None:
        rows = report.get("optimization", {}).get("rows", [])
    return [int(row["center"]) for row in rows if row.get("accepted")]


def registered_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def provenance(report: dict[str, Any]) -> dict[str, str | None]:
    source = report.get("source", {})
    configuration = report.get("configuration", {})
    blind_dir = registered_path(
        source.get("blind_reference_run_dir")
        or configuration.get("blind_reference_run_dir")
    )
    pullbacks_dir = registered_path(
        source.get("pullbacks_dir") or configuration.get("pullbacks_dir")
    )
    parent_report_hash = None
    derivation = "direct report fields"
    if blind_dir is None or pullbacks_dir is None:
        parent_run = registered_path(configuration.get("run_dir"))
        parent_report_path = None if parent_run is None else parent_run / "report.json"
        if parent_report_path is not None and parent_report_path.exists():
            parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
            parent_configuration = parent_report.get("configuration", {})
            blind_dir = blind_dir or registered_path(
                parent_configuration.get("blind_reference_run_dir")
            )
            pullbacks_dir = pullbacks_dir or registered_path(
                parent_configuration.get("pullbacks_dir")
            )
            parent_report_hash = sha256_file(parent_report_path)
            derivation = "resolved through registered parent run report"
    blind_points = None if blind_dir is None else blind_dir / "blind_points.npz"
    blind_pullbacks = (
        None if pullbacks_dir is None else pullbacks_dir / "blind_pullbacks.npy"
    )
    blind_points_hash = source.get("blind_points_sha256")
    if (
        blind_points_hash is None
        and blind_points is not None
        and blind_points.exists()
    ):
        blind_points_hash = sha256_file(blind_points)
    blind_pullbacks_hash = source.get("blind_pullbacks_sha256")
    if (
        blind_pullbacks_hash is None
        and blind_pullbacks is not None
        and blind_pullbacks.exists()
    ):
        blind_pullbacks_hash = sha256_file(blind_pullbacks)
    return {
        "blind_reference_run_dir": None if blind_dir is None else str(blind_dir),
        "pullbacks_dir": None if pullbacks_dir is None else str(pullbacks_dir),
        "blind_points_sha256": blind_points_hash,
        "blind_pullbacks_sha256": blind_pullbacks_hash,
        "parent_run_report_sha256": parent_report_hash,
        "derivation": derivation,
    }


def tensor_real_parameter_count(value: Any) -> int:
    return int(value.numel() * (2 if value.is_complex() else 1))


def architecture_summary(report: dict[str, Any]) -> dict[str, Any]:
    architecture = report.get("architecture", {})
    active = architecture.get("active_real_parameters")
    fixed = architecture.get("fixed_learned_dictionary_real_parameters")
    stored = architecture.get("stored_learned_real_parameters")
    bond_dimensions = architecture.get("bond_dimensions")
    branch_count = architecture.get("branch_count")
    if active is None:
        artifact_value = report.get("artifacts", {}).get("model")
        artifact_path = Path(artifact_value) if artifact_value else None
        if artifact_path is not None and artifact_path.exists():
            import torch

            artifact = torch.load(
                artifact_path, map_location="cpu", weights_only=False
            )
            if artifact.get("schema") in (
                "quintic-positive-tensor-network-coherent-sum-v1",
                "quintic-positive-tensor-network-positive-sum-v1",
            ):
                states = artifact["branch_states"]
                active = sum(
                    tensor_real_parameter_count(value)
                    for state in states
                    for name, value in state.items()
                    if name.startswith("coefficient_cores.")
                ) + tensor_real_parameter_count(artifact["new_gates"])
                fixed = sum(
                    tensor_real_parameter_count(state["physical_dictionary"])
                    for state in states
                    if "physical_dictionary" in state
                )
                stored = active + fixed
                metadata = artifact.get("branch_metadata", [])
                branch_count = len(states)
                bond_dimensions = [
                    int(item["bond_dimension"]) for item in metadata
                ]
    if bond_dimensions is None:
        bond = architecture.get("bond_dimension")
        bond_dimensions = None if bond is None else [bond]
    return {
        "branch_count": 1 if branch_count is None else branch_count,
        "bond_dimensions": bond_dimensions,
        "active_real_parameters": active,
        "fixed_learned_dictionary_real_parameters": fixed,
        "stored_learned_real_parameters": stored,
    }


def variant_row(label: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    baseline = metrics(report["benchmark"]["baseline"])
    final = metrics(report["benchmark"]["final"])
    return {
        "label": label,
        "report": str(path),
        "report_sha256": sha256_file(path),
        "schema": report.get("schema"),
        "architecture": architecture_summary(report),
        "accepted_centers": accepted_centers(report),
        "timing_seconds": float(report["timing_seconds"]),
        "benchmark_baseline": baseline,
        "benchmark_final": final,
        "relative_change_from_own_initialization": relative_changes(
            baseline, final
        ),
        "provenance": provenance(report),
    }


def validate_common_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    point_counts = {
        int(row["benchmark_final"]["n_points"]) for row in rows
    }
    if len(point_counts) != 1:
        raise RuntimeError(f"benchmark point counts differ: {sorted(point_counts)}")
    for row in rows:
        if row["benchmark_baseline"]["n_points"] not in point_counts:
            raise RuntimeError("baseline and final benchmark point counts differ")

    keys = ("blind_points_sha256", "blind_pullbacks_sha256")
    hash_checks = {}
    missing_hashes = {}
    for key in keys:
        observed = {
            row["provenance"].get(key)
            for row in rows
            if row["provenance"].get(key) is not None
        }
        if len(observed) > 1:
            raise RuntimeError(f"common benchmark provenance differs for {key}")
        hash_checks[key] = next(iter(observed), None)
        missing_hashes[key] = [
            row["label"]
            for row in rows
            if row["provenance"].get(key) is None
        ]
    hashes_complete = all(
        hash_checks[key] is not None and not missing_hashes[key] for key in keys
    )
    return {
        "point_count": next(iter(point_counts)),
        "hash_checks": hash_checks,
        "missing_hashes": missing_hashes,
        "hash_status": (
            "verified"
            if hashes_complete
            else "partial: legacy reports do not contain every direct data hash"
        ),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Quintic TN capacity controls",
        "",
        "All metric columns use the common blind benchmark. Negative deltas are improvements.",
        "",
        "| Model | Stored parameters | Accepted centers | sigma | chi | q99.9 | CVaR99 | max | Delta sigma vs leading | Time (s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        final = row["benchmark_final"]
        architecture = row["architecture"]
        parameters = architecture["stored_learned_real_parameters"]
        centers = ",".join(str(value) for value in row["accepted_centers"]) or "none"
        lines.append(
            "| {label} | {parameters} | {centers} | {sigma:.8g} | {chi:.8g} | "
            "{q999:.8g} | {cvar:.8g} | {maximum:.8g} | {delta:+.3%} | {seconds:.1f} |".format(
                label=row["label"],
                parameters="unknown" if parameters is None else parameters,
                centers=centers,
                sigma=final["sigma"],
                chi=final["chi"],
                q999=final["q999_abs_residual"],
                cvar=final["cvar99_abs_residual"],
                maximum=final["maximum_abs_residual"],
                delta=row["relative_change_from_common_leading"]["sigma"],
                seconds=row["timing_seconds"],
            )
        )
    lines.extend(
        (
            "",
            f"Benchmark points: {summary['common_benchmark']['point_count']}",
            f"Provenance status: {summary['common_benchmark']['hash_status']}",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    leading_path, leading_report = load_report(args.leading_benchmark_report)
    leading = metrics(leading_report["benchmark"]["baseline"])

    rows = []
    for label, report_argument in (
        ("coherent D8+D8", args.coherent_report),
        ("positive-sum D8+D8", args.positive_report),
        ("single-branch D12", args.d12_report),
    ):
        path, report = load_report(report_argument)
        row = variant_row(label, path, report)
        row["relative_change_from_common_leading"] = relative_changes(
            leading, row["benchmark_final"]
        )
        rows.append(row)

    summary = {
        "schema": "quintic-tn-capacity-controls-summary-v1",
        "leading_benchmark": {
            "report": str(leading_path),
            "report_sha256": sha256_file(leading_path),
            "metrics": leading,
        },
        "common_benchmark": validate_common_benchmark(rows),
        "rows": rows,
    }
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)

    if args.markdown_out is not None:
        markdown_path = args.markdown_out.expanduser().resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_markdown = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
        temporary_markdown.write_text(markdown(summary), encoding="utf-8")
        temporary_markdown.replace(markdown_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
