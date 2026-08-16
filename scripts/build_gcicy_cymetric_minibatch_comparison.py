#!/usr/bin/env python3
"""Build the type-(1,1) gCICY mini-batch mechanism comparison report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "outputs/pipeline/gcicy_cymetric_minibatch_mechanism_20260716"
)
SUMMARY_SUFFIX = "_summary.json"
CLUSTER_SIZE = 4

ARMS = (
    (
        "full_h_32k",
        "Full H, 32k points",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_pilot",
    ),
    (
        "full_h_131k",
        "Full H, 131k points",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_n131072_u6144",
    ),
    (
        "spectrum_w5",
        "Full H + spectrum w=5",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_specreg_w5",
    ),
    (
        "spectrum_w20",
        "Full H + spectrum w=20",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_specreg_w20",
    ),
    (
        "spectrum_w5_global_cvar",
        "Spectrum w=5 + global CVaR",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_specreg_w5_globalcvar",
    ),
    (
        "spectrum_w5_global_cvar_replay64",
        "Spectrum w=5 + CVaR + 64 replay batches",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_specreg_w5_globalcvar_replay64",
    ),
    (
        "eigen_diagonal",
        "Fixed eigenvectors, 273 parameters",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_eigendiag_e20",
    ),
    (
        "low_rank_32",
        "Reference low rank 32",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_lowrank32",
    ),
    (
        "full_h_lr1e4",
        "Full H, lr=1e-4, 50 epochs",
        "p4p1_type11_hirzebruch_k4_cymetric_minibatch_cold_lr1e4_e50",
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def optional_number(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key)
    return default if value is None else float(value)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "sigma": float(metrics["sigma"]),
        "volume_ratio_l2": float(metrics["sqrt_squared_energy"]),
        "ratio_min": float(metrics["normalized_ratio_min"]),
        "ratio_max": float(metrics["normalized_ratio_max"]),
        "ratio_above_3_weighted_mass": float(
            metrics["normalized_ratio_above_3_weighted_mass"]
        ),
    }


def summarize_arm(
    root: Path,
    arm_id: str,
    label: str,
    stem: str,
) -> dict[str, Any]:
    summary_path = root / "outputs/pipeline" / f"{stem}{SUMMARY_SUFFIX}"
    artifact_path = root / "outputs/pipeline" / f"{stem}.npz"
    summary = load_json(summary_path)
    request = summary["request"]
    optimization = summary["optimization"]
    checkpoints = [
        value["candidate"] for value in summary["checkpoint_validation"].values()
    ]
    if not checkpoints:
        raise ValueError(f"{summary_path} contains no independent checkpoints")

    log_span = float(
        optimization["selected_relative_h_update"][
            "relative_to_initial_log_eigenvalue_span"
        ]
    )
    cluster_count = int(optimization["training_cluster_count"])
    batch_clusters = int(optimization["clusters_per_optimizer_step"])
    cvar = optimization.get("global_volume_ratio_cvar") or {}
    replay_batches = int(optimization.get("cluster_tail_replay_batches_per_epoch") or 0)

    return {
        "id": arm_id,
        "label": label,
        "source_summary": str(summary_path.relative_to(root)),
        "source_artifact": (
            str(artifact_path.relative_to(root)) if artifact_path.exists() else None
        ),
        "configuration": {
            "parameterization": request["h_parameterization"],
            "coordinate_system": optimization["h_coordinate_system"],
            "real_parameter_count": int(optimization["parameterization_real_dimension"]),
            "training_clusters": cluster_count,
            "training_points": cluster_count * CLUSTER_SIZE,
            "clusters_per_step": batch_clusters,
            "points_per_step": int(optimization["nominal_points_per_optimizer_step"]),
            "epochs": int(request["epochs"]),
            "learning_rate": float(request["learning_rate"]),
            "optimizer_steps": int(optimization["optimizer_step_count"]),
            "relative_log_spectrum_weight": optional_number(
                optimization, "relative_log_spectrum_loss_weight"
            ),
            "global_cvar_weight": optional_number(cvar, "loss_weight"),
            "global_cvar_tail_fractions": [
                float(value) for value in cvar.get("tail_fractions", [])
            ],
            "tail_replay_batches_per_epoch": replay_batches,
        },
        "selection": {
            "best_epoch": int(summary["best_epoch"]),
            "pipeline_success": bool(summary["success"]),
            "passed_internal_gates": bool(summary["passed_internal_gates"]),
            "pipeline_success_interpretation": (
                "Exploratory runs deliberately used non-publication checkpoint "
                "policies; false is not a numerical crash."
            ),
        },
        "export": compact_metrics(summary["export_candidate"]),
        "independent_checkpoints": {
            "count": len(checkpoints),
            "mean_sigma": sum(float(value["sigma"]) for value in checkpoints)
            / len(checkpoints),
            "maximum_volume_ratio_l2": max(
                float(value["sqrt_squared_energy"]) for value in checkpoints
            ),
            "maximum_ratio": max(
                float(value["normalized_ratio_max"]) for value in checkpoints
            ),
            "maximum_ratio_above_3_weighted_mass": max(
                float(value["normalized_ratio_above_3_weighted_mass"])
                for value in checkpoints
            ),
        },
        "relative_h_spectrum": {
            "centered_log_eigenvalue_span": log_span,
            "condition_number": math.exp(log_span),
            "basis_covariant": True,
            "scale_invariant": True,
        },
        "runtime_seconds": {
            key: float(value) for key, value in summary["runtime_seconds"].items()
        },
    }


def phi_benchmark(root: Path) -> dict[str, Any]:
    path = root / "outputs/cymetric_phi_gcicy_type11_pilot_8k_e20_seed720xx/report.json"
    report = load_json(path)
    metrics = report["blind_test"]["trained_phi_self_normalized"]
    old_h = report["blind_test"]["current_k4_h_self_normalized"]
    return {
        "source": str(path.relative_to(root)),
        "method": report["scientific_scope"]["method"],
        "training_points": int(report["configuration"]["train_points"]),
        "batch_size": int(report["configuration"]["batch_size"]),
        "epochs": int(report["configuration"]["epochs"]),
        "parameter_count": int(report["network"]["parameter_count"]),
        "blind": {
            "sigma": float(metrics["sigma"]),
            "volume_ratio_l2": float(metrics["chi_l2"]),
            "ratio_min": float(metrics["min_ratio"]),
            "ratio_max": float(metrics["max_ratio"]),
            "ratio_above_3_weighted_mass": float(
                metrics["ratio_upper_tails"]["3.0"]["weighted_mass"]
            ),
        },
        "same_points_old_h": {
            "sigma": float(old_h["sigma"]),
            "volume_ratio_l2": float(old_h["chi_l2"]),
            "ratio_max": float(old_h["max_ratio"]),
        },
        "comparison_limit": (
            "This Phi pilot uses a different parameterization and training budget. "
            "It is a mechanism benchmark, not a matched final-accuracy comparison."
        ),
    }


def markdown_table(arms: list[dict[str, Any]]) -> str:
    rows = [
        "| Arm | Train points | Updates | Export sigma | Check max L2 | "
        "Check max r | mass(r>3) | relative cond |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        config = arm["configuration"]
        export = arm["export"]
        check = arm["independent_checkpoints"]
        spectrum = arm["relative_h_spectrum"]
        rows.append(
            f"| {arm['label']} | {config['training_points']:,} | "
            f"{config['optimizer_steps']:,} | {export['sigma']:.4f} | "
            f"{check['maximum_volume_ratio_l2']:.3f} | "
            f"{check['maximum_ratio']:.2f} | "
            f"{check['maximum_ratio_above_3_weighted_mass']:.3%} | "
            f"{spectrum['condition_number']:.3g} |"
        )
    return "\n".join(rows)


def render_markdown(payload: dict[str, Any]) -> str:
    phi = payload["phi_benchmark"]
    phi_metrics = phi["blind"]
    return f"""# gCICY transfer of the cymetric-style mini-batch mechanism

## Controlled change

The ordinary-quintic repair was transferred to the direct type-(1,1) gCICY
in `P4 x P1`, configuration `((1,4),(3,-1))`, at polarization degree `(4,4)`.
Each optimizer step contains 16 complete four-point fibres, hence 64 points.
The primary loss uses a fixed training-pool normalization, shuffled fibre
batches, Adam, and global gradient clipping at five. No symmetry is used.

## Results

{markdown_table(payload['arms'])}

The unregularized 32k-point arm reduces export sigma from the Fubini--Study
level near `0.70` to `{payload['arms'][0]['export']['sigma']:.4f}`, so the
small-batch mechanism fixes the earlier gradient-dilution failure in the bulk.
It does not remove the extreme tail: the largest independent-check ratio is
`{payload['arms'][0]['independent_checkpoints']['maximum_ratio']:.2f}`.

Increasing the distinct pool to 131,072 points improves export sigma and the
typical export tail, but an independent checkpoint still reaches
`{payload['arms'][1]['independent_checkpoints']['maximum_ratio']:.2f}`. Lowering
the learning rate to `1e-4` delays spectral drift but eventually reaches a
maximum independent-check ratio of
`{payload['arms'][-1]['independent_checkpoints']['maximum_ratio']:.2f}`.
Coverage and step size are therefore contributory, not sufficient causes.

Relative-spectrum regularization reduces the generalized H condition and the
observed maximum monotonically, but strong regularization worsens bulk sigma.
Exact global CVaR has little effect without enough optimizer exposure because
the `1e-4` tail contains only a few weighted points. Complete-fibre replay
improves the spectrum-w=5 export maximum from
`{payload['arms'][2]['export']['ratio_max']:.2f}` to
`{payload['arms'][5]['export']['ratio_max']:.2f}`, but does not close the gap.
The fixed-eigenvector and naive low-rank parameterizations underfit or retain
bad tails, so capacity reduction alone is not the repair.

## Phi benchmark

The independent PhiFS gCICY pilot has blind sigma `{phi_metrics['sigma']:.4f}`,
L2 `{phi_metrics['volume_ratio_l2']:.4f}`, and maximum ratio
`{phi_metrics['ratio_max']:.3f}` on 8,192 points. At comparable bulk sigma, the
full-H arms have much larger observed maxima. The same gCICY point generator
therefore supports a model without the order-40--140 H tails; the negative
degree gCICY structure does not by itself force the pathology.

This is not a strict architecture leaderboard: the Phi and H pilots have
different parameter counts and training budgets. It is a causal mechanism
comparison.

## Conclusion

The cymetric-style mini-batch schedule transfers successfully as a bulk
optimizer, but it is not a complete gCICY tail repair. The remaining mechanism
is the interaction of the 75,075-dimensional full-H coordinate system, weakly
identified spectral directions, and a bulk-dominated L1 objective. The next
controlled change should be a basis-covariant trust-region or stable-subspace
update for H, combined with global tail replay. More points or a smaller raw
Adam step alone are not justified by these ablations.

All maxima are finite-sample observations. They are not global sup-norm
certificates, and publication holdout seeds `706xx` and `710xx` were not used.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output

    arms = [summarize_arm(root, *arm) for arm in ARMS]
    payload = {
        "schema": "gcicy-cymetric-minibatch-mechanism-comparison-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": {
            "type": "(1,1)",
            "ambient": "P4 x P1",
            "configuration": [[1, 4], [3, -1]],
            "polarization_degree": [4, 4],
            "section_count": 274,
            "fibre_cluster_size": CLUSTER_SIZE,
        },
        "protocol": {
            "symmetry_used": False,
            "complete_fibres_preserved": True,
            "publication_holdout_seeds_used": False,
            "sealed_seed_namespaces": ["706xx", "710xx"],
            "claim_boundary": (
                "Independent finite samples measure observed tails but do not "
                "provide a deterministic global supremum certificate."
            ),
        },
        "arms": arms,
        "phi_benchmark": phi_benchmark(root),
        "interpretation": {
            "mini_batch_transfer": "improves bulk optimization",
            "tail_repair": "incomplete",
            "coverage_only_explanation": "rejected by the 131k-point arm",
            "large_learning_rate_only_explanation": "rejected by the lr=1e-4 arm",
            "spectrum_drift_is_causal": True,
            "recommended_next_test": (
                "basis-covariant relative-H trust region or empirically stable "
                "subspace, combined with complete-fibre global-tail replay"
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "comparison.md").write_text(render_markdown(payload), encoding="utf-8")
    print(output / "comparison.json")
    print(output / "comparison.md")


if __name__ == "__main__":
    main()
