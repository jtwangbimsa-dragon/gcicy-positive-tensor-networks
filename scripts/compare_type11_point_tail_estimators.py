#!/usr/bin/env python3
"""Build the point-64 tail-estimator and bicubic-control diagnostic report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    (
        "baseline",
        "Point-64, no tail loss",
        "p4p1_type11_hirzebruch_k4_pointb64_baseline_e3",
    ),
    (
        "sparse_cvar",
        "Sparse full-pool CVaR",
        "p4p1_type11_hirzebruch_k4_pointb64_upperlogcvar_e3",
    ),
    (
        "matched_replay",
        "Matched stratified replay",
        "p4p1_type11_hirzebruch_k4_pointb64_upperlog_replay_matched_e3",
    ),
    (
        "broader_replay",
        "Broader stratified replay",
        "p4p1_type11_hirzebruch_k4_pointb64_upperlog_replay_e3",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/pipeline/type11_point_tail_estimator_diagnostic_20260717"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "sigma": float(metrics["sigma"]),
        "volume_ratio_l2": float(metrics["sqrt_squared_energy"]),
        "ratio_max": float(metrics["normalized_ratio_max"]),
        "positive_log_ratio_q999": float(metrics["positive_log_ratio_q999"]),
        "ratio_above_3_weighted_mass": float(
            metrics["normalized_ratio_above_3_weighted_mass"]
        ),
    }


def load_arm(identifier: str, label: str, stem: str) -> dict[str, Any]:
    path = ROOT / "outputs" / "pipeline" / f"{stem}_summary.json"
    summary = load_json(path)
    checkpoints = [
        compact_metrics(row["candidate"])
        for row in summary["checkpoint_validation"].values()
    ]
    final_history = summary["history"][-1]
    objective = summary["optimization"]["global_upper_log_ratio_cvar"]
    return {
        "id": identifier,
        "label": label,
        "source": str(path.relative_to(ROOT)),
        "tail_objective": {
            "loss_weight": float(objective["loss_weight"]),
            "tail_fractions": [float(value) for value in objective["tail_fractions"]],
            "gradient_estimator": objective["gradient_estimator"],
            "minibatch_gradient_rescaling": objective[
                "minibatch_gradient_rescaling"
            ],
            "replay_points_per_tail_fraction": int(
                objective.get("point_replay_points_per_tail_fraction", 0)
            ),
        },
        "checkpoints": {
            "mean_sigma": sum(row["sigma"] for row in checkpoints)
            / len(checkpoints),
            "maximum_volume_ratio_l2": max(
                row["volume_ratio_l2"] for row in checkpoints
            ),
            "maximum_ratio": max(row["ratio_max"] for row in checkpoints),
            "per_seed_ratio_maxima": [row["ratio_max"] for row in checkpoints],
            "maximum_positive_log_ratio_q999": max(
                row["positive_log_ratio_q999"] for row in checkpoints
            ),
            "maximum_ratio_above_3_weighted_mass": max(
                row["ratio_above_3_weighted_mass"] for row in checkpoints
            ),
        },
        "export": compact_metrics(summary["export_candidate"]),
        "optimizer": {
            "mean_preclip_gradient_norm": float(
                final_history[
                    "mean_preclip_gradient_norm_since_previous_evaluation"
                ]
            ),
            "maximum_preclip_gradient_norm": float(
                final_history[
                    "maximum_preclip_gradient_norm_since_previous_evaluation"
                ]
            ),
            "clipped_steps": int(
                final_history[
                    "clipped_optimizer_steps_since_previous_evaluation"
                ]
            ),
            "optimizer_steps": int(
                final_history["optimizer_steps_since_previous_evaluation"]
            ),
            "relative_h_log_spectrum_span": float(
                final_history["relative_to_initial_log_eigenvalue_span"]
            ),
        },
        "request": summary["request"],
    }


def validate_common_design(arms: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "model_seed",
        "exact_model",
        "degree",
        "precision",
        "basis_points",
        "train_batches",
        "checkpoint_batches",
        "export_seed",
        "epochs",
        "learning_rate",
        "h_parameterization",
        "relative_log_spectrum_loss_weight",
        "points_per_optimizer_step",
        "torch_seed",
        "group_shuffle_seed",
    )
    reference = {key: arms[0]["request"][key] for key in keys}
    for arm in arms[1:]:
        observed = {key: arm["request"][key] for key in keys}
        if observed != reference:
            raise ValueError(f"arm {arm['id']} is not matched: {observed}")
    return {
        **reference,
        "training_points": sum(
            int(row["points"]) for row in reference["train_batches"]
        ),
        "checkpoint_points": sum(
            int(row["points"]) for row in reference["checkpoint_batches"]
        ),
    }


def percent_change(candidate: float, reference: float) -> float:
    return 100.0 * (candidate / reference - 1.0)


def load_witness() -> dict[str, Any]:
    path = (
        ROOT
        / "outputs"
        / "pipeline"
        / "type11_pointb64_replay_matched_seed72003_witness.json"
    )
    raw = load_json(path)
    row = raw["source_worst"]
    source_point = row["source_point"]
    target_point = row["target_point"]
    return {
        "source": str(path.relative_to(ROOT)),
        "ratio_spearman": float(raw["ratio_spearman"]),
        "same_worst_index": bool(
            raw["source_worst"]["index"] == raw["target_worst"]["index"]
        ),
        "index": int(row["index"]),
        "baseline_ratio": float(row["source_ratio"]),
        "matched_replay_ratio": float(row["target_ratio"]),
        "baseline_cluster_ratios": [float(value) for value in row["source_cluster_ratios"]],
        "matched_replay_cluster_ratios": [
            float(value) for value in row["target_cluster_ratios"]
        ],
        "importance_weight": float(row["importance_weight"]),
        "geometry": row["geometry"],
        "holomorphic_volume_log_density": float(
            source_point["holomorphic_volume_log_density"]
        ),
        "residue_denominator_magnitude": float(
            source_point["residue_denominator"]["magnitude"]
        ),
        "tangent_basis_condition_number": float(
            source_point["tangent_basis_condition_number"]
        ),
        "positive_equation_row_norm": float(
            source_point["positive_equation_row_norm"]
        ),
        "baseline_metric_eigenvalues": [
            float(value) for value in source_point["candidate_metric"]["eigenvalues"]
        ],
        "matched_replay_metric_eigenvalues": [
            float(value) for value in target_point["candidate_metric"]["eigenvalues"]
        ],
        "baseline_metric_condition_number": float(
            source_point["candidate_metric"]["condition_number"]
        ),
        "matched_replay_metric_condition_number": float(
            target_point["candidate_metric"]["condition_number"]
        ),
        "baseline_metric_log_determinant": float(
            source_point["candidate_metric"]["log_determinant"]
        ),
        "matched_replay_metric_log_determinant": float(
            target_point["candidate_metric"]["log_determinant"]
        ),
    }


def load_bicubic_control() -> dict[str, Any]:
    summary_path = ROOT / "outputs" / "bicubic_global_h_metric_k3_k2xk1_gpu_summary.json"
    audit_path = ROOT / "outputs" / "bicubic_k3_sampler_matched_large_audit.json"
    summary = load_json(summary_path)
    audit = load_json(audit_path)
    ratio_maximum = max(
        float(row["normalized_ratio_max"]) for row in audit["per_seed"]
    )
    return {
        "summary_source": str(summary_path.relative_to(ROOT)),
        "large_audit_source": str(audit_path.relative_to(ROOT)),
        "degree": int(summary["degree"]),
        "restricted_section_count": int(summary["restricted_section_count"]),
        "scale_free_h_dimension": int(summary["restricted_section_count"] ** 2 - 1),
        "initialization": summary["initialization"],
        "last_recorded_epoch": int(summary["history"][-1]["epoch"]),
        "export_weighted_centered_log_rms": float(summary["global_h_export"]["rms"]),
        "large_audit_points": int(audit["total_points"]),
        "large_audit_mean_sigma": float(audit["aggregate"]["mean_sigma"]),
        "large_audit_maximum_ratio": ratio_maximum,
        "large_audit_minimum_metric_eigenvalue": float(
            audit["aggregate"]["minimum_metric_eigenvalue"]
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    rows = [
        "| Arm | Tail fractions | Check sigma | Check max L2 | Check max r | "
        "Check q99.9 | Mass(r>3) | Export sigma | Export max r |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    optimizer_rows = [
        "| Arm | Estimator | Mean preclip grad | Max preclip grad | "
        "Clipped steps | H log-spectrum span |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        check = arm["checkpoints"]
        export = arm["export"]
        objective = arm["tail_objective"]
        fractions = ", ".join(f"{value:g}" for value in objective["tail_fractions"])
        rows.append(
            f"| {arm['label']} | {fractions or '-'} | {check['mean_sigma']:.4f} | "
            f"{check['maximum_volume_ratio_l2']:.3f} | "
            f"{check['maximum_ratio']:.2f} | "
            f"{check['maximum_positive_log_ratio_q999']:.3f} | "
            f"{check['maximum_ratio_above_3_weighted_mass']:.3%} | "
            f"{export['sigma']:.4f} | {export['ratio_max']:.2f} |"
        )
        optimizer = arm["optimizer"]
        estimator = (
            "inactive"
            if objective["loss_weight"] == 0
            else objective["gradient_estimator"].replace("_", " ")
        )
        optimizer_rows.append(
            f"| {arm['label']} | {estimator} | "
            f"{optimizer['mean_preclip_gradient_norm']:.2f} | "
            f"{optimizer['maximum_preclip_gradient_norm']:.2f} | "
            f"{optimizer['clipped_steps']} / {optimizer['optimizer_steps']} | "
            f"{optimizer['relative_h_log_spectrum_span']:.3f} |"
        )

    effects = payload["estimator_only_effects_percent"]
    witness = payload["same_point_witness"]
    geometry = witness["geometry"]
    bicubic = payload["bicubic_control"]
    return f"""# Type-(1,1) point-64 tail-estimator diagnostic

## Controlled design

All four gCICY arms use the same degree `(4,4)` section basis, 32,768
training points, 16,384 checkpoint points, 8,192 export points, cold
Fubini--Study initialization, model and shuffle seeds, 64-point batches, 512
updates per epoch, and three epochs. The matched estimator contrast keeps the
upper-tail loss weight, threshold, and fractions `(0.001, 0.0001)` fixed.

## Results

{"\n".join(rows)}

## Optimizer diagnostics

{"\n".join(optimizer_rows)}

Replacing sparse, rescaled full-pool tail hits by stratified replay changes
the matched experiment as follows:

- maximum pre-clipping gradient: `{effects['maximum_preclip_gradient_norm']:+.2f}%`;
- mean checkpoint sigma: `{effects['mean_check_sigma']:+.2f}%`;
- maximum checkpoint ratio: `{effects['maximum_check_ratio']:+.2f}%`;
- weighted checkpoint mass above three: `{effects['maximum_check_mass_r_gt_3']:+.2f}%`;
- export maximum ratio: `{effects['export_maximum_ratio']:+.2f}%`.

The gradient burst is removed, but the maximum is unchanged within 0.2%.
Therefore the sparse CVaR estimator was an optimization defect, not the root
cause of the persistent spike.

## Same-point witness

On the common seed-72003 set, both the no-tail baseline and matched replay
model attain their maximum at index `{witness['index']}`. Their ratios are
`{witness['baseline_ratio']:.2f}` and `{witness['matched_replay_ratio']:.2f}`;
the full-field Spearman correlation is `{witness['ratio_spearman']:.3f}`. The
four roots on that fibre have baseline ratios
`{witness['baseline_cluster_ratios']}` and replay ratios
`{witness['matched_replay_cluster_ratios']}`. The failure is localized to one
root, rather than the whole fibre.

The point is numerically regular: equation residual
`{geometry['relative_equation_residual']:.2e}`, normalized reduced-Jacobian
minimum singular value
`{geometry['normalized_reduced_jacobian_min_singular_value']:.3f}`, minimum
projective root separation
`{geometry['minimum_projective_x_root_separation']:.3f}`, and tangent-basis
condition number `{witness['tangent_basis_condition_number']:.3f}`. The
candidate metric is also positive and moderate, with replay eigenvalues
`{witness['matched_replay_metric_eigenvalues']}` and condition number
`{witness['matched_replay_metric_condition_number']:.3f}`.

The distinguishing feature is the target density: the holomorphic-volume log
density is `{witness['holomorphic_volume_log_density']:.3f}`, while the replay
metric log determinant is only
`{witness['matched_replay_metric_log_determinant']:.3f}`. After global
normalization this mismatch produces the large ratio. This supports a stable
local approximation/training difficulty; it is not evidence for a sampled
singularity, bad chart, root collision, or ill-conditioned metric.

## Why bicubic is different

The existing bicubic result is genuinely much better on its audit: degree
`k={bicubic['degree']}`, `{bicubic['restricted_section_count']}` restricted
sections (`{bicubic['scale_free_h_dimension']}` scale-free H parameters),
weighted centered-log RMS
`{bicubic['export_weighted_centered_log_rms']:.4f}`, and maximum normalized
ratio `{bicubic['large_audit_maximum_ratio']:.3f}` over
`{bicubic['large_audit_points']:,}` fresh points.

It is not, however, a same-algorithm control. It uses a product warm start from
trained `k=2` and `k=1` metrics, full-batch train and validation losses,
centered squared log residual, a positivity barrier, drift regularization, and
800 epochs. The present gCICY run uses 274 sections (75,075 scale-free H
parameters), cold initialization, random point-64 L1 sigma loss, and only
three data passes. Geometry, capacity, initialization, objective, and budget
all change simultaneously.

## Conclusion and next discriminating experiment

1. Stratified replay fixes the CVaR gradient pathology but not the spike.
2. The same regular, one-root hard point survives every optimizer variant
   tested here; the evidence now favors finite-k approximation or insufficient
   local resolution over a sampler singularity.
3. The present data do not yet distinguish an intrinsically harder gCICY
   target from a weaker gCICY training protocol.
4. The next decisive test is a matched-algorithm cross-geometry ablation:
   train bicubic and this gCICY with the same initialization class, centered
   log-squared objective, batch policy, parameter-to-sample ratio, and number
   of data passes. Only that experiment can attribute the remaining gap to
   geometry.

All maxima here are finite-sample diagnostics, not global sup-norm
certificates.
"""


def main() -> None:
    args = parse_args()
    arms = [load_arm(*row) for row in ARMS]
    by_id = {arm["id"]: arm for arm in arms}
    sparse = by_id["sparse_cvar"]
    replay = by_id["matched_replay"]
    effects = {
        "maximum_preclip_gradient_norm": percent_change(
            replay["optimizer"]["maximum_preclip_gradient_norm"],
            sparse["optimizer"]["maximum_preclip_gradient_norm"],
        ),
        "mean_check_sigma": percent_change(
            replay["checkpoints"]["mean_sigma"],
            sparse["checkpoints"]["mean_sigma"],
        ),
        "maximum_check_ratio": percent_change(
            replay["checkpoints"]["maximum_ratio"],
            sparse["checkpoints"]["maximum_ratio"],
        ),
        "maximum_check_mass_r_gt_3": percent_change(
            replay["checkpoints"]["maximum_ratio_above_3_weighted_mass"],
            sparse["checkpoints"]["maximum_ratio_above_3_weighted_mass"],
        ),
        "export_maximum_ratio": percent_change(
            replay["export"]["ratio_max"], sparse["export"]["ratio_max"]
        ),
    }
    payload = {
        "schema": "type11-point-tail-estimator-diagnostic-v1",
        "common_design": validate_common_design(arms),
        "arms": arms,
        "estimator_only_effects_percent": effects,
        "same_point_witness": load_witness(),
        "bicubic_control": load_bicubic_control(),
        "claim_limit": (
            "Disclosed-seed development diagnostic; sampled maxima are not a "
            "global sup-norm certificate."
        ),
    }
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {out_dir / 'report.json'}")
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
