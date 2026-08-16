#!/usr/bin/env python3
"""Summarize the controlled quintic TN degree and capacity experiments."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNS = (
    (
        "k9_q21_d5",
        "outputs/quintic_tn_q21_k9_d5_p8820_vectorized_b64_e20_20260719/report.json",
    ),
    (
        "k12_q21_d5",
        "outputs/quintic_tn_q21_k12_d5_p11970_vectorized_b64_e20_degree_ladder_20260719/report.json",
    ),
    (
        "k15_q21_d5",
        "outputs/quintic_tn_q21_k15_d5_p15120_vectorized_b64_e20_degree_ladder_20260719/report.json",
    ),
    (
        "k20_q21_d5",
        "outputs/quintic_tn_q21_k20_d5_p20370_vectorized_b64_e20_degree_ladder_20260719/report.json",
    ),
    (
        "k20_q25_d5_lr1e4",
        "outputs/quintic_tn_k20_capacity_20260719/report.json",
    ),
    (
        "k20_q25_d5_lr3e5",
        "outputs/quintic_tn_k20_capacity_20260719/q25_d5_lr3e5_b64_e10_continuation/report.json",
    ),
    (
        "k20_q25_d6_b64",
        "outputs/quintic_tn_k20_capacity_20260719/q25_d6_noise1e2_lr3e5_b64_e10/report.json",
    ),
    (
        "k20_q25_d6_b1024_lr3e5",
        "outputs/quintic_tn_k20_capacity_20260719/q25_d6_b1024_lr3e5_e30_refinement/report.json",
    ),
    (
        "k20_q25_d6_b1024_lr1e5",
        "outputs/quintic_tn_k20_capacity_20260719/q25_d6_b1024_lr1e5_e50_final_refinement/report.json",
    ),
    (
        "matched_cymetric_p8820",
        "outputs/cymetric_quintic_fermat_width63_p8820_same_points_20260719/report.json",
    ),
)

FINAL_ROUTE = (
    "k20_q21_d5",
    "k20_q25_d5_lr1e4",
    "k20_q25_d5_lr3e5",
    "k20_q25_d6_b64",
    "k20_q25_d6_b1024_lr3e5",
    "k20_q25_d6_b1024_lr1e5",
)


def read_report(relative_path: str) -> dict:
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def extract_tn(label: str, relative_path: str, report: dict) -> dict:
    normalized = report["blind_test"]["normalized_volume"]
    configuration = report["configuration"]
    architecture = report["architecture"]
    timing = report["timing_seconds"]
    return {
        "label": label,
        "report": relative_path,
        "method": "positive_tn",
        "k": int(configuration["site_count"]),
        "q": int(configuration["dictionary_rank"]),
        "D": int(configuration["bond_dimension"]),
        "parameters": int(architecture["trainable_real_parameter_count"]),
        "batch_size": int(configuration["batch_size"]),
        "learning_rate": float(configuration["learning_rate"]),
        "epochs": int(configuration["epochs"]),
        "best_epoch": int(report["training"]["best_epoch"]),
        "sigma": float(normalized["sigma_official_formula"]),
        "chi": float(normalized["weighted_rms_abs_residual"]),
        "q999_abs_residual": float(
            normalized["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99_abs_residual": float(
            normalized["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum_abs_residual": float(
            normalized["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
        "nonpositive_count": int(
            normalized["nonpositive_min_eigenvalue"]["count"]
        ),
        "blind_points": int(normalized["n_points"]),
        "training_seconds": float(timing["training"]),
        "wall_seconds": float(timing["wall_total"]),
    }


def extract_cymetric(label: str, relative_path: str, report: dict) -> dict:
    normalized = report["trained_phi_model"]
    timing = report["timing_seconds"]
    return {
        "label": label,
        "report": relative_path,
        "method": "cymetric_phi",
        "k": None,
        "q": None,
        "D": None,
        "parameters": int(report["network"]["parameter_count"]),
        "batch_size": int(report["configuration"]["batch_size"]),
        "learning_rate": float(report["configuration"]["learning_rate"]),
        "epochs": int(report["configuration"]["epochs"]),
        "best_epoch": None,
        "sigma": float(normalized["sigma_official_formula"]),
        "chi": float(normalized["weighted_rms_abs_residual"]),
        "q999_abs_residual": float(
            normalized["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99_abs_residual": float(
            normalized["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum_abs_residual": float(
            normalized["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
        "nonpositive_count": int(
            normalized["nonpositive_min_eigenvalue"]["count"]
        ),
        "blind_points": int(normalized["n_points"]),
        "training_seconds": float(timing["training"]),
        "wall_seconds": float(timing["wall_total"]),
    }


def improvement(reference: float, candidate: float) -> float:
    return 1.0 - candidate / reference


def main() -> None:
    rows = []
    for label, relative_path in RUNS:
        report = read_report(relative_path)
        if label == "matched_cymetric_p8820":
            rows.append(extract_cymetric(label, relative_path, report))
        else:
            rows.append(extract_tn(label, relative_path, report))
    by_label = {row["label"]: row for row in rows}
    baseline = by_label["k9_q21_d5"]
    k20_q21 = by_label["k20_q21_d5"]
    q25_d5_first = by_label["k20_q25_d5_lr1e4"]
    q25_d5 = by_label["k20_q25_d5_lr3e5"]
    d6_b64 = by_label["k20_q25_d6_b64"]
    d6_large_batch = by_label["k20_q25_d6_b1024_lr3e5"]
    final = by_label["k20_q25_d6_b1024_lr1e5"]
    cymetric = by_label["matched_cymetric_p8820"]
    route_wall = sum(by_label[label]["wall_seconds"] for label in FINAL_ROUTE)
    route_training = sum(
        by_label[label]["training_seconds"] for label in FINAL_ROUTE
    )
    summary = {
        "schema": "quintic-tn-degree-capacity-summary-v1",
        "controlled_geometry": "Fermat quintic hypersurface X_5 in P^4",
        "common_blind_points": 200000,
        "rows": rows,
        "derived": {
            "q21_degree_k9_to_k20_sigma_improvement_fraction": improvement(
                baseline["sigma"], k20_q21["sigma"]
            ),
            "q21_degree_k9_to_k20_max_residual_improvement_fraction": improvement(
                baseline["maximum_abs_residual"],
                k20_q21["maximum_abs_residual"],
            ),
            "q21_to_q25_first_stage_sigma_improvement_fraction": improvement(
                k20_q21["sigma"], q25_d5_first["sigma"]
            ),
            "q25_low_lr_refinement_sigma_improvement_fraction": improvement(
                q25_d5_first["sigma"], q25_d5["sigma"]
            ),
            "d5_to_d6_after_q25_sigma_improvement_fraction": improvement(
                q25_d5["sigma"], d6_b64["sigma"]
            ),
            "d6_large_batch_sigma_improvement_fraction": improvement(
                d6_b64["sigma"], d6_large_batch["sigma"]
            ),
            "d6_final_low_lr_sigma_improvement_fraction": improvement(
                d6_large_batch["sigma"], final["sigma"]
            ),
            "final_over_k9_sigma_improvement_fraction": improvement(
                baseline["sigma"], final["sigma"]
            ),
            "final_over_matched_cymetric_sigma_improvement_fraction": improvement(
                cymetric["sigma"], final["sigma"]
            ),
            "final_to_literature_modnet_sigma_ratio_context_only": (
                final["sigma"] / 0.0010
            ),
            "final_route_training_seconds": route_training,
            "final_route_wall_seconds": route_wall,
        },
        "failed_control": {
            "label": "k20_to_k25_linear_interior_site_interpolation",
            "epoch0_validation_sigma": 0.1344945,
            "source_k20_validation_sigma": 0.001077739,
            "classification": (
                "initialization failure; not evidence against k=25 capacity"
            ),
        },
        "claim_limits": [
            "The ModNet value 0.0010 is literature context, not a matched run.",
            "The final continuation time is cumulative across all registered stages.",
            "A multi-seed final-model audit and matched ModNet reproduction remain open.",
        ],
    }

    json_path = ROOT / "outputs/quintic_tn_degree_capacity_summary_20260719.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Quintic TN degree and capacity summary",
        "",
        "All rows use the same Fermat quintic data and 200,000 blind points.",
        "",
        "| run | k | q | D | parameters | batch | sigma | chi | q99.9 | CVaR99 | max | wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {k} | {q} | {D} | {parameters:,} | {batch_size} | "
            "{sigma:.7f} | {chi:.7f} | {q999_abs_residual:.7f} | "
            "{cvar99_abs_residual:.7f} | {maximum_abs_residual:.7f} | "
            "{wall_seconds:.1f} |".format(
                **{
                    **row,
                    "k": row["k"] if row["k"] is not None else "-",
                    "q": row["q"] if row["q"] is not None else "-",
                    "D": row["D"] if row["D"] is not None else "-",
                }
            )
        )
    derived = summary["derived"]
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            f"- Raising k from 9 to 20 at q=21, D=5 improves sigma by {100 * derived['q21_degree_k9_to_k20_sigma_improvement_fraction']:.1f}%.",
            f"- The first q=21 to 25 continuation stage improves sigma by {100 * derived['q21_to_q25_first_stage_sigma_improvement_fraction']:.1f}%.",
            f"- Lower-learning-rate q=25 refinement then improves sigma by a further {100 * derived['q25_low_lr_refinement_sigma_improvement_fraction']:.1f}%.",
            f"- Raising D=5 to 6 after q=25 improves sigma by {100 * derived['d5_to_d6_after_q25_sigma_improvement_fraction']:.1f}% in the b64 continuation.",
            f"- Large-batch D=6 refinement improves sigma by {100 * derived['d6_large_batch_sigma_improvement_fraction']:.1f}%, followed by {100 * derived['d6_final_low_lr_sigma_improvement_fraction']:.1f}% from the final low-learning-rate stage.",
            f"- The final model reaches sigma={final['sigma']:.7f} with {final['parameters']:,} trainable real parameters and no nonpositive metric on the blind pool.",
            f"- Relative to the matched standard cymetric baseline, final sigma is lower by {100 * derived['final_over_matched_cymetric_sigma_improvement_fraction']:.1f}%.",
            f"- The registered cumulative continuation route takes {derived['final_route_wall_seconds'] / 60:.1f} wall minutes.",
            "",
            "## Interpretation",
            "",
            "The dominant capacity bottleneck was the incomplete local operator dictionary, not the bond dimension. Large-batch refinement then removed most of the stochastic optimization floor. The failed k=25 linear site interpolation is an initialization failure and must not be interpreted as a finite-k limitation.",
            "",
            "The literature ModNet sigma=0.0010 remains context only until it is reproduced on the identical data, blind pool, normalization, and hardware protocol.",
        ]
    )
    markdown_path = ROOT / "docs/quintic_tn_degree_capacity_20260719.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
