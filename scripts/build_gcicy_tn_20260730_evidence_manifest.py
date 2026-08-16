#!/usr/bin/env python3
"""Build the post-control content-addressed evidence manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "gcicy paper"
    / "generated_tn"
    / "current_evidence_manifest_20260729.json"
)
OUTPUT = (
    ROOT
    / "gcicy paper"
    / "generated_tn"
    / "current_evidence_manifest_20260730.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(relative: str) -> dict[str, str]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "sha256": sha256(path)}


def main() -> None:
    manifest = json.loads(BASE.read_text(encoding="utf-8"))
    manifest["schema"] = "gcicy-tn-current-evidence-tables-v4"

    control_root = (
        "outputs/pipeline/"
        "type21_d8_continuation_capacity_control_20260730"
    )
    new_sources = {
        "x11_plateau_protocol": (
            "pipeline_specs/"
            "type11_h2_matched_tn_phi_three_seed_plateau_final_blind_20260731.json"
        ),
        "x11_plateau_final_blind_summary": (
            "outputs/pipeline/"
            "type11_h2_matched_tn_phi_three_seed_plateau_20260731/"
            "three_seed_plateau_final_blind_summary.json"
        ),
        "x21_equal_update_protocol": (
            "pipeline_specs/"
            "type21_d8_continuation_capacity_control_20260730.json"
        ),
        "x21_equal_update_preregistration": f"{control_root}/PREREGISTRATION.json",
        "x21_equal_update_summary": f"{control_root}/summary.json",
        "x21_equal_update_holdout_a_manifest": (
            f"{control_root}/X21_capacity_holdout_seed86206_n98304.manifest.json"
        ),
        "x21_equal_update_holdout_b_manifest": (
            f"{control_root}/X21_capacity_holdout_seed86207_n98304.manifest.json"
        ),
        "source_map_immersion_certificate": (
            "outputs/pipeline/gcicy_source_map_immersion_20260730/"
            "certificate.json"
        ),
        "source_restriction_rank_certificates": (
            "outputs/section_restriction_rank_certificates.json"
        ),
        "x21_kd_plateau_grid_protocol": (
            "pipeline_specs/type21_kd_plateau_grid_20260731.json"
        ),
        "x21_kd_plateau_grid_preregistration": (
            "outputs/pipeline/type21_kd_plateau_grid_floor1e10_20260731/"
            "PREREGISTRATION.json"
        ),
        "x21_kd_plateau_followups_protocol": (
            "pipeline_specs/type21_kd_plateau_followups_20260802.json"
        ),
        "x21_kd_plateau_followups_preregistration": (
            "outputs/pipeline/type21_kd_plateau_followups_20260802/"
            "PREREGISTRATION.json"
        ),
        "x21_nested_k_protocol": (
            "pipeline_specs/type21_nested_k8d10_to_k16d10_20260802.json"
        ),
        "x21_nested_k_preregistration": (
            "outputs/pipeline/type21_nested_k8d10_to_k16d10_20260802/"
            "PREREGISTRATION.json"
        ),
        "x21_kd_plateau_paper_summary": (
            "outputs/pipeline/type21_kd_plateau_paper_summary_20260802.json"
        ),
        "x21_h4_vs_d8_final_blind_bootstrap": (
            "outputs/pipeline/type21_q121_d12_final_blind_20260729/"
            "h4_vs_d8_final_blind_bootstrap_2000.json"
        ),
        "x22_q22_vs_q60_k4_bootstrap": (
            "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/"
            "q22_vs_q60_k4_paired_bootstrap_20260806.json"
        ),
        "x11_plateau_training_ledger": (
            "outputs/pipeline/"
            "type11_h2_matched_tn_phi_three_seed_plateau_20260731/"
            "training_ledger.json"
        ),
    }
    x11_plateau_root = (
        "outputs/pipeline/"
        "type11_h2_matched_tn_phi_three_seed_plateau_20260731"
    )
    for replicate in (1, 2, 3):
        new_sources[f"x11_plateau_replicate_{replicate}_tn_report"] = (
            f"{x11_plateau_root}/replicate_{replicate}/tn_final_blind.json"
        )
        new_sources[f"x11_plateau_replicate_{replicate}_phi_report"] = (
            f"{x11_plateau_root}/replicate_{replicate}/"
            "phi_final_blind/report.json"
        )
        new_sources[f"x11_plateau_replicate_{replicate}_bootstrap"] = (
            f"{x11_plateau_root}/replicate_{replicate}/"
            "tn_vs_phi_final_blind_bootstrap.json"
        )
        for architecture in ("tn", "phi"):
            for stage in ("primary", "precision"):
                new_sources[
                    f"x11_plateau_replicate_{replicate}_{architecture}_{stage}_training"
                ] = (
                    f"{x11_plateau_root}/training_reports/"
                    f"replicate_{replicate}_{architecture}_{stage}.json"
                )
    for seed in (86231, 86232, 86233):
        new_sources[f"x21_equal_update_seed_{seed}_training"] = (
            f"{control_root}/seed_{seed}/training_summary.json"
        )
        new_sources[f"x21_equal_update_seed_{seed}_parameter_audit"] = (
            f"{control_root}/seed_{seed}/parameter_change_audit.json"
        )
        new_sources[f"x21_equal_update_seed_{seed}_bootstrap"] = (
            f"{control_root}/seed_{seed}_control_vs_d12_bootstrap_2000.json"
        )

    new_outputs = {
        "x11_matched_solver_plateau_20260731.tex": (
            "gcicy paper/generated_tn/"
            "x11_matched_solver_plateau_20260731.tex"
        ),
        "x11_plateau_seedwise_20260731.tex": (
            "gcicy paper/generated_tn/"
            "x11_plateau_seedwise_20260731.tex"
        ),
        "gcicy_tn_x11_plateau_tail_survival.pdf": (
            "gcicy paper/figures/"
            "gcicy_tn_x11_plateau_tail_survival.pdf"
        ),
        "run_type11_h2_matched_tn_phi_three_seed_plateau_remote.sh": (
            "scripts/"
            "run_type11_h2_matched_tn_phi_three_seed_plateau_remote.sh"
        ),
        "x21_equal_update_capacity_control_20260730.tex": (
            "gcicy paper/generated_tn/"
            "x21_equal_update_capacity_control_20260730.tex"
        ),
        "source_immersion_20260730.tex": (
            "gcicy paper/generated_tn/source_immersion_20260730.tex"
        ),
        "gcicy_tn_x21_equal_update_control.pdf": (
            "gcicy paper/figures/gcicy_tn_x21_equal_update_control.pdf"
        ),
        "plot_type21_equal_update_capacity_control.py": (
            "scripts/plot_type21_equal_update_capacity_control.py"
        ),
        "certify_gcicy_source_map_immersion.py": (
            "scripts/certify_gcicy_source_map_immersion.py"
        ),
        "build_gcicy_tn_20260730_evidence_manifest.py": (
            "scripts/build_gcicy_tn_20260730_evidence_manifest.py"
        ),
        "verify_gcicy_tn_current_evidence_manifest.py": (
            "scripts/verify_gcicy_tn_current_evidence_manifest.py"
        ),
        "verify_gcicy_tn_manuscript_claims.py": (
            "scripts/verify_gcicy_tn_manuscript_claims.py"
        ),
        "x21_kd_plateau_nested_20260802.tex": (
            "gcicy paper/generated_tn/x21_kd_plateau_nested_20260802.tex"
        ),
        "build_type21_kd_plateau_paper_assets.py": (
            "scripts/build_type21_kd_plateau_paper_assets.py"
        ),
        "run_type21_kd_plateau_grid_remote.sh": (
            "scripts/run_type21_kd_plateau_grid_remote.sh"
        ),
        "run_type21_kd_plateau_followups_remote.sh": (
            "scripts/run_type21_kd_plateau_followups_remote.sh"
        ),
        "run_type21_nested_k8d10_to_k16d10_remote.sh": (
            "scripts/run_type21_nested_k8d10_to_k16d10_remote.sh"
        ),
        "x11_plateau_cost_summary_20260802.tex": (
            "gcicy paper/generated_tn/x11_plateau_cost_summary_20260802.tex"
        ),
        "x11_plateau_cost_seedwise_20260802.tex": (
            "gcicy paper/generated_tn/x11_plateau_cost_seedwise_20260802.tex"
        ),
        "x21_final_blind_compression_20260802.tex": (
            "gcicy paper/generated_tn/"
            "x21_final_blind_compression_20260802.tex"
        ),
        "gcicy_tn_x21_path_dependence_20260802.pdf": (
            "gcicy paper/figures/gcicy_tn_x21_path_dependence_20260802.pdf"
        ),
        "review_revision_assets_20260802.json": (
            "gcicy paper/generated_tn/review_revision_assets_20260802.json"
        ),
        "build_gcicy_tn_review_revision_assets.py": (
            "scripts/build_gcicy_tn_review_revision_assets.py"
        ),
        "bootstrap_type22_dictionary_capacity.py": (
            "scripts/bootstrap_type22_dictionary_capacity.py"
        ),
    }

    for name, relative in new_sources.items():
        manifest["sources"][name] = record(relative)
    for name, relative in new_outputs.items():
        manifest["outputs"][name] = record(relative)

    # Refresh every pre-existing digest because generated ledgers and
    # verification scripts may have changed during the controlled revision.
    for group in ("sources", "outputs"):
        for name, item in list(manifest[group].items()):
            manifest[group][name] = record(item["path"])

    manifest["notes"] = [
        note
        for note in manifest["notes"]
        if not note.startswith("D8 uses the representative")
    ]
    manifest["notes"].extend(
        [
            (
                "The X11 headline comparison continues each of three TN "
                "and residual-potential seeds to a preregistered two-stage "
                "validation-gain plateau before generating and opening one "
                "common 200000-point final blind pool."
            ),
            (
                "The historical X21 source-to-D12 result combines bond "
                "expansion with 30 additional epochs and is not interpreted "
                "as a causal capacity effect."
            ),
            (
                "The preregistered X21 equal-update control evaluates all "
                "three source, D8-to-D8 and D8-to-D12 triplets on a fresh "
                "196608-point holdout; its all-three-pairs directional gate "
                "failed."
            ),
            (
                "The source-map immersion certificate closes the global "
                "positivity hypothesis for all three headline TN families."
            ),
            (
                "The expanded X21 k-D ladder and nested-continuation "
                "controls are single-seed development evidence on an "
                "already inspected 49152-point pool; no new final blind "
                "pool was opened for architecture selection."
            ),
            (
                "The headline X11, historical/equal-update X21 and X22 "
                "runs use positive reference coefficient 1e-4; the "
                "single-seed X21 k-D development ladder uses 1e-10 and "
                "is not pooled with those causal comparisons."
            ),
            (
                "The X11 plateau cost table is reconstructed from twelve "
                "frozen primary/precision training reports and excludes "
                "the common 20-epoch warm-start."
            ),
            (
                "The X22 q22-to-q60 dictionary-capacity comparison uses "
                "matched optimization and a paired 2000-replicate "
                "fibre-cluster bootstrap; its extreme-quantile interval "
                "crosses zero."
            ),
        ]
    )

    OUTPUT.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    print(
        f"{len(manifest['sources'])} sources, "
        f"{len(manifest['outputs'])} outputs"
    )


if __name__ == "__main__":
    main()
