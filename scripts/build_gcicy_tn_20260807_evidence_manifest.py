#!/usr/bin/env python3
"""Bind the August 7 final-comparison evidence to the TN manuscript."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "gcicy paper"
    / "generated_tn"
    / "current_evidence_manifest_20260730.json"
)
OUTPUT = (
    ROOT
    / "gcicy paper"
    / "generated_tn"
    / "current_evidence_manifest_20260807.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(relative: str) -> dict[str, str]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "sha256": sha256(path)}


def main() -> None:
    manifest = json.loads(BASE.read_text(encoding="utf-8"))
    manifest["schema"] = "gcicy-tn-current-evidence-tables-v6"

    x11_root = "outputs/pipeline/type11_x11_equal_time_final_20260807"
    x11_full_h_root = (
        "outputs/pipeline/type11_k6_full_h_equal_time_plateau_20260807"
    )
    x22_root = "outputs/pipeline/type22_fixed_q289_capacity_20260806"
    new_sources = {
        "x11_final_completion_protocol": (
            "pipeline_specs/type11_x11_weekend_completion_20260806.json"
        ),
        "x11_full_h6_time_plateau_protocol": (
            "pipeline_specs/type11_k6_full_h_equal_time_plateau_20260807.json"
        ),
        "x11_equal_time_final_protocol": (
            "pipeline_specs/type11_x11_equal_time_final_20260807.json"
        ),
        "x11_full_h6_time_plateau_summary": (
            f"{x11_full_h_root}/summary.json"
        ),
        "x11_full_h6_training_frozen_manifest": (
            f"{x11_full_h_root}/frozen_checkpoint_manifest.json"
        ),
        "x11_final_frozen_checkpoint_manifest": (
            f"{x11_root}/frozen_checkpoint_manifest.json"
        ),
        "x11_final_sample_manifest": (
            f"{x11_root}/X11_final_seed86707_n200000.manifest.json"
        ),
        "x11_final_common_holdout_summary": (
            f"{x11_root}/final_summary.json"
        ),
        "x11_final_h2_start": f"{x11_root}/h2_start.json",
        "x22_q289_initial_equivalence": (
            f"{x22_root}/q22_to_q289_initial_equivalence_n8190.json"
        ),
        "x22_q22_vs_q289_bootstrap": (
            f"{x22_root}/q22_vs_q289_bootstrap_2000.json"
        ),
        "x22_q60_vs_q289_bootstrap": (
            f"{x22_root}/q60_vs_q289_bootstrap_2000.json"
        ),
        "x22_q289_final_report": (
            f"{x22_root}/q289_k4_capacity_blind83703_n65532.json"
        ),
        "x22_q289_training_summary": (
            f"{x22_root}/q289_k4_capacity_summary.json"
        ),
    }
    for replicate in (1, 2, 3):
        replicate_root = f"{x11_root}/replicate_{replicate}"
        new_sources[f"x11_final_replicate_{replicate}_tn"] = (
            f"{replicate_root}/tn.json"
        )
        new_sources[f"x11_final_replicate_{replicate}_phi"] = (
            f"{replicate_root}/source_density_phi/report.json"
        )
        new_sources[f"x11_final_replicate_{replicate}_full_h6"] = (
            f"{replicate_root}/full_h6_equal_time_plateau.json"
        )
        new_sources[f"x11_final_replicate_{replicate}_tn_vs_phi"] = (
            f"{replicate_root}/tn_vs_phi_bootstrap.json"
        )
        new_sources[f"x11_final_replicate_{replicate}_tn_vs_full_h6"] = (
            f"{replicate_root}/tn_vs_full_h6_bootstrap.json"
        )
        new_sources[f"x11_full_h6_replicate_{replicate}_development"] = (
            f"{x11_full_h_root}/replicate_{replicate}/"
            "development_confirmation.json"
        )
    new_outputs = {
        "x11_final_common_holdout_20260807.tex": (
            "gcicy paper/generated_tn/x11_final_common_holdout_20260807.tex"
        ),
        "x11_final_common_holdout_seedwise_20260807.tex": (
            "gcicy paper/generated_tn/"
            "x11_final_common_holdout_seedwise_20260807.tex"
        ),
        "x11_tn_free_full_h6_seedwise_20260807.tex": (
            "gcicy paper/generated_tn/"
            "x11_tn_free_full_h6_seedwise_20260807.tex"
        ),
        "x11_tn_vs_full_h6_equal_time_seedwise_20260807.tex": (
            "gcicy paper/generated_tn/"
            "x11_tn_vs_full_h6_equal_time_seedwise_20260807.tex"
        ),
        "x11_full_h6_equal_time_plateau_cost_20260807.tex": (
            "gcicy paper/generated_tn/"
            "x11_full_h6_equal_time_plateau_cost_20260807.tex"
        ),
        "x11_equal_time_final_summary_20260807.json": (
            "gcicy paper/generated_tn/"
            "x11_equal_time_final_summary_20260807.json"
        ),
        "x11_full_h6_equal_time_plateau_summary_20260807.json": (
            "gcicy paper/generated_tn/"
            "x11_full_h6_equal_time_plateau_summary_20260807.json"
        ),
        "x22_q289_capacity_20260807.tex": (
            "gcicy paper/generated_tn/x22_q289_capacity_20260807.tex"
        ),
        "summarize_type11_x11_equal_time_final.py": (
            "scripts/summarize_type11_x11_equal_time_final.py"
        ),
        "run_type11_k6_full_h_equal_time_plateau_remote.sh": (
            "scripts/run_type11_k6_full_h_equal_time_plateau_remote.sh"
        ),
        "evaluate_type11_x11_equal_time_final_200k_remote.sh": (
            "scripts/evaluate_type11_x11_equal_time_final_200k_remote.sh"
        ),
        "convert_positive_tensor_network_to_matrix_unit_dictionary.py": (
            "scripts/convert_positive_tensor_network_to_matrix_unit_dictionary.py"
        ),
        "run_type22_fixed_q289_capacity_remote.sh": (
            "scripts/run_type22_fixed_q289_capacity_remote.sh"
        ),
        "build_gcicy_tn_20260807_evidence_manifest.py": (
            "scripts/build_gcicy_tn_20260807_evidence_manifest.py"
        ),
    }

    for name, relative in new_sources.items():
        manifest["sources"][name] = record(relative)
    for name, relative in new_outputs.items():
        manifest["outputs"][name] = record(relative)

    # The manuscript and generated assets evolved together. Refresh all
    # previously registered local digests before writing the new manifest.
    for group in ("sources", "outputs", "inputs"):
        for name, item in list(manifest.get(group, {}).items()):
            manifest[group][name] = record(item["path"])

    obsolete_prefixes = (
        "The principal X11 row",
        "The X11 headline comparison",
        "The final X11 comparison",
        "The TN-free full-H6 paths",
        "The X22 row retains",
    )
    manifest["notes"] = [
        note
        for note in manifest.get("notes", [])
        if not note.startswith(obsolete_prefixes)
    ]
    manifest["notes"].extend(
        [
            (
                "The final X11 comparison contains three groups of three "
                "frozen checkpoints: the original TNs, source-density "
                "residual potentials selected by a validation-only learning-"
                "rate screen, and TN-free full-H6 paths. All nine were frozen "
                "before the common new "
                "200000-point sample was generated and opened."
            ),
            (
                "The TN-free full-H6 paths start directly from the exact "
                "cubic lift of H2. Each receives at least 1200 optimizer "
                "seconds, more than the slowest TN total including its common "
                "warm start, and is then continued to a validation plateau. "
                "The one-time full-H6 section-jet cache construction is not "
                "included in those optimizer times. These are path controls, "
                "not estimates of the full-H6 family optimum."
            ),
            (
                "The X22 q289 result is a single-run equal-update, same-sample "
                "capacity control. Its fixed complete matrix-unit dictionary "
                "is exactly equivalent to the q22 start before continuation, "
                "to the reported numerical precision."
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
