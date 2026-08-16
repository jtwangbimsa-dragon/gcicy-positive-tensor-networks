#!/usr/bin/env python3
"""Bind the completed fixed-D degree comparison to the TN manuscript."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "gcicy paper"
    / "generated_tn"
    / "current_evidence_manifest_20260807.json"
)
OUTPUT = (
    ROOT
    / "gcicy paper"
    / "generated_tn"
    / "current_evidence_manifest_20260809.json"
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
    manifest["schema"] = "gcicy-tn-current-evidence-tables-v7"

    root = (
        "outputs/pipeline/"
        "type21_fixed_d8_degree_scaling_floor1e14_20260807"
    )
    final = f"{root}/final_evidence_k8_k20"
    new_sources = {
        "x21_fixed_d8_degree_protocol": (
            "pipeline_specs/type21_fixed_d8_degree_scaling_20260807.json"
        ),
        "x21_fixed_d8_degree_protocol_amendment": (
            "pipeline_specs/"
            "type21_fixed_d8_degree_scaling_k8_k20_amendment_20260809.json"
        ),
        "x21_fixed_d8_degree_final_summary": (
            f"{final}/final_summary_k8_k20.json"
        ),
        "x21_fixed_d8_degree_final_pool_manifest": (
            f"{final}/X21_scaling_final_seed86807_n196608.manifest.json"
        ),
        "x21_fixed_d8_degree_frozen_checkpoints": (
            f"{final}/FROZEN_CHECKPOINTS_K8_K20.sha256"
        ),
        "x21_fixed_d8_degree_evidence_ledger": (
            f"{final}/FINAL_EVIDENCE_K8_K20.sha256"
        ),
        "x21_fixed_d8_degree_embedded_amendment": (
            f"{final}/PROTOCOL_AMENDMENT_K8_K20.json"
        ),
    }
    new_outputs = {
        "x21_fixed_d8_degree_scaling_final_20260809.tex": (
            "gcicy paper/generated_tn/"
            "x21_fixed_d8_degree_scaling_final_20260809.tex"
        ),
        "x21_fixed_d8_degree_scaling_seedwise_20260809.tex": (
            "gcicy paper/generated_tn/"
            "x21_fixed_d8_degree_scaling_seedwise_20260809.tex"
        ),
        "summarize_type21_fixed_d8_degree_scaling.py": (
            "scripts/summarize_type21_fixed_d8_degree_scaling.py"
        ),
        "run_type21_fixed_d8_degree_scaling_remote.sh": (
            "scripts/run_type21_fixed_d8_degree_scaling_remote.sh"
        ),
        "finalize_type21_fixed_d8_degree_scaling_k8_k20_remote.sh": (
            "scripts/finalize_type21_fixed_d8_degree_scaling_k8_k20_remote.sh"
        ),
        "build_gcicy_tn_20260809_evidence_manifest.py": (
            "scripts/build_gcicy_tn_20260809_evidence_manifest.py"
        ),
    }

    for name, relative in new_sources.items():
        manifest["sources"][name] = record(relative)
    for name, relative in new_outputs.items():
        manifest["outputs"][name] = record(relative)

    # Refresh registered local files because manuscript-facing verification
    # scripts evolve together with each evidence release.
    for group in ("sources", "outputs", "inputs"):
        for name, item in list(manifest.get(group, {}).items()):
            manifest[group][name] = record(item["path"])

    manifest["notes"].append(
        "The fixed-D=8 X21 degree comparison contains three independently "
        "optimized checkpoints at each of k=8,12,16,20. All twelve were "
        "frozen before generation of the common independent 196608-point "
        "final sample. The registered k=24 arm was removed by a recorded "
        "protocol amendment after repeated low-level precision-stage "
        "process failures and before the final sample was generated; no "
        "k=24 diagnostic enters the reported comparison."
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
