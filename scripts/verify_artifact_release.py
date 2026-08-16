#!/usr/bin/env python3
"""Verify that the ancillary ZIP is content-correct and self-contained."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_STEM = "gcicy_metric_artifact_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "outputs/release" / f"{ARCHIVE_STEM}.zip",
    )
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run(root: Path, command: list[str]) -> None:
    subprocess.run(command, cwd=root, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    archive_path = args.archive.expanduser().resolve()
    prefix = f"{ARCHIVE_STEM}/"
    required_members = {
        "gcicy_metric/pipeline/__init__.py",
        "gcicy_metric/pipeline/adapter.py",
        "gcicy_metric/pipeline/adapters/p4p1_hirzebruch_11.py",
        "gcicy_metric/simple_patch.py",
        "gcicy_metric/global_sections.py",
        "scripts/check_exact_smoothness.py",
        "scripts/check_type21_exact_smoothness.py",
        "scripts/certify_section_restriction_ranks.py",
        "scripts/certify_topological_targets.py",
        "scripts/run_sampler_audit_seed_parallel.py",
        "scripts/run_scalar_laplacian_seed_parallel.py",
        "scripts/audit_metric_volume_weight_tail.py",
        "scripts/run_metric_volume_tail_seed_parallel.py",
        "scripts/build_round4_publication_evidence.py",
        "scripts/build_artifact_release.py",
        "scripts/audit_paper_build.py",
        "scripts/record_run_provenance.py",
        "scripts/run_round4_hard_region_followup_remote.sh",
        "scripts/run_round4_multiseed_refinement_remote.sh",
        "scripts/run_round4_l2_tail_resolution_remote.sh",
        "scripts/run_round4_l2_tail_resolution_v2_remote.sh",
        "scripts/run_round4_l2_tail_resolution_v3_remote.sh",
        "scripts/run_round4_v3_failure_serial_diagnosis_remote.sh",
        "scripts/run_round4_l2_tail_resolution_v4_remote.sh",
        "scripts/run_round4_x22_recovery_remote.sh",
        "scripts/run_round4_x22_atlas_witness_remote.sh",
        "scripts/run_round4_x22_atlas_resolution_remote.sh",
        "scripts/run_round4_publication_complete_remote.sh",
        "scripts/verify_release.sh",
        "scripts/reproduce_small_remote.sh",
        "scripts/reproduce_full_remote.sh",
        "scripts/verify_artifact_release.py",
        "artifact_release/environment-lock.json",
        "output/pdf/gcicy_metric_paper_draft.pdf",
        "paper/main.tex",
        "paper/references.bib",
        "pipeline_specs/reviewer_round4_publication_manifest.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k2_tail_matched_gpu.json",
        "pipeline_specs/p4p1_type11_hirzebruch_scalar_metric_systematic_tail_matched_32768.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_hard_region_blind_scalar.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_multiseed_tail_refine.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_multiseed_scalar_65536.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_refine.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_discovery_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_scalar.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_scalar_65536.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_refine_v2.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_discovery_v2_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v2_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v2_scalar.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_v2_scalar_65536.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_refine_v3.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_discovery_v3_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v3_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v3_scalar.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_v3_scalar_65536.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_refine_v4.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_checkpoint_v4_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_discovery_v4_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_check_v4_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v4_audit.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v4_scalar.json",
        "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_v4_scalar_65536.json",
        "pipeline_specs/p1p1p5_type22_seed20260712_k1_k3_audit.json",
        "tests/test_gcicy_pipeline.py",
        "tests/test_round4_v4_protocol.py",
        "tests/test_type21_p5p1_candidate.py",
    }
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise SystemExit("ZIP integrity test failed")
        manifest = json.loads(archive.read(f"{prefix}MANIFEST.json"))
        entries = {entry["path"]: entry for entry in manifest["files"]}
        missing = sorted(required_members - entries.keys())
        if missing:
            raise SystemExit(f"release import/test closure is incomplete: {missing}")
        manuscript_binding = manifest.get("manuscript_binding", {})
        manuscript_path = manuscript_binding.get("path")
        if manuscript_path != "output/pdf/gcicy_metric_paper_draft.pdf":
            raise SystemExit("release manuscript binding is missing or invalid")
        manuscript_payload = archive.read(f"{prefix}{manuscript_path}")
        if (
            len(manuscript_payload) != int(manuscript_binding["bytes"])
            or sha256_bytes(manuscript_payload) != manuscript_binding["sha256"]
        ):
            raise SystemExit("release manuscript binding does not match its payload")
        for relative, entry in entries.items():
            payload = archive.read(f"{prefix}{relative}")
            if len(payload) != int(entry["bytes"]):
                raise SystemExit(f"release member size mismatch: {relative}")
            if sha256_bytes(payload) != entry["sha256"]:
                raise SystemExit(f"release member hash mismatch: {relative}")

        with tempfile.TemporaryDirectory(prefix="gcicy-release-") as temporary:
            archive.extractall(temporary)
            extracted_root = Path(temporary) / ARCHIVE_STEM
            run(
                extracted_root,
                [
                    sys.executable,
                    "-c",
                    (
                        "import gcicy_metric; "
                        "import gcicy_metric.pipeline; "
                        "import gcicy_metric.pipeline.adapters; "
                        "import gcicy_metric.simple_patch; "
                        "import gcicy_metric.global_sections"
                    ),
                ],
            )
            rebuild_targets = (
                "outputs/publication_evidence_summary.json",
                "outputs/multitype_publication_evidence.json",
                "outputs/reproducibility_evidence.json",
                "outputs/reviewer_round4_publication_evidence.json",
                "paper/generated/results_tables.md",
                "paper/generated/results_tables.tex",
                "paper/generated/multitype_tables.md",
                "paper/generated/multitype_tables.tex",
                "paper/generated/reproducibility_tables.tex",
                "paper/generated/round4_tables.md",
                "paper/generated/round4_tables.tex",
            )
            committed_hashes = {
                relative: sha256(extracted_root / relative)
                for relative in rebuild_targets
            }
            for builder in (
                "scripts/build_publication_evidence.py",
                "scripts/build_multitype_publication_evidence.py",
                "scripts/build_reproducibility_evidence.py",
                "scripts/build_round4_publication_evidence.py",
            ):
                run(extracted_root, [sys.executable, builder])
            rebuilt_hashes = {
                relative: sha256(extracted_root / relative)
                for relative in rebuild_targets
            }
            changed = [
                relative
                for relative in rebuild_targets
                if committed_hashes[relative] != rebuilt_hashes[relative]
            ]
            if changed:
                raise SystemExit(f"evidence rebuild is not byte-identical: {changed}")
            if not args.skip_tests:
                run(
                    extracted_root,
                    [
                        sys.executable,
                        "-m",
                        "unittest",
                        "tests.test_gcicy_pipeline",
                        "tests.test_round4_v4_protocol",
                        "tests.test_type21_p5p1_candidate",
                    ],
                )

    print(
        json.dumps(
            {
                "archive": archive_path.name,
                "payload_files": len(manifest["files"]),
                "manifest_and_payload_hashes_verified": True,
                "self_contained_imports_verified": True,
                "evidence_rebuild_verified": True,
                "regression_tests_run": not args.skip_tests,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
