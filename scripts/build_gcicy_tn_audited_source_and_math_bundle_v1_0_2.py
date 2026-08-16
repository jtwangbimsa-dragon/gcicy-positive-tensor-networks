#!/usr/bin/env python3
"""Build one self-contained audited-source bundle with mathematical notes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs" / "release"
STEM = "gcicy_tn_audited_source_and_math_v1_0_4"
STAGING = RELEASE / STEM
ARCHIVE = RELEASE / f"{STEM}.zip"
BASE = RELEASE / "gcicy_tn_full_production_code_audit_v1_0_0.zip"
PATCH = RELEASE / "gcicy_tn_full_production_code_audit_patch_v1_0_1.zip"
BASE_SHA256 = "0416632a36a0608f560d17661d23e52ad31c4560c3e39968789955bc25142af6"
PATCH_SHA256 = "e52c01764ef751992c746a2f9b8ea4edbd1f38be046cc1dc76ad0f0791e94956"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_payload(archive: Path, expected_root: str, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = Path(member.filename)
            if not path.parts or path.parts[0] != expected_root or member.is_dir():
                continue
            relative = Path(*path.parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def verifier_text() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
expected = {row["path"]: row for row in manifest["files"]}
actual = {
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_file() and path.name not in {"MANIFEST.json", "VERIFY_BUNDLE.py"}
}
failures = []
if actual != set(expected):
    failures.append(
        f"file-set mismatch: missing={sorted(set(expected)-actual)}, "
        f"extra={sorted(actual-set(expected))}"
    )
for relative, row in expected.items():
    path = ROOT / relative
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != row["bytes"] or digest != row["sha256"]:
            failures.append(f"content mismatch: {relative}")
if failures:
    raise SystemExit("\\n".join(failures))
print(f"verified {len(expected)} files in {manifest['release_id']}")
'''


def readme_text() -> str:
    return f'''# Audited gCICY metric production source and mathematical guide

This is the single-package delivery requested for independent source review.
It contains the complete production-code audit tree, the post-audit source
corrections, tests, mathematical explanations, production DAG and independent
audit reports.  It contains no large point arrays.  Two compact historical
model artifacts are included solely because a source-level protocol regression
test compares their epoch-zero re-expression; manuscript evidence and large
frozen checkpoints remain in the separate final reproducibility archive.

## Start here

1. `docs/full_production_code_audit/MATHEMATICAL_IMPLEMENTATION_GUIDE.md`
   explains the mathematics implemented by each production subsystem.
2. `docs/full_production_code_audit/FULL_PRODUCTION_DAG.md` identifies the
   scripts used at every stage of the X11, X21 and X22 calculations.
3. `docs/full_production_code_audit/AUDIT_INSTRUCTIONS.md` gives the intended
   independent-review procedure.
4. `audit_reports/gcicy_full_production_joint_postfix_audit_20260815.md`
   gives the consolidated findings and their effect on the paper.
5. `docs/full_production_code_audit/POST_AUDIT_PATCH_NOTES.md` lists every
   source correction made after the frozen historical audit.

Run `python VERIFY_BUNDLE.py` before reviewing, followed by
`python RUN_CRITICAL_TESTS.py`.  In the pinned project environment the latter
passes 184 source-level tests.  Tests that reconstruct manuscript numbers from
multi-gigabyte pointwise arrays belong to the separate final reproducibility
archive and are deliberately excluded from this source-only package.

## Provenance

The reported computations remain tied to the immutable historical archive
with SHA-256 `{BASE_SHA256}`.  The corrected source in this package is that
archive overlaid by patch SHA-256 `{PATCH_SHA256}`.  Original manifests and
release identifiers are retained under `provenance/`.
'''


def main() -> None:
    if sha256(BASE) != BASE_SHA256:
        raise RuntimeError("historical base archive is missing or changed")
    if sha256(PATCH) != PATCH_SHA256:
        raise RuntimeError("post-audit patch archive is missing or changed")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    extract_payload(BASE, "gcicy_tn_full_production_code_audit_v1_0_0", STAGING)

    provenance = STAGING / "provenance"
    provenance.mkdir()
    for name in ("MANIFEST.json", "README.md", "VERIFY_BUNDLE.py"):
        path = STAGING / name
        if path.exists():
            shutil.move(path, provenance / f"historical_v1_0_0_{name}")
    for name in (
        "AUDIT_INSTRUCTIONS.md",
        "FULL_PRODUCTION_DAG.md",
        "MATHEMATICAL_IMPLEMENTATION_GUIDE.md",
    ):
        path = STAGING / name
        if path.exists():
            path.unlink()

    extract_payload(PATCH, "gcicy_tn_full_production_code_audit_patch_v1_0_1", STAGING)
    for name in ("MANIFEST.json", "BASE_RELEASE.json"):
        path = STAGING / name
        if path.exists():
            shutil.move(path, provenance / f"patch_v1_0_1_{name}")
    stale_patch_verifier = STAGING / "VERIFY_PATCH.py"
    if stale_patch_verifier.exists():
        stale_patch_verifier.unlink()

    # Restore the reviewer guide omitted from v1.0.1.
    shutil.copy2(
        ROOT / "docs/full_production_code_audit/AUDIT_INSTRUCTIONS.md",
        STAGING / "docs/full_production_code_audit/AUDIT_INSTRUCTIONS.md",
    )

    # These compact records are the complete external-data dependency of the
    # retained source-level tests.  Large manuscript arrays remain in the
    # separate final reproducibility archive.
    fixtures = (
        ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k4_l2_tail_refined_v3_gpu_summary.json",
        ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k4_l2_tail_v3_seed69211_serial_failure_diagnostic.json",
        ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k4_l2_tail_refined_v2_gpu.npz",
        ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k4_l2_tail_refined_v3_gpu.npz",
    )
    for fixture in fixtures:
        fixture_target = STAGING / fixture.relative_to(ROOT)
        fixture_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture, fixture_target)

    # Manuscript-array replay is tested in the multi-gigabyte evidence bundle,
    # not in this package, which intentionally contains no large arrays.
    evidence_test = STAGING / "tests/test_final_reproducibility.py"
    if evidence_test.exists():
        evidence_test.unlink()
    fixture_scope = {
        "schema": "gcicy-tn-code-audit-test-fixtures-v3",
        "manuscript_pointwise_evidence_included": False,
        "compact_protocol_test_artifacts_included": True,
        "purpose": "Minimal deterministic fixtures required by source-level tests.",
        "excluded_test": "tests/test_final_reproducibility.py",
        "excluded_test_reason": (
            "It reconstructs manuscript claims from multi-gigabyte pointwise arrays "
            "provided by the separate final reproducibility archive."
        ),
        "files": sorted(
            path.relative_to(STAGING).as_posix()
            for path in (STAGING / "outputs").rglob("*")
            if path.is_file()
        ),
    }
    (STAGING / "TEST_FIXTURE_SCOPE.json").write_text(
        json.dumps(fixture_scope, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    provenance_record = {
        "release_id": STEM,
        "historical_base": BASE.name,
        "historical_base_sha256": BASE_SHA256,
        "post_audit_patch": PATCH.name,
        "post_audit_patch_sha256": PATCH_SHA256,
        "reported_results_remain_tied_to_historical_base": True,
        "large_pointwise_data_included": False,
        "compact_protocol_test_artifacts_included": True,
    }
    (provenance / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(provenance_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (STAGING / "README.md").write_text(readme_text(), encoding="utf-8")

    payloads = sorted(
        path
        for path in STAGING.rglob("*")
        if path.is_file() and path.name not in {"MANIFEST.json", "VERIFY_BUNDLE.py"}
    )
    records = [
        {
            "path": path.relative_to(STAGING).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in payloads
    ]
    manifest = {
        "schema": "gcicy-tn-audited-source-and-math-v1",
        "release_id": STEM,
        "historical_base_sha256": BASE_SHA256,
        "post_audit_patch_sha256": PATCH_SHA256,
        "experimental_data_included": False,
        "files": records,
    }
    (STAGING / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (STAGING / "VERIFY_BUNDLE.py").write_text(verifier_text(), encoding="utf-8")

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                output.write(path, f"{STEM}/{path.relative_to(STAGING).as_posix()}")
    checksum = sha256(ARCHIVE)
    (RELEASE / f"{STEM}.sha256").write_text(
        f"{checksum}  {ARCHIVE.name}\n", encoding="ascii"
    )
    print(ARCHIVE)
    print(checksum)


if __name__ == "__main__":
    main()
