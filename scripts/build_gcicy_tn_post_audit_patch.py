#!/usr/bin/env python3
"""Build the manifest-verified post-audit source patch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/release"
STEM = "gcicy_tn_full_production_code_audit_patch_v1_0_1"
STAGING = OUT / STEM
BASE_ARCHIVE = "gcicy_tn_full_production_code_audit_v1_0_0.zip"
BASE_SHA256 = "0416632a36a0608f560d17661d23e52ad31c4560c3e39968789955bc25142af6"

FILES = (
    "pytest.ini",
    "gcicy_metric/pipeline/adapter.py",
    "gcicy_metric/pipeline/algebraic_power_lift.py",
    "gcicy_metric/pipeline/parallel_sampling.py",
    "gcicy_metric/pipeline/projective_residual_phi.py",
    "gcicy_metric/pipeline/adapters/p1p1p5_22.py",
    "gcicy_metric/pipeline/adapters/p4p1_hirzebruch_11.py",
    "gcicy_metric/pipeline/adapters/p4p1p1_hirzebruch_21.py",
    "gcicy_metric/generic_model.py",
    "gcicy_metric/type21_hirzebruch_x3.py",
    "scripts/build_gcicy_tn_final_manuscript_assets.py",
    "scripts/build_gcicy_tn_full_production_code_audit.py",
    "scripts/build_gcicy_tn_post_audit_patch.py",
    "scripts/generate_gcicy_common_point_pool.py",
    "tests/test_algebraic_metric_power_lift.py",
    "tests/test_final_reproducibility.py",
    "tests/test_gcicy_pipeline.py",
    "tests/test_projective_residual_phi.py",
    "tests/test_quintic_compact_root_native_ma.py",
    "docs/full_production_code_audit/FULL_PRODUCTION_DAG.md",
    "docs/full_production_code_audit/MATHEMATICAL_IMPLEMENTATION_GUIDE.md",
    "docs/full_production_code_audit/POST_AUDIT_PATCH_NOTES.md",
    "audit_reports/local_gpt56_sol_ultra_full_production_code_audit_20260814.md",
    "audit_reports/remote_deepseek_v4pro_full_production_code_audit_full_20260814.md",
    "audit_reports/remote_deepseek_v4pro_postfix_audit_20260815.md",
    "audit_reports/gcicy_full_production_joint_postfix_audit_20260815.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if path.is_file() and path.name not in {"MANIFEST.json", "VERIFY_PATCH.py"}
}
failures = []
if actual != set(expected):
    failures.append(
        f"file-set mismatch: missing={sorted(set(expected)-actual)}, "
        f"extra={sorted(actual-set(expected))}"
    )
for relative, row in expected.items():
    path = ROOT / relative
    if path.is_file() and (
        path.stat().st_size != row["bytes"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
    ):
        failures.append(f"content mismatch: {relative}")
if failures:
    raise SystemExit("\\n".join(failures))
print(
    f"verified {len(expected)} patch files; "
    f"base={manifest['base_archive_sha256']}"
)
'''


def main() -> None:
    base = OUT / BASE_ARCHIVE
    if not base.is_file() or sha256(base) != BASE_SHA256:
        raise RuntimeError("the immutable base archive is absent or has changed")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = STAGING / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (STAGING / "BASE_RELEASE.json").write_text(
        json.dumps(
            {
                "base_archive": BASE_ARCHIVE,
                "base_archive_sha256": BASE_SHA256,
                "patch_release": STEM,
                "reported_results_remain_tied_to_base": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payloads = sorted(
        path
        for path in STAGING.rglob("*")
        if path.is_file() and path.name not in {"MANIFEST.json", "VERIFY_PATCH.py"}
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
        "schema": "gcicy-tn-post-audit-source-patch-v1",
        "release_id": STEM,
        "base_archive": BASE_ARCHIVE,
        "base_archive_sha256": BASE_SHA256,
        "experimental_data_included": False,
        "files": records,
    }
    (STAGING / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (STAGING / "VERIFY_PATCH.py").write_text(verifier_text(), encoding="utf-8")

    archive = OUT / f"{STEM}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                output.write(path, f"{STEM}/{path.relative_to(STAGING).as_posix()}")
    checksum = sha256(archive)
    (OUT / f"{STEM}.sha256").write_text(
        f"{checksum}  {archive.name}\n",
        encoding="ascii",
    )
    print(f"wrote {archive}")
    print(checksum)


if __name__ == "__main__":
    main()
