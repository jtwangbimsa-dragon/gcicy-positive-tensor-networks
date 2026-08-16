#!/usr/bin/env python3
"""Build a standalone supplemental-material and claim-evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STEM = "gcicy_tn_supplement_evidence_20260809_final"
CURRENT_MANIFEST = (
    ROOT / "gcicy paper" / "generated_tn" / "current_evidence_manifest_20260807.json"
)

EXPLICIT = (
    ROOT / "gcicy paper" / "gcicy_tn_supplement.pdf",
    ROOT / "gcicy paper" / "gcicy_tn_supplement.tex",
    ROOT / "gcicy paper" / "gcicy_tn_supplement.bbl",
    ROOT / "gcicy paper" / "gcicy_tn_supplementNotes.bib",
    ROOT / "gcicy paper" / "supplement_geometry.tex",
    ROOT / "gcicy paper" / "supplement_controls.tex",
    ROOT / "gcicy paper" / "supplement_ritz.tex",
    ROOT / "gcicy paper" / "supplement_reproducibility.tex",
    ROOT / "gcicy paper" / "gcicy_tn_paper.tex",
    ROOT / "gcicy paper" / "Makefile",
    ROOT / "paper" / "references.bib",
    ROOT / "artifact_release_tn" / "README.md",
    ROOT / "artifact_release_tn" / "ENVIRONMENT.md",
    ROOT / "artifact_release_tn" / "DEPOSITION.md",
    ROOT / "artifact_release_tn" / "requirements.txt",
    ROOT / "outputs" / "reproducibility_evidence.json",
    ROOT / "outputs" / "section_restriction_rank_certificates.json",
    ROOT
    / "outputs"
    / "pipeline"
    / "type21_q121_d12_final_blind_20260729"
    / "final_blind_summary.json",
    ROOT
    / "outputs"
    / "pipeline"
    / "type21_q121_d12_final_blind_20260729"
    / "h4_vs_d8_final_blind_bootstrap_2000.json",
    ROOT
    / "outputs"
    / "pipeline"
    / "type21_q121_d12_final_blind_20260729"
    / "d8_vs_d12_final_blind_bootstrap_2000.json",
    ROOT
    / "outputs"
    / "pipeline"
    / "type11_h2_matched_timing_replay_20260729"
    / "runtime_summary.json",
)

SOURCE_GLOBS = (
    (ROOT / "gcicy_metric", "*.py"),
    (ROOT / "scripts", "*.py"),
    (ROOT / "scripts", "*.sh"),
    (ROOT / "pipeline_specs", "*.json"),
    (ROOT / "gcicy paper" / "generated_tn", "*.tex"),
    (ROOT / "gcicy paper" / "generated_tn", "*.json"),
    (ROOT / "gcicy paper" / "figures", "*.pdf"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> Path:
    resolved = path.resolve()
    try:
        result = resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"bundle path outside workspace: {path}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise SystemExit(f"bundle path is not a regular file: {path}")
    return result


def manifest_paths(path: Path) -> set[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths: set[Path] = {relative(path)}
    for group in ("sources", "outputs", "inputs"):
        for record in payload.get(group, {}).values():
            artifact = ROOT / record["path"]
            artifact_rel = relative(artifact)
            observed = sha256(artifact)
            if observed != record["sha256"]:
                raise SystemExit(
                    f"registered hash mismatch: {artifact_rel}: "
                    f"expected {record['sha256']}, observed {observed}"
                )
            paths.add(artifact_rel)
    return paths


def collect_paths() -> list[Path]:
    paths = {relative(path) for path in EXPLICIT}
    paths.update(manifest_paths(CURRENT_MANIFEST))
    for directory, pattern in SOURCE_GLOBS:
        paths.update(relative(path) for path in directory.rglob(pattern))
    return sorted(paths, key=lambda value: value.as_posix())


def readme_text(file_count: int) -> str:
    return f"""# Supplemental material and evidence bundle

This is the standalone evidence bundle for `Positive Tensor-Network Kahler
Metrics on Sequential gCICY Threefolds`. It contains the frozen supplemental
PDF, its complete LaTeX source tree, the bibliography, every generated table
used by the supplement, the claim-level JSON evidence bound by the current
SHA-256 evidence index, the numerical specifications, and the source code
named by the supplement.

## Primary file

`gcicy paper/gcicy_tn_supplement.pdf`

## Verify the bundle

From this directory run:

```text
python3 VERIFY_BUNDLE.py
python3 scripts/verify_gcicy_tn_current_evidence_manifest.py \\
  "gcicy paper/generated_tn/current_evidence_manifest_20260807.json"
python3 scripts/verify_gcicy_tn_manuscript_claims.py
```

The first command needs only Python's standard library and verifies all
{file_count} archived evidence/source files. The second verifies the 136
evidence-bound artifacts. The third reconstructs the paper-facing numerical
claims and exclusions from the included reports.

## Rebuild the PDF

With a TeX Live installation containing REVTeX 4.2:

```text
cd "gcicy paper"
latexmk -pdf -interaction=nonstopmode -halt-on-error gcicy_tn_supplement.tex
```

No file from the original working directory is needed for these checks or
for rebuilding the supplemental PDF. Large raw training arrays and model
checkpoints are not silently referenced by these verification commands; the
included claim-level reports record their identities and digests.
"""


def verifier_text() -> str:
    return '''#!/usr/bin/env python3
"""Verify every regular file listed in MANIFEST.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    payload = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {record["path"]: record for record in payload["files"]}
    failures = []
    for relative, record in expected.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        observed = sha256(path)
        if observed != record["sha256"]:
            failures.append(
                f"hash mismatch: {relative}: expected {record['sha256']}, "
                f"observed {observed}"
            )
    ignored = {"MANIFEST.json", "README.md", "VERIFY_BUNDLE.py"}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in ignored
        and "__pycache__" not in path.parts
    }
    extras = sorted(actual - set(expected))
    if extras:
        failures.extend(f"unregistered: {path}" for path in extras)
    if failures:
        raise SystemExit("\\n".join(failures))
    print(f"verified {len(expected)} bundle files")


if __name__ == "__main__":
    main()
'''


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def main() -> None:
    paths = collect_paths()
    output_dir = ROOT / "outputs" / "release"
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / STEM
    if staging.exists():
        raise SystemExit(f"refusing to overwrite existing bundle directory: {staging}")
    staging.mkdir()

    records = []
    for path in paths:
        source = ROOT / path
        destination = staging / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": path.as_posix(),
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
            }
        )

    manifest = {
        "schema": "gcicy-tn-supplement-evidence-bundle-v1",
        "release_id": STEM,
        "release_date": "2026-08-09",
        "supplement": {
            "path": "gcicy paper/gcicy_tn_supplement.pdf",
            "sha256": sha256(ROOT / "gcicy paper" / "gcicy_tn_supplement.pdf"),
        },
        "evidence_index": {
            "path": CURRENT_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": sha256(CURRENT_MANIFEST),
        },
        "files": records,
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging / "VERIFY_BUNDLE.py").write_text(verifier_text(), encoding="utf-8")
    (staging / "README.md").write_text(readme_text(len(records)), encoding="utf-8")

    archive_path = output_dir / f"{STEM}.zip"
    if archive_path.exists():
        raise SystemExit(f"refusing to overwrite existing archive: {archive_path}")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                relative_name = path.relative_to(output_dir).as_posix()
                archive.writestr(zip_info(relative_name), path.read_bytes())

    checksum = sha256(archive_path)
    checksum_path = output_dir / f"{STEM}.sha256"
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="ascii")
    print(f"directory={staging}")
    print(f"archive={archive_path}")
    print(f"sha256={checksum}")
    print(f"files={len(records)}")


if __name__ == "__main__":
    main()
