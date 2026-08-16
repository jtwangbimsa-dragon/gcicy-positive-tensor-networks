#!/usr/bin/env python3
"""Build a deterministic, content-addressed ancillary computation archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "gcicy-metric-artifact-v1"
ARCHIVE_STEM = "gcicy_metric_artifact_v1"
EVIDENCE_FILES = (
    ROOT / "outputs/publication_evidence_summary.json",
    ROOT / "outputs/multitype_publication_evidence.json",
    ROOT / "outputs/reproducibility_evidence.json",
    ROOT / "outputs/reviewer_round4_publication_evidence.json",
)
EXPLICIT_FILES = (
    ROOT / "README.md",
    ROOT / "artifact_release/README.md",
    ROOT / "artifact_release/ENVIRONMENT.md",
    ROOT / "artifact_release/DEPOSITION.md",
    ROOT / "artifact_release/environment-lock.json",
    ROOT / "artifact_release/requirements.txt",
    ROOT / "paper/Makefile",
    ROOT / "output/pdf/gcicy_metric_paper_draft.pdf",
)
SOURCE_PATTERNS = (
    (ROOT / "gcicy_metric", "*.py"),
    (ROOT / "scripts", "*.py"),
    (ROOT / "scripts", "*.sh"),
    (ROOT / "tests", "*.py"),
    (ROOT / "pipeline_specs", "*.json"),
    (ROOT / "paper", "*.tex"),
    (ROOT / "paper", "*.bib"),
    (ROOT / "paper/generated", "*.md"),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_local_path(value: str | Path) -> Path:
    supplied = Path(value)
    path = (supplied if supplied.is_absolute() else ROOT / supplied).resolve()
    try:
        relative = path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"release source is outside the workspace: {path}") from exc
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"release source is not a regular file: {path}")
    return relative


def source_paths() -> list[Path]:
    paths = {relative_local_path(path) for path in (*EVIDENCE_FILES, *EXPLICIT_FILES)}
    for directory, pattern in SOURCE_PATTERNS:
        paths.update(relative_local_path(path) for path in directory.rglob(pattern))
    for evidence_path in EVIDENCE_FILES:
        if not evidence_path.exists():
            raise SystemExit(f"missing evidence index: {evidence_path}")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        for entry in evidence["source_files"]:
            value = entry.get("path", entry.get("file"))
            if value is None:
                raise SystemExit(f"source entry has no path: {entry}")
            relative = relative_local_path(value)
            expected = entry.get("sha256")
            if expected is not None and sha256(ROOT / relative) != expected:
                raise SystemExit(f"evidence source hash changed: {relative}")
            paths.add(relative)
    return sorted(paths, key=lambda path: path.as_posix())


def manifest_bytes(paths: list[Path]) -> bytes:
    manuscript = ROOT / "output/pdf/gcicy_metric_paper_draft.pdf"
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "release_date": "2026-07-13",
        "description": (
            "Content-addressed ancillary code, metric artifacts, raw audit rows, "
            "exact certificates, specifications, and evidence indices for the "
            "sequential gCICY metric manuscript."
        ),
        "source_identity": {
            "scheme": "sha256_file_manifest",
            "git_commit": None,
            "note": (
                "No verifiable Git history is present in the supplied workspace; "
                "the complete file manifest is the source identity."
            ),
        },
        "manuscript_binding": {
            "path": manuscript.relative_to(ROOT).as_posix(),
            "bytes": manuscript.stat().st_size,
            "sha256": sha256(manuscript),
        },
        "files": [
            {
                "path": path.as_posix(),
                "bytes": (ROOT / path).stat().st_size,
                "sha256": sha256(ROOT / path),
            }
            for path in paths
        ],
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def verify_archive(
    archive_path: Path,
    paths: list[Path],
    manifest_payload: bytes,
) -> None:
    prefix = f"{ARCHIVE_STEM}/"
    expected_names = [f"{prefix}{path.as_posix()}" for path in paths]
    expected_names.append(f"{prefix}MANIFEST.json")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != expected_names:
            raise SystemExit("archive member order or inventory mismatch")
        for path in paths:
            payload = archive.read(f"{prefix}{path.as_posix()}")
            if sha256_bytes(payload) != sha256(ROOT / path):
                raise SystemExit(f"archive member hash mismatch: {path}")
        if archive.read(f"{prefix}MANIFEST.json") != manifest_payload:
            raise SystemExit("embedded manifest mismatch")


def main() -> None:
    paths = source_paths()
    manifest_payload = manifest_bytes(paths)
    output_dir = ROOT / "outputs/release"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{ARCHIVE_STEM}.zip"
    temporary_path = output_dir / f".{ARCHIVE_STEM}.zip.tmp"
    prefix = f"{ARCHIVE_STEM}/"
    with zipfile.ZipFile(
        temporary_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in paths:
            archive.writestr(zip_info(f"{prefix}{path.as_posix()}"), (ROOT / path).read_bytes())
        archive.writestr(zip_info(f"{prefix}MANIFEST.json"), manifest_payload)
    temporary_path.replace(archive_path)
    verify_archive(archive_path, paths, manifest_payload)

    manifest_path = output_dir / f"{ARCHIVE_STEM}_manifest.json"
    manifest_path.write_bytes(manifest_payload)
    archive_hash = sha256(archive_path)
    checksum_path = output_dir / f"{ARCHIVE_STEM}.sha256"
    checksum_path.write_text(
        f"{archive_hash}  {archive_path.name}\n",
        encoding="ascii",
    )
    summary = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "archive": archive_path.relative_to(ROOT).as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_hash,
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_bytes(manifest_payload),
        "manuscript": "output/pdf/gcicy_metric_paper_draft.pdf",
        "manuscript_sha256": sha256(
            ROOT / "output/pdf/gcicy_metric_paper_draft.pdf"
        ),
        "file_count": len(paths),
        "deterministic_zip_verified": True,
        "external_doi": None,
    }
    summary_path = output_dir / f"{ARCHIVE_STEM}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"archive={archive_path}")
    print(f"sha256={archive_hash}")
    print(f"files={len(paths)}")


if __name__ == "__main__":
    main()
