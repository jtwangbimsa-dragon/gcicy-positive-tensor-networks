#!/usr/bin/env python3
"""Build a deterministic ancillary ZIP for the positive-TN gCICY paper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STEM = "gcicy_positive_tn_artifacts_v1_0_0"
CURRENT_MANIFEST = (
    ROOT / "gcicy paper" / "generated_tn" / "current_evidence_manifest_20260809.json"
)
OPTIONAL_HEADLINE_DIR = ROOT / "outputs" / "submission_headline_artifacts_20260730"

EXPLICIT = (
    ROOT / "README.md",
    ROOT / "artifact_release_tn" / "README.md",
    ROOT / "artifact_release_tn" / "ENVIRONMENT.md",
    ROOT / "artifact_release_tn" / "DEPOSITION.md",
    ROOT / "artifact_release_tn" / "requirements.txt",
    ROOT / "gcicy paper" / "gcicy_tn_paper.tex",
    ROOT / "gcicy paper" / "gcicy_tn_supplement.tex",
    ROOT / "gcicy paper" / "supplement_geometry.tex",
    ROOT / "gcicy paper" / "supplement_controls.tex",
    ROOT / "gcicy paper" / "supplement_ritz.tex",
    ROOT / "gcicy paper" / "supplement_reproducibility.tex",
    ROOT / "gcicy paper" / "Makefile",
    ROOT / "paper" / "references.bib",
    ROOT / "output" / "pdf" / "gcicy_tn_paper_latest.pdf",
    ROOT / "output" / "pdf" / "gcicy_tn_supplement_latest.pdf",
    ROOT
    / "outputs"
    / "pipeline"
    / "gcicy_source_map_immersion_20260730"
    / "certificate.json",
)

SOURCE_GLOBS = (
    (ROOT / "gcicy_metric", "*.py"),
    (ROOT / "scripts", "*.py"),
    (ROOT / "scripts", "*.sh"),
    (ROOT / "pipeline_specs", "*.json"),
    (ROOT / "gcicy paper" / "generated_tn", "*.tex"),
    (ROOT / "gcicy paper" / "figures", "*.pdf"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> Path:
    path = path.resolve()
    try:
        result = path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"release path outside workspace: {path}") from exc
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"release path is not a regular file: {path}")
    return result


def manifest_paths(path: Path) -> set[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths: set[Path] = {relative(path)}
    for group in ("sources", "outputs", "inputs"):
        for entry in payload.get(group, {}).values():
            artifact = ROOT / entry["path"]
            artifact_rel = relative(artifact)
            if sha256(artifact) != entry["sha256"]:
                raise SystemExit(f"registered hash mismatch: {artifact_rel}")
            paths.add(artifact_rel)
    for artifact_path, entry in payload.get("generated", {}).items():
        artifact = ROOT / artifact_path
        artifact_rel = relative(artifact)
        if sha256(artifact) != entry["sha256"]:
            raise SystemExit(f"registered hash mismatch: {artifact_rel}")
        paths.add(artifact_rel)
    return paths


def collect_paths() -> list[Path]:
    paths = {relative(path) for path in EXPLICIT}
    paths.update(manifest_paths(CURRENT_MANIFEST))
    for directory, pattern in SOURCE_GLOBS:
        paths.update(relative(path) for path in directory.rglob(pattern))
    if OPTIONAL_HEADLINE_DIR.exists():
        paths.update(
            relative(path)
            for path in OPTIONAL_HEADLINE_DIR.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    return sorted(paths, key=lambda value: value.as_posix())


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def main() -> None:
    paths = collect_paths()
    manuscript = ROOT / "output" / "pdf" / "gcicy_tn_paper_latest.pdf"
    manifest = {
        "schema": "gcicy-positive-tn-submission-release-v1",
        "release_id": STEM,
        "release_date": "2026-08-09",
        "source_identity": {
            "scheme": "sha256_file_manifest",
            "git_commit": None,
            "note": "No verifiable Git history was present in the supplied workspace.",
        },
        "manuscript": {
            "path": manuscript.relative_to(ROOT).as_posix(),
            "sha256": sha256(manuscript),
            "bytes": manuscript.stat().st_size,
        },
        "files": [
            {
                "path": path.as_posix(),
                "sha256": sha256(ROOT / path),
                "bytes": (ROOT / path).stat().st_size,
            }
            for path in paths
        ],
        "external_repository": None,
        "external_doi": None,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    output_dir = ROOT / "outputs" / "release"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{STEM}.zip"
    temporary_path = output_dir / f".{STEM}.zip.tmp"
    prefix = f"{STEM}/"
    with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            archive.writestr(zip_info(prefix + path.as_posix()), (ROOT / path).read_bytes())
        archive.writestr(zip_info(prefix + "MANIFEST.json"), manifest_bytes)
    temporary_path.replace(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        expected = [prefix + path.as_posix() for path in paths] + [prefix + "MANIFEST.json"]
        if archive.namelist() != expected:
            raise SystemExit("archive inventory or order mismatch")
        for path in paths:
            archived = hashlib.sha256(archive.read(prefix + path.as_posix())).hexdigest()
            if archived != sha256(ROOT / path):
                raise SystemExit(f"archive hash mismatch: {path}")

    manifest_path = output_dir / f"{STEM}_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    checksum = sha256(archive_path)
    (output_dir / f"{STEM}.sha256").write_text(
        f"{checksum}  {archive_path.name}\n", encoding="ascii"
    )
    print(f"archive={archive_path}")
    print(f"sha256={checksum}")
    print(f"files={len(paths)}")


if __name__ == "__main__":
    main()
