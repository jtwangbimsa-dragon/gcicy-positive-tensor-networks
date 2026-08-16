#!/usr/bin/env python3
"""Verify that the replay archive is bound to the exact submission snapshot."""

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
    binding = json.loads(
        (ROOT / "SUBMISSION_BINDING.json").read_text(encoding="utf-8")
    )
    failures: list[str] = []
    for record in binding["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            failures.append(f"missing bound submission file: {record['path']}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"size mismatch: {record['path']}")
        observed = sha256(path)
        if observed != record["sha256"]:
            failures.append(f"hash mismatch: {record['path']}")

    manuscript = (ROOT / "manuscript/gcicy_tn_paper.tex").read_text(
        encoding="utf-8"
    )
    required = (
        "included fractionally when its weight crosses the one-percent threshold",
        "The final evaluation sample combines two",
        "their common final sample pairs only the Monte Carlo evaluation",
        "Its primary\ntraining stage completed and wrote a model",
        "cross-degree nonmonotonicity does not by itself separate those effects",
        "claim-reconstruction scripts are provided in a content-addressed Zenodo",
        "figures/gcicy_tn_x11_plateau_tail_survival.pdf",
    )
    forbidden = (
        "both fitted metrics and this choice were\nfixed before the independent",
        "No step beyond \\(k=12\\) is supported by all three paired",
        "first\nstage failed repeatedly at the process level before writing a single",
        "The nonmonotone\nlandscape is an optimization effect",
        "so the control is not limited by its\noptimization budget",
    )
    for fragment in required:
        if fragment not in manuscript:
            failures.append(f"required manuscript wording is absent: {fragment}")
    for fragment in forbidden:
        if fragment in manuscript:
            failures.append(f"superseded manuscript wording is present: {fragment}")

    figure = ROOT / "manuscript/figures/gcicy_tn_x11_plateau_tail_survival.pdf"
    canonical = (
        ROOT
        / "outputs/pipeline/type11_x11_equal_time_final_20260807/figures/"
        "gcicy_tn_x11_plateau_tail_survival.pdf"
    )
    if figure.is_file() and canonical.is_file() and sha256(figure) != sha256(canonical):
        failures.append("Figure 1 is not the final source-density comparison asset")
    expected_figure = binding["canonical_figure_1_sha256"]
    if figure.is_file() and sha256(figure) != expected_figure:
        failures.append("Figure 1 does not match the bound canonical digest")

    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"verified {len(binding['files'])} bound submission files, wording gates, "
        "and canonical Figure 1"
    )


if __name__ == "__main__":
    main()
