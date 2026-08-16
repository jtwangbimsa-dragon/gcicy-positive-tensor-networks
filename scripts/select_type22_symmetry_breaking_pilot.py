#!/usr/bin/env python3
"""Select a preregistered type-(2,2) symmetry-breaking pilot arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = ("noise003", "noise010", "noise030")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    rows = []
    for arm in ARMS:
        summary_path = run_dir / f"pilot_{arm}_summary.json"
        spectrum_path = run_dir / f"pilot_{arm}_core_spectrum.json"
        model_path = run_dir / f"pilot_{arm}.pt"
        summary = load_json(summary_path)
        spectrum_report = load_json(spectrum_path)
        spectrum = spectrum_report["spectrum"]
        metadata = spectrum_report["metadata"]
        minimum_eigenvalue = float(
            summary["best_validation"]["minimum_metric_eigenvalue"]
        )
        eligible = bool(
            minimum_eigenvalue > 0
            and spectrum["numerical_rank"] >= 2
            and metadata["exactly_zero_bond_block_count"] == 0
        )
        rows.append(
            {
                "arm": arm,
                "initialization_noise": manifest["pilot_arms"][arm][
                    "initialization_noise"
                ],
                "model": str(model_path),
                "model_sha256": sha256_file(model_path),
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "spectrum": str(spectrum_path),
                "spectrum_sha256": sha256_file(spectrum_path),
                "best_epoch": int(summary["best_epoch"]),
                "best_validation_selection_score": float(
                    summary["best_validation_selection_score"]
                ),
                "best_validation_chi": float(
                    summary["best_validation"][
                        "fixed_teacher_kappa_sqrt_ma_energy"
                    ]
                ),
                "minimum_metric_eigenvalue": minimum_eigenvalue,
                "numerical_core_rank": int(spectrum["numerical_rank"]),
                "entropy_effective_core_rank": float(
                    spectrum["entropy_effective_rank"]
                ),
                "participation_effective_core_rank": float(
                    spectrum["participation_effective_rank"]
                ),
                "exactly_zero_bond_block_count": int(
                    metadata["exactly_zero_bond_block_count"]
                ),
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise SystemExit("no symmetry-breaking pilot arm passed the validity gates")
    selected = min(
        eligible_rows,
        key=lambda row: (row["best_validation_selection_score"], row["initialization_noise"]),
    )
    report = {
        "schema": "type22-tensor-network-symmetry-breaking-selection-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_rule": (
            "minimum best validation selection score among positive-metric arms "
            "with numerical core rank at least two and no exactly zero bond block; "
            "smaller initialization noise breaks exact ties"
        ),
        "arms": rows,
        "selected_arm": selected["arm"],
        "selected_model": selected["model"],
        "selected_model_sha256": selected["model_sha256"],
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
