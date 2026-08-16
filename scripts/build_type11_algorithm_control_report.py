#!/usr/bin/env python3
"""Build the paired type-(1,1) optimizer-mechanism comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.audit import effective_sample_size, weighted_quantile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pipeline" / "type11_algorithm_controls_20260717",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tail_statistics(path: Path) -> dict[str, float | int]:
    with np.load(path, allow_pickle=False) as payload:
        weights = np.asarray(payload["importance_weights"], dtype=np.float64)
        ratio = np.asarray(payload["normalized_volume_ratio"], dtype=np.float64)
    normalized_weights = weights / np.sum(weights)
    metric_point_mass = normalized_weights * ratio
    metric_point_mass /= np.sum(metric_point_mass)
    clusters = np.arange(len(ratio), dtype=np.int64) // 4
    fibre_mass = np.bincount(clusters, weights=metric_point_mass)
    error = ratio - 1.0
    return {
        "point_count": len(ratio),
        "sigma": float(np.sum(normalized_weights * np.abs(error))),
        "chi_l2": float(np.sqrt(np.sum(normalized_weights * error**2))),
        "ratio_q999": weighted_quantile(ratio, weights, 0.999),
        "max_ratio": float(np.max(ratio)),
        "ratio_above_3_weighted_mass": float(
            np.sum(normalized_weights[ratio > 3.0])
        ),
        "ratio_above_3_point_count": int(np.count_nonzero(ratio > 3.0)),
        "ratio_above_3_fibre_count": int(len(np.unique(clusters[ratio > 3.0]))),
        "metric_volume_point_ess": effective_sample_size(metric_point_mass),
        "metric_volume_maximum_point_mass": float(np.max(metric_point_mass)),
        "metric_volume_fibre_ess": effective_sample_size(fibre_mass),
        "metric_volume_maximum_fibre_mass": float(np.max(fibre_mass)),
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    eig_dir = ROOT / "outputs" / "pipeline" / "type11_eigensection_switching_20260717"
    paths = {
        "point64_l1": {
            "arrays": eig_dir / "baseline_seed72003.npz",
            "audit": eig_dir / "baseline_seed72003.json",
            "artifact": ROOT
            / "outputs/pipeline/p4p1_type11_hirzebruch_k4_pointb64_baseline_e3.npz",
        },
        "fullpool_e2": {
            "arrays": out_dir / "e2_fullpool_k4_seed72003_audit.npz",
            "audit": out_dir / "e2_fullpool_k4_seed72003_audit.json",
            "artifact": out_dir / "e2_fullpool_k4_train72001_e1000.npz",
            "training_summary": out_dir
            / "e2_fullpool_k4_train72001_e1000_summary.json",
        },
        "nu_balanced": {
            "arrays": out_dir / "nu_balanced_k4_seed72003_audit.npz",
            "audit": out_dir / "nu_balanced_k4_seed72003_audit.json",
            "artifact": out_dir / "nu_balanced_k4_train72001_c128.npz",
            "training_summary": out_dir
            / "nu_balanced_k4_train72001_c128_summary.json",
        },
    }
    paired_path = out_dir / "paired_blind_seeds73011_73012.json"
    required = [paired_path]
    for row in paths.values():
        required.extend(Path(value) for value in row.values())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing report inputs: " + ", ".join(missing))

    labels = {
        "point64_l1": "point64 L1 + Adam",
        "fullpool_e2": "full-pool E2 + Adam",
        "nu_balanced": "nu-balanced T-map",
    }
    known_seed_rows: dict[str, Any] = {}
    mechanism_rows: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for key, row in paths.items():
        audit = json.loads(Path(row["audit"]).read_text(encoding="utf-8"))
        known_seed_rows[key] = finite_tail_statistics(Path(row["arrays"]))
        mechanism_rows[key] = {
            "reference_relative_h_spectrum": audit["reference_relative_h_spectrum"],
            "correlations": audit["correlations"],
            "bulk": audit["groups"]["bulk_bottom_90_percent_by_count"],
            "top_1_percent": audit["groups"]["top_1_percent_by_count"],
            "worst_point": audit["worst_point"],
        }
        provenance[key] = {
            name: {
                "path": str(Path(value)),
                "sha256": sha256_file(Path(value)),
            }
            for name, value in row.items()
        }

    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    artifact_name_to_key = {
        "p4p1_type11_hirzebruch_k4_pointb64_baseline_e3.npz": "point64_l1",
        "e2_fullpool_k4_train72001_e1000.npz": "fullpool_e2",
        "nu_balanced_k4_train72001_c128.npz": "nu_balanced",
    }
    paired_rows = {}
    for row in paired["artifacts"]:
        key = artifact_name_to_key[Path(row["path"]).name]
        paired_rows[key] = {
            "mean_sigma": row["mean_sigma"],
            "mean_chi_l2": row["mean_sqrt_squared_energy"],
            "maximum_seed_chi_l2": row["max_seed_sqrt_squared_energy"],
            "maximum_ratio": row["max_normalized_ratio"],
            "mean_positive_log_ratio_q999": row["mean_positive_log_ratio_q999"],
            "maximum_ratio_above_3_weighted_mass": row[
                "max_seed_normalized_ratio_above_3_weighted_mass"
            ],
            "minimum_metric_volume_point_ess": row[
                "min_metric_volume_effective_sample_size"
            ],
            "seeds": row["seeds"],
        }

    report = {
        "schema": "type11-algorithm-control-report-v1",
        "model_seed": 20260731,
        "degree": [4, 4],
        "section_count": 274,
        "known_failure_seed": 72003,
        "known_failure_seed_rows": known_seed_rows,
        "paired_blind_seeds": [73011, 73012],
        "paired_blind_rows": paired_rows,
        "mechanism_rows": mechanism_rows,
        "atlas": {
            "projective_charts_seen": paired["atlas"]["projective_charts_seen"],
            "projective_charts_expected": paired["atlas"][
                "projective_charts_expected"
            ],
            "implicit_choices_seen": paired["atlas"][
                "implicit_coordinate_choices_seen"
            ],
            "implicit_choices_expected": paired["atlas"][
                "implicit_coordinate_choices_expected"
            ],
            "maximum_baseline_projective_ma_error": paired["atlas"][
                "max_baseline_projective_ma_error"
            ],
            "maximum_baseline_implicit_ma_error": paired["atlas"][
                "max_baseline_implicit_ma_error"
            ],
            "maximum_projective_importance_log_weight_error": paired["atlas"][
                "max_projective_importance_log_weight_error"
            ],
            "maximum_implicit_importance_log_weight_error": paired["atlas"][
                "max_implicit_importance_log_weight_error"
            ],
        },
        "conclusions": {
            "eigensection_switching": (
                "Rejected as the primary explanation for the point64 ridge: its worst "
                "point has many active eigensections, and the high-ratio top-two gap is "
                "not smaller than the bulk gap."
            ),
            "spectral_conditioning": (
                "A large reference-relative H condition number is not sufficient for an "
                "MA spike: the nu-balanced solution is far more spectrally anisotropic "
                "but has much smaller paired blind tails."
            ),
            "sampling_geometry": (
                "The paired atlas and importance-weight invariance checks pass at roughly "
                "1e-10 or better, so the observed optimizer differences are not explained "
                "by chart, Jacobian, or point-weight inconsistencies."
            ),
            "squared_loss": (
                "Full-pool E2 strongly improves its fixed training pool but retains large "
                "blind tails; a tail-sensitive loss alone does not solve the undersampled "
                "75075-dimensional optimization problem."
            ),
            "balanced_update": (
                "On the same k=4 section space and an 8192-point integration set, the "
                "double-precision nu-balanced T-map is the only tested update that "
                "suppresses the known ridge and improves both additional blind seeds."
            ),
        },
        "claim_limit": (
            "This is a targeted causal pilot on one known failure seed and two additional "
            "8192-point paired blind seeds. It is not a global sup-norm certificate and "
            "does not yet establish large-sample or k-scaling convergence."
        ),
        "provenance": {
            **provenance,
            "paired_blind_audit": {
                "path": str(paired_path),
                "sha256": sha256_file(paired_path),
            },
        },
    }

    json_path = out_dir / "report.json"
    markdown_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def number(value: float) -> str:
        return f"{value:.6g}"

    lines = [
        "# Type-(1,1) k=4 algorithm-control experiment",
        "",
        "All three rows use the same 274-section algebraic metric family and target-measure weights.",
        "",
        "## Known failure seed 72003 (8192 paired points)",
        "",
        "| Method | sigma | chi | q99.9(r) | max r | mass(r>3) | r>3 fibres |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("point64_l1", "fullpool_e2", "nu_balanced"):
        row = known_seed_rows[key]
        lines.append(
            f"| {labels[key]} | {number(row['sigma'])} | {number(row['chi_l2'])} | "
            f"{number(row['ratio_q999'])} | {number(row['max_ratio'])} | "
            f"{number(row['ratio_above_3_weighted_mass'])} | "
            f"{row['ratio_above_3_fibre_count']} |"
        )
    lines.extend(
        [
            "",
            "## Additional paired blind seeds 73011 and 73012",
            "",
            "| Method | mean sigma | mean chi | worst chi | worst max r | worst mass(r>3) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("point64_l1", "fullpool_e2", "nu_balanced"):
        row = paired_rows[key]
        lines.append(
            f"| {labels[key]} | {number(row['mean_sigma'])} | "
            f"{number(row['mean_chi_l2'])} | "
            f"{number(row['maximum_seed_chi_l2'])} | "
            f"{number(row['maximum_ratio'])} | "
            f"{number(row['maximum_ratio_above_3_weighted_mass'])} |"
        )
    current_mechanism = mechanism_rows["point64_l1"]
    worst = current_mechanism["worst_point"]
    bulk_gap = current_mechanism["bulk"]["top_two_logit_gap"]["median"]
    tail_gap = current_mechanism["top_1_percent"]["top_two_logit_gap"]["median"]
    balanced_condition = mechanism_rows["nu_balanced"][
        "reference_relative_h_spectrum"
    ]["condition_number"]
    current_condition = current_mechanism["reference_relative_h_spectrum"][
        "condition_number"
    ]
    lines.extend(
        [
            "",
            "## Mechanism verdict",
            "",
            f"- The point64 worst point has N_eff={number(worst['effective_eigensection_count'])} "
            f"and leading mass={number(worst['leading_eigensection_mass'])}; it is not a two-section-dominance event.",
            f"- The top-one-percent median top-two gap is {number(tail_gap)}, versus "
            f"{number(bulk_gap)} in the bottom 90 percent. A switching ridge would predict the opposite ordering.",
            f"- The point64 H condition number is {number(current_condition)}, while the "
            f"better nu-balanced H has condition number {number(balanced_condition)}. Spectrum width alone is not causal.",
            "- Exact full-pool E2 overfits its 8192-point integration set; changing only the residual loss is insufficient.",
            "- The nu-balanced T-map enforces a full global moment equation at every update and is the only tested method that removes the disclosed ridge and improves both added blind seeds.",
            "",
            "## Limits and next calculation",
            "",
            "This is not a sup-norm certificate. The next controlled run should scale the nu-balanced integration pool to 32768 and 131072 points, use complex128 for the Hilbert inversion, and evaluate at least eight untouched seeds. After that, use the balanced artifact as the initializer for a low-step MA-energy refinement and compare k=3,4,5.",
            "",
            f"Machine-readable report: `{json_path}`",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
