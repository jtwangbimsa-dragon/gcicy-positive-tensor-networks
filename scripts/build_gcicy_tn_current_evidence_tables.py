#!/usr/bin/env python3
"""Build current shared-data gCICY TN manuscript tables with source hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "gcicy paper" / "generated_tn"

SOURCES = {
    "x11_common_pool": ROOT
    / "outputs/pipeline/type11_h2_residual_phi_common_pool_20260728_seed86111/"
    "common_base_three_model_bilateral_summary.json",
    "x11_tn_vs_phi_bootstrap": ROOT
    / "outputs/pipeline/type11_h2_residual_phi_common_pool_20260728_seed86111/"
    "d5_tn_vs_residual_phi_confirmation_bootstrap.json",
    "x11_h2_vs_phi_bootstrap": ROOT
    / "outputs/pipeline/type11_h2_residual_phi_common_pool_20260728_seed86111/"
    "trained_h2_vs_residual_phi_confirmation_bootstrap.json",
    "x11_h2_vs_tn_bootstrap": ROOT
    / "outputs/pipeline/type11_h2_residual_phi_common_pool_20260728_seed86111/"
    "trained_h2_vs_d5_tn_confirmation_bootstrap.json",
    "x11_plateau_final_blind": ROOT
    / "outputs/pipeline/type11_h2_matched_tn_phi_three_seed_plateau_20260731/"
    "three_seed_plateau_final_blind_summary.json",
    "x21_h4_d8": ROOT
    / "outputs/pipeline/type21_common_confirmation_20260728/"
    "h4_vs_d8_tn_bilateral_summary.json",
    "x21_d8_multiseed": ROOT
    / "outputs/pipeline/type21_q121_d8_common_pool_three_seed_20260728/"
    "multiseed_confirmation_summary.json",
    "x21_d12": ROOT
    / "outputs/pipeline/type21_q121_d12_common_pool_pilot_20260729/"
    "bilateral_confirmation.json",
    "x21_d12_vs_d8": ROOT
    / "outputs/pipeline/type21_q121_d12_common_pool_pilot_20260729/"
    "paired_bootstrap_vs_d8_seed86231.json",
    "x21_d12_vs_h4": ROOT
    / "outputs/pipeline/type21_q121_d12_common_pool_pilot_20260729/"
    "paired_bootstrap_vs_h4.json",
    "x21_lift_rank": ROOT
    / "outputs/pipeline/type21_h4_lift_schmidt_audit_20260729/"
    "canonical_lift_first_cut_spectrum_seed86211.json",
    "legacy_publication": ROOT / "outputs/gcicy_tn_publication_summary_20260719.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tabular(column_spec: str, header: str, rows: list[str], *, resize: bool = True) -> str:
    body = "\n".join(
        [
            f"\\begin{{tabular}}{{{column_spec}}}",
            "\\toprule",
            header + " \\tabularnewline",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    return "\\resizebox{\\textwidth}{!}{%\n" + body + "\n}\n" if resize else body + "\n"


def write(name: str, content: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing numerical evidence files: {missing}")

    data = {key: load(path) for key, path in SOURCES.items()}
    x11 = data["x11_common_pool"]["models"]
    x21_h4_d8 = data["x21_h4_d8"]["models"]
    d8_summary = data["x21_d8_multiseed"]
    d12 = data["x21_d12"]["models"]["q121_D12_seed_86241"]["metrics"]
    legacy_x22 = next(
        case for case in data["legacy_publication"]["cases"] if case["type"] == "(2,2)"
    )

    headline_rows = [
        "(1,1) & 45 & 3 & fixed complete $q=2025$ & 141,750 & "
        "3-seed solver test & 200,000 & 0 \\tabularnewline",
        "(2,1) & 11 & 8 & fixed complete $q=121$ & 96,800 / 214,896 & "
        "compression + causal test & 196,608 & 0 \\tabularnewline",
        "(2,2) & 17 & 6 & learned $q=60/289$ & 47,880 & "
        "portability control & 65,532 & 0 \\tabularnewline",
    ]
    outputs = [
        write(
            "current_headline_20260729.tex",
            tabular(
                "lrrlrlrl",
                "type & $d$ & sites & dictionary & $P$ & numerical role & "
                "$N$ & nonpositive",
                headline_rows,
            ),
        )
    ]

    x11_rows = []
    for label, parameters, key in [
        (r"trained full-$H_2$", 2025, "trained_H2_full_H"),
        (r"$H_2+\phi$", 141642, "H2_plus_residual_phi"),
        (r"$k=6,D=5$ direct TN", 141750, "k6_D5_direct_TN"),
    ]:
        metrics = x11[key]["metrics"]
        x11_rows.append(
            "{} & {:,} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & "
            "[{:.3f},{:.3f}] & {:.4f} \\tabularnewline".format(
                label,
                parameters,
                metrics["sigma"],
                metrics["chi"],
                metrics["absolute_log_ratio_q999"],
                metrics["absolute_log_ratio_cvar_1pct"],
                metrics["normalized_ratio_min"],
                metrics["normalized_ratio_max"],
                metrics["minimum_metric_eigenvalue"],
            )
        )
    outputs.append(
        write(
            "x11_common_base_20260728.tex",
            tabular(
                "lrrrrrrr",
                "model & $P$ & $\\sigma$ & $\\chi$ & "
                "$Q_{0.999}(|\\log r|)$ & $\\CVaR_{1\\%}(|\\log r|)$ & "
                "$[r_{\\min},r_{\\max}]$ & $\\lambda_{\\min}(g)$",
                x11_rows,
            ),
        )
    )

    h4 = x21_h4_d8["h4_full_H"]["metrics"]
    d8_run = next(
        run
        for run in d8_summary["runs"]
        if run["activation_seed"]
        == d8_summary["representative_seeds"]["median_sigma_activation_seed"]
    )
    x21_rows = [
        "{} & {:,} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & "
        "[{:.3f},{:.3f}] & {} \\tabularnewline".format(
            r"$k=4$ full-$H$",
            104976,
            h4["sigma"],
            h4["chi"],
            h4["absolute_log_ratio_q999"],
            h4["absolute_log_ratio_cvar_1pct"],
            h4["normalized_ratio_min"],
            h4["normalized_ratio_max"],
            "separate training",
        ),
        "{} & {:,} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & "
        "[{:.3f},{:.3f}] & {} \\tabularnewline".format(
            r"$k=8,q=121,D=8$ TN",
            int(d8_summary["frozen_protocol"]["trainable_real_parameter_count"]),
            d8_run["sigma"],
            d8_run["chi"],
            d8_run["absolute_log_ratio_q999"],
            d8_run["absolute_log_ratio_cvar_1pct"],
            d8_run["normalized_ratio_min"],
            d8_run["normalized_ratio_max"],
            "median seed",
        ),
        "{} & {:,} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & "
        "[{:.3f},{:.3f}] & {} \\tabularnewline".format(
            r"$k=8,q=121,D=12$ TN",
            214896,
            d12["sigma"],
            d12["chi"],
            d12["absolute_log_ratio_q999"],
            d12["absolute_log_ratio_cvar_1pct"],
            d12["normalized_ratio_min"],
            d12["normalized_ratio_max"],
            "single-seed pilot",
        ),
    ]
    outputs.append(
        write(
            "x21_capacity_20260729.tex",
            tabular(
                "lrrrrrrl",
                "model & $P$ & $\\sigma$ & $\\chi$ & "
                "$Q_{0.999}(|\\log r|)$ & $\\CVaR_{1\\%}(|\\log r|)$ & "
                "$[r_{\\min},r_{\\max}]$ & status",
                x21_rows,
            ),
        )
    )

    parameter_rows = [
        "(1,1) & 6 & 871 & 141,750 & 758,641 & 5.4 \\tabularnewline",
        "(2,1), $D=8$ & 8 & 2,440 & 96,800 & 5,953,600 & 61.5 \\tabularnewline",
        "(2,1), $D=12$ & 8 & 2,440 & 214,896 & 5,953,600 & 27.7 \\tabularnewline",
        "(2,1), $D=10$ & 16 & 19,216 & 343,640 & 369,254,656 & 1,074.5 \\tabularnewline",
        "(2,1), $D=12$ & 16 & 19,216 & 493,680 & 369,254,656 & 748.0 \\tabularnewline",
        "(2,2) & 6 & 1,852 & 47,880 & 3,429,904 & 71.6 \\tabularnewline",
    ]
    outputs.append(
        write(
            "current_parameter_comparison_20260729.tex",
            tabular(
                "lrrrrr",
                "type & target degree & $h^0(L^k)$ & $P_{\\TN}$ & "
                "$P_{\\mathrm{full}\\text{-}H}$ & full-$H$/TN",
                parameter_rows,
                resize=False,
            ),
        )
    )

    protocol_rows = [
        "(1,1) & 3 independent runs & 196,608 & 24,576 & 200,000 & final evaluation & "
        "50,000 & 4 \\tabularnewline",
        "(2,1) & selected model & 196,608 & 24,576 & 196,608 & earlier evaluation & "
        "32,768 & 6 \\tabularnewline",
        "(2,1) & 3 independent runs, same updates & 196,608 & 24,576 & 196,608 & bond-growth comparison & "
        "32,768 & 6 \\tabularnewline",
        "(2,1) & exploratory $k$--$D$ study & 196,608 & 24,576 & 49,152 & optimization study & "
        "8,192 & 6 \\tabularnewline",
        "(2,2) & earlier calculation & 131,064 & 32,766 & 65,532 & portability test & "
        "21,844 & 3 \\tabularnewline",
    ]
    outputs.append(
        write(
            "current_protocol_ledger_20260729.tex",
            tabular(
                "llrrrrrr",
                "type & calculation & $N_{\\rm train}$ & $N_{\\rm val}$ & $N_{\\rm eval}$ & use & "
                "fibres & roots/fibre",
                protocol_rows,
            ),
        )
    )

    manifest = {
        "schema": "gcicy-tn-current-evidence-tables-v1",
        "sources": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for key, path in SOURCES.items()
        },
        "outputs": {
            path.name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for path in outputs
        },
        "notes": [
            "The principal X11 row is the common-start three-seed comparison on an independent final sample.",
            "The X11 residual-phi network and geometric outputs use float64; the TN parameters use complex128.",
            "The earlier X21 evaluation, independent equal-update sample, and inspected development landscape have distinct evidential roles.",
            "The expanded X21 k-D landscape is single-seed development evidence, not independent evidence for a scaling law.",
            "The X22 row retains the earlier independent evaluation.",
        ],
    }
    manifest_path = write(
        "current_evidence_manifest_20260729.json",
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
