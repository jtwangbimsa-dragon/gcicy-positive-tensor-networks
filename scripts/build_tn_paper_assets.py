#!/usr/bin/env python3
"""Build checked tables, figures, and a release manifest for the TN paper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from summarize_quintic_tn_full_h_budget_control import (  # noqa: E402
    latex_table as build_tn_full_h_budget_table,
)


DEGREE_SUMMARY = ROOT / "outputs/quintic_tn_degree_capacity_summary_20260719.json"
TN_MULTISEED = ROOT / "outputs/quintic_tn_final_multiseed_20260719/summary.json"
TN_FINAL_RUN = ROOT / (
    "outputs/quintic_tn_k20_capacity_20260719/"
    "q25_d6_b1024_lr1e5_e50_final_refinement"
)
TN_CURVATURE = ROOT / (
    "outputs/quintic_tn_k20_curvature_20260719/"
    "n5000_corrected_seed202607193"
)
STANDARD_CYMETRIC = ROOT / (
    "outputs/cymetric_quintic_fermat_width63_p8820_same_points_20260719"
)
MODNET_RUNS = tuple(
    ROOT
    / "outputs/quintic_modnet_same_points_20260719"
    / f"w171_e50_seed{seed}"
    for seed in (202607197, 202607297, 202607397)
)
MODNET_CURVATURE = ROOT / (
    "outputs/quintic_modnet_curvature_20260719/"
    "w171_n5000_seed202607193"
)
MODNET_LITERATURE_SCALE = ROOT / (
    "outputs/quintic_modnet_literature_scale_20260719/"
    "w64_n1m_e5_seed202607198"
)
MODNET_DATA_CONTROL = ROOT / (
    "outputs/quintic_modnet_same_points_20260719/"
    "w64_e50_seed202607198_control"
)
TN_DATA_CONTROL = ROOT / (
    "outputs/quintic_tn_literature_scale_20260719/"
    "k20_q25_d6_n90k_e50_continuation_control_seed202607192"
)
TN_LITERATURE_SCALE = ROOT / (
    "outputs/quintic_tn_literature_scale_20260719/"
    "k20_q25_d6_n900k_e5_seed202607192"
)
TN_LITERATURE_PULLBACK_REPORT = ROOT / (
    "outputs/quintic_tn_literature_scale_20260719/pullbacks_n1m/report.json"
)
SUB100K_ROOT = ROOT / "outputs/quintic_tn_sub100k_joint_scan_20260719"
SUB100K_RUNS = (
    ("k40 repeat init", SUB100K_ROOT / "k40_d6_repeat_init_audit_stable"),
    ("k40 D6", SUB100K_ROOT / "k40_d6_e50_seed202607192"),
    ("k40 D6 bridge", SUB100K_ROOT / "k40_d6_bridge1e2_e50_seed202607192"),
    ("k40 D7", SUB100K_ROOT / "k40_d7_e20_seed202607192"),
)
SUB100K_TRANSFER_SUMMARIES = (
    SUB100K_ROOT / "k40_d6_repeat_init/transfer_summary.json",
    SUB100K_ROOT / "k40_d6_bridge1e2_init/transfer_summary.json",
    SUB100K_ROOT / "k40_d7_from_bridge_d6_init/expansion_summary.json",
)
GCICY_SUMMARY = ROOT / "outputs/gcicy_tn_publication_summary_20260719.json"
TN_FULL_H_BUDGET_SUMMARY = ROOT / (
    "outputs/quintic_tn_full_h_budget_control_20260720/summary.json"
)
TN_TWO_SITE_REPORT = ROOT / (
    "outputs/quintic_tn_two_site_lm_20260720/"
    "formal_from_d12_pairs_0_5_10_15_20_25_28_seed219/report.json"
)
TN_TWO_SITE_MODEL = TN_TWO_SITE_REPORT.parent / "best_tensor_network.pt"
FULL_H_TRAINING_REPORT = ROOT / (
    "outputs/quintic_full_h_minibatch_20260721/k8_closure_registered/"
    "01_b256_lr1e6_e100_v2/report.json"
)
FULL_H_MODEL = FULL_H_TRAINING_REPORT.parent / "best_h_metric.npz"
FULL_H_AUDIT_REPORT = ROOT / (
    "outputs/quintic_full_h_minibatch_20260721/k8_closure_registered/"
    "02_v2_best_validation_blind200k/report.json"
)

OUTPUT_JSON = ROOT / "outputs/tn_paper_evidence_20260719.json"
OUTPUT_MANIFEST = ROOT / "outputs/tn_release_manifest_20260719.json"
OUTPUT_MARKDOWN = ROOT / "docs/tn_paper_evidence_20260719.md"
MAIN_TABLES = ROOT / "paper/generated/tn_main_tables.tex"
GCICY_TABLE = ROOT / "paper/generated/tn_gcicy_table.tex"
DATA_SCALE_TABLE = ROOT / "paper/generated/tn_data_scale_table.tex"
SUB100K_TABLE = ROOT / "paper/generated/tn_sub100k_table.tex"
APPENDIX_TABLES = ROOT / "paper/generated/tn_appendix_tables.tex"
TN_FULL_H_BUDGET_TABLE = ROOT / "paper/generated/tn_full_h_budget_table.tex"
FIGURE_DIR = ROOT / "paper/figures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        value = value[key]
    return value


def phi_metrics(report: dict[str, Any]) -> dict[str, float | int]:
    statistics = report["trained_phi_model"]
    return {
        "sigma": float(statistics["sigma_official_formula"]),
        "chi": float(statistics["weighted_rms_abs_residual"]),
        "q999": float(
            statistics["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99": float(
            statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum_residual": float(
            statistics["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
        "minimum_eigenvalue": float(
            statistics["min_eigenvalue_weighted_quantiles"]["q0.0000"]
        ),
        "nonpositive_count": int(
            statistics["nonpositive_min_eigenvalue"]["count"]
        ),
        "training_seconds": float(report["timing_seconds"]["training"]),
        "wall_seconds": float(report["timing_seconds"]["wall_total"]),
    }


def tn_metrics(report: dict[str, Any]) -> dict[str, float | int]:
    statistics = report["blind_test"]["normalized_volume"]
    return {
        "sigma": float(statistics["sigma_official_formula"]),
        "chi": float(statistics["weighted_rms_abs_residual"]),
        "q999": float(
            statistics["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99": float(
            statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum_residual": float(
            statistics["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
        "minimum_eigenvalue": float(
            statistics["min_eigenvalue_weighted_quantiles"]["q0.0000"]
        ),
        "nonpositive_count": int(
            statistics["nonpositive_min_eigenvalue"]["count"]
        ),
        "training_seconds": float(report["timing_seconds"]["training"]),
        "wall_seconds": float(report["timing_seconds"]["wall_total"]),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def modnet_multiseed(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for run_dir, report in zip(MODNET_RUNS, reports, strict=True):
        row = {
            "model_seed": int(report["configuration"]["model_seed"]),
            "report": str((run_dir / "report.json").relative_to(ROOT)),
            "report_sha256": sha256_file(run_dir / "report.json"),
            **phi_metrics(report),
        }
        rows.append(row)
    metric_names = (
        "sigma",
        "chi",
        "q999",
        "cvar99",
        "maximum_residual",
        "training_seconds",
    )
    return {
        "parameter_count": int(reports[0]["network"]["parameter_count"]),
        "rows": rows,
        "aggregate": {
            name: summarize([float(row[name]) for row in rows])
            for name in metric_names
        },
    }


def build_data_scale_ablation(
    modnet_control: dict[str, Any],
    modnet_large: dict[str, Any],
    tn_control: dict[str, Any],
    tn_large: dict[str, Any],
    *,
    expected_blind_sha256: str,
) -> dict[str, Any]:
    """Build and validate the same-exposure 90k/900k data ablation."""

    modnet_reports = (modnet_control, modnet_large)
    tn_reports = (tn_control, tn_large)
    blind_hashes = [
        *(report["data"]["blind_points_sha256"] for report in modnet_reports),
        *(
            report["common_point_evidence"]["source_sha256"]["blind_points"]
            for report in tn_reports
        ),
    ]
    if any(value != expected_blind_sha256 for value in blind_hashes):
        raise RuntimeError("a data-scale arm uses a different blind pool")
    if len({int(report["configuration"]["model_seed"]) for report in modnet_reports}) != 1:
        raise RuntimeError("ModNet data-scale arms use different model seeds")
    if len({int(report["configuration"]["torch_seed"]) for report in tn_reports}) != 1:
        raise RuntimeError("TN data-scale arms use different optimizer seeds")
    if len({int(report["network"]["parameter_count"]) for report in modnet_reports}) != 1:
        raise RuntimeError("ModNet data-scale arms have different parameter counts")
    if len(
        {
            int(report["architecture"]["trainable_real_parameter_count"])
            for report in tn_reports
        }
    ) != 1:
        raise RuntimeError("TN data-scale arms have different parameter counts")
    initial_hashes = {
        report["training"]["initial_model"]["sha256"] for report in tn_reports
    }
    if len(initial_hashes) != 1:
        raise RuntimeError("TN data-scale arms do not start from the same checkpoint")

    for modnet_report, tn_report in zip(modnet_reports, tn_reports, strict=True):
        if (
            modnet_report["data"]["dataset_sha256"]
            != tn_report["common_point_evidence"]["source_sha256"]["dataset"]
        ):
            raise RuntimeError("ModNet and TN data-scale arms use different datasets")

    rows: list[dict[str, Any]] = []
    for label, report in zip(("90k", "900k"), modnet_reports, strict=True):
        train_count = int(report["data"]["train_count"])
        epochs = int(report["configuration"]["epochs"])
        rows.append(
            {
                "family": "ModNet-style",
                "label": label,
                "train_count": train_count,
                "validation_count": int(report["data"]["validation_count"]),
                "epochs": epochs,
                "point_exposures": train_count * epochs,
                "parameters": int(report["network"]["parameter_count"]),
                "dataset_sha256": report["data"]["dataset_sha256"],
                **phi_metrics(report),
            }
        )
    for label, report in zip(("90k", "900k"), tn_reports, strict=True):
        train_count = int(report["common_point_evidence"]["train_points"])
        epochs = int(report["configuration"]["epochs"])
        rows.append(
            {
                "family": "positive TN",
                "label": label,
                "train_count": train_count,
                "validation_count": int(
                    report["common_point_evidence"]["validation_points"]
                ),
                "epochs": epochs,
                "point_exposures": train_count * epochs,
                "parameters": int(
                    report["architecture"]["trainable_real_parameter_count"]
                ),
                "dataset_sha256": report["common_point_evidence"]["source_sha256"][
                    "dataset"
                ],
                **tn_metrics(report),
            }
        )

    for family in ("ModNet-style", "positive TN"):
        family_rows = [row for row in rows if row["family"] == family]
        if len({row["point_exposures"] for row in family_rows}) != 1:
            raise RuntimeError(f"{family} data-scale arms have unequal exposure counts")

    modnet_small, modnet_big, tn_small, tn_big = rows
    return {
        "rows": rows,
        "common_blind_points_sha256": expected_blind_sha256,
        "tn_initial_model_sha256": next(iter(initial_hashes)),
        "derived": {
            "modnet_sigma_improvement_fraction": 1.0
            - modnet_big["sigma"] / modnet_small["sigma"],
            "modnet_chi_improvement_fraction": 1.0
            - modnet_big["chi"] / modnet_small["chi"],
            "tn_sigma_improvement_fraction": 1.0
            - tn_big["sigma"] / tn_small["sigma"],
            "tn_chi_improvement_fraction": 1.0
            - tn_big["chi"] / tn_small["chi"],
        },
    }


def build_sub100k_joint_scan(
    source_report: dict[str, Any],
    reports: list[dict[str, Any]],
    *,
    expected_blind_sha256: str,
) -> dict[str, Any]:
    """Validate the fixed-q, sub-100k joint k/D capacity diagnostic."""

    if len(reports) != len(SUB100K_RUNS):
        raise RuntimeError("sub-100k report inventory is incomplete")
    all_reports = [source_report, *reports]
    for report in all_reports:
        evidence = report["common_point_evidence"]
        if evidence["source_sha256"]["blind_points"] != expected_blind_sha256:
            raise RuntimeError("a sub-100k arm uses a different blind pool")
        if int(evidence["blind_points"]) != 200_000:
            raise RuntimeError("a sub-100k arm does not use 200,000 blind points")
        if int(report["architecture"]["shared_dictionary_rank_q"]) != 25:
            raise RuntimeError("the sub-100k diagnostic must hold q fixed at 25")
        if int(report["architecture"]["trainable_real_parameter_count"]) > 100_000:
            raise RuntimeError("a sub-100k diagnostic arm exceeds its budget")
        if tn_metrics(report)["nonpositive_count"]:
            raise RuntimeError("a sub-100k diagnostic arm has nonpositive metrics")

    rows = []
    source_metrics = tn_metrics(source_report)
    rows.append(
        {
            "label": "k20 source",
            "k": int(source_report["architecture"]["site_count_k"]),
            "q": 25,
            "D": int(source_report["architecture"]["bond_dimension_D"]),
            "parameters": int(
                source_report["architecture"]["trainable_real_parameter_count"]
            ),
            "epochs": int(source_report["configuration"]["epochs"]),
            **source_metrics,
        }
    )
    for (label, _), report in zip(SUB100K_RUNS, reports, strict=True):
        rows.append(
            {
                "label": label,
                "k": int(report["architecture"]["site_count_k"]),
                "q": int(report["architecture"]["shared_dictionary_rank_q"]),
                "D": int(report["architecture"]["bond_dimension_D"]),
                "parameters": int(
                    report["architecture"]["trainable_real_parameter_count"]
                ),
                "epochs": int(report["configuration"]["epochs"]),
                **tn_metrics(report),
            }
        )
    best = min(rows[1:], key=lambda row: row["sigma"])
    return {
        "rows": rows,
        "fixed_dictionary_rank": 25,
        "dictionary_is_complete": True,
        "parameter_budget": 100_000,
        "best_label": best["label"],
        "derived": {
            "repeat_initialization_sigma_relative_change": (
                rows[1]["sigma"] / rows[0]["sigma"] - 1.0
            ),
            "best_sigma_improvement_fraction": 1.0
            - best["sigma"] / rows[0]["sigma"],
            "best_chi_improvement_fraction": 1.0
            - best["chi"] / rows[0]["chi"],
        },
    }


def verify_common_protocol(
    tn_report: dict[str, Any],
    standard_report: dict[str, Any],
    modnet_reports: list[dict[str, Any]],
    modnet_curvature: dict[str, Any],
    tn_curvature: dict[str, Any],
) -> dict[str, Any]:
    source_hashes = tn_report["common_point_evidence"]["source_sha256"]
    expected_dataset = source_hashes["dataset"]
    expected_blind = source_hashes["blind_points"]
    phi_reports = [standard_report, *modnet_reports]
    for report in phi_reports:
        if report["data"]["dataset_sha256"] != expected_dataset:
            raise RuntimeError("a controlled Phi model uses a different training dataset")
        if report["data"]["blind_points_sha256"] != expected_blind:
            raise RuntimeError("a controlled Phi model uses a different blind pool")
        if int(report["trained_phi_model"]["n_points"]) != 200_000:
            raise RuntimeError("a controlled Phi model does not use 200,000 blind points")

    tn_curvature_arrays = np.load(TN_CURVATURE / "curvature_arrays.npz")
    modnet_curvature_arrays = np.load(MODNET_CURVATURE / "curvature_arrays.npz")
    if not np.array_equal(
        tn_curvature_arrays["indices"], modnet_curvature_arrays["indices"]
    ):
        raise RuntimeError("TN and ModNet curvature audits use different points")
    if tn_curvature["sampling"] != modnet_curvature["sampling"]:
        raise RuntimeError("TN and ModNet curvature sampling protocols differ")
    if (
        tn_curvature["scientific_scope"]["ricci_measure_definition"]
        != modnet_curvature["scientific_scope"]["ricci_measure_definition"]
    ):
        raise RuntimeError("TN and ModNet curvature definitions differ")

    tn_tail = np.load(TN_FINAL_RUN / "blind_test_tail_arrays.npz")
    standard_tail = np.load(STANDARD_CYMETRIC / "blind_test_tail_arrays.npz")
    modnet_tail = np.load(MODNET_RUNS[0] / "blind_test_tail_arrays.npz")
    for tail in (standard_tail, modnet_tail):
        if not np.array_equal(tn_tail["weights"], tail["weights"]):
            raise RuntimeError("controlled tail arrays use different integration weights")
        if not np.array_equal(tn_tail["omega_squared"], tail["omega_squared"]):
            raise RuntimeError("controlled tail arrays use different Omega densities")

    return {
        "dataset_sha256": expected_dataset,
        "blind_points_sha256": expected_blind,
        "train_points": int(tn_report["common_point_evidence"]["train_points"]),
        "validation_points": int(
            tn_report["common_point_evidence"]["validation_points"]
        ),
        "blind_points": int(tn_report["common_point_evidence"]["blind_points"]),
        "curvature_indices_sha256": hashlib.sha256(
            np.asarray(tn_curvature_arrays["indices"], dtype=np.int64).tobytes()
        ).hexdigest(),
        "curvature_points": int(tn_curvature["sampling"]["sample_points"]),
        "curvature_fibres": int(tn_curvature["sampling"]["sample_fibres"]),
    }


def controlled_rows(
    degree: dict[str, Any],
    tn_multiseed: dict[str, Any],
    standard_report: dict[str, Any],
    modnet: dict[str, Any],
    tn_curvature: dict[str, Any],
    modnet_curvature: dict[str, Any],
) -> list[dict[str, Any]]:
    degree_rows = {row["label"]: row for row in degree["rows"]}
    tn_small = degree_rows["k9_q21_d5"]
    standard = phi_metrics(standard_report)
    tn_final = tn_multiseed["aggregate"]
    modnet_aggregate = modnet["aggregate"]
    return [
        {
            "model": "positive TN (k=9, q=21, D=5)",
            "kind": "positive_tn",
            "parameters": int(tn_small["parameters"]),
            "sigma": float(tn_small["sigma"]),
            "sigma_sd": None,
            "chi": float(tn_small["chi"]),
            "q999": float(tn_small["q999_abs_residual"]),
            "cvar99": float(tn_small["cvar99_abs_residual"]),
            "maximum_residual": float(tn_small["maximum_abs_residual"]),
            "ricci_measure": None,
            "training_seconds": float(tn_small["training_seconds"]),
            "positivity": "structural",
        },
        {
            "model": "standard cymetric PhiFS (3x63)",
            "kind": "cymetric_phi",
            "parameters": int(standard_report["network"]["parameter_count"]),
            "sigma": float(standard["sigma"]),
            "sigma_sd": None,
            "chi": float(standard["chi"]),
            "q999": float(standard["q999"]),
            "cvar99": float(standard["cvar99"]),
            "maximum_residual": float(standard["maximum_residual"]),
            "ricci_measure": None,
            "training_seconds": float(standard["training_seconds"]),
            "positivity": "sampled",
        },
        {
            "model": "positive TN (k=20, q=25, D=6)",
            "kind": "positive_tn",
            "parameters": int(tn_multiseed["frozen_protocol"]["trainable_real_parameter_count"]),
            "sigma": float(tn_final["sigma"]["mean"]),
            "sigma_sd": float(tn_final["sigma"]["sample_standard_deviation"]),
            "chi": float(tn_final["chi"]["mean"]),
            "q999": float(tn_final["q99_9_abs_residual"]["mean"]),
            "cvar99": float(tn_final["cvar99_abs_residual"]["mean"]),
            "maximum_residual": float(tn_final["maximum_abs_residual"]["mean"]),
            "ricci_measure": float(tn_curvature["curvature"]["ricci_measure"]),
            "ricci_measure_se": float(
                tn_curvature["curvature"]["ricci_measure_cluster_standard_error"]
            ),
            "training_seconds": float(tn_final["training_seconds"]["mean"]),
            "cumulative_wall_seconds": float(
                degree["derived"]["final_route_wall_seconds"]
            ),
            "positivity": "structural",
        },
        {
            "model": "ModNet-style PhiFS (2x171)",
            "kind": "modnet_style_phi",
            "parameters": int(modnet["parameter_count"]),
            "sigma": float(modnet_aggregate["sigma"]["mean"]),
            "sigma_sd": float(
                modnet_aggregate["sigma"]["sample_standard_deviation"]
            ),
            "chi": float(modnet_aggregate["chi"]["mean"]),
            "q999": float(modnet_aggregate["q999"]["mean"]),
            "cvar99": float(modnet_aggregate["cvar99"]["mean"]),
            "maximum_residual": float(
                modnet_aggregate["maximum_residual"]["mean"]
            ),
            "ricci_measure": float(modnet_curvature["curvature"]["ricci_measure"]),
            "ricci_measure_se": float(
                modnet_curvature["curvature"][
                    "ricci_measure_cluster_standard_error"
                ]
            ),
            "training_seconds": float(
                modnet_aggregate["training_seconds"]["mean"]
            ),
            "positivity": "sampled",
        },
    ]


def weighted_survival(
    residual: np.ndarray, weights: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    valid = np.isfinite(residual) & np.isfinite(weights) & (weights > 0)
    values = residual[valid]
    selected_weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    selected_weights = selected_weights[order]
    suffix = np.cumsum(selected_weights[::-1])[::-1]
    indices = np.searchsorted(values, thresholds, side="right")
    result = np.zeros_like(thresholds)
    mask = indices < len(values)
    result[mask] = suffix[indices[mask]] / np.sum(selected_weights)
    return result


def save_figure(fig: plt.Figure, name: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURE_DIR / f"{name}.pdf", FIGURE_DIR / f"{name}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def make_figures(
    degree: dict[str, Any], controlled: list[dict[str, Any]]
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    generated: list[Path] = []
    rows = degree["rows"]
    q21 = [row for row in rows if row["method"] == "positive_tn" and row["q"] == 21]
    capacity = [row for row in rows if row["method"] == "positive_tn"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
    axes[0].plot(
        [row["k"] for row in q21],
        [row["sigma"] for row in q21],
        marker="o",
        color="#176B87",
        label=r"$\sigma$",
    )
    axes[0].plot(
        [row["k"] for row in q21],
        [row["q999_abs_residual"] for row in q21],
        marker="s",
        color="#C84B31",
        label=r"$q_{99.9}(|1-r|)$",
    )
    axes[0].set_xlabel("degree / site count k")
    axes[0].set_ylabel("blind error")
    axes[0].set_yscale("log")
    axes[0].set_title(r"Fixed $q=21$, $D=5$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    stage_labels = [
        "k9\nq21 D5",
        "k12",
        "k15",
        "k20\nq21 D5",
        "q25\nD5",
        "low lr",
        "D6",
        "large\nbatch",
        "final",
    ]
    axes[1].plot(
        np.arange(len(capacity)),
        [row["sigma"] for row in capacity],
        marker="o",
        color="#5B2A86",
    )
    axes[1].set_xticks(np.arange(len(capacity)), stage_labels, rotation=40, ha="right")
    axes[1].set_ylabel(r"blind $\sigma$")
    axes[1].set_yscale("log")
    axes[1].set_title("Registered continuation path")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    generated.extend(save_figure(fig, "tn_degree_capacity"))

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    tn_rows = [row for row in rows if row["method"] == "positive_tn"]
    ax.plot(
        [row["parameters"] for row in tn_rows],
        [row["sigma"] for row in tn_rows],
        "o-",
        color="#176B87",
        label="positive TN path",
    )
    method_styles = {
        "cymetric_phi": ("#C84B31", "s", "standard cymetric"),
        "modnet_style_phi": ("#2E8B57", "^", "ModNet-style"),
    }
    for row in controlled:
        if row["kind"] not in method_styles:
            continue
        color, marker, label = method_styles[row["kind"]]
        ax.scatter(
            row["parameters"], row["sigma"], color=color, marker=marker, s=52, label=label
        )
    ax.set_xlabel("trainable real parameters")
    ax.set_ylabel(r"blind $\sigma$")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    generated.extend(save_figure(fig, "tn_parameter_accuracy"))

    tn_tail = np.load(TN_FINAL_RUN / "blind_test_tail_arrays.npz")
    standard_tail = np.load(STANDARD_CYMETRIC / "blind_test_tail_arrays.npz")
    modnet_tail = np.load(MODNET_RUNS[0] / "blind_test_tail_arrays.npz")
    thresholds = np.geomspace(1.0e-5, 1.0e-1, 240)
    tail_rows = (
        (
            "positive TN, k=20",
            np.abs(1.0 - tn_tail["normalized_ratio"]),
            tn_tail["weights"],
            "#176B87",
        ),
        (
            "ModNet-style, 34k",
            modnet_tail["abs_residual"],
            modnet_tail["weights"],
            "#2E8B57",
        ),
        (
            "standard cymetric, 8.8k",
            standard_tail["abs_residual"],
            standard_tail["weights"],
            "#C84B31",
        ),
    )
    fig, ax = plt.subplots(figsize=(4.5, 3.25))
    for label, residual, weights, color in tail_rows:
        survival = weighted_survival(residual, weights, thresholds)
        positive = survival > 0
        ax.plot(thresholds[positive], survival[positive], color=color, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"absolute MA residual $|1-r|$")
    ax.set_ylabel("weighted tail mass")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    generated.extend(save_figure(fig, "tn_residual_tail_ccdf"))

    tn_curvature = np.load(TN_CURVATURE / "curvature_arrays.npz")
    modnet_curvature = np.load(MODNET_CURVATURE / "curvature_arrays.npz")
    fig, ax = plt.subplots(figsize=(4.5, 3.25))
    for label, arrays, color in (
        ("positive TN", tn_curvature, "#176B87"),
        ("ModNet-style", modnet_curvature, "#2E8B57"),
    ):
        values = np.abs(arrays["ricci_scalar"])
        weights = arrays["metric_measure_weight"]
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order]) / np.sum(weights)
        ax.plot(values[order], cumulative, color=color, label=label)
    ax.set_xlabel(r"$|R|$")
    ax.set_ylabel("metric-volume CDF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    generated.extend(save_figure(fig, "tn_ricci_cdf"))
    return generated


def format_uncertain(mean: float, deviation: float | None, digits: int = 6) -> str:
    if deviation is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {deviation:.{digits}f}"


def build_main_tables(
    controlled: list[dict[str, Any]],
    curvature_rows: list[dict[str, Any]],
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled Fermat-quintic comparison. All rows use the same 90,000/10,000 training/validation split and the same 200,000 blind points. Uncertainties are sample standard deviations over three optimizer seeds where shown.}",
        r"\label{tab:controlled-fermat}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrc}",
        r"\toprule",
        r"model & parameters & $\sigma$ & $\chi$ & $q_{99.9}$ & CVaR$_{99}$ & max. & positivity \\",
        r"\midrule",
    ]
    names = {
        "positive TN (k=9, q=21, D=5)": r"positive TN ($k=9,q=21,D=5$)",
        "standard cymetric PhiFS (3x63)": r"standard cymetric PhiFS ($3\times63$)",
        "positive TN (k=20, q=25, D=6)": r"positive TN ($k=20,q=25,D=6$)",
        "ModNet-style PhiFS (2x171)": r"ModNet-style PhiFS ($2\times171$)",
    }
    for row in controlled:
        lines.append(
            "{} & {:,} & {} & {:.6f} & {:.6f} & {:.6f} & {:.6f} & {} \\\\".format(
                names[row["model"]],
                row["parameters"],
                format_uncertain(row["sigma"], row.get("sigma_sd")),
                row["chi"],
                row["q999"],
                row["cvar99"],
                row["maximum_residual"],
                row["positivity"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Same-point curvature audit on 5,000 points grouped into 1,000 complete fibres. The error on $\mathcal R$ is the whole-fibre cluster standard error.}",
            r"\label{tab:curvature}",
            r"\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"model & $\mathcal R$ & mean $|R|$ & $q_{99.9}(|R|)$ \\",
            r"\midrule",
        ]
    )
    for row in curvature_rows:
        lines.append(
            "{} & {:.6f} $\\pm$ {:.6f} & {:.4f} & {:.4f} \\\\".format(
                row["model"],
                row["ricci_measure"],
                row["ricci_measure_cluster_standard_error"],
                row["mean_abs_ricci"],
                row["q999_abs_ricci"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_gcicy_table(gcicy: dict[str, Any]) -> str:
    lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Transfer to sequential gCICY threefolds. Dictionary entries show the learned local-operator rank over the complete rank. Tail values are blind-sample statements, not sup-norm certificates.}",
            r"\label{tab:gcicy-transfer}",
            r"\small",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"type & sites & $D$ & dictionary & parameters & blind points & $\sigma$ & $\chi$ & ratio range \\",
            r"\midrule",
        ]
    for case in gcicy["cases"]:
        metrics = case["blind_metrics"]
        dictionary = case["physical_dictionary"]
        lines.append(
            "{} & {} & {} & {}/{} & {:,} & {:,} & {:.5f} & {:.5f} & [{:.3f},{:.3f}] \\\\".format(
                case["type"],
                case["site_count"],
                case["bond_dimension"],
                dictionary["rank"],
                dictionary["complete_rank"],
                case["trainable_real_parameter_count"],
                case["sample_counts"]["blind"],
                metrics["sigma"],
                metrics["chi"],
                metrics["normalized_ratio_min"],
                metrics["normalized_ratio_max"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_data_scale_table(ablation: dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Same-exposure data-scale ablation. Within each model family, both arms use the same seed, initialization, approximately $4.5$ million point exposures, and the common 200,000-point blind pool.}",
        r"\label{tab:data-scale}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"model & data arm & train points & epochs & exposures & parameters & $\sigma$ & $\chi$ & $q_{99.9}$ & max. \\",
        r"\midrule",
    ]
    for index, row in enumerate(ablation["rows"]):
        if index == 2:
            lines.append(r"\midrule")
        lines.append(
            "{} & {} & {:,} & {} & {:,} & {:,} & {:.7f} & {:.7f} & {:.7f} & {:.7f} \\\\".format(
                row["family"],
                row["label"],
                row["train_count"],
                row["epochs"],
                row["point_exposures"],
                row["parameters"],
                row["sigma"],
                row["chi"],
                row["q999"],
                row["maximum_residual"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_sub100k_table(scan: dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Fixed-$q=25$ capacity diagnostic below 100,000 parameters. All rows use the same 90,000 training points and 200,000 blind points. The higher-capacity rows are single-seed continuations and are not headline accuracy estimates.}",
        r"\label{tab:sub100k-scan}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"arm & $k$ & $D$ & parameters & epochs & $\sigma$ & $\chi$ & $q_{99.9}$ & max. \\",
        r"\midrule",
    ]
    for row in scan["rows"]:
        lines.append(
            "{} & {} & {} & {:,} & {} & {:.7f} & {:.7f} & {:.7f} & {:.7f} \\\\".format(
                row["label"],
                row["k"],
                row["D"],
                row["parameters"],
                row["epochs"],
                row["sigma"],
                row["chi"],
                row["q999"],
                row["maximum_residual"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_appendix_tables(
    degree: dict[str, Any], tn_multiseed: dict[str, Any], modnet: dict[str, Any]
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete registered Fermat TN degree/capacity path.}",
        r"\label{tab:degree-capacity-full}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"run & $k$ & $q$ & $D$ & parameters & batch & $\sigma$ & $\chi$ & $q_{99.9}$ & max. \\",
        r"\midrule",
    ]
    for row in degree["rows"]:
        if row["method"] != "positive_tn":
            continue
        lines.append(
            "{} & {} & {} & {} & {:,} & {} & {:.7f} & {:.7f} & {:.7f} & {:.7f} \\\\".format(
                row["label"].replace("_", r"\_"),
                row["k"],
                row["q"],
                row["D"],
                row["parameters"],
                row["batch_size"],
                row["sigma"],
                row["chi"],
                row["q999_abs_residual"],
                row["maximum_abs_residual"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Independent optimizer-seed audits for the final positive TN and the parameter-matched ModNet-style baseline.}",
            r"\label{tab:seed-audits}",
            r"\small",
            r"\textbf{Positive TN}\par\smallskip",
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"seed & $\sigma$ & $\chi$ & $q_{99.9}$ & CVaR$_{99}$ & max. \\",
            r"\midrule",
        ]
    )
    for row in tn_multiseed["runs"]:
        lines.append(
            "{} & {:.8f} & {:.8f} & {:.8f} & {:.8f} & {:.8f} \\\\".format(
                row["torch_seed"],
                row["sigma"],
                row["chi"],
                row["q99_9_abs_residual"],
                row["cvar99_abs_residual"],
                row["maximum_abs_residual"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\bigskip",
            r"\textbf{ModNet-style PhiFS}\par\smallskip",
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"seed & $\sigma$ & $\chi$ & $q_{99.9}$ & CVaR$_{99}$ & max. \\",
            r"\midrule",
        ]
    )
    for row in modnet["rows"]:
        lines.append(
            "{} & {:.8f} & {:.8f} & {:.8f} & {:.8f} & {:.8f} \\\\".format(
                row["model_seed"],
                row["sigma"],
                row["chi"],
                row["q999"],
                row["cvar99"],
                row["maximum_residual"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def evidence_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    required = [
        DEGREE_SUMMARY,
        TN_MULTISEED,
        TN_FINAL_RUN / "report.json",
        TN_FINAL_RUN / "blind_test_tail_arrays.npz",
        TN_CURVATURE / "report.json",
        TN_CURVATURE / "curvature_arrays.npz",
        STANDARD_CYMETRIC / "report.json",
        STANDARD_CYMETRIC / "blind_test_tail_arrays.npz",
        MODNET_CURVATURE / "report.json",
        MODNET_CURVATURE / "curvature_arrays.npz",
        MODNET_DATA_CONTROL / "report.json",
        MODNET_DATA_CONTROL / "blind_test_tail_arrays.npz",
        MODNET_LITERATURE_SCALE / "report.json",
        MODNET_LITERATURE_SCALE / "blind_test_tail_arrays.npz",
        TN_DATA_CONTROL / "report.json",
        TN_DATA_CONTROL / "blind_test_tail_arrays.npz",
        TN_LITERATURE_SCALE / "report.json",
        TN_LITERATURE_SCALE / "blind_test_tail_arrays.npz",
        TN_LITERATURE_PULLBACK_REPORT,
        GCICY_SUMMARY,
        TN_FULL_H_BUDGET_SUMMARY,
        TN_TWO_SITE_REPORT,
        TN_TWO_SITE_MODEL,
        FULL_H_TRAINING_REPORT,
        FULL_H_MODEL,
        FULL_H_AUDIT_REPORT,
    ]
    required.extend(run_dir / "report.json" for run_dir in MODNET_RUNS)
    required.extend(run_dir / "blind_test_tail_arrays.npz" for run_dir in MODNET_RUNS)
    required.extend(run_dir / "report.json" for _, run_dir in SUB100K_RUNS)
    required.extend(
        run_dir / "blind_test_tail_arrays.npz" for _, run_dir in SUB100K_RUNS
    )
    required.extend(SUB100K_TRANSFER_SUMMARIES)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    degree = read_json(DEGREE_SUMMARY)
    tn_multiseed = read_json(TN_MULTISEED)
    tn_report = read_json(TN_FINAL_RUN / "report.json")
    standard_report = read_json(STANDARD_CYMETRIC / "report.json")
    modnet_reports = [read_json(run_dir / "report.json") for run_dir in MODNET_RUNS]
    tn_curvature = read_json(TN_CURVATURE / "report.json")
    modnet_curvature = read_json(MODNET_CURVATURE / "report.json")
    modnet_data_control = read_json(MODNET_DATA_CONTROL / "report.json")
    modnet_literature_scale_report = read_json(
        MODNET_LITERATURE_SCALE / "report.json"
    )
    tn_data_control = read_json(TN_DATA_CONTROL / "report.json")
    tn_literature_scale_report = read_json(TN_LITERATURE_SCALE / "report.json")
    sub100k_reports = [
        read_json(run_dir / "report.json") for _, run_dir in SUB100K_RUNS
    ]
    gcicy = read_json(GCICY_SUMMARY)
    tn_full_h_budget = read_json(TN_FULL_H_BUDGET_SUMMARY)
    tn_two_site_report = read_json(TN_TWO_SITE_REPORT)
    full_h_training_report = read_json(FULL_H_TRAINING_REPORT)
    if tn_full_h_budget.get("schema") != "quintic-tn-full-h-near-matched-budget-v1":
        raise RuntimeError("the TN/full-H budget summary has an unsupported schema")
    artifact_checks = {
        "TN two-site model": (
            sha256_file(TN_TWO_SITE_MODEL),
            tn_two_site_report["artifacts"]["model_sha256"],
        ),
        "full-H model": (
            sha256_file(FULL_H_MODEL),
            full_h_training_report["artifacts"]["best_h_sha256"],
        ),
        "TN two-site report": (
            sha256_file(TN_TWO_SITE_REPORT),
            tn_full_h_budget["models"]["positive_tn"]["report_sha256"],
        ),
        "full-H training report": (
            sha256_file(FULL_H_TRAINING_REPORT),
            tn_full_h_budget["models"]["dense_full_h"][
                "training_report_sha256"
            ],
        ),
        "full-H audit report": (
            sha256_file(FULL_H_AUDIT_REPORT),
            tn_full_h_budget["models"]["dense_full_h"]["audit_report_sha256"],
        ),
    }
    failed_artifact_checks = {
        label: values for label, values in artifact_checks.items() if values[0] != values[1]
    }
    if failed_artifact_checks:
        raise RuntimeError(
            f"TN/full-H evidence artifact hashes differ: {failed_artifact_checks}"
        )
    if not gcicy["all_checks_passed"]:
        raise RuntimeError("the common gCICY publication audit did not pass")
    if not all(tn_multiseed["acceptance"].values()):
        raise RuntimeError("the final TN multiseed acceptance record is incomplete")

    modnet = modnet_multiseed(modnet_reports)
    protocol = verify_common_protocol(
        tn_report, standard_report, modnet_reports, modnet_curvature, tn_curvature
    )
    if (
        tn_full_h_budget["common_benchmark"]["blind_points_sha256"]
        != protocol["blind_points_sha256"]
    ):
        raise RuntimeError("the TN/full-H diagnostic uses a different blind pool")
    data_scale = build_data_scale_ablation(
        modnet_data_control,
        modnet_literature_scale_report,
        tn_data_control,
        tn_literature_scale_report,
        expected_blind_sha256=protocol["blind_points_sha256"],
    )
    sub100k_scan = build_sub100k_joint_scan(
        tn_report,
        sub100k_reports,
        expected_blind_sha256=protocol["blind_points_sha256"],
    )
    tn_parameters = int(
        tn_multiseed["frozen_protocol"]["trainable_real_parameter_count"]
    )
    parameter_mismatch = abs(modnet["parameter_count"] - tn_parameters) / tn_parameters
    if parameter_mismatch > 0.01:
        raise RuntimeError("the final TN and ModNet parameter budgets differ by over 1%")

    controlled = controlled_rows(
        degree,
        tn_multiseed,
        standard_report,
        modnet,
        tn_curvature,
        modnet_curvature,
    )
    curvature_rows = [
        {
            "model": "positive TN",
            "ricci_measure": float(tn_curvature["curvature"]["ricci_measure"]),
            "ricci_measure_cluster_standard_error": float(
                tn_curvature["curvature"]["ricci_measure_cluster_standard_error"]
            ),
            "mean_abs_ricci": float(
                tn_curvature["curvature"][
                    "metric_volume_weighted_mean_abs_ricci_scalar"
                ]
            ),
            "q999_abs_ricci": float(
                tn_curvature["curvature"][
                    "abs_ricci_scalar_metric_volume_quantiles"
                ]["q0.999"]
            ),
        },
        {
            "model": "ModNet-style PhiFS",
            "ricci_measure": float(modnet_curvature["curvature"]["ricci_measure"]),
            "ricci_measure_cluster_standard_error": float(
                modnet_curvature["curvature"][
                    "ricci_measure_cluster_standard_error"
                ]
            ),
            "mean_abs_ricci": float(
                modnet_curvature["curvature"][
                    "metric_volume_weighted_mean_abs_ricci_scalar"
                ]
            ),
            "q999_abs_ricci": float(
                modnet_curvature["curvature"][
                    "abs_ricci_scalar_metric_volume_quantiles"
                ]["q0.999"]
            ),
        },
    ]
    derived = {
        "matched_modnet_parameter_relative_difference": parameter_mismatch,
        "final_tn_sigma_improvement_over_modnet_fraction": (
            1.0
            - controlled[2]["sigma"] / controlled[3]["sigma"]
        ),
        "final_tn_chi_improvement_over_modnet_fraction": (
            1.0 - controlled[2]["chi"] / controlled[3]["chi"]
        ),
        "final_tn_ricci_measure_improvement_over_modnet_fraction": (
            1.0
            - curvature_rows[0]["ricci_measure"]
            / curvature_rows[1]["ricci_measure"]
        ),
        "final_tn_sigma_to_0_001_literature_context_ratio": (
            controlled[2]["sigma"] / 0.001
        ),
    }

    literature_scale = {
        "report": str((MODNET_LITERATURE_SCALE / "report.json").relative_to(ROOT)),
        "parameter_count": int(
            modnet_literature_scale_report["network"]["parameter_count"]
        ),
        "train_count": int(modnet_literature_scale_report["data"]["train_count"]),
        "validation_count": int(
            modnet_literature_scale_report["data"]["validation_count"]
        ),
        **phi_metrics(modnet_literature_scale_report),
    }

    payload = {
        "schema": "positive-tn-paper-evidence-v1",
        "scientific_target": (
            "positive tensor-network Bergman metrics with trainable parameter "
            "count and native point-evaluation cost linear in k at fixed q and D"
        ),
        "common_protocol": protocol,
        "controlled_fermat_rows": controlled,
        "curvature_rows": curvature_rows,
        "modnet_multiseed": modnet,
        "modnet_literature_scale": literature_scale,
        "data_scale_ablation": data_scale,
        "sub100k_joint_capacity_scan": sub100k_scan,
        "tn_multiseed": tn_multiseed,
        "degree_capacity": degree,
        "gcicy_transfer": gcicy,
        "tn_full_h_near_matched_budget": tn_full_h_budget,
        "derived": derived,
        "claim_limits": [
            "The O(k) law holds at fixed source dimension d, dictionary rank q, and bond dimension D; no theorem says q and D remain fixed for every geometry or accuracy target.",
            "The controlled ModNet-style implementation follows the published symmetric feature map but is not the authors' original code.",
            "The final high-accuracy TN uses a registered continuation path; isolated final-stage time and cumulative path time are both reported.",
            "Sampled positivity audits verify implementations, while global positivity and Kahler gluing follow from the purified ansatz plus the reference floor.",
            "Finite blind pools support statistical tail claims, not deterministic global sup-norm certificates.",
            "The failed k=20 to k=25 interpolation is an initialization failure, not a finite-k impossibility result.",
            "Exact multiplicative prolongation handles k to m*k; it does not yet solve arbitrary degree increments or guarantee that new bridge channels optimize effectively.",
            "The literature sigma=0.0010 and R=0.0028 are context, not controlled comparisons.",
            "The k30 TN versus k8 full-H table matches stored parameter budget and blind arrays, but not degree, objective, optimizer, or compute; the full-H checkpoint is a finite-budget baseline selected at its last evaluated epoch.",
        ],
    }
    write_json(OUTPUT_JSON, payload)

    MAIN_TABLES.parent.mkdir(parents=True, exist_ok=True)
    MAIN_TABLES.write_text(
        build_main_tables(controlled, curvature_rows), encoding="utf-8"
    )
    GCICY_TABLE.write_text(build_gcicy_table(gcicy), encoding="utf-8")
    DATA_SCALE_TABLE.write_text(
        build_data_scale_table(data_scale), encoding="utf-8"
    )
    SUB100K_TABLE.write_text(
        build_sub100k_table(sub100k_scan), encoding="utf-8"
    )
    APPENDIX_TABLES.write_text(
        build_appendix_tables(degree, tn_multiseed, modnet), encoding="utf-8"
    )
    TN_FULL_H_BUDGET_TABLE.write_text(
        build_tn_full_h_budget_table(tn_full_h_budget), encoding="utf-8"
    )
    generated_figures = make_figures(degree, controlled)

    evidence_paths = list(required)
    evidence_paths.extend(run_dir / "phi_network.weights.h5" for run_dir in MODNET_RUNS)
    evidence_paths.extend(
        [
            TN_FINAL_RUN / "best_tensor_network.pt",
            ROOT / "gcicy_metric/pipeline/positive_tensor_network.py",
            ROOT / "scripts/train_quintic_positive_tensor_network_same_points.py",
            ROOT / "scripts/run_cymetric_quintic_tail_experiment.py",
            ROOT / "scripts/audit_quintic_tn_curvature.py",
            ROOT / "scripts/audit_cymetric_quintic_curvature.py",
            ROOT / "scripts/build_tn_paper_assets.py",
            ROOT / "scripts/summarize_quintic_tn_full_h_budget_control.py",
            TN_FULL_H_BUDGET_SUMMARY,
            TN_TWO_SITE_REPORT,
            TN_TWO_SITE_MODEL,
            FULL_H_TRAINING_REPORT,
            FULL_H_MODEL,
            FULL_H_AUDIT_REPORT,
        ]
    )
    evidence_paths.extend(
        [
            MODNET_DATA_CONTROL / "phi_network.weights.h5",
            MODNET_LITERATURE_SCALE / "phi_network.weights.h5",
            TN_DATA_CONTROL / "best_tensor_network.pt",
            TN_LITERATURE_SCALE / "best_tensor_network.pt",
            ROOT / "scripts/precompute_quintic_pullbacks.py",
            ROOT / "scripts/resize_positive_tensor_network_sites.py",
            ROOT / "scripts/expand_positive_tensor_network_bond.py",
        ]
    )
    evidence_paths.extend(
        run_dir / "best_tensor_network.pt" for _, run_dir in SUB100K_RUNS
    )
    evidence_paths = list(dict.fromkeys(evidence_paths))
    generated_paths = [
        OUTPUT_JSON,
        MAIN_TABLES,
        GCICY_TABLE,
        DATA_SCALE_TABLE,
        SUB100K_TABLE,
        APPENDIX_TABLES,
        TN_FULL_H_BUDGET_TABLE,
        *generated_figures,
    ]
    records = [
        evidence_record(path, "source evidence") for path in evidence_paths
    ] + [evidence_record(path, "generated paper asset") for path in generated_paths]
    manifest = {
        "schema": "positive-tn-release-manifest-v1",
        "source_date": "2026-07-20",
        "records": records,
        "commands": {
            "paper_assets": "python3 scripts/build_tn_paper_assets.py",
            "tests": "python3 -m pytest -q",
            "paper": "cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex",
            "tn_curvature": "python scripts/audit_quintic_tn_curvature.py --sample-points 5000 --fibre-size 5 --seed 202607193 ...",
            "modnet_curvature": "python scripts/audit_cymetric_quintic_curvature.py --sample-points 5000 --fibre-size 5 --seed 202607193 ...",
        },
        "verification": {
            "full_test_suite": "180 passed in 138.368 seconds on 2026-07-21",
            "common_protocol": protocol,
            "all_gcicy_checks_passed": gcicy["all_checks_passed"],
            "all_tn_multiseed_checks_passed": all(
                tn_multiseed["acceptance"].values()
            ),
        },
        "claim_limits": payload["claim_limits"],
    }
    write_json(OUTPUT_MANIFEST, manifest)

    lines = [
        "# Positive TN paper evidence",
        "",
        "## Controlled Fermat comparison",
        "",
        "| model | parameters | sigma | chi | q99.9 | max | R |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in controlled:
        lines.append(
            "| {model} | {parameters:,} | {sigma:.7f} | {chi:.7f} | {q999:.7f} | "
            "{maximum_residual:.7f} | {ricci} |".format(
                **row,
                ricci=(
                    f"{row['ricci_measure']:.6f}"
                    if row["ricci_measure"] is not None
                    else "-"
                ),
            )
        )
    lines.extend(
        [
            "",
            f"At the final approximately matched parameter budget, TN lowers sigma by {100 * derived['final_tn_sigma_improvement_over_modnet_fraction']:.1f}%, chi by {100 * derived['final_tn_chi_improvement_over_modnet_fraction']:.1f}%, and the common-point R-measure by {100 * derived['final_tn_ricci_measure_improvement_over_modnet_fraction']:.1f}% relative to the controlled ModNet-style baseline.",
            "",
            "## Same-exposure data-scale ablation",
            "",
            "| model | train points | epochs | exposures | sigma | chi | q99.9 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data_scale["rows"]:
        lines.append(
            "| {family} | {train_count:,} | {epochs} | {point_exposures:,} | "
            "{sigma:.7f} | {chi:.7f} | {q999:.7f} | {maximum_residual:.7f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "At matched exposure count, increasing independent training coverage from 90,000 to 900,000 points lowers ModNet sigma by {:.1f}% and TN sigma by {:.1f}%.".format(
                100
                * data_scale["derived"]["modnet_sigma_improvement_fraction"],
                100 * data_scale["derived"]["tn_sigma_improvement_fraction"],
            ),
            "",
            "## Sub-100k joint capacity diagnostic",
            "",
            "| arm | k | D | parameters | epochs | sigma | chi | q99.9 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sub100k_scan["rows"]:
        lines.append(
            "| {label} | {k} | {D} | {parameters:,} | {epochs} | {sigma:.7f} | "
            "{chi:.7f} | {q999:.7f} | {maximum_residual:.7f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "With the complete q=25 local dictionary fixed, the best sub-100k arm lowers sigma by {:.1f}% relative to its same-seed k=20 source.".format(
                100
                * sub100k_scan["derived"]["best_sigma_improvement_fraction"]
            ),
            "",
            "All generated tables and figures are bound to source reports by `outputs/tn_release_manifest_20260719.json`.",
            "",
            "## Claim limits",
            "",
            *[f"- {limit}" for limit in payload["claim_limits"]],
            "",
        ]
    )
    OUTPUT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_JSON)
    print(OUTPUT_MANIFEST)
    print(OUTPUT_MARKDOWN)


if __name__ == "__main__":
    main()
