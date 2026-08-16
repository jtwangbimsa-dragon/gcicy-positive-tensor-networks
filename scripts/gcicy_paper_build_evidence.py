#!/usr/bin/env python3
"""Build machine-verifiable tables, plots, and manifests for the gCICY paper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, load_active_point_pool  # noqa: E402


PAPER = ROOT / "gcicy paper"
FIGURES = PAPER / "figures"
EVIDENCE = ROOT / "outputs" / "multitype_publication_evidence.json"
TYPE21_RITZ = (
    ROOT
    / "outputs"
    / "pipeline"
    / "type21_tensor_network_k4_teacher_free_ritz_20260718"
    / "dense_teacher_free_ritz_seed83401_n8190_l23.json"
)
TYPE21_RITZ_AUDIT = (
    ROOT / "outputs" / "gcicy_paper_type21_ritz_existing_artifact_audit.json"
)
METRIC_TAIL_SUMMARY = ROOT / "outputs" / "gcicy_paper_metric_tail_summary.json"
TYPE11_METRIC_SYSTEMATIC = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_scalar_metric_systematic_32768.json"
)
TYPE11_METRIC_SYSTEMATIC_SUMMARY = (
    ROOT / "outputs" / "gcicy_paper_type11_metric_systematic_summary.json"
)
FIXED_MODEL_SMOOTHNESS_SUMMARY = (
    ROOT / "outputs" / "gcicy_paper_fixed_model_smoothness.json"
)
TAIL_MECHANISM_SUMMARY = (
    ROOT / "outputs" / "gcicy_paper_tail_mechanism_summary.json"
)
ENVIRONMENT_LOCK = PAPER / "environment_lock.json"
ACCEPTED_TYPE11 = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_k4_tail_refined_gpu.npz"
)
REJECTED_TYPE11 = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_k4_v5_targeted_rank_two_b1_2p5_b2_0p5.npz"
)
REJECTED_DIAGNOSTIC = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_k4_l2_tail_v4_seed69604_failure_diagnostic.json"
)
TYPE21_SMOOTHNESS = tuple(
    ROOT
    / "outputs"
    / "pipeline"
    / f"p5p1_type21_candidate_1223_smoothness_p{prime}.json"
    for prime in (31991, 32003, 65521)
)
TYPE11_SMOOTHNESS = tuple(
    ROOT
    / "outputs"
    / "pipeline"
    / f"p4p1p1_type21_hirzebruch_x3_smoothness_p{prime}.json"
    for prime in (31991, 32003, 65521)
)
TYPE22_SMOOTHNESS = tuple(
    ROOT / "outputs" / f"gcicy_exact_smoothness_check_p{prime}.json"
    for prime in (31991, 32003, 65521)
)
ACTIVE_POOL = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_k4_adaptive_round1_active_pool.npz"
)
ACTIVE_DISCOVERY = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_k4_adaptive_round1_discovery_summary.json"
)
ACTIVE_SOURCE_METRIC = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_k4_full_h_1m_continuation_stabilized.npz"
)
RANK_TWO_SCAN = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_k4_v5_targeted_rank_two_joint_seed69604_extended_scan.json"
)
RANK_TWO_FRESH_FAILURE = (
    ROOT
    / "outputs"
    / "pipeline"
    / "p4p1_type11_hirzebruch_k4_v5_targeted_rank_two_b1_2p5_b2_0p5_seed69723_checkpoint_audit.json"
)

SMOOTHNESS_GROUPS = (
    {
        "geometry": "X11",
        "adapter": "p4p1_type11_hirzebruch_x3",
        "model_seed": 20260731,
        "presentation": "equivalent stabilized type-(2,1)",
        "paths": TYPE11_SMOOTHNESS,
    },
    {
        "geometry": "X21",
        "adapter": "p5p1_type21_k3_1223",
        "model_seed": 20260802,
        "presentation": "direct sequential type-(2,1)",
        "paths": TYPE21_SMOOTHNESS,
    },
    {
        "geometry": "X22",
        "adapter": "p1p1p5_type22",
        "model_seed": 20260711,
        "presentation": "direct sequential type-(2,2)",
        "paths": TYPE22_SMOOTHNESS,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_change(new: np.ndarray, old: np.ndarray) -> list[float]:
    return ((new - old) / old).tolist()


def model_payload_sha256(adapter_key: str, model_seed: int) -> str:
    adapter = get_adapter(adapter_key)
    model = adapter.make_model(model_seed, exact=True)
    payload = adapter.artifact_model_payload(model, exact=True)
    digest = hashlib.sha256()
    for key in sorted(payload):
        array = np.ascontiguousarray(np.asarray(payload[key]))
        digest.update(key.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_type21_observable_audit() -> dict:
    source = json.loads(TYPE21_RITZ.read_text())
    dataset = source["datasets"][0]
    teacher = dataset["metrics"]["full_h_teacher"]
    level2 = np.asarray(teacher["trial_levels"]["2"]["eigenvalues"], dtype=float)
    level3 = np.asarray(teacher["trial_levels"]["3"]["eigenvalues"], dtype=float)
    audit = {
        "schema": "gcicy-paper-existing-observable-audit-v1",
        "status": "preliminary_single_dataset_control",
        "source_path": str(TYPE21_RITZ.relative_to(ROOT)),
        "source_sha256": sha256(TYPE21_RITZ),
        "extracted_metric": "full_h_teacher",
        "teacher_artifact": source["teacher_artifact"],
        "teacher_artifact_sha256": source["teacher_artifact_sha256"],
        "adapter": source["adapter"],
        "sample_seed": dataset["dataset"]["seed"],
        "sample_points": dataset["dataset"]["points"],
        "complete_fibre_clusters": dataset["fibre_cluster_count"],
        "cluster_size": source["sampling_cluster_size"],
        "integration_measure": source["integration_measure"],
        "level2": teacher["trial_levels"]["2"],
        "level3": teacher["trial_levels"]["3"],
        "level2_to_level3_relative_changes": relative_change(level3, level2),
        "maximum_absolute_level_change": float(
            np.max(np.abs((level3 - level2) / level2))
        ),
        "limitations": [
            "Only one independent 8190-point dataset is present.",
            "No delete-group jackknife was run in this source artifact.",
            "The fibre ESS per retained trial rank is below 2 at level 3.",
            "The result is a preliminary portability control, not a continuum spectrum claim.",
        ],
    }
    TYPE21_RITZ_AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def build_metric_tail_summary(evidence: dict) -> dict:
    source_by_key = {row["key"]: row for row in evidence["source_files"]}
    audit_paths = {
        "(1,1)": ROOT / source_by_key["type11"]["path"],
        "(2,1)": ROOT / source_by_key["type21"]["path"],
        "(2,2)": ROOT / source_by_key["type22"]["path"],
    }
    rows = []
    sources = []
    for type_label, path in audit_paths.items():
        audit = json.loads(path.read_text())
        sources.append(
            {
                "type": type_label,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
        )
        for artifact in audit["artifacts"]:
            rows.append(
                {
                    "type": type_label,
                    "degree": artifact["degree"],
                    "section_count": artifact["section_count"],
                    "mean_sigma": artifact["mean_sigma"],
                    "sigma_95_percent_ci": artifact["sigma_95_percent_ci"],
                    "mean_chi": artifact["mean_sqrt_squared_energy"],
                    "maximum_seed_chi": artifact["max_seed_sqrt_squared_energy"],
                    "minimum_normalized_ratio": artifact["min_normalized_ratio"],
                    "maximum_normalized_ratio": artifact["max_normalized_ratio"],
                    "minimum_importance_effective_sample_size": artifact[
                        "min_importance_effective_sample_size"
                    ],
                }
            )
    summary = {
        "schema": "gcicy-paper-metric-tail-summary-v1",
        "definitions": {
            "chi": "importance-weighted RMS of 1-r on each fresh seed, then averaged across seeds",
            "maximum_normalized_ratio": "largest r observed across every fresh seed for the artifact",
            "limitation": (
                "The accepted aggregate audits retain seedwise RMS values and "
                "extrema but not point arrays, so common q99.9 and CVaR values "
                "cannot be reconstructed from these summaries."
            ),
        },
        "sources": sources,
        "rows": rows,
    }
    METRIC_TAIL_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_type11_metric_systematic_summary() -> dict:
    source = json.loads(TYPE11_METRIC_SYSTEMATIC.read_text())
    if not source["success"]:
        raise RuntimeError("The 32768-point metric-systematic audit did not pass")
    comparison = source["artifact_pairwise_comparisons"][0]
    artifacts = {row["key"]: row for row in source["artifacts"]}
    source_artifact = artifacts[comparison["source_artifact"]]
    target_artifact = artifacts[comparison["target_artifact"]]
    source_values = [
        row["mean"]
        for row in source_artifact["trial_levels"][0]["eigenvalues"][:3]
    ]
    target_values = [
        row["mean"]
        for row in target_artifact["trial_levels"][0]["eigenvalues"][:3]
    ]
    rows = []
    for row in comparison["eigenvalues"][:3]:
        rows.append(
            {
                "index": row["index"],
                "source_mean": row["source_mean"],
                "target_mean": row["target_mean"],
                "mean_relative_change": row["mean_relative_change"],
                "relative_change_95_percent_ci": row[
                    "relative_change_95_percent_ci"
                ],
            }
        )
    summary = {
        "schema": "gcicy-paper-type11-metric-systematic-v1",
        "source_path": str(TYPE11_METRIC_SYSTEMATIC.relative_to(ROOT)),
        "source_sha256": sha256(TYPE11_METRIC_SYSTEMATIC),
        "status": "accepted_same_points_same_basis_control",
        "sample_points_per_seed": source["points_per_seed"],
        "seeds": source["seeds"],
        "trial_level": comparison["trial_level"],
        "source_metric": comparison["source_artifact"],
        "source_degree": source_artifact["degree"],
        "target_metric": comparison["target_artifact"],
        "target_degree": target_artifact["degree"],
        "source_first_three_ritz_values": source_values,
        "target_first_three_ritz_values": target_values,
        "first_three_comparison": rows,
        "maximum_absolute_first_three_relative_change": float(
            max(abs(row["mean_relative_change"]) for row in rows)
        ),
        "interpretation": (
            "This is a metric-degree systematic at fixed points and fixed "
            "level-two trial basis; it is not a sampling standard error."
        ),
    }
    TYPE11_METRIC_SYSTEMATIC_SUMMARY.write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def build_fixed_model_smoothness_summary() -> dict:
    rows = []
    for group in SMOOTHNESS_GROUPS:
        artifacts = []
        for path in group["paths"]:
            audit = json.loads(path.read_text())
            if not audit["all_proved_smooth"]:
                raise RuntimeError(f"Smoothness artifact did not pass: {path}")
            artifacts.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "prime": audit["characteristic"],
                    "charts_requested": audit["charts_requested"],
                    "all_proved_smooth": audit["all_proved_smooth"],
                }
            )
        rows.append(
            {
                "geometry": group["geometry"],
                "adapter": group["adapter"],
                "model_seed": group["model_seed"],
                "fixed_coefficient_sha256": model_payload_sha256(
                    group["adapter"], group["model_seed"]
                ),
                "certificate_presentation": group["presentation"],
                "method": (
                    "complete affine-chart finite-field Jacobian-minor "
                    "good-reduction check"
                ),
                "primes": [row["prime"] for row in artifacts],
                "result": "smooth fixed characteristic-zero model",
                "artifacts": artifacts,
            }
        )
    summary = {
        "schema": "gcicy-paper-fixed-model-smoothness-v1",
        "logic": (
            "For each fixed integral model, emptiness of the singular locus "
            "on every standard projective chart at a good prime certifies a "
            "smooth characteristic-zero generic fibre; the extra primes are "
            "regression controls."
        ),
        "rows": rows,
    }
    FIXED_MODEL_SMOOTHNESS_SUMMARY.write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def build_tail_mechanism_summary() -> dict:
    discovery = json.loads(ACTIVE_DISCOVERY.read_text())
    adapter = get_adapter(discovery["adapter"])
    model = adapter.make_model(discovery["model_seed"], exact=True)
    pool = load_active_point_pool(
        ACTIVE_POOL,
        adapter,
        model,
        expected_model_seed=discovery["model_seed"],
        expected_exact_model=True,
    )
    artifact = adapter.load_h_artifact(ACTIVE_SOURCE_METRIC, model)
    active_metrics = adapter.h_metrics(pool.points, artifact)
    active_raw = adapter.residual_values(pool.points, active_metrics)
    positive_log_ratio = (
        active_raw - float(pool.metadata["global_log_normalization"])
    )
    radius_rows = []
    for radius in sorted(set(float(value) for value in pool.radii)):
        mask = np.isclose(pool.radii, radius)
        values = positive_log_ratio[mask]
        radius_rows.append(
            {
                "radius": radius,
                "point_count": int(np.sum(mask)),
                "minimum_ratio": float(np.exp(np.min(values))),
                "q10_ratio": float(np.exp(np.quantile(values, 0.10))),
                "median_ratio": float(np.exp(np.median(values))),
                "q90_ratio": float(np.exp(np.quantile(values, 0.90))),
                "maximum_ratio": float(np.exp(np.max(values))),
            }
        )

    failure = json.loads(REJECTED_DIAGNOSTIC.read_text())
    source_row = failure["artifacts"][0]
    source_point = source_row["top_points"][0]
    rank_two_scan = json.loads(RANK_TWO_SCAN.read_text())
    selected = next(
        row
        for row in rank_two_scan["candidates"]
        if row["first_strength"] == 2.5 and row["second_strength"] == 0.5
    )
    fresh_failure = json.loads(RANK_TWO_FRESH_FAILURE.read_text())
    fresh_row = fresh_failure["artifacts"][0]
    fresh_stats = fresh_row["volume_ratio_error_statistics"]
    summary = {
        "schema": "gcicy-paper-tail-mechanism-v1",
        "local_neighborhood_audit": {
            "source_path": str(ACTIVE_DISCOVERY.relative_to(ROOT)),
            "source_sha256": sha256(ACTIVE_DISCOVERY),
            "active_pool_path": str(ACTIVE_POOL.relative_to(ROOT)),
            "active_pool_sha256": sha256(ACTIVE_POOL),
            "source_metric_path": str(ACTIVE_SOURCE_METRIC.relative_to(ROOT)),
            "source_metric_sha256": sha256(ACTIVE_SOURCE_METRIC),
            "distinct_centers": discovery["audited_distinct_center_count"],
            "point_count": len(pool.points),
            "radius_rows": radius_rows,
            "reported_monte_carlo_estimate": False,
        },
        "extreme_point_anisotropy": {
            "source_path": str(REJECTED_DIAGNOSTIC.relative_to(ROOT)),
            "source_sha256": sha256(REJECTED_DIAGNOSTIC),
            "seed": failure["seed"],
            "maximum_normalized_ratio": source_row[
                "volume_ratio_error_statistics"
            ]["normalized_ratio_max"],
            "metric_eigenvalues": source_point["metric_eigenvalues"],
            "squared_error_fraction": source_point[
                "squared_error_fraction"
            ],
        },
        "targeted_repair_and_migration": {
            "scan_path": str(RANK_TWO_SCAN.relative_to(ROOT)),
            "scan_sha256": sha256(RANK_TWO_SCAN),
            "source_seed": 69604,
            "source_maximum_ratio": source_row[
                "volume_ratio_error_statistics"
            ]["normalized_ratio_max"],
            "same_seed_repaired_maximum_ratio": selected[
                "maximum_normalized_ratio"
            ],
            "fresh_failure_path": str(RANK_TWO_FRESH_FAILURE.relative_to(ROOT)),
            "fresh_failure_sha256": sha256(RANK_TWO_FRESH_FAILURE),
            "fresh_seed": fresh_failure["seed"],
            "fresh_maximum_ratio": fresh_stats["normalized_ratio_max"],
            "fresh_sigma": fresh_stats["sigma"],
            "fresh_chi": fresh_stats["sqrt_squared_energy"],
            "fresh_top_squared_error_fraction": fresh_row[
                "volume_ratio_tail"
            ]["maximum_ratio_point"]["squared_error_fraction"],
        },
    }
    TAIL_MECHANISM_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_environment_lock() -> dict:
    lock = {
        "schema": "gcicy-paper-environment-lock-v1",
        "scope": (
            "Manuscript assembly, deterministic evidence extraction, figures, "
            "and consistency checks on the local workstation."
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "remote_training_environment": (
            "Device and precision are retained in each training artifact; a "
            "full remote package lock is required in the public release."
        ),
    }
    try:
        import torch

        lock["torch"] = torch.__version__
        lock["cuda_available_locally"] = bool(torch.cuda.is_available())
    except ImportError:
        lock["torch"] = None
        lock["cuda_available_locally"] = False
    ENVIRONMENT_LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def build_workflow_figure() -> None:
    labels = [
        "configuration + local\ngeneralized sections",
        "base/fibre sampling +\ncomplete root clusters",
        "proposal / residue\ndensities and weights",
        "projective + implicit\natlas audit",
        "restricted section\nbasis",
        "positive full-$H$\nmetric",
        "independent bulk, tail\nand positivity audit",
        "cluster-aware\nRayleigh--Ritz estimate",
    ]
    positions = [
        (0.13, 0.76),
        (0.37, 0.76),
        (0.61, 0.76),
        (0.85, 0.76),
        (0.85, 0.24),
        (0.61, 0.24),
        (0.37, 0.24),
        (0.13, 0.24),
    ]
    colors = [
        "#DCEAF7",
        "#E7F2E5",
        "#FFF0D5",
        "#F1E5F4",
        "#DCEAF7",
        "#E7F2E5",
        "#FFF0D5",
        "#F1E5F4",
    ]
    fig, ax = plt.subplots(figsize=(8.0, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    width, height = 0.19, 0.25
    for (x, y), label, color in zip(positions, labels, colors, strict=True):
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=0.9,
            edgecolor="#444444",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.6)
    for left, right in zip(positions[:-1], positions[1:], strict=True):
        if abs(left[1] - right[1]) < 1e-8:
            start = (
                left[0] + (width / 2 if right[0] > left[0] else -width / 2),
                left[1],
            )
            end = (
                right[0] - (width / 2 if right[0] > left[0] else -width / 2),
                right[1],
            )
        else:
            start = (left[0], left[1] - height / 2)
            end = (right[0], right[1] + height / 2)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
        )
    fig.tight_layout(pad=0.3)
    fig.savefig(
        FIGURES / "workflow.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "workflow.png", dpi=220)
    plt.close(fig)


def build_figures(evidence: dict, tail_summary: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.figsize": (5.8, 3.7),
        }
    )

    by_type: dict[str, list[dict]] = {}
    for row in evidence["metric_rows"]:
        if row["type"] in {"$(1,1)$", "$(2,1)$", "$(2,2)$"}:
            by_type.setdefault(row["type"], []).append(row)
    labels = {"$(1,1)$": r"$(1,1)$", "$(2,1)$": r"$(2,1)$", "$(2,2)$": r"$(2,2)$"}
    colors = {"$(1,1)$": "#0072B2", "$(2,1)$": "#009E73", "$(2,2)$": "#D55E00"}
    markers = {"$(1,1)$": "o", "$(2,1)$": "s", "$(2,2)$": "^"}
    fig, ax = plt.subplots()
    for key in ["$(1,1)$", "$(2,1)$", "$(2,2)$"]:
        rows = sorted(by_type[key], key=lambda item: item["degree"][0])
        degree = np.asarray([row["degree"][0] for row in rows], dtype=int)
        sigma = np.asarray([row["mean_sigma"] for row in rows], dtype=float)
        lo = np.asarray([row["sigma_95_percent_ci"][0] for row in rows])
        hi = np.asarray([row["sigma_95_percent_ci"][1] for row in rows])
        ax.errorbar(
            degree,
            sigma,
            yerr=np.vstack([sigma - lo, hi - sigma]),
            color=colors[key],
            marker=markers[key],
            linewidth=1.6,
            capsize=3,
            label=labels[key],
        )
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("algebraic degree $k$")
    ax.set_ylabel(r"independent mean $\sigma$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "degree_convergence.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "degree_convergence.png", dpi=220)
    plt.close(fig)

    spectrum: dict[tuple[int, int], list[float]] = {}
    for row in evidence["spectrum_rows"]:
        spectrum[(row["points"], row["level"])] = [
            row["lambda_1"],
            row["lambda_2"],
            row["lambda_3"],
        ]
    fig, ax = plt.subplots()
    for level, color, marker in [(2, "#0072B2", "o"), (3, "#D55E00", "s")]:
        points = np.asarray([8192, 32768, 65536])
        values = np.asarray([spectrum[(point, level)] for point in points])
        for mode in range(3):
            ax.plot(
                points,
                values[:, mode],
                color=color,
                marker=marker,
                linewidth=1.3,
                alpha=0.95 - 0.16 * mode,
                label=f"level {level}, mode {mode + 1}",
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([8192, 32768, 65536], labels=["8192", "32768", "65536"])
    ax.set_xlabel("sample points per seed")
    ax.set_ylabel(r"Ritz value $\lambda$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "type11_ritz_convergence.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "type11_ritz_convergence.png", dpi=220)
    plt.close(fig)

    build_workflow_figure()

    local_rows = [
        row
        for row in tail_summary["local_neighborhood_audit"]["radius_rows"]
        if row["radius"] > 0
    ]
    radii = np.asarray([row["radius"] for row in local_rows])
    medians = np.asarray([row["median_ratio"] for row in local_rows])
    q10 = np.asarray([row["q10_ratio"] for row in local_rows])
    q90 = np.asarray([row["q90_ratio"] for row in local_rows])
    repair = tail_summary["targeted_repair_and_migration"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    left, right = axes
    left.fill_between(radii, q10, q90, color="#56B4E9", alpha=0.28)
    left.plot(radii, medians, color="#0072B2", marker="o", linewidth=1.6)
    left.axhline(3.0, color="#D55E00", linestyle="--", linewidth=1.0)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("local Fubini--Study radius")
    left.set_ylabel("normalized ratio $r$")
    left.set_title("finite-width local tail")
    left.grid(True, which="both", alpha=0.22)
    anisotropy = tail_summary["extreme_point_anisotropy"]["metric_eigenvalues"]
    left.text(
        0.04,
        0.04,
        "$\\lambda(g)=(%.2f,\\,%.2f,\\,%.2f)$\\nat an independent extreme point"
        % tuple(anisotropy),
        transform=left.transAxes,
        fontsize=7.5,
        va="bottom",
    )

    stages = ["source\n(seed 69604)", "targeted\nsame seed", "fresh\nseed 69723"]
    maxima = [
        repair["source_maximum_ratio"],
        repair["same_seed_repaired_maximum_ratio"],
        repair["fresh_maximum_ratio"],
    ]
    bars = right.bar(
        stages,
        maxima,
        color=["#CC79A7", "#009E73", "#D55E00"],
        edgecolor="#444444",
        linewidth=0.7,
    )
    right.set_yscale("log")
    right.set_ylabel("maximum observed $r$")
    right.set_title("targeted repair and migration")
    right.grid(True, axis="y", which="both", alpha=0.22)
    for bar, value in zip(bars, maxima, strict=True):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "tail_migration.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "tail_migration.png", dpi=220)
    plt.close(fig)


def build_manifest(
    evidence: dict,
    type21_audit: dict,
    metric_tail_summary: dict,
    metric_systematic: dict,
    smoothness_summary: dict,
    tail_summary: dict,
) -> None:
    source_files = []
    for record in evidence["source_files"]:
        path = ROOT / record["path"]
        actual = sha256(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"Hash mismatch for {path}")
        source_files.append({**record, "verified": True})

    manifest = {
        "schema": "gcicy-paper-evidence-manifest-v1",
        "publication_evidence": {
            "path": str(EVIDENCE.relative_to(ROOT)),
            "sha256": sha256(EVIDENCE),
            "all_gates_passed": evidence["all_gates_passed"],
        },
        "source_files": source_files,
        "type11_checkpoint_distinction": {
            "accepted_tail_refined": {
                "path": str(ACCEPTED_TYPE11.relative_to(ROOT)),
                "sha256": sha256(ACCEPTED_TYPE11),
                "status": "accepted_for_metric_and_scalar_ritz_results",
            },
            "rejected_rank_two_continuation": {
                "path": str(REJECTED_TYPE11.relative_to(ROOT)),
                "sha256": sha256(REJECTED_TYPE11),
                "status": "rejected_after_fresh_seed_tail_failure",
            },
            "failure_diagnostic": {
                "path": str(REJECTED_DIAGNOSTIC.relative_to(ROOT)),
                "sha256": sha256(REJECTED_DIAGNOSTIC),
            },
        },
        "derived_existing_observable_audit": {
            "path": str(TYPE21_RITZ_AUDIT.relative_to(ROOT)),
            "sha256": sha256(TYPE21_RITZ_AUDIT),
            "status": type21_audit["status"],
        },
        "derived_metric_tail_summary": {
            "path": str(METRIC_TAIL_SUMMARY.relative_to(ROOT)),
            "sha256": sha256(METRIC_TAIL_SUMMARY),
            "row_count": len(metric_tail_summary["rows"]),
        },
        "derived_type11_metric_systematic": {
            "path": str(TYPE11_METRIC_SYSTEMATIC_SUMMARY.relative_to(ROOT)),
            "sha256": sha256(TYPE11_METRIC_SYSTEMATIC_SUMMARY),
            "source_sha256": metric_systematic["source_sha256"],
            "status": metric_systematic["status"],
        },
        "fixed_model_smoothness": {
            "path": str(FIXED_MODEL_SMOOTHNESS_SUMMARY.relative_to(ROOT)),
            "sha256": sha256(FIXED_MODEL_SMOOTHNESS_SUMMARY),
            "geometry_count": len(smoothness_summary["rows"]),
        },
        "tail_mechanism": {
            "path": str(TAIL_MECHANISM_SUMMARY.relative_to(ROOT)),
            "sha256": sha256(TAIL_MECHANISM_SUMMARY),
            "fresh_failure_sha256": tail_summary[
                "targeted_repair_and_migration"
            ]["fresh_failure_sha256"],
        },
        "environment_lock": {
            "path": str(ENVIRONMENT_LOCK.relative_to(ROOT)),
            "sha256": sha256(ENVIRONMENT_LOCK),
        },
        "type21_good_reduction": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "all_proved_smooth": json.loads(path.read_text())[
                    "all_proved_smooth"
                ],
            }
            for path in TYPE21_SMOOTHNESS
        ],
        "figures": [
            {
                "path": str((FIGURES / "degree_convergence.pdf").relative_to(ROOT)),
                "sha256": sha256(FIGURES / "degree_convergence.pdf"),
            },
            {
                "path": str((FIGURES / "type11_ritz_convergence.pdf").relative_to(ROOT)),
                "sha256": sha256(FIGURES / "type11_ritz_convergence.pdf"),
            },
            {
                "path": str((FIGURES / "workflow.pdf").relative_to(ROOT)),
                "sha256": sha256(FIGURES / "workflow.pdf"),
            },
            {
                "path": str((FIGURES / "tail_migration.pdf").relative_to(ROOT)),
                "sha256": sha256(FIGURES / "tail_migration.pdf"),
            },
        ],
    }
    (PAPER / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(EVIDENCE.read_text())
    if not evidence["all_gates_passed"]:
        raise RuntimeError("Publication evidence gate is not satisfied")
    type21_audit = build_type21_observable_audit()
    metric_tail_summary = build_metric_tail_summary(evidence)
    metric_systematic = build_type11_metric_systematic_summary()
    smoothness_summary = build_fixed_model_smoothness_summary()
    tail_summary = build_tail_mechanism_summary()
    build_environment_lock()
    build_figures(evidence, tail_summary)
    build_manifest(
        evidence,
        type21_audit,
        metric_tail_summary,
        metric_systematic,
        smoothness_summary,
        tail_summary,
    )
    print(PAPER / "evidence_manifest.json")
    print(TYPE21_RITZ_AUDIT)
    print(METRIC_TAIL_SUMMARY)
    print(TYPE11_METRIC_SYSTEMATIC_SUMMARY)
    print(FIXED_MODEL_SMOOTHNESS_SUMMARY)
    print(TAIL_MECHANISM_SUMMARY)


if __name__ == "__main__":
    main()
