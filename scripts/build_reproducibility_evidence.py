#!/usr/bin/env python3
"""Build training, certificate, and numerical degree-completeness tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def workspace_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


PIPELINE_TRAINING = (
    (
        "$X_{11}$",
        ROOT / "outputs/pipeline/p4p1_type11_hirzebruch_k2_gpu_summary.json",
    ),
    (
        "$X_{11}$",
        ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_tail_refined_gpu_summary.json",
    ),
    (
        "$X_{11}$ matched control",
        ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k2_tail_matched_gpu_summary.json",
    ),
    (
        "$X_{21}$",
        ROOT / "outputs/pipeline/p5p1_type21_k3_1223_k1_refined_summary.json",
    ),
    (
        "$X_{21}$",
        ROOT / "outputs/pipeline/p5p1_type21_k3_1223_k2_gpu_summary.json",
    ),
    (
        "$X_{21}$",
        ROOT / "outputs/pipeline/p5p1_type21_k3_1223_k3_gpu_large_summary.json",
    ),
    (
        "$X_{21}$",
        ROOT / "outputs/pipeline/p5p1_type21_k3_1223_k4_gpu_large_summary.json",
    ),
    (
        "$X_{11}^{(4)}$",
        ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_m4_k1_out_of_sample_summary.json",
    ),
)


SMOOTHNESS_GROUPS = (
    (
        "$X_{11}$",
        "(1,1)",
        20260731,
        (
            "outputs/pipeline/p4p1p1_type21_hirzebruch_x3_smoothness_p31991.json",
            "outputs/pipeline/p4p1p1_type21_hirzebruch_x3_smoothness_p32003.json",
            "outputs/pipeline/p4p1p1_type21_hirzebruch_x3_smoothness_p65521.json",
        ),
    ),
    (
        "$X_{21}$",
        "(2,1)",
        20260802,
        (
            "outputs/pipeline/p5p1_type21_candidate_1223_smoothness_p31991.json",
            "outputs/pipeline/p5p1_type21_candidate_1223_smoothness_p32003.json",
            "outputs/pipeline/p5p1_type21_candidate_1223_smoothness_p65521.json",
        ),
    ),
    (
        "$X_{22}$",
        "(2,2)",
        20260711,
        (
            "outputs/gcicy_exact_smoothness_check_p31991.json",
            "outputs/gcicy_exact_smoothness_check_p32003.json",
            "outputs/gcicy_exact_smoothness_check_p65521.json",
        ),
    ),
    (
        "$X_{11}^{(4)}$",
        "(1,1), degree $-2$",
        20260831,
        (
            "outputs/pipeline/p4p1_type11_hirzebruch_m4_smoothness_p31991.json",
            "outputs/pipeline/p4p1_type11_hirzebruch_m4_smoothness_p32003.json",
            "outputs/pipeline/p4p1_type11_hirzebruch_m4_smoothness_p65521.json",
        ),
    ),
)


METRIC_AUDITS = (
    (
        "$X_{11}$",
        ROOT / "outputs/pipeline/p4p1_type11_hirzebruch_k24_tail_refined_audit.json",
    ),
    (
        "$X_{11}$ matched control",
        ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k2_tail_matched_audit.json",
    ),
    (
        "$X_{21}$",
        ROOT / "outputs/pipeline/p5p1_type21_k3_1223_k1234_large_audit.json",
    ),
    (
        "$X_{22}$",
        ROOT / "outputs/pipeline/p1p1p5_type22_k123_audit.json",
    ),
    (
        "$X_{11}^{(4)}$",
        ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_m4_k1_out_of_sample_audit.json",
    ),
)


IMPLEMENTATION_SOURCES = (
    ROOT / "gcicy_metric/pipeline/train.py",
    ROOT / "gcicy_metric/pipeline/audit.py",
    ROOT / "gcicy_metric/pipeline/spectrum.py",
    ROOT / "gcicy_metric/type21_hirzebruch_x3.py",
    ROOT / "gcicy_metric/type21_candidate_p5p1_1223.py",
    ROOT / "gcicy_metric/generic_model.py",
    ROOT / "gcicy_metric/bicubic_model.py",
    ROOT / "scripts/audit_sampler_fibres.py",
    ROOT / "scripts/audit_large_metric_sample.py",
    ROOT / "scripts/audit_bicubic_large_sample.py",
    ROOT / "scripts/audit_scalar_laplacian.py",
    ROOT / "scripts/run_scalar_laplacian_seed_parallel.py",
    ROOT / "scripts/build_publication_evidence.py",
    ROOT / "scripts/build_multitype_publication_evidence.py",
    ROOT / "scripts/build_reproducibility_evidence.py",
    ROOT / "scripts/build_artifact_release.py",
    ROOT / "scripts/verify_artifact_release.py",
    ROOT / "scripts/run_sampler_audit_seed_parallel.py",
    ROOT / "scripts/certify_section_restriction_ranks.py",
    ROOT / "scripts/certify_topological_targets.py",
    ROOT / "artifact_release/README.md",
    ROOT / "artifact_release/ENVIRONMENT.md",
    ROOT / "artifact_release/requirements.txt",
    ROOT / "tests/test_gcicy_pipeline.py",
    ROOT / "tests/test_type21_p5p1_candidate.py",
    ROOT / "outputs/gpu_runs/20260710T065519Z-hermitian/environment.txt",
    ROOT / "pipeline_specs/p4p1_type11_hirzebruch_k2_gpu.json",
    ROOT / "pipeline_specs/p4p1_type11_hirzebruch_k4_tail_refine_gpu.json",
    ROOT / "pipeline_specs/p5p1_type21_k3_1223_k1_refined.json",
    ROOT / "pipeline_specs/p5p1_type21_k3_1223_k2_gpu.json",
    ROOT / "pipeline_specs/p5p1_type21_k3_1223_k3_gpu_large.json",
    ROOT / "pipeline_specs/p5p1_type21_k3_1223_k4_gpu_large.json",
    ROOT / "pipeline_specs/p4p1_type11_hirzebruch_m4_k1_out_of_sample.json",
    ROOT / "pipeline_specs/p4p1_type11_hirzebruch_k24_tail_refined_audit.json",
    ROOT / "pipeline_specs/p5p1_type21_k3_1223_k1234_large_audit.json",
    ROOT / "pipeline_specs/p1p1p5_type22_k123_audit.json",
    ROOT / "pipeline_specs/p4p1_type11_hirzebruch_m4_k1_out_of_sample_audit.json",
    ROOT
    / "pipeline_specs/p4p1_type11_hirzebruch_scalar_metric_volume_cluster_65536.json",
    ROOT
    / "pipeline_specs/p4p1_type11_hirzebruch_scalar_metric_systematic_8192.json",
    ROOT
    / "pipeline_specs/p4p1_type11_hirzebruch_scalar_metric_systematic_32768.json",
    ROOT
    / "pipeline_specs/p4p1_type11_hirzebruch_scalar_rank_robustness_65536.json",
    ROOT / "pipeline_specs/p4p1_type11_hirzebruch_k2_tail_matched_gpu.json",
    ROOT / "pipeline_specs/p4p1_type11_hirzebruch_k2_tail_matched_audit.json",
    ROOT
    / "pipeline_specs/p4p1_type11_hirzebruch_scalar_metric_systematic_tail_matched_32768.json",
    ROOT
    / "pipeline_specs/p4p1_type11_hirzebruch_scalar_basis_seed_161803_65536.json",
    ROOT
    / "pipeline_specs/p4p1_type11_hirzebruch_scalar_basis_seed_271828_65536.json",
    ROOT
    / "pipeline_specs/p4p1_type11_hirzebruch_scalar_basis_seed_577215_65536.json",
    ROOT / "pipeline_specs/reviewer_followups_large_metric_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type22-training",
        type=Path,
        default=ROOT / "pipeline_specs/p1p1p5_type22_training_manifest.json",
    )
    parser.add_argument(
        "--type11-spectrum",
        type=Path,
        default=ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_scalar_metric_volume_cluster_65536.json",
    )
    parser.add_argument(
        "--type11-spectrum-8192",
        type=Path,
        default=ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_scalar_metric_volume_cluster_8192.json",
    )
    parser.add_argument(
        "--type11-spectrum-32768",
        type=Path,
        default=ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_scalar_metric_volume_cluster_32768.json",
    )
    parser.add_argument(
        "--type21-sampler",
        type=Path,
        default=ROOT / "outputs/pipeline/p5p1_type21_sampler_audit_formal.json",
    )
    parser.add_argument(
        "--type22-sampler",
        type=Path,
        default=ROOT / "outputs/pipeline/p1p1p5_type22_sampler_audit_formal.json",
    )
    parser.add_argument(
        "--m4-sampler",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_m4_sampler_audit_formal.json",
    )
    parser.add_argument(
        "--type22-large-metric",
        type=Path,
        default=ROOT / "outputs/pipeline/p1p1p5_type22_k3_large_sample_audit.json",
    )
    parser.add_argument(
        "--bicubic-large-metric",
        type=Path,
        default=ROOT / "outputs/bicubic_k3_sampler_matched_large_audit.json",
    )
    parser.add_argument(
        "--large-metric-manifest",
        type=Path,
        default=ROOT / "pipeline_specs/reviewer_followups_large_metric_manifest.json",
    )
    parser.add_argument(
        "--metric-systematic-8192",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_scalar_metric_systematic_8192.json",
    )
    parser.add_argument(
        "--metric-systematic-32768",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_scalar_metric_systematic_32768.json",
    )
    parser.add_argument(
        "--metric-systematic-tail-matched-32768",
        type=Path,
        default=ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_scalar_metric_systematic_tail_matched_32768.json",
    )
    parser.add_argument(
        "--rank-robustness",
        type=Path,
        default=ROOT
        / "outputs/pipeline/"
        "p4p1_type11_hirzebruch_scalar_rank_robustness_65536.json",
    )
    parser.add_argument(
        "--section-rank-certificates",
        type=Path,
        default=ROOT / "outputs/section_restriction_rank_certificates.json",
    )
    parser.add_argument(
        "--topological-targets",
        type=Path,
        default=ROOT / "outputs/topological_target_certificates.json",
    )
    parser.add_argument(
        "--multitype-evidence",
        type=Path,
        default=ROOT / "outputs/multitype_publication_evidence.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "outputs/reproducibility_evidence.json",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=ROOT / "paper/generated/reproducibility_tables.tex",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing reproducibility input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_training_rows() -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for geometry, path in PIPELINE_TRAINING:
        data = load(path)
        request = data["request"]
        section_count = int(data["restricted_section_count"])
        artifact_value = (
            data["artifact"]["path"]
            if isinstance(data["artifact"], dict)
            else str(data["artifact"])
        )
        local_artifact = ROOT / "outputs/pipeline" / Path(artifact_value).name
        if not local_artifact.exists():
            raise SystemExit(f"missing local accepted artifact: {local_artifact}")
        rows.append(
            {
                "geometry": geometry,
                "degree": int(request["degree"][0]),
                "section_count": section_count,
                "parameterization": "full Cholesky",
                "train_points": int(request["train_points"]),
                "validation_points": int(request["validation_points"]),
                "check_points_per_seed": int(request["check_points"]),
                "check_seed_count": len(request["check_seeds"]),
                "export_points": int(request["export_points"]),
                "maximum_epochs": int(request["epochs"]),
                "selected_epoch": int(data["best_epoch"]),
                "learning_rate": float(request["learning_rate"]),
                "sigma_loss_weight": float(request.get("sigma_loss_weight", 0.0)),
                "selection_sigma_weight": float(
                    request.get("selection_sigma_weight", 0.0)
                ),
                "basis_points": int(request["basis_points"]),
                "seeds": {
                    "torch": int(request["torch_seed"]),
                    "basis": int(request["basis_seed"]),
                    "train": int(request["train_seed"]),
                    "validation": int(request["validation_seed"]),
                    "checks": [int(value) for value in request["check_seeds"]],
                    "export": int(request["export_seed"]),
                },
                "optimizer": "Adam",
                "learning_rate_schedule": "constant",
                "evaluation_interval": int(request["eval_every"]),
                "patience_evaluations": int(request["patience_evaluations"]),
                "validation_weight": float(request["validation_weight"]),
                "regularization_weight": float(request["drift_weight"]),
                "summary": workspace_path(path),
                "artifact": workspace_path(local_artifact),
                "real_parameter_count": section_count**2,
            }
        )
        sources.extend((path, local_artifact))
    return rows, sources


def type22_training_rows(path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    manifest = load(path)
    common = manifest["common"]
    rows: list[dict[str, Any]] = []
    sources = [path]
    for run in manifest["runs"]:
        summary_path = (path.parent / run["summary"]).resolve()
        artifact_path = (path.parent / run["artifact"]).resolve()
        summary = load(summary_path)
        if int(summary["restricted_section_count"]) != int(run["section_count"]):
            raise SystemExit("type-(2,2) training manifest section count mismatch")
        if int(summary["degree"][0]) != int(run["degree"]):
            raise SystemExit("type-(2,2) training manifest degree mismatch")
        rows.append(
            {
                "geometry": "$X_{22}$",
                "degree": int(run["degree"]),
                "section_count": int(run["section_count"]),
                "parameterization": run["parameterization"],
                "train_points": int(run["train_points"]),
                "validation_points": int(run["validation_points"]),
                "check_points_per_seed": int(run["check_points_per_seed"]),
                "check_seed_count": len(run["check_seeds"]),
                "export_points": int(run["export_points"]),
                "maximum_epochs": int(run["maximum_epochs"]),
                "selected_epoch": int(run["selected_epoch"]),
                "learning_rate": float(run["learning_rate"]),
                "sigma_loss_weight": 0.0,
                "selection_sigma_weight": 0.0,
                "basis_points": int(run["basis_points"]),
                "seeds": {
                    "torch": run["torch_seed"],
                    "basis": int(run["basis_seed"]),
                    "train": int(run["train_seed"]),
                    "validation": int(run["validation_seed"]),
                    "checks": [int(value) for value in run["check_seeds"]],
                    "export": int(run["export_seed"]),
                },
                "optimizer": common["optimizer"],
                "learning_rate_schedule": common["learning_rate_schedule"],
                "evaluation_interval": int(run["evaluation_interval"]),
                "patience_evaluations": int(run["patience_evaluations"]),
                "validation_weight": float(common["validation_weight"]),
                "regularization_weight": float(run["regularization_weight"]),
                "summary": workspace_path(summary_path),
                "artifact": workspace_path(artifact_path),
                "rank": run.get("rank"),
                "epsilon": run.get("epsilon"),
            }
        )
        sources.extend((summary_path, artifact_path))
    return rows, sources


def metric_test_rows() -> tuple[list[dict[str, Any]], list[Path]]:
    rows = []
    sources = []
    for geometry, path in METRIC_AUDITS:
        data = load(path)
        if not data["success"] or not all(data["gates"].values()):
            raise SystemExit(f"failed metric audit: {path}")
        primary = data["artifacts"][-1]
        thresholds = data["request"]["thresholds"]
        rows.append(
            {
                "geometry": geometry,
                "degree": int(primary["degree"][0]),
                "seeds": [int(value) for value in data["request"]["seeds"]],
                "points_per_seed": int(data["request"]["points_per_seed"]),
                "mean_sigma": float(primary["mean_sigma"]),
                "sigma_95_percent_ci": [
                    float(value) for value in primary["sigma_95_percent_ci"]
                ],
                "minimum_importance_effective_sample_size": float(
                    primary["min_importance_effective_sample_size"]
                ),
                "hard_gates": {
                    "maximum_atlas_error": float(thresholds["max_atlas_error"]),
                    "minimum_jacobian_singular_value": float(
                        thresholds["min_jacobian_singular_value"]
                    ),
                    "minimum_metric_eigenvalue": float(
                        thresholds["min_metric_eigenvalue"]
                    ),
                    "minimum_effective_sample_size": float(
                        thresholds["min_effective_sample_size"]
                    ),
                    "maximum_primary_mean_sigma": (
                        None
                        if thresholds["max_primary_mean_sigma"] is None
                        else float(thresholds["max_primary_mean_sigma"])
                    ),
                    "require_mean_sigma_improvement": bool(
                        thresholds["require_mean_sigma_improvement"]
                    ),
                    "require_every_seed_sigma_improvement": bool(
                        thresholds["require_every_seed_sigma_improvement"]
                    ),
                    "require_ordered_artifact_sigma_improvement": bool(
                        thresholds.get(
                            "require_ordered_artifact_sigma_improvement", False
                        )
                    ),
                },
            }
        )
        sources.append(path)
    return rows, sources


def smoothness_rows() -> tuple[list[dict[str, Any]], list[Path]]:
    rows = []
    sources: list[Path] = []
    for geometry, type_label, model_seed, names in SMOOTHNESS_GROUPS:
        primes = []
        chart_counts = []
        for name in names:
            path = ROOT / name
            data = load(path)
            characteristic = int(data["characteristic"])
            proved = int(
                data.get(
                    "gcicy_charts_proved_smooth",
                    data.get("charts_proved_smooth", 0),
                )
            )
            requested = int(data["charts_requested"])
            if not data["all_proved_smooth"] or proved != requested:
                raise SystemExit(f"incomplete smoothness certificate: {path}")
            primes.append(characteristic)
            chart_counts.append(proved)
            sources.append(path)
        if len(set(chart_counts)) != 1:
            raise SystemExit(f"smoothness chart counts differ for {geometry}")
        rows.append(
            {
                "geometry": geometry,
                "type": type_label,
                "model_seed": model_seed,
                "primes": primes,
                "charts_per_prime": chart_counts[0],
                "chart_prime_certificates": sum(chart_counts),
                "files": [workspace_path(ROOT / name) for name in names],
            }
        )
    family_path = ROOT / "outputs/gcicy_exact_model_screen.json"
    family = load(family_path)
    if int(family["models_accepted_all_characteristics"]) != 5:
        raise SystemExit("type-(2,2) exact model-family screen is incomplete")
    rows.append(
        {
            "geometry": "$X_{22}$ family",
            "type": "(2,2), five models",
            "model_seed": "20260711--20260715",
            "primes": [int(value) for value in family["characteristics"]],
            "charts_per_prime": "5x24",
            "chart_prime_certificates": int(
                family["models_accepted_all_characteristics"]
                * len(family["characteristics"])
                * family["charts_per_model_prime"]
            ),
            "files": [workspace_path(family_path)],
        }
    )
    sources.append(family_path)
    return rows, sources


def section_rank_rows(path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    data = load(path)
    if not data["all_certificates_passed"]:
        raise SystemExit("finite-field section-rank certificates failed")
    rows = []
    for certificate in data["certificates"]:
        for degree in certificate["degrees"]:
            determinant = int(degree["pivot_minor_determinant_mod_prime"])
            target = int(degree["riemann_roch_target"])
            rank = int(degree["finite_field_rank"])
            if rank != target or determinant % int(certificate["prime"]) == 0:
                raise SystemExit("invalid finite-field section-rank witness")
            artifact_rank = degree.get("reported_artifact_finite_field_rank")
            artifact_determinant = degree.get(
                "reported_artifact_pivot_minor_determinant_mod_prime"
            )
            if artifact_rank is not None and (
                int(artifact_rank) != target
                or int(artifact_determinant) % int(certificate["prime"]) == 0
            ):
                raise SystemExit("reported metric artifact section basis is not certified")
            rows.append(
                {
                    "geometry": certificate["geometry"],
                    "degree": [int(value) for value in degree["degree"]],
                    "prime": int(certificate["prime"]),
                    "ambient_monomial_count": int(
                        degree["ambient_monomial_count"]
                    ),
                    "riemann_roch_target": target,
                    "finite_field_rank": rank,
                    "pivot_minor_determinant_mod_prime": determinant,
                    "reported_artifact": degree.get("reported_artifact"),
                    "reported_artifact_sha256": degree.get(
                        "reported_artifact_sha256"
                    ),
                    "reported_artifact_finite_field_rank": artifact_rank,
                    "reported_artifact_pivot_minor_determinant_mod_prime": (
                        artifact_determinant
                    ),
                    "certificate_sha256": sha256(path),
                }
            )
    return rows, [path]


def topological_target_rows(path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    data = load(path)
    if not data["all_certificates_passed"]:
        raise SystemExit("exact topological target certificates failed")
    expected = {
        "X11_m3": (23, 86),
        "X21": (28, 76),
        "X22": (50, 104),
        "X11_m4": (26, 92),
    }
    rows = data["certificates"]
    if {row["geometry"] for row in rows} != set(expected):
        raise SystemExit("topological target inventory mismatch")
    for row in rows:
        observed = (
            int(row["polarization_cube"]),
            int(row["c2_polarization"]),
        )
        if observed != expected[row["geometry"]] or not row["passed"]:
            raise SystemExit(f"invalid topological target: {row['geometry']}")
    return rows, [path]


def sampler_rows(
    spectrum_path: Path,
    type21_path: Path,
    type22_path: Path,
    m4_path: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    spectrum = load(spectrum_path)
    diagnostics = spectrum["sampling_diagnostics"]
    x11 = {
        "geometry": "$X_{11}$",
        "dataset": "$X_{11}$ scalar Ritz audit",
        "roots_per_fibre": 4,
        "seeds": len(diagnostics),
        "points_per_seed": int(spectrum["points_per_seed"]),
        "attempted_clusters": sum(int(row["attempted_clusters"]) for row in diagnostics),
        "accepted_clusters": sum(int(row["accepted_clusters"]) for row in diagnostics),
        "rejected_clusters": sum(int(row["rejected_clusters"]) for row in diagnostics),
        "minimum_projective_root_separation": min(
            float(row["minimum_projective_root_separation"]) for row in diagnostics
        ),
        "maximum_accepted_relative_residual": max(
            float(row["maximum_accepted_relative_residual"]) for row in diagnostics
        ),
        "minimum_jacobian_singular_value": None,
        "importance_weight_used_for_acceptance": False,
    }
    rows = [x11]
    for geometry, path, roots in (
        ("$X_{21}$", type21_path, 6),
        ("$X_{22}$", type22_path, 3),
        ("$X_{11}^{(4)}$", m4_path, 4),
    ):
        data = load(path)
        if not data["success"] or not all(data["gates"].values()):
            raise SystemExit(f"failed numerical degree-completeness audit: {path}")
        aggregate = data["aggregate"]
        rows.append(
            {
                "geometry": geometry,
                "dataset": f"{geometry} formal audit",
                "roots_per_fibre": roots,
                "seeds": len(data["seeds"]),
                "points_per_seed": int(data["points_per_seed"]),
                "attempted_clusters": int(aggregate["attempted_clusters"]),
                "accepted_clusters": int(aggregate["accepted_clusters"]),
                "rejected_clusters": int(aggregate["rejected_clusters"]),
                "minimum_projective_root_separation": float(
                    aggregate["minimum_projective_root_separation"]
                ),
                "maximum_accepted_relative_residual": float(
                    aggregate["maximum_accepted_relative_residual"]
                ),
                "minimum_jacobian_singular_value": float(
                    aggregate["minimum_jacobian_singular_value"]
                ),
                "importance_weight_used_for_acceptance": False,
                "audit_type": "sampler",
                "hard_gates": {
                    "require_all_requested_points_returned": True,
                    "require_numerically_degree_complete_clusters": True,
                    "required_roots_per_fibre": roots,
                    "minimum_projective_root_separation": 0.0,
                    "maximum_accepted_relative_residual": float(
                        min(row["residual_tolerance"] for row in data["per_seed"])
                    ),
                    "require_no_importance_weight_input_to_acceptance": True,
                },
            }
        )
    return rows, [spectrum_path, type21_path, type22_path, m4_path]


def large_metric_rows(
    type22_path: Path,
    bicubic_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    manifest = load(manifest_path)
    manifest_runs = {
        (manifest_path.parent / row["output"]).resolve(): row
        for row in manifest["runs"]
    }
    rows = []
    for geometry, roots, path in (
        ("$X_{22}$", 3, type22_path),
        ("bicubic", 3, bicubic_path),
    ):
        resolved_path = path.resolve()
        if resolved_path not in manifest_runs:
            raise SystemExit(f"large-metric result is absent from manifest: {path}")
        specification = manifest_runs[resolved_path]
        data = load(path)
        if not data["success"] or not all(data["gates"].values()):
            raise SystemExit(f"failed large-sample metric audit: {path}")
        expected_seeds = [int(value) for value in specification["seeds"]]
        if [int(value) for value in data["seeds"]] != expected_seeds:
            raise SystemExit(f"large-metric seed mismatch: {path}")
        if int(data["points_per_seed"]) != int(specification["points_per_seed"]):
            raise SystemExit(f"large-metric point-count mismatch: {path}")
        if int(specification["required_roots_per_fibre"]) != roots:
            raise SystemExit(f"large-metric root-count mismatch: {path}")
        if Path(data["artifact"]).name != Path(specification["artifact"]).name:
            raise SystemExit(f"large-metric artifact mismatch: {path}")
        aggregate = data["aggregate"]
        required_point_ess = float(
            specification["minimum_point_effective_sample_size"]
        )
        if float(aggregate["minimum_point_effective_sample_size"]) < required_point_ess:
            raise SystemExit(f"large-metric specified ESS gate failed: {path}")
        sampler_rows = [entry["sampler"] for entry in data["per_seed"]]
        if any(
            int(entry["expected_points_per_cluster"]) != roots
            or bool(entry["importance_weight_used_for_acceptance"])
            for entry in sampler_rows
        ):
            raise SystemExit(f"large-metric acceptance contract failed: {path}")
        rows.append(
            {
                "geometry": geometry,
                "dataset": f"{geometry} large metric",
                "roots_per_fibre": roots,
                "points_per_seed": int(data["points_per_seed"]),
                "seed_count": len(data["seeds"]),
                "total_points": int(data["total_points"]),
                "mean_sigma": float(aggregate["mean_sigma"]),
                "sigma_95_percent_ci": [
                    float(value) for value in aggregate["sigma_95_percent_ci"]
                ],
                "minimum_point_effective_sample_size": float(
                    aggregate["minimum_point_effective_sample_size"]
                ),
                "minimum_cluster_weight_effective_sample_size": float(
                    aggregate["minimum_cluster_weight_effective_sample_size"]
                ),
                "minimum_metric_eigenvalue": float(
                    aggregate["minimum_metric_eigenvalue"]
                ),
                "minimum_point_effective_sample_size_required": required_point_ess,
                "attempted_clusters": sum(
                    int(entry["attempted_clusters"]) for entry in sampler_rows
                ),
                "accepted_clusters": sum(
                    int(entry["accepted_clusters"]) for entry in sampler_rows
                ),
                "rejected_clusters": sum(
                    int(entry["rejected_clusters"]) for entry in sampler_rows
                ),
                "minimum_projective_root_separation": min(
                    float(entry["minimum_projective_root_separation"])
                    for entry in sampler_rows
                ),
                "maximum_accepted_relative_residual": max(
                    float(entry["maximum_accepted_relative_residual"])
                    for entry in sampler_rows
                ),
                "minimum_jacobian_singular_value": None,
                "importance_weight_used_for_acceptance": False,
                "hard_gates": {
                    "minimum_point_effective_sample_size": required_point_ess,
                    "require_positive_metric": bool(
                        specification["require_positive_metric"]
                    ),
                    "require_complete_sampling_clusters": bool(
                        specification["require_complete_sampling_clusters"]
                    ),
                    "require_no_importance_weight_input_to_acceptance": bool(
                        specification[
                            "require_no_importance_weight_input_to_acceptance"
                        ]
                    ),
                    "required_roots_per_fibre": roots,
                },
            }
        )
    return rows, [type22_path, bicubic_path, manifest_path]


def rank_robustness_rows(
    path: Path,
    reference_spectrum_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    data = load(path)
    reference = load(reference_spectrum_path)
    if not data["success"] or not all(data["gates"].values()):
        raise SystemExit(f"failed mass-rank robustness audit: {path}")
    if float(data["mass_relative_threshold"]) != 1e-9:
        raise SystemExit("unexpected primary mass threshold")
    if len(data["artifacts"]) != 1:
        raise SystemExit("mass-rank robustness audit must use one metric artifact")
    levels = data["artifacts"][0]["trial_levels"]
    if len(levels) != 1 or int(levels[0]["level"]) != 3:
        raise SystemExit("mass-rank robustness audit must use only trial level 3")
    level = levels[0]
    sweep = level.get("mass_threshold_sweep", [])
    if [float(row["mass_relative_threshold"]) for row in sweep] != [
        1e-8,
        1e-9,
        1e-10,
    ]:
        raise SystemExit("mass-rank robustness threshold grid mismatch")

    reference_level = next(
        row
        for row in reference["artifacts"][0]["trial_levels"]
        if int(row["level"]) == 3
    )
    primary = next(
        row for row in sweep if float(row["mass_relative_threshold"]) == 1e-9
    )
    for observed, expected in zip(
        primary["eigenvalues"][:3],
        reference_level["eigenvalues"][:3],
        strict=True,
    ):
        tolerance = 1e-12 * max(1.0, abs(float(expected["mean"])))
        if abs(float(observed["mean"]) - float(expected["mean"])) > tolerance:
            raise SystemExit("primary-threshold spectrum changed in robustness rerun")

    rows = [
        {
            "mass_relative_threshold": float(row["mass_relative_threshold"]),
            "retained_trial_ranks": [
                int(value) for value in row["retained_trial_ranks"]
            ],
            "eigenvalues": [
                float(entry["mean"]) for entry in row["eigenvalues"][:3]
            ],
            "minimum_smallest_retained_relative_mass_eigenvalue": float(
                row["minimum_smallest_retained_relative_mass_eigenvalue"]
            ),
            "maximum_largest_discarded_relative_mass_eigenvalue": (
                None
                if row["maximum_largest_discarded_relative_mass_eigenvalue"] is None
                else float(
                    row["maximum_largest_discarded_relative_mass_eigenvalue"]
                )
            ),
            "first_three_relative_changes_from_primary": [
                float(value)
                for value in row["first_three_relative_changes_from_primary"]
            ],
        }
        for row in sweep
    ]
    summary = {
        "primary_threshold": float(level["mass_threshold_sweep_primary_threshold"]),
        "maximum_absolute_first_three_threshold_relative_change": float(
            level["maximum_absolute_first_three_threshold_relative_change"]
        ),
        "hard_gates": {
            "minimum_effective_sample_size": float(
                data["minimum_effective_sample_size_required"]
            ),
            "minimum_effective_samples_per_retained_direction": float(
                data["minimum_ess_per_retained_rank_required"]
            ),
            "minimum_cluster_effective_samples_per_retained_direction": float(
                data["minimum_cluster_ess_per_retained_rank_required"]
            ),
            "require_complete_sampling_clusters": True,
        },
    }
    worker_dir = path.parent / f"{path.stem}_workers"
    worker_results = sorted(worker_dir.glob("seed_*.json"))
    worker_specs = sorted(worker_dir.glob("seed_*_spec.json"))
    worker_logs = sorted(worker_dir.glob("seed_*.log"))
    result_only = [entry for entry in worker_results if not entry.name.endswith("_spec.json")]
    if not (
        len(result_only) == len(data["seeds"])
        and len(worker_specs) == len(data["seeds"])
        and len(worker_logs) == len(data["seeds"])
    ):
        raise SystemExit("mass-rank robustness worker inventory is incomplete")
    return rows, summary, [
        path,
        reference_spectrum_path,
        *result_only,
        *worker_specs,
        *worker_logs,
    ]


def metric_systematic_rows(
    paths: tuple[Path, ...],
    *,
    study: str,
    retain_first_as_failed_diagnostic: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    rows = []
    comparisons = []
    expected_diagnostic_failures = {
        "integration_effective_sample_size",
        "effective_samples_per_retained_direction",
    }
    for index, path in enumerate(paths):
        data = load(path)
        retained_failed_diagnostic = bool(
            index == 0 and retain_first_as_failed_diagnostic
        )
        failed_gates = {
            key for key, value in data["gates"].items() if not value
        }
        if retained_failed_diagnostic:
            if bool(data["success"]) or failed_gates != expected_diagnostic_failures:
                raise SystemExit(
                    "the retained N=8192 diagnostic no longer has exactly the "
                    "two specified ESS failures"
                )
        elif not bool(data["success"]) or failed_gates:
            raise SystemExit(
                "the required N=32768 metric-systematic escalation failed"
            )
        for artifact in data["artifacts"]:
            level = artifact["trial_levels"][0]
            point_ratio = float(
                level["minimum_effective_samples_per_retained_direction"]
            )
            cluster_ratio = float(
                level.get(
                    "minimum_cluster_weight_ess_per_retained_rank",
                    level["minimum_effective_clusters_per_retained_direction"],
                )
            )
            rows.append(
                {
                    "study": study,
                    "points": int(data["points_per_seed"]),
                    "metric_degree": int(artifact["degree"][0]),
                    "retained_rank": list(level["retained_trial_ranks"]),
                    "eigenvalues": [
                        float(entry["mean"]) for entry in level["eigenvalues"][:3]
                    ],
                    "minimum_point_ess_per_rank": point_ratio,
                    "minimum_cluster_weight_ess_per_rank": cluster_ratio,
                    "required": not retained_failed_diagnostic,
                    "status": (
                        "expected_gate_failure"
                        if retained_failed_diagnostic
                        else "passed"
                    ),
                    "superseded_by": (
                        workspace_path(paths[1])
                        if retained_failed_diagnostic and len(paths) > 1
                        else None
                    ),
                    "manuscript_table": "tab:spectrum-metric-systematic",
                    "ess_gates_passed": bool(
                        point_ratio
                        >= float(data["minimum_ess_per_retained_rank_required"])
                        and cluster_ratio
                        >= float(
                            data["minimum_cluster_ess_per_retained_rank_required"]
                        )
                        and level["minimum_integration_effective_sample_size"]
                        >= float(data["minimum_effective_sample_size_required"])
                    ),
                }
            )
        paired = data["artifact_pairwise_comparisons"][0]
        comparisons.append(
            {
                "study": study,
                "points": int(data["points_per_seed"]),
                "source_degree": int(data["artifacts"][0]["degree"][0]),
                "target_degree": int(data["artifacts"][1]["degree"][0]),
                "first_three_modes": paired["eigenvalues"][:3],
                "audit_success": bool(data["success"]),
                "failed_gates": sorted(failed_gates),
                "required": not retained_failed_diagnostic,
                "status": (
                    "expected_gate_failure"
                    if retained_failed_diagnostic
                    else "passed"
                ),
                "superseded_by": (
                    workspace_path(paths[1])
                    if retained_failed_diagnostic and len(paths) > 1
                    else None
                ),
                "manuscript_table": "tab:spectrum-metric-systematic",
            }
        )
    return rows, comparisons, list(paths)


def hard_gate_rows(
    metric_tests: list[dict[str, Any]],
    samplers: list[dict[str, Any]],
    large_metrics: list[dict[str, Any]],
    rank_summary: dict[str, Any],
    final_spectrum_path: Path,
    metric_systematic_path: Path,
    matched_metric_systematic_path: Path,
    spectrum_8192_path: Path,
    spectrum_32768_path: Path,
    multitype_evidence_path: Path,
) -> list[dict[str, Any]]:
    rows = [
        {
            "dataset": f"{row['geometry']} final metric",
            "audit_type": "metric",
            "hard_gates": row["hard_gates"],
            "passed": True,
        }
        for row in metric_tests
    ]
    rows.extend(
        {
            "dataset": row["dataset"],
            "audit_type": "sampler",
            "hard_gates": row["hard_gates"],
            "passed": True,
        }
        for row in samplers
        if row.get("audit_type") == "sampler"
    )
    rows.extend(
        {
            "dataset": row["dataset"],
            "audit_type": "large_metric",
            "hard_gates": row["hard_gates"],
            "passed": True,
        }
        for row in large_metrics
    )

    for dataset, path in (
        ("$X_{11}$ final scalar Ritz audit", final_spectrum_path),
        ("$X_{11}$ metric-sensitivity escalation", metric_systematic_path),
        (
            "$X_{11}$ matched-loss metric sensitivity",
            matched_metric_systematic_path,
        ),
    ):
        data = load(path)
        if not data["success"] or not all(data["gates"].values()):
            raise SystemExit(f"failed scalar-Ritz hard gates: {path}")
        jackknife = data["cluster_jackknife_configuration"]
        rows.append(
            {
                "dataset": dataset,
                "audit_type": "scalar_spectrum",
                "hard_gates": {
                    "minimum_effective_sample_size": float(
                        data["minimum_effective_sample_size_required"]
                    ),
                    "minimum_effective_samples_per_retained_direction": float(
                        data["minimum_ess_per_retained_rank_required"]
                    ),
                    "minimum_cluster_effective_samples_per_retained_direction": float(
                        data["minimum_cluster_ess_per_retained_rank_required"]
                    ),
                    "maximum_cluster_jackknife_relative_standard_error": (
                        None
                        if jackknife["maximum_relative_standard_error"] is None
                        else float(jackknife["maximum_relative_standard_error"])
                    ),
                    "require_complete_sampling_clusters": True,
                    "require_positive_finite_spectrum": True,
                    "require_stable_trial_rank": True,
                },
                "passed": True,
            }
        )
    rows.append(
        {
            "dataset": "$X_{11}$ mass-threshold sweep",
            "audit_type": "rank_robustness",
            "hard_gates": rank_summary["hard_gates"],
            "passed": True,
        }
    )
    spectra = [
        load(spectrum_8192_path),
        load(spectrum_32768_path),
        load(final_spectrum_path),
    ]

    def means(data: dict[str, Any], level_number: int) -> list[float]:
        level = next(
            entry
            for entry in data["artifacts"][-1]["trial_levels"]
            if int(entry["level"]) == level_number
        )
        return [float(entry["mean"]) for entry in level["eigenvalues"][:3]]

    level3 = [means(data, 3) for data in spectra]
    level2_final = means(spectra[-1], 2)
    sample_8192_to_32768 = max(
        abs((target - source) / source)
        for source, target in zip(level3[0], level3[1], strict=True)
    )
    sample_32768_to_65536 = max(
        abs((target - source) / source)
        for source, target in zip(level3[1], level3[2], strict=True)
    )
    trial_level2_to_level3 = max(
        abs((target - source) / source)
        for source, target in zip(level2_final, level3[2], strict=True)
    )
    stability_gates = {
        "maximum_sample_shift_8192_to_32768": 0.03,
        "maximum_sample_shift_32768_to_65536": 0.02,
        "maximum_level2_to_level3_shift_at_65536": 0.01,
    }
    stability_passed = bool(
        sample_8192_to_32768
        <= stability_gates["maximum_sample_shift_8192_to_32768"]
        and sample_32768_to_65536
        <= stability_gates["maximum_sample_shift_32768_to_65536"]
        and trial_level2_to_level3
        <= stability_gates["maximum_level2_to_level3_shift_at_65536"]
    )
    if not stability_passed:
        raise SystemExit("failed scalar sample/trial stability hard gates")
    rows.append(
        {
            "dataset": "$X_{11}$ scalar stability",
            "audit_type": "scalar_stability",
            "hard_gates": stability_gates,
            "observed": {
                "sample_shift_8192_to_32768": sample_8192_to_32768,
                "sample_shift_32768_to_65536": sample_32768_to_65536,
                "level2_to_level3_shift_at_65536": trial_level2_to_level3,
            },
            "passed": stability_passed,
        }
    )
    multitype = load(multitype_evidence_path)
    basis = multitype["basis_seed_robustness"]
    basis_gates = {
        "maximum_relative_standard_deviation": float(
            basis["maximum_relative_standard_deviation_gate"]
        ),
        "maximum_relative_range": float(basis["maximum_relative_range_gate"]),
    }
    basis_passed = bool(
        max(float(value) for value in basis["relative_standard_deviations"])
        <= basis_gates["maximum_relative_standard_deviation"]
        and max(float(value) for value in basis["relative_ranges"])
        <= basis_gates["maximum_relative_range"]
    )
    if not basis_passed:
        raise SystemExit("failed scalar cubic basis-seed hard gates")
    rows.append(
        {
            "dataset": "$X_{11}$ cubic basis seeds",
            "audit_type": "scalar_basis_seed",
            "hard_gates": basis_gates,
            "observed": {
                "relative_standard_deviations": basis[
                    "relative_standard_deviations"
                ],
                "relative_ranges": basis["relative_ranges"],
            },
            "passed": basis_passed,
        }
    )
    return rows


def hard_gate_latex(row: dict[str, Any]) -> str:
    gates = row["hard_gates"]
    if row["audit_type"] == "metric":
        pieces = [
            f"atlas error $\\leq {gates['maximum_atlas_error']:.0e}$",
            (
                "$s_{\\min}(J)\\geq "
                f"{gates['minimum_jacobian_singular_value']:.0e}$"
            ),
            "$\\lambda_{\\min}(g)>0$",
            f"pESS $\\geq {gates['minimum_effective_sample_size']:.0f}$",
        ]
        if gates["maximum_primary_mean_sigma"] is not None:
            pieces.append(
                f"mean $\\sigma\\leq {gates['maximum_primary_mean_sigma']:.2f}$"
            )
        if gates["require_every_seed_sigma_improvement"]:
            pieces.append("every-seed $\\sigma$ improvement")
        if gates["require_ordered_artifact_sigma_improvement"]:
            pieces.append("ordered artifact improvement")
        return "; ".join(pieces)
    if row["audit_type"] == "large_metric":
        return "; ".join(
            (
                f"pESS $\\geq {gates['minimum_point_effective_sample_size']:.0f}$",
                "$\\lambda_{\\min}(g)>0$",
                (
                    "numerically degree-complete "
                    f"{gates['required_roots_per_fibre']}-root fibres"
                ),
                "no importance-weight input to acceptance",
            )
        )
    if row["audit_type"] == "sampler":
        return "; ".join(
            (
                "all requested points returned",
                (
                    "numerically degree-complete "
                    f"{gates['required_roots_per_fibre']}-root clusters"
                ),
                "projective separation $>0$",
                (
                    "accepted residual $\\leq "
                    f"{gates['maximum_accepted_relative_residual']:.0e}$"
                ),
                "no importance-weight input to acceptance",
            )
        )
    if row["audit_type"] == "scalar_stability":
        return "; ".join(
            (
                "$N:8192\\to32768$ shift $\\leq 3\\%$",
                "$N:32768\\to65536$ shift $\\leq 2\\%$",
                "level-2 to level-3 shift $\\leq 1\\%$",
            )
        )
    if row["audit_type"] == "scalar_basis_seed":
        return "; ".join(
            (
                "four cubic seeds",
                "max. modewise RSD $\\leq 2\\%$",
                "max. modewise range $\\leq 5\\%$",
            )
        )
    pieces = [
        f"pESS $\\geq {gates['minimum_effective_sample_size']:.0f}$",
        (
            "pESS/rank $\\geq "
            f"{gates['minimum_effective_samples_per_retained_direction']:.0f}$"
        ),
        (
            "cwESS/rank $\\geq "
            f"{gates['minimum_cluster_effective_samples_per_retained_direction']:.0f}$"
        ),
        "numerically degree-complete clusters",
    ]
    jackknife = gates.get("maximum_cluster_jackknife_relative_standard_error")
    if jackknife is not None:
        pieces.append(f"jackknife RSE $\\leq {100 * jackknife:.0f}\\%$")
    return "; ".join(pieces)


def latex(summary: dict[str, Any]) -> str:
    lines = [
        "% Generated by scripts/build_reproducibility_evidence.py; do not edit.",
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Accepted metric-training sample specification.  The check column gives the number of fixed checkpoint seeds times the points per seed.}",
        "\\label{tab:training-samples}",
        "\\begin{tabular}{lrrlrrrrr}",
        "\\toprule",
        "geometry & $k$ & $N_k$ & parameterization & basis & train & validation & check & export \\\\",
        "\\midrule",
    ]
    for row in summary["training"]:
        parameterization = (
            row["parameterization"]
            .replace("full complex Cholesky", "full Cholesky")
            .replace("direct positive ", "")
            .replace("whitened positive ", "whitened ")
            .replace(" congruence", "")
        )
        lines.append(
            f"{row['geometry']} & {row['degree']} & {row['section_count']} & "
            f"{parameterization} & {row['basis_points']} & "
            f"{row['train_points']} & "
            f"{row['validation_points']} & "
            f"{row['check_seed_count']}x{row['check_points_per_seed']} & "
            f"{row['export_points']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Sampler-matched large-sample metric tests.  Both rows use eight seeds, $6144$ points per seed, complete three-root fibres, and descriptive seed-level normal-approximation intervals.}",
            "\\label{tab:large-metric-controls}",
            "\\begin{tabular}{lrrrrrrr}",
            "\\toprule",
            "geometry & total $N$ & $\\sigma$ & 95\\% interval & min. pESS & min. cwESS & min. $\\lambda(g)$ \\\\",
            "\\midrule",
        ]
    )
    for row in summary["large_metric_tests"]:
        interval = row["sigma_95_percent_ci"]
        lines.append(
            f"{row['geometry']} & {row['total_points']} & "
            f"{row['mean_sigma']:.5f} & "
            f"[{interval[0]:.5f},{interval[1]:.5f}] & "
            f"{row['minimum_point_effective_sample_size']:.0f} & "
            f"{row['minimum_cluster_weight_effective_sample_size']:.0f} & "
            f"{row['minimum_metric_eigenvalue']:.2e} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Paired metric-systematic scalar Ritz test on identical samples and the complete level-2 trial space.  A failed ESS entry is retained as a diagnostic rather than reclassified after inspection.}",
            "\\label{tab:spectrum-metric-systematic}",
            "\\begin{tabular}{lrrrrrrr}",
            "\\toprule",
            "study & $N$ & metric $k$ & $\\lambda_1$ & $\\lambda_2$ & $\\lambda_3$ & pESS/rank & cwESS/rank \\\\",
            "\\midrule",
        ]
    )
    for row in summary["metric_systematic_rows"]:
        eigenvalues = row["eigenvalues"]
        suffix = "" if row["ess_gates_passed"] else "$^{*}$"
        lines.append(
            f"{row['study']} & {row['points']} & "
            f"{row['metric_degree']}{suffix} & "
            f"{eigenvalues[0]:.5f} & {eigenvalues[1]:.5f} & "
            f"{eigenvalues[2]:.5f} & "
            f"{row['minimum_point_ess_per_rank']:.2f} & "
            f"{row['minimum_cluster_weight_ess_per_rank']:.2f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{minipage}{0.94\\textwidth}\\footnotesize",
            "$^{*}$The specified point-ESS gate failed; the row is shown only as a retained diagnostic.",
            "\\end{minipage}",
            "\\end{table}",
            "",
        ]
    )
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Mass-matrix rank-threshold robustness on the same eight $N=65536$ samples and level-3 trial matrices as the final scalar calculation.  Eigenvalue intervals and threshold changes are descriptive diagnostics.}",
            "\\label{tab:spectrum-rank-robustness}",
            "\\begin{tabular}{lrrrrrrr}",
            "\\toprule",
            "$\\tau_M$ & retained ranks & $\\lambda_1$ & $\\lambda_2$ & $\\lambda_3$ & max. discarded & min. retained & max. $|\\Delta_{1:3}|$ \\\\",
            "\\midrule",
        ]
    )
    maximum_change = summary["rank_robustness_summary"][
        "maximum_absolute_first_three_threshold_relative_change"
    ]
    for row in summary["rank_robustness"]:
        eigenvalues = row["eigenvalues"]
        ranks = ",".join(str(value) for value in row["retained_trial_ranks"])
        discarded = row["maximum_largest_discarded_relative_mass_eigenvalue"]
        discarded_text = "--" if discarded is None else f"{discarded:.2e}"
        lines.append(
            f"{row['mass_relative_threshold']:.0e} & {ranks} & "
            f"{eigenvalues[0]:.5f} & {eigenvalues[1]:.5f} & "
            f"{eigenvalues[2]:.5f} & {discarded_text} & "
            f"{row['minimum_smallest_retained_relative_mass_eigenvalue']:.2e} & "
            f"{maximum_change:.2e} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Hard numerical gates encoded in the audit specifications and used for acceptance.  Quantities reported elsewhere but absent here are descriptive diagnostics, not additional pass/fail criteria.}",
            "\\label{tab:hard-numerical-gates}",
            "\\begin{tabular}{@{}p{0.21\\textwidth}p{0.75\\textwidth}@{}}",
            "\\toprule",
            "dataset & hard acceptance gates \\\\",
            "\\midrule",
        ]
    )
    for row in summary["hard_numerical_gates"]:
        lines.append(f"{row['dataset']} & {hard_gate_latex(row)} \\\\ ")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Training split seeds.  A dash in the Torch column means that the deterministic full-Cholesky initialization introduced no random optimizer parameter.}",
            "\\label{tab:training-seeds}",
            "\\begin{tabular}{lrrrrrl}",
            "\\toprule",
            "geometry & $k$ & Torch & basis & train & validation & check seeds / export \\\\",
            "\\midrule",
        ]
    )
    for row in summary["training"]:
        seeds = row["seeds"]
        torch_seed = "--" if seeds["torch"] is None else str(seeds["torch"])
        check_text = ",".join(str(value) for value in seeds["checks"])
        lines.append(
            f"{row['geometry']} & {row['degree']} & {torch_seed} & "
            f"{seeds['basis']} & {seeds['train']} & {seeds['validation']} & "
            f"{check_text} / {seeds['export']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Adam optimization and checkpoint specification.  All learning rates are constant.  The columns $\\beta$ and $\\delta$ are the normalized-$\\sigma$ loss and checkpoint-score weights; zero means centered log-MA training and selection.}",
            "\\label{tab:training-optimization}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "geometry & $k$ & max. epoch & selected & learning rate & $\\beta$ & $\\delta$ \\\\",
            "\\midrule",
        ]
    )
    for row in summary["training"]:
        lines.append(
            f"{row['geometry']} & {row['degree']} & {row['maximum_epochs']} & "
            f"{row['selected_epoch']} & {row['learning_rate']:.2e} & "
            f"{row['sigma_loss_weight']:.2f} & "
            f"{row['selection_sigma_weight']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Untouched final metric tests.  Intervals are descriptive seed-level normal-approximation intervals; no hypothesis-test interpretation is made.}",
            "\\label{tab:metric-test-specification}",
            "\\begin{tabular}{lrrlrr}",
            "\\toprule",
            "geometry & $k$ & points/seed & seeds & $\\sigma$ & 95\\% interval \\\\",
            "\\midrule",
        ]
    )
    for row in summary["metric_tests"]:
        seed_text = f"{row['seeds'][0]}--{row['seeds'][-1]}"
        interval = row["sigma_95_percent_ci"]
        lines.append(
            f"{row['geometry']} & {row['degree']} & {row['points_per_seed']} & "
            f"{seed_text} & {row['mean_sigma']:.5f} & "
            f"[{interval[0]:.5f},{interval[1]:.5f}] \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Complete finite-field smoothness manifests.  Each entry reports complete standard-chart covers at three good primes.}",
            "\\label{tab:smoothness-manifest}",
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "geometry & type & model seed & primes & charts/prime \\\\",
            "\\midrule",
        ]
    )
    for row in summary["smoothness"]:
        prime_text = ",".join(str(value) for value in row["primes"])
        lines.append(
            f"{row['geometry']} & {row['type']} & {row['model_seed']} & "
            f"{prime_text} & {row['charts_per_prime']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Exact finite-field restriction-rank witnesses.  Rank/determinant pairs give a target-size evaluation minor modulo the listed prime.  The span pair proves completeness of the ambient restriction space; the artifact pair separately certifies the section basis stored in the published NPZ.}",
            "\\label{tab:section-rank-certificates}",
            "\\begin{tabular}{lrrrrll}",
            "\\toprule",
            "geometry & degree & $p$ & ambient & RR & span $r/\\det$ & artifact $r/\\det$ \\\\",
            "\\midrule",
        ]
    )
    geometry_labels = {
        "X11_m3": "$X_{11}$",
        "X21": "$X_{21}$",
        "X22": "$X_{22}$",
        "X22_seed20260712": "$X_{22}^{(20260712)}$",
        "X11_m4": "$X_{11}^{(4)}$",
    }
    for row in summary["section_restriction_ranks"]:
        degree_text = "(" + ",".join(str(value) for value in row["degree"]) + ")"
        artifact_rank = row["reported_artifact_finite_field_rank"]
        artifact_determinant = row[
            "reported_artifact_pivot_minor_determinant_mod_prime"
        ]
        artifact_rank_text = (
            "--"
            if artifact_rank is None
            else f"{artifact_rank}/{artifact_determinant}"
        )
        lines.append(
            f"{geometry_labels[row['geometry']]} & {degree_text} & {row['prime']} & "
            f"{row['ambient_monomial_count']} & {row['riemann_roch_target']} & "
            f"{row['finite_field_rank']}/"
            f"{row['pivot_minor_determinant_mod_prime']} & "
            f"{artifact_rank_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{Numerical degree-completeness accounting for every dataset used in the sampler claims and large-sample controls.  The listed audits have zero rejected candidate fibres.  Computing importance weights after acceptance rules out explicit weight-based selection; no broader independence claim is made.}",
            "\\label{tab:complete-fibre-audits}",
            "\\begin{tabular}{lrrrrrrr}",
            "\\toprule",
            "dataset & roots & attempted & accepted & rejected & points/seed & min. sep. & max. residual \\\\",
            "\\midrule",
        ]
    )
    for row in summary["samplers"]:
        lines.append(
            f"{row['dataset']} & {row['roots_per_fibre']} & "
            f"{row['attempted_clusters']} & {row['accepted_clusters']} & "
            f"{row['rejected_clusters']} & "
            f"{row['points_per_seed']} & "
            f"{row['minimum_projective_root_separation']:.2e} & "
            f"{row['maximum_accepted_relative_residual']:.2e} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pipeline_rows, pipeline_sources = pipeline_training_rows()
    type22_rows, type22_sources = type22_training_rows(args.type22_training)
    smoothness, smoothness_sources = smoothness_rows()
    section_ranks, section_rank_sources = section_rank_rows(
        args.section_rank_certificates
    )
    topological_targets, topological_target_sources = topological_target_rows(
        args.topological_targets
    )
    metric_tests, metric_test_sources = metric_test_rows()
    samplers, sampler_sources = sampler_rows(
        args.type11_spectrum,
        args.type21_sampler,
        args.type22_sampler,
        args.m4_sampler,
    )
    large_metrics, large_metric_sources = large_metric_rows(
        args.type22_large_metric,
        args.bicubic_large_metric,
        args.large_metric_manifest,
    )
    samplers.extend(large_metrics)
    systematic_rows, systematic_comparisons, systematic_sources = (
        metric_systematic_rows(
            (args.metric_systematic_8192, args.metric_systematic_32768),
            study="original",
            retain_first_as_failed_diagnostic=True,
        )
    )
    matched_rows, matched_comparisons, matched_sources = metric_systematic_rows(
        (args.metric_systematic_tail_matched_32768,),
        study="matched loss",
        retain_first_as_failed_diagnostic=False,
    )
    systematic_rows.extend(matched_rows)
    systematic_comparisons.extend(matched_comparisons)
    systematic_sources.extend(matched_sources)
    rank_rows, rank_summary, rank_sources = rank_robustness_rows(
        args.rank_robustness,
        args.type11_spectrum,
    )
    hard_gates = hard_gate_rows(
        metric_tests,
        samplers,
        large_metrics,
        rank_summary,
        args.type11_spectrum,
        args.metric_systematic_32768,
        args.metric_systematic_tail_matched_32768,
        args.type11_spectrum_8192,
        args.type11_spectrum_32768,
        args.multitype_evidence,
    )
    sources = [
        *IMPLEMENTATION_SOURCES,
        *pipeline_sources,
        *type22_sources,
        *smoothness_sources,
        *section_rank_sources,
        *topological_target_sources,
        *metric_test_sources,
        *sampler_sources,
        *large_metric_sources,
        *systematic_sources,
        *rank_sources,
        args.type11_spectrum_8192,
        args.type11_spectrum_32768,
        args.multitype_evidence,
        args.metric_systematic_tail_matched_32768,
    ]
    unique_sources = list(dict.fromkeys(path.resolve() for path in sources))
    required_evidence_gates_passed = all(
        bool(row["passed"]) for row in hard_gates
    )
    all_indexed_audits_passed = bool(
        required_evidence_gates_passed
        and all(comparison["audit_success"] for comparison in systematic_comparisons)
    )
    summary = {
        "schema_version": 2,
        "description": (
            "Machine-readable training, checkpoint, exact-smoothness, and "
            "whole-fibre numerical degree-completeness reproducibility evidence."
        ),
        "training": [*pipeline_rows, *type22_rows],
        "smoothness": smoothness,
        "section_restriction_ranks": section_ranks,
        "topological_targets": topological_targets,
        "metric_tests": metric_tests,
        "samplers": samplers,
        "large_metric_tests": large_metrics,
        "metric_systematic_rows": systematic_rows,
        "metric_systematic_comparisons": systematic_comparisons,
        "rank_robustness": rank_rows,
        "rank_robustness_summary": rank_summary,
        "hard_numerical_gates": hard_gates,
        "statistical_conventions": {
            "seed_level_interval": (
                "descriptive mean +/- 1.96 sample_sd / sqrt(number_of_seeds)"
            ),
            "inferential_scope": (
                "Intervals summarize between-seed dispersion; no null-hypothesis "
                "test, coverage guarantee, or significance claim is attached."
            ),
            "spectrum_primary_estimator": "full-sample Monte Carlo matrices",
            "cluster_weight_ess_interpretation": (
                "importance-weight concentration diagnostic, not an "
                "observable-specific independent-sample count"
            ),
            "jackknife_grouping": (
                "clusters sorted by integer id and assigned index modulo 16"
            ),
            "spectrum_mass_relative_threshold": 1e-9,
            "mass_rank_robustness_thresholds": [1e-8, 1e-9, 1e-10],
        },
        "source_files": [
            {"path": workspace_path(path), "sha256": sha256(path)}
            for path in unique_sources
        ],
        "required_evidence_gates_passed": required_evidence_gates_passed,
        "audit_scope_semantics": {
            "required_evidence_gates_passed": (
                "Required publication evidence set, using final or escalated "
                "runs and excluding explicitly retained failed diagnostics."
            ),
            "all_recorded_audits_passed": (
                "Required evidence plus the explicitly indexed retained diagnostic "
                "rows, including the N=8192 point-ESS failure."
            ),
        },
        "all_recorded_audits_passed": all_indexed_audits_passed,
        "retained_failed_diagnostics": [
            comparison
            for comparison in systematic_comparisons
            if not comparison["audit_success"]
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_tex.write_text(latex(summary) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()
