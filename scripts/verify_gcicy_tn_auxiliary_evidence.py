#!/usr/bin/env python3
"""Recompute the sampler, timing, and rank summaries cited in the paper."""

from __future__ import annotations

import json
import math
from pathlib import Path
import zipfile


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent if (SCRIPT.parent / "outputs").is_dir() else SCRIPT.parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(actual: float, expected: float, *, tolerance: float = 5e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise SystemExit(f"numerical mismatch: expected {expected}, observed {actual}")


def check_samplers() -> None:
    cases = (
        (
            "outputs/pipeline/p4p1_type11_hirzebruch_x3_sampler_audit_formal_262k_20260717.json",
            65536,
            4,
            0.034145838835433595,
            3.1978044209379817e-15,
            33.50852819663335,
            0.07452656799376571,
        ),
        (
            "outputs/pipeline/p5p1_type21_sampler_audit_formal.json",
            8192,
            6,
            0.07022172845757975,
            7.430728978225895e-16,
            13.933501533416187,
            0.04968275239443896,
        ),
        (
            "outputs/pipeline/p1p1p5_type22_sampler_audit_formal.json",
            16384,
            3,
            0.05883010449865508,
            8.880174965485581e-16,
            116.50906814106307,
            0.1139243891235126,
        ),
    )
    for relative, clusters, roots, separation, residual, maximum_weight, top_one in cases:
        report = load(relative)
        aggregate = report["aggregate"]
        if not report["success"]:
            raise SystemExit(f"sampler audit did not pass: {relative}")
        for key in ("attempted_clusters", "accepted_clusters", "returned_complete_clusters"):
            if int(aggregate[key]) != clusters:
                raise SystemExit(f"sampler cluster-count mismatch: {relative}:{key}")
        if aggregate["root_count_histogram"] != {str(roots): clusters}:
            raise SystemExit(f"sampler root-count mismatch: {relative}")
        close(float(aggregate["minimum_projective_root_separation"]), separation)
        close(float(aggregate["maximum_accepted_relative_residual"]), residual)
        close(float(aggregate["maximum_mean_normalized_importance_weight"]), maximum_weight)
        close(float(aggregate["maximum_top_one_percent_importance_weight_fraction"]), top_one)


def check_timing() -> None:
    scaling = load(
        "outputs/pipeline/reviewer_timing_20260805/kernel_m1_m8_b256_c128.json"
    )
    rows = [row for row in scaling["results"] if int(row["site_count"]) >= 3]
    x = [float(row["site_count"]) for row in rows]
    y = [float(row["median_forward_seconds"]) for row in rows]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sum(
        (a - x_mean) ** 2 for a in x
    )
    intercept = y_mean - slope * x_mean
    residual = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(x, y))
    total = sum((b - y_mean) ** 2 for b in y)
    r_squared = 1.0 - residual / total
    close(slope, 0.002145931962877512)
    close(r_squared, 0.9999440599217156)

    synchronized = load(
        "outputs/pipeline/reviewer_timing_20260805/m3_d5_b256_c128.json"
    )
    close(float(synchronized["forward"]["median_seconds"]), 0.003869613050483167)
    close(
        float(synchronized["training_update"]["median_seconds"]),
        0.013338997378014028,
    )


def check_ranks_and_inputs() -> None:
    report = load("outputs/section_restriction_rank_certificates.json")
    if not report["all_certificates_passed"]:
        raise SystemExit("section restriction rank certificates do not all pass")
    expected = {
        ("X11_m3", (2, 2, 0)): (45, 45),
        ("X21", (1, 1)): (11, 11),
        ("X22", (1, 1, 1)): (17, 17),
    }
    observed: dict[tuple[str, tuple[int, ...]], tuple[int, int]] = {}
    for certificate in report["certificates"]:
        for row in certificate["degrees"]:
            observed[(certificate["geometry"], tuple(row["degree"]))] = (
                int(row["finite_field_rank"]),
                int(row["riemann_roch_target"]),
            )
    for key, value in expected.items():
        if observed.get(key) != value:
            raise SystemExit(f"source-rank mismatch: {key}")

    inputs = (
        "outputs/pipeline/p4p1_type11_hirzebruch_k2_gpu.npz",
        "outputs/pipeline/p4p1_type11_hirzebruch_k4_tail_refined_gpu.npz",
        "outputs/gcicy_generic_global_h_metric_k2_rank40_weighted.npz",
        "outputs/gcicy_generic_global_h_metric_k3_rank64_whitened_gpu.npz",
        "outputs/pipeline/gcicy_model_20260712_k3_rank64_whitened_gpu.npz",
    )
    for relative in inputs:
        path = ROOT / relative
        if not zipfile.is_zipfile(path):
            raise SystemExit(f"rank-generator input is missing or invalid: {relative}")
        with zipfile.ZipFile(path) as archive:
            if not any(name.endswith(".npy") for name in archive.namelist()):
                raise SystemExit(f"rank-generator input has no arrays: {relative}")


def main() -> None:
    check_samplers()
    check_timing()
    check_ranks_and_inputs()
    print("verified sampler audits, synchronized timing, scaling fit, and rank inputs")


if __name__ == "__main__":
    main()
