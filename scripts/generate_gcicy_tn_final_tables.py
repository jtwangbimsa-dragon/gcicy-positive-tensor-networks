#!/usr/bin/env python3
"""Generate the five final manuscript tables from frozen evidence.

The generator deliberately recomputes pointwise statistics from the archived
NPZ arrays.  JSON summaries are not used as the numerical source of the table
entries.  The only embedded constants are geometry and architecture choices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent if (SCRIPT.parent / "manuscript").is_dir() else SCRIPT.parents[1]
MANIFEST_NAME = "final_tables_generation_manifest.json"


@dataclass(frozen=True)
class GeneratedTable:
    filename: str
    content: str
    rows: tuple[tuple[str, ...], ...]
    column_counts: tuple[int, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, q, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def weighted_upper_mass_mean(
    values: np.ndarray, weights: np.ndarray, mass: float
) -> float:
    order = np.argsort(values)[::-1]
    remaining = float(mass)
    total = 0.0
    for index in order:
        contribution = min(float(weights[index]), remaining)
        total += contribution * float(values[index])
        remaining -= contribution
        if remaining <= 1e-15:
            break
    if remaining > 1e-12:
        raise ValueError("weights do not contain the requested upper-tail mass")
    return total / mass


def pointwise_metrics(relative: str) -> dict[str, float]:
    path = ROOT / relative
    with np.load(path) as arrays:
        log_eta = np.asarray(arrays["model_log_eta"], dtype=np.float64)
        weights = np.asarray(arrays["importance_weights"], dtype=np.float64)
        minimum_eigenvalues = (
            np.asarray(arrays["metric_minimum_eigenvalues"], dtype=np.float64)
            if "metric_minimum_eigenvalues" in arrays.files
            else None
        )
    weights = weights / weights.sum()
    shift = float(np.max(log_eta))
    log_mean = shift + math.log(float(np.sum(weights * np.exp(log_eta - shift))))
    log_ratio = log_eta - log_mean
    ratio = np.exp(log_ratio)
    absolute_log_ratio = np.abs(log_ratio)
    result = {
        "sigma": float(np.sum(weights * np.abs(ratio - 1.0))),
        "chi": float(np.sqrt(np.sum(weights * np.square(ratio - 1.0)))),
        "absolute_log_ratio_q999": weighted_quantile(
            absolute_log_ratio, weights, 0.999
        ),
        "absolute_log_ratio_cvar_1pct": weighted_upper_mass_mean(
            absolute_log_ratio, weights, 0.01
        ),
        "maximum_absolute_log_ratio": float(np.max(absolute_log_ratio)),
    }
    if minimum_eigenvalues is not None:
        result["minimum_metric_eigenvalue"] = float(np.min(minimum_eigenvalues))
    return result


def aggregate_metrics(rows: Iterable[dict[str, float]]) -> dict[str, tuple[float, float]]:
    records = tuple(rows)
    keys = tuple(records[0])
    return {
        key: (
            statistics.mean(float(row[key]) for row in records),
            statistics.stdev(float(row[key]) for row in records),
        )
        for key in keys
        if all(key in row for row in records)
    }


def table_content(
    preamble: list[str], rows: tuple[tuple[str, ...], ...], postamble: list[str]
) -> str:
    rendered_rows = [" & ".join(row) + r" \\" for row in rows]
    return "\n".join([*preamble, *rendered_rows, *postamble, ""])


def architecture_specification() -> dict:
    return load_json("pipeline_specs/final_table_architectures.json")


def h0_dimension(
    geometry: str, degree: int, specification: dict | None = None
) -> int:
    if specification is None:
        specification = architecture_specification()
    polynomial = specification["hilbert_polynomials"][geometry]
    numerator = (
        int(polynomial["cubic_coefficient"]) * degree**3
        + int(polynomial["linear_coefficient"]) * degree
    )
    denominator = int(polynomial["denominator"])
    if numerator % denominator:
        raise ValueError("nonintegral Hilbert-polynomial value")
    return numerator // denominator


def fixed_dictionary_parameters(q: int, sites: int, bond: int) -> int:
    return 2 * q * (2 * bond + (sites - 2) * bond**2)


def learned_dictionary_parameters(d: int, q: int, sites: int, bond: int) -> int:
    return fixed_dictionary_parameters(q, sites, bond) + 2 * d**2 * q


def generate_table_1() -> GeneratedTable:
    specification = architecture_specification()
    rows = []
    for record in specification["table_1_rows"]:
        label = str(record["label_tex"])
        geometry = str(record["geometry"])
        degree = int(record["target_degree"])
        d = int(record["source_dimension"])
        sites = int(record["site_count"])
        q = int(record["dictionary_size"])
        bond = int(record["bond_dimension"])
        dictionary = str(record["dictionary"])
        if geometry != "X11" and degree != sites:
            raise ValueError("registered table uses base degree one except X11")
        if geometry == "X11" and (d, sites, degree) != (45, 3, 6):
            raise ValueError("unexpected X11 source-degree registration")
        if dictionary not in {"fixed", "learned"}:
            raise ValueError(f"unknown dictionary type: {dictionary}")
        sections = h0_dimension(geometry, degree, specification)
        tn = (
            learned_dictionary_parameters(d, q, sites, bond)
            if dictionary == "learned"
            else fixed_dictionary_parameters(q, sites, bond)
        )
        hermitian = sections**2
        rows.append(
            (
                label,
                str(degree),
                f"{sections:,}",
                f"{tn:,}",
                f"{hermitian:,}",
                f"{hermitian / tn:,.1f}",
            )
        )
    tuple_rows = tuple(rows)
    content = table_content(
        [
            "% Generated deterministically from geometry and architecture data by",
            "% scripts/generate_gcicy_tn_final_tables.py.",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"type & target degree & \(h^0(L^k)\) & \(P_{\rm TN}\) & \(P_{\rm Herm}\) & \(P_{\rm Herm}/P_{\rm TN}\) \\",
            r"\midrule",
        ],
        tuple_rows,
        [r"\bottomrule", r"\end{tabular}"],
    )
    return GeneratedTable(
        "current_parameter_comparison_20260729.tex", content, tuple_rows, (6,)
    )


def generate_table_2() -> GeneratedTable:
    base = "outputs/pipeline/type11_x11_equal_time_final_20260807"
    model_paths = {
        "positive_TN": [f"{base}/replicate_{i}/tn_arrays.npz" for i in (1, 2, 3)],
        "phi": [
            f"{base}/replicate_{i}/source_density_phi/blind_tail_arrays.npz"
            for i in (1, 2, 3)
        ],
        "full_h6": [
            f"{base}/replicate_{i}/full_h6_equal_time_plateau_arrays.npz"
            for i in (1, 2, 3)
        ],
    }
    aggregates = {
        key: aggregate_metrics(pointwise_metrics(path) for path in paths)
        for key, paths in model_paths.items()
    }
    h2 = pointwise_metrics(f"{base}/h2_start_arrays.npz")
    phi_counts = {
        int(
            load_json(
                f"{base}/replicate_{i}/source_density_phi/report.json"
            )["network"]["parameter_count"]
        )
        for i in (1, 2, 3)
    }
    if len(phi_counts) != 1:
        raise ValueError("residual-phi parameter count differs between runs")
    phi_count = phi_counts.pop()

    def pm(key: str, metric: str, digits: int = 5) -> str:
        mean, sd = aggregates[key][metric]
        return f"${mean:.{digits}f}\\pm{sd:.{digits}f}$"

    main_rows = (
        (
            r"common $H_2$ start",
            f"{45**2:,}",
            f"${h2['sigma']:.5f}$",
            f"${h2['chi']:.5f}$",
            f"${h2['minimum_metric_eigenvalue']:.5f}$",
        ),
        (
            r"positive $k=6,D=5$ TN",
            f"{fixed_dictionary_parameters(45**2, 3, 5):,}",
            pm("positive_TN", "sigma"),
            pm("positive_TN", "chi"),
            pm("positive_TN", "minimum_metric_eigenvalue"),
        ),
        (
            r"neural $\phi$-model",
            f"{phi_count:,}",
            pm("phi", "sigma"),
            pm("phi", "chi"),
            pm("phi", "minimum_metric_eigenvalue"),
        ),
        (
            r"direct $H_6$, extended to plateau",
            f"{h0_dimension('X11', 6)**2:,}",
            pm("full_h6", "sigma"),
            pm("full_h6", "chi"),
            pm("full_h6", "minimum_metric_eigenvalue"),
        ),
    )
    tail_rows = (
        (
            r"positive $k=6,D=5$ TN",
            pm("positive_TN", "absolute_log_ratio_q999"),
            pm("positive_TN", "absolute_log_ratio_cvar_1pct"),
            pm("positive_TN", "maximum_absolute_log_ratio"),
        ),
        (
            r"neural $\phi$-model",
            pm("phi", "absolute_log_ratio_q999"),
            pm("phi", "absolute_log_ratio_cvar_1pct"),
            pm("phi", "maximum_absolute_log_ratio"),
        ),
        (
            r"direct $H_6$, extended to plateau",
            pm("full_h6", "absolute_log_ratio_q999"),
            pm("full_h6", "absolute_log_ratio_cvar_1pct"),
            pm("full_h6", "maximum_absolute_log_ratio"),
        ),
    )
    first = table_content(
        [
            "% Generated from frozen pointwise X11 arrays by",
            "% scripts/generate_gcicy_tn_final_tables.py.",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"model & trainable real parameters & $\sigma$ & $\chi$ & $\lambda_{\min}(g)$ \\",
            r"\midrule",
        ],
        main_rows,
        [r"\bottomrule", r"\end{tabular}", "}", ""],
    )
    second = table_content(
        [
            r"\begin{tabular}{@{}lrrr@{}}",
            r"\toprule",
            r"model & $Q_{0.999}(|\log r|)$ & $\operatorname{CVaR}_{1\%}(|\log r|)$ & $\max |\log r|$ \\",
            r"\midrule",
        ],
        tail_rows,
        [r"\bottomrule", r"\end{tabular}"],
    )
    return GeneratedTable(
        "x11_equal_time_final_20260807.tex",
        first + second,
        main_rows + tail_rows,
        (5, 4),
    )


def generate_table_3() -> GeneratedTable:
    base = "outputs/pipeline/type21_q121_d12_final_blind_20260729"
    records = (
        (r"full \(H_4\)", h0_dimension("X21", 4) ** 2, pointwise_metrics(f"{base}/h4_final_blind_arrays.npz")),
        (
            r"positive \(k=8,D=8\) TN",
            fixed_dictionary_parameters(11**2, 8, 8),
            pointwise_metrics(f"{base}/d8_final_blind_arrays.npz"),
        ),
    )
    rows = tuple(
        (
            label,
            f"{parameters:,}",
            f"{metrics['sigma']:.6f}",
            f"{metrics['chi']:.6f}",
            f"{metrics['absolute_log_ratio_q999']:.3f}",
            f"{metrics['absolute_log_ratio_cvar_1pct']:.3f}",
        )
        for label, parameters, metrics in records
    )
    content = table_content(
        [
            "% Generated from frozen pointwise X21 arrays by",
            "% scripts/generate_gcicy_tn_final_tables.py.",
            r"\begin{tabular}{lrcccc}",
            r"\toprule",
            r"model & \(P\) & \(\sigma\) & \(\chi\) & \(Q^{\rm abs}_{0.999}\) & \(\CVaR^{\rm abs}_{1\%}\) \\",
            r"\midrule",
        ],
        rows,
        [r"\bottomrule", r"\end{tabular}"],
    )
    return GeneratedTable(
        "x21_final_blind_compression_20260802.tex", content, rows, (6,)
    )


def generate_table_4() -> GeneratedTable:
    base = "outputs/pipeline/type21_fixed_d8_degree_scaling_floor1e14_20260807"
    rows = []
    for degree in (8, 12, 16, 20):
        metrics = aggregate_metrics(
            pointwise_metrics(
                f"{base}/k{degree}/replicate_{replicate}/final_common/arrays.npz"
            )
            for replicate in (1, 2, 3)
        )

        def pm(metric: str, digits: int) -> str:
            mean, sd = metrics[metric]
            return rf"\({mean:.{digits}f}\pm{sd:.{digits}f}\)"

        rows.append(
            (
                str(degree),
                f"{fixed_dictionary_parameters(11**2, degree, 8):,}",
                pm("sigma", 6),
                pm("chi", 6),
                pm("absolute_log_ratio_q999", 5),
                pm("absolute_log_ratio_cvar_1pct", 5),
            )
        )
    tuple_rows = tuple(rows)
    content = table_content(
        [
            "% Generated from 12 frozen pointwise X21 arrays by",
            "% scripts/generate_gcicy_tn_final_tables.py.",
            r"\small",
            r"\begin{tabularx}{\textwidth}{@{}rrXXXX@{}}",
            r"\toprule",
            r"\(k\) & trainable \(P\) & \(\sigma\) & \(\chi\) & \(Q^{\rm abs}_{0.999}\) & \(\CVaR^{\rm abs}_{1\%}\) \\",
            r"\midrule",
        ],
        tuple_rows,
        [r"\bottomrule", r"\end{tabularx}"],
    )
    return GeneratedTable(
        "x21_fixed_bond_degree_ladder.tex", content, tuple_rows, (6,)
    )


def generate_table_5() -> GeneratedTable:
    base = "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719"
    records = (
        (
            r"trainable \(q=22\)",
            learned_dictionary_parameters(17, 22, 4, 5),
            pointwise_metrics(f"{base}/active_q22_k4_control_blind83703_n65532_arrays.npz"),
        ),
        (
            r"trainable \(q=60\)",
            learned_dictionary_parameters(17, 60, 4, 5),
            pointwise_metrics(f"{base}/active_q60_k4_capacity_blind83703_n65532_arrays.npz"),
        ),
        (
            r"fixed complete \(q=289\)",
            fixed_dictionary_parameters(17**2, 4, 5),
            pointwise_metrics(
                "outputs/pipeline/type22_fixed_q289_capacity_20260806/"
                "q289_k4_capacity_blind83703_n65532_arrays.npz"
            ),
        ),
    )
    rows = tuple(
        (
            label,
            f"{parameters:,}",
            f"{metrics['sigma']:.5f}",
            f"{metrics['chi']:.5f}",
            f"{metrics['absolute_log_ratio_q999']:.5f}",
            f"{metrics['absolute_log_ratio_cvar_1pct']:.5f}",
        )
        for label, parameters, metrics in records
    )
    content = table_content(
        [
            "% Generated from frozen pointwise X22 arrays by",
            "% scripts/generate_gcicy_tn_final_tables.py.",
            r"\begin{tabular}{lrcccc}",
            r"\toprule",
            r"local dictionary & \(P\) & \(\sigma\) & \(\chi\) & \(Q_{0.999}(|\log r|)\) & \(\CVaR_{1\%}(|\log r|)\) \\",
            r"\midrule",
        ],
        rows,
        [r"\bottomrule", r"\end{tabular}"],
    )
    return GeneratedTable("x22_dictionary_comparison.tex", content, rows, (6,))


def generated_tables() -> tuple[GeneratedTable, ...]:
    return (
        generate_table_1(),
        generate_table_2(),
        generate_table_3(),
        generate_table_4(),
        generate_table_5(),
    )


def source_paths() -> tuple[Path, ...]:
    paths: list[Path] = [ROOT / "pipeline_specs/final_table_architectures.json"]
    x11 = "outputs/pipeline/type11_x11_equal_time_final_20260807"
    paths.append(ROOT / f"{x11}/h2_start_arrays.npz")
    for replicate in (1, 2, 3):
        paths.extend(
            [
                ROOT / f"{x11}/replicate_{replicate}/tn_arrays.npz",
                ROOT
                / f"{x11}/replicate_{replicate}/source_density_phi/blind_tail_arrays.npz",
                ROOT
                / f"{x11}/replicate_{replicate}/source_density_phi/report.json",
                ROOT
                / f"{x11}/replicate_{replicate}/full_h6_equal_time_plateau_arrays.npz",
            ]
        )
    x21_blind = "outputs/pipeline/type21_q121_d12_final_blind_20260729"
    paths.extend(
        [
            ROOT / f"{x21_blind}/h4_final_blind_arrays.npz",
            ROOT / f"{x21_blind}/d8_final_blind_arrays.npz",
        ]
    )
    ladder = "outputs/pipeline/type21_fixed_d8_degree_scaling_floor1e14_20260807"
    for degree in (8, 12, 16, 20):
        for replicate in (1, 2, 3):
            paths.append(
                ROOT
                / f"{ladder}/k{degree}/replicate_{replicate}/final_common/arrays.npz"
            )
    x22 = "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719"
    paths.extend(
        [
            ROOT / f"{x22}/active_q22_k4_control_blind83703_n65532_arrays.npz",
            ROOT / f"{x22}/active_q60_k4_capacity_blind83703_n65532_arrays.npz",
            ROOT
            / "outputs/pipeline/type22_fixed_q289_capacity_20260806/"
            "q289_k4_capacity_blind83703_n65532_arrays.npz",
        ]
    )
    return tuple(paths)


def generation_manifest(tables: tuple[GeneratedTable, ...]) -> dict:
    missing = [str(path) for path in source_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing final-table evidence: " + ", ".join(missing))
    return {
        "schema": "gcicy-tn-final-table-generation-v2",
        "generator": {
            "path": "scripts/generate_gcicy_tn_final_tables.py",
            "sha256": sha256(SCRIPT),
        },
        "statistics": {
            "ratio_normalization": "self-normalized weighted mean",
            "standard_deviation": "sample standard deviation (n-1)",
            "cvar": "exact worst one-percent weight mass with fractional boundary",
        },
        "sources": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_paths()
        },
        "tables": {
            table.filename: {
                "bytes": len(table.content.encode("utf-8")),
                "sha256": hashlib.sha256(table.content.encode("utf-8")).hexdigest(),
                "rows": len(table.rows),
                "column_counts": list(table.column_counts),
            }
            for table in tables
        },
    }


def default_output_dir() -> Path:
    archive_manuscript = ROOT / "manuscript/generated_tn"
    if (ROOT / "manuscript/gcicy_tn_paper.tex").is_file():
        return archive_manuscript
    return ROOT / "release_work/final_repro_20260815/manuscript/generated_tn"


def compare(output_dir: Path, tables: tuple[GeneratedTable, ...], manifest: dict) -> None:
    failures = []
    for table in tables:
        path = output_dir / table.filename
        if not path.is_file():
            failures.append(f"missing generated table: {path}")
        elif path.read_text(encoding="utf-8") != table.content:
            failures.append(f"generated table differs byte-for-byte: {path}")
    manifest_path = output_dir / MANIFEST_NAME
    expected_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if not manifest_path.is_file():
        failures.append(f"missing generation manifest: {manifest_path}")
    elif manifest_path.read_text(encoding="utf-8") != expected_manifest:
        failures.append(f"generation manifest differs byte-for-byte: {manifest_path}")
    if failures:
        raise SystemExit("\n".join(failures))


def write(output_dir: Path, tables: tuple[GeneratedTable, ...], manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table in tables:
        (output_dir / table.filename).write_text(table.content, encoding="utf-8")
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tables = generated_tables()
    manifest = generation_manifest(tables)
    output_dir = args.output_dir.expanduser().resolve()
    if args.check:
        compare(output_dir, tables, manifest)
        print(
            f"verified byte-exact regeneration of {len(tables)} tables from "
            f"{len(manifest['sources'])} frozen source files"
        )
    else:
        write(output_dir, tables, manifest)
        print(f"wrote {len(tables)} tables and {output_dir / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
