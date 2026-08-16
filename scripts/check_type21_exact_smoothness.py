#!/usr/bin/env python3
"""Check accepted and rejected type-(2,1) models chart by chart with Singular."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.adapters.p3p1p1p1_21 import (  # noqa: E402
    P3P1P1P1Type21Adapter,
)
from gcicy_metric.pipeline.adapters.p4p1p1_hirzebruch_21 import (  # noqa: E402
    P4P1P1HirzebruchType21Adapter,
)
from gcicy_metric.pipeline.adapters.p4p1_hirzebruch_11 import (  # noqa: E402
    P4P1HirzebruchM4Type11Adapter,
)
from gcicy_metric import type21_hirzebruch_x3 as hirzebruch_case  # noqa: E402
from gcicy_metric import type21_candidate_p5p1 as p5p1_case  # noqa: E402
from gcicy_metric import type21_candidate_p5p1_1223 as p5p1_1223_case  # noqa: E402
from gcicy_metric import type21_p3p1p1p1 as rejected_case  # noqa: E402
from gcicy_metric.pipeline import GCICYConfiguration  # noqa: E402


def parse_chart(text: str, sizes: tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != len(sizes) or any(
        not 0 <= value < size
        for value, size in zip(values, sizes, strict=True)
    ):
        ranges = ",".join(str(size) for size in sizes)
        raise argparse.ArgumentTypeError(f"chart must have ranges {ranges}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "hirzebruch_x3",
            "hirzebruch_m4",
            "rejected_hodge_9_13",
            "p5p1_1124",
            "p5p1_1223",
        ),
        default="hirzebruch_x3",
    )
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--charts", nargs="*")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--characteristic", type=int, default=32003)
    parser.add_argument("--algorithm", choices=("std", "slimgb"), default="slimgb")
    parser.add_argument("--singular", default="/opt/homebrew/bin/Singular")
    parser.add_argument(
        "--out",
        type=Path,
    )
    return parser.parse_args()


def polynomial_string(exponents, coefficients, chart, factor_offsets):
    selected = {
        offset + chart_index
        for offset, chart_index in zip(factor_offsets, chart, strict=True)
    }
    coordinate_count = int(np.asarray(exponents).shape[1])
    active = [index for index in range(coordinate_count) if index not in selected]
    variable_for_index = {
        homogeneous_index: f"u{column}"
        for column, homogeneous_index in enumerate(active)
    }
    terms = []
    for exponent, coefficient_value in zip(exponents, coefficients, strict=True):
        value = complex(coefficient_value)
        if abs(value.imag) > 1e-10:
            raise ValueError("exact smoothness coefficients must be real integers")
        coefficient = int(round(value.real))
        if coefficient == 0:
            continue
        factors = []
        for index, power_value in enumerate(exponent):
            power = int(power_value)
            if power == 0 or index in selected:
                continue
            variable = variable_for_index[index]
            factors.append(variable if power == 1 else f"{variable}^{power}")
        monomial = "*".join(factors) if factors else "1"
        magnitude = abs(coefficient)
        body = monomial if magnitude == 1 else f"{magnitude}*{monomial}"
        if not terms:
            terms.append(body if coefficient > 0 else f"-{body}")
        else:
            terms.append(("+" if coefficient > 0 else "-") + body)
    return "".join(terms) if terms else "0"


def singular_program(
    model,
    chart,
    characteristic,
    algorithm,
    polynomial_data,
    factor_offsets,
    affine_dimension,
):
    polynomials = [
        polynomial_string(exponents, coefficients, chart, factor_offsets)
        for exponents, coefficients in polynomial_data(model, chart)
    ]
    variables = ",".join(f"u{index}" for index in range(affine_dimension))
    return "\n".join(
        [
            "option(redSB);",
            f"ring r={characteristic},({variables}),dp;",
            *(f"poly f{index + 1}={polynomial};" for index, polynomial in enumerate(polynomials)),
            "ideal IM=f1,f2;",
            f"ideal GM={algorithm}(IM);",
            'print("INTERMEDIATE_DIMENSION");',
            "print(dim(GM));",
            "matrix JM=jacob(IM);",
            "ideal MM=minor(JM,2);",
            "ideal SM=IM+MM;",
            f"ideal GSM={algorithm}(SM);",
            'if (reduce(1,GSM)==0) { print("INTERMEDIATE_SMOOTH"); } else { print("INTERMEDIATE_SINGULAR"); }',
            "ideal IX=f1,f2,f3;",
            f"ideal GX={algorithm}(IX);",
            'print("GCICY_DIMENSION");',
            "print(dim(GX));",
            "matrix JX=jacob(IX);",
            "ideal MX=minor(JX,3);",
            "ideal SX=IX+MX;",
            f"ideal GSX={algorithm}(SX);",
            'if (reduce(1,GSX)==0) { print("GCICY_SMOOTH"); } else { print("GCICY_SINGULAR_OR_UNRESOLVED"); print(dim(GSX)); }',
            "quit;",
        ]
    )


def _marker_integer(stdout: str, marker: str) -> int | None:
    lines = [line.strip() for line in stdout.splitlines()]
    if marker not in lines:
        return None
    index = lines.index(marker)
    if index + 1 >= len(lines):
        return None
    try:
        return int(lines[index + 1])
    except ValueError:
        return None


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def check_chart(
    singular,
    model,
    chart,
    timeout,
    characteristic,
    algorithm,
    polynomial_data,
    factor_offsets,
    affine_dimension,
):
    program = singular_program(
        model,
        chart,
        characteristic,
        algorithm,
        polynomial_data,
        factor_offsets,
        affine_dimension,
    )
    started = time.monotonic()
    try:
        result = subprocess.run(
            [singular, "-q"],
            input=program,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout = result.stdout
        return {
            "chart": list(chart),
            "intermediate_smooth": (
                result.returncode == 0
                and "INTERMEDIATE_SMOOTH" in stdout
                and "INTERMEDIATE_SINGULAR" not in stdout
            ),
            "gcicy_smooth": (
                result.returncode == 0
                and "GCICY_SMOOTH" in stdout
                and "GCICY_SINGULAR_OR_UNRESOLVED" not in stdout
            ),
            "timed_out": False,
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "intermediate_dimension": _marker_integer(stdout, "INTERMEDIATE_DIMENSION"),
            "gcicy_dimension": _marker_integer(stdout, "GCICY_DIMENSION"),
            "stdout": stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "chart": list(chart),
            "intermediate_smooth": False,
            "gcicy_smooth": False,
            "timed_out": True,
            "returncode": None,
            "elapsed_seconds": time.monotonic() - started,
            "intermediate_dimension": None,
            "gcicy_dimension": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
        }


def main() -> None:
    args = parse_args()
    singular = Path(args.singular)
    if not singular.exists():
        raise SystemExit(f"Singular executable not found: {singular}")
    if args.case == "hirzebruch_x3":
        adapter = P4P1P1HirzebruchType21Adapter()
        model_seed = 20260731 if args.model_seed is None else args.model_seed
        polynomial_data = hirzebruch_case.local_polynomial_data
        sizes = hirzebruch_case.FACTOR_SIZES
        offsets = hirzebruch_case.FACTOR_OFFSETS
        model = adapter.make_model(model_seed, exact=True)
        case_key = adapter.key
        configuration = adapter.configuration.to_dict()
        model_metadata = adapter.model_metadata(model)
        all_charts = list(adapter.projective_charts())
    elif args.case == "hirzebruch_m4":
        adapter = P4P1HirzebruchM4Type11Adapter()
        model_seed = 20260831 if args.model_seed is None else args.model_seed
        polynomial_data = hirzebruch_case.local_polynomial_data
        sizes = hirzebruch_case.FACTOR_SIZES
        offsets = hirzebruch_case.FACTOR_OFFSETS
        model = adapter.make_model(model_seed, exact=True)
        case_key = adapter.key
        configuration = adapter.configuration.to_dict()
        model_metadata = adapter.model_metadata(model)
        all_charts = list(product(*(range(size) for size in sizes)))
    elif args.case == "rejected_hodge_9_13":
        adapter = P3P1P1P1Type21Adapter()
        model_seed = 20260721 if args.model_seed is None else args.model_seed
        polynomial_data = rejected_case.local_polynomial_data
        sizes = rejected_case.FACTOR_SIZES
        offsets = rejected_case.FACTOR_OFFSETS
        model = adapter.make_model(model_seed, exact=True)
        case_key = adapter.key
        configuration = adapter.configuration.to_dict()
        model_metadata = adapter.model_metadata(model)
        all_charts = list(adapter.projective_charts())
    elif args.case == "p5p1_1124":
        model_seed = 20260801 if args.model_seed is None else args.model_seed
        polynomial_data = p5p1_case.local_polynomial_data
        sizes = p5p1_case.FACTOR_SIZES
        offsets = p5p1_case.FACTOR_OFFSETS
        model = p5p1_case.make_p5p1_type21_candidate_model(model_seed)
        candidate_configuration = GCICYConfiguration(
            key="p5p1_type21_candidate_1124",
            name="P5 x P1 non-stabilized type-(2,1) candidate",
            ambient_dimensions=(5, 1),
            positive_columns=((1, 1), (1, 2)),
            generalized_columns=((4, -1),),
            kahler_line_bundle=(1, 1),
            source="arXiv:1507.03235 and arXiv:2209.10157 P5 x P1 scan class",
        )
        case_key = candidate_configuration.key
        configuration = candidate_configuration.to_dict()
        model_metadata = {
            "seed": int(model.seed),
            "p1_term_count": int(len(model.p1_coefficients)),
            "p2_term_count": int(len(model.p2_coefficients)),
            "generalized_section_parameter_count_before_koszul_quotient": int(
                model.q_cubic_coefficients.size
            ),
            "generalized_section_space_dimension": 105,
            "candidate_status": "screening_only_not_registered",
        }
        all_charts = list(product(*(range(size) for size in sizes)))
    else:
        model_seed = 20260802 if args.model_seed is None else args.model_seed
        polynomial_data = p5p1_1223_case.local_polynomial_data
        sizes = p5p1_1223_case.FACTOR_SIZES
        offsets = p5p1_1223_case.FACTOR_OFFSETS
        model = p5p1_1223_case.make_p5p1_type21_candidate_1223_model(model_seed)
        candidate_configuration = GCICYConfiguration(
            key="p5p1_type21_candidate_1223",
            name="P5 x P1 K3-fibered non-stabilized type-(2,1) candidate",
            ambient_dimensions=(5, 1),
            positive_columns=((1, 1), (2, 2)),
            generalized_columns=((3, -1),),
            kahler_line_bundle=(1, 1),
            source="arXiv:1507.03235 and arXiv:2209.10157 P5 x P1 scan class",
        )
        case_key = candidate_configuration.key
        configuration = candidate_configuration.to_dict()
        model_metadata = {
            "seed": int(model.seed),
            "p1_term_count": int(len(model.p1_coefficients)),
            "p2_term_count": int(len(model.p2_coefficients)),
            "generalized_section_parameter_count_before_koszul_quotient": int(
                len(model.r_a_coefficients) + model.r_b_coefficients.size
            ),
            "generalized_section_space_dimension": 30,
            "expected_euler_characteristic": -100,
            "expected_favorable_hodge_numbers": [2, 52],
            "generic_fiber": "complete-intersection K3 of degrees (2,3) in P4",
            "candidate_status": "screening_only_not_registered",
        }
        all_charts = list(product(*(range(size) for size in sizes)))
    charts = (
        [parse_chart(value, sizes) for value in args.charts]
        if args.charts
        else all_charts
    )
    affine_dimension = sum(size - 1 for size in sizes)
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(
                check_chart,
                str(singular),
                model,
                chart,
                args.timeout,
                args.characteristic,
                args.algorithm,
                polynomial_data,
                offsets,
                affine_dimension,
            ): chart
            for chart in charts
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"chart={tuple(row['chart'])} M={row['intermediate_smooth']} "
                f"X={row['gcicy_smooth']} dim=({row['intermediate_dimension']},"
                f"{row['gcicy_dimension']}) seconds={row['elapsed_seconds']:.2f}"
            )
    rows.sort(key=lambda row: row["chart"])
    expected_dimensions = all(
        row["intermediate_dimension"] in (-1, 4)
        and row["gcicy_dimension"] in (-1, 3)
        for row in rows
    )
    requested_chart_set = {tuple(row["chart"]) for row in rows}
    complete_chart_set = {tuple(chart) for chart in all_charts}
    complete_projective_atlas = (
        requested_chart_set == complete_chart_set
        and len(rows) == len(complete_chart_set)
    )
    all_requested_charts_proved_smooth = bool(
        all(row["intermediate_smooth"] and row["gcicy_smooth"] for row in rows)
        and expected_dimensions
        and any(row["gcicy_dimension"] == 3 for row in rows)
    )
    complete_atlas_proved_smooth = bool(
        complete_projective_atlas and all_requested_charts_proved_smooth
    )
    good_reduction_certificate = bool(
        _is_prime(args.characteristic) and complete_atlas_proved_smooth
    )
    summary = {
        "description": (
            "Complete finite-field atlas check for the intermediate complete intersection "
            "and its sequential generalized hypersurface."
        ),
        "adapter": None if args.case.startswith("p5p1_") else case_key,
        "candidate_key": case_key,
        "case": args.case,
        "certificate_presentation": (
            "stabilized_type_21_for_direct_type_11"
            if args.case == "hirzebruch_m4"
            else "configured_presentation"
        ),
        "configuration": configuration,
        "model": model_metadata,
        "characteristic": args.characteristic,
        "algorithm": args.algorithm,
        "charts_requested": len(charts),
        "complete_projective_atlas": complete_projective_atlas,
        "intermediate_charts_proved_smooth": sum(
            row["intermediate_smooth"] for row in rows
        ),
        "gcicy_charts_proved_smooth": sum(row["gcicy_smooth"] for row in rows),
        "charts_timed_out": sum(row["timed_out"] for row in rows),
        "expected_dimensions": expected_dimensions,
        "nonempty_intermediate_charts": sum(
            row["intermediate_dimension"] == 4 for row in rows
        ),
        "nonempty_gcicy_charts": sum(row["gcicy_dimension"] == 3 for row in rows),
        "all_requested_charts_proved_smooth": all_requested_charts_proved_smooth,
        "all_proved_smooth": all_requested_charts_proved_smooth,
        "complete_atlas_proved_smooth": complete_atlas_proved_smooth,
        "characteristic_zero_good_reduction_certificate": {
            "applicable": good_reduction_certificate,
            "conclusion": (
                "The fixed integral model has a nonempty smooth generic fibre over "
                "Q, hence a smooth base change over C."
                if good_reduction_certificate
                else None
            ),
            "logical_requirements": {
                "positive_prime_characteristic": _is_prime(args.characteristic),
                "integer_coefficient_model": True,
                "integral_local_section_gluing": True,
                "complete_standard_projective_atlas": complete_projective_atlas,
                "empty_singular_locus_on_every_chart": complete_atlas_proved_smooth,
                "expected_nonempty_dimensions": expected_dimensions
                and any(row["gcicy_dimension"] == 3 for row in rows),
                "projective_family": True,
            },
            "proof": (
                "docs/hirzebruch_m4_out_of_sample.md"
                if args.case == "hirzebruch_m4"
                else "docs/type21_good_reduction_smoothness.md"
            ),
        },
        "rows": rows,
    }
    default_name = (
        f"{case_key}_smoothness_p{args.characteristic}.json"
    )
    output = (
        args.out.expanduser().resolve()
        if args.out is not None
        else (ROOT / "outputs" / "pipeline" / default_name).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(
        f"proved smooth: M={summary['intermediate_charts_proved_smooth']}/"
        f"{len(charts)}, X={summary['gcicy_charts_proved_smooth']}/{len(charts)}"
    )
    if not summary["all_proved_smooth"]:
        raise SystemExit("the exact type-(2,1) model did not pass every smoothness gate")


if __name__ == "__main__":
    main()
