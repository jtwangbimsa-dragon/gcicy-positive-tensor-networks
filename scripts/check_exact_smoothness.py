#!/usr/bin/env python3
"""Use Singular to test exact smoothness chart by chart for the integer model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import all_projective_charts, make_exact_generic_model  # noqa: E402


def parse_chart(text: str) -> tuple[int, int, int]:
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 3 or not (0 <= values[0] < 2 and 0 <= values[1] < 2 and 0 <= values[2] < 6):
        raise argparse.ArgumentTypeError("chart must have the form x,y,z with ranges 2,2,6")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=20260711)
    parser.add_argument("--charts", nargs="*", type=parse_chart)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--characteristic", type=int, default=0)
    parser.add_argument("--algorithm", choices=("std", "slimgb"), default="slimgb")
    parser.add_argument("--singular", default="/opt/homebrew/bin/Singular")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_exact_smoothness_check.json")
    return parser.parse_args()


def q_data(model, chart, q_index):
    exponents = []
    coefficients = []
    if q_index == 1:
        tensor_index = 1 if chart[0] == 0 else 0
        sign = 1 if chart[0] == 0 else -1
        for y_index in range(2):
            for z_index in range(6):
                exponent = [0] * 10
                exponent[2 + y_index] = 1
                exponent[4 + z_index] = 1
                exponents.append(exponent)
                coefficients.append(sign * int(round(model.p2_tensor[tensor_index, y_index, z_index].real)))
    else:
        tensor_index = 1 if chart[1] == 0 else 0
        sign = 1 if chart[1] == 0 else -1
        for x_index in range(2):
            for z_index in range(6):
                exponent = [0] * 10
                exponent[x_index] = 1
                exponent[4 + z_index] = 1
                exponents.append(exponent)
                coefficients.append(sign * int(round(model.p2_tensor[x_index, tensor_index, z_index].real)))
    return np.asarray(exponents, dtype=np.int64), np.asarray(coefficients, dtype=np.int64)


def polynomial_string(exponents, coefficients, chart):
    selected = {chart[0], 2 + chart[1], 4 + chart[2]}
    active = [index for index in range(10) if index not in selected]
    variable_for_index = {index: f"u{column}" for column, index in enumerate(active)}
    terms = []
    for exponent, coefficient_value in zip(exponents, coefficients, strict=True):
        coefficient = int(round(complex(coefficient_value).real))
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


def singular_program(model, chart, characteristic, algorithm):
    q1_exponents, q1_coefficients = q_data(model, chart, 1)
    q2_exponents, q2_coefficients = q_data(model, chart, 2)
    polynomials = [
        polynomial_string(model.p1_exponents, model.p1_coefficients, chart),
        polynomial_string(model.p2_exponents, model.p2_coefficients, chart),
        polynomial_string(q1_exponents, q1_coefficients, chart),
        polynomial_string(q2_exponents, q2_coefficients, chart),
    ]
    return "\n".join(
        [
            "option(redSB);",
            f"ring r={characteristic},(u0,u1,u2,u3,u4,u5,u6),dp;",
            *(f"poly f{index + 1}={polynomial};" for index, polynomial in enumerate(polynomials)),
            "ideal I=f1,f2,f3,f4;",
            f"ideal GI={algorithm}(I);",
            'print("VARIETY_DIMENSION");',
            "print(dim(GI));",
            "matrix J=jacob(I);",
            "ideal M=minor(J,4);",
            "ideal S=I+M;",
            f"ideal G={algorithm}(S);",
            'if (reduce(1,G)==0) { print("SMOOTH"); } else { print("SINGULAR_OR_UNRESOLVED"); print(dim(G)); }',
            "quit;",
        ]
    )


def check_chart(singular, model, chart, timeout, characteristic, algorithm):
    program = singular_program(model, chart, characteristic, algorithm)
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
        elapsed = time.monotonic() - started
        smooth = result.returncode == 0 and "SMOOTH" in result.stdout and "SINGULAR_OR_UNRESOLVED" not in result.stdout
        stdout_lines = [line.strip() for line in result.stdout.splitlines()]
        dimension = None
        if "VARIETY_DIMENSION" in stdout_lines:
            marker = stdout_lines.index("VARIETY_DIMENSION")
            if marker + 1 < len(stdout_lines):
                try:
                    dimension = int(stdout_lines[marker + 1])
                except ValueError:
                    dimension = None
        return {
            "chart": list(chart),
            "smooth": bool(smooth),
            "timed_out": False,
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "variety_dimension": dimension,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "chart": list(chart),
            "smooth": False,
            "timed_out": True,
            "returncode": None,
            "elapsed_seconds": time.monotonic() - started,
            "variety_dimension": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
        }


def main() -> None:
    args = parse_args()
    singular = Path(args.singular)
    if not singular.exists():
        raise SystemExit(f"Singular executable not found: {singular}")
    model = make_exact_generic_model(args.model_seed)
    charts = args.charts if args.charts else all_projective_charts()
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
            ): chart
            for chart in charts
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"chart={tuple(row['chart'])} smooth={row['smooth']} timeout={row['timed_out']} "
                f"dimension={row['variety_dimension']} seconds={row['elapsed_seconds']:.2f}"
            )
    rows.sort(key=lambda row: row["chart"])
    summary = {
        "description": "Exact Singular check that each affine-chart singular ideal contains 1.",
        "model_seed": args.model_seed,
        "characteristic": args.characteristic,
        "algorithm": args.algorithm,
        "charts_requested": len(charts),
        "charts_proved_smooth": sum(row["smooth"] for row in rows),
        "charts_timed_out": sum(row["timed_out"] for row in rows),
        "nonempty_dimension_three_charts": sum(row["variety_dimension"] == 3 for row in rows),
        "unexpected_variety_dimensions": sorted(
            {row["variety_dimension"] for row in rows if row["variety_dimension"] not in (None, -1, 3)}
        ),
        "all_proved_smooth": all(row["smooth"] for row in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"proved smooth: {summary['charts_proved_smooth']} / {summary['charts_requested']}")
    print(f"nonempty dimension-three charts: {summary['nonempty_dimension_three_charts']}")
    if not summary["all_proved_smooth"]:
        raise SystemExit("not every requested chart was proved smooth")
    if summary["nonempty_dimension_three_charts"] == 0 or summary["unexpected_variety_dimensions"]:
        raise SystemExit("exact dimension check did not certify a nonempty pure threefold")


if __name__ == "__main__":
    main()
