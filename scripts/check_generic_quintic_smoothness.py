#!/usr/bin/env python3
"""Certify the fixed generic quintic as smooth in all projective charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.generic_quintic import (  # noqa: E402
    GENERIC_QUINTIC_COEFFICIENTS,
    GENERIC_QUINTIC_EXPONENTS,
    experiment_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--singular", default="/opt/homebrew/bin/Singular")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "generic_quintic_kill_test_20260726"
        / "smoothness_report.json",
    )
    return parser.parse_args()


def monomial_string(exponents: np.ndarray, coefficient: int) -> str:
    factors = []
    for coordinate, power in enumerate(exponents):
        if power:
            variable = f"u{coordinate}"
            factors.append(variable if power == 1 else f"{variable}^{int(power)}")
    body = "*".join(factors) if factors else "1"
    magnitude = abs(int(coefficient))
    if magnitude != 1:
        body = f"{magnitude}*{body}"
    return body if coefficient > 0 else f"-{body}"


def affine_polynomial(chart: int) -> str:
    terms = []
    for exponents, coefficient in zip(
        GENERIC_QUINTIC_EXPONENTS,
        GENERIC_QUINTIC_COEFFICIENTS,
        strict=True,
    ):
        affine_exponents = np.delete(exponents, chart)
        term = monomial_string(affine_exponents, int(coefficient))
        if terms and not term.startswith("-"):
            term = "+" + term
        terms.append(term)
    return "".join(terms)


def singular_program(chart: int) -> str:
    polynomial = affine_polynomial(chart)
    return "\n".join(
        [
            "option(redSB);",
            "ring r=0,(u0,u1,u2,u3),dp;",
            f"poly f={polynomial};",
            "ideal X=f;",
            "ideal GX=slimgb(X);",
            'print("VARIETY_DIMENSION");',
            "print(dim(GX));",
            "ideal S=f,diff(f,u0),diff(f,u1),diff(f,u2),diff(f,u3);",
            "ideal GS=slimgb(S);",
            'if (reduce(1,GS)==0) { print("SMOOTH"); }'
            ' else { print("SINGULAR"); print(dim(GS)); }',
            "quit;",
        ]
    )


def check_chart(
    singular: str,
    chart: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [singular, "-q"],
            input=singular_program(chart),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "chart": chart,
            "smooth": False,
            "timed_out": True,
            "returncode": None,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
        }
    lines = [line.strip() for line in result.stdout.splitlines()]
    dimension = None
    if "VARIETY_DIMENSION" in lines:
        index = lines.index("VARIETY_DIMENSION")
        if index + 1 < len(lines):
            try:
                dimension = int(lines[index + 1])
            except ValueError:
                pass
    return {
        "chart": chart,
        "smooth": (
            result.returncode == 0
            and "SMOOTH" in lines
            and "SINGULAR" not in lines
        ),
        "timed_out": False,
        "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "variety_dimension": dimension,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    args = parse_args()
    singular = Path(args.singular).expanduser().resolve()
    if not singular.exists():
        raise FileNotFoundError(singular)
    rows = []
    for chart in range(5):
        row = check_chart(str(singular), chart, args.timeout)
        rows.append(row)
        print(
            f"chart={chart} smooth={row['smooth']} "
            f"dimension={row.get('variety_dimension')} "
            f"seconds={row['elapsed_seconds']:.3f}",
            flush=True,
        )
    manifest = experiment_manifest()
    report = {
        "schema": "generic-quintic-exact-smoothness-v1",
        "geometry": manifest,
        "charts_requested": 5,
        "charts_proved_smooth": sum(row["smooth"] for row in rows),
        "all_proved_smooth": all(row["smooth"] for row in rows),
        "all_affine_dimensions_three": all(
            row.get("variety_dimension") == 3 for row in rows
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    if not report["all_proved_smooth"]:
        raise SystemExit("at least one projective chart was not proved smooth")
    if not report["all_affine_dimensions_three"]:
        raise SystemExit("the affine hypersurface dimension check failed")
    if len(manifest["support_preserving_coordinate_permutations"]) != 1:
        raise SystemExit("monomial support has an unexpected coordinate symmetry")
    if len(manifest["projective_fifth_root_phase_symmetries"]) != 1:
        raise SystemExit("monomial support has an unexpected fifth-root phase symmetry")


if __name__ == "__main__":
    main()

