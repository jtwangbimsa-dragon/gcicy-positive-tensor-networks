#!/usr/bin/env python3
"""Use Singular to certify smoothness of the exact bicubic control model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.bicubic_model import make_exact_bicubic_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=20260721)
    parser.add_argument("--characteristic", type=int, default=32003)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--singular", default="/opt/homebrew/bin/Singular")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "bicubic_exact_smoothness_check_p32003.json",
    )
    return parser.parse_args()


def polynomial_string(model, chart):
    selected = {chart[0], 3 + chart[1]}
    active = [index for index in range(6) if index not in selected]
    variable = {index: f"u{column}" for column, index in enumerate(active)}
    terms = []
    for exponent, coefficient_value in zip(model.exponents, model.coefficients, strict=True):
        coefficient = int(round(complex(coefficient_value).real))
        factors = []
        for index, power_value in enumerate(exponent):
            power = int(power_value)
            if power == 0 or index in selected:
                continue
            factors.append(variable[index] if power == 1 else f"{variable[index]}^{power}")
        monomial = "*".join(factors) if factors else "1"
        magnitude = abs(coefficient)
        body = monomial if magnitude == 1 else f"{magnitude}*{monomial}"
        if not terms:
            terms.append(body if coefficient > 0 else f"-{body}")
        else:
            terms.append(("+" if coefficient > 0 else "-") + body)
    return "".join(terms)


def singular_program(model, chart, characteristic):
    polynomial = polynomial_string(model, chart)
    return "\n".join(
        [
            "option(redSB);",
            f"ring r={characteristic},(u0,u1,u2,u3),dp;",
            f"poly f={polynomial};",
            "ideal S=f,diff(f,u0),diff(f,u1),diff(f,u2),diff(f,u3);",
            "ideal G=slimgb(S);",
            'if (reduce(1,G)==0) { print("SMOOTH"); } else { print("SINGULAR_OR_UNRESOLVED"); print(dim(G)); }',
            "quit;",
        ]
    )


def check_chart(singular, model, chart, characteristic, timeout):
    started = time.monotonic()
    try:
        result = subprocess.run(
            [singular, "-q"],
            input=singular_program(model, chart, characteristic),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        smooth = result.returncode == 0 and "SMOOTH" in result.stdout and "SINGULAR_OR_UNRESOLVED" not in result.stdout
        return {
            "chart": list(chart),
            "smooth": bool(smooth),
            "timed_out": False,
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
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
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
        }


def main() -> None:
    args = parse_args()
    if not Path(args.singular).exists():
        raise SystemExit(f"Singular executable not found: {args.singular}")
    model = make_exact_bicubic_model(args.model_seed)
    charts = [(x_index, y_index) for x_index in range(3) for y_index in range(3)]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(
                check_chart,
                args.singular,
                model,
                chart,
                args.characteristic,
                args.timeout,
            ): chart
            for chart in charts
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"chart={tuple(row['chart'])} smooth={row['smooth']} "
                f"timeout={row['timed_out']} seconds={row['elapsed_seconds']:.2f}",
                flush=True,
            )
    rows.sort(key=lambda row: row["chart"])
    summary = {
        "description": "Exact finite-field smoothness certificate for the ordinary bicubic control.",
        "model_seed": model.seed,
        "characteristic": args.characteristic,
        "charts_requested": len(charts),
        "charts_proved_smooth": sum(row["smooth"] for row in rows),
        "charts_timed_out": sum(row["timed_out"] for row in rows),
        "all_proved_smooth": all(row["smooth"] for row in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if not summary["all_proved_smooth"]:
        raise SystemExit("not every bicubic chart was proved smooth")


if __name__ == "__main__":
    main()
