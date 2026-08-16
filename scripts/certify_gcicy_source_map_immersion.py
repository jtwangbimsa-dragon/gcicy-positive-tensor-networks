#!/usr/bin/env python3
"""Bind the three headline source maps to ambient very-ample embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from certify_section_restriction_ranks import (
    cases as restriction_rank_cases,
    integer_coefficients,
)


EXPECTED = {
    "X11_m3": {
        "reported_name": "X_{11}",
        "degree": [2, 2, 0],
        "effective_ambient": "P4 x P1",
        "effective_degree": [2, 2],
        "ambient_line_bundle": "O_{P4 x P1}(2,2)",
        "source_dimension": 45,
        "special_reduction": (
            "The second positive equation fixes the auxiliary P1 at "
            "[1:-1], so projection to P4 x P1 is an isomorphism onto the "
            "direct Hirzebruch presentation."
        ),
    },
    "X21": {
        "reported_name": "X_{21}",
        "degree": [1, 1],
        "effective_ambient": "P5 x P1",
        "effective_degree": [1, 1],
        "ambient_line_bundle": "O_{P5 x P1}(1,1)",
        "source_dimension": 11,
        "special_reduction": None,
    },
    "X22": {
        "reported_name": "X_{22}",
        "degree": [1, 1, 1],
        "effective_ambient": "P1 x P1 x P5",
        "effective_degree": [1, 1, 1],
        "ambient_line_bundle": "O_{P1 x P1 x P5}(1,1,1)",
        "source_dimension": 17,
        "special_reduction": None,
    },
}


def affine_polynomial(
    exponents: np.ndarray,
    coefficients: np.ndarray,
    *,
    chart: tuple[int, ...],
    ambient_dimensions: tuple[int, ...],
    prime: int,
) -> str:
    selected_coordinates = []
    offset = 0
    for dimension, selected in zip(ambient_dimensions, chart, strict=True):
        selected_coordinates.append(offset + selected)
        offset += dimension + 1
    free_coordinates = [
        index for index in range(offset) if index not in selected_coordinates
    ]

    combined: dict[tuple[int, ...], int] = {}
    for powers, coefficient in zip(
        np.asarray(exponents, dtype=np.int64),
        integer_coefficients(coefficients),
        strict=True,
    ):
        affine_powers = tuple(int(powers[index]) for index in free_coordinates)
        combined[affine_powers] = (
            combined.get(affine_powers, 0) + int(coefficient)
        ) % prime

    terms = []
    for powers, coefficient in sorted(combined.items()):
        coefficient %= prime
        if coefficient == 0:
            continue
        factors = [str(coefficient)]
        for index, power in enumerate(powers, start=1):
            if power == 1:
                factors.append(f"u{index}")
            elif power > 1:
                factors.append(f"u{index}^{power}")
        terms.append("*".join(factors))
    return "+".join(terms) if terms else "0"


def singular_affine_dimension(
    *,
    polynomials: list[tuple[np.ndarray, np.ndarray]],
    chart: tuple[int, ...],
    ambient_dimensions: tuple[int, ...],
    prime: int,
    singular: str,
) -> int:
    variable_count = sum(ambient_dimensions)
    variables = ",".join(f"u{index}" for index in range(1, variable_count + 1))
    equations = [
        affine_polynomial(
            exponents,
            coefficients,
            chart=chart,
            ambient_dimensions=ambient_dimensions,
            prime=prime,
        )
        for exponents, coefficients in polynomials
    ]
    source = "\n".join(
        [
            f"ring r={prime},({variables}),dp;",
            f"ideal I={','.join(equations)};",
            "ideal G=std(I);",
            'print("CODEX_DIM_BEGIN");',
            "print(dim(G));",
            'print("CODEX_DIM_END");',
            "quit;",
            "",
        ]
    )
    completed = subprocess.run(
        [singular, "-q"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    match = re.search(
        r"CODEX_DIM_BEGIN\s*[\r\n]+(-?\d+)\s*[\r\n]+CODEX_DIM_END",
        completed.stdout,
    )
    if match is None:
        raise SystemExit(
            "could not parse Singular dimension output:\n" + completed.stdout
        )
    return int(match.group(1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank-certificates",
        type=Path,
        default=Path("outputs/section_restriction_rank_certificates.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/pipeline/gcicy_source_map_immersion_20260730/certificate.json"
        ),
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=Path("gcicy paper/generated_tn/source_immersion_20260730.tex"),
    )
    parser.add_argument("--singular", default="Singular")
    args = parser.parse_args()

    rank_path = args.rank_certificates.expanduser().resolve()
    rank_payload = json.loads(rank_path.read_text())
    by_geometry = {
        item["geometry"]: item for item in rank_payload["certificates"]
    }
    exact_cases = {
        item.key: item
        for item in restriction_rank_cases()
        if item.key in EXPECTED
    }
    certificates = []
    for geometry, expected in EXPECTED.items():
        source = by_geometry[geometry]
        degree_record = next(
            item
            for item in source["degrees"]
            if item["degree"] == expected["degree"]
        )
        if degree_record["finite_field_rank"] != expected["source_dimension"]:
            raise SystemExit(f"{geometry}: finite-field source rank mismatch")
        if degree_record["riemann_roch_target"] != expected["source_dimension"]:
            raise SystemExit(f"{geometry}: Riemann--Roch target mismatch")
        if degree_record["pivot_minor_determinant_mod_prime"] == 0:
            raise SystemExit(f"{geometry}: stored rank minor is zero")
        if (
            degree_record.get("reported_artifact_finite_field_rank")
            != expected["source_dimension"]
        ):
            raise SystemExit(f"{geometry}: implemented source-vector rank mismatch")
        if (
            degree_record.get(
                "reported_artifact_pivot_minor_determinant_mod_prime"
            )
            == 0
        ):
            raise SystemExit(f"{geometry}: implemented source-vector minor is zero")
        if any(value <= 0 for value in expected["effective_degree"]):
            raise SystemExit(f"{geometry}: effective ambient degree is not positive")

        exact_case = exact_cases[geometry]
        local_polynomials, _ = exact_case.polynomial_factory()
        affine_dimension = singular_affine_dimension(
            polynomials=local_polynomials,
            chart=exact_case.chart,
            ambient_dimensions=exact_case.ambient_dimensions,
            prime=source["prime"],
            singular=args.singular,
        )
        expected_affine_dimension = (
            sum(exact_case.ambient_dimensions) - len(local_polynomials)
        )
        if affine_dimension != expected_affine_dimension:
            raise SystemExit(
                f"{geometry}: special-fibre dimension {affine_dimension} "
                f"does not match expected {expected_affine_dimension}"
            )

        certificates.append(
            {
                "geometry": geometry,
                **expected,
                "restriction_rank": degree_record["finite_field_rank"],
                "riemann_roch_dimension": degree_record["riemann_roch_target"],
                "rank_prime": source["prime"],
                "implemented_section_artifact": degree_record[
                    "reported_artifact"
                ],
                "implemented_section_artifact_sha256": degree_record[
                    "reported_artifact_sha256"
                ],
                "implemented_section_count": degree_record[
                    "reported_artifact_section_count"
                ],
                "implemented_section_rank_mod_prime": degree_record[
                    "reported_artifact_finite_field_rank"
                ],
                "implemented_nonzero_minor_mod_prime": degree_record[
                    "reported_artifact_pivot_minor_determinant_mod_prime"
                ],
                "ambient_affine_dimension": sum(exact_case.ambient_dimensions),
                "local_equation_count": len(local_polynomials),
                "special_fibre_affine_dimension": affine_dimension,
                "expected_affine_dimension": expected_affine_dimension,
                "regular_sequence_mod_prime": True,
                "integral_coefficients": True,
                "characteristic_zero_bridge": (
                    "The exact model and ambient monomial sections are defined "
                    "over Z. At the recorded prime, the affine equations have "
                    "the expected codimension and hence form a regular sequence. "
                    "The resulting Z_(p)-algebra is p-torsion-free. A nontrivial "
                    "Q-linear relation among the implemented sections could "
                    "therefore be scaled to a relation over Z_(p) with a unit "
                    "coefficient and reduced modulo p, contradicting the stored "
                    "nonzero evaluation minor."
                ),
                "h0_justification": (
                    "The polarization is ample and K_X is trivial, so Kodaira "
                    "vanishing gives h^i(X,L)=0 for i>0; threefold "
                    "Riemann--Roch therefore identifies h^0 with the exact "
                    "displayed Euler characteristic."
                ),
                "very_ample_reason": (
                    "An external tensor product O(a_1,...,a_r) with every "
                    "a_i>0 is very ample on a product of projective spaces."
                ),
                "restriction_reason": (
                    "Restricting a closed immersion to a closed subvariety "
                    "is a closed immersion. Removing ambient sections in "
                    "the restriction kernel only identifies the linear "
                    "span containing the image and does not change the map."
                ),
                "immersion_certified": True,
            }
        )

    output = {
        "schema": "gcicy-headline-source-immersion-certificate-v2",
        "rank_certificate_artifact": str(rank_path),
        "rank_certificate_sha256": sha256(rank_path),
        "logical_certificate": [
            "Each effective source line bundle is ambient very ample.",
            "Every model and implemented source vector is defined over the integers.",
            "At the rank prime, the local equations retain expected codimension; regular-sequence flatness makes the chart algebra p-torsion-free.",
            "The implemented source vectors have a nonzero finite-field evaluation minor, so p-torsion-freeness lifts their independence to Q and C.",
            "Kodaira vanishing and threefold Riemann--Roch give the exact characteristic-zero h^0, so the implemented independent vectors form a basis of the complete restricted section space.",
            "The restricted ambient morphism is therefore a closed immersion.",
            "A positive-definite reference Hermitian form changes projective coordinates invertibly, and its m-fold power is followed by a Veronese embedding; both preserve immersion.",
        ],
        "certificates": certificates,
        "all_headline_source_maps_are_immersions": all(
            item["immersion_certified"] for item in certificates
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")

    rows = []
    for item in certificates:
        degree = ",".join(str(value) for value in item["degree"])
        effective_degree = ",".join(
            str(value) for value in item["effective_degree"]
        )
        rows.append(
            f"${item['reported_name']}$ & $({degree})$ & "
            f"$({effective_degree})$ & {item['source_dimension']} & "
            f"{item['rank_prime']} & "
            f"{item['implemented_nonzero_minor_mod_prime']} & "
            f"{item['special_fibre_affine_dimension']} \\\\"
        )
    tex = "\n".join(
        [
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"geometry & stored degree & effective degree & $h^0_{\mathbb C}$ & prime & implemented minor & $\dim X_p$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.write_text(tex)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
