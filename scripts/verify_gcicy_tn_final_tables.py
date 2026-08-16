#!/usr/bin/env python3
"""Verify byte-exact, structured regeneration of all five final tables."""

from __future__ import annotations

import difflib
import importlib.util
import json
from pathlib import Path
import re
import sys


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent if (SCRIPT.parent / "manuscript").is_dir() else SCRIPT.parents[1]
MANUSCRIPT = (
    ROOT / "manuscript"
    if (ROOT / "manuscript/gcicy_tn_paper.tex").is_file()
    else ROOT / "release_work/final_repro_20260815/manuscript"
)


def load_generator():
    candidates = (
        SCRIPT.parent / "REGENERATE_MANUSCRIPT_TABLES.py",
        ROOT / "scripts/generate_gcicy_tn_final_tables.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("final-table generator is absent")
    spec = importlib.util.spec_from_file_location("gcicy_final_tables", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load table generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parsed_rows(text: str) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    in_body = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == r"\midrule":
            in_body = True
            continue
        if line == r"\bottomrule":
            in_body = False
            continue
        if not in_body or not line.endswith(r"\\"):
            continue
        body = line[:-2].rstrip()
        rows.append(tuple(cell.strip() for cell in body.split(" & ")))
    return tuple(rows)


def exact_difference(expected: str, observed: str, filename: str) -> str:
    difference = difflib.unified_diff(
        observed.splitlines(),
        expected.splitlines(),
        fromfile=f"manuscript/{filename}",
        tofile=f"regenerated/{filename}",
        lineterm="",
    )
    return "\n".join(list(difference)[:80])


def check_exact_degree_wording() -> int:
    main = (MANUSCRIPT / "gcicy_tn_paper.tex").read_text(encoding="utf-8")
    supplement_path = ROOT / "supplement/supplement_controls.tex"
    if not supplement_path.is_file():
        supplement_path = ROOT / "gcicy paper/supplement_controls.tex"
    supplement = supplement_path.read_text(encoding="utf-8")
    combined = main + supplement
    required = (
        "cross-degree re-embedding is an",
        "initialization scheme rather than the exact identity",
        "registered repeat re-embedding",
    )
    forbidden = (
        "through exact degree continuation and function-preserving bond enlargement",
        "trained directly after exact degree and bond embeddings",
    )
    for fragment in required:
        if fragment not in combined:
            raise SystemExit(f"exact-degree clarification missing: {fragment}")
    for fragment in forbidden:
        if fragment in combined:
            raise SystemExit(f"superseded exact-degree wording present: {fragment}")
    return len(required) + len(forbidden)


def main() -> None:
    generator = load_generator()
    tables = generator.generated_tables()
    manifest = generator.generation_manifest(tables)
    generated_dir = MANUSCRIPT / "generated_tn"
    failures: list[str] = []
    row_counts: dict[str, int] = {}

    for table in tables:
        path = generated_dir / table.filename
        if not path.is_file():
            failures.append(f"missing manuscript table: {path}")
            continue
        observed = path.read_text(encoding="utf-8")
        if observed != table.content:
            failures.append(
                f"{table.filename} is not byte-exact generator output\n"
                + exact_difference(table.content, observed, table.filename)
            )
        observed_rows = parsed_rows(observed)
        if observed_rows != table.rows:
            failures.append(
                f"{table.filename} row/column structure differs from regenerated rows"
            )
        observed_column_counts = tuple(
            dict.fromkeys(len(row) for row in observed_rows)
        )
        if observed_column_counts != table.column_counts:
            failures.append(
                f"{table.filename} column layout {observed_column_counts} != "
                f"{table.column_counts}"
            )
        row_counts[table.filename] = len(observed_rows)

    manifest_path = generated_dir / generator.MANIFEST_NAME
    expected_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if not manifest_path.is_file():
        failures.append(f"missing final-table generation manifest: {manifest_path}")
    elif manifest_path.read_text(encoding="utf-8") != expected_manifest:
        failures.append("final-table generation manifest is not byte-exact")

    paper = (MANUSCRIPT / "gcicy_tn_paper.tex").read_text(encoding="utf-8")
    offsets = []
    for table in tables:
        token = rf"\input{{generated_tn/{table.filename}}}"
        matches = [match.start() for match in re.finditer(re.escape(token), paper)]
        if len(matches) != 1:
            failures.append(
                f"main manuscript contains {len(matches)} references to {table.filename}"
            )
        else:
            offsets.append(matches[0])
    if offsets != sorted(offsets):
        failures.append("main manuscript does not reference Tables 1--5 in generator order")

    wording_count = check_exact_degree_wording()
    if failures:
        raise SystemExit("\n\n".join(failures))
    print(
        "verified byte-exact and row/column-exact regeneration of final "
        "manuscript Tables 1--5"
    )
    print(
        json.dumps(
            {
                "tables": row_counts,
                "frozen_sources": len(manifest["sources"]),
                "wording_gates": wording_count,
                "manuscript_input_order": "verified",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
