#!/usr/bin/env python3
"""Build a compact, self-verifying source audit bundle for the gCICY TN code."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STEM = "gcicy_tn_core_code_audit_v1_0_4"
OUT = ROOT / "outputs/release"
STAGING = OUT / STEM

CORE_SCRIPTS = (
    "audit_gcicy_section_multiplication.py",
    "audit_h_precision_factorization.py",
    "audit_positive_tensor_network_blocking.py",
    "audit_type11_positive_tensor_network.py",
    "bootstrap_gcicy_metric_comparison.py",
    "bootstrap_type22_dictionary_capacity.py",
    "build_type11_h2_cubic_full_h_lift.py",
    "canonicalize_positive_tensor_network_artifact.py",
    "certify_gcicy_source_map_immersion.py",
    "complete_positive_tensor_network_dictionary.py",
    "compress_positive_tensor_network_local_dictionary.py",
    "convert_positive_tensor_network_to_matrix_unit_dictionary.py",
    "evaluate_gcicy_metric_tail_arrays.py",
    "expand_positive_tensor_network_bond.py",
    "expand_positive_tensor_network_dictionary_rank.py",
    "generate_gcicy_common_point_pool.py",
    "initialize_positive_tensor_network_dictionary.py",
    "materialize_type11_k6_tn_as_full_h.py",
    "resize_positive_tensor_network_sites.py",
    "run_cymetric_phi_gcicy_type11.py",
    "summarize_type11_x11_equal_time_final.py",
    "summarize_type21_d8_continuation_capacity_control.py",
    "summarize_type21_fixed_d8_degree_scaling.py",
    "summarize_type21_q121_d12_final_blind.py",
    "train_gcicy_pipeline.py",
    "train_quintic_full_h_same_points.py",
    "train_quintic_positive_tensor_network_same_points.py",
    "train_type11_k6_reference_whitened_full_h.py",
    "train_type11_positive_tensor_network.py",
    "verify_gcicy_tn_final_reproducibility.py",
)

PROTOCOLS = (
    "run_type11_tn_source_phi_lr_sweep_remote.sh",
    "run_type11_selected_lr_three_seed_remote.sh",
    "run_type11_k6_full_h_equal_time_plateau_remote.sh",
    "run_type21_q121_d8_common_pool_three_seed_remote.sh",
    "run_type21_fixed_d8_degree_scaling_remote.sh",
    "run_type22_tensor_network_q22_transfer_remote.sh",
    "run_type22_tensor_network_q60_capacity_remote.sh",
    "run_type22_fixed_q289_capacity_remote.sh",
)

TESTS = (
    "test_positive_tensor_network.py",
    "test_positive_tensor_network_blocking.py",
    "test_positive_tensor_network_training_objectives.py",
    "test_gcicy_pipeline.py",
    "test_gcicy_tail_statistics.py",
    "test_algebraic_metric_power_lift.py",
    "test_type21_p5p1_candidate.py",
    "test_metric_regularity.py",
    "test_positive_multiplication_tree.py",
    "test_type11_reference_whitened_full_h.py",
)

FIXTURES = (
    "outputs/gcicy_generic_global_h_metric_k3_rank64_whitened_gpu.npz",
    "outputs/pipeline/p4p1p1_type21_hirzebruch_k1_pilot.npz",
    "outputs/pipeline/p5p1_type21_k3_1223_k1_refined.npz",
    "outputs/pipeline/p5p1_type21_k3_1223_k2_gpu.npz",
)

KEY_PURPOSES = {
    "gcicy_metric/pipeline/positive_tensor_network.py": (
        "positive TN",
        "Purified MPS/MPO map, shared local dictionary, direct contraction of F and its derivatives.",
    ),
    "gcicy_metric/pipeline/algebraic_power_lift.py": (
        "degree lift",
        "Exact F_b^m lift and metric-preserving degree continuation.",
    ),
    "gcicy_metric/pipeline/adapter.py": (
        "geometry interface",
        "Common interface joining charts, sections, residue density and training.",
    ),
    "gcicy_metric/pipeline/audit.py": (
        "evaluation",
        "Metric, residual, positivity, uncertainty and effective-sample diagnostics.",
    ),
    "gcicy_metric/pipeline/tail.py": (
        "tail statistics",
        "Weighted quantiles, CVaR and paired tail comparisons.",
    ),
    "gcicy_metric/pipeline/risk.py": (
        "training risk",
        "Differentiable weighted tail and log-mean-exp operations.",
    ),
    "gcicy_metric/pipeline/projective_residual_phi.py": (
        "comparison model",
        "Residual scalar-potential network and differentiated metric.",
    ),
    "gcicy_metric/pipeline/factorized_h.py": (
        "full-H",
        "Positive factorized Hermitian-form metric and training support.",
    ),
    "gcicy_metric/type21_hirzebruch_x3.py": (
        "X11 geometry",
        "Equations, complete four-root sampler, tangent basis, residue and sections.",
    ),
    "gcicy_metric/type21_candidate_p5p1_1223.py": (
        "X21 geometry",
        "Equations, complete six-root sampler, tangent basis, residue and sections.",
    ),
    "gcicy_metric/generic_model.py": (
        "X22 geometry",
        "Two-stage gCICY equations, complete three-root sampler, residue and sections.",
    ),
    "gcicy_metric/global_sections.py": (
        "algebraic metric",
        "Section values/Jacobians and full-H Kahler metric.",
    ),
    "gcicy_metric/implicit_atlas.py": (
        "local coordinates",
        "Implicit tangent coordinates and metric chart transformations.",
    ),
    "gcicy_metric/product_projective.py": (
        "ambient geometry",
        "Product-projective charts and Fubini--Study forms.",
    ),
    "scripts/train_type11_positive_tensor_network.py": (
        "TN training",
        "Native geometric losses, fixed training normalization, validation and checkpointing.",
    ),
    "scripts/audit_type11_positive_tensor_network.py": (
        "TN evaluation",
        "Independent checkpoint evaluation on registered point samples.",
    ),
    "scripts/train_type11_k6_reference_whitened_full_h.py": (
        "full-H training",
        "Same-degree positive full-H6 comparison in whitened coordinates.",
    ),
    "scripts/run_cymetric_phi_gcicy_type11.py": (
        "residual-phi training",
        "Residual-potential comparison arm on the common X11 geometry.",
    ),
    "scripts/materialize_type11_k6_tn_as_full_h.py": (
        "equivalence check",
        "Materializes a TN operator in the degree-six section space.",
    ),
    "scripts/generate_gcicy_common_point_pool.py": (
        "sampling driver",
        "Creates immutable common train/validation/evaluation point sets.",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_python_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        copy_file(path, destination / path.relative_to(source))


def write_verifier(path: Path) -> None:
    text = '''#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
expected = {row["path"]: row for row in manifest["files"]}
actual = {
    p.relative_to(ROOT).as_posix()
    for p in ROOT.rglob("*")
    if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts
}
actual.discard("MANIFEST.json")
actual.discard("VERIFY_BUNDLE.py")
if actual != set(expected):
    raise SystemExit(f"file-set mismatch: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}")
for relative, row in expected.items():
    p = ROOT / relative
    if p.stat().st_size != row["bytes"]:
        raise SystemExit(f"size mismatch: {relative}")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != row["sha256"]:
        raise SystemExit(f"hash mismatch: {relative}")
if manifest["file_count"] != len(expected):
    raise SystemExit("manifest file_count mismatch")
if manifest["payload_bytes"] != sum(row["bytes"] for row in expected.values()):
    raise SystemExit("manifest payload_bytes mismatch")
print(f"verified {len(expected)} core-audit payload files")
'''
    path.write_text(text, encoding="utf-8")


def purpose_for(relative: str) -> tuple[str, str]:
    if relative in KEY_PURPOSES:
        return KEY_PURPOSES[relative]
    if relative.startswith("gcicy_metric/pipeline/adapters/"):
        return "geometry adapter", "Connects a registered gCICY model to the common pipeline."
    if relative.startswith("gcicy_metric/pipeline/"):
        return "pipeline support", "Supporting numerical geometry, training or evaluation module."
    if relative.startswith("gcicy_metric/"):
        return "geometry support", "Supporting projective, section or model-specific implementation."
    if relative.startswith("tests/"):
        return "test", "Deterministic formula- or pipeline-level regression tests."
    if relative.startswith("fixtures/"):
        return "test fixture", "Small fixed numerical artifact used by the included tests."
    if relative.startswith("protocol_examples/"):
        return "protocol", "Recorded command recipe exposing the reported training schedule."
    if relative.startswith("scripts/"):
        return "driver", "Command-line implementation or audit utility used by the project."
    if relative.startswith("paper/"):
        return "paper reference", "Equation and convention reference for the code audit."
    return "documentation", "Audit-package documentation or environment metadata."


def main() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    for source in sorted((ROOT / "core_audit_bundle").iterdir()):
        if source.is_file():
            copy_file(source, STAGING / source.name)

    copy_python_tree(ROOT / "gcicy_metric", STAGING / "gcicy_metric")

    for name in CORE_SCRIPTS:
        copy_file(ROOT / "scripts" / name, STAGING / "scripts" / name)
    for name in PROTOCOLS:
        copy_file(
            ROOT / "scripts" / name,
            STAGING / "protocol_examples" / name,
        )
    for name in TESTS:
        copy_file(ROOT / "tests" / name, STAGING / "tests" / name)
    for relative in FIXTURES:
        copy_file(ROOT / relative, STAGING / "fixtures" / relative.removeprefix("outputs/"))

    # Preserve the locations expected by two legacy-compatible tests.
    for relative in FIXTURES:
        copy_file(ROOT / relative, STAGING / relative)

    copy_file(
        ROOT / "release_work/final_repro_20260813/manuscript/gcicy_tn_paper.pdf",
        STAGING / "paper/gcicy_tn_paper.pdf",
    )
    copy_file(
        ROOT / "gcicy paper/gcicy_tn_supplement.pdf",
        STAGING / "paper/gcicy_tn_supplement.pdf",
    )
    copy_file(
        Path(__file__),
        STAGING / "build_provenance/build_gcicy_tn_core_audit_bundle.py",
    )

    inventory_paths = sorted(
        p.relative_to(STAGING).as_posix()
        for p in STAGING.rglob("*")
        if p.is_file()
    )
    with (STAGING / "CORE_FILE_INDEX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "category", "mathematical_or_audit_role"))
        for relative in inventory_paths:
            category, purpose = purpose_for(relative)
            writer.writerow((relative, category, purpose))

    payloads = sorted(
        p for p in STAGING.rglob("*")
        if p.is_file() and p.name not in {"MANIFEST.json", "VERIFY_BUNDLE.py"}
    )
    records = [
        {
            "path": p.relative_to(STAGING).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in payloads
    ]
    manifest = {
        "schema": "gcicy-tn-core-code-audit-bundle-v1",
        "release_id": STEM,
        "file_count": len(records),
        "payload_bytes": sum(row["bytes"] for row in records),
        "reference_test_count": 148,
        "files": records,
    }
    (STAGING / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_verifier(STAGING / "VERIFY_BUNDLE.py")

    archive = OUT / f"{STEM}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                output.write(path, f"{STEM}/{path.relative_to(STAGING).as_posix()}")
    checksum = sha256(archive)
    checksum_path = OUT / f"{STEM}.sha256"
    checksum_path.write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    print(STAGING)
    print(archive)
    print(checksum_path)
    print(f"archive_sha256={checksum}")


if __name__ == "__main__":
    main()
