#!/usr/bin/env python3
"""Build the final self-contained reproducibility archive for the gCICY paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "outputs/release/gcicy_tn_section4_evidence_20260810"
MANUSCRIPT = ROOT / "release_work/final_repro_20260813/manuscript"
SUPPLEMENT = ROOT / "gcicy paper"
STEM = "gcicy_tn_final_reproducibility_v1_0_8"
IGNORED_NAMES = {".DS_Store"}
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}
ALREADY_COMPRESSED = {".npz", ".npy", ".pt", ".pdf", ".png", ".zip", ".gz"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    return (
        path.name in IGNORED_NAMES
        or path.suffix == ".pyc"
        or path.suffix == ".pid"
        or any(part in IGNORED_PARTS for part in path.parts)
    )


class Collector:
    def __init__(self) -> None:
        self.files: dict[str, Path] = {}

    def add(self, source: Path, destination: str, *, replace: bool = False) -> None:
        source = source.expanduser().resolve()
        if not source.is_file() or source.is_symlink() or ignored(source):
            raise FileNotFoundError(source)
        if destination in self.files and not replace:
            if self.files[destination] == source:
                return
            raise RuntimeError(f"duplicate archive path: {destination}")
        self.files[destination] = source

    def add_tree(
        self,
        source: Path,
        destination: str,
        *,
        suffixes: set[str] | None = None,
        replace: bool = False,
    ) -> None:
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink() or ignored(path):
                continue
            if suffixes is not None and path.suffix not in suffixes:
                continue
            relative = path.relative_to(source).as_posix()
            self.add(path, f"{destination}/{relative}", replace=replace)


def local_tex_dependencies(root_file: Path) -> set[Path]:
    found: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in found:
            return
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(path)
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\input\{([^}]+)\}", text):
            candidate = path.parent / match.group(1)
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".tex")
            visit(candidate)

    visit(root_file)
    return found


def collect() -> Collector:
    collector = Collector()

    # The previous large archive already contains the complete selected raw
    # evidence closure, including frozen models and pointwise arrays.
    collector.add_tree(LEGACY / "outputs", "outputs")

    # Overlay the one corrected aggregate and preserve the pre-fix record.
    collector.add(
        ROOT / "outputs/pipeline/type11_x11_equal_time_final_20260807/final_summary.json",
        "outputs/pipeline/type11_x11_equal_time_final_20260807/final_summary.json",
        replace=True,
    )
    collector.add(
        ROOT
        / "audit_reports/repro_bundle_20260813/"
        "x11_final_summary_before_normalization_fix.json",
        "audits/x11_final_summary_before_normalization_fix.json",
    )

    # Current implementation, tests and registered protocols.
    collector.add_tree(ROOT / "gcicy_metric", "gcicy_metric", suffixes={".py"})
    collector.add_tree(ROOT / "scripts", "scripts", suffixes={".py", ".sh"})
    collector.add_tree(ROOT / "tests", "tests", suffixes={".py"})
    collector.add_tree(ROOT / "pipeline_specs", "pipeline_specs", suffixes={".json"})
    for name in ("ENVIRONMENT.md", "requirements.txt", "DEPOSITION.md"):
        path = ROOT / "artifact_release_tn" / name
        if path.is_file():
            collector.add(path, f"environment/{name}")
    # Final manuscript: source dependencies, figures, bibliography and PDF.
    for path in local_tex_dependencies(MANUSCRIPT / "gcicy_tn_paper.tex"):
        collector.add(path, f"manuscript/{path.relative_to(MANUSCRIPT).as_posix()}")
    for name in ("references.bib", "jheppub.sty", "gcicy_tn_paper.bbl", "gcicy_tn_paper.pdf"):
        collector.add(MANUSCRIPT / name, f"manuscript/{name}")
    for figure in sorted((MANUSCRIPT / "figures").glob("*")):
        if figure.suffix in {".pdf", ".png"}:
            collector.add(figure, f"manuscript/figures/{figure.name}")

    # Current Supplemental Material and precisely the generated inputs it uses.
    for path in local_tex_dependencies(SUPPLEMENT / "gcicy_tn_supplement.tex"):
        collector.add(path, f"supplement/{path.relative_to(SUPPLEMENT).as_posix()}")
    for name in (
        "gcicy_tn_supplement.bbl",
        "gcicy_tn_supplement.pdf",
        "references.bib",
    ):
        collector.add(SUPPLEMENT / name, f"supplement/{name}")
    # These generated summaries are consumed by the portable manuscript-claim
    # checker but are not direct \input dependencies of the final documents.
    for name in (
        "x11_plateau_cost_summary_20260802.tex",
        "x21_fixed_d8_degree_scaling_final_20260809.tex",
    ):
        collector.add(
            SUPPLEMENT / "generated_tn" / name,
            f"supplement/generated_tn/{name}",
        )
    collector.add(
        ROOT
        / "outputs/pipeline/type11_h2_matched_timing_replay_20260729/"
        "runtime_summary.json",
        "outputs/pipeline/type11_h2_matched_timing_replay_20260729/"
        "runtime_summary.json",
    )
    collector.add(
        ROOT / "audit_reports/repro_bundle_20260813/FIGURE_DATA_REFERENCE.json",
        "FIGURE_DATA_REFERENCE.json",
    )
    # Reconciliation and independent audit records.
    collector.add(
        ROOT / "audit_reports/repro_bundle_20260813/DISCREPANCY_REPORT.md",
        "audits/DISCREPANCY_REPORT.md",
    )
    collector.add(
        ROOT / "audit_reports/repro_bundle_20260813/pointwise_reconstruction.json",
        "audits/pointwise_reconstruction.json",
    )
    collector.add(
        ROOT / "audit_reports/DEEPSEEK_V4PRO_GCICY_AUDIT_20260813.md",
        "audits/DEEPSEEK_V4PRO_GCICY_AUDIT_20260813.md",
    )
    collector.add(
        ROOT / "audit_reports/repro_bundle_20260813/PINNED_TEST_REPORT.txt",
        "audits/PINNED_TEST_REPORT.txt",
    )
    independent = ROOT / "outputs/release/independent_audit_20260811/data_and_claim_audit.md"
    if independent.is_file():
        collector.add(independent, "audits/independent_data_and_claim_audit_20260811.md")
    return collector


def claim_map() -> dict[str, object]:
    x11_base = "outputs/pipeline/type11_x11_equal_time_final_20260807"
    x11_pointwise = [
        f"{x11_base}/replicate_{replicate}/{relative}"
        for replicate in (1, 2, 3)
        for relative in (
            "tn_arrays.npz",
            "source_density_phi/blind_tail_arrays.npz",
            "full_h6_equal_time_plateau_arrays.npz",
        )
    ]
    x21_control_base = (
        "outputs/pipeline/type21_d8_continuation_capacity_control_20260730"
    )
    x21_control_pointwise = [
        f"{x21_control_base}/seed_{seed}_{arm}_holdout_arrays.npz"
        for seed in (86231, 86232, 86233)
        for arm in ("source", "control", "d12")
    ]
    x21_ladder_base = (
        "outputs/pipeline/type21_fixed_d8_degree_scaling_floor1e14_20260807"
    )
    x21_ladder_pointwise = [
        f"{x21_ladder_base}/k{degree}/replicate_{replicate}/final_common/arrays.npz"
        for degree in (8, 12, 16, 20)
        for replicate in (1, 2, 3)
    ]
    return {
        "schema": "gcicy-tn-final-claim-evidence-map-v1",
        "claims": [
            {
                "id": "X11_COMMON_FINAL_COMPARISON",
                "evidence_level": "three-run independent final sample",
                "paper": [
                    "manuscript/gcicy_tn_paper.tex",
                    "manuscript/generated_tn/x11_equal_time_final_20260807.tex",
                    "manuscript/figures/gcicy_tn_x11_plateau_tail_survival.pdf",
                    "supplement/generated_tn/x11_final_common_holdout_20260807.tex",
                ],
                "data": [
                    f"{x11_base}/final_summary.json",
                    f"{x11_base}/X11_final_seed86707_n200000.npz",
                    "FIGURE_DATA_REFERENCE.json",
                    *x11_pointwise,
                ],
                "code": [
                    "scripts/verify_gcicy_tn_final_reproducibility.py",
                    "scripts/summarize_type11_x11_equal_time_final.py",
                    "scripts/build_gcicy_tn_review_figures.py",
                ],
            },
            {
                "id": "X11_REPRESENTATION_AND_LIFT",
                "evidence_level": "numerical representation audit",
                "paper": ["manuscript/gcicy_tn_paper.tex"],
                "data": [
                    "outputs/pipeline/type11_k6_full_h_same_degree_20260805/h2_cubic_lift_report.json",
                    "outputs/pipeline/type11_k6_full_h_same_degree_20260805/tn_replicate1_materialized_full_h_report.json",
                    "outputs/pipeline/type11_k2_to_k6_multiplication_audit_20260805/multiplication_rank_audit.json",
                ],
                "code": ["scripts/build_type11_h2_cubic_full_h_lift.py"],
            },
            {
                "id": "X21_FINITE_BUDGET_COMPRESSION",
                "evidence_level": "single fitted metric per representation on independent final sample",
                "paper": [
                    "manuscript/gcicy_tn_paper.tex",
                    "manuscript/generated_tn/x21_final_blind_compression_20260802.tex",
                ],
                "data": [
                    "outputs/pipeline/type21_q121_d12_final_blind_20260729/final_blind_summary.json",
                    "outputs/pipeline/type21_q121_d12_final_blind_20260729/h4_final_blind_arrays.npz",
                    "outputs/pipeline/type21_q121_d12_final_blind_20260729/d8_final_blind_arrays.npz",
                    "outputs/pipeline/type21_q121_d12_final_blind_20260729/d12_final_blind_arrays.npz",
                    "outputs/pipeline/type21_section_multiplication_audit_20260728/multiplication_rank_audit.json",
                ],
                "code": [
                    "scripts/summarize_type21_q121_d12_final_blind.py",
                    "scripts/run_type21_section_multiplication_audit_remote.sh",
                ],
            },
            {
                "id": "X21_EQUAL_UPDATE_BOND_CONTROL",
                "evidence_level": "three paired equal-update controls",
                "paper": ["manuscript/gcicy_tn_paper.tex"],
                "data": [
                    f"{x21_control_base}/summary.json",
                    *x21_control_pointwise,
                ],
                "code": ["scripts/summarize_type21_d8_continuation_capacity_control.py"],
            },
            {
                "id": "X21_FIXED_BOND_DEGREE_LADDER",
                "evidence_level": "three runs per degree on a common independent final sample",
                "paper": [
                    "manuscript/gcicy_tn_paper.tex",
                    "manuscript/generated_tn/x21_fixed_bond_degree_ladder.tex",
                ],
                "data": [
                    f"{x21_ladder_base}/final_summary_k8_k20.json",
                    *x21_ladder_pointwise,
                ],
                "code": ["scripts/summarize_type21_fixed_d8_degree_scaling.py"],
            },
            {
                "id": "X21_DEGREE_BOND_DEVELOPMENT_FIGURE",
                "evidence_level": "single-run development landscape",
                "paper": [
                    "manuscript/gcicy_tn_paper.tex",
                    "manuscript/figures/gcicy_tn_x21_path_dependence_20260802.pdf",
                ],
                "data": [
                    "outputs/pipeline/type21_kd_plateau_paper_summary_20260802.json",
                    "FIGURE_DATA_REFERENCE.json",
                ],
                "code": ["scripts/build_gcicy_tn_review_revision_assets.py"],
            },
            {
                "id": "X22_DICTIONARY_CAPACITY",
                "evidence_level": "single-run equal-update paired comparison",
                "paper": [
                    "manuscript/gcicy_tn_paper.tex",
                    "manuscript/generated_tn/x22_dictionary_comparison.tex",
                ],
                "data": [
                    "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/bilateral_evaluation_20260805.json",
                    "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/active_q22_k4_control_blind83703_n65532_arrays.npz",
                    "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/active_q60_k4_capacity_blind83703_n65532_arrays.npz",
                    "outputs/pipeline/type22_fixed_q289_capacity_20260806/q289_k4_capacity_blind83703_n65532.json",
                    "outputs/pipeline/type22_fixed_q289_capacity_20260806/q289_k4_capacity_blind83703_n65532_arrays.npz",
                ],
                "code": ["scripts/run_type22_fixed_q289_capacity_remote.sh"],
            },
            {
                "id": "X22_DEGREE_CONTINUATION",
                "evidence_level": "single-run common-sample paired comparison",
                "paper": ["manuscript/gcicy_tn_paper.tex"],
                "data": [
                    "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/k4_k6_common_bilateral_20260805.json",
                    "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/active_q60_k4_on_common_blind83713_n65532_arrays.npz",
                    "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/active_q60_k6_on_common_blind83713_n65532_arrays.npz",
                ],
                "code": ["scripts/bootstrap_gcicy_metric_comparison.py"],
            },
            {
                "id": "SOURCE_MAP_IMMERSION",
                "evidence_level": "exact finite-field and geometric argument",
                "paper": [
                    "manuscript/gcicy_tn_paper.tex",
                    "manuscript/appendix_geometry.tex",
                ],
                "data": [
                    "outputs/pipeline/gcicy_source_map_immersion_20260730/certificate.json",
                    "outputs/section_restriction_rank_certificates.json",
                ],
                "code": ["scripts/certify_gcicy_source_map_immersion.py"],
            },
        ],
    }


def verifier_source() -> str:
    return '''#!/usr/bin/env python3
"""Strictly verify the file set, sizes, hashes and claim references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"MANIFEST.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {record["path"]: record for record in manifest["files"]}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.relative_to(ROOT).as_posix() not in EXCLUDED
    }
    failures = []
    if manifest.get("payload_file_count") != len(expected):
        failures.append(
            "manifest payload_file_count does not match the indexed file count"
        )
    indexed_bytes = sum(int(record["bytes"]) for record in expected.values())
    if manifest.get("payload_bytes") != indexed_bytes:
        failures.append("manifest payload_bytes does not match indexed byte counts")
    if manifest.get("release_id") != ROOT.name:
        failures.append("manifest release_id does not match the package directory")
    for path in sorted(set(expected) - actual):
        failures.append(f"missing: {path}")
    for path in sorted(actual - set(expected)):
        failures.append(f"unregistered extra file: {path}")
    for relative, record in sorted(expected.items()):
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.is_symlink():
            failures.append(f"unexpected symlink: {relative}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"size mismatch: {relative}")
            continue
        observed = sha256(path)
        if observed != record["sha256"]:
            failures.append(
                f"hash mismatch: {relative}: expected {record['sha256']}, observed {observed}"
            )
    claims = json.loads((ROOT / "CLAIM_EVIDENCE_MAP.json").read_text(encoding="utf-8"))
    for claim in claims["claims"]:
        for group in ("paper", "data", "code"):
            for relative in claim[group]:
                if relative not in expected or not (ROOT / relative).is_file():
                    failures.append(f"claim reference missing: {claim['id']}:{relative}")
    for key, relative in (
        ("manuscript_pdf_sha256", "manuscript/gcicy_tn_paper.pdf"),
        ("supplement_pdf_sha256", "supplement/gcicy_tn_supplement.pdf"),
    ):
        if relative in expected and manifest.get(key) != expected[relative]["sha256"]:
            failures.append(f"manifest {key} does not match the indexed PDF digest")
    if failures:
        raise SystemExit("\\n".join(failures))
    print(
        f"verified exact set of {len(expected)} payload files and "
        f"{len(claims['claims'])} claim groups"
    )


if __name__ == "__main__":
    main()
'''


def readme_english(file_count: int, total_bytes: int) -> str:
    return f"""# Final gCICY tensor-network reproducibility bundle

This archive is bound to the final manuscript and Supplemental Material under
`manuscript/` and `supplement/`.  It contains {file_count:,} payload files
({total_bytes / 1024**3:.2f} GiB before ZIP compression), including frozen
models, common point samples, pointwise arrays, protocols, source code and
tests.

Run from a clean extraction, in this order:

```text
python3 VERIFY_BUNDLE.py
python3 -m pip install -r REPLAY_REQUIREMENTS.txt
python3 REPRODUCE_CLAIMS.py
python3 VERIFY_MANUSCRIPT_CLAIMS.py
python3 REPRODUCE_FIGURE_DATA.py --check FIGURE_DATA_REFERENCE.json
```

`VERIFY_BUNDLE.py` uses only the Python standard library and rejects missing,
modified, or unregistered extra files. `REPRODUCE_CLAIMS.py` requires NumPy and
independently reconstructs the principal X11, X21 and X22 statistics from the
pointwise arrays. It does not trust the generated LaTeX tables.
`VERIFY_MANUSCRIPT_CLAIMS.py` then reconstructs every displayed row of final
Tables 1--5 from the frozen summaries and checks the two corrected exact-degree
wording gates in the manuscript and Supplemental Material.
`REPRODUCE_FIGURE_DATA.py` reconstructs the two-sided X11 survival curves and
all X21 development-landscape points and continuation arrows from their frozen
numerical inputs.

The raw `importance_weights` arrays come from several historical evaluators
and may differ by an overall positive scale (sum one, sum equal to the number
of points, or the sum of concatenated shards). The replay normalizes every
array by its own sum before computing observables; same-sample weights then
agree to numerical precision.

After the three evidence-reconstruction commands, the documents can be rebuilt with:

```text
(cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error gcicy_tn_paper.tex)
(cd supplement && latexmk -pdf -interaction=nonstopmode -halt-on-error gcicy_tn_supplement.tex)
```

Compilation creates auxiliary files, so the exact-set verifier should be run
first on a fresh extraction. The three top-level verification entry points are
portable. Historical training scripts are included as provenance and may
require the recorded CUDA environment and explicit replacement of original
run-directory defaults before full retraining.

For the final X11 TN/residual-potential comparison, the operative recipes are
`scripts/run_type11_tn_source_phi_lr_sweep_remote.sh` and
`scripts/run_type11_selected_lr_three_seed_remote.sh`. They record the
source-density residual model with 2025 input features and width 66. The
`run_type11_h2_matched_*` launchers are earlier ambient-density experiments
retained only as historical provenance.

Read `DISCREPANCY_REPORT.md` before comparing this release with earlier paper
snapshots. One descriptive X11 maximum was corrected because the old summary
mixed final-sample and training-sample normalizations; all headline bulk,
tail-CVaR, positivity and ordering claims are unchanged.

The only large numerical objects intentionally omitted are the rebuildable
full-H6 section-value/derivative caches (about 18.5 GB). Their point samples,
models, reconstruction reports and final pointwise arrays are included.
Historical absolute filesystem paths inside JSON records are provenance
strings. Some historical shell launchers also retain their original host
defaults; they are provenance rather than portable one-command retraining
entry points. The two top-level verification commands use only
package-relative paths.
"""


def readme_chinese(file_count: int, total_bytes: int) -> str:
    return f"""# 最终 gCICY 张量网络复现包

本包与 `manuscript/` 中的最终主文和 `supplement/` 中的补充材料绑定，共含
{file_count:,} 个有效载荷文件，未压缩体积约 {total_bytes / 1024**3:.2f} GiB。
冻结模型、共同评估点、逐点数组、协议、代码和测试均在包内。

全新解压后依次运行：

```text
python3 VERIFY_BUNDLE.py
python3 -m pip install -r REPLAY_REQUIREMENTS.txt
python3 REPRODUCE_CLAIMS.py
python3 VERIFY_MANUSCRIPT_CLAIMS.py
python3 REPRODUCE_FIGURE_DATA.py --check FIGURE_DATA_REFERENCE.json
```

第一条只用 Python 标准库，严格检查文件集合、大小和 SHA-256，并拒绝未登记
的额外文件。第二条需要 NumPy，从逐点 `log eta`、重要性权重和 fibre 编号
独立重算 X11、X21、X22 的主要统计量，不读取 LaTeX 表格作为数值依据。
第三条进一步核对正文中的参数量、表格舍入值、相对改善、统计门槛、负结果和
已经排除的旧措辞。第四条从逐点数组重建 Figure 1 的双侧生存曲线，并从冻结
摘要重建 Figure 2 的全部散点和 continuation 箭头。

不同历史评估器保存的 `importance_weights` 可能相差一个整体正常数：权重和
可能为 1、点数或拼接分片数。重算脚本先将每个数组除以自身总和；完成该
归一化后，同一评估样本上的模型权重在数值精度内一致。

完成上述证据重建后，可编译主文和补充材料：

```text
(cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error gcicy_tn_paper.tex)
(cd supplement && latexmk -pdf -interaction=nonstopmode -halt-on-error gcicy_tn_supplement.tex)
```

编译会产生辅助文件，因此严格文件集合验证应先在全新解压目录执行。顶层三
个证据验证入口可直接移植；历史训练脚本作为来源记录保留，完整重训仍需要记录
的 CUDA 环境，并可能需要显式替换原始运行目录默认值。

最终 X11 TN/residual-potential 比较对应的正式配方是
`scripts/run_type11_tn_source_phi_lr_sweep_remote.sh` 与
`scripts/run_type11_selected_lr_three_seed_remote.sh`；其中 residual 模型使用
2025 维 source-density 输入和宽度 66。`run_type11_h2_matched_*` 是较早的
ambient-density 实验，仅作为历史来源记录保留。

`DISCREPANCY_REPORT.md` 解释了与旧版本的差异。真正的数值修正只有 X11
residual-phi 行的一个描述性最大值；旧汇总误用了训练样本归一化分支。所有
headline bulk、CVaR、正定性、模型排序和相对改善结论均未改变。

唯一主动排除的是约 18.5 GB、可由包内模型和点集重建的 full-H6 截面及导数
缓存；它们不是独立实验结果。
"""


def data_scope() -> dict[str, object]:
    return {
        "schema": "gcicy-tn-final-data-scope-v1",
        "included": {
            "frozen_models": True,
            "common_evaluation_samples": True,
            "pointwise_metric_arrays": True,
            "training_histories_and_protocols": True,
            "manuscript_and_supplement_sources": True,
        },
        "excluded_rebuildable_intermediates": [
            {
                "paths": [
                    "outputs/pipeline/type11_k6_full_h_same_degree_20260805/feature_cache_c64",
                    "outputs/pipeline/type11_k6_full_h_same_degree_20260805/feature_cache_c128",
                ],
                "total_bytes_on_training_host": 18494522368,
                "content_hashes_available": False,
                "reason": (
                    "Rebuildable full-H6 section-value and derivative caches; "
                    "they are not independent experimental results."
                ),
                "included_inputs_and_outputs": [
                    "input point samples",
                    "H2 cubic lift",
                    "fitted full-H6 models",
                    "training histories",
                    "reconstruction reports",
                    "final pointwise arrays",
                ],
                "rebuild_entry_points": [
                    "scripts/run_type11_k6_full_h_same_degree_remote.sh",
                    "scripts/run_type11_k6_full_h_equal_time_plateau_remote.sh",
                ],
                "note": (
                    "The launchers preserve original host defaults; set ROOT, "
                    "PYTHON and run directories for a new machine."
                ),
            }
        ],
    }


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def role(relative: str) -> str:
    if relative.startswith("manuscript/"):
        return "manuscript"
    if relative.startswith("supplement/"):
        return "supplement"
    if relative.startswith(("gcicy_metric/", "scripts/", "tests/")):
        return "source_code"
    if relative.startswith("pipeline_specs/"):
        return "protocol"
    if relative.startswith("outputs/"):
        return "numerical_evidence"
    if relative.startswith("audits/"):
        return "audit"
    return "documentation"


def build(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if not LEGACY.is_dir() or not MANUSCRIPT.is_dir():
        raise FileNotFoundError("required frozen inputs are missing")
    collector = collect()
    output = ROOT / "outputs/release"
    staging = output / STEM
    temporary = output / f".{STEM}.building"
    archive = output / f"{STEM}.zip"
    temporary_archive = output / f".{STEM}.zip.building"
    checksum = output / f"{STEM}.sha256"
    for path in (staging, temporary, archive, temporary_archive, checksum):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {path}")
    output.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for relative, source in sorted(collector.files.items()):
            hardlink_or_copy(source, temporary / relative)

        claims = claim_map()
        for claim in claims["claims"]:  # type: ignore[index]
            for group in ("paper", "data", "code"):
                for relative in claim[group]:
                    if not (temporary / relative).is_file():
                        raise FileNotFoundError(
                            f"claim reference missing: {claim['id']}:{relative}"
                        )
        write_json(temporary / "CLAIM_EVIDENCE_MAP.json", claims)
        shutil.copy2(
            ROOT / "audit_reports/repro_bundle_20260813/DISCREPANCY_REPORT.md",
            temporary / "DISCREPANCY_REPORT.md",
        )
        shutil.copy2(
            ROOT / "scripts/verify_gcicy_tn_final_reproducibility.py",
            temporary / "REPRODUCE_CLAIMS.py",
        )
        shutil.copy2(
            ROOT / "scripts/verify_gcicy_tn_final_tables.py",
            temporary / "VERIFY_MANUSCRIPT_CLAIMS.py",
        )
        shutil.copy2(
            ROOT / "scripts/reproduce_gcicy_tn_figure_data.py",
            temporary / "REPRODUCE_FIGURE_DATA.py",
        )
        (temporary / "VERIFY_BUNDLE.py").write_text(
            verifier_source(), encoding="utf-8"
        )
        (temporary / "REPLAY_REQUIREMENTS.txt").write_text(
            "numpy==1.26.4\n", encoding="ascii"
        )
        write_json(temporary / "DATA_SCOPE.json", data_scope())

        # Inventory all files except the manifest, which is the signed index.
        records = []
        for path in sorted(temporary.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(temporary).as_posix()
            records.append(
                {
                    "path": relative,
                    "role": role(relative),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        total_bytes = sum(record["bytes"] for record in records)
        (temporary / "README.md").write_text(
            readme_english(len(records) + 3, total_bytes), encoding="utf-8"
        )
        (temporary / "README_zh.md").write_text(
            readme_chinese(len(records) + 3, total_bytes), encoding="utf-8"
        )

        # README files are also payloads, so append their hashes now.
        for name in ("README.md", "README_zh.md"):
            path = temporary / name
            records.append(
                {
                    "path": name,
                    "role": "documentation",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        records.sort(key=lambda item: item["path"])
        total_bytes = sum(record["bytes"] for record in records)
        manifest = {
            "schema": "gcicy-tn-final-reproducibility-bundle-v1",
            "release_id": STEM,
            "release_date": "2026-08-13",
            "payload_file_count": len(records),
            "payload_bytes": total_bytes,
            "manuscript_pdf_sha256": sha256(MANUSCRIPT / "gcicy_tn_paper.pdf"),
            "supplement_pdf_sha256": sha256(SUPPLEMENT / "gcicy_tn_supplement.pdf"),
            "files": records,
        }
        write_json(temporary / "MANIFEST.json", manifest)
        inventory = io.StringIO(newline="")
        writer = csv.DictWriter(
            inventory, fieldnames=("path", "role", "bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(records)
        # Inventory is deliberately generated before the manifest and is not
        # itself included; the manifest already provides the canonical index.
        (temporary / "FILE_INVENTORY.csv").write_text(
            inventory.getvalue(), encoding="utf-8"
        )

        # FILE_INVENTORY must be covered too. Rebuild the manifest once.
        inventory_path = temporary / "FILE_INVENTORY.csv"
        records.append(
            {
                "path": "FILE_INVENTORY.csv",
                "role": "documentation",
                "bytes": inventory_path.stat().st_size,
                "sha256": sha256(inventory_path),
            }
        )
        records.sort(key=lambda item: item["path"])
        manifest["payload_file_count"] = len(records)
        manifest["payload_bytes"] = sum(record["bytes"] for record in records)
        manifest["files"] = records
        write_json(temporary / "MANIFEST.json", manifest)

        temporary.rename(staging)
        with zipfile.ZipFile(temporary_archive, "w", allowZip64=True) as handle:
            for path in sorted(staging.rglob("*")):
                if not path.is_file():
                    continue
                compression = (
                    zipfile.ZIP_STORED
                    if path.suffix in ALREADY_COMPRESSED
                    else zipfile.ZIP_DEFLATED
                )
                handle.write(
                    path,
                    (Path(STEM) / path.relative_to(staging)).as_posix(),
                    compress_type=compression,
                    compresslevel=None if compression == zipfile.ZIP_STORED else 6,
                )
        temporary_archive.rename(archive)
        checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if temporary_archive.exists():
            temporary_archive.unlink()
        raise
    return staging, archive, checksum


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    staging, archive, checksum = build(parse_args())
    print(staging)
    print(archive)
    print(checksum)
    print(f"archive_sha256={sha256(archive)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
