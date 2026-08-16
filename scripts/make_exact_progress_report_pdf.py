#!/usr/bin/env python3
"""Build the corrected exact-model gCICY metric progress report."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

from make_progress_report_pdf import bullet_list, make_styles, p, page_footer, register_fonts, table


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "gcicy_metric_exact_global_progress_report.pdf"


def load(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sci(value: float) -> str:
    return f"{value:.3e}"


def build_story(styles):
    prototype = load("outputs/gcicy_global_geometry_check.json")
    exact_geometry = load("outputs/gcicy_generic_global_model_check.json")
    metric_geometry = load("outputs/gcicy_generic_metric_geometry_check.json")
    training = load("outputs/gcicy_generic_global_h_metric_summary.json")
    audit = load("outputs/gcicy_generic_global_h_metric_audit.json")
    smooth_primes = [
        load("outputs/gcicy_exact_smoothness_check_p31991.json"),
        load("outputs/gcicy_exact_smoothness_check_p32003.json"),
        load("outputs/gcicy_exact_smoothness_check_p65521.json"),
    ]

    story = [
        p("gCICY Ricci-flat Metric：Exact Global Model 进展报告", styles, "Title"),
        p("更新日期：2026-07-10。本文取代 2026-05-02 的阶段性结论。", styles, "Subtitle"),
        p("当前结论", styles, "H1"),
        p(
            "项目已经从单一 reference patch 的局部数值原型，推进到一个固定小整数系数、"
            "具有全局 rational sections、plane-cubic sampler、24 个 projective charts、"
            "35 种 implicit coordinate choices 和 positive full-H metric 的 exact global model。",
            styles,
        ),
        bullet_list(
            [
                "重要更正：旧 simple_patch.py 的 sparse p1 在 z5 chart 上存在边界奇异 locus；旧 metric artifacts 只能作为数值基础设施测试。",
                "新模型 seed=20260711，p1/p2 使用固定非零小整数系数，可以交给 Singular 做精确有限域 Gröbner 检查。",
                "三个素数 31991、32003、65521 上，24/24 affine charts 的 singular ideal 都包含 1。",
                "exact model 上的 O(1,1,1)|X section space 从 24 个 ambient monomials 降到 17 个 independent restrictions。",
                f"fresh audit：8×512 unseen points 全过，mean RMS={audit['mean_candidate_rms']:.4f}，worst RMS={audit['max_candidate_rms']:.4f}。",
                "当前结果仍是 section degree one 的 numerical Ricci-flat approximation，不是精确 Ricci-flat metric。",
            ],
            styles,
        ),
        p("最重要的科学变化，是我们不再把 reference-patch 的好 residual 误认为全局 Calabi-Yau 结果。", styles),
        PageBreak(),
        p("1. 为什么旧原型必须淘汰", styles, "H1"),
        p(
            "旧模型为了显式写出 w5=-p1_rest，只保留了 x0 y0 z0^2 z5 型线性项。"
            "reference sampler 永远假设 x0 y0 z0 不为零，因此看不到 z=[0:0:0:0:0:1] 的边界。"
            "在 z5!=0 chart 中，这个边界满足四个 defining equations，但第四个 Jacobian singular value 约为 1e-13。",
            styles,
        ),
        table(
            [
                ["Prototype check", "Result"],
                ["rational representative error", sci(prototype["max_rational_representation_error"])],
                ["q transition error", sci(prototype["max_section_transition_error"])],
                ["reference-patch min Jacobian singular value", sci(prototype["min_gcicy_jacobian_singular_value"])],
                ["boundary min Jacobian singular value", sci(prototype["min_boundary_jacobian_singular_value"])],
                ["globally smooth", str(prototype["prototype_is_globally_smooth"])],
            ],
            [8.0 * cm, 7.5 * cm],
            styles,
        ),
        p(
            "这说明 rational-section transition 可以完全正确，但 underlying variety 仍可能奇异。"
            "因此多 patch smoothness 检查是 gCICY metric pipeline 的必要组成部分，而不是附加测试。",
            styles,
        ),
        p("2. Exact generic model", styles, "H1"),
        p(
            "新模型不再强迫任何 z coordinate 线性出现。固定 x,y 后，p2=q1=q2 是 z 的三个线性条件，"
            "其 kernel 是 projective P2；p1 限制成该 P2 上的 generic plane cubic。采样器用随机直线与 cubic 求交，"
            "保留全部多项式 roots。",
            styles,
        ),
        table(
            [
                ["Exact model numerical check", "Result"],
                ["model seed", str(exact_geometry["model_seed"])],
                ["max homogeneous residual", sci(exact_geometry["max_homogeneous_residual"])],
                ["max local residual", sci(exact_geometry["max_local_equation_residual"])],
                ["min sampled Jacobian singular value", sci(exact_geometry["minimum_sampled_jacobian_singular_value"])],
                ["q transition charts", str(exact_geometry["transition_charts"])],
            ],
            [8.0 * cm, 7.5 * cm],
            styles,
        ),
        PageBreak(),
        p("3. Exact smoothness evidence", styles, "H1"),
        p(
            "对每个 P1×P1×P5 affine chart，构造 ideal <f1,f2,q1,q2, all 4x4 Jacobian minors>。"
            "Singular 的 exact finite-field Gröbner basis 包含 1，表示该 chart 的 singular locus 为空。",
            styles,
        ),
        table(
            [["Characteristic", "Charts proved smooth", "Timeouts"]]
            + [
                [str(item["characteristic"]), f"{item['charts_proved_smooth']}/{item['charts_requested']}", str(item["charts_timed_out"])]
                for item in smooth_primes
            ],
            [5.0 * cm, 6.0 * cm, 4.5 * cm],
            styles,
        ),
        p(
            "三个互不相同的大素数得到相同结果。论文中还需要把 exact model 视为整数基底上的 proper complete-intersection family，"
            "明确陈述 good reduction 和 flatness 如何推出 characteristic-zero generic fibre smooth。",
            styles,
        ),
        p("4. Residue metric geometry", styles, "H1"),
        p(
            "每个 sampled point 选择 well-conditioned projective chart，再从 7 个 ambient affine coordinates 中选择"
            "三个 independent coordinates。其余四个 coordinates 由 4×4 equation Jacobian minor 隐式求导。"
            "Poincare residue density 为 -2 log|det J_dep|。",
            styles,
        ),
        table(
            [
                ["Atlas check", "Result"],
                ["projective charts", f"{metric_geometry['projective_charts_seen']}/24"],
                ["implicit coordinate choices", f"{metric_geometry['implicit_coordinate_choices_seen']}/35"],
                ["max projective MA error", sci(metric_geometry["max_projective_monge_ampere_error"])],
                ["max implicit MA error", sci(metric_geometry["max_implicit_monge_ampere_error"])],
                ["minimum baseline metric eigenvalue", sci(metric_geometry["minimum_metric_eigenvalue"])],
            ],
            [8.0 * cm, 7.5 * cm],
            styles,
        ),
        PageBreak(),
        p("5. Global-section full-H metric", styles, "H1"),
        p(
            "对 L=O(1,1,1)|X 枚举 24 个 ambient homogeneous monomials，在 sampled X 上用 rank-revealing QR"
            "得到 17 个 independent restrictions。H 使用 Cholesky parameterization H=LL†，因此始终正定。"
            "Kahler potential 为 K=log(s†Hs)，metric 由 section values 和 analytic derivatives 直接计算。",
            styles,
        ),
        table(
            [
                ["Training/export", "Baseline", "Full H"],
                ["centered MA RMS", f"{training['baseline_export']['rms']:.4f}", f"{training['global_h_export']['rms']:.4f}"],
                ["max absolute residual", f"{training['baseline_export']['max_abs']:.4f}", f"{training['global_h_export']['max_abs']:.4f}"],
                ["minimum metric eigenvalue", sci(training["baseline_export"]["min_eigenvalue"]), sci(training["global_h_export"]["min_eigenvalue"])],
            ],
            [6.0 * cm, 4.7 * cm, 4.7 * cm],
            styles,
        ),
        p("6. 完全独立 fresh-seed audit", styles, "H1"),
        table(
            [["Seed", "Baseline RMS", "H RMS", "Min eig", "Pass"]]
            + [
                [str(row["seed"]), f"{row['baseline_rms']:.4f}", f"{row['candidate_rms']:.4f}", sci(row["candidate_min_eigenvalue"]), str(row["passed"])]
                for row in audit["rows"]
            ],
            [2.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm, 2.2 * cm],
            styles,
        ),
        p(
            f"汇总：mean RMS={audit['mean_candidate_rms']:.4f}，worst RMS={audit['max_candidate_rms']:.4f}，"
            f"minimum metric eigenvalue={audit['min_candidate_eigenvalue']:.4f}，"
            f"minimum defining-Jacobian singular value={audit['min_sampled_jacobian_singular_value']:.4f}，"
            f"trained-H 24-chart MA error={sci(audit['max_trained_h_chart_ma_error'])}。",
            styles,
        ),
        PageBreak(),
        p("7. 已经完成与尚未完成", styles, "H1"),
        bullet_list(
            [
                "已完成：exact global defining data、rational representatives、24-chart transition、plane-cubic sampler、implicit residue、global section quotient、positive full H、disjoint audit。",
                "已完成：三个有限域上的 exact Singular smoothness checks，24/24 charts 全过。",
                "未完成：characteristic-zero smoothness 的正式 theorem-proof 写作和 machine-checkable certificate archive。",
                "未完成：exact model 上 k=2,k=3 section degree 与 H-rank 收敛。",
                "未完成：第二、第三个 generic gCICY benchmark，以及 HYM/Yukawa/Laplacian 等 metric-dependent application。",
            ],
            styles,
        ),
        p("8. 可复现命令", styles, "H1"),
        p(
            "python3 scripts/check_generic_global_model.py<br/>"
            "python3 scripts/check_generic_metric_geometry.py<br/>"
            "python3 scripts/check_exact_smoothness.py --characteristic 32003<br/>"
            "python3 scripts/train_generic_global_h_metric.py<br/>"
            "python3 scripts/audit_generic_global_h_metric.py",
            styles,
            "Code",
        ),
        p(
            "当前最准确的表述是：我们已经得到一份 exact smooth gCICY candidate 上、全局一致且独立验证的"
            "degree-one numerical Ricci-flat approximation；下一道发表门槛是 section-degree 收敛和至少一个物理 observable。",
            styles,
        ),
    ]
    return story


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    font_name, bold_name = register_fonts()
    styles = make_styles(font_name, bold_name)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="gCICY Exact Global Metric Progress Report",
        author="Codex",
    )
    doc.build(build_story(styles), onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUT)


if __name__ == "__main__":
    main()
