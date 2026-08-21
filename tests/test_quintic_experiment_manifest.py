from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST = (
    ROOT / "experiments" / "manifests" / "quintic_kd_progressive_joint_paths_v1.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workflow = _load_module(
    "quintic_manifest_workflow",
    ROOT / "gcicy_metric" / "pipeline" / "experiment_workflow.py",
)
study_arm = _load_module(
    "quintic_manifest_study_arm", ROOT / "scripts" / "run_quintic_tn_study_arm.py"
)
plateau = _load_module(
    "quintic_manifest_plateau", ROOT / "scripts" / "run_quintic_tn_plateau_stage.py"
)
precision_cast = _load_module(
    "quintic_manifest_cast",
    ROOT / "scripts" / "cast_positive_tensor_network_precision.py",
)
precision_audit = _load_module(
    "quintic_manifest_audit", ROOT / "scripts" / "audit_quintic_tn_scaling_preflight.py"
)


def _plan(tmp_path: Path):
    return workflow.load_workflow_plan(
        MANIFEST,
        repo_root=ROOT,
        frozen_root=tmp_path / "frozen",
        run_root=tmp_path / "run",
        python_executable=sys.executable,
    )


def _parse_script_args(module, script: str, argv: list[str], monkeypatch):
    monkeypatch.setattr(sys, "argv", [script, *argv])
    return module.parse_args()


def test_manifest_closure_counts_and_real_argparse(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    assert len(plan.jobs) == 113
    assert sum(job.result is not None for job in plan.jobs) == 60
    assert Counter(job.phase for job in plan.jobs) == {
        "workflow-calibration": 1,
        "resource-preflight-d14": 2,
        "conditional-d16-preflight": 2,
        "main-mechanism": 42,
        "secondary-cold-initialization": 6,
        "precision-replay": 60,
    }

    parser_counts = Counter()
    for job in plan.jobs:
        script = Path(job.command[1]).name
        argv = list(job.command[2:])
        parser_counts[script] += 1
        if script == "run_quintic_tn_study_arm.py":
            study_arm.parse_args(argv)
        elif script == "run_quintic_tn_plateau_stage.py":
            plateau.parse_cli(argv)
        elif script == "cast_positive_tensor_network_precision.py":
            _parse_script_args(precision_cast, script, argv, monkeypatch)
        elif script == "audit_quintic_tn_scaling_preflight.py":
            _parse_script_args(precision_audit, script, argv, monkeypatch)
        else:  # pragma: no cover - protects future manifest edits
            raise AssertionError(f"unvalidated command entry point: {script}")
    assert parser_counts == {
        "run_quintic_tn_study_arm.py": 50,
        "run_quintic_tn_plateau_stage.py": 3,
        "cast_positive_tensor_network_precision.py": 30,
        "audit_quintic_tn_scaling_preflight.py": 30,
    }


def test_endpoint_results_pairing_and_precision_replay(tmp_path):
    plan = _plan(tmp_path)
    by_id = plan.jobs_by_id
    endpoint_ids = {
        f"main-endpoint-{arm}-r{replicate}"
        for arm in "abcdefghij"
        for replicate in (1, 2, 3)
    }
    assert endpoint_ids.issubset(by_id)
    assert len([job for job in plan.jobs if job.id.startswith("main-endpoint-")]) == 30

    required_metrics = {
        "sigma_official_formula": "development_metrics.sigma_official_formula",
        "weighted_rms_abs_residual": "development_metrics.weighted_rms_abs_residual",
        "q0.9990": "development_metrics.q0.9990",
        "cvar_0.9900": "development_metrics.cvar_0.9900",
        "q1.0000": "development_metrics.q1.0000",
        "minimum_metric_eigenvalue": "development_metrics.minimum_metric_eigenvalue",
        "nonpositive_metric_count": "development_metrics.nonpositive_metric_count",
    }
    for endpoint_id in sorted(endpoint_ids):
        endpoint = by_id[endpoint_id]
        assert endpoint.result is not None
        assert "final_training_report.json" in {
            path.path.name for path in endpoint.expected_outputs
        }
        assert required_metrics.items() <= endpoint.result["fields"].items()
        gate_pairs = {
            (gate["field"], gate.get("equals"), gate.get("gt"))
            for gate in endpoint.json_gates
        }
        assert ("development_metrics.minimum_metric_eigenvalue", None, 0) in gate_pairs
        assert ("development_metrics.nonpositive_metric_count", 0, None) in gate_pairs

        suffix = endpoint_id.removeprefix("main-endpoint-")
        cast = by_id[f"precision-cast-{suffix}"]
        replay = by_id[f"precision-replay-{suffix}"]
        assert cast.needs == (endpoint_id,)
        assert replay.needs == (cast.id,)
        assert "--precision" in cast.command and "complex128" in cast.command
        replay_tokens = set(replay.command)
        assert {
            "--split",
            "validation",
            "--limit",
            "10000",
            "--skip-gradients",
        } <= replay_tokens
        assert "--blind-reference-run-dir" not in replay_tokens
        assert replay.result is not None
        assert replay.scientific["included_in_precision_results"] is True


def test_scientific_controls_and_resource_gates(tmp_path):
    plan = _plan(tmp_path)
    by_id = plan.jobs_by_id
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert raw["scientific_policy"]["primary_result_rows"] == 27
    assert raw["scientific_policy"]["secondary_result_rows"] == 3
    assert raw["scientific_policy"]["precision_replay_rows"] == 30
    assert all(
        role not in {"source_report", "anchor_report"}
        for role in (entry["role"] for entry in raw["frozen_inputs"])
    )
    frozen_paths = "\n".join(entry["path"] for entry in raw["frozen_inputs"])
    assert "${SOURCE_RUN}/report.json" not in frozen_paths

    for scope in ("cores", "joint"):
        d14 = by_id[f"resource-preflight-d14-{scope}"]
        d16 = by_id[f"resource-preflight-d16-{scope}"]
        for job in (d14, d16):
            parsed = study_arm.parse_args(list(job.command[2:]))
            assert len(parsed.stage) == 1
            assert parsed.stage[0].epochs == parsed.stage[0].max_rounds == 1
            gates = {(gate["field"], gate.get("le")) for gate in job.json_gates}
            assert ("device_memory.maximum_allocated_bytes", 21474836480) in gates
            assert ("device_memory.maximum_reserved_bytes", 23085449216) in gates
    assert not any(
        job.id.startswith("main-endpoint-")
        and "--target-d" in job.command
        and job.command[job.command.index("--target-d") + 1] == "16"
        for job in plan.jobs
    )

    g = study_arm.parse_args(list(by_id["main-endpoint-g-r1"].command[2:]))
    i = study_arm.parse_args(list(by_id["main-endpoint-i-r1"].command[2:]))
    assert [stage.scope for stage in g.stage] == ["joint"] * 3
    assert [stage.scope for stage in i.stage] == ["joint"] * 3
    assert [stage.learning_rate for stage in g.stage] == [1e-5] * 3
    assert [stage.learning_rate for stage in i.stage] == [3e-6] * 3
    for field, value in vars(g).items():
        if field not in {"arm_dir", "stage"}:
            assert getattr(i, field) == value
    for left, right in zip(g.stage, i.stage, strict=True):
        assert left.name == right.name
        assert left.scope == right.scope
        assert left.min_relative_gain == right.min_relative_gain
        assert left.patience_rounds == right.patience_rounds
        assert left.epochs == right.epochs
        assert left.max_rounds == right.max_rounds


def test_cold_control_and_output_producers(tmp_path):
    plan = _plan(tmp_path)
    by_id = plan.jobs_by_id
    for replicate in (1, 2, 3):
        cold = by_id[f"secondary-j-cold-block1-r{replicate}"]
        wrapper, trainer = plateau.parse_cli(list(cold.command[2:]))
        assert wrapper.initial_model is None
        assert wrapper.max_rounds == 1 and wrapper.accept_round_cap
        assert wrapper.parameter_scope == "cores"
        assert trainer[trainer.index("--epochs") + 1] == "50"
        assert trainer[trainer.index("--initialization-noise") + 1] == "1e-2"
        assert trainer[trainer.index("--torch-seed") + 1] == f"2026900{replicate}"
        assert {path.path.name for path in cold.expected_outputs} == {
            "plateau_model.pt",
            "plateau_report.json",
            "stage_summary.json",
        }

        endpoint = by_id[f"main-endpoint-j-r{replicate}"]
        parsed = study_arm.parse_args(list(endpoint.command[2:]))
        assert [stage.name for stage in parsed.stage] == ["budget-2", "budget-3"]
        assert all(
            stage.epochs == 50 and stage.max_rounds == 1 for stage in parsed.stage
        )
        assert endpoint.needs == (cold.id,)

    for job in plan.jobs:
        command_text = " ".join(job.command)
        assert "blind_points.npz" not in command_text
        assert "blind_test_tail_arrays.npz" not in command_text
        assert "blind_pullbacks.npy" not in command_text
        if Path(job.command[1]).name == "run_quintic_tn_study_arm.py":
            parsed = study_arm.parse_args(list(job.command[2:]))
            assert parsed.skip_blind_audit
            assert parsed.train_limit == 90000 and parsed.validation_limit == 10000
            assert parsed.batch_size == 1024
            assert parsed.fixed_log_kappa == -4.0396350923022535
            assert parsed.bond_initialization_noise == 0
            assert parsed.bond_one_sided_activation_scale == 1e-3


def test_every_run_root_input_has_a_transitive_producer(tmp_path):
    plan = _plan(tmp_path)
    output_producers = {
        output.path: job.id for job in plan.jobs for output in job.expected_outputs
    }

    def dependency_closure(job_id: str) -> set[str]:
        closure: set[str] = set()
        pending = list(plan.jobs_by_id[job_id].needs)
        while pending:
            dependency = pending.pop()
            if dependency in closure:
                continue
            closure.add(dependency)
            pending.extend(plan.jobs_by_id[dependency].needs)
        return closure

    input_flags = {
        "run_quintic_tn_study_arm.py": ("--initial-model",),
        "run_quintic_tn_plateau_stage.py": ("--initial-model",),
        "cast_positive_tensor_network_precision.py": ("--model",),
        "audit_quintic_tn_scaling_preflight.py": ("--model", "--run-report"),
    }
    for job in plan.jobs:
        script = Path(job.command[1]).name
        closure = dependency_closure(job.id)
        for flag in input_flags[script]:
            if flag not in job.command:
                continue
            value = Path(job.command[job.command.index(flag) + 1]).resolve()
            if not value.is_relative_to(plan.run_root):
                continue
            assert value in output_producers, (job.id, flag, value)
            assert output_producers[value] in closure, (job.id, flag, value)

        if script == "audit_quintic_tn_scaling_preflight.py":
            run_dir = Path(job.command[job.command.index("--run-dir") + 1]).resolve()
            producers = {
                producer
                for output, producer in output_producers.items()
                if output.is_relative_to(run_dir)
            }
            assert producers & closure, (job.id, run_dir)
