from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = (
    ROOT
    / "experiments"
    / "manifests"
    / "x21_architecture_capacity_v1.json"
)


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workflow = _load_module(
    "x21_capacity_workflow",
    "gcicy_metric/pipeline/experiment_workflow.py",
)
plateau = _load_module(
    "x21_capacity_plateau",
    "scripts/run_gcicy_tn_plateau_stage.py",
)

SCRIPT_MODULES = {
    script: _load_module(f"x21_capacity_{script.removesuffix('.py')}", f"scripts/{script}")
    for script in (
        "audit_positive_tensor_network_equivalence.py",
        "audit_type11_positive_tensor_network.py",
        "canonicalize_positive_tensor_network_artifact.py",
        "cast_positive_tensor_network_precision.py",
        "complete_positive_tensor_network_dictionary.py",
        "decide_x21_architecture_capacity_d16.py",
        "evaluate_gcicy_metric_tail_arrays.py",
        "expand_positive_tensor_network_bond.py",
        "resize_positive_tensor_network_sites.py",
        "set_positive_tensor_network_floor.py",
        "train_type11_positive_tensor_network.py",
    )
}


def _plan(tmp_path: Path):
    return workflow.load_workflow_plan(
        MANIFEST,
        repo_root=ROOT,
        frozen_root=tmp_path / "frozen",
        run_root=tmp_path / "runs" / "x21-architecture-capacity-v1",
        python_executable=sys.executable,
    )


def _option(command: tuple[str, ...] | list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def _parse_standard(module, script: str, argv: list[str], monkeypatch):
    monkeypatch.setattr(sys, "argv", [script, *argv])
    return module.parse_args()


def _parse_plateau_and_trainer(command: tuple[str, ...], monkeypatch) -> None:
    wrapper, trainer_args = plateau.parse_cli(list(command[2:]))
    synthetic = [
        *trainer_args,
        "--initial-model",
        str(wrapper.initial_model),
        "--kappa-source",
        wrapper.first_kappa_source,
        "--checkpoint",
        "/tmp/x21-capacity-checkpoint.pt",
        "--out",
        "/tmp/x21-capacity-model.pt",
        "--summary",
        "/tmp/x21-capacity-summary.json",
    ]
    _parse_standard(
        SCRIPT_MODULES["train_type11_positive_tensor_network.py"],
        "train_type11_positive_tensor_network.py",
        synthetic,
        monkeypatch,
    )


def test_manifest_cartesian_closure_and_real_argparse(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    assert len(plan.jobs) == 109
    assert Counter(job.phase for job in plan.jobs) == {
        "prepare": 4,
        "numerical-calibration": 1,
        "capacity-initialization": 42,
        "resource-preflight-d14": 1,
        "capacity-grid": 24,
        "precision-replay": 36,
        "promotion-decision": 1,
    }
    assert sum(job.result is not None for job in plan.jobs) == 12

    expected_cells = {
        (k, D, replicate)
        for k in (20, 24)
        for D in (8, 14)
        for replicate in (1, 2, 3)
    }
    endpoints = [job for job in plan.jobs if job.id.startswith("endpoint-")]
    observed_cells = {
        (
            int(job.scientific["factors"]["k"]),
            int(job.scientific["factors"]["D"]),
            int(job.scientific["factors"]["replicate"]),
        )
        for job in endpoints
    }
    assert observed_cells == expected_cells

    parser_counts = Counter()
    for job in plan.jobs:
        script = Path(job.command[1]).name
        parser_counts[script] += 1
        if script == "run_gcicy_tn_plateau_stage.py":
            _parse_plateau_and_trainer(job.command, monkeypatch)
        else:
            assert script in SCRIPT_MODULES
            _parse_standard(
                SCRIPT_MODULES[script], script, list(job.command[2:]), monkeypatch
            )
    assert parser_counts == {
        "complete_positive_tensor_network_dictionary.py": 1,
        "set_positive_tensor_network_floor.py": 1,
        "cast_positive_tensor_network_precision.py": 14,
        "audit_positive_tensor_network_equivalence.py": 13,
        "train_type11_positive_tensor_network.py": 1,
        "resize_positive_tensor_network_sites.py": 6,
        "expand_positive_tensor_network_bond.py": 12,
        "canonicalize_positive_tensor_network_artifact.py": 12,
        "run_gcicy_tn_plateau_stage.py": 24,
        "audit_type11_positive_tensor_network.py": 12,
        "evaluate_gcicy_metric_tail_arrays.py": 12,
        "decide_x21_architecture_capacity_d16.py": 1,
    }


def test_paired_seeds_complex64_training_and_fixed_q121_dictionary(tmp_path):
    plan = _plan(tmp_path)
    by_id = plan.jobs_by_id

    for replicate in (1, 2, 3):
        site_seeds = {
            _option(by_id[f"initial-sites-k{k}-r{replicate}"].command, "--bridge-noise-seed")
            for k in (20, 24)
        }
        assert site_seeds == {f"863000{replicate}"}

        bond_seeds = {
            _option(
                by_id[f"initial-bond-k{k}-d{D}-r{replicate}"].command,
                "--seed",
            )
            for k in (20, 24)
            for D in (8, 14)
        }
        assert bond_seeds == {f"864000{replicate}"}

        optimizer_seeds = set()
        for prefix in ("capacity-primary", "capacity-precision"):
            for k in (20, 24):
                for D in (8, 14):
                    command = by_id[f"{prefix}-k{k}-d{D}-r{replicate}"].command
                    optimizer_seeds.add(_option(command, "--torch-seed"))
                    assert _option(command, "--precision") == "complex64"
                    assert _option(command, "--batch-size") == "1024"
                    assert _option(command, "--train-seed") == "86201"
                    assert _option(command, "--validation-seed") == "86202"
                    assert "--train-physical-dictionary" not in command
        assert optimizer_seeds == {f"866000{replicate}"}

    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert raw["scientific_policy"]["physical_dictionary"].startswith(
        "fixed complete q=121"
    )
    assert by_id["prepare-q121-k4"].command[1].endswith(
        "complete_positive_tensor_network_dictionary.py"
    )
    for job in plan.jobs:
        if job.id.startswith(("capacity-primary-", "capacity-precision-")):
            gates = {
                (gate["field"], gate.get("equals")) for gate in job.json_gates
            }
            k = int(_option(job.command, "--site-count"))
            D = int(_option(job.command, "--bond-dimension"))
            parameter_count = 2 * 121 * (2 * D + (k - 2) * D * D)
            assert {
                ("site_count", k),
                ("bond_dimension", D),
                ("trainable_real_parameter_count", parameter_count),
                ("positive_floor", 1.0e-14),
                ("physical_dictionary_rank", 121),
                ("trainable_physical_dictionary", False),
                ("physical_dictionary_gauge", "fixed"),
                (
                    "train.common_pool",
                    str(
                        plan.frozen_root
                        / "outputs/pipeline/gcicy_tn_common_pools_20260728"
                        / "X21_train_seed86201_n196608.npz"
                    ),
                ),
                (
                    "train.common_pool_sha256",
                    "050092ddc3f0847cde4b730ddd33c4e7faa6cbaffbfdafd147b171a5f191cf94",
                ),
                (
                    "validation.common_pool",
                    str(
                        plan.frozen_root
                        / "outputs/pipeline/gcicy_tn_common_pools_20260728"
                        / "X21_selection_seed86202_n24576.npz"
                    ),
                ),
                (
                    "validation.common_pool_sha256",
                    "d39779ad4bc0dc7bd4381486b198124a563d9a798bfe425a60775e87a1ef7ee9",
                ),
            } <= gates


def test_parameter_counts_and_full_d14_resource_gate(tmp_path):
    plan = _plan(tmp_path)
    by_id = plan.jobs_by_id
    expected = {
        (20, 8): 282656,
        (24, 8): 344608,
        (20, 14): 860552,
        (24, 14): 1050280,
    }
    for (k, D), count in expected.items():
        assert count == 2 * 121 * (2 * D + (k - 2) * D * D)
        for replicate in (1, 2, 3):
            canonical = by_id[f"initial-canonical-k{k}-d{D}-r{replicate}"]
            count_gates = [
                gate
                for gate in canonical.json_gates
                if gate["field"] == "trainable_real_parameter_count"
            ]
            assert count_gates == [
                {
                    "path": str(
                        plan.run_root
                        / "jobs"
                        / "initial"
                        / f"k{k}_d{D}_r{replicate}"
                        / "canonical"
                        / "summary.json"
                    ),
                    "field": "trainable_real_parameter_count",
                    "equals": count,
                }
            ]
            endpoint = by_id[f"endpoint-k{k}-d{D}-r{replicate}"]
            assert (
                endpoint.scientific["factors"]["trainable_real_parameter_count"]
                == count
            )

    preflight = by_id["resource-preflight-k24-d14"]
    assert preflight.needs == ("initial-equivalence-k24-d14-r1",)
    assert _option(preflight.command, "--site-count") == "24"
    assert _option(preflight.command, "--bond-dimension") == "14"
    assert _option(preflight.command, "--precision") == "complex64"
    assert _option(preflight.command, "--train-points") == "196608"
    assert _option(preflight.command, "--validation-points") == "24576"
    assert _option(preflight.command, "--epochs") == "1"
    assert _option(preflight.command, "--batch-size") == "1024"
    assert preflight.resume is not None

    gate_map = defaultdict(list)
    for gate in preflight.json_gates:
        gate_map[gate["field"]].append(gate)
    assert gate_map["trainable_real_parameter_count"] == [
        {
            "path": str(plan.run_root / "resource_preflight/k24_d14/summary.json"),
            "field": "trainable_real_parameter_count",
            "equals": 1050280,
        }
    ]
    assert {gate.get("le") for gate in gate_map["device_memory.maximum_allocated_bytes"]} == {
        None,
        21474836480,
    }
    assert gate_map["device_memory.maximum_reserved_bytes"][0]["le"] == 23085449216


def test_complex128_replay_dependency_and_validity_gates(tmp_path):
    plan = _plan(tmp_path)
    by_id = plan.jobs_by_id
    for k in (20, 24):
        for D in (8, 14):
            for replicate in (1, 2, 3):
                suffix = f"k{k}-d{D}-r{replicate}"
                cast = by_id[f"replay-cast-{suffix}"]
                audit = by_id[f"replay-audit-{suffix}"]
                endpoint = by_id[f"endpoint-{suffix}"]
                assert cast.needs == (f"capacity-precision-{suffix}",)
                assert audit.needs == (cast.id,)
                assert endpoint.needs == (audit.id,)
                assert _option(cast.command, "--precision") == "complex128"
                assert _option(audit.command, "--common-pool-split") == "confirmation"
                assert endpoint.result is not None

                cast_gates = {
                    (gate["field"], gate.get("equals"))
                    for gate in cast.json_gates
                }
                assert {
                    ("source_precision", "complex64"),
                    ("target_precision", "complex128"),
                    ("optimizer_updates", 0),
                    ("exact_binary_promotion", True),
                    ("maximum_source_dtype_roundtrip_difference", 0),
                } <= cast_gates
                endpoint_gates = {
                    (gate["field"].rsplit(".", 1)[-1], gate.get("equals"), gate.get("gt"))
                    for gate in endpoint.json_gates
                }
                assert ("nonpositive_metric_count", 0, None) in endpoint_gates
                assert ("minimum_metric_eigenvalue", None, 0) in endpoint_gates


def test_no_blind_or_d16_accuracy_job_and_promotion_is_closed_by_decision(tmp_path):
    plan = _plan(tmp_path)
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    promotion = raw["scientific_policy"]["d16_promotion"]
    assert promotion["status"].startswith("preregistered deterministic decision job")
    assert promotion["accuracy_jobs_in_this_manifest"] == 0
    assert promotion["target"] == {
        "k": 24,
        "D": 16,
        "trainable_real_parameter_count": 1370688,
    }

    assert all("blind" not in entry["role"].lower() for entry in raw["frozen_inputs"])
    assert all("blind" not in entry["path"].lower() for entry in raw["frozen_inputs"])
    forbidden = raw["forbidden_inputs"]
    assert len(forbidden["paths"]) >= 4
    assert any("final_blind" in path for path in forbidden["paths"])
    forbidden_resolved = [
        path.replace("${FROZEN_ROOT}", str(plan.frozen_root))
        for path in forbidden["paths"]
    ]
    for job in plan.jobs:
        if "d16" in job.id:
            assert job.id == "decide-d16-promotion"
            assert job.scientific["accuracy_job"] is False
        if "--bond-dimension" in job.command:
            assert _option(job.command, "--bond-dimension") in {"8", "14"}
        if "--common-pool-split" in job.command:
            assert _option(job.command, "--common-pool-split") == "confirmation"
        factors = job.scientific.get("factors", {})
        assert factors.get("D") != 16
        serialized = json.dumps(
            {
                "command": job.command,
                "gates": job.json_gates,
                "result": job.result,
            }
        )
        assert all(path not in serialized for path in forbidden_resolved)
        assert all(
            field not in serialized
            for field in forbidden["forbidden_report_fields"]
        )

    decision_job = plan.jobs_by_id["decide-d16-promotion"]
    endpoint_ids = {
        f"endpoint-k{k}-d{D}-r{replicate}"
        for k in (20, 24)
        for D in (8, 14)
        for replicate in (1, 2, 3)
    }
    assert set(decision_job.needs) == endpoint_ids
    assert decision_job.retry["max_attempts"] == 1
    assert decision_job.result is None
    assert _option(decision_job.command, "--out") == str(
        plan.run_root / "adjudication" / "d16_promotion_decision.json"
    )
    assert {
        (gate["field"], gate.get("equals")) for gate in decision_job.json_gates
    } >= {
        ("complete", True),
        ("row_count", 12),
        ("completeness.exact_cartesian_closure", True),
        ("rules_evaluated", True),
    }


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
        "complete_positive_tensor_network_dictionary.py": ("--model",),
        "set_positive_tensor_network_floor.py": ("--model",),
        "cast_positive_tensor_network_precision.py": ("--model",),
        "audit_positive_tensor_network_equivalence.py": ("--model-a", "--model-b"),
        "train_type11_positive_tensor_network.py": ("--initial-model",),
        "resize_positive_tensor_network_sites.py": ("--model",),
        "expand_positive_tensor_network_bond.py": ("--model",),
        "canonicalize_positive_tensor_network_artifact.py": ("--model",),
        "run_gcicy_tn_plateau_stage.py": ("--initial-model",),
        "audit_type11_positive_tensor_network.py": ("--model",),
        "evaluate_gcicy_metric_tail_arrays.py": ("--arrays",),
        "decide_x21_architecture_capacity_d16.py": (),
    }
    for job in plan.jobs:
        script = Path(job.command[1]).name
        closure = dependency_closure(job.id)
        for flag in input_flags[script]:
            if flag not in job.command:
                continue
            value = Path(_option(job.command, flag)).resolve()
            if not value.is_relative_to(plan.run_root):
                continue
            assert value in output_producers, (job.id, flag, value)
            assert output_producers[value] in closure, (job.id, flag, value)
