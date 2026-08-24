from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from gcicy_metric.pipeline.experiment_workflow import (
    WorkflowError,
    _retryable,
    active_python_executable,
    atomic_write_json,
    bootstrap_lock,
    campaign_lock,
    initialize_run_root,
    job_state_path,
    load_workflow_plan,
    record_cuda_identity,
    read_job_state,
    run_workflow,
    selection_requires_gpu,
    status_rows,
    summarize_results,
    verify_frozen_inputs,
    verify_cuda_requirements,
)


def test_active_python_executable_preserves_virtualenv_symlink(tmp_path, monkeypatch):
    target = Path(sys.executable).resolve()
    link = tmp_path / "venv-python"
    link.symlink_to(target)
    monkeypatch.setattr(sys, "executable", str(link))
    assert active_python_executable() == str(link.absolute())
    assert active_python_executable() != str(link.resolve())


def write_manifest(
    tmp_path: Path,
    *,
    jobs: list[dict] | None = None,
    job_templates: list[dict] | None = None,
    run_root: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    immutable = frozen_root / "input.bin"
    immutable.write_bytes(b"immutable")
    protected = frozen_root / "release"
    protected.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    resolved_run_root = run_root or tmp_path / "runs" / "campaign"
    manifest = {
        "schema": "gcicy-tn-gpu-workflow-v1",
        "campaign_id": "test-campaign",
        "protected_paths": ["${FROZEN_ROOT}/release"],
        "frozen_inputs": [
            {
                "role": "input",
                "path": "${FROZEN_ROOT}/input.bin",
                "sha256": hashlib.sha256(b"immutable").hexdigest(),
                "bytes": len(b"immutable"),
            }
        ],
        "jobs": jobs or [],
        "job_templates": job_templates or [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, repo_root, frozen_root, resolved_run_root


def load_test_plan(paths):
    manifest, repo_root, frozen_root, run_root = paths
    return load_workflow_plan(
        manifest,
        repo_root=repo_root,
        frozen_root=frozen_root,
        run_root=run_root,
        python_executable=sys.executable,
    )


def test_matrix_expansion_and_topological_order_are_deterministic(tmp_path):
    templates = [
        {
            "id": "train-k{{k}}-d{{D}}",
            "phase": "screen",
            "matrix": {"k": [8, 12], "D": [5, 8]},
            "needs": ["prepare"],
            "command": ["${PYTHON}", "-c", "pass"],
            "expected_outputs": [],
            "scientific": {"factors": {"path": "direct"}},
        }
    ]
    jobs = [
        {
            "id": "prepare",
            "phase": "prepare",
            "command": ["${PYTHON}", "-c", "pass"],
            "expected_outputs": [],
        }
    ]
    paths = write_manifest(tmp_path, jobs=jobs, job_templates=templates)
    first = load_test_plan(paths)
    second = load_test_plan(paths)
    assert [job.id for job in first.jobs] == [
        "prepare",
        "train-k12-d5",
        "train-k12-d8",
        "train-k8-d5",
        "train-k8-d8",
    ]
    assert first.digest == second.digest
    assert first.jobs_by_id["train-k8-d5"].scientific["factors"] == {
        "path": "direct",
        "k": 8,
        "D": 5,
    }


def test_rejects_run_root_overlapping_protected_release(tmp_path):
    paths = write_manifest(tmp_path)
    manifest, repo_root, frozen_root, _ = paths
    with pytest.raises(WorkflowError, match="overlap"):
        load_workflow_plan(
            manifest,
            repo_root=repo_root,
            frozen_root=frozen_root,
            run_root=frozen_root / "release" / "new-results",
        )


def test_rejects_run_root_inside_source_or_frozen_tree(tmp_path):
    manifest, repo_root, frozen_root, _ = write_manifest(tmp_path)
    for unsafe in (repo_root / "runs", frozen_root / "runs"):
        with pytest.raises(WorkflowError, match="must be disjoint"):
            load_workflow_plan(
                manifest,
                repo_root=repo_root,
                frozen_root=frozen_root,
                run_root=unsafe,
            )


def test_forbidden_input_paths_cannot_be_registered_in_commands(tmp_path):
    paths = write_manifest(
        tmp_path,
        jobs=[
            {
                "id": "inspect",
                "command": [
                    "${PYTHON}",
                    "-c",
                    "pass",
                    "${FROZEN_ROOT}/blind/private.npz",
                ],
                "expected_outputs": [],
            }
        ],
    )
    manifest_path = paths[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["forbidden_inputs"] = {
        "paths": ["${FROZEN_ROOT}/blind/private.npz"],
        "forbidden_report_fields": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(WorkflowError, match="command references forbidden input"):
        load_test_plan(paths)


@pytest.mark.parametrize("location", ["gate", "result"])
def test_forbidden_report_fields_cannot_be_read(tmp_path, location):
    output = "${RUN_ROOT}/report.json"
    job = {
        "id": "inspect",
        "command": ["${PYTHON}", "-c", "pass"],
        "expected_outputs": [output],
    }
    if location == "gate":
        job["json_gates"] = [
            {"path": output, "field": "metrics.blind_test.sigma", "gt": 0}
        ]
    else:
        job["result"] = {
            "path": output,
            "fields": {"sigma": "metrics.blind_test.sigma"},
        }
    paths = write_manifest(tmp_path, jobs=[job])
    manifest_path = paths[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["forbidden_inputs"] = {
        "paths": [],
        "forbidden_report_fields": ["blind_test"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(WorkflowError, match="forbidden report field"):
        load_test_plan(paths)


def test_rejects_dependency_cycle(tmp_path):
    jobs = [
        {
            "id": "a",
            "needs": ["b"],
            "command": ["${PYTHON}", "-c", "pass"],
            "expected_outputs": [],
        },
        {
            "id": "b",
            "needs": ["a"],
            "command": ["${PYTHON}", "-c", "pass"],
            "expected_outputs": [],
        },
    ]
    with pytest.raises(WorkflowError, match="cycle"):
        load_test_plan(write_manifest(tmp_path, jobs=jobs))


def test_verifies_frozen_input_size_and_hash(tmp_path):
    paths = write_manifest(
        tmp_path,
        jobs=[
            {
                "id": "noop",
                "command": ["${PYTHON}", "-c", "pass"],
                "expected_outputs": [],
            }
        ],
    )
    plan = load_test_plan(paths)
    assert (
        verify_frozen_inputs(plan)[0]["sha256"]
        == hashlib.sha256(b"immutable").hexdigest()
    )
    (paths[2] / "input.bin").write_bytes(b"changed")
    with pytest.raises(WorkflowError, match="size mismatch|hash mismatch"):
        verify_frozen_inputs(plan)


def test_cpu_job_is_content_verified_and_skipped_on_resume(tmp_path):
    output = "${RUN_ROOT}/jobs/write/result.txt"
    jobs = [
        {
            "id": "write",
            "command": [
                "${PYTHON}",
                "-c",
                "from pathlib import Path; Path(r'${RUN_ROOT}/jobs/write').mkdir(parents=True, exist_ok=True); Path(r'${RUN_ROOT}/jobs/write/result.txt').write_text('ok')",
            ],
            "expected_outputs": [output],
        }
    ]
    plan = load_test_plan(write_manifest(tmp_path, jobs=jobs))
    initialize_run_root(
        plan,
        frozen_inputs=verify_frozen_inputs(plan),
        source_identity={
            "commit": "test",
            "branch": "exp/test",
            "status_porcelain": "",
        },
    )
    first = run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
    second = run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
    assert first["succeeded"] == 1
    assert second["skipped"] == 1
    state = read_job_state(plan, "write")
    assert state["outputs"][0]["sha256"] == hashlib.sha256(b"ok").hexdigest()
    Path(state["outputs"][0]["path"]).write_text("tampered", encoding="utf-8")
    assert status_rows(plan)[0]["status"] == "invalidated"


def test_json_paths_support_list_indices_and_keys_containing_dots(tmp_path):
    output = "${RUN_ROOT}/jobs/replay/result.json"
    payload = {
        "scale_and_precision": [
            {"precision": "complex64"},
            {
                "precision": "complex128",
                "renormalized": {
                    "min_eigenvalue_weighted_quantiles": {"q0.0000": 0.03},
                    "abs_residual_weighted_quantiles": {"q0.9990": 0.006},
                },
            },
        ]
    }
    jobs = [
        {
            "id": "replay",
            "phase": "precision-replay",
            "command": [
                "${PYTHON}",
                "-c",
                (
                    "from pathlib import Path; "
                    "p=Path(r'${RUN_ROOT}/jobs/replay/result.json'); "
                    "p.parent.mkdir(parents=True, exist_ok=True); "
                    f"p.write_text({json.dumps(json.dumps(payload))})"
                ),
            ],
            "expected_outputs": [output],
            "json_gates": [
                {
                    "path": output,
                    "field": (
                        "scale_and_precision.1.renormalized."
                        "min_eigenvalue_weighted_quantiles.q0.0000"
                    ),
                    "gt": 0,
                }
            ],
            "result": {
                "path": output,
                "fields": {
                    "q0.9990": (
                        "scale_and_precision.1.renormalized."
                        "abs_residual_weighted_quantiles.q0.9990"
                    )
                },
            },
        }
    ]
    plan = load_test_plan(write_manifest(tmp_path, jobs=jobs))
    initialize_run_root(
        plan,
        frozen_inputs=verify_frozen_inputs(plan),
        source_identity={
            "commit": "test",
            "branch": "exp/test",
            "status_porcelain": "",
        },
    )

    counts = run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
    assert counts == {"succeeded": 1, "skipped": 0, "failed": 0, "blocked": 0}
    summary_dir = summarize_results(plan)
    summary = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["rows"][0]["q0.9990"] == 0.006


def test_campaign_lock_rejects_a_second_runner(tmp_path):
    paths = write_manifest(
        tmp_path,
        jobs=[
            {
                "id": "noop",
                "command": ["${PYTHON}", "-c", "pass"],
                "expected_outputs": [],
            }
        ],
    )
    plan = load_test_plan(paths)
    initialize_run_root(
        plan,
        frozen_inputs=verify_frozen_inputs(plan),
        source_identity={
            "commit": "test",
            "branch": "exp/test",
            "status_porcelain": "",
        },
    )

    with campaign_lock(plan):
        with pytest.raises(WorkflowError, match="another runner"):
            run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")


def test_bootstrap_lock_serializes_first_run_publication(tmp_path):
    plan = load_test_plan(
        write_manifest(
            tmp_path,
            jobs=[
                {
                    "id": "noop",
                    "command": ["${PYTHON}", "-c", "pass"],
                    "expected_outputs": [],
                }
            ],
        )
    )
    with bootstrap_lock(plan, tmp_path / "locks", timeout_seconds=0):
        with pytest.raises(WorkflowError, match="bootstrap lock"):
            with bootstrap_lock(plan, tmp_path / "locks", timeout_seconds=0):
                pass


def test_run_root_rejects_source_identity_drift(tmp_path):
    plan = load_test_plan(
        write_manifest(
            tmp_path,
            jobs=[
                {
                    "id": "noop",
                    "command": ["${PYTHON}", "-c", "pass"],
                    "expected_outputs": [],
                }
            ],
        )
    )
    frozen = verify_frozen_inputs(plan)
    source = {
        "commit": "a" * 40,
        "branch": "exp/test",
        "status_porcelain": " M file.py",
        "worktree_sha256": "1" * 64,
    }
    initialize_run_root(plan, frozen_inputs=frozen, source_identity=source)
    changed_source = {**source, "worktree_sha256": "2" * 64}
    with pytest.raises(WorkflowError, match="source identity"):
        initialize_run_root(
            plan,
            frozen_inputs=frozen,
            source_identity=changed_source,
        )


def test_cuda_identity_is_locked_for_a_campaign(tmp_path):
    plan = load_test_plan(
        write_manifest(
            tmp_path,
            jobs=[
                {
                    "id": "noop",
                    "command": ["${PYTHON}", "-c", "pass"],
                    "expected_outputs": [],
                }
            ],
        )
    )
    initialize_run_root(
        plan,
        frozen_inputs=verify_frozen_inputs(plan),
        source_identity={
            "commit": "test",
            "branch": "exp/test",
            "status_porcelain": "",
        },
    )
    cuda = {
        "torch": "2.7.1",
        "cuda": "12.8",
        "device": "test-gpu",
        "capability": [9, 0],
    }
    record_cuda_identity(plan, gpu_id="0", cuda=cuda)
    record_cuda_identity(plan, gpu_id="0", cuda=cuda)
    with pytest.raises(WorkflowError, match="CUDA/PyTorch identity"):
        record_cuda_identity(
            plan,
            gpu_id="0",
            cuda={**cuda, "device": "different-gpu"},
        )


def test_retry_appends_resume_checkpoint_only_after_failure(tmp_path):
    helper = tmp_path / "resume_helper.py"
    helper.write_text(
        """
from pathlib import Path
import sys
checkpoint = Path(sys.argv[1])
output = Path(sys.argv[2])
if '--resume-checkpoint' not in sys.argv:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text('checkpoint')
    raise SystemExit(75)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('resumed')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    jobs = [
        {
            "id": "resumable",
            "command": [
                "${PYTHON}",
                str(helper),
                "${RUN_ROOT}/jobs/resumable/checkpoint.pt",
                "${RUN_ROOT}/jobs/resumable/result.txt",
            ],
            "expected_outputs": ["${RUN_ROOT}/jobs/resumable/result.txt"],
            "resume": {"checkpoint": "${RUN_ROOT}/jobs/resumable/checkpoint.pt"},
            "retry": {"max_attempts": 2, "exit_codes": [75], "backoff_seconds": 0},
        }
    ]
    plan = load_test_plan(write_manifest(tmp_path, jobs=jobs))
    initialize_run_root(
        plan,
        frozen_inputs=verify_frozen_inputs(plan),
        source_identity={
            "commit": "test",
            "branch": "exp/test",
            "status_porcelain": "",
        },
    )
    counts = run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
    state = read_job_state(plan, "resumable")
    assert counts["succeeded"] == 1
    assert len(state["attempts"]) == 2
    assert "--resume-checkpoint" not in state["attempts"][0]["command"]
    assert "--resume-checkpoint" in state["attempts"][1]["command"]


def test_encoded_nested_signal_is_retryable_but_plain_failure_is_not(tmp_path):
    plan = load_test_plan(
        write_manifest(
            tmp_path,
            jobs=[
                {
                    "id": "signal",
                    "command": ["${PYTHON}", "-c", "pass"],
                    "expected_outputs": [],
                    "retry": {"signals": [4, 5, 11], "log_patterns": []},
                }
            ],
        )
    )
    job = plan.jobs_by_id["signal"]
    assert _retryable(job, 139, "") is True
    assert _retryable(job, -11, "") is True
    assert _retryable(job, 1, "trainer failed in round") is False


def test_nonretryable_failure_stays_terminal_across_invocations(tmp_path):
    jobs = [
        {
            "id": "terminal",
            "command": ["${PYTHON}", "-c", "raise SystemExit(2)"],
            "expected_outputs": [],
            "retry": {"max_attempts": 3, "backoff_seconds": 0},
        }
    ]
    plan = load_test_plan(write_manifest(tmp_path, jobs=jobs))
    initialize_run_root(
        plan,
        frozen_inputs=verify_frozen_inputs(plan),
        source_identity={
            "commit": "test",
            "branch": "exp/test",
            "status_porcelain": "",
        },
    )
    with pytest.raises(WorkflowError, match="failed"):
        run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
    first_state = read_job_state(plan, "terminal")
    assert first_state["retryable"] is False
    assert len(first_state["attempts"]) == 1

    with pytest.raises(WorkflowError, match="terminal failure"):
        run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
    second_state = read_job_state(plan, "terminal")
    assert len(second_state["attempts"]) == 1


def test_retryable_failure_becomes_terminal_when_attempt_budget_is_exhausted(
    tmp_path,
):
    jobs = [
        {
            "id": "exhausted",
            "command": ["${PYTHON}", "-c", "raise SystemExit(75)"],
            "expected_outputs": [],
            "retry": {
                "max_attempts": 2,
                "exit_codes": [75],
                "backoff_seconds": 0,
            },
        }
    ]
    plan = load_test_plan(write_manifest(tmp_path, jobs=jobs))
    initialize_run_root(
        plan,
        frozen_inputs=verify_frozen_inputs(plan),
        source_identity={
            "commit": "test",
            "branch": "exp/test",
            "status_porcelain": "",
        },
    )
    with pytest.raises(WorkflowError, match="failed"):
        run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
    first_state = read_job_state(plan, "exhausted")
    assert len(first_state["attempts"]) == 2
    assert first_state["retryable"] is False
    assert first_state["failure_reason"] == "attempt_budget_exhausted"

    with pytest.raises(WorkflowError, match="terminal failure"):
        run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
    second_state = read_job_state(plan, "exhausted")
    assert len(second_state["attempts"]) == 2


@pytest.mark.parametrize("complete_output", [False, True])
def test_exhausted_running_attempt_is_finalized_after_runner_interruption(
    tmp_path, complete_output
):
    jobs = [
        {
            "id": "interrupted",
            "command": ["${PYTHON}", "-c", "raise AssertionError('must not rerun')"],
            "expected_outputs": ["${RUN_ROOT}/result.txt"],
            "retry": {"max_attempts": 1},
        }
    ]
    plan = load_test_plan(write_manifest(tmp_path, jobs=jobs))
    initialize_run_root(
        plan,
        frozen_inputs=verify_frozen_inputs(plan),
        source_identity={
            "commit": "test",
            "branch": "exp/test",
            "status_porcelain": "",
        },
    )
    if complete_output:
        output = plan.run_root / "result.txt"
        output.write_text("complete", encoding="utf-8")
    job = plan.jobs_by_id["interrupted"]
    atomic_write_json(
        job_state_path(plan, job.id),
        {
            "schema": "gcicy-experiment-job-state-v1",
            "campaign_id": plan.campaign_id,
            "plan_sha256": plan.digest,
            "job_id": job.id,
            "job_digest": job.digest,
            "status": "running",
            "attempts": [{"attempt": 1}],
        },
    )

    if complete_output:
        counts = run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
        assert counts["succeeded"] == 1
        state = read_job_state(plan, job.id)
        assert state["status"] == "succeeded"
        assert state["recovered_after_runner_interruption"] is True
    else:
        with pytest.raises(WorkflowError, match="exhausted its attempt budget"):
            run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
        state = read_job_state(plan, job.id)
        assert state["status"] == "failed"
        assert state["retryable"] is False
        assert state["failure_reason"] == "attempt_budget_exhausted"


def test_aggregation_is_create_only_and_records_result_fields(tmp_path):
    jobs = [
        {
            "id": "measure",
            "phase": "audit",
            "command": [
                "${PYTHON}",
                "-c",
                "from pathlib import Path; p=Path(r'${RUN_ROOT}/jobs/measure/result.json'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{\"metrics\":{\"sigma\":0.1}}')",
            ],
            "expected_outputs": ["${RUN_ROOT}/jobs/measure/result.json"],
            "scientific": {"factors": {"k": 8, "D": 5}},
            "result": {
                "path": "${RUN_ROOT}/jobs/measure/result.json",
                "fields": {"sigma": "metrics.sigma"},
            },
        }
    ]
    plan = load_test_plan(write_manifest(tmp_path, jobs=jobs))
    initialize_run_root(
        plan,
        frozen_inputs=verify_frozen_inputs(plan),
        source_identity={
            "commit": "test",
            "branch": "exp/test",
            "status_porcelain": "",
        },
    )
    run_workflow(plan, gpu_id="0", lock_root=tmp_path / "locks")
    output = summarize_results(plan)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    assert summary["rows"] == [
        {"job_id": "measure", "phase": "audit", "k": 8, "D": 5, "sigma": 0.1}
    ]


def test_aggregation_requires_an_initialized_run_root(tmp_path):
    plan = load_test_plan(
        write_manifest(
            tmp_path,
            jobs=[
                {
                    "id": "measure",
                    "command": ["${PYTHON}", "-c", "pass"],
                    "expected_outputs": [],
                    "result": {"path": "${RUN_ROOT}/missing.json", "fields": {}},
                }
            ],
        )
    )
    with pytest.raises(WorkflowError, match="not initialized"):
        summarize_results(plan, allow_incomplete=True)


def test_checked_in_x21_manifest_expands_all_registered_comparisons(tmp_path):
    from scripts.run_gcicy_tn_study_arm import parse_args as parse_study_arm_args

    repo_root = Path(__file__).resolve().parents[1]
    plan = load_workflow_plan(
        repo_root
        / "experiments"
        / "manifests"
        / "x21_kd_progressive_joint_paths_v1.json",
        repo_root=repo_root,
        frozen_root=tmp_path / "frozen",
        run_root=tmp_path / "runs" / "x21",
        python_executable=sys.executable,
    )
    phase_counts: dict[str, int] = {}
    for job in plan.jobs:
        phase_counts[job.phase] = phase_counts.get(job.phase, 0) + 1
    assert len(plan.jobs) == 279
    assert phase_counts == {
        "prepare": 2,
        "capacity-grid": 252,
        "resource-preflight": 1,
        "progressive-paths": 12,
        "parameter-scope": 9,
        "optimizer-paths": 3,
    }
    assert sum(job.result is not None for job in plan.jobs) == 60
    assert (
        "--first-kappa-source",
        "initial_model",
    ) in tuple(
        zip(
            plan.jobs_by_id["path-k-jump-k16-d12-r1"].command,
            plan.jobs_by_id["path-k-jump-k16-d12-r1"].command[1:],
        )
    )
    assert (
        "--first-kappa-source",
        "saved_model",
    ) in tuple(
        zip(
            plan.jobs_by_id["path-endpoint-control-k16-d12-r1"].command,
            plan.jobs_by_id["path-endpoint-control-k16-d12-r1"].command[1:],
        )
    )
    assert (
        "--first-kappa-source",
        "initial_model",
    ) in tuple(
        zip(
            plan.jobs_by_id["path-endpoint-rekappa-control-k16-d12-r1"].command,
            plan.jobs_by_id["path-endpoint-rekappa-control-k16-d12-r1"].command[1:],
        )
    )
    assert (
        "--train-physical-dictionary"
        in plan.jobs_by_id["joint-epochzero-k16-d12-r1"].command
    )
    assert (
        "--train-physical-dictionary"
        not in plan.jobs_by_id["joint-cores-control-k16-d12-r1"].command
    )
    targeted = [
        job for job in plan.jobs if job.id.startswith(("path-", "joint-", "optimizer-"))
    ]
    assert len(targeted) == 24
    for job in targeted:
        parsed = parse_study_arm_args(list(job.command[2:]))
        assert parsed.target_k == 16
        assert parsed.target_d == 12
    assert selection_requires_gpu(
        plan,
        phases=None,
        only=["path-k-jump-k16-d12-r1"],
    )
    assert not selection_requires_gpu(
        plan,
        phases=None,
        only=["direct-init-bond-k8-d8-r1"],
    )
    verify_cuda_requirements(plan, {"torch": "2.7.1+cu128"})
    with pytest.raises(WorkflowError, match="PyTorch version"):
        verify_cuda_requirements(plan, {"torch": "2.8.0"})
