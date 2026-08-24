#!/usr/bin/env python3
"""Compare matched compiled-tree relaxation under audited per-arm policies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_generic_quintic_h4_architecture_arms import (  # noqa: E402
    build_checkpoint_model,
    infer_architecture,
)
from scripts.refine_generic_quintic_compiled_tree_native_gn import (  # noqa: E402
    load_fixed_split_indices,
    load_disjoint_splits,
    load_excluded_indices,
)
from scripts.train_generic_quintic_adaptive_tree_rank import (  # noqa: E402
    make_batch_plan,
    state_to_cpu,
    train_arm,
    zero_nonpositive,
)
from scripts.train_generic_quintic_compiled_tree import (  # noqa: E402
    metric_row,
    sha256_file,
    whiten_dataset,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    make_dataset,
    paired_improvement,
    tail_guard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--train-points", type=Path, required=True)
    parser.add_argument("--train-pullbacks", type=Path, required=True)
    parser.add_argument("--selection-points", type=Path, required=True)
    parser.add_argument("--selection-pullbacks", type=Path, required=True)
    parser.add_argument("--confirmation-points", type=Path)
    parser.add_argument("--confirmation-pullbacks", type=Path)
    parser.add_argument("--development-evaluation-points", type=Path)
    parser.add_argument("--development-evaluation-pullbacks", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-indices-file",
        type=Path,
        nargs="*",
        default=(),
    )
    parser.add_argument("--fixed-indices-file", type=Path)
    parser.add_argument("--fixed-batch-plan-file", type=Path)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument(
        "--scheduler",
        choices=("constant", "cosine"),
        default="cosine",
        help=(
            "Common scheduler used by both arms unless a role-specific "
            "scheduler override is supplied."
        ),
    )
    parser.add_argument(
        "--control-scheduler",
        choices=("constant", "cosine"),
        default=None,
        help="Optional control-arm override for --scheduler.",
    )
    parser.add_argument(
        "--candidate-scheduler",
        choices=("constant", "cosine"),
        default=None,
        help="Optional candidate-arm override for --scheduler.",
    )
    parser.add_argument(
        "--control-trainable-scope",
        choices=("internal", "all"),
        default="all",
    )
    parser.add_argument(
        "--candidate-trainable-scope",
        choices=("internal", "all"),
        default="all",
    )
    parser.add_argument(
        "--require-equal-total-parameters",
        action="store_true",
        help=(
            "Reject checkpoints with unequal total parameter counts. This is "
            "automatically required when the two role-specific training "
            "policies differ."
        ),
    )
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--selection-size", type=int, default=5_000)
    parser.add_argument("--confirmation-size", type=int, default=5_000)
    parser.add_argument("--development-evaluation-size", type=int, default=5_000)
    parser.add_argument("--stochastic-batch-size", type=int, default=2_048)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--minimum-relative-sigma-gain",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--minimum-relative-chi-gain",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.02,
    )
    parser.add_argument("--train-chunk-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=202607477)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help=(
            "Optional fixed split seed. When omitted, preserve the historical "
            "behavior and reuse --seed for data sampling."
        ),
    )
    parser.add_argument(
        "--batch-plan-seed",
        type=int,
        default=None,
        help=(
            "Optional explicit minibatch-plan seed. When omitted, preserve the "
            "historical --seed + 17 behavior."
        ),
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Skip the independent confirmation split.",
    )
    parser.add_argument(
        "--development-evaluation",
        action="store_true",
        help=(
            "With --development-only, evaluate paired confidence intervals on "
            "a preregistered development split without using it for checkpoint "
            "selection."
        ),
    )
    return parser.parse_args()


def checkpoint_contract(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    configuration = payload["configuration"]
    return (
        np.asarray(configuration["exponents"], dtype=np.int64),
        np.asarray(configuration["whitening"], dtype=np.complex128),
        str(configuration["precision"]),
    )


def real_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
    )


def role_training_policies(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """Resolve backward-compatible common settings into per-arm policies."""

    return {
        "control": {
            "trainable_scope": str(args.control_trainable_scope),
            "scheduler": str(args.control_scheduler or args.scheduler),
        },
        "candidate": {
            "trainable_scope": str(args.candidate_trainable_scope),
            "scheduler": str(args.candidate_scheduler or args.scheduler),
        },
    }


def configure_trainable_scope(
    model: torch.nn.Module,
    scope: str,
) -> list[str]:
    """Apply the registered parameter scope and return its exact tensor names."""

    if scope not in {"internal", "all"}:
        raise ValueError("trainable scope must be internal or all")
    trainable_names = []
    for name, parameter in model.named_parameters():
        trainable = scope == "all" or name.startswith("internal_tensors.")
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError(f"trainable scope {scope} selected no parameters")
    return trainable_names


def named_real_parameter_count(
    model: torch.nn.Module,
    names: set[str],
) -> int:
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for name, parameter in model.named_parameters()
        if name in names
    )


def batch_plan_digest(batch_plan: np.ndarray) -> str:
    """Hash an array including its dtype and shape, independent of its path."""

    contiguous = np.ascontiguousarray(batch_plan)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def assert_batch_plan_unchanged(
    batch_plan: np.ndarray,
    reference: np.ndarray,
    reference_sha256: str,
) -> None:
    """Fail closed if an arm receives or mutates a different minibatch plan."""

    if (
        batch_plan.dtype != reference.dtype
        or batch_plan.shape != reference.shape
        or not np.array_equal(batch_plan, reference)
        or batch_plan_digest(batch_plan) != reference_sha256
    ):
        raise RuntimeError("paired arms did not use the same fixed batch plan")


def total_parameter_parity_audit(
    *,
    control: int,
    candidate: int,
    required: bool,
) -> dict[str, Any]:
    equal = control == candidate
    audit = {
        "required": bool(required),
        "equal": equal,
        "control_total_real_parameters": int(control),
        "candidate_total_real_parameters": int(candidate),
        "candidate_minus_control": int(candidate - control),
    }
    if required and not equal:
        raise ValueError(
            "paired scope/scheduler comparison requires equal total parameters"
        )
    return audit


def parameter_audit(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
    *,
    trainable_names: set[str] | None = None,
) -> dict[str, Any]:
    rows = []
    total_changed_real = 0
    maximum_absolute = 0.0
    for name, parameter in model.named_parameters():
        before = initial[name].to(parameter.device)
        delta = parameter.detach() - before
        absolute = torch.abs(delta)
        max_absolute = float(torch.max(absolute)) if absolute.numel() else 0.0
        delta_norm = float(torch.linalg.vector_norm(delta.reshape(-1)))
        before_norm = float(torch.linalg.vector_norm(before.reshape(-1)))
        threshold = 32.0 * torch.finfo(parameter.real.dtype).eps
        changed_scalars = int(torch.count_nonzero(absolute > threshold))
        changed_real = changed_scalars * (2 if parameter.is_complex() else 1)
        trainable = trainable_names is None or name in trainable_names
        exactly_unchanged = bool(torch.equal(parameter.detach(), before))
        total_changed_real += changed_real
        maximum_absolute = max(maximum_absolute, max_absolute)
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "real_parameters": (
                    parameter.numel() * (2 if parameter.is_complex() else 1)
                ),
                "trainable": trainable,
                "exactly_unchanged": exactly_unchanged,
                "changed_real_parameters": changed_real,
                "max_absolute_change": max_absolute,
                "l2_change": delta_norm,
                "relative_l2_change": delta_norm / max(before_norm, 1.0e-30),
            }
        )
    audit = {
        "total_changed_real_parameters": total_changed_real,
        "maximum_absolute_change": maximum_absolute,
        "tensors": rows,
    }
    frozen = [row for row in rows if not row["trainable"]]
    audit["frozen_tensor_count"] = len(frozen)
    audit["frozen_real_parameters"] = sum(row["real_parameters"] for row in frozen)
    audit["all_frozen_parameters_exactly_unchanged"] = all(
        row["exactly_unchanged"] for row in frozen
    )
    return audit


def assert_frozen_parameters_unchanged(audit: dict[str, Any]) -> None:
    if not audit["all_frozen_parameters_exactly_unchanged"]:
        changed = [
            row["name"]
            for row in audit["tensors"]
            if not row["trainable"] and not row["exactly_unchanged"]
        ]
        raise RuntimeError(f"frozen parameters changed: {changed}")


def relative_gain(start: float, final: float) -> float:
    return (start - final) / start


def adjudication_winner(
    *,
    selection_passes: bool,
    confirmation_passes: bool | None,
) -> str:
    """Choose from selection, unless an independent confirmation was opened.

    Development-evaluation is intentionally absent from this interface: it is
    post-training paired-CI evidence and cannot affect checkpoints or winner.
    """

    passes = (
        confirmation_passes if confirmation_passes is not None else selection_passes
    )
    return "candidate" if passes else "control"


def model_payload(
    parent: dict[str, Any],
    state: dict[str, torch.Tensor],
    *,
    role: str,
    source_path: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(parent)
    result["state_dict"] = state
    result["teacher_runtime_dependency"] = False
    result["joint_relaxation_role"] = role
    result["joint_relaxation_source_sha256"] = sha256_file(source_path)
    result["joint_relaxation_parameter_audit"] = audit
    return result


def main() -> None:
    args = parse_args()
    positive = (
        args.steps,
        args.learning_rate,
        args.eval_every,
        args.train_size,
        args.selection_size,
        args.confirmation_size,
        args.development_evaluation_size,
        args.stochastic_batch_size,
        args.gradient_clip_norm,
        args.train_chunk_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("joint-relaxation sizes must be positive")
    if not 0 <= args.minimum_relative_sigma_gain < 1:
        raise ValueError("minimum sigma gain must lie in [0, 1)")
    if not 0 <= args.minimum_relative_chi_gain < 1:
        raise ValueError("minimum chi gain must lie in [0, 1)")
    if args.maximum_selection_tail_relative_degradation < 0:
        raise ValueError("tail degradation allowance must be nonnegative")
    if args.data_seed is not None and args.data_seed < 0:
        raise ValueError("data seed must be nonnegative")
    if args.batch_plan_seed is not None and args.batch_plan_seed < 0:
        raise ValueError("batch-plan seed must be nonnegative")
    if args.development_evaluation and not args.development_only:
        raise ValueError("development evaluation requires --development-only")
    if args.development_only:
        if (
            args.confirmation_points is not None
            or args.confirmation_pullbacks is not None
        ):
            raise ValueError(
                "development-only runs must not receive confirmation paths"
            )
        if args.development_evaluation and (
            args.development_evaluation_points is None
            or args.development_evaluation_pullbacks is None
        ):
            raise ValueError("development-evaluation paths are required")
    elif args.confirmation_points is None or args.confirmation_pullbacks is None:
        raise ValueError("confirmation paths are required outside development-only")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    training_policies = role_training_policies(args)
    require_equal_total_parameters = bool(
        args.require_equal_total_parameters
        or training_policies["control"] != training_policies["candidate"]
    )

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a joint-relaxation run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    started = time.perf_counter()

    try:
        control_path = args.control_checkpoint.expanduser().resolve()
        candidate_path = args.candidate_checkpoint.expanduser().resolve()
        control_payload = torch.load(
            control_path,
            map_location="cpu",
            weights_only=False,
        )
        candidate_payload = torch.load(
            candidate_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            infer_architecture(control_payload) != "compiled-tree"
            or infer_architecture(candidate_payload) != "compiled-tree"
        ):
            raise ValueError("both checkpoints must be compiled trees")
        if bool(
            control_payload.get("teacher_runtime_dependency", False)
            or candidate_payload.get("teacher_runtime_dependency", False)
        ):
            raise ValueError("joint relaxation requires teacher-free checkpoints")
        control_contract = checkpoint_contract(control_payload)
        candidate_contract = checkpoint_contract(candidate_payload)
        if (
            control_contract[2] != candidate_contract[2]
            or not np.array_equal(control_contract[0], candidate_contract[0])
            or not np.allclose(
                control_contract[1],
                candidate_contract[1],
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise ValueError("control and candidate feature contracts differ")
        exponents, whitening, precision = control_contract
        dtype = torch.complex64 if precision == "complex64" else torch.complex128

        specifications = {
            "fit": (
                args.train_points,
                args.train_pullbacks,
                args.train_size,
            ),
            "selection": (
                args.selection_points,
                args.selection_pullbacks,
                args.selection_size,
            ),
        }
        if args.development_evaluation:
            specifications["development_evaluation"] = (
                args.development_evaluation_points,
                args.development_evaluation_pullbacks,
                args.development_evaluation_size,
            )
        elif not args.development_only:
            specifications["confirmation"] = (
                args.confirmation_points,
                args.confirmation_pullbacks,
                args.confirmation_size,
            )
        excluded = load_excluded_indices(args.exclude_indices_file)
        arrays, indices = load_disjoint_splits(
            specifications,
            seed=args.seed if args.data_seed is None else args.data_seed,
            exclusions=excluded,
            fixed_indices=load_fixed_split_indices(args.fixed_indices_file),
        )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **indices)
        datasets = {
            name: whiten_dataset(
                make_dataset(
                    rows,
                    exponents,
                    feature_batch_size=args.feature_batch_size,
                    complex_dtype=dtype,
                    device=device,
                ),
                whitening,
            )
            for name, rows in arrays.items()
        }
        if args.fixed_batch_plan_file is None:
            batch_plan = make_batch_plan(
                count=datasets["fit"]["count"],
                batch_size=args.stochastic_batch_size,
                steps=args.steps,
                seed=(
                    args.seed + 17
                    if args.batch_plan_seed is None
                    else args.batch_plan_seed
                ),
            )
        else:
            raw_plan = np.load(
                args.fixed_batch_plan_file.expanduser().resolve(),
                allow_pickle=False,
            )
            batch_plan = np.asarray(raw_plan)
            expected_shape = (
                args.steps,
                min(datasets["fit"]["count"], args.stochastic_batch_size),
            )
            if batch_plan.shape != expected_shape or not np.issubdtype(
                batch_plan.dtype, np.integer
            ):
                raise ValueError("fixed batch plan has the wrong shape or dtype")
            batch_plan = np.asarray(batch_plan, dtype=np.int64)
            if np.any((batch_plan < 0) | (batch_plan >= datasets["fit"]["count"])):
                raise ValueError("fixed batch plan contains out-of-range rows")
            if any(len(np.unique(row)) != len(row) for row in batch_plan):
                raise ValueError("fixed batch plan repeats a row within a step")
        batch_plan_path = output_dir / "batch_plan.npy"
        np.save(batch_plan_path, batch_plan, allow_pickle=False)
        batch_plan_reference = np.array(batch_plan, copy=True)
        batch_plan_reference.setflags(write=False)
        fixed_batch_plan_sha256 = batch_plan_digest(batch_plan_reference)

        results: dict[str, Any] = {}
        payloads = {
            "control": (control_payload, control_path),
            "candidate": (candidate_payload, candidate_path),
        }
        for role, (payload, source_path) in payloads.items():
            policy = training_policies[role]
            write_json(
                status_path,
                {"state": "running", "phase": "training", "role": role},
            )
            model = build_checkpoint_model(payload, device=device)
            trainable_names = configure_trainable_scope(
                model,
                policy["trainable_scope"],
            )
            trainable_name_set = set(trainable_names)
            initial_state = state_to_cpu(model)
            count = real_parameter_count(model)
            trainable_count = named_real_parameter_count(
                model,
                trainable_name_set,
            )
            if role == "candidate":
                total_parameter_parity_audit(
                    control=results["control"]["total_real_parameters"],
                    candidate=count,
                    required=require_equal_total_parameters,
                )
            assert_batch_plan_unchanged(
                batch_plan,
                batch_plan_reference,
                fixed_batch_plan_sha256,
            )
            result = train_arm(
                model,
                training=datasets["fit"],
                selection=datasets["selection"],
                batch_plan=batch_plan,
                learning_rate=args.learning_rate,
                eval_every=args.eval_every,
                train_chunk_size=args.train_chunk_size,
                eval_batch_size=args.eval_batch_size,
                gradient_clip_norm=args.gradient_clip_norm,
                maximum_tail_degradation=(
                    args.maximum_selection_tail_relative_degradation
                ),
                scheduler=policy["scheduler"],
            )
            assert_batch_plan_unchanged(
                batch_plan,
                batch_plan_reference,
                fixed_batch_plan_sha256,
            )
            audit = parameter_audit(
                result["model"],
                initial_state,
                trainable_names=trainable_name_set,
            )
            assert_frozen_parameters_unchanged(audit)
            polished = model_payload(
                payload,
                result["best_state"],
                role=role,
                source_path=source_path,
                audit=audit,
            )
            checkpoint = output_dir / f"{role}_polished.pt"
            torch.save(polished, checkpoint)
            results[role] = {
                "model": result["model"],
                "payload": polished,
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256_file(checkpoint),
                "total_real_parameters": count,
                "trainable_real_parameters": trainable_count,
                "new_real_parameters_vs_control": 0,
                "training_policy": policy,
                "trainable_tensor_names": trainable_names,
                "fixed_batch_plan_sha256": fixed_batch_plan_sha256,
                "initial_statistics": result["initial_statistics"],
                "initial_tail": result["initial_tail"],
                "best_statistics": result["best_statistics"],
                "best_tail": result["best_tail"],
                "history": result["history"],
                "parameter_audit": audit,
            }
        results["candidate"]["new_real_parameters_vs_control"] = (
            results["candidate"]["total_real_parameters"]
            - results["control"]["total_real_parameters"]
        )
        parameter_parity = total_parameter_parity_audit(
            control=results["control"]["total_real_parameters"],
            candidate=results["candidate"]["total_real_parameters"],
            required=require_equal_total_parameters,
        )

        control_selection = results["control"]["best_statistics"]
        candidate_selection = results["candidate"]["best_statistics"]
        sigma_gain = relative_gain(
            control_selection["sigma_official_formula"],
            candidate_selection["sigma_official_formula"],
        )
        chi_gain = relative_gain(
            control_selection["weighted_rms_abs_residual"],
            candidate_selection["weighted_rms_abs_residual"],
        )
        selection_passes = bool(
            sigma_gain >= args.minimum_relative_sigma_gain
            and chi_gain >= args.minimum_relative_chi_gain
            and zero_nonpositive(candidate_selection)
            and tail_guard(
                results["candidate"]["best_tail"],
                results["control"]["best_tail"],
                relative_degradation=(args.maximum_selection_tail_relative_degradation),
            )
        )

        confirmation_report = None
        development_evaluation_report = None
        confirmation_passes = None
        evaluation_split = None
        if args.development_evaluation:
            evaluation_split = "development_evaluation"
        elif not args.development_only:
            evaluation_split = "confirmation"
        if evaluation_split is not None:
            write_json(
                status_path,
                {"state": "running", "phase": evaluation_split},
            )
            evaluation_rows = {}
            ratios = {}
            for role in ("control", "candidate"):
                statistics, ratio, tail = metric_row(
                    results[role]["model"],
                    datasets[evaluation_split],
                    chunk_size=args.eval_batch_size,
                )
                evaluation_rows[role] = {
                    "statistics": statistics,
                    "tail": tail,
                }
                ratios[role] = ratio
            paired = paired_improvement(
                ratios["control"],
                ratios["candidate"],
                datasets[evaluation_split]["weights_numpy"],
            )
            control_evaluation = evaluation_rows["control"]["statistics"]
            candidate_evaluation = evaluation_rows["candidate"]["statistics"]
            evaluation_passes = bool(
                selection_passes
                and candidate_evaluation["sigma_official_formula"]
                < control_evaluation["sigma_official_formula"]
                and candidate_evaluation["weighted_rms_abs_residual"]
                < control_evaluation["weighted_rms_abs_residual"]
                and paired["e2"]["ci95_low"] > 0
                and paired["sigma"]["ci95_low"] > 0
                and zero_nonpositive(candidate_evaluation)
                and tail_guard(
                    evaluation_rows["candidate"]["tail"],
                    evaluation_rows["control"]["tail"],
                    relative_degradation=(
                        args.maximum_selection_tail_relative_degradation
                    ),
                )
            )
            evaluation_report = {
                **evaluation_rows,
                "paired_improvement": paired,
                "passes": evaluation_passes,
                "selection_or_checkpoint_role": "none",
            }
            if evaluation_split == "confirmation":
                confirmation_passes = evaluation_passes
                confirmation_report = evaluation_report
            else:
                development_evaluation_report = evaluation_report

        winner = adjudication_winner(
            selection_passes=selection_passes,
            confirmation_passes=confirmation_passes,
        )
        winner_path = output_dir / (
            "accepted_candidate.pt" if winner == "candidate" else "control_retained.pt"
        )
        torch.save(results[winner]["payload"], winner_path)

        serializable_results = {
            role: {
                key: value
                for key, value in row.items()
                if key not in {"model", "payload"}
            }
            for role, row in results.items()
        }
        report = {
            "schema": "generic-quintic-tree-joint-relaxation-v1",
            "contract": {
                "precision": control_contract[2],
            },
            "configuration": {
                **vars(args),
                "control_checkpoint": str(control_path),
                "candidate_checkpoint": str(candidate_path),
                "output_dir": str(output_dir),
            },
            "training_policies": training_policies,
            "total_parameter_parity": parameter_parity,
            "source_checkpoint_sha256": {
                "control": sha256_file(control_path),
                "candidate": sha256_file(candidate_path),
            },
            "data": {
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "fixed_indices_file": (
                    None
                    if args.fixed_indices_file is None
                    else str(args.fixed_indices_file.expanduser().resolve())
                ),
                "fixed_indices_file_sha256": (
                    None
                    if args.fixed_indices_file is None
                    else sha256_file(args.fixed_indices_file.expanduser().resolve())
                ),
                "input_sha256": {
                    "train_points": sha256_file(
                        args.train_points.expanduser().resolve()
                    ),
                    "train_pullbacks": sha256_file(
                        args.train_pullbacks.expanduser().resolve()
                    ),
                    "selection_points": sha256_file(
                        args.selection_points.expanduser().resolve()
                    ),
                    "selection_pullbacks": sha256_file(
                        args.selection_pullbacks.expanduser().resolve()
                    ),
                    "development_evaluation_points": (
                        None
                        if args.development_evaluation_points is None
                        else sha256_file(
                            args.development_evaluation_points.expanduser().resolve()
                        )
                    ),
                    "development_evaluation_pullbacks": (
                        None
                        if args.development_evaluation_pullbacks is None
                        else sha256_file(
                            args.development_evaluation_pullbacks.expanduser().resolve()
                        )
                    ),
                    "confirmation_points": (
                        None
                        if args.confirmation_points is None
                        else sha256_file(
                            args.confirmation_points.expanduser().resolve()
                        )
                    ),
                    "confirmation_pullbacks": (
                        None
                        if args.confirmation_pullbacks is None
                        else sha256_file(
                            args.confirmation_pullbacks.expanduser().resolve()
                        )
                    ),
                },
                "counts": {name: int(len(rows)) for name, rows in indices.items()},
                "data_seed": (args.seed if args.data_seed is None else args.data_seed),
            },
            "batch_plan": {
                "path": str(batch_plan_path),
                "sha256": sha256_file(batch_plan_path),
                "array_sha256": fixed_batch_plan_sha256,
                "same_for_both_arms": all(
                    row["fixed_batch_plan_sha256"] == fixed_batch_plan_sha256
                    for row in results.values()
                ),
                "fixed_source_sha256": (
                    None
                    if args.fixed_batch_plan_file is None
                    else sha256_file(args.fixed_batch_plan_file.expanduser().resolve())
                ),
                "seed": (
                    args.seed + 17
                    if args.batch_plan_seed is None
                    else args.batch_plan_seed
                ),
                "steps": args.steps,
                "batch_size": min(
                    args.stochastic_batch_size,
                    datasets["fit"]["count"],
                ),
            },
            "results": serializable_results,
            "selection_comparison": {
                "sigma_gain": sigma_gain,
                "chi_gain": chi_gain,
                "passes": selection_passes,
            },
            "development_evaluation": development_evaluation_report,
            "confirmation": confirmation_report,
            "confirmation_passes": confirmation_passes,
            "winner": winner,
            "checkpoint": str(winner_path),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "winner": winner,
                "checkpoint": str(winner_path),
                "wall_seconds": report["wall_seconds"],
            },
        )
        print(
            json.dumps(
                {
                    "state": "complete",
                    "winner": winner,
                    "selection_comparison": report["selection_comparison"],
                    "confirmation_passes": confirmation_passes,
                    "checkpoint": str(winner_path),
                    "wall_seconds": report["wall_seconds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
