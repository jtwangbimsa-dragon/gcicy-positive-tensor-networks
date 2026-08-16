#!/usr/bin/env python3
"""Compare matched full-model relaxation before and after tree-rank growth."""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--confirmation-points", type=Path, required=True)
    parser.add_argument("--confirmation-pullbacks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-indices-file",
        type=Path,
        nargs="*",
        default=(),
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--selection-size", type=int, default=5_000)
    parser.add_argument("--confirmation-size", type=int, default=5_000)
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
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Skip the independent confirmation split.",
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


def parameter_audit(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
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
        total_changed_real += changed_real
        maximum_absolute = max(maximum_absolute, max_absolute)
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "real_parameters": (
                    parameter.numel() * (2 if parameter.is_complex() else 1)
                ),
                "changed_real_parameters": changed_real,
                "max_absolute_change": max_absolute,
                "l2_change": delta_norm,
                "relative_l2_change": delta_norm / max(before_norm, 1.0e-30),
            }
        )
    return {
        "total_changed_real_parameters": total_changed_real,
        "maximum_absolute_change": maximum_absolute,
        "tensors": rows,
    }


def relative_gain(start: float, final: float) -> float:
    return (start - final) / start


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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

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
        dtype = (
            torch.complex64
            if precision == "complex64"
            else torch.complex128
        )

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
        if not args.development_only:
            specifications["confirmation"] = (
                args.confirmation_points,
                args.confirmation_pullbacks,
                args.confirmation_size,
            )
        excluded = load_excluded_indices(args.exclude_indices_file)
        arrays, indices = load_disjoint_splits(
            specifications,
            seed=args.seed,
            exclusions=excluded,
        )
        np.savez_compressed(output_dir / "data_indices.npz", **indices)
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
        batch_plan = make_batch_plan(
            count=datasets["fit"]["count"],
            batch_size=args.stochastic_batch_size,
            steps=args.steps,
            seed=args.seed + 17,
        )

        results: dict[str, Any] = {}
        payloads = {
            "control": (control_payload, control_path),
            "candidate": (candidate_payload, candidate_path),
        }
        for role, (payload, source_path) in payloads.items():
            write_json(
                status_path,
                {"state": "running", "phase": "training", "role": role},
            )
            model = build_checkpoint_model(payload, device=device)
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            initial_state = state_to_cpu(model)
            count = real_parameter_count(model)
            trainable_names = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
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
            )
            audit = parameter_audit(result["model"], initial_state)
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
                "total_real_parameters": count,
                "trainable_real_parameters": count,
                "new_real_parameters_vs_control": 0,
                "trainable_tensor_names": trainable_names,
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
                relative_degradation=(
                    args.maximum_selection_tail_relative_degradation
                ),
            )
        )

        confirmation_report = None
        confirmation_passes = None
        if not args.development_only:
            write_json(
                status_path,
                {"state": "running", "phase": "confirmation"},
            )
            confirmation_rows = {}
            ratios = {}
            for role in ("control", "candidate"):
                statistics, ratio, tail = metric_row(
                    results[role]["model"],
                    datasets["confirmation"],
                    chunk_size=args.eval_batch_size,
                )
                confirmation_rows[role] = {
                    "statistics": statistics,
                    "tail": tail,
                }
                ratios[role] = ratio
            paired = paired_improvement(
                ratios["control"],
                ratios["candidate"],
                datasets["confirmation"]["weights_numpy"],
            )
            control_confirmation = confirmation_rows["control"]["statistics"]
            candidate_confirmation = confirmation_rows["candidate"]["statistics"]
            confirmation_passes = bool(
                selection_passes
                and candidate_confirmation["sigma_official_formula"]
                < control_confirmation["sigma_official_formula"]
                and candidate_confirmation["weighted_rms_abs_residual"]
                < control_confirmation["weighted_rms_abs_residual"]
                and paired["e2"]["ci95_low"] > 0
                and paired["sigma"]["ci95_low"] > 0
                and zero_nonpositive(candidate_confirmation)
                and tail_guard(
                    confirmation_rows["candidate"]["tail"],
                    confirmation_rows["control"]["tail"],
                    relative_degradation=(
                        args.maximum_selection_tail_relative_degradation
                    ),
                )
            )
            confirmation_report = {
                **confirmation_rows,
                "paired_improvement": paired,
                "passes": confirmation_passes,
            }

        winner = (
            "candidate"
            if (
                confirmation_passes
                if confirmation_passes is not None
                else selection_passes
            )
            else "control"
        )
        winner_path = output_dir / (
            "accepted_candidate.pt"
            if winner == "candidate"
            else "control_retained.pt"
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
            "configuration": {
                **vars(args),
                "control_checkpoint": str(control_path),
                "candidate_checkpoint": str(candidate_path),
                "output_dir": str(output_dir),
            },
            "results": serializable_results,
            "selection_comparison": {
                "sigma_gain": sigma_gain,
                "chi_gain": chi_gain,
                "passes": selection_passes,
            },
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
