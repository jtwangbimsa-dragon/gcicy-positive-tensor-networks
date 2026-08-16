#!/usr/bin/env python3
"""Refine a hard-symmetry quintic TN against a Reynolds teacher metric."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.audit_quintic_tn_reynolds_derivative_fidelity import (  # noqa: E402
    model_feature_jet,
    reynolds_feature_jet,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    ratio_statistics,
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    fixed_fermat_actions_torch,
    tensor_split,
    training_log_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--group-samples", type=int, default=64)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-action-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--metric-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-determinant-loss-weight", type=float, default=0.0)
    parser.add_argument("--log-feature-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--selection-objective",
        choices=("metric", "logdet", "combined"),
        default="metric",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--expected-parameter-count", type=int, default=0)
    parser.add_argument("--torch-seed", type=int, default=202607234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integers = (
        "train_limit",
        "validation_limit",
        "group_samples",
        "teacher_batch_size",
        "teacher_action_batch_size",
        "batch_size",
        "eval_batch_size",
    )
    for name in positive_integers:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.epochs < 0:
        raise ValueError("--epochs cannot be negative")
    if args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        raise ValueError("learning rate and gradient clipping must be positive")
    if (
        args.metric_loss_weight < 0
        or args.log_determinant_loss_weight < 0
        or args.log_feature_loss_weight < 0
    ):
        raise ValueError("loss weights cannot be negative")
    if (
        args.metric_loss_weight == 0
        and args.log_determinant_loss_weight == 0
        and args.log_feature_loss_weight == 0
    ):
        raise ValueError("at least one refinement loss must be active")
    if args.selection_objective == "metric" and args.metric_loss_weight == 0:
        raise ValueError("metric selection requires a nonzero metric loss weight")
    if args.selection_objective == "logdet" and args.log_determinant_loss_weight == 0:
        raise ValueError("logdet selection requires a nonzero log-determinant weight")
    if args.expected_parameter_count < 0:
        raise ValueError("--expected-parameter-count cannot be negative")


def teacher_targets(
    teacher: torch.nn.Module,
    dataset: dict[str, object],
    actions: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    *,
    batch_size: int,
    action_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_feature_rows = []
    metric_rows = []
    teacher.eval()
    with torch.no_grad():
        for start in range(0, int(dataset["count"]), batch_size):
            stop = min(start + batch_size, int(dataset["count"]))
            log_feature, _, metric = reynolds_feature_jet(
                teacher,
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
                actions,
                action_chunk_size=action_batch_size,
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(eigenvalues > 0)):
                raise FloatingPointError(
                    "Reynolds teacher target contains a nonpositive metric"
                )
            log_feature_rows.append(log_feature)
            metric_rows.append(metric)
    return torch.cat(log_feature_rows), torch.cat(metric_rows)


def whitened_metric_squared(
    student_metric: torch.Tensor,
    teacher_metric: torch.Tensor,
    teacher_cholesky: torch.Tensor | None = None,
) -> torch.Tensor:
    """Squared Frobenius norm after whitening by the teacher metric."""

    factor = (
        torch.linalg.cholesky(teacher_metric)
        if teacher_cholesky is None
        else teacher_cholesky
    )
    delta = student_metric - teacher_metric
    left_solved = torch.linalg.solve_triangular(
        factor,
        delta,
        upper=False,
    )
    whitened = torch.linalg.solve_triangular(
        torch.conj(factor),
        torch.transpose(left_solved, -2, -1),
        upper=False,
    ).transpose(-2, -1)
    return torch.sum(torch.abs(whitened) ** 2, dim=(-2, -1))


def evaluate(
    model: torch.nn.Module,
    dataset: dict[str, object],
    target_log_feature: torch.Tensor,
    target_metric: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, object]:
    log_feature_rows = []
    metric_squared_rows = []
    log_determinant_squared_rows = []
    raw_rows = []
    minimum_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(dataset["count"]), batch_size):
            stop = min(start + batch_size, int(dataset["count"]))
            log_feature, _, metric = model_feature_jet(
                model,
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(eigenvalues > 0)):
                raise FloatingPointError("student metric is not positive")
            metric_squared = whitened_metric_squared(
                metric,
                target_metric[start:stop],
            )
            target_log_determinant = training_log_volume(
                target_metric[start:stop],
                "cholesky",
            )
            student_log_determinant = torch.sum(torch.log(eigenvalues), dim=1)
            log_determinant_squared_rows.append(
                torch.square(student_log_determinant - target_log_determinant)
            )
            raw = torch.sum(torch.log(eigenvalues), dim=1)
            raw -= dataset["log_omega"][start:stop]
            log_feature_rows.append(log_feature)
            metric_squared_rows.append(metric_squared)
            raw_rows.append(raw)
            minimum_rows.append(torch.min(eigenvalues, dim=1).values)

    weights = np.asarray(dataset["weights_numpy"], dtype=np.float64)
    log_feature = torch.cat(log_feature_rows).detach().cpu().numpy().astype(np.float64)
    target_log = target_log_feature.detach().cpu().numpy().astype(np.float64)
    centered = log_feature - target_log
    centered -= float(np.sum(weights * centered))
    metric_squared = (
        torch.cat(metric_squared_rows).detach().cpu().numpy().astype(np.float64)
    )
    log_determinant_squared = (
        torch.cat(log_determinant_squared_rows)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    raw = torch.cat(raw_rows).detach().cpu().numpy().astype(np.float64)
    minimum = torch.cat(minimum_rows).detach().cpu().numpy().astype(np.float64)
    normalized_volume, _ = ratio_statistics(raw, weights, minimum)
    return {
        "centered_log_feature_rms": math.sqrt(
            float(np.sum(weights * np.square(centered)))
        ),
        "teacher_whitened_metric_frobenius_rms": math.sqrt(
            float(np.sum(weights * metric_squared))
        ),
        "log_determinant_rms": math.sqrt(
            float(np.sum(weights * log_determinant_squared))
        ),
        "normalized_volume": normalized_volume,
    }


def checkpoint_score(
    validation_result: dict[str, object],
    *,
    objective: str,
    metric_weight: float,
    log_determinant_weight: float,
) -> float:
    metric_rms = float(
        validation_result["teacher_whitened_metric_frobenius_rms"]
    )
    logdet_rms = float(validation_result["log_determinant_rms"])
    if objective == "metric":
        return metric_rms
    if objective == "logdet":
        return logdet_rms
    if objective != "combined":
        raise ValueError(f"unsupported checkpoint objective: {objective}")
    return math.sqrt(
        metric_weight * metric_rms**2
        + log_determinant_weight * logdet_rms**2
    )


def target_volume_statistics(
    target_metric: torch.Tensor,
    dataset: dict[str, object],
) -> dict[str, object]:
    eigenvalues = torch.linalg.eigvalsh(target_metric)
    if not bool(torch.all(eigenvalues > 0)):
        raise FloatingPointError("teacher metric is not positive")
    raw = torch.sum(torch.log(eigenvalues), dim=1) - dataset["log_omega"]
    minimum = torch.min(eigenvalues, dim=1).values
    normalized, _ = ratio_statistics(
        raw.detach().cpu().numpy().astype(np.float64),
        np.asarray(dataset["weights_numpy"], dtype=np.float64),
        minimum.detach().cpu().numpy().astype(np.float64),
    )
    return normalized


def save_model(
    path: Path,
    initial_payload: dict[str, object],
    state: dict[str, torch.Tensor],
    evidence: dict[str, object],
) -> None:
    payload = dict(initial_payload)
    payload["state_dict"] = {
        key: value.detach().cpu().clone() for key, value in state.items()
    }
    payload["derivative_refinement"] = evidence
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    started = time.perf_counter()
    write_json(status_path, {"state": "running", "phase": "loading"})

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.torch_seed)

    initial_path = args.initial_model.expanduser().resolve()
    teacher_path = args.teacher_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    train_pullbacks_path = pullbacks_dir / "train_pullbacks.npy"
    validation_pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    for path in (
        initial_path,
        teacher_path,
        dataset_path,
        train_pullbacks_path,
        validation_pullbacks_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
    source_degree = int(initial_payload.get("source_degree", 1))
    for key in ("source_degree", "site_count", "total_degree"):
        default = 1 if key == "source_degree" else -1
        if int(initial_payload.get(key, default)) != int(
            teacher_payload.get(key, default)
        ):
            raise ValueError(f"student and teacher {key} differ")
    reference_h = np.asarray(
        initial_payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )

    data = np.load(dataset_path, allow_pickle=False)
    train_x = np.asarray(data["X_train"][: args.train_limit], dtype=np.float32)
    train_y = np.asarray(data["y_train"][: args.train_limit], dtype=np.float64)
    validation_x = np.asarray(
        data["X_val"][: args.validation_limit], dtype=np.float32
    )
    validation_y = np.asarray(
        data["y_val"][: args.validation_limit], dtype=np.float64
    )
    train_pullbacks = np.load(train_pullbacks_path, mmap_mode="r")[: args.train_limit]
    validation_pullbacks = np.load(validation_pullbacks_path, mmap_mode="r")[
        : args.validation_limit
    ]

    write_json(status_path, {"state": "running", "phase": "teacher_targets"})
    target_train_data = tensor_split(
        train_x,
        train_pullbacks,
        train_y,
        source_degree=source_degree,
        complex_dtype=torch.complex128,
        real_dtype=torch.float64,
        device=device,
    )
    target_validation_data = tensor_split(
        validation_x,
        validation_pullbacks,
        validation_y,
        source_degree=source_degree,
        complex_dtype=torch.complex128,
        real_dtype=torch.float64,
        device=device,
    )
    teacher = positive_tensor_network_from_artifact_payload(
        reference_h,
        teacher_payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=torch.complex128)
    teacher.requires_grad_(False).eval()
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.torch_seed + 1009)
    actions = fixed_fermat_actions_torch(
        args.group_samples,
        generator=action_generator,
        complex_dtype=torch.complex128,
        device=device,
    )
    target_started = time.perf_counter()
    train_target_log, train_target_metric = teacher_targets(
        teacher,
        target_train_data,
        actions,
        batch_size=args.teacher_batch_size,
        action_batch_size=args.teacher_action_batch_size,
    )
    validation_target_log, validation_target_metric = teacher_targets(
        teacher,
        target_validation_data,
        actions,
        batch_size=args.teacher_batch_size,
        action_batch_size=args.teacher_action_batch_size,
    )
    target_seconds = time.perf_counter() - target_started
    teacher_validation_volume = target_volume_statistics(
        validation_target_metric,
        target_validation_data,
    )
    del teacher, actions, target_train_data, target_validation_data

    student_dtype = (
        torch.complex64
        if str(initial_payload["precision"]) == "complex64"
        else torch.complex128
    )
    student_real_dtype = (
        torch.float32 if student_dtype == torch.complex64 else torch.float64
    )
    train = tensor_split(
        train_x,
        train_pullbacks,
        train_y,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=student_real_dtype,
        device=device,
    )
    validation = tensor_split(
        validation_x,
        validation_pullbacks,
        validation_y,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=student_real_dtype,
        device=device,
    )
    train_target_log = train_target_log.to(dtype=student_real_dtype)
    train_target_metric = train_target_metric.to(dtype=student_dtype)
    validation_target_log = validation_target_log.to(dtype=student_real_dtype)
    validation_target_metric = validation_target_metric.to(dtype=student_dtype)
    train_target_cholesky = torch.linalg.cholesky(train_target_metric)
    train_target_log_determinant = training_log_volume(
        train_target_metric,
        "cholesky",
    ).detach()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        initial_payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=student_dtype)
    active_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not active_parameters:
        raise RuntimeError("student model has no trainable parameters")
    parameter_count = int(model.trainable_real_parameter_count)
    if (
        args.expected_parameter_count
        and parameter_count != args.expected_parameter_count
    ):
        raise RuntimeError(
            f"parameter-count gate failed: {parameter_count} != "
            f"{args.expected_parameter_count}"
        )
    optimizer = torch.optim.Adam(active_parameters, lr=args.learning_rate)
    permutation_generator = torch.Generator(device=device)
    permutation_generator.manual_seed(args.torch_seed + 1013)

    initial_train_log = []
    model.eval()
    with torch.no_grad():
        for start in range(0, train["count"], args.eval_batch_size):
            stop = min(start + args.eval_batch_size, train["count"])
            initial_train_log.append(
                model.log_feature_norm(train["values"][start:stop])
            )
    initial_train_log = torch.cat(initial_train_log)
    gauge_shift = torch.sum(
        train["weights"] * (initial_train_log - train_target_log)
    ).detach()

    history = []
    best_epoch = -1
    best_score = float("inf")
    best_state = None
    optimizer_steps = 0
    train_started = time.perf_counter()
    for epoch in range(args.epochs + 1):
        validation_result = evaluate(
            model,
            validation,
            validation_target_log,
            validation_target_metric,
            batch_size=args.eval_batch_size,
        )
        score = checkpoint_score(
            validation_result,
            objective=args.selection_objective,
            metric_weight=args.metric_loss_weight,
            log_determinant_weight=args.log_determinant_loss_weight,
        )
        accepted = score < best_score
        if accepted:
            best_epoch = epoch
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch,
            "accepted": accepted,
            "selection_score": score,
            "validation": validation_result,
        }
        history.append(row)
        write_json(
            output_dir / "training_history.json",
            {"rows": history, "best_epoch": best_epoch, "best_score": best_score},
        )
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "training",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "selection_objective": args.selection_objective,
                "validation_score": score,
                "validation_metric_rms": validation_result[
                    "teacher_whitened_metric_frobenius_rms"
                ],
                "validation_logdet_rms": validation_result[
                    "log_determinant_rms"
                ],
            },
        )
        print(
            f"epoch={epoch} "
            f"metric_rms={validation_result['teacher_whitened_metric_frobenius_rms']:.6e} "
            f"logdet_rms={validation_result['log_determinant_rms']:.6e} "
            f"sigma={validation_result['normalized_volume']['sigma_official_formula']:.6e} "
            f"selection_score={score:.6e} "
            f"accepted={accepted}",
            flush=True,
        )
        if epoch == args.epochs:
            break

        model.train()
        permutation = torch.randperm(
            train["count"],
            generator=permutation_generator,
            device=device,
        )
        for start in range(0, train["count"], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch_weights = train["weights"][indices]
            batch_weights = batch_weights / torch.sum(batch_weights)
            optimizer.zero_grad(set_to_none=True)
            log_feature, _, metric = model_feature_jet(
                model,
                train["values"][indices],
                train["derivatives"][indices],
            )
            metric_squared = whitened_metric_squared(
                metric,
                train_target_metric[indices],
                train_target_cholesky[indices],
            )
            metric_loss = torch.sum(batch_weights * metric_squared)
            student_log_determinant = training_log_volume(metric, "cholesky")
            log_determinant_loss = torch.sum(
                batch_weights
                * torch.square(
                    student_log_determinant
                    - train_target_log_determinant[indices]
                )
            )
            log_residual = log_feature - train_target_log[indices] - gauge_shift
            log_feature_loss = torch.sum(batch_weights * torch.square(log_residual))
            loss = (
                args.metric_loss_weight * metric_loss
                + args.log_determinant_loss_weight * log_determinant_loss
                + args.log_feature_loss_weight * log_feature_loss
            )
            torch._assert_async(torch.isfinite(loss), "nonfinite refinement objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                active_parameters,
                args.gradient_clip_norm,
            )
            optimizer.step()
            optimizer_steps += 1

    if best_state is None:
        raise RuntimeError("refinement produced no finite checkpoint")
    model.load_state_dict(best_state)
    final_validation = evaluate(
        model,
        validation,
        validation_target_log,
        validation_target_metric,
        batch_size=args.eval_batch_size,
    )
    evidence = {
        "schema": "quintic-reynolds-metric-refinement-v1",
        "initial_model": str(initial_path),
        "initial_model_sha256": sha256_file(initial_path),
        "teacher_model": str(teacher_path),
        "teacher_model_sha256": sha256_file(teacher_path),
        "group_samples": args.group_samples,
        "train_points": train["count"],
        "validation_points": validation["count"],
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "metric_loss_weight": args.metric_loss_weight,
        "log_determinant_loss_weight": args.log_determinant_loss_weight,
        "log_feature_loss_weight": args.log_feature_loss_weight,
        "selection_objective": args.selection_objective,
        "best_epoch": best_epoch,
        "optimizer_steps": optimizer_steps,
        "trainable_real_parameter_count": parameter_count,
    }
    save_model(
        output_dir / "best_tensor_network.pt",
        initial_payload,
        best_state,
        evidence,
    )
    report = {
        "schema": "quintic-reynolds-metric-refinement-report-v1",
        "configuration": evidence,
        "teacher_validation_normalized_volume": teacher_validation_volume,
        "history": history,
        "final_validation": final_validation,
        "teacher_target_seconds": target_seconds,
        "training_seconds": time.perf_counter() - train_started,
        "wall_seconds": time.perf_counter() - started,
        "source_sha256": {
            "dataset": sha256_file(dataset_path),
            "train_pullbacks": sha256_file(train_pullbacks_path),
            "validation_pullbacks": sha256_file(validation_pullbacks_path),
        },
    }
    write_json(output_dir / "report.json", report)
    write_json(status_path, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "final_validation": final_validation,
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
