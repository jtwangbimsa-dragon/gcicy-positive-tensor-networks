#!/usr/bin/env python3
"""Distill a degree-lifted Fermat full-H oracle into a trained quintic TN."""

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

from gcicy_metric.fermat_quintic import (  # noqa: E402
    fermat_quintic_quotient_basis,
    fermat_quotient_fubini_study_h,
)
from gcicy_metric.pipeline import (  # noqa: E402
    AlgebraicMetricPowerLift,
    FermatSymmetricFullH,
    positive_tensor_network_from_artifact_payload,
)
from scripts.audit_quintic_tn_reynolds_derivative_fidelity import (  # noqa: E402
    model_feature_jet,
)
from scripts.refine_quintic_tn_reynolds_metric import (  # noqa: E402
    checkpoint_score,
    evaluate,
    save_model,
    target_volume_statistics,
    whitened_metric_squared,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    prepare_features,
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    tensor_split,
    training_log_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--teacher-h-artifact", type=Path, required=True)
    parser.add_argument("--teacher-h-key", default="global_h_matrix")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-degree", type=int, default=10)
    parser.add_argument("--target-degree", type=int, default=30)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--validation-start", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--teacher-feature-batch-size", type=int, default=1024)
    parser.add_argument("--teacher-eval-batch-size", type=int, default=128)
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
    parser.add_argument("--expected-stored-parameter-count", type=int, default=0)
    parser.add_argument("--require-teacher-better", action="store_true")
    parser.add_argument("--torch-seed", type=int, default=202607261)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.teacher_degree,
        args.target_degree,
        args.train_limit,
        args.validation_limit,
        args.teacher_feature_batch_size,
        args.teacher_eval_batch_size,
        args.batch_size,
        args.eval_batch_size,
        args.gradient_clip_norm,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("degrees, sample sizes, batches, and clipping must be positive")
    if args.target_degree % args.teacher_degree:
        raise ValueError("target degree must be an integer teacher-degree multiple")
    if args.train_start < 0 or args.validation_start < 0 or args.epochs < 0:
        raise ValueError("dataset starts and epochs cannot be negative")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    weights = (
        args.metric_loss_weight,
        args.log_determinant_loss_weight,
        args.log_feature_loss_weight,
    )
    if any(value < 0 for value in weights) or not any(value > 0 for value in weights):
        raise ValueError("distillation weights must be non-negative and nonzero")
    if args.selection_objective == "metric" and args.metric_loss_weight == 0:
        raise ValueError("metric selection requires metric loss")
    if (
        args.selection_objective == "logdet"
        and args.log_determinant_loss_weight == 0
    ):
        raise ValueError("logdet selection requires log-determinant loss")
    if (
        args.expected_parameter_count < 0
        or args.expected_stored_parameter_count < 0
    ):
        raise ValueError("expected parameter counts cannot be negative")


def array_slice(
    value: np.ndarray,
    *,
    start: int,
    count: int,
) -> np.ndarray:
    stop = start + count
    if stop > len(value):
        raise ValueError(f"requested rows [{start}:{stop}] from {len(value)} entries")
    return np.asarray(value[start:stop])


def full_h_teacher_targets(
    teacher: AlgebraicMetricPowerLift,
    x_values: np.ndarray,
    pullbacks: np.ndarray,
    exponents: np.ndarray,
    *,
    feature_batch_size: int,
    eval_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    section_values, section_derivatives = prepare_features(
        x_values,
        pullbacks,
        exponents,
        feature_batch_size,
        complex_dtype=np.dtype(np.complex128),
    )
    values = torch.tensor(
        section_values, dtype=torch.complex128, device=device
    )
    derivatives = torch.tensor(
        section_derivatives, dtype=torch.complex128, device=device
    )
    log_rows = []
    metric_rows = []
    teacher.eval()
    with torch.no_grad():
        for start in range(0, len(values), eval_batch_size):
            stop = min(start + eval_batch_size, len(values))
            log_feature, metric = teacher.log_feature_and_metric(
                values[start:stop],
                derivatives[start:stop],
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(torch.isfinite(eigenvalues) & (eigenvalues > 0))):
                raise FloatingPointError("full-H teacher metric is not positive")
            log_rows.append(log_feature)
            metric_rows.append(metric)
    return torch.cat(log_rows), torch.cat(metric_rows)


def load_teacher(
    artifact_path: Path,
    *,
    matrix_key: str,
    source_degree: int,
    target_degree: int,
    device: torch.device,
) -> tuple[AlgebraicMetricPowerLift, np.ndarray]:
    artifact = np.load(artifact_path, allow_pickle=False)
    if int(np.asarray(artifact["degree"]).item()) != source_degree:
        raise ValueError("teacher H artifact degree does not match --teacher-degree")
    if matrix_key not in artifact.files:
        raise KeyError(f"{matrix_key!r} is absent from {artifact_path}")
    ambient, exponents, quotient_lift = fermat_quintic_quotient_basis(source_degree)
    initial_h = fermat_quotient_fubini_study_h(ambient, quotient_lift)
    source = FermatSymmetricFullH(
        exponents,
        initial_h,
        normalization=1.0 / (math.pi * source_degree),
        device=device,
        conjugation_invariant=True,
    )
    source.set_coordinates_from_h_(
        np.asarray(artifact[matrix_key], dtype=np.complex128)
    )
    source.requires_grad_(False).eval()
    return (
        AlgebraicMetricPowerLift(
            source,
            source_degree=source_degree,
            target_degree=target_degree,
        ).to(device),
        exponents,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
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
    teacher_path = args.teacher_h_artifact.expanduser().resolve()
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
    if int(initial_payload["total_degree"]) != args.target_degree:
        raise ValueError("student total degree does not match the lifted teacher")
    source_degree = int(initial_payload.get("source_degree", 1))
    reference_h = np.asarray(
        initial_payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )

    data = np.load(dataset_path, allow_pickle=False)
    train_x = array_slice(
        np.asarray(data["X_train"], dtype=np.float32),
        start=args.train_start,
        count=args.train_limit,
    )
    train_y = array_slice(
        np.asarray(data["y_train"], dtype=np.float64),
        start=args.train_start,
        count=args.train_limit,
    )
    validation_x = array_slice(
        np.asarray(data["X_val"], dtype=np.float32),
        start=args.validation_start,
        count=args.validation_limit,
    )
    validation_y = array_slice(
        np.asarray(data["y_val"], dtype=np.float64),
        start=args.validation_start,
        count=args.validation_limit,
    )
    train_pullbacks = array_slice(
        np.load(train_pullbacks_path, mmap_mode="r"),
        start=args.train_start,
        count=args.train_limit,
    )
    validation_pullbacks = array_slice(
        np.load(validation_pullbacks_path, mmap_mode="r"),
        start=args.validation_start,
        count=args.validation_limit,
    )

    write_json(status_path, {"state": "running", "phase": "teacher_targets"})
    teacher, teacher_exponents = load_teacher(
        teacher_path,
        matrix_key=args.teacher_h_key,
        source_degree=args.teacher_degree,
        target_degree=args.target_degree,
        device=device,
    )
    target_started = time.perf_counter()
    train_target_log, train_target_metric = full_h_teacher_targets(
        teacher,
        train_x,
        train_pullbacks,
        teacher_exponents,
        feature_batch_size=args.teacher_feature_batch_size,
        eval_batch_size=args.teacher_eval_batch_size,
        device=device,
    )
    validation_target_log, validation_target_metric = full_h_teacher_targets(
        teacher,
        validation_x,
        validation_pullbacks,
        teacher_exponents,
        feature_batch_size=args.teacher_feature_batch_size,
        eval_batch_size=args.teacher_eval_batch_size,
        device=device,
    )
    target_seconds = time.perf_counter() - target_started
    del teacher

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
    teacher_validation_volume = target_volume_statistics(
        validation_target_metric,
        {
            **validation,
            "log_omega": validation["log_omega"].to(torch.float64),
        },
    )
    train_target_log = train_target_log.to(dtype=student_real_dtype)
    train_target_metric = train_target_metric.to(dtype=student_dtype)
    validation_target_log = validation_target_log.to(dtype=student_real_dtype)
    validation_target_metric = validation_target_metric.to(dtype=student_dtype)
    train_target_cholesky = torch.linalg.cholesky(train_target_metric)
    train_target_log_determinant = training_log_volume(
        train_target_metric, "cholesky"
    ).detach()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        initial_payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=student_dtype)
    active_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not active_parameters:
        raise RuntimeError("student model has no trainable parameters")
    parameter_count = int(model.trainable_real_parameter_count)
    fixed_dictionary_real_parameters = 0
    if initial_payload.get("trainable_physical_dictionary", False):
        dictionary = model.physical_dictionary
        fixed_dictionary_real_parameters = dictionary.numel() * (
            2 if dictionary.is_complex() else 1
        )
    stored_parameter_count = parameter_count + fixed_dictionary_real_parameters
    if (
        args.expected_parameter_count
        and parameter_count != args.expected_parameter_count
    ):
        raise RuntimeError(
            f"parameter-count gate failed: {parameter_count} != "
            f"{args.expected_parameter_count}"
        )
    if (
        args.expected_stored_parameter_count
        and stored_parameter_count != args.expected_stored_parameter_count
    ):
        raise RuntimeError(
            f"stored parameter-count gate failed: {stored_parameter_count} != "
            f"{args.expected_stored_parameter_count}"
        )
    optimizer = torch.optim.Adam(active_parameters, lr=args.learning_rate)
    permutation_generator = torch.Generator(device=device)
    permutation_generator.manual_seed(args.torch_seed + 1013)

    model.eval()
    initial_train_log = []
    with torch.no_grad():
        for start in range(0, train["count"], args.eval_batch_size):
            stop = min(start + args.eval_batch_size, train["count"])
            initial_train_log.append(
                model.log_feature_norm(train["values"][start:stop])
            )
    gauge_shift = torch.sum(
        train["weights"] * (torch.cat(initial_train_log) - train_target_log)
    ).detach()

    history = []
    best_epoch = -1
    best_score = float("inf")
    best_state = None
    optimizer_steps = 0
    training_started = time.perf_counter()
    initial_validation = evaluate(
        model,
        validation,
        validation_target_log,
        validation_target_metric,
        batch_size=args.eval_batch_size,
    )
    teacher_sigma = float(
        teacher_validation_volume["sigma_official_formula"]
    )
    initial_student_sigma = float(
        initial_validation["normalized_volume"]["sigma_official_formula"]
    )
    if args.require_teacher_better and teacher_sigma >= initial_student_sigma:
        failure = {
            "state": "failed",
            "phase": "teacher_quality_gate",
            "teacher_sigma": teacher_sigma,
            "initial_student_sigma": initial_student_sigma,
        }
        write_json(status_path, failure)
        raise RuntimeError(
            "full-H oracle quality gate failed: "
            f"teacher sigma {teacher_sigma:.6e} is not below "
            f"student sigma {initial_student_sigma:.6e}"
        )
    for epoch in range(args.epochs + 1):
        validation_result = (
            initial_validation
            if epoch == 0
            else evaluate(
                model,
                validation,
                validation_target_log,
                validation_target_metric,
                batch_size=args.eval_batch_size,
            )
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
            save_model(
                output_dir / "best_tensor_network.pt",
                initial_payload,
                best_state,
                {
                    "schema": "quintic-full-h-oracle-distillation-checkpoint-v1",
                    "checkpoint_phase": "training",
                    "teacher_h_artifact": str(teacher_path),
                    "teacher_degree": args.teacher_degree,
                    "target_degree": args.target_degree,
                    "teacher_validation_sigma": teacher_sigma,
                    "initial_student_validation_sigma": initial_student_sigma,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "optimizer_steps": optimizer_steps,
                    "active_real_parameters": parameter_count,
                    "fixed_learned_dictionary_real_parameters": (
                        fixed_dictionary_real_parameters
                    ),
                    "stored_learned_real_parameters": stored_parameter_count,
                },
            )
        history.append(
            {
                "epoch": epoch,
                "accepted": accepted,
                "selection_score": score,
                "validation": validation_result,
            }
        )
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
                "validation_metric_rms": validation_result[
                    "teacher_whitened_metric_frobenius_rms"
                ],
                "validation_sigma": validation_result["normalized_volume"][
                    "sigma_official_formula"
                ],
            },
        )
        print(
            f"epoch={epoch} "
            f"metric_rms={validation_result['teacher_whitened_metric_frobenius_rms']:.6e} "
            f"logdet_rms={validation_result['log_determinant_rms']:.6e} "
            f"sigma={validation_result['normalized_volume']['sigma_official_formula']:.6e} "
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
            logdet_loss = torch.sum(
                batch_weights
                * torch.square(
                    student_log_determinant
                    - train_target_log_determinant[indices]
                )
            )
            log_residual = log_feature - train_target_log[indices] - gauge_shift
            log_feature_loss = torch.sum(
                batch_weights * torch.square(log_residual)
            )
            loss = (
                args.metric_loss_weight * metric_loss
                + args.log_determinant_loss_weight * logdet_loss
                + args.log_feature_loss_weight * log_feature_loss
            )
            torch._assert_async(torch.isfinite(loss), "nonfinite oracle loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                active_parameters, args.gradient_clip_norm
            )
            optimizer.step()
            optimizer_steps += 1

    if best_state is None:
        raise RuntimeError("oracle distillation produced no finite checkpoint")
    model.load_state_dict(best_state)
    final_validation = evaluate(
        model,
        validation,
        validation_target_log,
        validation_target_metric,
        batch_size=args.eval_batch_size,
    )
    evidence = {
        "schema": "quintic-full-h-oracle-distillation-v1",
        "initial_model": str(initial_path),
        "initial_model_sha256": sha256_file(initial_path),
        "teacher_h_artifact": str(teacher_path),
        "teacher_h_artifact_sha256": sha256_file(teacher_path),
        "teacher_h_key": args.teacher_h_key,
        "teacher_degree": args.teacher_degree,
        "target_degree": args.target_degree,
        "train_start": args.train_start,
        "train_points": train["count"],
        "validation_start": args.validation_start,
        "validation_points": validation["count"],
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "metric_loss_weight": args.metric_loss_weight,
        "log_determinant_loss_weight": args.log_determinant_loss_weight,
        "log_feature_loss_weight": args.log_feature_loss_weight,
        "selection_objective": args.selection_objective,
        "require_teacher_better": args.require_teacher_better,
        "teacher_validation_sigma": teacher_sigma,
        "initial_student_validation_sigma": initial_student_sigma,
        "best_epoch": best_epoch,
        "optimizer_steps": optimizer_steps,
        "active_real_parameters": parameter_count,
        "fixed_learned_dictionary_real_parameters": (
            fixed_dictionary_real_parameters
        ),
        "stored_learned_real_parameters": stored_parameter_count,
    }
    save_model(
        output_dir / "best_tensor_network.pt",
        initial_payload,
        best_state,
        evidence,
    )
    report = {
        "schema": "quintic-full-h-oracle-distillation-report-v1",
        "configuration": evidence,
        "teacher_validation_normalized_volume": teacher_validation_volume,
        "history": history,
        "final_validation": final_validation,
        "teacher_target_seconds": target_seconds,
        "training_seconds": time.perf_counter() - training_started,
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
                "teacher_sigma": teacher_validation_volume[
                    "sigma_official_formula"
                ],
                "student_sigma": final_validation["normalized_volume"][
                    "sigma_official_formula"
                ],
                "metric_rms": final_validation[
                    "teacher_whitened_metric_frobenius_rms"
                ],
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
