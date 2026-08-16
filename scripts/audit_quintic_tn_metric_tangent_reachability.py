#!/usr/bin/env python3
"""Audit local teacher-metric and log-determinant reachability of a quintic TN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.func import functional_call, jvp, vjp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    ComplexParameterVectorizer,
    lanczos_tridiagonal,
    real_inner,
    ridge_dual_from_lanczos,
    select_disjoint_indices,
    vector_norm,
)
from scripts.refine_quintic_tn_reynolds_metric import teacher_targets  # noqa: E402
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    fixed_fermat_actions_torch,
    tensor_split,
    training_log_volume,
)


@dataclass(frozen=True)
class MaskedComplexParameterVectorizer:
    """Expose selected entries of one parameter to functional_call."""

    name: str
    shape: torch.Size
    flat_indices: torch.Tensor
    baseline: torch.Tensor

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        parameter_name: str,
        mask: torch.Tensor,
    ) -> "MaskedComplexParameterVectorizer":
        parameters = dict(module.named_parameters())
        if parameter_name not in parameters:
            raise ValueError(f"unknown trainable parameter: {parameter_name}")
        parameter = parameters[parameter_name]
        if not parameter.requires_grad:
            raise ValueError("masked parameter must be trainable")
        resolved_mask = mask.to(device=parameter.device, dtype=torch.bool)
        if resolved_mask.shape != parameter.shape:
            raise ValueError("parameter mask has the wrong shape")
        flat_indices = torch.nonzero(resolved_mask.reshape(-1), as_tuple=False).reshape(-1)
        if flat_indices.numel() == 0:
            raise ValueError("parameter mask cannot be empty")
        return cls(
            name=parameter_name,
            shape=parameter.shape,
            flat_indices=flat_indices,
            baseline=parameter.detach().clone(),
        )

    def pack(self, module: torch.nn.Module) -> torch.Tensor:
        parameter = dict(module.named_parameters())[self.name]
        return torch.index_select(parameter.detach().reshape(-1), 0, self.flat_indices)

    def unpack(self, vector: torch.Tensor) -> dict[str, torch.Tensor]:
        if vector.ndim != 1 or vector.numel() != self.flat_indices.numel():
            raise ValueError("masked parameter vector has the wrong shape")
        value = self.baseline.reshape(-1).scatter(0, self.flat_indices, vector)
        return {self.name: value.reshape(self.shape)}

    def commit_(self, module: torch.nn.Module, vector: torch.Tensor) -> None:
        replacement = self.unpack(vector)[self.name]
        parameter = dict(module.named_parameters())[self.name]
        with torch.no_grad():
            parameter.copy_(replacement)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--probe-size", type=int, default=128)
    parser.add_argument("--holdout-size", type=int, default=128)
    parser.add_argument("--group-samples", type=int, default=64)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-action-batch-size", type=int, default=64)
    parser.add_argument("--operator-chunk-size", type=int, default=2)
    parser.add_argument("--lanczos-steps", type=int, default=8)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(100.0, 10.0, 1.0, 0.1),
    )
    parser.add_argument("--expected-parameter-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202607236)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.probe_size,
        args.holdout_size,
        args.group_samples,
        args.teacher_batch_size,
        args.teacher_action_batch_size,
        args.operator_chunk_size,
        args.lanczos_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sample, batch, chunk, and Lanczos sizes must be positive")
    if args.candidate_limit < 0 or args.expected_parameter_count < 0:
        raise ValueError("candidate and expected parameter counts cannot be negative")
    if not args.ridge_factors or any(value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be positive")


def whiten_metric(
    metric: torch.Tensor,
    target_metric: torch.Tensor,
    target_cholesky: torch.Tensor,
) -> torch.Tensor:
    delta = metric - target_metric
    left_solved = torch.linalg.solve_triangular(
        target_cholesky,
        delta,
        upper=False,
    )
    return torch.linalg.solve_triangular(
        torch.conj(target_cholesky),
        torch.transpose(left_solved, -2, -1),
        upper=False,
    ).transpose(-2, -1)


class MatrixFreeTeacherMetricJacobian:
    """JVP/VJP operator for a fixed Reynolds teacher metric."""

    def __init__(
        self,
        model: torch.nn.Module,
        vectorizer: ComplexParameterVectorizer | MaskedComplexParameterVectorizer,
        theta: torch.Tensor,
        dataset: dict[str, Any],
        target_metric: torch.Tensor,
        *,
        mode: str,
        chunk_size: int,
    ) -> None:
        if mode not in {"metric", "logdet"}:
            raise ValueError("mode must be metric or logdet")
        if target_metric.shape[0] != dataset["count"]:
            raise ValueError("teacher target count does not match the dataset")
        self.model = model
        self.vectorizer = vectorizer
        self.theta = theta
        self.dataset = dataset
        self.target_metric = target_metric
        self.target_cholesky = torch.linalg.cholesky(target_metric)
        self.target_logdet = training_log_volume(target_metric, "cholesky")
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.components_per_point = (
            2 * target_metric.shape[-1] ** 2 if mode == "metric" else 1
        )
        self._slices = tuple(
            slice(start, min(start + chunk_size, dataset["count"]))
            for start in range(0, dataset["count"], chunk_size)
        )
        self.output_count = int(dataset["count"] * self.components_per_point)

    def _chunk_residual_function(
        self,
        selection: slice,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        values = self.dataset["values"][selection]
        derivatives = self.dataset["derivatives"][selection]
        square_root_weights = torch.sqrt(self.dataset["weights"][selection])
        target = self.target_metric[selection]
        target_cholesky = self.target_cholesky[selection]
        target_logdet = self.target_logdet[selection]

        def residual(parameter_vector: torch.Tensor) -> torch.Tensor:
            metric = functional_call(
                self.model,
                self.vectorizer.unpack(parameter_vector),
                (values, derivatives),
                strict=False,
            )
            if self.mode == "metric":
                relative = whiten_metric(metric, target, target_cholesky)
                real_components = torch.view_as_real(relative).reshape(
                    relative.shape[0], -1
                )
            else:
                logdet = training_log_volume(metric, "cholesky")
                real_components = (logdet - target_logdet).reshape(-1, 1)
            return (square_root_weights[:, None] * real_components).reshape(-1)

        return residual

    def residual(self, parameter_vector: torch.Tensor | None = None) -> torch.Tensor:
        value = self.theta if parameter_vector is None else parameter_vector
        return torch.cat(
            [self._chunk_residual_function(part)(value) for part in self._slices]
        )

    def jvp(self, direction: torch.Tensor) -> torch.Tensor:
        rows = []
        for part in self._slices:
            function = self._chunk_residual_function(part)
            _, tangent = jvp(function, (self.theta,), (direction,))
            rows.append(tangent)
        return torch.cat(rows)

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        if cotangent.ndim != 1 or cotangent.numel() != self.output_count:
            raise ValueError("teacher residual cotangent has the wrong shape")
        result = torch.zeros_like(self.theta)
        offset = 0
        for part in self._slices:
            function = self._chunk_residual_function(part)
            chunk_count = (part.stop - part.start) * self.components_per_point
            _, pullback = vjp(function, self.theta)
            result = result + pullback(cotangent[offset : offset + chunk_count])[0]
            offset += chunk_count
        return result

    def normal(self, cotangent: torch.Tensor) -> torch.Tensor:
        return self.jvp(self.vjp(cotangent))


def residual_metrics(residual: torch.Tensor) -> dict[str, float]:
    squared = float(real_inner(residual, residual))
    return {
        "squared_norm": squared,
        "rms": math.sqrt(max(0.0, squared)),
        "maximum_weighted_component": float(torch.max(torch.abs(residual))),
    }


def capture_fraction(baseline: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = max(float(real_inner(baseline, baseline)), np.finfo(float).tiny)
    return 1.0 - float(real_inner(candidate, candidate)) / denominator


def reachability_curve(
    name: str,
    probe: MatrixFreeTeacherMetricJacobian,
    holdout: MatrixFreeTeacherMetricJacobian,
    *,
    theta_norm: float,
    lanczos_steps: int,
    ridge_factors: tuple[float, ...],
    quiet: bool,
) -> dict[str, Any]:
    residual = probe.residual()
    holdout_residual = holdout.residual()
    residual_squared = float(real_inner(residual, residual))
    gradient = probe.vjp(residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(
        residual_squared,
        np.finfo(float).tiny,
    )
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        raise FloatingPointError(f"{name} Jacobian has no finite descent direction")

    def progress(iteration: int, alpha: float, beta: float) -> None:
        if not quiet:
            print(
                f"mode={name} lanczos={iteration}/{lanczos_steps} "
                f"alpha={alpha:.6e} beta={beta:.6e}",
                flush=True,
            )

    lanczos = lanczos_tridiagonal(
        probe.normal,
        residual,
        steps=lanczos_steps,
        callback=progress,
    )
    rows = []
    for factor in ridge_factors:
        ridge = factor * rayleigh_scale
        dual = ridge_dual_from_lanczos(lanczos, ridge)
        delta = -probe.vjp(dual)
        predicted_probe = residual + probe.jvp(delta)
        predicted_holdout = holdout_residual + holdout.jvp(delta)
        row = {
            "ridge_factor": float(factor),
            "ridge": float(ridge),
            "relative_parameter_step": float(
                vector_norm(delta) / max(theta_norm, np.finfo(float).tiny)
            ),
            "probe_linear_capture": capture_fraction(residual, predicted_probe),
            "holdout_linear_capture": capture_fraction(
                holdout_residual,
                predicted_holdout,
            ),
            "probe_linear_residual": residual_metrics(predicted_probe),
            "holdout_linear_residual": residual_metrics(predicted_holdout),
        }
        rows.append(row)
        if not quiet:
            print(
                f"mode={name} ridge_factor={factor:g} "
                f"probe_capture={row['probe_linear_capture']:.4f} "
                f"holdout_capture={row['holdout_linear_capture']:.4f}",
                flush=True,
            )
    return {
        "baseline_probe": residual_metrics(residual),
        "baseline_holdout": residual_metrics(holdout_residual),
        "probe_jt_residual_norm": gradient_norm,
        "residual_rayleigh_scale": rayleigh_scale,
        "lanczos": {
            "steps_completed": lanczos.operator_applications,
            "breakdown": lanczos.breakdown,
            "basis_orthogonality_error": lanczos.orthogonality_error,
            "tridiagonal_eigenvalue_minimum": float(
                torch.min(torch.linalg.eigvalsh(lanczos.tridiagonal))
            ),
            "tridiagonal_eigenvalue_maximum": float(
                torch.max(torch.linalg.eigvalsh(lanczos.tridiagonal))
            ),
        },
        "curve": rows,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output = args.output.expanduser().resolve()
    status = output.with_suffix(output.suffix + ".status.json")
    if output.exists() or status.exists():
        raise FileExistsError("refusing to overwrite an existing audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(status, {"state": "running", "phase": "loading"})

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model_path = args.model.expanduser().resolve()
    teacher_path = args.teacher_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    for path in (model_path, teacher_path, dataset_path, pullbacks_path):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
    source_degree = int(payload.get("source_degree", 1))
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    data = np.load(dataset_path, allow_pickle=False)
    count = len(data["X_val"])
    if args.candidate_limit:
        count = min(count, args.candidate_limit)
    probe_indices, holdout_indices = select_disjoint_indices(
        count,
        args.probe_size,
        args.holdout_size,
        seed=args.seed,
    )
    validation_pullbacks = np.load(pullbacks_path, mmap_mode="r")

    def arrays(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(data["X_val"][indices], dtype=np.float32),
            np.asarray(validation_pullbacks[indices]),
            np.asarray(data["y_val"][indices], dtype=np.float64),
        )

    probe_arrays = arrays(probe_indices)
    holdout_arrays = arrays(holdout_indices)
    target_probe = tensor_split(
        *probe_arrays,
        source_degree=source_degree,
        complex_dtype=torch.complex128,
        real_dtype=torch.float64,
        device=device,
    )
    target_holdout = tensor_split(
        *holdout_arrays,
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
    action_generator.manual_seed(args.seed + 1009)
    actions = fixed_fermat_actions_torch(
        args.group_samples,
        generator=action_generator,
        complex_dtype=torch.complex128,
        device=device,
    )
    write_json(status, {"state": "running", "phase": "teacher_targets"})
    _, probe_target_metric = teacher_targets(
        teacher,
        target_probe,
        actions,
        batch_size=args.teacher_batch_size,
        action_batch_size=args.teacher_action_batch_size,
    )
    _, holdout_target_metric = teacher_targets(
        teacher,
        target_holdout,
        actions,
        batch_size=args.teacher_batch_size,
        action_batch_size=args.teacher_action_batch_size,
    )
    del teacher, actions, target_probe, target_holdout

    student_dtype = (
        torch.complex64 if str(payload["precision"]) == "complex64" else torch.complex128
    )
    student_real_dtype = (
        torch.float32 if student_dtype == torch.complex64 else torch.float64
    )
    probe = tensor_split(
        *probe_arrays,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=student_real_dtype,
        device=device,
    )
    holdout = tensor_split(
        *holdout_arrays,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=student_real_dtype,
        device=device,
    )
    probe_target_metric = probe_target_metric.to(dtype=student_dtype)
    holdout_target_metric = holdout_target_metric.to(dtype=student_dtype)
    if not bool(torch.all(torch.linalg.eigvalsh(probe_target_metric) > 0)):
        raise FloatingPointError("probe teacher metric lost positivity after casting")
    if not bool(torch.all(torch.linalg.eigvalsh(holdout_target_metric) > 0)):
        raise FloatingPointError("holdout teacher metric lost positivity after casting")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=student_dtype)
    model.force_indexed_block_contraction = True
    vectorizer = ComplexParameterVectorizer.from_module(model)
    theta = vectorizer.pack(model)
    parameter_count = int(model.trainable_real_parameter_count)
    if args.expected_parameter_count and parameter_count != args.expected_parameter_count:
        raise RuntimeError(
            f"parameter-count gate failed: {parameter_count} != "
            f"{args.expected_parameter_count}"
        )
    theta_norm = float(vector_norm(theta))

    results = {}
    for mode in ("metric", "logdet"):
        write_json(status, {"state": "running", "phase": mode})
        probe_operator = MatrixFreeTeacherMetricJacobian(
            model,
            vectorizer,
            theta,
            probe,
            probe_target_metric,
            mode=mode,
            chunk_size=args.operator_chunk_size,
        )
        holdout_operator = MatrixFreeTeacherMetricJacobian(
            model,
            vectorizer,
            theta,
            holdout,
            holdout_target_metric,
            mode=mode,
            chunk_size=args.operator_chunk_size,
        )
        results[mode] = reachability_curve(
            mode,
            probe_operator,
            holdout_operator,
            theta_norm=theta_norm,
            lanczos_steps=args.lanczos_steps,
            ridge_factors=tuple(args.ridge_factors),
            quiet=args.quiet,
        )

    indices_path = output.with_suffix(output.suffix + ".indices.npz")
    np.savez_compressed(
        indices_path,
        probe_indices=probe_indices,
        holdout_indices=holdout_indices,
    )
    report = {
        "schema": "quintic-tn-teacher-metric-tangent-reachability-v1",
        "scientific_scope": {
            "purpose": (
                "Separate local tangent-space reachability from optimizer and "
                "finite-sample effects for a fixed Reynolds teacher."
            ),
            "metric_residual": (
                "sqrt(weight) times the real vectorization of "
                "g_T^{-1/2}(g_S-g_T)g_T^{-1/2}"
            ),
            "logdet_residual": "sqrt(weight) times (log det g_S - log det g_T)",
            "limitation": (
                "This is a local linear diagnostic in the stored parameter gauge, "
                "not a global approximation theorem."
            ),
        },
        "configuration": {
            "device": str(device),
            "student_precision": str(payload["precision"]),
            "teacher_precision": "complex128",
            "candidate_count": count,
            "probe_size": args.probe_size,
            "holdout_size": args.holdout_size,
            "group_samples": args.group_samples,
            "operator_chunk_size": args.operator_chunk_size,
            "lanczos_steps": args.lanczos_steps,
            "ridge_factors": list(args.ridge_factors),
            "seed": args.seed,
            "real_parameter_count": parameter_count,
        },
        "source": {
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "teacher_model": str(teacher_path),
            "teacher_model_sha256": sha256_file(teacher_path),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "validation_pullbacks": str(pullbacks_path),
            "validation_pullbacks_sha256": sha256_file(pullbacks_path),
            "indices": str(indices_path),
            "indices_sha256": sha256_file(indices_path),
        },
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output, report)
    write_json(status, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "output": str(output),
                "wall_seconds": report["wall_seconds"],
                "best_holdout_capture": {
                    mode: max(row["holdout_linear_capture"] for row in value["curve"])
                    for mode, value in results.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
