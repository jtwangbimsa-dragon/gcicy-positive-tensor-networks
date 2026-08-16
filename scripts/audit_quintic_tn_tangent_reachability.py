#!/usr/bin/env python3
"""Measure local Monge--Ampere residual reachability of a trained quintic TN."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.func import functional_call, jvp, vjp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_quintic_tn_scaling_preflight import (  # noqa: E402
    build_model,
    load_numpy_split,
    make_tensor_split,
    resolve_inputs,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    ratio_statistics,
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    training_log_volume,
    weighted_log_mean_exp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--blind-reference-run-dir", type=Path)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("blind", "validation"), default="validation")
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--probe-size", type=int, default=2000)
    parser.add_argument("--holdout-size", type=int, default=2000)
    parser.add_argument("--operator-chunk-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lanczos-steps", type=int, default=12)
    parser.add_argument(
        "--ridge-factors", type=float, nargs="+", default=(100.0, 10.0, 1.0, 0.1)
    )
    parser.add_argument(
        "--line-search-alphas", type=float, nargs="+", default=(0.125, 0.25, 0.5, 1.0)
    )
    parser.add_argument("--seed", type=int, default=202607206)
    parser.add_argument("--precision", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-canonicalization", action="store_true")
    parser.add_argument("--skip-global-scale-normalization", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integers = (
        args.probe_size,
        args.holdout_size,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.lanczos_steps,
    )
    if any(value <= 0 for value in positive_integers):
        raise ValueError("probe, holdout, chunk, batch, and Lanczos sizes must be positive")
    if args.candidate_limit < 0:
        raise ValueError("candidate limit must be non-negative")
    if not args.ridge_factors or any(value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be positive")
    if not args.line_search_alphas or any(
        value <= 0 or value > 1 for value in args.line_search_alphas
    ):
        raise ValueError("line-search alphas must lie in (0, 1]")


@dataclass(frozen=True)
class ComplexParameterVectorizer:
    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    counts: tuple[int, ...]

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        parameter_names: Iterable[str] | None = None,
    ) -> "ComplexParameterVectorizer":
        selected = None if parameter_names is None else frozenset(parameter_names)
        parameters = tuple(
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
            and (selected is None or name in selected)
        )
        if not parameters:
            raise ValueError("module has no trainable parameters")
        if selected is not None and selected != frozenset(name for name, _ in parameters):
            missing = sorted(selected - frozenset(name for name, _ in parameters))
            raise ValueError(f"requested parameters are unavailable: {missing}")
        parameter_kinds = {parameter.is_complex() for _, parameter in parameters}
        if len(parameter_kinds) != 1:
            raise ValueError(
                "selected parameters must be uniformly real or uniformly complex"
            )
        return cls(
            names=tuple(name for name, _ in parameters),
            shapes=tuple(parameter.shape for _, parameter in parameters),
            counts=tuple(parameter.numel() for _, parameter in parameters),
        )

    def pack(self, module: torch.nn.Module) -> torch.Tensor:
        named = dict(module.named_parameters())
        return torch.cat([named[name].detach().reshape(-1) for name in self.names])

    def unpack(self, vector: torch.Tensor) -> dict[str, torch.Tensor]:
        if vector.ndim != 1 or vector.numel() != sum(self.counts):
            raise ValueError("parameter vector has the wrong shape")
        result: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape, count in zip(
            self.names, self.shapes, self.counts, strict=True
        ):
            result[name] = vector[offset : offset + count].reshape(shape)
            offset += count
        return result


def real_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.is_complex() or right.is_complex():
        return torch.real(torch.vdot(left, right))
    return torch.dot(left, right)


def vector_norm(vector: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(real_inner(vector, vector), min=0.0))


class MatrixFreeResidualJacobian:
    """JVP/VJP operator for weighted fixed-normalization MA residuals."""

    def __init__(
        self,
        model: torch.nn.Module,
        vectorizer: ComplexParameterVectorizer,
        theta: torch.Tensor,
        dataset: dict[str, Any],
        *,
        fixed_log_kappa: float,
        chunk_size: int,
    ) -> None:
        self.model = model
        self.vectorizer = vectorizer
        self.theta = theta
        self.dataset = dataset
        self.fixed_log_kappa = float(fixed_log_kappa)
        self.chunk_size = int(chunk_size)
        self._slices = tuple(
            slice(start, min(start + chunk_size, dataset["count"]))
            for start in range(0, dataset["count"], chunk_size)
        )

    def _chunk_residual_function(self, selection: slice) -> Callable[[torch.Tensor], torch.Tensor]:
        values = self.dataset["values"][selection]
        derivatives = self.dataset["derivatives"][selection]
        log_omega = self.dataset["log_omega"][selection]
        square_root_weights = torch.sqrt(self.dataset["weights"][selection])
        log_kappa = log_omega.new_tensor(self.fixed_log_kappa)

        def residual(parameter_vector: torch.Tensor) -> torch.Tensor:
            metric = functional_call(
                self.model,
                self.vectorizer.unpack(parameter_vector),
                (values, derivatives),
                strict=False,
            )
            raw = training_log_volume(metric, "cholesky") - log_omega
            ratio = torch.exp(torch.clamp(raw - log_kappa, -20.0, 20.0))
            return square_root_weights * (ratio - 1.0)

        return residual

    def residual(self, parameter_vector: torch.Tensor | None = None) -> torch.Tensor:
        value = self.theta if parameter_vector is None else parameter_vector
        rows = [self._chunk_residual_function(part)(value) for part in self._slices]
        return torch.cat(rows)

    def jvp(self, direction: torch.Tensor) -> torch.Tensor:
        rows = []
        for part in self._slices:
            function = self._chunk_residual_function(part)
            _, tangent = jvp(function, (self.theta,), (direction,))
            rows.append(tangent)
        return torch.cat(rows)

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        if cotangent.ndim != 1 or cotangent.numel() != self.dataset["count"]:
            raise ValueError("residual cotangent has the wrong shape")
        result = torch.zeros_like(self.theta)
        offset = 0
        for part in self._slices:
            count = part.stop - part.start
            function = self._chunk_residual_function(part)
            _, pullback = vjp(function, self.theta)
            result = result + pullback(cotangent[offset : offset + count])[0]
            offset += count
        return result

    def normal(self, cotangent: torch.Tensor) -> torch.Tensor:
        return self.jvp(self.vjp(cotangent))


@dataclass(frozen=True)
class LanczosResult:
    basis: torch.Tensor
    tridiagonal: torch.Tensor
    initial_norm: float
    operator_applications: int
    breakdown: bool
    orthogonality_error: float


def lanczos_tridiagonal(
    operator: Callable[[torch.Tensor], torch.Tensor],
    initial: torch.Tensor,
    *,
    steps: int,
    tolerance: float = 1.0e-10,
    callback: Callable[[int, float, float], None] | None = None,
) -> LanczosResult:
    initial_norm_tensor = vector_norm(initial)
    initial_norm = float(initial_norm_tensor)
    if not np.isfinite(initial_norm) or initial_norm <= 0:
        raise ValueError("Lanczos initial vector must be finite and nonzero")
    basis: list[torch.Tensor] = []
    diagonal: list[torch.Tensor] = []
    off_diagonal: list[torch.Tensor] = []
    previous = torch.zeros_like(initial)
    previous_beta = initial.new_tensor(0.0)
    current = initial / initial_norm_tensor
    breakdown = False

    for iteration in range(steps):
        basis.append(current)
        candidate = operator(current) - previous_beta * previous
        alpha = real_inner(current, candidate)
        candidate = candidate - alpha * current
        for _ in range(2):
            for vector in basis:
                candidate = candidate - real_inner(vector, candidate) * vector
        beta = vector_norm(candidate)
        diagonal.append(alpha)
        if callback is not None:
            callback(iteration + 1, float(alpha), float(beta))
        scale = max(1.0, float(torch.max(torch.abs(torch.stack(diagonal)))))
        if iteration == steps - 1 or float(beta) <= tolerance * scale:
            breakdown = iteration < steps - 1
            break
        off_diagonal.append(beta)
        previous, current = current, candidate / beta
        previous_beta = beta

    basis_matrix = torch.stack(basis, dim=1)
    dimension = len(diagonal)
    tridiagonal = torch.zeros(
        (dimension, dimension), dtype=initial.dtype, device=initial.device
    )
    tridiagonal.diagonal().copy_(torch.stack(diagonal))
    if off_diagonal:
        values = torch.stack(off_diagonal[: dimension - 1])
        tridiagonal.diagonal(1).copy_(values)
        tridiagonal.diagonal(-1).copy_(values)
    gram = torch.transpose(basis_matrix, 0, 1) @ basis_matrix
    orthogonality_error = float(
        torch.linalg.matrix_norm(
            gram - torch.eye(dimension, dtype=gram.dtype, device=gram.device),
            ord=2,
        )
    )
    return LanczosResult(
        basis=basis_matrix,
        tridiagonal=tridiagonal,
        initial_norm=initial_norm,
        operator_applications=dimension,
        breakdown=breakdown,
        orthogonality_error=orthogonality_error,
    )


def ridge_dual_from_lanczos(
    result: LanczosResult, ridge: float
) -> torch.Tensor:
    coefficients = ridge_coefficients_from_lanczos(result, ridge)
    return result.basis @ coefficients


def ridge_coefficients_from_lanczos(
    result: LanczosResult, ridge: float
) -> torch.Tensor:
    """Solve the projected ridge system without lifting out of Krylov space."""

    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    right = torch.zeros(
        result.tridiagonal.shape[0],
        dtype=result.tridiagonal.dtype,
        device=result.tridiagonal.device,
    )
    right[0] = result.initial_norm
    coefficients = torch.linalg.solve(
        result.tridiagonal
        + ridge
        * torch.eye(
            len(right), dtype=result.tridiagonal.dtype, device=result.tridiagonal.device
        ),
        right,
    )
    return coefficients


def select_disjoint_indices(
    count: int, probe_size: int, holdout_size: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if probe_size + holdout_size > count:
        raise ValueError("probe and holdout sizes exceed the candidate pool")
    permutation = np.random.default_rng(seed).permutation(count)
    return permutation[:probe_size], permutation[probe_size : probe_size + holdout_size]


def subset_numpy_split(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray], indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.asarray(value[indices]) for value in arrays)  # type: ignore[return-value]


def model_raw_and_minimum(
    model: torch.nn.Module,
    vectorizer: ComplexParameterVectorizer,
    parameter_vector: torch.Tensor,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_rows = []
    minimum_rows = []
    parameters = vectorizer.unpack(parameter_vector)
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            metric = functional_call(
                model,
                parameters,
                (dataset["values"][start:stop], dataset["derivatives"][start:stop]),
                strict=False,
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(torch.isfinite(eigenvalues))):
                raise FloatingPointError("nonfinite metric eigenvalue")
            raw = torch.sum(torch.log(eigenvalues), dim=1)
            raw -= dataset["log_omega"][start:stop]
            raw_rows.append(raw.detach().cpu().numpy().astype(np.float64))
            minimum_rows.append(
                torch.min(eigenvalues, dim=1).values.detach().cpu().numpy().astype(np.float64)
            )
    return np.concatenate(raw_rows), np.concatenate(minimum_rows)


def empirical_statistics(
    model: torch.nn.Module,
    vectorizer: ComplexParameterVectorizer,
    parameter_vector: torch.Tensor,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[dict[str, Any], float]:
    raw, minimum = model_raw_and_minimum(
        model, vectorizer, parameter_vector, dataset, chunk_size=chunk_size
    )
    statistics, _ = ratio_statistics(raw, dataset["weights_numpy"], minimum)
    return statistics, weighted_log_mean_exp(raw, dataset["weights_numpy"])


def left_isometry_errors(model: torch.nn.Module) -> list[float]:
    errors = []
    for core in model.coefficient_cores[:-1]:
        left, right, physical = core.shape
        matrix = core.permute(0, 2, 1).reshape(left * physical, right)
        gram = torch.conj(matrix.T) @ matrix
        errors.append(
            float(
                torch.linalg.matrix_norm(
                    gram
                    - torch.eye(right, dtype=gram.dtype, device=gram.device),
                    ord=2,
                )
            )
        )
    return errors


def fixed_residual_metrics(residual: torch.Tensor) -> dict[str, float]:
    squared_norm = float(real_inner(residual, residual))
    return {
        "squared_norm": squared_norm,
        "rms": math.sqrt(max(0.0, squared_norm)),
        "maximum_weighted_component": float(torch.max(torch.abs(residual))),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = resolve_inputs(args)
    if args.output is None:
        paths["output"] = paths["run_dir"] / "tangent_reachability_smoke.json"
    started = time.perf_counter()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    complex_dtype = torch.complex64 if args.precision == "complex64" else torch.complex128
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64

    payload = torch.load(paths["model"], map_location="cpu", weights_only=False)
    if args.precision != payload.get("precision", "complex64"):
        from scripts.audit_quintic_tn_scaling_preflight import cast_artifact_precision

        payload = cast_artifact_precision(payload, args.precision)
    source_degree = int(payload.get("source_degree", 1))
    arrays = load_numpy_split(paths, args.split, args.candidate_limit)
    probe_indices, holdout_indices = select_disjoint_indices(
        len(arrays[0]), args.probe_size, args.holdout_size, seed=args.seed
    )
    probe_arrays = subset_numpy_split(arrays, probe_indices)
    holdout_arrays = subset_numpy_split(arrays, holdout_indices)
    probe = make_tensor_split(
        *probe_arrays,
        source_degree=source_degree,
        precision=args.precision,
        device=device,
    )
    holdout = make_tensor_split(
        *holdout_arrays,
        source_degree=source_degree,
        precision=args.precision,
        device=device,
    )

    model = build_model(payload, device)
    vectorizer = ComplexParameterVectorizer.from_module(model)
    theta_before = vectorizer.pack(model)
    preservation_count = min(128, probe["count"])
    preservation_dataset = {
        key: (value[:preservation_count] if torch.is_tensor(value) else value)
        for key, value in probe.items()
    }
    preservation_dataset["count"] = preservation_count
    before_raw, _ = model_raw_and_minimum(
        model,
        vectorizer,
        theta_before,
        preservation_dataset,
        chunk_size=args.eval_batch_size,
    )
    errors_before = left_isometry_errors(model)
    positive_floor_before = float(model.positive_floor)
    global_scale = 1.0
    if not args.skip_canonicalization:
        model.left_canonicalize_coefficient_cores_()
        if not args.skip_global_scale_normalization:
            global_scale = model.normalize_coefficient_chain_scale_()
    theta = vectorizer.pack(model)
    after_raw, _ = model_raw_and_minimum(
        model,
        vectorizer,
        theta,
        preservation_dataset,
        chunk_size=args.eval_batch_size,
    )
    errors_after = left_isometry_errors(model)
    canonicalization = {
        "applied": not args.skip_canonicalization,
        "global_scale_normalization_applied": (
            not args.skip_canonicalization
            and not args.skip_global_scale_normalization
        ),
        "global_purification_scale": global_scale,
        "positive_floor_before": positive_floor_before,
        "positive_floor_after": float(model.positive_floor),
        "parameter_norm_before": float(vector_norm(theta_before)),
        "parameter_norm_after": float(vector_norm(theta)),
        "maximum_raw_log_volume_change": float(np.max(np.abs(after_raw - before_raw))),
        "parameter_vector_relative_change": float(
            vector_norm(theta - theta_before)
            / max(float(vector_norm(theta_before)), np.finfo(float).tiny)
        ),
        "maximum_left_isometry_error_before": max(errors_before, default=0.0),
        "maximum_left_isometry_error_after": max(errors_after, default=0.0),
    }

    probe_baseline, probe_log_kappa = empirical_statistics(
        model, vectorizer, theta, probe, chunk_size=args.eval_batch_size
    )
    holdout_baseline, holdout_log_kappa = empirical_statistics(
        model, vectorizer, theta, holdout, chunk_size=args.eval_batch_size
    )
    probe_operator = MatrixFreeResidualJacobian(
        model,
        vectorizer,
        theta,
        probe,
        fixed_log_kappa=probe_log_kappa,
        chunk_size=args.operator_chunk_size,
    )
    holdout_operator = MatrixFreeResidualJacobian(
        model,
        vectorizer,
        theta,
        holdout,
        fixed_log_kappa=holdout_log_kappa,
        chunk_size=args.operator_chunk_size,
    )
    residual = probe_operator.residual()
    residual_norm_squared = float(real_inner(residual, residual))
    gradient = probe_operator.vjp(residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(
        residual_norm_squared, np.finfo(float).tiny
    )
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        raise FloatingPointError("residual Jacobian has no finite descent direction")

    def progress(iteration: int, alpha: float, beta: float) -> None:
        if not args.quiet:
            print(
                f"lanczos={iteration}/{args.lanczos_steps} "
                f"alpha={alpha:.6e} beta={beta:.6e}",
                flush=True,
            )

    lanczos = lanczos_tridiagonal(
        probe_operator.normal,
        residual,
        steps=args.lanczos_steps,
        callback=progress,
    )

    theta_norm = float(vector_norm(theta))
    rows = []
    for factor in args.ridge_factors:
        ridge = factor * rayleigh_scale
        dual = ridge_dual_from_lanczos(lanczos, ridge)
        delta = -probe_operator.vjp(dual)
        predicted = residual + probe_operator.jvp(delta)
        predicted_capture = 1.0 - float(real_inner(predicted, predicted)) / max(
            residual_norm_squared, np.finfo(float).tiny
        )
        line_rows = []
        for line_alpha in args.line_search_alphas:
            candidate = theta + line_alpha * delta
            probe_fixed = probe_operator.residual(candidate)
            line_rows.append(
                {
                    "alpha": float(line_alpha),
                    "probe_fixed": fixed_residual_metrics(probe_fixed),
                }
            )
        best_line = min(line_rows, key=lambda row: row["probe_fixed"]["squared_norm"])
        best_alpha = float(best_line["alpha"])
        selected = theta + best_alpha * delta
        holdout_fixed = holdout_operator.residual(selected)
        probe_empirical, _ = empirical_statistics(
            model, vectorizer, selected, probe, chunk_size=args.eval_batch_size
        )
        holdout_empirical, _ = empirical_statistics(
            model, vectorizer, selected, holdout, chunk_size=args.eval_batch_size
        )
        rows.append(
            {
                "ridge_factor": float(factor),
                "ridge": float(ridge),
                "relative_parameter_step_at_alpha_1": float(
                    vector_norm(delta) / max(theta_norm, np.finfo(float).tiny)
                ),
                "predicted_linear_capture_at_alpha_1": predicted_capture,
                "predicted_linear_residual": fixed_residual_metrics(predicted),
                "line_search": line_rows,
                "selected_alpha_from_probe": best_alpha,
                "selected_relative_parameter_step": float(
                    best_alpha
                    * vector_norm(delta)
                    / max(theta_norm, np.finfo(float).tiny)
                ),
                "holdout_fixed": fixed_residual_metrics(holdout_fixed),
                "probe_empirical": probe_empirical,
                "holdout_empirical": holdout_empirical,
            }
        )
        if not args.quiet:
            print(
                f"ridge_factor={factor:g} predicted_capture={predicted_capture:.4f} "
                f"alpha={best_alpha:g} "
                f"holdout_sigma={holdout_empirical['sigma_official_formula']:.6e}",
                flush=True,
            )

    index_path = paths["output"].with_suffix(".indices.npz")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        probe_indices=probe_indices,
        holdout_indices=holdout_indices,
    )
    report = {
        "schema": "quintic-tn-tangent-reachability-v1",
        "scientific_scope": {
            "purpose": (
                "Local representation-versus-optimization diagnostic; not a final "
                "generalization or global sup-norm certificate."
            ),
            "pool_status": "diagnostic subset of an existing architecture benchmark",
            "residual": (
                "sqrt(normalized quadrature weight) * (r-1), with empirical baseline "
                "log kappa held fixed while differentiating"
            ),
            "parameter_metric": (
                "Euclidean norm after left-canonical gauge; ridge controls remaining "
                "redundant tangent directions"
            ),
        },
        "configuration": {
            "device": str(device),
            "precision": args.precision,
            "split": args.split,
            "candidate_count": len(arrays[0]),
            "probe_size": args.probe_size,
            "holdout_size": args.holdout_size,
            "operator_chunk_size": args.operator_chunk_size,
            "eval_batch_size": args.eval_batch_size,
            "lanczos_steps_requested": args.lanczos_steps,
            "ridge_factors": list(args.ridge_factors),
            "line_search_alphas": list(args.line_search_alphas),
            "seed": args.seed,
        },
        "source": {
            "model": str(paths["model"]),
            "model_sha256": sha256_file(paths["model"]),
            "run_report": str(paths["report"]),
            "run_report_sha256": sha256_file(paths["report"]),
            "indices": str(index_path),
            "indices_sha256": sha256_file(index_path),
        },
        "parameterization": {
            "complex_parameters": int(theta.numel()),
            "real_parameters": int(2 * theta.numel()),
            "canonicalization": canonicalization,
        },
        "baseline": {
            "probe_empirical": probe_baseline,
            "holdout_empirical": holdout_baseline,
            "probe_empirical_log_kappa": probe_log_kappa,
            "holdout_empirical_log_kappa": holdout_log_kappa,
            "probe_fixed_residual": fixed_residual_metrics(residual),
            "probe_jt_residual_norm": gradient_norm,
            "residual_rayleigh_scale": rayleigh_scale,
        },
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
        "reachability_curve": rows,
        "timing_seconds": time.perf_counter() - started,
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    write_json(paths["output"], report)
    print(
        json.dumps(
            {
                "output": str(paths["output"]),
                "steps_completed": lanczos.operator_applications,
                "timing_seconds": report["timing_seconds"],
                "rows": [
                    {
                        "ridge_factor": row["ridge_factor"],
                        "predicted_capture": row["predicted_linear_capture_at_alpha_1"],
                        "selected_alpha": row["selected_alpha_from_probe"],
                        "holdout_sigma": row["holdout_empirical"]["sigma_official_formula"],
                    }
                    for row in rows
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
