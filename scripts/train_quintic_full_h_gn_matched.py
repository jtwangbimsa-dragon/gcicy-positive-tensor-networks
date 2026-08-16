#!/usr/bin/env python3
"""Train a quotient-basis quintic full H with matched HN Gauss--Newton steps."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.fermat_full_h import (  # noqa: E402
    FermatSymmetricFullH,
)
from scripts.audit_quintic_tn_scaling_preflight import (  # noqa: E402
    load_numpy_split,
    resolve_inputs,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    ComplexParameterVectorizer,
    MatrixFreeResidualJacobian,
    fixed_residual_metrics,
    lanczos_tridiagonal,
    real_inner,
    ridge_dual_from_lanczos,
    vector_norm,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    exponent_compositions,
    fermat_quintic_quotient_basis,
    fermat_quotient_fubini_study_h,
    fubini_study_h,
    prepare_features,
    ratio_statistics,
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    evaluate_raw,
)
from scripts.train_quintic_tn_one_site_lm import (  # noqa: E402
    copy_parameter_vector_into_model,
    direct_empirical_statistics,
    evaluate_validation_candidate,
    load_training_arrays,
)
from scripts.refine_quintic_hard_symmetry_channel_native_ma import (  # noqa: E402
    MatrixFreeNativeMARatioJacobian,
)


class DenseCholeskyAlgebraicMetric(torch.nn.Module):
    """Positive Hermitian algebraic metric with exactly ``n^2`` real parameters."""

    def __init__(
        self,
        initial_h: np.ndarray,
        *,
        normalization: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        matrix = np.asarray(initial_h, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("initial H must be square")
        if normalization <= 0 or not np.isfinite(normalization):
            raise ValueError("normalization must be finite and positive")
        cholesky = np.linalg.cholesky(matrix)
        size = len(matrix)
        rows, columns = np.tril_indices(size, k=-1)
        coordinates = np.concatenate(
            (
                np.log(np.real(np.diag(cholesky))),
                np.real(cholesky[rows, columns]),
                np.imag(cholesky[rows, columns]),
            )
        )
        if len(coordinates) != size**2:
            raise RuntimeError("dense Cholesky coordinate count is inconsistent")
        self.size = int(size)
        self.normalization = float(normalization)
        self.coordinates = torch.nn.Parameter(
            torch.tensor(coordinates, dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "lower_rows", torch.tensor(rows, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "lower_columns", torch.tensor(columns, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "reference_h",
            torch.tensor(matrix, dtype=torch.complex128, device=device),
        )

    @property
    def trainable_real_parameter_count(self) -> int:
        return int(self.coordinates.numel())

    def cholesky_factor(self) -> torch.Tensor:
        pair_count = self.lower_rows.numel()
        diagonal = torch.exp(self.coordinates[: self.size])
        real = self.coordinates[self.size : self.size + pair_count]
        imaginary = self.coordinates[self.size + pair_count :]
        lower = torch.diag(diagonal).to(torch.complex128)
        lower = lower.index_put(
            (self.lower_rows, self.lower_columns),
            torch.complex(real, imaginary),
        )
        trace = torch.sum(torch.real(torch.conj(lower) * lower))
        return lower * torch.sqrt(self.size / trace)

    def h_matrix(self) -> torch.Tensor:
        lower = self.cholesky_factor()
        return lower @ torch.conj(lower.T)

    def potential_and_metric(
        self, section_values: torch.Tensor, section_derivatives: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference_values = torch.einsum(
            "ab,nb->na", self.reference_h, section_values
        )
        reference_norm = torch.real(
            torch.einsum("na,na->n", torch.conj(section_values), reference_values)
        )
        if not bool(torch.all(reference_norm > 0)):
            raise FloatingPointError("reference section norm is not positive")
        inverse_scale = torch.rsqrt(reference_norm)
        values = section_values * inverse_scale[:, None]
        derivatives = section_derivatives * inverse_scale[:, None, None]

        lower = self.cholesky_factor()
        transformed_values = values @ torch.conj(lower)
        transformed_derivatives = torch.einsum(
            "mc,nmj->ncj", torch.conj(lower), derivatives
        )
        denominator = torch.sum(
            torch.real(torch.conj(transformed_values) * transformed_values), dim=1
        )
        if not bool(torch.all(denominator > 0)):
            raise FloatingPointError("dense H section norm is not positive")
        first = torch.einsum(
            "nci,ncj->nij",
            torch.conj(transformed_derivatives),
            transformed_derivatives,
        )
        gradient = torch.einsum(
            "nc,ncj->nj", torch.conj(transformed_values), transformed_derivatives
        )
        metric = first / denominator[:, None, None]
        metric = metric - (
            torch.conj(gradient)[:, :, None]
            * gradient[:, None, :]
            / denominator[:, None, None] ** 2
        )
        metric = self.normalization * metric
        metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        potential = self.normalization * (
            torch.log(denominator) + torch.log(reference_norm)
        )
        return potential, metric

    def forward(
        self, section_values: torch.Tensor, section_derivatives: torch.Tensor
    ) -> torch.Tensor:
        return self.potential_and_metric(section_values, section_derivatives)[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--blind-reference-run-dir", type=Path)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="continue from coordinates saved by a compatible earlier GN run",
    )
    parser.add_argument(
        "--initial-h-file",
        type=Path,
        help="initialize a Fermat-symmetric model from a compatible H artifact",
    )
    parser.add_argument(
        "--exclude-indices-file",
        type=Path,
        nargs="*",
        default=(),
        help="NPZ files whose train_indices and validation_indices are excluded",
    )
    parser.add_argument("--initial-h-key", default="global_h_matrix")
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument(
        "--fermat-symmetric",
        action="store_true",
        help="optimize the complete positive H cone invariant under (Z5)^4 semidirect S5",
    )
    parser.add_argument(
        "--fermat-conjugation",
        action="store_true",
        help="also impose the anti-holomorphic complex-conjugation symmetry",
    )
    parser.add_argument(
        "--native-normalization",
        action="store_true",
        help="differentiate the empirical volume normalization in the E2 residual",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=20000)
    parser.add_argument("--blind-limit", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--operator-chunk-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--gn-backend",
        choices=("matrix-free", "finite-difference"),
        default="matrix-free",
    )
    parser.add_argument("--lanczos-steps", type=int, default=8)
    parser.add_argument("--finite-difference-relative-step", type=float, default=2.0e-5)
    parser.add_argument("--maximum-finite-difference-parameters", type=int, default=512)
    parser.add_argument(
        "--ridge-factors", type=float, nargs="+", default=(1000.0, 100.0, 10.0, 1.0)
    )
    parser.add_argument(
        "--line-search-alphas", type=float, nargs="+", default=(0.125, 0.25, 0.5, 1.0)
    )
    parser.add_argument("--maximum-relative-step", type=float, default=2.0e-2)
    parser.add_argument(
        "--maximum-validation-tail-relative-degradation",
        type=float,
        default=2.0e-3,
    )
    parser.add_argument(
        "--maximum-validation-maximum-relative-degradation",
        type=float,
        help=(
            "optional separate allowance for the single largest residual; "
            "q999 and CVaR continue to use the robust-tail allowance"
        ),
    )
    parser.add_argument("--minimum-validation-chi-improvement", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=202607221)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-blind", action="store_true")
    parser.add_argument("--stop-after-rejected-iteration", action="store_true")
    parser.add_argument("--preserve-run-tail-baseline", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.degree,
        args.train_size,
        args.validation_size,
        args.feature_batch_size,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.lanczos_steps,
        args.finite_difference_relative_step,
        args.maximum_finite_difference_parameters,
        args.maximum_relative_step,
    )
    if any(value <= 0 for value in positive) or args.iterations < 0:
        raise ValueError("degree, sample, solver, and trust values must be valid")
    if args.fermat_conjugation and not args.fermat_symmetric:
        raise ValueError("Fermat conjugation requires the Fermat-symmetric model")
    if args.initial_checkpoint is not None and args.initial_h_file is not None:
        raise ValueError("choose either an initial checkpoint or an initial H file")
    if args.initial_h_file is not None and not args.fermat_symmetric:
        raise ValueError("initial H projection is registered for Fermat symmetry only")
    if (
        args.blind_limit < 0
        or args.minimum_validation_chi_improvement < 0
        or args.maximum_validation_tail_relative_degradation < 0
        or (
            args.maximum_validation_maximum_relative_degradation is not None
            and args.maximum_validation_maximum_relative_degradation < 0
        )
    ):
        raise ValueError("blind limit and validation tolerances must be non-negative")
    if not args.ridge_factors or any(value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be positive")
    if not args.line_search_alphas or any(
        value <= 0 or value > 1 for value in args.line_search_alphas
    ):
        raise ValueError("line-search alphas must lie in (0,1]")


def make_dense_split(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    exponents: np.ndarray,
    *,
    feature_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    x_values, labels, pullbacks = arrays
    values, derivatives = prepare_features(
        x_values,
        pullbacks,
        exponents,
        feature_batch_size,
        complex_dtype=np.dtype(np.complex128),
    )
    weights = np.asarray(labels[:, 0], dtype=np.float64)
    omega = np.asarray(labels[:, 1], dtype=np.float64)
    if not (
        np.all(np.isfinite(weights) & (weights > 0))
        and np.all(np.isfinite(omega) & (omega > 0))
    ):
        raise ValueError("weights and holomorphic-volume labels must be positive")
    weights = weights / np.sum(weights)
    return {
        "count": len(x_values),
        "values": torch.tensor(values, dtype=torch.complex128, device=device),
        "derivatives": torch.tensor(
            derivatives, dtype=torch.complex128, device=device
        ),
        "weights": torch.tensor(weights, dtype=torch.float64, device=device),
        "weights_numpy": weights,
        "log_omega": torch.log(
            torch.tensor(omega, dtype=torch.float64, device=device)
        ),
    }


def tail_summary(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "q999": float(statistics["abs_residual_weighted_quantiles"]["q0.9990"]),
        "cvar99": float(statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]),
        "maximum": float(statistics["abs_residual_weighted_quantiles"]["q1.0000"]),
    }


def random_subset_excluding(
    arrays: tuple[np.ndarray, ...],
    count: int,
    *,
    seed: int,
    excluded_indices: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    if not arrays or any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("subset arrays must be nonempty and have equal length")
    excluded = np.unique(np.asarray(excluded_indices, dtype=np.int64))
    if np.any((excluded < 0) | (excluded >= len(arrays[0]))):
        raise ValueError("excluded subset index is outside the source pool")
    available = np.ones(len(arrays[0]), dtype=bool)
    available[excluded] = False
    eligible = np.flatnonzero(available)
    if count > len(eligible):
        raise ValueError(
            f"requested {count} rows from only {len(eligible)} eligible entries"
        )
    rng = np.random.default_rng(seed)
    indices = np.asarray(
        rng.choice(eligible, size=count, replace=False), dtype=np.int64
    )
    return tuple(np.asarray(array[indices]) for array in arrays), indices


def fs_reproduction_check(
    model: torch.nn.Module,
    validation: dict[str, Any],
    reference_model: torch.nn.Module,
    reference_validation: dict[str, Any],
    *,
    count: int = 256,
) -> dict[str, float]:
    sample_count = min(
        count, validation["count"], reference_validation["count"]
    )
    with torch.no_grad():
        observed = model(
            validation["values"][:sample_count],
            validation["derivatives"][:sample_count],
        ).detach().cpu().numpy()
        expected = reference_model(
            reference_validation["values"][:sample_count],
            reference_validation["derivatives"][:sample_count],
        ).detach().cpu().numpy()
    relative = np.linalg.norm(observed - expected, axis=(1, 2)) / np.maximum(
        np.linalg.norm(expected, axis=(1, 2)), np.finfo(float).tiny
    )
    result = {
        "count": sample_count,
        "median_relative_tensor_error": float(np.median(relative)),
        "maximum_relative_tensor_error": float(np.max(relative)),
    }
    if result["maximum_relative_tensor_error"] > 2.0e-4:
        raise RuntimeError(f"quotient FS reproduction failed: {result}")
    return result


def parameter_trust_scale(theta: torch.Tensor) -> float:
    """Use a finite per-coordinate scale even when the identity is theta=0."""

    return max(float(vector_norm(theta)), math.sqrt(theta.numel()))


def tail_guard_passes(
    observed: dict[str, float],
    baseline: dict[str, float],
    *,
    robust_relative_degradation: float,
    maximum_relative_degradation: float,
) -> bool:
    robust_limit = 1.0 + robust_relative_degradation
    maximum_limit = 1.0 + maximum_relative_degradation
    return bool(
        observed["q999"] <= robust_limit * baseline["q999"]
        and observed["cvar99"] <= robust_limit * baseline["cvar99"]
        and observed["maximum"] <= maximum_limit * baseline["maximum"]
    )


def finite_difference_residual_jacobian(
    residual_function: Any,
    theta: torch.Tensor,
    *,
    relative_step: float,
    maximum_parameters: int,
    callback: Any = None,
) -> torch.Tensor:
    """Build a central-difference Jacobian for a small real parameter space."""

    if theta.is_complex():
        raise ValueError("finite-difference GN requires real coordinates")
    if theta.numel() > maximum_parameters:
        raise ValueError(
            "finite-difference GN parameter gate failed: "
            f"{theta.numel()} > {maximum_parameters}"
        )
    baseline = residual_function(theta)
    jacobian = torch.empty(
        (baseline.numel(), theta.numel()),
        dtype=baseline.dtype,
        device=baseline.device,
    )
    with torch.no_grad():
        for index in range(theta.numel()):
            scale = max(1.0, abs(float(theta[index])))
            step = relative_step * scale
            positive = theta.clone()
            negative = theta.clone()
            positive[index] += step
            negative[index] -= step
            jacobian[:, index] = (
                residual_function(positive) - residual_function(negative)
            ) / (2.0 * step)
            if callback is not None:
                callback(index + 1, theta.numel(), step)
    if not bool(torch.all(torch.isfinite(jacobian))):
        raise FloatingPointError("finite-difference residual Jacobian is nonfinite")
    return jacobian


def fermat_raw_and_minimum_from_coordinates(
    model: FermatSymmetricFullH,
    coordinates: torch.Tensor,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one Fermat H candidate while reusing its blocks over all points."""

    raw_rows = []
    minimum_rows = []
    with torch.no_grad():
        blocks = model.h_blocks(coordinates)
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            _, metric = model.potential_and_metric_from_blocks(
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
                blocks,
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(torch.isfinite(eigenvalues) & (eigenvalues > 0))):
                raise FloatingPointError("nonpositive cached Fermat metric")
            raw_rows.append(
                torch.sum(torch.log(eigenvalues), dim=1)
                - dataset["log_omega"][start:stop]
            )
            minimum_rows.append(torch.min(eigenvalues, dim=1).values)
    return torch.cat(raw_rows), torch.cat(minimum_rows)


def fermat_native_residual_from_coordinates(
    model: FermatSymmetricFullH,
    coordinates: torch.Tensor,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> torch.Tensor:
    raw, _ = fermat_raw_and_minimum_from_coordinates(
        model, coordinates, dataset, chunk_size=chunk_size
    )
    maximum = torch.max(raw)
    log_kappa = maximum + torch.log(
        torch.sum(dataset["weights"] * torch.exp(raw - maximum))
    )
    ratio = torch.exp(raw - log_kappa)
    return torch.sqrt(dataset["weights"]) * (ratio - 1.0)


def fermat_statistics_from_coordinates(
    model: FermatSymmetricFullH,
    coordinates: torch.Tensor,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> dict[str, Any]:
    raw, minimum = fermat_raw_and_minimum_from_coordinates(
        model, coordinates, dataset, chunk_size=chunk_size
    )
    statistics, _ = ratio_statistics(
        raw.detach().cpu().numpy().astype(np.float64),
        dataset["weights_numpy"],
        minimum.detach().cpu().numpy().astype(np.float64),
    )
    return statistics


def empirical_statistics_for_model(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> dict[str, Any]:
    if isinstance(model, FermatSymmetricFullH):
        return fermat_statistics_from_coordinates(
            model,
            model.coordinates.detach(),
            dataset,
            chunk_size=chunk_size,
        )
    return direct_empirical_statistics(model, dataset, chunk_size=chunk_size)


def optimize_iteration(
    model: torch.nn.Module,
    iteration: int,
    train: dict[str, Any],
    validation: dict[str, Any],
    *,
    fixed_log_kappa: float,
    run_baseline_tail: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.requires_grad_(True)
    vectorizer = ComplexParameterVectorizer.from_module(
        model, parameter_names=("coordinates",)
    )
    theta = vectorizer.pack(model)
    cached_fermat_backend = bool(
        args.gn_backend == "finite-difference"
        and args.native_normalization
        and isinstance(model, FermatSymmetricFullH)
    )
    if cached_fermat_backend:
        def validation_statistics(candidate: torch.Tensor) -> dict[str, Any]:
            return fermat_statistics_from_coordinates(
                model,
                candidate,
                validation,
                chunk_size=args.eval_batch_size,
            )
    else:
        def validation_statistics(candidate: torch.Tensor) -> dict[str, Any]:
            return evaluate_validation_candidate(
                model,
                vectorizer,
                candidate,
                validation,
                chunk_size=args.eval_batch_size,
            )
    baseline_validation = validation_statistics(theta)
    baseline_tail = tail_summary(baseline_validation)
    operator = None
    if cached_fermat_backend:
        exact_residual = lambda candidate: fermat_native_residual_from_coordinates(
            model,
            candidate,
            train,
            chunk_size=args.operator_chunk_size,
        )
        residual = exact_residual(theta)
    else:
        operator = (
            MatrixFreeNativeMARatioJacobian(
                model,
                vectorizer,
                theta,
                train,
                chunk_size=args.operator_chunk_size,
            )
            if args.native_normalization
            else MatrixFreeResidualJacobian(
                model,
                vectorizer,
                theta,
                train,
                fixed_log_kappa=fixed_log_kappa,
                chunk_size=args.operator_chunk_size,
            )
        )
        exact_residual = (
            operator.exact_residual
            if args.native_normalization
            else operator.residual
        )
        residual = operator.residual()
    baseline_train = fixed_residual_metrics(residual)
    baseline_energy = baseline_train["squared_norm"]
    jacobian = None
    lanczos = None
    if args.gn_backend == "finite-difference":
        def finite_difference_progress(
            completed: int, total: int, step: float
        ) -> None:
            if not args.quiet:
                print(
                    f"iteration={iteration} finite_difference="
                    f"{completed}/{total} step={step:.6e}",
                    flush=True,
                )

        jacobian = finite_difference_residual_jacobian(
            exact_residual,
            theta,
            relative_step=args.finite_difference_relative_step,
            maximum_parameters=args.maximum_finite_difference_parameters,
            callback=finite_difference_progress,
        )
        gradient = torch.transpose(jacobian, 0, 1) @ residual
    else:
        if operator is None:
            raise RuntimeError("matrix-free GN operator was not constructed")
        gradient = operator.vjp(residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(
        baseline_energy, np.finfo(float).tiny
    )
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        raise FloatingPointError("dense H has no finite residual descent direction")

    def progress(step: int, alpha: float, beta: float) -> None:
        if not args.quiet:
            print(
                f"iteration={iteration} lanczos={step}/{args.lanczos_steps} "
                f"alpha={alpha:.6e} beta={beta:.6e}",
                flush=True,
            )

    gram = None
    if args.gn_backend == "finite-difference":
        if jacobian is None:
            raise RuntimeError("finite-difference Jacobian was not constructed")
        gram = torch.transpose(jacobian, 0, 1) @ jacobian
        gram = 0.5 * (gram + torch.transpose(gram, 0, 1))
        eigenvalues = torch.linalg.eigvalsh(gram)
        solver_diagnostics = {
            "backend": "finite_difference",
            "steps_completed": int(theta.numel()),
            "breakdown": False,
            "orthogonality_error": None,
            "eigenvalue_minimum": float(torch.min(eigenvalues)),
            "eigenvalue_maximum": float(torch.max(eigenvalues)),
            "finite_difference_relative_step": (
                args.finite_difference_relative_step
            ),
        }
    else:
        lanczos = lanczos_tridiagonal(
            # This branch necessarily has a matrix-free operator.
            operator.normal,
            residual,
            steps=args.lanczos_steps,
            callback=progress,
        )
        solver_diagnostics = {
            "backend": "matrix_free_lanczos",
            "steps_completed": lanczos.operator_applications,
            "breakdown": lanczos.breakdown,
            "orthogonality_error": lanczos.orthogonality_error,
            "eigenvalue_minimum": float(
                torch.min(torch.linalg.eigvalsh(lanczos.tridiagonal))
            ),
            "eigenvalue_maximum": float(
                torch.max(torch.linalg.eigvalsh(lanczos.tridiagonal))
            ),
            "finite_difference_relative_step": None,
        }
    theta_norm = float(vector_norm(theta))
    trust_scale = parameter_trust_scale(theta)
    maximum_tail_degradation = (
        args.maximum_validation_tail_relative_degradation
        if args.maximum_validation_maximum_relative_degradation is None
        else args.maximum_validation_maximum_relative_degradation
    )
    candidates = []
    candidate_count = len(args.ridge_factors) * len(args.line_search_alphas)
    for factor in args.ridge_factors:
        ridge = factor * rayleigh_scale
        if args.gn_backend == "finite-difference":
            if gram is None or jacobian is None:
                raise RuntimeError("finite-difference GN state is incomplete")
            delta = torch.linalg.solve(
                gram
                + ridge
                * torch.eye(
                    theta.numel(), dtype=theta.dtype, device=theta.device
                ),
                -gradient,
            )
            predicted = residual + jacobian @ delta
        else:
            if lanczos is None or operator is None:
                raise RuntimeError("Lanczos state is incomplete")
            dual = ridge_dual_from_lanczos(lanczos, ridge)
            delta = -operator.vjp(dual)
            predicted = residual + operator.jvp(delta)
        predicted_capture = 1.0 - float(real_inner(predicted, predicted)) / max(
            baseline_energy, np.finfo(float).tiny
        )
        for alpha in args.line_search_alphas:
            candidate = theta + alpha * delta
            relative_step = float(
                alpha * vector_norm(delta) / trust_scale
            )
            train_residual = exact_residual(candidate)
            train_metrics = fixed_residual_metrics(train_residual)
            validation_metrics = validation_statistics(candidate)
            observed_tail = tail_summary(validation_metrics)
            local_tail = tail_guard_passes(
                observed_tail,
                baseline_tail,
                robust_relative_degradation=(
                    args.maximum_validation_tail_relative_degradation
                ),
                maximum_relative_degradation=maximum_tail_degradation,
            )
            cumulative_tail = tail_guard_passes(
                observed_tail,
                run_baseline_tail,
                robust_relative_degradation=(
                    args.maximum_validation_tail_relative_degradation
                ),
                maximum_relative_degradation=maximum_tail_degradation,
            )
            candidates.append(
                {
                    "ridge_factor": float(factor),
                    "ridge": float(ridge),
                    "alpha": float(alpha),
                    "relative_step": relative_step,
                    "predicted_capture_at_alpha_1": predicted_capture,
                    "train_fixed": train_metrics,
                    "validation": validation_metrics,
                    "validation_tail": observed_tail,
                    "local_tail_guard_passed": local_tail,
                    "cumulative_tail_guard_passed": cumulative_tail,
                    "eligible": bool(
                        train_metrics["squared_norm"] < baseline_energy
                        and relative_step <= args.maximum_relative_step
                        and local_tail
                        and cumulative_tail
                    ),
                    "parameter_vector": candidate,
                }
            )
            if not args.quiet:
                print(
                    f"iteration={iteration} "
                    f"candidate={len(candidates)}/{candidate_count} "
                    f"ridge_factor={factor:.6g} alpha={alpha:.6g} "
                    f"train_e2={train_metrics['squared_norm']:.6e} "
                    f"validation_chi="
                    f"{validation_metrics['weighted_rms_abs_residual']:.6e} "
                    f"eligible={candidates[-1]['eligible']}",
                    flush=True,
                )
    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda row: row["validation"]["weighted_rms_abs_residual"],
        )
        if eligible
        else None
    )
    baseline_chi = float(baseline_validation["weighted_rms_abs_residual"])
    selected_chi = (
        float(selected["validation"]["weighted_rms_abs_residual"])
        if selected is not None
        else float("inf")
    )
    accepted = bool(
        selected is not None
        and baseline_chi - selected_chi
        > args.minimum_validation_chi_improvement
    )
    if accepted:
        copy_parameter_vector_into_model(model, vectorizer, selected["parameter_vector"])
    model.requires_grad_(False)
    return {
        "iteration": iteration,
        "active_real_parameters": int(theta.numel()),
        "baseline": {
            "train_fixed": baseline_train,
            "validation": baseline_validation,
            "validation_tail": baseline_tail,
            "run_validation_tail": run_baseline_tail,
            "jt_residual_norm": gradient_norm,
            "residual_rayleigh_scale": rayleigh_scale,
            "parameter_norm": theta_norm,
            "parameter_trust_scale": trust_scale,
        },
        "lanczos": solver_diagnostics,
        "candidates": [
            {key: value for key, value in row.items() if key != "parameter_vector"}
            for row in candidates
        ],
        "selected": (
            None
            if selected is None
            else {
                key: value
                for key, value in selected.items()
                if key != "parameter_vector"
            }
        ),
        "accepted": accepted,
        "validation_chi_change": (
            selected_chi - baseline_chi if selected is not None else None
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})
    try:
        paths = resolve_inputs(args)
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        reference_payload = torch.load(
            paths["model"], map_location="cpu", weights_only=False
        )
        fixed_log_kappa = float(reference_payload["fixed_log_kappa"])
        ambient_exponents, exponents, quotient_lift = fermat_quintic_quotient_basis(
            args.degree
        )
        initial_h = fermat_quotient_fubini_study_h(
            ambient_exponents, quotient_lift
        )
        model = (
            FermatSymmetricFullH(
                exponents,
                initial_h,
                normalization=1.0 / (math.pi * args.degree),
                device=device,
                conjugation_invariant=args.fermat_conjugation,
            )
            if args.fermat_symmetric
            else DenseCholeskyAlgebraicMetric(
                initial_h,
                normalization=1.0 / (math.pi * args.degree),
                device=device,
            )
        )
        expected_parameters = model.trainable_real_parameter_count
        if model.trainable_real_parameter_count != expected_parameters:
            raise RuntimeError("full-H parameter count gate failed")

        excluded_train_rows = []
        excluded_validation_rows = []
        exclusion_sources = []
        for exclusion_path in args.exclude_indices_file:
            resolved_exclusion_path = exclusion_path.expanduser().resolve()
            if not resolved_exclusion_path.exists():
                raise FileNotFoundError(resolved_exclusion_path)
            exclusion = np.load(resolved_exclusion_path, allow_pickle=False)
            for key in ("train_indices", "validation_indices"):
                if key not in exclusion.files:
                    raise KeyError(
                        f"{key!r} is absent from {resolved_exclusion_path}"
                    )
            excluded_train_rows.append(
                np.asarray(exclusion["train_indices"], dtype=np.int64)
            )
            excluded_validation_rows.append(
                np.asarray(exclusion["validation_indices"], dtype=np.int64)
            )
            exclusion_sources.append(
                {
                    "path": str(resolved_exclusion_path),
                    "sha256": sha256_file(resolved_exclusion_path),
                }
            )
        excluded_train = (
            np.unique(np.concatenate(excluded_train_rows))
            if excluded_train_rows
            else np.empty(0, dtype=np.int64)
        )
        excluded_validation = (
            np.unique(np.concatenate(excluded_validation_rows))
            if excluded_validation_rows
            else np.empty(0, dtype=np.int64)
        )

        training_arrays, train_indices = random_subset_excluding(
            load_training_arrays(paths),
            args.train_size,
            seed=args.seed,
            excluded_indices=excluded_train,
        )
        validation_pool = load_numpy_split(paths, "validation", 0)
        validation_arrays, validation_indices = random_subset_excluding(
            validation_pool,
            args.validation_size,
            seed=args.seed + 1,
            excluded_indices=excluded_validation,
        )
        write_json(status_path, {"state": "running", "phase": "features"})
        train = make_dense_split(
            training_arrays,
            exponents,
            feature_batch_size=args.feature_batch_size,
            device=device,
        )
        validation = make_dense_split(
            validation_arrays,
            exponents,
            feature_batch_size=args.feature_batch_size,
            device=device,
        )
        indices_path = output_dir / "optimization_indices.npz"
        np.savez_compressed(
            indices_path,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
        fs_sample_count = min(256, len(validation_pool[0]))
        fs_arrays = tuple(
            np.asarray(array[:fs_sample_count]) for array in validation_pool
        )
        fs_validation = make_dense_split(
            fs_arrays,
            exponents,
            feature_batch_size=args.feature_batch_size,
            device=device,
        )
        degree_one_exponents = exponent_compositions(1, 5)
        fs_reference_validation = make_dense_split(
            fs_arrays,
            degree_one_exponents,
            feature_batch_size=args.feature_batch_size,
            device=device,
        )
        fs_reference_model = DenseCholeskyAlgebraicMetric(
            fubini_study_h(degree_one_exponents),
            normalization=1.0 / math.pi,
            device=device,
        )
        fs_reference_model.requires_grad_(False)
        fs_check = fs_reproduction_check(
            model,
            fs_validation,
            fs_reference_model,
            fs_reference_validation,
        )
        fs_check["reference"] = "degree-one ambient FS metric on identical points"
        del fs_reference_model, fs_reference_validation, fs_validation
        initial_checkpoint = None
        if args.initial_checkpoint is not None:
            initial_checkpoint_path = args.initial_checkpoint.expanduser().resolve()
            initial_checkpoint = torch.load(
                initial_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if int(initial_checkpoint["degree"]) != args.degree:
                raise ValueError("initial checkpoint degree does not match the run")
            initial_coordinates = torch.as_tensor(
                initial_checkpoint["coordinates"],
                dtype=model.coordinates.dtype,
                device=device,
            )
            if initial_coordinates.shape != model.coordinates.shape:
                raise ValueError(
                    "initial checkpoint coordinates do not match the selected model"
                )
            with torch.no_grad():
                model.coordinates.copy_(initial_coordinates)
        elif args.initial_h_file is not None:
            initial_h_path = args.initial_h_file.expanduser().resolve()
            initial_h_artifact = np.load(initial_h_path, allow_pickle=False)
            if args.initial_h_key not in initial_h_artifact.files:
                raise KeyError(
                    f"{args.initial_h_key!r} is absent from {initial_h_path}"
                )
            model.set_coordinates_from_h_(
                np.asarray(
                    initial_h_artifact[args.initial_h_key],
                    dtype=np.complex128,
                )
            )
        validation_baseline = empirical_statistics_for_model(
            model, validation, chunk_size=args.eval_batch_size
        )
        run_baseline_tail = tail_summary(validation_baseline)
        if args.preserve_run_tail_baseline:
            if initial_checkpoint is None:
                raise ValueError(
                    "preserving the run tail baseline requires an initial checkpoint"
                )
            saved_tail = initial_checkpoint.get("run_baseline_tail")
            if not isinstance(saved_tail, dict) or set(saved_tail) != {
                "q999",
                "cvar99",
                "maximum",
            }:
                raise ValueError("initial checkpoint has no compatible tail baseline")
            run_baseline_tail = {
                key: float(saved_tail[key]) for key in saved_tail
            }

        blind = None
        blind_baseline = None
        if not args.skip_blind:
            blind_arrays = load_numpy_split(paths, "blind", args.blind_limit)
            blind = make_dense_split(
                blind_arrays,
                exponents,
                feature_batch_size=args.feature_batch_size,
                device=device,
            )
            blind_baseline = empirical_statistics_for_model(
                model, blind, chunk_size=args.eval_batch_size
            )

        rows = []
        for iteration in range(1, args.iterations + 1):
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "gauss_newton",
                    "iteration": iteration,
                    "iterations": args.iterations,
                },
            )
            row = optimize_iteration(
                model,
                iteration,
                train,
                validation,
                fixed_log_kappa=fixed_log_kappa,
                run_baseline_tail=run_baseline_tail,
                args=args,
            )
            rows.append(row)
            write_json(output_dir / "optimization_history.json", {"rows": rows})
            torch.save(
                {
                    "schema": "quintic-full-h-gn-checkpoint-v1",
                    "degree": args.degree,
                    "coordinates": model.coordinates.detach().cpu(),
                    "completed_iterations": iteration,
                    "fixed_log_kappa": fixed_log_kappa,
                    "run_baseline_tail": run_baseline_tail,
                    "fermat_symmetric": args.fermat_symmetric,
                    "fermat_conjugation": args.fermat_conjugation,
                    "native_normalization": args.native_normalization,
                },
                output_dir / "optimization_checkpoint.pt",
            )
            if args.stop_after_rejected_iteration and not row["accepted"]:
                if not args.quiet:
                    print(
                        f"iteration={iteration} rejected; stopping deterministic "
                        "continuation",
                        flush=True,
                    )
                break

        validation_final = (
            validation_baseline
            if args.iterations == 0
            else empirical_statistics_for_model(
                model, validation, chunk_size=args.eval_batch_size
            )
        )
        blind_final = (
            None
            if blind is None
            else (
                blind_baseline
                if args.iterations == 0
                else empirical_statistics_for_model(
                    model, blind, chunk_size=args.eval_batch_size
                )
            )
        )
        h_matrix = model.h_matrix().detach().cpu().numpy()
        artifact_path = output_dir / "best_h_metric.npz"
        np.savez_compressed(
            artifact_path,
            degree=np.asarray(args.degree),
            ambient_exponents=ambient_exponents,
            exponents=exponents,
            quotient_lift=quotient_lift,
            initial_h_matrix=initial_h,
            global_h_matrix=h_matrix,
            coordinates=model.coordinates.detach().cpu().numpy(),
            normalization=np.asarray(model.normalization),
            fixed_log_kappa=np.asarray(fixed_log_kappa),
        )
        report = {
            "schema": (
                "quintic-fermat-symmetric-full-h-native-gn-v1"
                if args.fermat_symmetric and args.native_normalization
                else "quintic-full-h-matched-gn-v1"
            ),
            "scientific_scope": {
                "geometry": "Fermat quintic X_5 in P^4",
                "objective": (
                    "sampled Headrick--Nassar sum w_i (r_i-1)^2 with "
                    "differentiated empirical volume normalization"
                    if args.native_normalization
                    else "sampled fixed-normalization sum w_i (r_i-1)^2"
                ),
                "optimizer": "matrix-free damped Gauss-Newton/Lanczos",
                "symmetry_usage": (
                    (
                        "exact (Z5)^4 phase blocks, exact S5 Reynolds tying, "
                        "and exact complex conjugation"
                    )
                    if args.fermat_conjugation
                    else "exact (Z5)^4 phase blocks and exact S5 Reynolds tying"
                    if args.fermat_symmetric
                    else "none"
                ),
                "pool_status": "existing architecture benchmark, not sealed final test",
            },
            "configuration": {
                **{
                    key: value
                    for key, value in vars(args).items()
                    if not isinstance(value, Path)
                    and key != "exclude_indices_file"
                },
                "exclude_indices_files": [
                    str(path.expanduser().resolve())
                    for path in args.exclude_indices_file
                ],
                "run_dir": str(args.run_dir),
                "output_dir": str(output_dir),
                "device": str(device),
            },
            "basis": {
                "degree": args.degree,
                "ambient_monomial_count": int(len(ambient_exponents)),
                "section_count": int(len(exponents)),
                "unrestricted_real_hermitian_parameter_count": int(
                    len(exponents) ** 2
                ),
                "real_hermitian_parameter_count": expected_parameters,
                "registered_real_parameter_count": expected_parameters,
                "effective_invariant_real_dimension": (
                    model.effective_invariant_real_dimension
                    if args.fermat_symmetric
                    else expected_parameters
                ),
                "phase_block_real_dimension": (
                    model.blueprint.phase_block_real_dimension
                    if args.fermat_symmetric
                    else int(len(exponents) ** 2)
                ),
                "representative_reynolds_real_parameter_count": (
                    model.blueprint.representative_real_parameter_count
                    if args.fermat_symmetric
                    else int(len(exponents) ** 2)
                ),
                "quotient_convention": (
                    "standard monomials exponent(z0)<5 with the Fermat relation"
                ),
                "fs_reproduction": fs_check,
            },
            "source": {
                "reference_model": str(paths["model"]),
                "reference_model_sha256": sha256_file(paths["model"]),
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "excluded_indices": {
                    "sources": exclusion_sources,
                    "unique_train_count": int(len(excluded_train)),
                    "unique_validation_count": int(len(excluded_validation)),
                },
                "initial_checkpoint": (
                    None
                    if args.initial_checkpoint is None
                    else str(args.initial_checkpoint.expanduser().resolve())
                ),
                "initial_checkpoint_sha256": (
                    None
                    if args.initial_checkpoint is None
                    else sha256_file(args.initial_checkpoint.expanduser().resolve())
                ),
                "initial_h_file": (
                    None
                    if args.initial_h_file is None
                    else str(args.initial_h_file.expanduser().resolve())
                ),
                "initial_h_file_sha256": (
                    None
                    if args.initial_h_file is None
                    else sha256_file(args.initial_h_file.expanduser().resolve())
                ),
                "initial_h_key": (
                    None if args.initial_h_file is None else args.initial_h_key
                ),
            },
            "optimization": {
                "fixed_log_kappa": fixed_log_kappa,
                "validation_baseline": validation_baseline,
                "rows": rows,
                "validation_final": validation_final,
            },
            "benchmark": {
                "baseline": blind_baseline,
                "final": blind_final,
            },
            "artifacts": {
                "h_metric": str(artifact_path),
                "h_metric_sha256": sha256_file(artifact_path),
                "optimization_checkpoint": str(
                    output_dir / "optimization_checkpoint.pt"
                ),
                "optimization_checkpoint_sha256": (
                    sha256_file(output_dir / "optimization_checkpoint.pt")
                    if args.iterations
                    else None
                ),
            },
            "timing_seconds": time.perf_counter() - started,
        }
        report_path = output_dir / "report.json"
        write_json(report_path, report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
            },
        )
        print(
            json.dumps(
                {
                    "accepted_iterations": [
                        row["iteration"] for row in rows if row["accepted"]
                    ],
                    "parameters": expected_parameters,
                    "validation_sigma_baseline": validation_baseline[
                        "sigma_official_formula"
                    ],
                    "validation_sigma_final": validation_final[
                        "sigma_official_formula"
                    ],
                    "blind_sigma_final": (
                        None
                        if blind_final is None
                        else blind_final["sigma_official_formula"]
                    ),
                    "report": str(report_path),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "exception",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
