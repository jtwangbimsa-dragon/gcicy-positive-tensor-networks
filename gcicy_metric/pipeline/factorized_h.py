"""Train product-factor algebraic metrics and export a standard H artifact.

The first implementation deliberately supports two equal-degree factors.  If

    F = (s^* H_1 s) (s^* H_2 s),

then the degree-doubled metric is the arithmetic mean of the two source-level
metrics.  Training therefore uses two small H matrices and only materializes
the full doubled-degree H matrix once, during export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .active_set import file_sha256
from .adapter import (
    GCICYAdapter,
    HMetricArtifact,
    H_POSITIVE_RELATIVE_FLOOR,
    positive_hermitian_projection,
)
from .audit import normalized_volume_ratios, standard_errors
from .parallel_sampling import sample_points_parallel
from .risk import (
    smooth_upper_log_ratio_excess_torch,
    weighted_cvar_numpy,
    weighted_cvar_selected_linearization_torch,
    weighted_cvar_torch,
)


FACTORIZED_H_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FactorizedHTrainingRequest:
    """Configuration for the two-factor ``k -> 2k`` pilot."""

    model_seed: int
    exact_model: bool
    source_artifact: Path
    target_degree: tuple[int, ...] | None = None
    expected_source_section_count: int | None = None
    expected_target_section_count: int | None = None
    factor_count: int = 2
    device: str = "auto"
    precision: str = "complex64"
    importance_weighted: bool = True
    lift_basis_points: int = 512
    lift_basis_seed: int = 70499
    train_points: int = 8192
    train_seed: int = 72001
    checkpoint_points: int = 2048
    checkpoint_seed: int = 72002
    export_points: int = 8192
    export_seed: int = 72003
    equivalence_points: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    points_per_optimizer_step: int = 64
    shuffle_seed: int = 72801
    torch_seed: int = 72800
    symmetry_breaking_scale: float = 1e-3
    symmetry_breaking_seed: int = 72802
    sigma_loss_weight: float = 1.0
    volume_ratio_l2_loss_weight: float = 0.0
    positive_log_ratio_cvar_loss_weight: float = 0.0
    positive_log_ratio_cvar_tail_fraction: float = 0.01
    upper_log_ratio_cvar_loss_weight: float = 0.0
    upper_log_ratio_cvar_tail_fraction: float = 0.01
    upper_log_ratio_threshold: float = 3.0
    upper_log_ratio_smooth_temperature: float = 0.1
    centered_log_variance_loss_weight: float = 0.0
    metric_barrier_weight: float = 0.0
    drift_weight: float = 0.0
    relative_log_spectrum_loss_weight: float = 0.0
    gradient_clip_norm: float | None = 5.0
    selection_volume_ratio_l2_weight: float = 1e-12
    selection_positive_log_ratio_cvar_weight: float = 0.0
    training_sampling_mode: str = "one_random_root_per_fibre"
    sampling_workers: int = 1
    sampling_cluster_size: int = 4
    sampling_backend: str = "process"
    lift_rcond: float = 1e-10
    maximum_lift_relation_error: float = 1e-8
    maximum_metric_equivalence_error: float = 1e-7
    require_source_sigma_improvement: bool = False

    def __post_init__(self) -> None:
        if self.factor_count != 2:
            raise ValueError(
                "the first factorized-H implementation requires factor_count=2"
            )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.precision not in {"complex64", "complex128"}:
            raise ValueError("precision must be complex64 or complex128")
        if self.training_sampling_mode not in {
            "complete_fibres",
            "one_random_root_per_fibre",
        }:
            raise ValueError(
                "training_sampling_mode must be complete_fibres or "
                "one_random_root_per_fibre"
            )
        integer_counts = (
            self.lift_basis_points,
            self.train_points,
            self.checkpoint_points,
            self.export_points,
            self.equivalence_points,
            self.points_per_optimizer_step,
            self.sampling_workers,
            self.sampling_cluster_size,
        )
        if min(integer_counts) <= 0:
            raise ValueError(
                "all point, batch, worker, and cluster counts must be positive"
            )
        if self.equivalence_points > self.export_points:
            raise ValueError("equivalence_points cannot exceed export_points")
        for name, count in (
            ("expected_source_section_count", self.expected_source_section_count),
            ("expected_target_section_count", self.expected_target_section_count),
        ):
            if count is not None and count <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.epochs < 0:
            raise ValueError("epochs cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.symmetry_breaking_scale < 0:
            raise ValueError("symmetry_breaking_scale cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        nonnegative_weights = (
            self.sigma_loss_weight,
            self.volume_ratio_l2_loss_weight,
            self.positive_log_ratio_cvar_loss_weight,
            self.upper_log_ratio_cvar_loss_weight,
            self.centered_log_variance_loss_weight,
            self.metric_barrier_weight,
            self.drift_weight,
            self.relative_log_spectrum_loss_weight,
            self.selection_volume_ratio_l2_weight,
            self.selection_positive_log_ratio_cvar_weight,
        )
        if min(nonnegative_weights) < 0 or not any(nonnegative_weights[:5]):
            raise ValueError(
                "loss weights must be non-negative with one data loss enabled"
            )
        if not 0 < self.positive_log_ratio_cvar_tail_fraction <= 1:
            raise ValueError("CVaR tail fraction must lie in (0, 1]")
        if not 0 < self.upper_log_ratio_cvar_tail_fraction <= 1:
            raise ValueError("upper-log CVaR tail fraction must lie in (0, 1]")
        if (
            not np.isfinite(self.upper_log_ratio_threshold)
            or self.upper_log_ratio_threshold <= 1.0
        ):
            raise ValueError("upper-log ratio threshold must be greater than one")
        if (
            not np.isfinite(self.upper_log_ratio_smooth_temperature)
            or self.upper_log_ratio_smooth_temperature <= 0.0
        ):
            raise ValueError("upper-log smooth temperature must be positive")
        if self.sampling_backend not in {"process", "thread"}:
            raise ValueError("sampling_backend must be process or thread")
        if self.training_sampling_mode == "one_random_root_per_fibre":
            if self.sampling_cluster_size <= 1:
                raise ValueError("independent-fibre sampling requires cluster_size > 1")
        if self.sampling_workers > 1:
            counts = (
                self.lift_basis_points,
                self.checkpoint_points,
                self.export_points,
            )
            if any(count % self.sampling_cluster_size for count in counts):
                raise ValueError(
                    "parallel non-training point counts must be divisible by cluster_size"
                )
        if (
            min(
                self.lift_rcond,
                self.maximum_lift_relation_error,
                self.maximum_metric_equivalence_error,
            )
            <= 0
        ):
            raise ValueError("lift and equivalence tolerances must be positive")


def _trace_normalize(matrix: np.ndarray) -> np.ndarray:
    candidate = positive_hermitian_projection(matrix)
    return candidate * (len(candidate) / float(np.trace(candidate).real))


def symmetry_broken_relative_cholesky_factors(
    dimension: int,
    *,
    scale: float,
    seed: int,
) -> np.ndarray:
    """Return deterministic opposite SPD perturbations for two factors.

    At ``scale=0`` both factors are exactly the identity.  For positive scale,
    their relative log matrices are ``+scale*A`` and ``-scale*A``, where A is
    trace-free and has spectral radius one.
    """

    if dimension <= 0 or scale < 0:
        raise ValueError("dimension must be positive and scale non-negative")
    if scale == 0:
        return np.stack([np.eye(dimension, dtype=np.complex128)] * 2)
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    generator = 0.5 * (raw + raw.conjugate().T)
    generator -= np.trace(generator) / dimension * np.eye(dimension)
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    radius = float(np.max(np.abs(eigenvalues)))
    if not np.isfinite(radius) or radius <= 0:
        raise FloatingPointError("failed to construct a symmetry-breaking generator")
    eigenvalues /= radius
    factors = []
    for sign in (1.0, -1.0):
        relative_h = (
            eigenvectors * np.exp(sign * scale * eigenvalues)[None, :]
        ) @ eigenvectors.conjugate().T
        factors.append(np.linalg.cholesky(0.5 * (relative_h + relative_h.conj().T)))
    return np.asarray(factors, dtype=np.complex128)


def _factorized_metrics_from_prepared_numpy(
    values: np.ndarray,
    derivatives: np.ndarray,
    h_matrices: np.ndarray,
    *,
    source_normalization: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex128)
    derivatives = np.asarray(derivatives, dtype=np.complex128)
    factors = np.asarray(h_matrices, dtype=np.complex128)
    if values.ndim != 2 or derivatives.shape[:2] != values.shape:
        raise ValueError("prepared section values and derivatives are not aligned")
    if factors.ndim != 3 or factors.shape[1:] != (values.shape[1], values.shape[1]):
        raise ValueError("factor H matrices do not match the source section basis")
    metrics = np.zeros(
        (len(values), derivatives.shape[2], derivatives.shape[2]),
        dtype=np.complex128,
    )
    for h_matrix in factors:
        h_values = np.einsum("ab,nb->na", h_matrix, values, optimize=True)
        denominator = np.real(
            np.einsum("na,na->n", np.conjugate(values), h_values, optimize=True)
        )
        if np.any(denominator <= 0):
            raise FloatingPointError("factorized H denominator is not positive")
        h_derivatives = np.einsum("ab,nbj->naj", h_matrix, derivatives, optimize=True)
        first = np.einsum(
            "nmi,nmj->nij",
            np.conjugate(derivatives),
            h_derivatives,
            optimize=True,
        )
        gradient = np.einsum(
            "nm,nmj->nj", np.conjugate(values), h_derivatives, optimize=True
        )
        contribution = first / denominator[:, None, None]
        contribution -= (
            np.conjugate(gradient)[:, :, None]
            * gradient[:, None, :]
            / denominator[:, None, None] ** 2
        )
        metrics += source_normalization * contribution
    metrics /= len(factors)
    return 0.5 * (metrics + np.conjugate(np.swapaxes(metrics, 1, 2)))


def factorized_h_metrics(
    adapter: GCICYAdapter,
    points: Sequence[Any],
    *,
    source_exponents: np.ndarray,
    h_matrices: np.ndarray,
    source_normalization: float,
) -> np.ndarray:
    """Evaluate a product-factor metric while computing source sections once."""

    evaluated = [
        adapter.section_values_and_jacobian(point, source_exponents) for point in points
    ]
    values = np.asarray([item[0] for item in evaluated], dtype=np.complex128)
    derivatives = np.asarray([item[1] for item in evaluated], dtype=np.complex128)
    return _factorized_metrics_from_prepared_numpy(
        values,
        derivatives,
        h_matrices,
        source_normalization=source_normalization,
    )


def lift_two_factor_h_matrix(
    adapter: GCICYAdapter,
    points: Sequence[Any],
    *,
    source_degree: tuple[int, ...],
    source_exponents: np.ndarray,
    h_matrices: np.ndarray,
    target_basis: Any,
    rcond: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Lift two distinct source H matrices to one doubled-degree H matrix."""

    factors = np.asarray(h_matrices, dtype=np.complex128)
    exponents = np.asarray(source_exponents, dtype=np.int64)
    if factors.shape != (2, len(exponents), len(exponents)):
        raise ValueError("exactly two source H matrices are required")
    expected_degree = tuple(2 * int(value) for value in source_degree)
    target_degree = tuple(int(value) for value in target_basis.degree)
    if target_degree != expected_degree:
        raise ValueError(
            f"target basis degree {target_degree} is not doubled source degree "
            f"{expected_degree}"
        )
    source_values = np.asarray(
        [adapter.section_values_and_jacobian(point, exponents)[0] for point in points],
        dtype=np.complex128,
    )
    product_values = np.einsum(
        "ni,nj->nij", source_values, source_values, optimize=True
    ).reshape(len(points), -1)
    target_values = np.asarray(
        [
            adapter.section_values_and_jacobian(point, target_basis.selected_exponents)[
                0
            ]
            for point in points
        ],
        dtype=np.complex128,
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        target_values,
        product_values,
        rcond=rcond,
    )
    if rank != target_values.shape[1]:
        raise FloatingPointError(
            f"target lift evaluation matrix has rank {rank}, expected "
            f"{target_values.shape[1]}"
        )
    reconstructed = target_values @ coefficients
    relation_error = float(
        np.linalg.norm(reconstructed - product_values)
        / max(np.finfo(float).tiny, np.linalg.norm(product_values))
    )
    coefficient_tensor = coefficients.reshape(
        coefficients.shape[0], len(exponents), len(exponents)
    )
    first_contraction = np.einsum(
        "ik,bkl->bil", factors[0], coefficient_tensor, optimize=True
    )
    product_contraction = np.einsum(
        "jl,bil->bij", factors[1], first_contraction, optimize=True
    )
    target_h = np.einsum(
        "aij,bij->ab",
        np.conjugate(coefficient_tensor),
        product_contraction,
        optimize=True,
    )
    return _trace_normalize(target_h), relation_error


def _dataset_stats(
    dataset: dict[str, Any],
    metrics: np.ndarray,
    *,
    importance_weighted: bool,
) -> dict[str, float]:
    eigenvalues = np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128))
    minimum = float(np.min(eigenvalues))
    if minimum <= 0:
        raise FloatingPointError("candidate metric is not positive definite")
    log_eta = np.sum(np.log(eigenvalues), axis=1) - dataset["log_omega"]
    weights = (
        dataset["importance_weights"]
        if importance_weighted
        else np.ones(len(log_eta), dtype=float)
    )
    stats = standard_errors(log_eta, weights)
    normalized_ratio, _ = normalized_volume_ratios(log_eta, weights)
    cluster_ids = np.asarray(dataset["sampling_cluster_ids"], dtype=np.int64)
    failure = normalized_ratio > 3.0
    stats["sampling_cluster_count"] = int(len(np.unique(cluster_ids)))
    stats["normalized_ratio_above_3_cluster_count"] = int(
        len(np.unique(cluster_ids[failure]))
    )
    stats["min_metric_eigenvalue"] = minimum
    return stats


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def train_factorized_h_metric(
    adapter: GCICYAdapter,
    request: FactorizedHTrainingRequest,
    *,
    artifact_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Train two source-level H factors and export their exact product lift."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for factorized-H training") from exc

    started = time.perf_counter()
    if request.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(request.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    real_dtype = torch.float32 if request.precision == "complex64" else torch.float64
    complex_dtype = (
        torch.complex64 if request.precision == "complex64" else torch.complex128
    )
    prepared_dtype = np.complex64 if request.precision == "complex64" else np.complex128
    torch.manual_seed(request.torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(request.torch_seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = adapter.make_model(request.model_seed, exact=request.exact_model)
    source_path = request.source_artifact.expanduser().resolve()
    source = adapter.load_h_artifact(source_path, model)
    source_degree = tuple(int(value) for value in source.degree)
    target_degree = tuple(2 * value for value in source_degree)
    if (
        request.target_degree is not None
        and tuple(request.target_degree) != target_degree
    ):
        raise ValueError(
            f"requested target degree {request.target_degree} does not equal "
            f"the doubled source degree {target_degree}"
        )
    source_power = adapter.configuration.kahler_power(source_degree)
    target_power = adapter.configuration.kahler_power(target_degree)
    if target_power != 2 * source_power:
        raise ValueError("adapter Kahler powers are not additive under degree doubling")
    expected_source_normalization = 1.0 / source_power
    if not np.isclose(source.normalization, expected_source_normalization, rtol=1e-12):
        raise ValueError(
            "source artifact normalization is inconsistent with its polarization power"
        )
    target_normalization = 1.0 / target_power
    source_h = _trace_normalize(source.h_matrix)
    source_exponents = np.asarray(source.section_exponents, dtype=np.int64)
    section_count = len(source_exponents)
    if (
        request.expected_source_section_count is not None
        and section_count != request.expected_source_section_count
    ):
        raise ValueError(
            f"source artifact has {section_count} sections, expected "
            f"{request.expected_source_section_count}"
        )
    initial_cholesky = np.linalg.cholesky(source_h)
    relative_factors = symmetry_broken_relative_cholesky_factors(
        section_count,
        scale=request.symmetry_breaking_scale,
        seed=request.symmetry_breaking_seed,
    )

    sampling_started = time.perf_counter()
    sampling_shards: dict[str, list[dict[str, int]]] = {}

    def sample_stage(label: str, count: int, seed: int) -> list[Any]:
        independent = (
            label == "train"
            and request.training_sampling_mode == "one_random_root_per_fibre"
        )
        generated_count = (
            count * request.sampling_cluster_size if independent else count
        )
        if request.sampling_workers > 1:
            points, shards = sample_points_parallel(
                adapter,
                model_seed=request.model_seed,
                exact_model=request.exact_model,
                count=generated_count,
                seed=seed,
                workers=request.sampling_workers,
                cluster_size=request.sampling_cluster_size,
                backend=request.sampling_backend,
            )
            sampling_shards[label] = shards
        else:
            points = adapter.sample_points(model, generated_count, seed=seed)
        if independent:
            cluster_ids = np.asarray(
                adapter.sampling_cluster_ids(points), dtype=np.int64
            )
            unique_ids, first_indices = np.unique(cluster_ids, return_index=True)
            if len(unique_ids) != count:
                raise RuntimeError(
                    f"independent-fibre sampler returned {len(unique_ids)} clusters, "
                    f"expected {count}"
                )
            points = [points[index] for index in np.sort(first_indices)]
        if len(points) != count:
            raise RuntimeError(
                f"{label} sampler returned {len(points)} points, expected {count}"
            )
        return points

    def prepare(label: str, count: int, seed: int) -> dict[str, Any]:
        stage_started = time.perf_counter()
        print(f"preparing factorized {label}: points={count}, seed={seed}", flush=True)
        points = sample_stage(label, count, seed)
        evaluated = [
            adapter.section_values_and_jacobian(point, source_exponents)
            for point in points
        ]
        cluster_ids = (
            np.arange(count, dtype=np.int64)
            if label == "train"
            and request.training_sampling_mode == "one_random_root_per_fibre"
            else np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
        )
        dataset = {
            "points": points,
            "values": np.asarray([item[0] for item in evaluated], dtype=prepared_dtype),
            "derivatives": np.asarray(
                [item[1] for item in evaluated], dtype=prepared_dtype
            ),
            "log_omega": np.asarray(
                [adapter.holomorphic_volume_log_density(point) for point in points],
                dtype=np.float64,
            ),
            "importance_weights": adapter.importance_weights(points),
            "sampling_cluster_ids": cluster_ids,
        }
        print(
            f"prepared factorized {label}: seconds="
            f"{time.perf_counter() - stage_started:.1f}",
            flush=True,
        )
        return dataset

    train = prepare("train", request.train_points, request.train_seed)
    checkpoint = prepare(
        "checkpoint", request.checkpoint_points, request.checkpoint_seed
    )
    export = prepare("export", request.export_points, request.export_seed)
    sampling_seconds = time.perf_counter() - sampling_started

    lower_rows, lower_columns = np.tril_indices(section_count, k=-1)
    diagonal_log = torch.nn.Parameter(
        torch.tensor(
            np.log(np.real(np.diagonal(relative_factors, axis1=1, axis2=2))),
            dtype=real_dtype,
            device=device,
        )
    )
    lower_real = torch.nn.Parameter(
        torch.tensor(
            relative_factors[:, lower_rows, lower_columns].real,
            dtype=real_dtype,
            device=device,
        )
    )
    lower_imag = torch.nn.Parameter(
        torch.tensor(
            relative_factors[:, lower_rows, lower_columns].imag,
            dtype=real_dtype,
            device=device,
        )
    )
    trainable_parameters = [diagonal_log, lower_real, lower_imag]
    initial_cholesky_t = torch.tensor(
        initial_cholesky, dtype=complex_dtype, device=device
    )
    lower_rows_t = torch.tensor(lower_rows, dtype=torch.int64, device=device)
    lower_columns_t = torch.tensor(lower_columns, dtype=torch.int64, device=device)

    def current_relative_factors():
        factors = torch.diag_embed(torch.exp(diagonal_log)).to(complex_dtype)
        factors[:, lower_rows_t, lower_columns_t] = lower_real.to(
            complex_dtype
        ) + 1j * lower_imag.to(complex_dtype)
        return factors

    def current_h_matrices():
        transformed = torch.einsum(
            "ab,fbc->fac", initial_cholesky_t, current_relative_factors()
        )
        matrices = transformed @ torch.conj(torch.transpose(transformed, 1, 2))
        traces = torch.real(torch.diagonal(matrices, dim1=1, dim2=2).sum(dim=1))
        return matrices * (section_count / traces)[:, None, None]

    def current_h_numpy() -> np.ndarray:
        with torch.no_grad():
            matrices = current_h_matrices().detach().cpu().numpy()
        return np.asarray([_trace_normalize(item) for item in matrices])

    initial_relative_h_t = current_relative_factors().detach() @ torch.conj(
        torch.transpose(current_relative_factors().detach(), 1, 2)
    )

    def dataset_tensors(dataset: dict[str, Any]):
        weights = (
            dataset["importance_weights"]
            if request.importance_weighted
            else np.ones(len(dataset["values"]), dtype=np.float64)
        )
        return (
            torch.tensor(dataset["values"], dtype=complex_dtype, device=device),
            torch.tensor(dataset["derivatives"], dtype=complex_dtype, device=device),
            torch.tensor(dataset["log_omega"], dtype=real_dtype, device=device),
            torch.tensor(weights, dtype=real_dtype, device=device),
        )

    train_tensors = dataset_tensors(train)

    def raw_and_barrier(values, derivatives, log_omega, h_matrices):
        metrics = torch.zeros(
            (len(values), derivatives.shape[2], derivatives.shape[2]),
            dtype=complex_dtype,
            device=device,
        )
        for h_matrix in h_matrices:
            h_values = torch.einsum("ab,nb->na", h_matrix, values)
            denominator = torch.real(
                torch.einsum("na,na->n", torch.conj(values), h_values)
            )
            h_derivatives = torch.einsum("ab,nbj->naj", h_matrix, derivatives)
            first = torch.einsum("nmi,nmj->nij", torch.conj(derivatives), h_derivatives)
            gradient = torch.einsum("nm,nmj->nj", torch.conj(values), h_derivatives)
            contribution = first / denominator[:, None, None]
            contribution -= (
                torch.conj(gradient)[:, :, None]
                * gradient[:, None, :]
                / denominator[:, None, None] ** 2
            )
            metrics = metrics + source.normalization * contribution
        metrics = metrics / request.factor_count
        metrics = 0.5 * (metrics + torch.conj(torch.transpose(metrics, 1, 2)))
        eigenvalues = torch.linalg.eigvalsh(metrics)
        raw = torch.log(torch.clamp(eigenvalues, min=1e-12)).sum(dim=1) - log_omega
        scale = torch.clamp(
            torch.mean(torch.abs(eigenvalues), dim=1, keepdim=True), min=1e-14
        )
        relative_eigenvalues = eigenvalues / scale
        barrier = (
            torch.nn.functional.softplus((1e-8 - relative_eigenvalues) * 80.0).mean()
            / 80.0
        )
        return raw, barrier

    with torch.no_grad():
        initial_raw, _ = raw_and_barrier(
            train_tensors[0],
            train_tensors[1],
            train_tensors[2],
            current_h_matrices(),
        )
        positive_weights = torch.clamp(
            train_tensors[3], min=torch.finfo(real_dtype).tiny
        )
        fixed_log_kappa = (
            torch.logsumexp(torch.log(positive_weights) + initial_raw, dim=0)
            - torch.log(torch.sum(positive_weights))
        ).detach()
    print(
        f"fixed factorized initial-pool log(kappa)={float(fixed_log_kappa.cpu()):.12e}",
        flush=True,
    )

    training_batch_count = (
        request.train_points + request.points_per_optimizer_step - 1
    ) // request.points_per_optimizer_step
    upper_tail_selected_fractions = None
    upper_tail_total_weight = None

    def refresh_upper_tail_selection() -> dict[str, float] | None:
        nonlocal upper_tail_selected_fractions, upper_tail_total_weight
        if not request.upper_log_ratio_cvar_loss_weight:
            upper_tail_selected_fractions = None
            upper_tail_total_weight = None
            return None
        values, derivatives, log_omega, weights = train_tensors
        loss_rows = []
        probe_batch_size = max(request.points_per_optimizer_step, 1024)
        with torch.no_grad():
            for start in range(0, request.train_points, probe_batch_size):
                stop = min(start + probe_batch_size, request.train_points)
                raw, _ = raw_and_barrier(
                    values[start:stop],
                    derivatives[start:stop],
                    log_omega[start:stop],
                    current_h_matrices(),
                )
                loss_rows.append(
                    smooth_upper_log_ratio_excess_torch(
                        raw - fixed_log_kappa,
                        ratio_threshold=request.upper_log_ratio_threshold,
                        smooth_temperature=(
                            request.upper_log_ratio_smooth_temperature
                        ),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
        losses = np.concatenate(loss_rows).astype(np.float64, copy=False)
        weight_values = weights.detach().cpu().numpy().astype(np.float64, copy=False)
        audit = weighted_cvar_numpy(
            losses,
            weight_values,
            tail_fraction=request.upper_log_ratio_cvar_tail_fraction,
        )
        boundary_selection_fraction = (
            audit.selected_boundary_mass / audit.boundary_mass
        )
        selected = (losses > audit.var_threshold).astype(np.float64)
        selected += boundary_selection_fraction * (
            losses == audit.var_threshold
        ).astype(np.float64)
        upper_tail_selected_fractions = torch.as_tensor(
            selected,
            dtype=real_dtype,
            device=device,
        )
        upper_tail_total_weight = float(np.sum(weight_values))
        return {
            "value": audit.value,
            "var_threshold": audit.var_threshold,
            "selected_tail_mass": audit.selected_tail_mass,
            "strict_tail_mass": audit.strict_tail_mass,
            "boundary_mass": audit.boundary_mass,
            "selected_boundary_mass": audit.selected_boundary_mass,
            "boundary_selection_fraction": boundary_selection_fraction,
        }

    def loss_for_indices(indices):
        values, derivatives, log_omega, weights = train_tensors
        raw, barrier = raw_and_barrier(
            values[indices],
            derivatives[indices],
            log_omega[indices],
            current_h_matrices(),
        )
        selected_weights = weights[indices]
        normalized_weights = selected_weights / torch.sum(selected_weights)
        centered = raw - fixed_log_kappa
        ratio = torch.exp(centered)
        absolute_error = torch.abs(1.0 - ratio)
        sigma = torch.sum(normalized_weights * absolute_error)
        volume_ratio_l2 = torch.sqrt(
            torch.sum(normalized_weights * (1.0 - ratio) ** 2)
            + torch.finfo(real_dtype).eps
        )
        positive_log_ratio = torch.relu(centered)
        positive_log_cvar = (
            weighted_cvar_torch(
                positive_log_ratio,
                selected_weights,
                tail_fraction=request.positive_log_ratio_cvar_tail_fraction,
            )
            if request.positive_log_ratio_cvar_loss_weight
            else torch.zeros((), dtype=real_dtype, device=device)
        )
        if request.upper_log_ratio_cvar_loss_weight:
            if (
                upper_tail_selected_fractions is None
                or upper_tail_total_weight is None
            ):
                raise RuntimeError("upper-tail selection was not prepared")
            upper_log_excess = smooth_upper_log_ratio_excess_torch(
                centered,
                ratio_threshold=request.upper_log_ratio_threshold,
                smooth_temperature=request.upper_log_ratio_smooth_temperature,
            )
            upper_log_cvar = (
                training_batch_count
                * weighted_cvar_selected_linearization_torch(
                    upper_log_excess,
                    selected_weights,
                    upper_tail_selected_fractions[indices],
                    tail_fraction=request.upper_log_ratio_cvar_tail_fraction,
                    total_weight=upper_tail_total_weight,
                )
            )
        else:
            upper_log_cvar = torch.zeros((), dtype=real_dtype, device=device)
        centered_log_variance = torch.sum(
            normalized_weights * (raw - torch.sum(normalized_weights * raw)) ** 2
        )
        relative = current_relative_factors()
        relative_h = relative @ torch.conj(torch.transpose(relative, 1, 2))
        drift = torch.mean(torch.abs(relative_h - initial_relative_h_t) ** 2)
        relative_eigenvalues = torch.linalg.eigvalsh(relative_h)
        relative_logs = torch.log(torch.clamp(relative_eigenvalues, min=1e-12))
        relative_logs -= torch.mean(relative_logs, dim=1, keepdim=True)
        spectrum_penalty = torch.mean(relative_logs**2)
        total = (
            request.sigma_loss_weight * sigma
            + request.volume_ratio_l2_loss_weight * volume_ratio_l2
            + request.positive_log_ratio_cvar_loss_weight * positive_log_cvar
            + request.upper_log_ratio_cvar_loss_weight * upper_log_cvar
            + request.centered_log_variance_loss_weight * centered_log_variance
            + request.metric_barrier_weight * barrier
            + request.drift_weight * drift
            + request.relative_log_spectrum_loss_weight * spectrum_penalty
        )
        return total

    source_factors = np.stack([source_h, source_h])

    def evaluate(dataset: dict[str, Any], matrices: np.ndarray) -> dict[str, float]:
        metrics = _factorized_metrics_from_prepared_numpy(
            dataset["values"],
            dataset["derivatives"],
            matrices,
            source_normalization=source.normalization,
        )
        return _dataset_stats(
            dataset,
            metrics,
            importance_weighted=request.importance_weighted,
        )

    source_checkpoint = evaluate(checkpoint, source_factors)
    initial_h_matrices = current_h_numpy()
    initial_checkpoint = evaluate(checkpoint, initial_h_matrices)

    def selection_score(stats: dict[str, float]) -> float:
        return float(
            stats["sigma"]
            + request.selection_volume_ratio_l2_weight * stats["sqrt_squared_energy"]
            + request.selection_positive_log_ratio_cvar_weight
            * stats["positive_log_ratio_cvar_1pct"]
        )

    best_h_matrices = initial_h_matrices.copy()
    best_epoch = 0
    best_score = selection_score(initial_checkpoint)
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "checkpoint": initial_checkpoint,
            "selection_score": best_score,
            "factor_relative_frobenius_distance": float(
                np.linalg.norm(initial_h_matrices[0] - initial_h_matrices[1])
                / max(np.finfo(float).tiny, np.linalg.norm(source_h))
            ),
            "optimizer_step_count": 0,
            "global_upper_log_ratio_cvar_tail_audit": None,
        }
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=request.learning_rate)
    shuffle_rng = np.random.default_rng(request.shuffle_seed)
    optimizer_step_count = 0
    clipped_step_count = 0
    optimization_started = time.perf_counter()
    for epoch in range(1, request.epochs + 1):
        upper_tail_audit = refresh_upper_tail_selection()
        permutation = shuffle_rng.permutation(request.train_points)
        for start in range(0, request.train_points, request.points_per_optimizer_step):
            indices = torch.tensor(
                permutation[start : start + request.points_per_optimizer_step],
                dtype=torch.int64,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_indices(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite factorized loss at epoch {epoch}")
            loss.backward()
            if request.gradient_clip_norm is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    request.gradient_clip_norm,
                )
                if float(gradient_norm.detach().cpu()) > request.gradient_clip_norm:
                    clipped_step_count += 1
            optimizer.step()
            optimizer_step_count += 1
        current = current_h_numpy()
        checkpoint_stats = evaluate(checkpoint, current)
        score = selection_score(checkpoint_stats)
        distance = float(
            np.linalg.norm(current[0] - current[1])
            / max(np.finfo(float).tiny, np.linalg.norm(source_h))
        )
        history.append(
            {
                "epoch": epoch,
                "checkpoint": checkpoint_stats,
                "selection_score": score,
                "factor_relative_frobenius_distance": distance,
                "optimizer_step_count": optimizer_step_count,
                "global_upper_log_ratio_cvar_tail_audit": upper_tail_audit,
            }
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_h_matrices = current.copy()
        print(
            f"factorized epoch={epoch} checkpoint_sigma="
            f"{checkpoint_stats['sigma']:.6e} checkpoint_l2="
            f"{checkpoint_stats['sqrt_squared_energy']:.6e} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimization_seconds = time.perf_counter() - optimization_started

    lift_started = time.perf_counter()
    print(
        f"preparing doubled-degree lift: points={request.lift_basis_points}, "
        f"seed={request.lift_basis_seed}",
        flush=True,
    )
    lift_points = sample_stage(
        "lift_basis", request.lift_basis_points, request.lift_basis_seed
    )
    target_basis = adapter.restricted_section_basis(lift_points, target_degree)
    if (
        request.expected_target_section_count is not None
        and target_basis.numerical_rank != request.expected_target_section_count
    ):
        raise FloatingPointError(
            f"target basis has rank {target_basis.numerical_rank}, expected "
            f"{request.expected_target_section_count}"
        )
    target_h, relation_error = lift_two_factor_h_matrix(
        adapter,
        lift_points,
        source_degree=source_degree,
        source_exponents=source_exponents,
        h_matrices=best_h_matrices,
        target_basis=target_basis,
        rcond=request.lift_rcond,
    )
    lift_seconds = time.perf_counter() - lift_started
    target_exponents = np.asarray(target_basis.selected_exponents, dtype=np.int64)
    in_memory_artifact = HMetricArtifact(
        path=artifact_path.expanduser().resolve(),
        degree=target_degree,
        section_exponents=target_exponents,
        h_matrix=target_h,
        normalization=target_normalization,
    )
    equivalence_slice = slice(0, request.equivalence_points)
    direct_equivalence_metrics = _factorized_metrics_from_prepared_numpy(
        export["values"][equivalence_slice],
        export["derivatives"][equivalence_slice],
        best_h_matrices,
        source_normalization=source.normalization,
    )
    lifted_equivalence_metrics = adapter.h_metrics(
        export["points"][equivalence_slice], in_memory_artifact
    )
    relative_metric_errors = np.linalg.norm(
        lifted_equivalence_metrics - direct_equivalence_metrics, axis=(1, 2)
    ) / np.maximum(
        np.finfo(float).tiny,
        np.linalg.norm(direct_equivalence_metrics, axis=(1, 2)),
    )
    maximum_metric_equivalence_error = float(np.max(relative_metric_errors))
    rms_metric_equivalence_error = float(np.sqrt(np.mean(relative_metric_errors**2)))

    best_checkpoint = evaluate(checkpoint, best_h_matrices)
    source_export_metrics = _factorized_metrics_from_prepared_numpy(
        export["values"],
        export["derivatives"],
        source_factors,
        source_normalization=source.normalization,
    )
    source_export = _dataset_stats(
        export,
        source_export_metrics,
        importance_weighted=request.importance_weighted,
    )
    best_export_metrics = _factorized_metrics_from_prepared_numpy(
        export["values"],
        export["derivatives"],
        best_h_matrices,
        source_normalization=source.normalization,
    )
    best_export = _dataset_stats(
        export,
        best_export_metrics,
        importance_weighted=request.importance_weighted,
    )
    best_export_eigenvalues = np.linalg.eigvalsh(best_export_metrics)
    corrected_log_ma = (
        np.sum(np.log(best_export_eigenvalues), axis=1) - export["log_omega"]
    )
    baseline_metrics = adapter.baseline_metrics(export["points"])
    baseline_log_ma = adapter.residual_values(export["points"], baseline_metrics)

    source_metric_errors = np.linalg.norm(
        best_export_metrics - source_export_metrics, axis=(1, 2)
    ) / np.maximum(
        np.finfo(float).tiny,
        np.linalg.norm(source_export_metrics, axis=(1, 2)),
    )
    maximum_source_metric_error = float(np.max(source_metric_errors))
    zero_control_requested = bool(
        request.epochs == 0 and request.symmetry_breaking_scale == 0
    )
    zero_control_gate = bool(
        not zero_control_requested
        or maximum_source_metric_error <= request.maximum_metric_equivalence_error
    )

    relation_gate = relation_error <= request.maximum_lift_relation_error
    equivalence_gate = (
        maximum_metric_equivalence_error <= request.maximum_metric_equivalence_error
    )
    positivity_gate = bool(best_export["min_metric_eigenvalue"] > 0)
    source_sigma_improved = bool(
        best_checkpoint["sigma"] < source_checkpoint["sigma"]
        and best_export["sigma"] < source_export["sigma"]
    )
    implementation_gate = bool(
        relation_gate and equivalence_gate and positivity_gate and zero_control_gate
    )
    success = bool(
        implementation_gate
        and (source_sigma_improved or not request.require_source_sigma_improvement)
    )

    artifact_path = artifact_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **adapter.artifact_model_payload(model, exact=request.exact_model),
        "pipeline_schema_version": np.asarray(1, dtype=np.int64),
        "pipeline_adapter": np.asarray(adapter.key),
        "factorized_h_schema_version": np.asarray(
            FACTORIZED_H_SCHEMA_VERSION, dtype=np.int64
        ),
        "importance_weighted_training": np.asarray(request.importance_weighted),
        "global_section_degree": np.asarray(target_degree, dtype=np.int64),
        "global_section_exponents": target_exponents,
        "global_h_matrix": target_h,
        "global_h_positive_relative_floor": np.asarray(
            H_POSITIVE_RELATIVE_FLOOR, dtype=np.float64
        ),
        "global_section_normalization": np.asarray(
            target_normalization, dtype=np.float64
        ),
        "basis_selected_indices": np.asarray(
            target_basis.selected_indices, dtype=np.int64
        ),
        "basis_relation_error": np.asarray(relation_error, dtype=np.float64),
        "h_parameterization": np.asarray("two_factor_product_relative_cholesky"),
        "h_optimization_coordinate_system": np.asarray(
            "source_h_relative_cholesky_per_factor"
        ),
        "parameterization_real_dimension": np.asarray(
            request.factor_count * section_count**2, dtype=np.int64
        ),
        "scale_invariant_h_dimension_upper_bound": np.asarray(
            request.factor_count * (section_count**2 - 1), dtype=np.int64
        ),
        "factorized_source_artifact_sha256": np.asarray(file_sha256(source_path)),
        "factorized_source_degree": np.asarray(source_degree, dtype=np.int64),
        "factorized_source_exponents": source_exponents,
        "factorized_source_normalization": np.asarray(
            source.normalization, dtype=np.float64
        ),
        "factorized_h_matrices": best_h_matrices,
        "factorized_factor_count": np.asarray(request.factor_count, dtype=np.int64),
        "factorized_symmetry_breaking_scale": np.asarray(
            request.symmetry_breaking_scale, dtype=np.float64
        ),
        "factorized_lift_relation_error": np.asarray(relation_error, dtype=np.float64),
        "factorized_maximum_metric_equivalence_error": np.asarray(
            maximum_metric_equivalence_error, dtype=np.float64
        ),
        "factorized_maximum_source_metric_error": np.asarray(
            maximum_source_metric_error, dtype=np.float64
        ),
        "fixed_group_log_kappa": np.asarray(
            float(fixed_log_kappa.cpu()), dtype=np.float64
        ),
        "upper_log_ratio_cvar_loss_weight": np.asarray(
            request.upper_log_ratio_cvar_loss_weight, dtype=np.float64
        ),
        "upper_log_ratio_cvar_tail_fraction": np.asarray(
            request.upper_log_ratio_cvar_tail_fraction, dtype=np.float64
        ),
        "upper_log_ratio_threshold": np.asarray(
            request.upper_log_ratio_threshold, dtype=np.float64
        ),
        "upper_log_ratio_smooth_temperature": np.asarray(
            request.upper_log_ratio_smooth_temperature, dtype=np.float64
        ),
        "group_normalization_mode": np.asarray("fixed_initial_training_pool"),
        "group_optimizer_step_mode": np.asarray("shuffled_point_minibatch"),
        "training_sampling_mode": np.asarray(request.training_sampling_mode),
        "points_per_optimizer_step": np.asarray(
            request.points_per_optimizer_step, dtype=np.int64
        ),
        "gradient_clip_norm": np.asarray(
            (
                request.gradient_clip_norm
                if request.gradient_clip_norm is not None
                else np.nan
            ),
            dtype=np.float64,
        ),
        "optimizer_step_count": np.asarray(optimizer_step_count, dtype=np.int64),
        "baseline_log_ma": baseline_log_ma,
        "corrected_log_ma": corrected_log_ma,
        "importance_weights": export["importance_weights"],
    }
    np.savez_compressed(artifact_path, **payload)

    device_memory = None
    if device.type == "cuda":
        device_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    summary: dict[str, Any] = {
        "schema_version": FACTORIZED_H_SCHEMA_VERSION,
        "adapter": adapter.key,
        "configuration": adapter.model_metadata(model),
        "request": _json_safe(asdict(request)),
        "source": {
            "artifact": str(source_path),
            "sha256": file_sha256(source_path),
            "degree": list(source_degree),
            "section_count": section_count,
            "normalization": source.normalization,
        },
        "target": {
            "degree": list(target_degree),
            "section_count": int(target_basis.numerical_rank),
            "normalization": target_normalization,
        },
        "parameterization": {
            "name": "two_factor_product_relative_cholesky",
            "factor_count": request.factor_count,
            "real_dimension": request.factor_count * section_count**2,
            "scale_invariant_dimension_upper_bound": request.factor_count
            * (section_count**2 - 1),
            "full_target_scale_invariant_dimension": int(
                target_basis.numerical_rank**2 - 1
            ),
        },
        "device": str(device),
        "device_memory": device_memory,
        "sampling": {
            "seconds": sampling_seconds,
            "derivation": "SeedSequence(base_seed).spawn(active_workers)",
            "shards": sampling_shards,
        },
        "training": {
            "fixed_initial_log_kappa": float(fixed_log_kappa.cpu()),
            "optimizer": "Adam",
            "optimizer_step_count": optimizer_step_count,
            "clipped_optimizer_step_count": clipped_step_count,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "history": history,
        },
        "checkpoint": {
            "source_reference": source_checkpoint,
            "initial_factorized": initial_checkpoint,
            "selected_factorized": best_checkpoint,
        },
        "export": {
            "source_reference": source_export,
            "selected_factorized": best_export,
        },
        "lift": {
            "basis_points": request.lift_basis_points,
            "relation_error": relation_error,
            "maximum_allowed_relation_error": request.maximum_lift_relation_error,
            "relation_gate_passed": relation_gate,
            "equivalence_points": request.equivalence_points,
            "maximum_relative_metric_error": maximum_metric_equivalence_error,
            "rms_relative_metric_error": rms_metric_equivalence_error,
            "maximum_allowed_metric_error": (request.maximum_metric_equivalence_error),
            "equivalence_gate_passed": equivalence_gate,
            "zero_training_source_metric_error": maximum_source_metric_error,
        },
        "gates": {
            "positivity_passed": positivity_gate,
            "zero_control_requested": zero_control_requested,
            "zero_control_passed": zero_control_gate,
            "implementation_passed": implementation_gate,
            "source_sigma_improved_on_checkpoint_and_export": source_sigma_improved,
            "source_improvement_required": request.require_source_sigma_improvement,
        },
        "runtime_seconds": {
            "sampling": sampling_seconds,
            "optimization": optimization_seconds,
            "lift": lift_seconds,
            "total": time.perf_counter() - started,
        },
        "artifact": str(artifact_path),
        "success": success,
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
