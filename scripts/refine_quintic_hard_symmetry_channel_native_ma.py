#!/usr/bin/env python3
"""Fit one nested hard-symmetry channel with the native normalized MA residual."""

from __future__ import annotations

import argparse
import copy
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
from scripts.audit_quintic_tn_metric_tangent_reachability import (  # noqa: E402
    MaskedComplexParameterVectorizer,
)
from scripts.audit_quintic_tn_paired_native_tail import tail_values  # noqa: E402
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    lanczos_tridiagonal,
    real_inner,
    ridge_dual_from_lanczos,
    select_disjoint_indices,
    vector_norm,
)
from scripts.initialize_quintic_hard_symmetry_channel import (  # noqa: E402
    one_sided_channel_masks,
    seed_one_sided_channel_,
)
from scripts.refine_quintic_hard_symmetry_channel_gn import (  # noqa: E402
    candidate_used_indices,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    ratio_statistics,
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    tensor_split,
    training_log_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--exclude-report", type=Path, nargs="*", default=())
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--candidate-start", type=int, default=65_000)
    parser.add_argument("--candidate-limit", type=int, default=10_000)
    parser.add_argument("--fit-size", type=int, default=4096)
    parser.add_argument("--selection-size", type=int, default=4096)
    parser.add_argument("--bond-index", type=int, required=True)
    parser.add_argument("--candidate-seed", type=int, required=True)
    parser.add_argument("--relative-seed-scale", type=float, default=1.0)
    parser.add_argument(
        "--seed-reference-scale",
        choices=("inherited_rms", "unit_rms"),
        default="unit_rms",
    )
    parser.add_argument("--operator-chunk-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lanczos-steps", type=int, default=6)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(30.0, 10.0, 3.0, 1.0),
    )
    parser.add_argument(
        "--step-scales",
        type=float,
        nargs="+",
        default=(0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--expected-parameter-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202607251)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.candidate_limit,
        args.fit_size,
        args.selection_size,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.lanczos_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sample, chunk, batch, and Lanczos sizes must be positive")
    if args.candidate_start < 0 or args.bond_index < 0:
        raise ValueError("candidate start and bond index cannot be negative")
    if args.expected_parameter_count < 0:
        raise ValueError("expected parameter count cannot be negative")
    if args.relative_seed_scale <= 0:
        raise ValueError("relative seed scale must be positive")
    if any(
        not np.isfinite(value) or value <= 0 for value in args.ridge_factors
    ):
        raise ValueError("ridge factors must be finite and positive")
    if any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in args.step_scales
    ):
        raise ValueError("step scales must lie in (0, 1]")


def weighted_log_mean_exp(
    raw: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    maximum = torch.max(raw)
    return maximum + torch.log(torch.sum(weights * torch.exp(raw - maximum)))


class MatrixFreeNativeMARatioJacobian:
    """JVP/VJP for sqrt(w) * (r - 1), including d(volume normalization)."""

    def __init__(
        self,
        model: torch.nn.Module,
        vectorizer: MaskedComplexParameterVectorizer,
        theta: torch.Tensor,
        dataset: dict[str, Any],
        *,
        chunk_size: int,
        raw_function_factory: (
            Callable[[slice, MaskedComplexParameterVectorizer], Callable[
                [torch.Tensor], torch.Tensor
            ]]
            | None
        ) = None,
    ) -> None:
        if dataset["count"] <= 0 or chunk_size <= 0:
            raise ValueError("native MA operator requires nonempty positive chunks")
        weights = dataset["weights"]
        if weights.ndim != 1 or weights.numel() != dataset["count"]:
            raise ValueError("native MA weights have the wrong shape")
        if not bool(torch.all(weights > 0)):
            raise ValueError("native MA weights must be positive")
        if not bool(
            torch.isclose(
                torch.sum(weights),
                torch.ones((), dtype=weights.dtype, device=weights.device),
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        ):
            raise ValueError("native MA weights must be normalized")
        self.model = model
        self.vectorizer = vectorizer
        self.theta = theta
        self.dataset = dataset
        self.chunk_size = int(chunk_size)
        self.raw_function_factory = raw_function_factory
        self._slices = tuple(
            slice(start, min(start + chunk_size, dataset["count"]))
            for start in range(0, dataset["count"], chunk_size)
        )
        self.output_count = int(dataset["count"])

        raw = self.raw(theta)
        log_kappa = weighted_log_mean_exp(raw, weights)
        ratio = torch.exp(raw - log_kappa)
        probability = weights * ratio
        if not bool(
            torch.isclose(
                torch.sum(probability),
                torch.ones(
                    (),
                    dtype=probability.dtype,
                    device=probability.device,
                ),
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        ):
            raise FloatingPointError("native MA normalization failed")
        self.raw_baseline = raw.detach()
        self.log_kappa = log_kappa.detach()
        self.ratio = ratio.detach()
        self.probability = probability.detach()
        self.square_root_weights = torch.sqrt(weights)
        self._residual = self.square_root_weights * (self.ratio - 1.0)

    def _chunk_raw_function(
        self,
        selection: slice,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.raw_function_factory is not None:
            return self.raw_function_factory(selection, self.vectorizer)
        values = self.dataset["values"][selection]
        derivatives = self.dataset["derivatives"][selection]
        log_omega = self.dataset["log_omega"][selection]

        def raw(parameter_vector: torch.Tensor) -> torch.Tensor:
            metric = functional_call(
                self.model,
                self.vectorizer.unpack(parameter_vector),
                (values, derivatives),
                strict=False,
            )
            return training_log_volume(metric, "cholesky") - log_omega

        return raw

    def raw(self, parameter_vector: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self._chunk_raw_function(part)(parameter_vector)
                for part in self._slices
            ]
        )

    def residual(self) -> torch.Tensor:
        return self._residual

    def exact_residual(self, parameter_vector: torch.Tensor) -> torch.Tensor:
        raw = self.raw(parameter_vector)
        log_kappa = weighted_log_mean_exp(raw, self.dataset["weights"])
        ratio = torch.exp(raw - log_kappa)
        return self.square_root_weights * (ratio - 1.0)

    def jvp(self, direction: torch.Tensor) -> torch.Tensor:
        raw_tangent = []
        for part in self._slices:
            function = self._chunk_raw_function(part)
            _, tangent = jvp(function, (self.theta,), (direction,))
            raw_tangent.append(tangent)
        delta_log_volume = torch.cat(raw_tangent)
        delta_log_kappa = torch.sum(self.probability * delta_log_volume)
        delta_ratio = self.ratio * (
            delta_log_volume - delta_log_kappa
        )
        return self.square_root_weights * delta_ratio

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        if cotangent.ndim != 1 or cotangent.numel() != self.output_count:
            raise ValueError("native MA residual cotangent has the wrong shape")
        return self.vjp_many(cotangent.unsqueeze(0))[0]

    def vjp_many(self, cotangents: torch.Tensor) -> torch.Tensor:
        """Apply several explicit VJPs while sharing each chunk's forward graph."""

        if (
            cotangents.ndim != 2
            or cotangents.shape[1] != self.output_count
            or cotangents.shape[0] <= 0
        ):
            raise ValueError("native MA residual cotangent batch has the wrong shape")
        scaled = (
            self.square_root_weights[None, :]
            * self.ratio[None, :]
            * cotangents
        )
        raw_cotangent = scaled - self.probability[None, :] * torch.sum(
            scaled,
            dim=1,
            keepdim=True,
        )
        result = torch.zeros(
            (cotangents.shape[0],) + self.theta.shape,
            dtype=self.theta.dtype,
            device=self.theta.device,
        )
        offset = 0
        for part in self._slices:
            function = self._chunk_raw_function(part)
            count = part.stop - part.start
            _, pullback = vjp(function, self.theta)
            for row in range(cotangents.shape[0]):
                result[row] = result[row] + pullback(
                    raw_cotangent[row, offset : offset + count]
                )[0]
            offset += count
        return result

    def normal(self, cotangent: torch.Tensor) -> torch.Tensor:
        return self.jvp(self.vjp(cotangent))


def evaluate_raw(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_rows = []
    minimum_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], batch_size):
            stop = min(start + batch_size, dataset["count"])
            metric = model(
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(torch.isfinite(eigenvalues))):
                raise FloatingPointError("nonfinite metric eigenvalue")
            raw_rows.append(
                (
                    training_log_volume(metric, "cholesky")
                    - dataset["log_omega"][start:stop]
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            minimum_rows.append(
                torch.min(eigenvalues, dim=1)
                .values.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    return np.concatenate(raw_rows), np.concatenate(minimum_rows)


def model_summary(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    raw, minimum = evaluate_raw(model, dataset, batch_size=batch_size)
    statistics, _ = ratio_statistics(
        raw,
        dataset["weights_numpy"],
        minimum,
    )
    tail = tail_values(statistics)
    return {
        "sigma": float(statistics["sigma_official_formula"]),
        "chi": float(statistics["weighted_rms_abs_residual"]),
        "e2": float(statistics["weighted_rms_abs_residual"]) ** 2,
        "q999": float(tail["abs_residual_q999"]),
        "cvar99": float(tail["abs_residual_cvar99"]),
        "cvar999": float(tail["abs_residual_cvar999"]),
        "maximum": float(tail["abs_residual_max"]),
        "ratio_max": float(tail["ratio_max"]),
        "minimum_metric_eigenvalue": float(
            statistics["min_eigenvalue_weighted_quantiles"]["q0.0000"]
        ),
    }


def capture(baseline: float, candidate: float) -> float:
    return 1.0 - candidate / max(baseline, np.finfo(float).tiny)


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_model = args.output_model.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    status = output_report.with_suffix(output_report.suffix + ".status.json")
    indices_path = output_report.with_suffix(output_report.suffix + ".indices.npz")
    if (
        output_model.exists()
        or output_report.exists()
        or status.exists()
        or indices_path.exists()
    ):
        raise FileExistsError("refusing to overwrite a native-MA channel artifact")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
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

    model_path = args.expanded_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    exclude_reports = tuple(
        path.expanduser().resolve() for path in args.exclude_report
    )
    for path in (model_path, dataset_path, pullbacks_path, *exclude_reports):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    expansion = payload.get("phase_multiplicity_expansion")
    if not isinstance(expansion, dict):
        raise ValueError("input model is not an exact phase-multiplicity expansion")
    source_multiplicity = int(expansion["source_multiplicity"])
    target_multiplicity = int(expansion["target_multiplicity"])
    source_degree = int(payload.get("source_degree", 1))
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    complex_dtype = (
        torch.complex64
        if str(payload["precision"]) == "complex64"
        else torch.complex128
    )
    real_dtype = (
        torch.float32 if complex_dtype == torch.complex64 else torch.float64
    )

    data = np.load(dataset_path, allow_pickle=False)
    candidate_stop = min(
        args.candidate_start + args.candidate_limit,
        len(data["X_val"]),
    )
    candidate_count = candidate_stop - args.candidate_start
    if candidate_count <= 0:
        raise ValueError("candidate window is empty")
    local_fit, local_selection = select_disjoint_indices(
        candidate_count,
        args.fit_size,
        args.selection_size,
        seed=args.seed,
    )
    fit_indices = local_fit + args.candidate_start
    selection_indices = local_selection + args.candidate_start
    stage_indices = np.concatenate((fit_indices, selection_indices))

    excluded_rows = []
    excluded_artifacts = []
    for report_path in exclude_reports:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        values, artifact = candidate_used_indices(
            report,
            report_path=report_path,
        )
        excluded_rows.append(values)
        excluded_artifacts.append(artifact)
    previous = (
        np.unique(np.concatenate(excluded_rows))
        if excluded_rows
        else np.empty(0, dtype=np.int64)
    )
    overlap = np.intersect1d(previous, stage_indices, assume_unique=False)
    if overlap.size:
        raise ValueError(
            f"native-MA fit/selection overlaps preceding data at "
            f"{overlap.size} indices"
        )
    np.savez_compressed(
        indices_path,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
    )

    validation_pullbacks = np.load(pullbacks_path, mmap_mode="r")

    def make_split(indices: np.ndarray) -> dict[str, Any]:
        return tensor_split(
            np.asarray(data["X_val"][indices], dtype=np.float32),
            np.asarray(validation_pullbacks[indices]),
            np.asarray(data["y_val"][indices], dtype=np.float64),
            source_degree=source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )

    fit = make_split(fit_indices)
    selection = make_split(selection_indices)
    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=complex_dtype)
    model.force_indexed_block_contraction = True
    parameter_count = int(model.trainable_real_parameter_count)
    if args.expected_parameter_count and parameter_count != args.expected_parameter_count:
        raise RuntimeError(
            f"parameter-count gate failed: {parameter_count} != "
            f"{args.expected_parameter_count}"
        )

    probe_count = min(16, fit["count"])
    probe = {
        **fit,
        "count": probe_count,
        "values": fit["values"][:probe_count],
        "derivatives": fit["derivatives"][:probe_count],
        "log_omega": fit["log_omega"][:probe_count],
    }
    pre_seed_raw, _ = evaluate_raw(
        model,
        probe,
        batch_size=args.eval_batch_size,
    )
    seed_generator = torch.Generator(device=device)
    seed_generator.manual_seed(args.candidate_seed)
    activation = seed_one_sided_channel_(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        relative_scale=args.relative_seed_scale,
        reference_scale=args.seed_reference_scale,
        generator=seed_generator,
    )
    left_mask, _ = one_sided_channel_masks(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
    )
    parameter_name = f"blocked_two_site_orbit_parameters.{args.bond_index}"
    parameter = dict(model.named_parameters())[parameter_name]
    vectorizer = MaskedComplexParameterVectorizer.from_module(
        model,
        parameter_name,
        left_mask.reshape(parameter.shape),
    )
    theta = vectorizer.pack(model)
    model.requires_grad_(False).eval()
    post_seed_raw, _ = evaluate_raw(
        model,
        probe,
        batch_size=args.eval_batch_size,
    )
    epoch_zero_max_change = float(np.max(np.abs(post_seed_raw - pre_seed_raw)))
    if epoch_zero_max_change > 5.0e-6:
        raise RuntimeError(
            "one-sided native-MA activation did not preserve the baseline metric"
        )

    write_json(status, {"state": "running", "phase": "operators"})
    fit_operator = MatrixFreeNativeMARatioJacobian(
        model,
        vectorizer,
        theta,
        fit,
        chunk_size=args.operator_chunk_size,
    )
    selection_operator = MatrixFreeNativeMARatioJacobian(
        model,
        vectorizer,
        theta,
        selection,
        chunk_size=args.operator_chunk_size,
    )
    fit_residual = fit_operator.residual()
    selection_residual = selection_operator.residual()
    fit_e2 = float(real_inner(fit_residual, fit_residual))
    selection_e2 = float(real_inner(selection_residual, selection_residual))
    gradient = fit_operator.vjp(fit_residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(
        fit_e2,
        np.finfo(float).tiny,
    )
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        raise FloatingPointError("native E2 channel has no finite GN scale")

    def progress(iteration: int, alpha: float, beta: float) -> None:
        if not args.quiet:
            print(
                f"native_gn_lanczos={iteration}/{args.lanczos_steps} "
                f"alpha={alpha:.6e} beta={beta:.6e}",
                flush=True,
            )

    write_json(status, {"state": "running", "phase": "gn_lanczos"})
    lanczos = lanczos_tridiagonal(
        fit_operator.normal,
        fit_residual,
        steps=args.lanczos_steps,
        callback=progress,
    )
    directions = []
    for ridge_factor in args.ridge_factors:
        ridge = float(ridge_factor * rayleigh_scale)
        dual = ridge_dual_from_lanczos(lanczos, ridge)
        delta = -fit_operator.vjp(dual)
        directions.append(
            {
                "ridge_factor": float(ridge_factor),
                "ridge": ridge,
                "delta": delta,
                "fit_tangent": fit_operator.jvp(delta),
                "selection_tangent": selection_operator.jvp(delta),
            }
        )

    write_json(status, {"state": "running", "phase": "baseline"})
    fit_baseline = model_summary(
        model,
        fit,
        batch_size=args.eval_batch_size,
    )
    selection_baseline = model_summary(
        model,
        selection,
        batch_size=args.eval_batch_size,
    )
    if not np.isclose(fit_baseline["e2"], fit_e2, rtol=5.0e-5, atol=1.0e-10):
        raise RuntimeError("fit native E2 operator and nonlinear audit disagree")
    if not np.isclose(
        selection_baseline["e2"],
        selection_e2,
        rtol=5.0e-5,
        atol=1.0e-10,
    ):
        raise RuntimeError("selection native E2 operator and audit disagree")

    rows = []
    deltas = []
    write_json(status, {"state": "running", "phase": "line_search"})
    for direction in directions:
        for step_scale in args.step_scales:
            delta = float(step_scale) * direction["delta"]
            linear_fit = (
                fit_residual + float(step_scale) * direction["fit_tangent"]
            )
            linear_selection = (
                selection_residual
                + float(step_scale) * direction["selection_tangent"]
            )
            vectorizer.commit_(model, theta + delta)
            fit_trial = model_summary(
                model,
                fit,
                batch_size=args.eval_batch_size,
            )
            selection_trial = model_summary(
                model,
                selection,
                batch_size=args.eval_batch_size,
            )
            fit_actual_capture = capture(fit_e2, fit_trial["e2"])
            selection_actual_capture = capture(
                selection_e2,
                selection_trial["e2"],
            )
            fit_linear_capture = capture(
                fit_e2,
                float(real_inner(linear_fit, linear_fit)),
            )
            selection_linear_capture = capture(
                selection_e2,
                float(real_inner(linear_selection, linear_selection)),
            )
            positive = bool(
                fit_trial["minimum_metric_eigenvalue"] > 0
                and selection_trial["minimum_metric_eigenvalue"] > 0
            )
            eligible = bool(
                positive
                and fit_actual_capture > 0
                and selection_actual_capture > 0
                and selection_trial["sigma"] <= selection_baseline["sigma"]
            )
            row = {
                "ridge_factor": direction["ridge_factor"],
                "ridge": direction["ridge"],
                "step_scale": float(step_scale),
                "step_rms": float(torch.sqrt(torch.mean(torch.abs(delta) ** 2))),
                "fit_linear_capture": fit_linear_capture,
                "selection_linear_capture": selection_linear_capture,
                "fit_actual_capture": fit_actual_capture,
                "selection_actual_capture": selection_actual_capture,
                "fit": fit_trial,
                "selection": selection_trial,
                "positive_metric": positive,
                "sigma_nonworse_on_selection": bool(
                    selection_trial["sigma"] <= selection_baseline["sigma"]
                ),
                "eligible": eligible,
            }
            rows.append(row)
            deltas.append(delta.detach().clone())
            print(
                f"ridge_factor={direction['ridge_factor']:.6g} "
                f"step={step_scale:.6g} "
                f"fit_E2_capture={fit_actual_capture:.6f} "
                f"selection_E2_capture={selection_actual_capture:.6f} "
                f"selection_sigma={selection_trial['sigma']:.8e} "
                f"eligible={eligible}",
                flush=True,
            )
            vectorizer.commit_(model, theta)

    eligible_indices = [
        index for index, row in enumerate(rows) if bool(row["eligible"])
    ]
    selected_index = (
        min(
            eligible_indices,
            key=lambda index: (
                rows[index]["selection"]["e2"],
                rows[index]["selection"]["sigma"],
                rows[index]["ridge_factor"],
                rows[index]["step_scale"],
            ),
        )
        if eligible_indices
        else None
    )
    accepted = selected_index is not None
    selected = None if selected_index is None else rows[selected_index]
    if accepted:
        vectorizer.commit_(model, theta + deltas[selected_index])
        output_payload = copy.deepcopy(payload)
        output_payload["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        output_payload["native_ma_channel_refinement"] = {
            "bond_index": args.bond_index,
            "candidate_seed": args.candidate_seed,
            "ridge_factor": selected["ridge_factor"],
            "step_scale": selected["step_scale"],
            "source_model": str(model_path),
            "source_model_sha256": sha256_file(model_path),
            "selection_is_not_confirmation": True,
        }
        temporary = output_model.with_suffix(output_model.suffix + ".tmp")
        torch.save(output_payload, temporary)
        temporary.replace(output_model)

    report = {
        "schema": "quintic-hard-symmetry-channel-native-ma-v2",
        "scientific_scope": {
            "purpose": (
                "Hold the architecture, bond, and one-sided channel seed fixed "
                "while replacing teacher-logdet fitting by pure native E2."
            ),
            "normalization_derivative": (
                "J_r = diag(r) (I - 1 pi^T) J_logdet, with "
                "pi_i = w_i r_i; the volume normalization is not detached."
            ),
            "acceptance": (
                "Fit and independent-selection E2 must improve, selection sigma "
                "must be nonworse, and the metric must remain positive. A fresh "
                "confirmation set remains unopened."
            ),
        },
        "configuration": {
            "candidate_start": args.candidate_start,
            "candidate_limit": args.candidate_limit,
            "fit_size": args.fit_size,
            "selection_size": args.selection_size,
            "bond_index": args.bond_index,
            "candidate_seed": args.candidate_seed,
            "operator_chunk_size": args.operator_chunk_size,
            "eval_batch_size": args.eval_batch_size,
            "lanczos_steps": args.lanczos_steps,
            "ridge_factors": list(args.ridge_factors),
            "step_scales": list(args.step_scales),
            "device": str(device),
            "precision": str(payload["precision"]),
            "seed": args.seed,
        },
        "source": {
            "expanded_model": str(model_path),
            "expanded_model_sha256": sha256_file(model_path),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "validation_pullbacks": str(pullbacks_path),
            "validation_pullbacks_sha256": sha256_file(pullbacks_path),
            "excluded_reports": [str(path) for path in exclude_reports],
            "excluded_index_artifacts": [
                str(path) for path in excluded_artifacts
            ],
            "indices": str(indices_path),
            "indices_sha256": sha256_file(indices_path),
            "output_model": str(output_model) if accepted else None,
            "output_model_sha256": sha256_file(output_model) if accepted else None,
        },
        "data_isolation": {
            "preceding_used_index_count": int(previous.size),
            "fit_index_count": int(fit_indices.size),
            "selection_index_count": int(selection_indices.size),
            "overlap_count": int(overlap.size),
        },
        "parameter_count": parameter_count,
        "activation": activation,
        "active_complex_parameter_count": int(theta.numel()),
        "epoch_zero_probe_raw_max_abs_change": epoch_zero_max_change,
        "fit_baseline": fit_baseline,
        "selection_baseline": selection_baseline,
        "gradient_norm": gradient_norm,
        "rayleigh_scale": rayleigh_scale,
        "lanczos": {
            "steps_completed": int(lanczos.tridiagonal.shape[0]),
            "breakdown": bool(lanczos.breakdown),
            "basis_orthogonality_error": lanczos.orthogonality_error,
        },
        "rows": rows,
        "accepted_on_selection": accepted,
        "selected": selected,
        "next_gate": (
            "Open one fresh paired native-MA confirmation set, then stop the "
            "one-sided branch and proceed to the full bond 5-6 supercore."
            if accepted
            else "Do not open confirmation; proceed directly to the full bond "
            "5-6 native-E2 supercore."
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(status, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "accepted_on_selection": accepted,
                "selected": selected,
                "output_model": str(output_model) if accepted else None,
                "output_report": str(output_report),
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
