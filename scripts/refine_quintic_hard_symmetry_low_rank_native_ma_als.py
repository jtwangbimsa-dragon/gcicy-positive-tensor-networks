#!/usr/bin/env python3
"""Rotate both factors of a nested low-rank native-MA supercore update."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    lanczos_tridiagonal,
    real_inner,
    ridge_coefficients_from_lanczos,
    select_disjoint_indices,
    vector_norm,
)
from scripts.refine_quintic_hard_symmetry_channel_native_ma import (  # noqa: E402
    MatrixFreeNativeMARatioJacobian,
    model_summary,
)
from scripts.refine_quintic_hard_symmetry_supercore_native_ma import (  # noqa: E402
    BondPairPrefixCache,
    capture,
    masks_and_vectorizer,
    nonlinear_screen_shortlist,
    orthonormal_right_seed_subspace,
    renormalized_dataset_prefix,
    set_whitened_channel_,
    tail_nonworse,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    tensor_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-model", type=Path, required=True)
    parser.add_argument("--right-seed-model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--exclude-report", type=Path, nargs="*", default=())
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--data-start", type=int, default=240_000)
    parser.add_argument("--data-limit", type=int, default=4096)
    parser.add_argument("--fit-size", type=int, default=512)
    parser.add_argument("--selection-size", type=int, default=512)
    parser.add_argument("--bond-index", type=int, default=5)
    parser.add_argument("--retained-rank", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--operator-chunk-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--cache-fit-prefix", action="store_true")
    parser.add_argument("--prefix-cache-batch-size", type=int, default=2)
    parser.add_argument("--screen-fit-size", type=int, default=128)
    parser.add_argument("--screen-selection-size", type=int, default=128)
    parser.add_argument("--nonlinear-shortlist", type=int, default=4)
    parser.add_argument("--lanczos-steps", type=int, default=4)
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
        default=(0.25, 0.5, 1.0),
    )
    parser.add_argument("--expected-parameter-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202607265)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.data_limit,
        args.fit_size,
        args.selection_size,
        args.retained_rank,
        args.sweeps,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.prefix_cache_batch_size,
        args.screen_fit_size,
        args.screen_selection_size,
        args.nonlinear_shortlist,
        args.lanczos_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all sample, rank, sweep, batch, and iteration sizes are positive")
    if args.data_start < 0 or args.bond_index < 0:
        raise ValueError("data start and bond index cannot be negative")
    if args.fit_size + args.selection_size > args.data_limit:
        raise ValueError("fit and selection sets exceed the data window")
    if args.screen_fit_size > args.fit_size:
        raise ValueError("screen fit prefix exceeds the fit set")
    if args.screen_selection_size > args.selection_size:
        raise ValueError("screen selection prefix exceeds the selection set")
    if args.expected_parameter_count < 0:
        raise ValueError("expected parameter count cannot be negative")
    if any(not np.isfinite(value) or value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be finite and positive")
    if any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in args.step_scales
    ):
        raise ValueError("step scales must lie in (0, 1]")


def report_indices(report_path: Path) -> tuple[np.ndarray, Path]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    source = report.get("source")
    if not isinstance(source, dict) or not source.get("indices"):
        raise ValueError(f"report has no indices artifact: {report_path}")
    indices_path = Path(source["indices"]).expanduser().resolve()
    if not indices_path.exists():
        raise FileNotFoundError(indices_path)
    with np.load(indices_path, allow_pickle=False) as payload:
        arrays = [
            np.asarray(payload[key], dtype=np.int64).reshape(-1)
            for key in payload.files
            if "indice" in key
        ]
    if not arrays:
        raise ValueError(f"indices artifact is empty: {indices_path}")
    return np.unique(np.concatenate(arrays)), indices_path


def selection_gate(
    fit_candidate: dict[str, Any],
    selection_candidate: dict[str, Any],
    fit_baseline: dict[str, Any],
    selection_baseline: dict[str, Any],
) -> bool:
    return bool(
        fit_candidate["minimum_metric_eigenvalue"] > 0
        and selection_candidate["minimum_metric_eigenvalue"] > 0
        and fit_candidate["e2"] < fit_baseline["e2"]
        and selection_candidate["e2"] < selection_baseline["e2"]
        and selection_candidate["sigma"] <= selection_baseline["sigma"]
        and tail_nonworse(selection_candidate, selection_baseline)
    )


def balanced_factors(
    update: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return balanced rank-r factors and their reconstruction diagnostic."""

    left, singular, right_adjoint = torch.linalg.svd(update, full_matrices=False)
    retained = min(rank, singular.numel())
    square_root = torch.sqrt(singular[:retained])
    left_factor = left[:, :retained] * square_root[None, :]
    right_factor = torch.transpose(
        right_adjoint[:retained] * square_root[:, None],
        0,
        1,
    )
    reconstructed = left_factor @ torch.transpose(right_factor, 0, 1)
    relative_error = float(
        torch.linalg.vector_norm(reconstructed - update)
        / torch.clamp(
            torch.linalg.vector_norm(update),
            min=torch.finfo(update.real.dtype).tiny,
        )
    )
    return left_factor, right_factor, singular[:retained], relative_error


def canonicalize_channels_(
    model: torch.nn.Module,
    *,
    bond_index: int,
    left_masks: list[torch.Tensor],
    right_masks: list[torch.Tensor],
    left_scales: torch.Tensor,
    right_scales: torch.Tensor,
) -> dict[str, Any]:
    left_parameter = dict(model.named_parameters())[
        f"blocked_two_site_orbit_parameters.{bond_index}"
    ]
    right_parameter = dict(model.named_parameters())[
        f"blocked_two_site_orbit_parameters.{bond_index + 1}"
    ]
    left_columns = [
        left_parameter[mask] * left_scales[mask] for mask in left_masks
    ]
    right_columns = [
        right_parameter[mask] * right_scales[mask] for mask in right_masks
    ]
    update = torch.stack(left_columns, dim=1) @ torch.transpose(
        torch.stack(right_columns, dim=1),
        0,
        1,
    )
    left_factor, right_factor, singular, relative_error = balanced_factors(
        update,
        len(left_masks),
    )
    tolerance = 2.0e-5 if update.dtype == torch.complex64 else 1.0e-11
    if relative_error > tolerance:
        raise RuntimeError(
            "balanced factorization changed the supercore update: "
            f"{relative_error:.6e} > {tolerance:.6e}"
        )
    with torch.no_grad():
        for mask in left_masks:
            left_parameter[mask] = 0
        for mask in right_masks:
            right_parameter[mask] = 0
    for index, (left_mask, right_mask) in enumerate(
        zip(left_masks, right_masks, strict=True)
    ):
        set_whitened_channel_(
            left_parameter,
            left_mask,
            left_scales,
            left_factor[:, index],
        )
        set_whitened_channel_(
            right_parameter,
            right_mask,
            right_scales,
            right_factor[:, index],
        )
    return {
        "singular_values": [float(value) for value in singular.detach().cpu()],
        "relative_reconstruction_error": relative_error,
        "tolerance": tolerance,
    }


def half_step(
    *,
    model: torch.nn.Module,
    side: str,
    vectorizer: Any,
    fit: dict[str, Any],
    selection: dict[str, Any],
    screen_fit: dict[str, Any],
    screen_selection: dict[str, Any],
    raw_function_factory: Any,
    args: argparse.Namespace,
    sweep: int,
    status: Path,
) -> dict[str, Any]:
    theta = vectorizer.pack(model)
    fit_operator = MatrixFreeNativeMARatioJacobian(
        model,
        vectorizer,
        theta,
        fit,
        chunk_size=args.operator_chunk_size,
        raw_function_factory=raw_function_factory,
    )
    selection_operator = MatrixFreeNativeMARatioJacobian(
        model,
        vectorizer,
        theta,
        selection,
        chunk_size=args.operator_chunk_size,
    )
    fit_residual = fit_operator.residual()
    fit_e2 = float(real_inner(fit_residual, fit_residual))
    gradient = fit_operator.vjp(fit_residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(fit_e2, np.finfo(float).tiny)
    fit_baseline = model_summary(model, fit, batch_size=args.eval_batch_size)
    selection_baseline = model_summary(
        model,
        selection,
        batch_size=args.eval_batch_size,
    )
    screen_fit_baseline = model_summary(
        model,
        screen_fit,
        batch_size=args.eval_batch_size,
    )
    screen_selection_baseline = model_summary(
        model,
        screen_selection,
        batch_size=args.eval_batch_size,
    )
    report: dict[str, Any] = {
        "sweep": sweep,
        "side": side,
        "active_complex_parameter_count": int(theta.numel()),
        "gradient_norm": gradient_norm,
        "rayleigh_scale": rayleigh_scale,
        "fit_baseline": fit_baseline,
        "selection_baseline": selection_baseline,
        "rows": [],
        "accepted": False,
        "selected": None,
    }
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        report["reason"] = "no finite residual-aligned tangent"
        return report

    def progress(iteration: int, alpha: float, beta: float) -> None:
        write_json(
            status,
            {
                "state": "running",
                "phase": "native_low_rank_als_lanczos",
                "sweep": sweep,
                "side": side,
                "iteration": iteration,
                "count": args.lanczos_steps,
            },
        )
        if not args.quiet:
            print(
                f"sweep={sweep} side={side} lanczos="
                f"{iteration}/{args.lanczos_steps} "
                f"alpha={alpha:.6e} beta={beta:.6e}",
                flush=True,
            )

    lanczos = lanczos_tridiagonal(
        fit_operator.normal,
        fit_residual,
        steps=args.lanczos_steps,
        callback=progress,
    )
    ridge_entries = []
    for ridge_factor in args.ridge_factors:
        ridge = float(ridge_factor * rayleigh_scale)
        coefficients = ridge_coefficients_from_lanczos(lanczos, ridge)
        ridge_entries.append(
            {
                "ridge_factor": float(ridge_factor),
                "ridge": ridge,
                "dual": lanczos.basis @ coefficients,
            }
        )
    ridge_deltas = -fit_operator.vjp_many(
        torch.stack([entry["dual"] for entry in ridge_entries])
    )
    rows = []
    deltas = []
    for ridge_index, entry in enumerate(ridge_entries):
        for step_scale in args.step_scales:
            delta = float(step_scale) * ridge_deltas[ridge_index]
            vectorizer.commit_(model, theta + delta)
            fit_trial = model_summary(
                model,
                screen_fit,
                batch_size=args.eval_batch_size,
            )
            selection_trial = model_summary(
                model,
                screen_selection,
                batch_size=args.eval_batch_size,
            )
            vectorizer.commit_(model, theta)
            row = {
                "ridge_factor": entry["ridge_factor"],
                "ridge": entry["ridge"],
                "step_scale": float(step_scale),
                "step_rms": float(torch.sqrt(torch.mean(torch.abs(delta) ** 2))),
                "screen": {
                    "fit": fit_trial,
                    "selection": selection_trial,
                    "fit_actual_capture": capture(
                        screen_fit_baseline["e2"],
                        fit_trial["e2"],
                    ),
                    "selection_actual_capture": capture(
                        screen_selection_baseline["e2"],
                        selection_trial["e2"],
                    ),
                    "positive_metric": bool(
                        fit_trial["minimum_metric_eigenvalue"] > 0
                        and selection_trial["minimum_metric_eigenvalue"] > 0
                    ),
                },
                "screen_shortlisted": False,
                "fit": None,
                "selection": None,
                "fit_actual_capture": None,
                "selection_actual_capture": None,
                "intermediate_eligible": False,
            }
            rows.append(row)
            deltas.append(delta.detach().clone())

    shortlisted = nonlinear_screen_shortlist(rows, args.nonlinear_shortlist)
    shortlisted_set = set(shortlisted)
    for index, row in enumerate(rows):
        row["screen_shortlisted"] = index in shortlisted_set
    for index in shortlisted:
        row = rows[index]
        vectorizer.commit_(model, theta + deltas[index])
        fit_trial = model_summary(model, fit, batch_size=args.eval_batch_size)
        selection_trial = model_summary(
            model,
            selection,
            batch_size=args.eval_batch_size,
        )
        vectorizer.commit_(model, theta)
        fit_capture = capture(fit_baseline["e2"], fit_trial["e2"])
        selection_capture = capture(
            selection_baseline["e2"],
            selection_trial["e2"],
        )
        intermediate_eligible = bool(
            fit_trial["minimum_metric_eigenvalue"] > 0
            and selection_trial["minimum_metric_eigenvalue"] > 0
            and fit_capture > 0
            and selection_capture > 0
            and selection_trial["sigma"] <= selection_baseline["sigma"]
        )
        row.update(
            {
                "fit": fit_trial,
                "selection": selection_trial,
                "fit_actual_capture": fit_capture,
                "selection_actual_capture": selection_capture,
                "intermediate_eligible": intermediate_eligible,
            }
        )
        if not args.quiet:
            print(
                f"sweep={sweep} side={side} "
                f"ridge_factor={row['ridge_factor']:.6g} "
                f"step={row['step_scale']:.6g} "
                f"fit_E2_capture={fit_capture:.6f} "
                f"selection_E2_capture={selection_capture:.6f} "
                f"selection_sigma={selection_trial['sigma']:.8e} "
                f"eligible={intermediate_eligible}",
                flush=True,
            )
    eligible = [
        index for index, row in enumerate(rows) if row["intermediate_eligible"]
    ]
    selected_index = (
        min(
            eligible,
            key=lambda index: (
                rows[index]["selection"]["e2"],
                rows[index]["selection"]["sigma"],
            ),
        )
        if eligible
        else None
    )
    report.update(
        {
            "lanczos": {
                "steps_completed": int(lanczos.tridiagonal.shape[0]),
                "breakdown": bool(lanczos.breakdown),
                "basis_orthogonality_error": lanczos.orthogonality_error,
            },
            "shortlisted_candidate_indices": shortlisted,
            "rows": rows,
            "selected": None if selected_index is None else rows[selected_index],
            "accepted": selected_index is not None,
        }
    )
    if selected_index is not None:
        vectorizer.commit_(model, theta + deltas[selected_index])
    return report


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_model = args.output_model.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    status = output_report.with_suffix(output_report.suffix + ".status.json")
    indices_path = output_report.with_suffix(output_report.suffix + ".indices.npz")
    for path in (output_model, output_report, status, indices_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {path}")
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
    right_seed_path = args.right_seed_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    pullbacks_path = pullbacks_dir / "train_pullbacks.npy"
    exclude_reports = tuple(path.expanduser().resolve() for path in args.exclude_report)
    for path in (
        model_path,
        right_seed_path,
        dataset_path,
        pullbacks_path,
        *exclude_reports,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    expansion = payload.get("phase_multiplicity_expansion")
    if not isinstance(expansion, dict):
        raise ValueError("input model is not an exact multiplicity expansion")
    source_multiplicity = int(expansion["source_multiplicity"])
    target_multiplicity = int(expansion["target_multiplicity"])
    if args.retained_rank > target_multiplicity - source_multiplicity:
        raise ValueError("expanded model has too few new multiplicity channels")
    copy_indices = tuple(
        range(
            source_multiplicity,
            source_multiplicity + args.retained_rank,
        )
    )
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
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64

    data = np.load(dataset_path, allow_pickle=False)
    data_stop = min(args.data_start + args.data_limit, len(data["X_train"]))
    data_count = data_stop - args.data_start
    if data_count < args.fit_size + args.selection_size:
        raise ValueError("data window is too small for fit and selection")
    local_fit, local_selection = select_disjoint_indices(
        data_count,
        args.fit_size,
        args.selection_size,
        seed=args.seed,
    )
    fit_indices = local_fit + args.data_start
    selection_indices = local_selection + args.data_start
    previous_arrays = []
    excluded_index_paths = []
    for report_path in exclude_reports:
        indices, path = report_indices(report_path)
        previous_arrays.append(indices)
        excluded_index_paths.append(path)
    previous_indices = (
        np.unique(np.concatenate(previous_arrays))
        if previous_arrays
        else np.empty(0, dtype=np.int64)
    )
    overlap = np.intersect1d(
        previous_indices,
        np.concatenate((fit_indices, selection_indices)),
        assume_unique=False,
    )
    if overlap.size:
        raise ValueError(f"ALS data overlap excluded reports at {overlap.size} points")
    np.savez_compressed(
        indices_path,
        fit_train_indices=fit_indices,
        selection_train_indices=selection_indices,
    )

    pullbacks = np.load(pullbacks_path, mmap_mode="r")

    def make_split(indices: np.ndarray) -> dict[str, Any]:
        return tensor_split(
            np.asarray(data["X_train"][indices], dtype=np.float32),
            np.asarray(pullbacks[indices]),
            np.asarray(data["y_train"][indices], dtype=np.float64),
            source_degree=source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )

    fit = make_split(fit_indices)
    selection = make_split(selection_indices)
    screen_fit = renormalized_dataset_prefix(fit, args.screen_fit_size)
    screen_selection = renormalized_dataset_prefix(
        selection,
        args.screen_selection_size,
    )
    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=complex_dtype)
    model.force_indexed_block_contraction = True
    model.requires_grad_(False).eval()
    parameter_count = int(model.trainable_real_parameter_count)
    if args.expected_parameter_count and parameter_count != args.expected_parameter_count:
        raise RuntimeError(
            f"parameter-count gate failed: {parameter_count} != "
            f"{args.expected_parameter_count}"
        )

    left_masks, left_vectorizer, left_scales = masks_and_vectorizer(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        copy_indices=copy_indices,
        side="left",
    )
    right_masks, right_vectorizer, right_scales = masks_and_vectorizer(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        copy_indices=copy_indices,
        side="right",
    )
    if bool(torch.any(torch.abs(left_vectorizer.pack(model)) > 0)):
        raise ValueError("new left channels are not zero in the exact embedding")
    if bool(torch.any(torch.abs(right_vectorizer.pack(model)) > 0)):
        raise ValueError("new right channels are not zero in the exact embedding")

    fit_baseline = model_summary(model, fit, batch_size=args.eval_batch_size)
    selection_baseline = model_summary(
        model,
        selection,
        batch_size=args.eval_batch_size,
    )
    seed_payload = torch.load(right_seed_path, map_location="cpu", weights_only=False)
    seed_expansion = seed_payload.get("phase_multiplicity_expansion")
    if not isinstance(seed_expansion, dict) or int(
        seed_expansion["target_multiplicity"]
    ) != target_multiplicity:
        raise ValueError("right seed model has incompatible multiplicity")
    right_parameter = dict(model.named_parameters())[right_vectorizer.name]
    seed_parameter = seed_payload["state_dict"][right_vectorizer.name].to(
        device=device,
        dtype=right_parameter.dtype,
    )
    right_scale_tensor = right_scales.reshape(right_parameter.shape)
    right_subspace, seed_norms, seed_orthogonality_error = (
        orthonormal_right_seed_subspace(
            seed_parameter,
            right_masks,
            right_scale_tensor,
        )
    )
    for column, mask in enumerate(right_masks):
        set_whitened_channel_(
            right_parameter,
            mask,
            right_scale_tensor,
            torch.conj(right_subspace[:, column]),
        )
    seeded_fit = model_summary(model, fit, batch_size=args.eval_batch_size)
    seeded_selection = model_summary(
        model,
        selection,
        batch_size=args.eval_batch_size,
    )
    epoch_zero_change = {
        "fit_sigma": seeded_fit["sigma"] - fit_baseline["sigma"],
        "fit_e2": seeded_fit["e2"] - fit_baseline["e2"],
        "selection_sigma": seeded_selection["sigma"]
        - selection_baseline["sigma"],
        "selection_e2": seeded_selection["e2"] - selection_baseline["e2"],
    }
    epoch_zero_tolerance = 1.0e-10 if complex_dtype == torch.complex128 else 1.0e-7
    if max(abs(value) for value in epoch_zero_change.values()) > epoch_zero_tolerance:
        raise RuntimeError("right-only seed changed the epoch-zero metric")

    prefix_cache = None
    if args.cache_fit_prefix:
        prefix_cache = BondPairPrefixCache(
            model,
            fit,
            bond_index=args.bond_index,
            batch_size=args.prefix_cache_batch_size,
        )
    raw_function_factory = (
        prefix_cache.raw_function if prefix_cache is not None else None
    )
    half_steps = []
    canonicalizations = []
    best_state = None
    best_fit = None
    best_selection = None
    best_score = None
    stopped_early = False
    for sweep in range(args.sweeps):
        for side in ("left", "right"):
            write_json(
                status,
                {
                    "state": "running",
                    "phase": "native_low_rank_als",
                    "sweep": sweep,
                    "side": side,
                },
            )
            _, vectorizer, _ = masks_and_vectorizer(
                model,
                bond_index=args.bond_index,
                source_multiplicity=source_multiplicity,
                target_multiplicity=target_multiplicity,
                copy_indices=copy_indices,
                side=side,
            )
            step = half_step(
                model=model,
                side=side,
                vectorizer=vectorizer,
                fit=fit,
                selection=selection,
                screen_fit=screen_fit,
                screen_selection=screen_selection,
                raw_function_factory=raw_function_factory,
                args=args,
                sweep=sweep,
                status=status,
            )
            half_steps.append(step)
            if not step["accepted"]:
                stopped_early = True
                break
            current_fit = model_summary(
                model,
                fit,
                batch_size=args.eval_batch_size,
            )
            current_selection = model_summary(
                model,
                selection,
                batch_size=args.eval_batch_size,
            )
            gate = selection_gate(
                current_fit,
                current_selection,
                fit_baseline,
                selection_baseline,
            )
            step["global_selection_gate"] = gate
            if gate:
                score = (current_selection["e2"], current_selection["sigma"])
                if best_score is None or score < best_score:
                    best_score = score
                    best_fit = current_fit
                    best_selection = current_selection
                    best_state = copy.deepcopy(model.state_dict())
        if stopped_early:
            break
        canonicalization = canonicalize_channels_(
            model,
            bond_index=args.bond_index,
            left_masks=left_masks,
            right_masks=right_masks,
            left_scales=left_scales.reshape(
                dict(model.named_parameters())[left_vectorizer.name].shape
            ),
            right_scales=right_scale_tensor,
        )
        canonicalization["sweep"] = sweep
        canonicalizations.append(canonicalization)

    accepted = best_state is not None
    if accepted:
        model.load_state_dict(best_state)
        output_payload = copy.deepcopy(payload)
        output_payload["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        output_payload["native_ma_low_rank_als"] = {
            "schema": "quintic-hard-symmetry-low-rank-native-ma-als-v1",
            "source_model": str(model_path),
            "source_model_sha256": sha256_file(model_path),
            "right_seed_model": str(right_seed_path),
            "right_seed_model_sha256": sha256_file(right_seed_path),
            "bond_index": args.bond_index,
            "retained_rank": args.retained_rank,
            "selection_is_not_confirmation": True,
        }
        temporary = output_model.with_suffix(output_model.suffix + ".tmp")
        torch.save(output_payload, temporary)
        temporary.replace(output_model)

    report = {
        "schema": "quintic-hard-symmetry-low-rank-native-ma-als-v1",
        "scientific_scope": {
            "purpose": (
                "Test whether rotating both factors of a strictly nested "
                "rank-limited supercore update restores bulk-and-tail "
                "generalization lost by a frozen right Schmidt span."
            ),
            "intermediate_rule": (
                "Each ALS half-step must improve fit and selection E2, retain "
                "positivity, and not worsen selection sigma. Tail worsening is "
                "allowed only as a temporary factor-rotation state."
            ),
            "commit_rule": (
                "Only a complete candidate that improves fit and selection E2 "
                "relative to the original baseline, keeps sigma and every "
                "registered selection tail statistic nonworse, and remains "
                "positive is saved."
            ),
            "claim_limit": (
                "The saved model is a development candidate. It requires a new "
                "one-time confirmation and tail audit before scientific use."
            ),
        },
        "configuration": {
            "data_domain": "X_train",
            "data_start": args.data_start,
            "data_limit": args.data_limit,
            "fit_size": args.fit_size,
            "selection_size": args.selection_size,
            "screen_fit_size": args.screen_fit_size,
            "screen_selection_size": args.screen_selection_size,
            "nonlinear_shortlist": args.nonlinear_shortlist,
            "bond_index": args.bond_index,
            "retained_rank": args.retained_rank,
            "sweeps": args.sweeps,
            "operator_chunk_size": args.operator_chunk_size,
            "eval_batch_size": args.eval_batch_size,
            "cache_fit_prefix": bool(args.cache_fit_prefix),
            "fit_prefix_host_bytes": (
                prefix_cache.host_bytes if prefix_cache is not None else 0
            ),
            "prefix_cache_batch_size": args.prefix_cache_batch_size,
            "lanczos_steps": args.lanczos_steps,
            "ridge_factors": list(args.ridge_factors),
            "step_scales": list(args.step_scales),
            "precision": str(payload["precision"]),
            "device": str(device),
            "seed": args.seed,
        },
        "source": {
            "expanded_model": str(model_path),
            "expanded_model_sha256": sha256_file(model_path),
            "right_seed_model": str(right_seed_path),
            "right_seed_model_sha256": sha256_file(right_seed_path),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "train_pullbacks": str(pullbacks_path),
            "train_pullbacks_sha256": sha256_file(pullbacks_path),
            "indices": str(indices_path),
            "indices_sha256": sha256_file(indices_path),
            "excluded_reports": [str(path) for path in exclude_reports],
            "excluded_index_artifacts": [
                str(path) for path in excluded_index_paths
            ],
            "output_model": str(output_model) if accepted else None,
            "output_model_sha256": sha256_file(output_model) if accepted else None,
        },
        "data_isolation": {
            "preceding_used_index_count": int(previous_indices.size),
            "current_used_index_count": int(args.fit_size + args.selection_size),
            "overlap_count": int(overlap.size),
        },
        "parameter_count": parameter_count,
        "source_multiplicity": source_multiplicity,
        "target_multiplicity": target_multiplicity,
        "right_seed": {
            "channel_norms": seed_norms,
            "orthogonality_error": seed_orthogonality_error,
            "epoch_zero_change": epoch_zero_change,
            "epoch_zero_tolerance": epoch_zero_tolerance,
        },
        "fit_baseline": fit_baseline,
        "selection_baseline": selection_baseline,
        "half_steps": half_steps,
        "canonicalizations": canonicalizations,
        "stopped_early": stopped_early,
        "accepted_on_selection": accepted,
        "selected_fit": best_fit,
        "selected_selection": best_selection,
        "next_gate": (
            "Generate a fresh one-time confirmation and 20k tail audit."
            if accepted
            else "Do not open confirmation; test rank 8 on unused development data."
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(status, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "accepted_on_selection": accepted,
                "output_model": str(output_model) if accepted else None,
                "output_report": str(output_report),
                "half_steps": len(half_steps),
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
