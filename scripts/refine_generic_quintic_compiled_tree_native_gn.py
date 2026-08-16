#!/usr/bin/env python3
"""Teacher-free native-E2 GN/LM refinement of a compiled generic tree.

The input checkpoint is self-contained.  This script never reads the
full-H teacher or compiler artifact.  It learns one tree-node update on a fit
split, selects ridge and step size on a disjoint split, and opens the
confirmation split only for the unique frozen candidate.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.native_e2_gauss_newton import (  # noqa: E402
    MatrixFreeNormalizedE2Jacobian,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    ComplexParameterVectorizer,
    lanczos_tridiagonal,
    real_inner,
    ridge_dual_from_lanczos,
    vector_norm,
)
from scripts.evaluate_generic_quintic_h4_architecture_arms import (  # noqa: E402
    build_checkpoint_model,
    infer_architecture,
)
from scripts.train_generic_quintic_compiled_tree import (  # noqa: E402
    metric_row,
    sha256_file,
    whiten_dataset,
)
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    make_dataset,
    paired_improvement,
    tail_guard,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--fit-points", type=Path, required=True)
    parser.add_argument("--fit-pullbacks", type=Path, required=True)
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
        help="previous data_indices NPZ files excluded from every matching split",
    )
    parser.add_argument("--fit-size", type=int, default=1024)
    parser.add_argument("--selection-size", type=int, default=2048)
    parser.add_argument("--confirmation-size", type=int, default=5000)
    parser.add_argument(
        "--stage",
        choices=("root", "internal"),
        default="root",
        help="Optimize only the root tensor or all existing internal tensors.",
    )
    parser.add_argument("--lanczos-steps", type=int, default=12)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(100.0, 10.0, 1.0, 0.1),
    )
    parser.add_argument(
        "--line-search-alphas",
        type=float,
        nargs="+",
        default=(0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--operator-chunk-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--maximum-relative-parameter-step",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument("--seed", type=int, default=202607435)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.fit_size,
        args.selection_size,
        args.confirmation_size,
        args.lanczos_steps,
        args.operator_chunk_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all split, batch, thread, and Lanczos sizes must be positive")
    if (
        not args.ridge_factors
        or any(not np.isfinite(value) or value <= 0 for value in args.ridge_factors)
    ):
        raise ValueError("ridge factors must be finite and positive")
    if (
        not args.line_search_alphas
        or any(
            not np.isfinite(value) or value <= 0 or value > 1
            for value in args.line_search_alphas
        )
    ):
        raise ValueError("line-search alphas must lie in (0, 1]")
    if (
        not np.isfinite(args.maximum_relative_parameter_step)
        or args.maximum_relative_parameter_step <= 0
    ):
        raise ValueError("maximum relative parameter step must be positive")
    if args.maximum_selection_tail_relative_degradation < 0:
        raise ValueError("tail degradation allowance cannot be negative")


def _pool_arrays(
    points_path: Path,
    pullbacks_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.load(points_path, allow_pickle=False)
    values = np.asarray(points["X"], dtype=np.float32)
    labels = np.column_stack(
        (
            np.asarray(points["weights"], dtype=np.float64),
            np.asarray(points["omega_squared"], dtype=np.float64),
        )
    )
    pullbacks = np.load(pullbacks_path, mmap_mode="r")
    if len(values) != len(labels) or len(values) != len(pullbacks):
        raise RuntimeError("generic-quintic point-pool arrays are not aligned")
    return values, labels, pullbacks


def load_disjoint_splits(
    specifications: dict[str, tuple[Path, Path, int]],
    *,
    seed: int,
    exclusions: dict[str, np.ndarray] | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, np.ndarray],
]:
    """Sample disjoint rows whenever two logical splits share one pool."""

    grouped: dict[tuple[Path, Path], list[tuple[str, int]]] = {}
    for name, (points, pullbacks, size) in specifications.items():
        key = (points.expanduser().resolve(), pullbacks.expanduser().resolve())
        grouped.setdefault(key, []).append((name, size))

    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    indices: dict[str, np.ndarray] = {}
    for group_index, ((points_path, pullbacks_path), rows) in enumerate(
        grouped.items()
    ):
        values, labels, pullbacks = _pool_arrays(points_path, pullbacks_path)
        total = sum(size for _, size in rows)
        excluded_rows = (
            np.unique(
                np.concatenate(
                    [
                        exclusions.get(name, np.empty(0, dtype=np.int64))
                        for name, _ in rows
                    ]
                )
            )
            if exclusions is not None
            else np.empty(0, dtype=np.int64)
        )
        if np.any((excluded_rows < 0) | (excluded_rows >= len(values))):
            raise ValueError("excluded point index is outside its source pool")
        available = np.setdiff1d(
            np.arange(len(values), dtype=np.int64),
            excluded_rows,
            assume_unique=False,
        )
        if total > len(available):
            raise ValueError("requested disjoint splits exceed a shared point pool")
        permutation = np.random.default_rng(
            seed + 1009 * group_index
        ).permutation(available)[:total]
        offset = 0
        for name, size in rows:
            selected = np.asarray(
                permutation[offset : offset + size],
                dtype=np.int64,
            )
            arrays[name] = (
                np.asarray(values[selected]),
                np.asarray(labels[selected]),
                np.asarray(pullbacks[selected]),
            )
            indices[name] = selected
            offset += size
    return arrays, indices


def load_excluded_indices(
    paths: tuple[Path, ...] | list[Path],
) -> dict[str, np.ndarray]:
    """Merge prior fit/native, selection, and confirmation index ledgers."""

    result: dict[str, list[np.ndarray]] = {
        "fit": [],
        "selection": [],
        "confirmation": [],
    }
    for path in paths:
        artifact = np.load(path.expanduser().resolve(), allow_pickle=False)
        for logical_name, alternatives in {
            "fit": ("fit", "native"),
            "selection": ("selection",),
            "confirmation": ("confirmation",),
        }.items():
            for key in alternatives:
                if key in artifact.files:
                    result[logical_name].append(
                        np.asarray(artifact[key], dtype=np.int64)
                    )
                    break
    return {
        name: (
            np.unique(np.concatenate(rows))
            if rows
            else np.empty(0, dtype=np.int64)
        )
        for name, rows in result.items()
    }


def set_parameter_vector_(
    model: torch.nn.Module,
    vectorizer: ComplexParameterVectorizer,
    vector: torch.Tensor,
) -> None:
    values = vectorizer.unpack(vector)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            named[name].copy_(value)


def json_candidate(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("_vector", None)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a native GN run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    started = time.perf_counter()

    try:
        checkpoint_path = args.initial_checkpoint.expanduser().resolve()
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if infer_architecture(payload) != "compiled-tree":
            raise ValueError("native tree GN requires a compiled-tree checkpoint")
        if bool(payload.get("teacher_runtime_dependency", False)):
            raise ValueError("input checkpoint still declares a teacher dependency")
        configuration = payload["configuration"]
        dtype = (
            torch.complex64
            if str(configuration["precision"]) == "complex64"
            else torch.complex128
        )
        exponents = np.asarray(configuration["exponents"], dtype=np.int64)
        whitening = np.asarray(
            configuration["whitening"],
            dtype=np.complex128,
        )
        model = build_checkpoint_model(payload, device=device)
        model.set_trainable_stage_(args.stage)
        parameter_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        vectorizer = ComplexParameterVectorizer.from_module(
            model,
            parameter_names,
        )
        theta = vectorizer.pack(model).detach()
        baseline_state = copy.deepcopy(model.state_dict())

        write_json(status_path, {"state": "running", "phase": "features"})
        excluded_indices = load_excluded_indices(args.exclude_indices_file)
        split_arrays, split_indices = load_disjoint_splits(
            {
                "fit": (
                    args.fit_points,
                    args.fit_pullbacks,
                    args.fit_size,
                ),
                "selection": (
                    args.selection_points,
                    args.selection_pullbacks,
                    args.selection_size,
                ),
                "confirmation": (
                    args.confirmation_points,
                    args.confirmation_pullbacks,
                    args.confirmation_size,
                ),
            },
            seed=args.seed,
            exclusions=excluded_indices,
        )
        datasets = {
            name: whiten_dataset(
                make_dataset(
                    arrays,
                    exponents,
                    feature_batch_size=args.feature_batch_size,
                    complex_dtype=dtype,
                    device=device,
                ),
                whitening,
            )
            for name, arrays in split_arrays.items()
        }

        baseline_selection, baseline_selection_ratio, baseline_selection_tail = (
            metric_row(
                model,
                datasets["selection"],
                chunk_size=args.eval_batch_size,
            )
        )
        write_json(status_path, {"state": "running", "phase": "linearization"})
        fit_operator = MatrixFreeNormalizedE2Jacobian(
            model,
            vectorizer.unpack,
            theta,
            datasets["fit"],
            chunk_size=args.operator_chunk_size,
        )
        selection_operator = MatrixFreeNormalizedE2Jacobian(
            model,
            vectorizer.unpack,
            theta,
            datasets["selection"],
            chunk_size=args.operator_chunk_size,
        )
        residual = fit_operator.residual()
        residual_norm_squared = float(real_inner(residual, residual))
        gradient = fit_operator.vjp(residual)
        gradient_norm = float(vector_norm(gradient))
        rayleigh_scale = gradient_norm**2 / max(
            residual_norm_squared,
            np.finfo(np.float64).tiny,
        )
        if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
            raise FloatingPointError("native E2 has no finite root-node direction")

        def progress(iteration: int, alpha: float, beta: float) -> None:
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "lanczos",
                    "iteration": iteration,
                    "iterations": args.lanczos_steps,
                    "alpha": alpha,
                    "beta": beta,
                },
            )
            if not args.quiet:
                print(
                    f"lanczos={iteration}/{args.lanczos_steps} "
                    f"alpha={alpha:.6e} beta={beta:.6e}",
                    flush=True,
                )

        lanczos = lanczos_tridiagonal(
            fit_operator.normal,
            residual,
            steps=args.lanczos_steps,
            callback=progress,
        )
        theta_norm = max(
            float(vector_norm(theta)),
            np.finfo(np.float64).tiny,
        )
        selection_base_e2 = selection_operator.native_e2
        candidates: list[dict[str, Any]] = []
        write_json(status_path, {"state": "running", "phase": "selection"})
        for ridge_factor in args.ridge_factors:
            ridge = float(ridge_factor * rayleigh_scale)
            dual = ridge_dual_from_lanczos(lanczos, ridge)
            base_delta = -fit_operator.vjp(dual)
            predicted_fit = residual + fit_operator.jvp(base_delta)
            predicted_selection = (
                selection_operator.residual()
                + selection_operator.jvp(base_delta)
            )
            for line_alpha in args.line_search_alphas:
                delta = float(line_alpha) * base_delta
                candidate_vector = (theta + delta).detach()
                relative_step = float(vector_norm(delta)) / theta_norm
                row: dict[str, Any] = {
                    "ridge_factor": float(ridge_factor),
                    "ridge": ridge,
                    "alpha": float(line_alpha),
                    "relative_parameter_step": relative_step,
                    "predicted_fit_capture_at_alpha_1": (
                        1.0
                        - float(real_inner(predicted_fit, predicted_fit))
                        / residual_norm_squared
                    ),
                    "predicted_selection_capture_at_alpha_1": (
                        1.0
                        - float(
                            real_inner(
                                predicted_selection,
                                predicted_selection,
                            )
                        )
                        / selection_base_e2
                    ),
                    "eligible": False,
                    "failure": None,
                    "_vector": candidate_vector.detach().clone(),
                }
                if relative_step > args.maximum_relative_parameter_step:
                    row["failure"] = "relative_parameter_step"
                    candidates.append(row)
                    continue
                try:
                    fit_candidate_residual = fit_operator.residual(
                        candidate_vector
                    )
                    selection_candidate_residual = selection_operator.residual(
                        candidate_vector
                    )
                    set_parameter_vector_(model, vectorizer, candidate_vector)
                    statistics, _, tail = metric_row(
                        model,
                        datasets["selection"],
                        chunk_size=args.eval_batch_size,
                    )
                    actual_fit_e2 = float(
                        real_inner(
                            fit_candidate_residual,
                            fit_candidate_residual,
                        )
                    )
                    actual_selection_e2 = float(
                        real_inner(
                            selection_candidate_residual,
                            selection_candidate_residual,
                        )
                    )
                    row.update(
                        {
                            "actual_fit_e2": actual_fit_e2,
                            "actual_fit_capture": (
                                1.0 - actual_fit_e2 / residual_norm_squared
                            ),
                            "actual_selection_e2": actual_selection_e2,
                            "actual_selection_capture": (
                                1.0
                                - actual_selection_e2 / selection_base_e2
                            ),
                            "selection_statistics": statistics,
                            "selection_tail": tail,
                        }
                    )
                    row["eligible"] = bool(
                        actual_fit_e2 < residual_norm_squared
                        and actual_selection_e2 < selection_base_e2
                        and statistics["sigma_official_formula"]
                        < baseline_selection["sigma_official_formula"]
                        and tail_guard(
                            tail,
                            baseline_selection_tail,
                            relative_degradation=(
                                args.maximum_selection_tail_relative_degradation
                            ),
                        )
                    )
                except (RuntimeError, FloatingPointError) as error:
                    row["failure"] = f"{type(error).__name__}: {error}"
                finally:
                    model.load_state_dict(baseline_state)
                candidates.append(row)

        eligible = [row for row in candidates if row["eligible"]]
        selected = (
            min(
                eligible,
                key=lambda row: (
                    row["actual_selection_e2"],
                    row["selection_statistics"]["sigma_official_formula"],
                    row["relative_parameter_step"],
                ),
            )
            if eligible
            else None
        )

        confirmation_passes = False
        baseline_confirmation = None
        baseline_confirmation_tail = None
        candidate_confirmation = None
        candidate_confirmation_tail = None
        paired = None
        if selected is not None:
            model.load_state_dict(baseline_state)
            baseline_confirmation, baseline_confirmation_ratio, (
                baseline_confirmation_tail
            ) = metric_row(
                model,
                datasets["confirmation"],
                chunk_size=args.eval_batch_size,
            )
            set_parameter_vector_(model, vectorizer, selected["_vector"])
            candidate_confirmation, candidate_confirmation_ratio, (
                candidate_confirmation_tail
            ) = metric_row(
                model,
                datasets["confirmation"],
                chunk_size=args.eval_batch_size,
            )
            paired = paired_improvement(
                baseline_confirmation_ratio,
                candidate_confirmation_ratio,
                datasets["confirmation"]["weights_numpy"],
            )
            confirmation_passes = bool(
                candidate_confirmation["sigma_official_formula"]
                < baseline_confirmation["sigma_official_formula"]
                and candidate_confirmation["weighted_rms_abs_residual"]
                < baseline_confirmation["weighted_rms_abs_residual"]
                and paired["e2"]["ci95_low"] > 0
                and paired["sigma"]["ci95_low"] > 0
                and tail_guard(
                    candidate_confirmation_tail,
                    baseline_confirmation_tail,
                    relative_degradation=0.0,
                )
            )
        if not confirmation_passes:
            model.load_state_dict(baseline_state)

        output_payload = copy.deepcopy(payload)
        output_payload["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        output_payload["parent_checkpoint_sha256"] = sha256_file(
            checkpoint_path
        )
        output_payload["teacher_runtime_dependency"] = False
        output_payload["confirmation_passes"] = confirmation_passes
        output_payload["native_tree_gn"] = {
            "stage": args.stage,
            "selected_parameter_names": list(parameter_names),
            "selected_complex_parameter_count": int(theta.numel()),
            "selected_real_parameter_count": int(2 * theta.numel()),
            "normalization_derivative_included": True,
            "confirmation_passes": confirmation_passes,
        }
        saved_checkpoint = output_dir / (
            "accepted.pt" if confirmation_passes else "baseline_retained.pt"
        )
        torch.save(output_payload, saved_checkpoint)
        if selected is not None:
            torch.save(
                {
                    "schema": "generic-quintic-native-tree-gn-proposal-v1",
                    "parameter_names": parameter_names,
                    "parameter_vector": selected["_vector"].detach().cpu(),
                    "accepted": confirmation_passes,
                },
                output_dir / "candidate_proposal.pt",
            )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **split_indices)

        report = {
            "schema": "generic-quintic-compiled-tree-native-gn-v1",
            "teacher_role": "absent_after_round_zero",
            "teacher_runtime_dependency": False,
            "objective": {
                "name": "model-normalized native MA E2",
                "normalization_derivative_included": True,
                "formula": (
                    "sum_i w_i (exp(ell_i-logsumexp_j(log(w_j)+ell_j))-1)^2"
                ),
            },
            "configuration": {
                **vars(args),
                "initial_checkpoint": str(checkpoint_path),
                "output_dir": str(output_dir),
            },
            "model": {
                "stage": args.stage,
                "edge_dimensions": list(model.edge_dimensions),
                "total_real_parameter_count": model.trainable_real_parameter_count,
                "selected_parameter_names": list(parameter_names),
                "selected_complex_parameter_count": int(theta.numel()),
                "selected_real_parameter_count": int(2 * theta.numel()),
            },
            "linearization": {
                "fit_native_e2": residual_norm_squared,
                "selection_native_e2": selection_base_e2,
                "gradient_norm": gradient_norm,
                "rayleigh_scale": rayleigh_scale,
                "lanczos": {
                    "dimension": int(lanczos.tridiagonal.shape[0]),
                    "operator_applications": lanczos.operator_applications,
                    "breakdown": lanczos.breakdown,
                    "orthogonality_error": lanczos.orthogonality_error,
                },
            },
            "selection": {
                "baseline": baseline_selection,
                "baseline_tail": baseline_selection_tail,
                "candidates": [json_candidate(row) for row in candidates],
                "selected": (
                    json_candidate(selected) if selected is not None else None
                ),
            },
            "confirmation": {
                "opened": selected is not None,
                "baseline": baseline_confirmation,
                "baseline_tail": baseline_confirmation_tail,
                "candidate": candidate_confirmation,
                "candidate_tail": candidate_confirmation_tail,
                "paired_improvement": paired,
                "passes": confirmation_passes,
            },
            "data": {
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "excluded_indices_files": [
                    {
                        "path": str(path.expanduser().resolve()),
                        "sha256": sha256_file(path.expanduser().resolve()),
                    }
                    for path in args.exclude_indices_file
                ],
                "excluded_counts": {
                    name: int(len(values))
                    for name, values in excluded_indices.items()
                },
                "counts": {
                    name: int(len(values))
                    for name, values in split_indices.items()
                },
            },
            "checkpoint": str(saved_checkpoint),
            "checkpoint_sha256": sha256_file(saved_checkpoint),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "selection_found_candidate": selected is not None,
                "confirmation_passes": confirmation_passes,
                "checkpoint": str(saved_checkpoint),
            },
        )
        print(status_path.read_text(encoding="utf-8"), flush=True)
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
