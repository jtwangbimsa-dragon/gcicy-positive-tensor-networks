#!/usr/bin/env python3
"""Test a stable three-site Tucker space with a small native-E2 GN problem."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.func import functional_call


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    positive_tensor_network_from_artifact_payload,
)
from scripts.refine_quintic_hard_symmetry_channel_native_ma import (  # noqa: E402
    model_summary,
)
from scripts.refine_quintic_tn_rank_growth_native_ma_als import (  # noqa: E402
    orbit_augment_dataset,
    three_site_normal_gradient_analysis,
)
from scripts.refine_quintic_tn_stable_schmidt_coefficients import (  # noqa: E402
    consensus_principal_basis,
    explicit_native_residual_jacobian,
    random_subspace_ordered_null,
    significant_prefix,
)
from scripts.scan_quintic_tn_three_site_native_ma import (  # noqa: E402
    normalize_three_site_supercore_scale_,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--three-site-start", type=int, required=True)
    parser.add_argument("--subspace-rank", type=int, default=4)
    parser.add_argument("--maximum-mode-rank", type=int, default=4)
    parser.add_argument("--subspace-fit-train-start", type=int, default=431_000)
    parser.add_argument("--subspace-fit-size", type=int, default=512)
    parser.add_argument(
        "--subspace-selection-validation-start",
        type=int,
        default=81_000,
    )
    parser.add_argument("--subspace-selection-size", type=int, default=512)
    parser.add_argument("--coefficient-fit-train-start", type=int, default=432_000)
    parser.add_argument("--coefficient-fit-size", type=int, default=512)
    parser.add_argument(
        "--coefficient-selection-validation-start",
        type=int,
        default=82_000,
    )
    parser.add_argument("--coefficient-selection-size", type=int, default=512)
    parser.add_argument("--subspace-group-samples", type=int, default=2)
    parser.add_argument("--optimization-group-samples", type=int, default=4)
    parser.add_argument("--operator-chunk-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--nonlinear-shortlist", type=int, default=4)
    parser.add_argument("--minimum-selection-capture", type=float, default=0.005)
    parser.add_argument("--null-draws", type=int, default=10_000)
    parser.add_argument("--null-quantile", type=float, default=0.99)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01),
    )
    parser.add_argument(
        "--step-scales",
        type=float,
        nargs="+",
        default=(0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--seed", type=int, default=202607284)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.subspace_rank,
        args.maximum_mode_rank,
        args.subspace_fit_size,
        args.subspace_selection_size,
        args.coefficient_fit_size,
        args.coefficient_selection_size,
        args.subspace_group_samples,
        args.optimization_group_samples,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.nonlinear_shortlist,
        args.null_draws,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("rank, sample, batch, and draw counts must be positive")
    starts = (
        args.three_site_start,
        args.subspace_fit_train_start,
        args.subspace_selection_validation_start,
        args.coefficient_fit_train_start,
        args.coefficient_selection_validation_start,
    )
    if any(value < 0 for value in starts):
        raise ValueError("dataset and three-site starts cannot be negative")
    if args.maximum_mode_rank > args.subspace_rank:
        raise ValueError("maximum mode rank cannot exceed discovery rank")
    if not 0 < args.null_quantile < 1:
        raise ValueError("null quantile must lie in (0, 1)")
    if not 0 < args.minimum_selection_capture < 1:
        raise ValueError("minimum selection capture must lie in (0, 1)")
    if any(not np.isfinite(value) or value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be finite and positive")
    if any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in args.step_scales
    ):
        raise ValueError("step scales must lie in (0, 1]")


def real_coefficients_to_three_site_core(
    parameter: torch.Tensor,
    baseline_core: torch.Tensor,
    left_basis: torch.Tensor,
    middle_basis: torch.Tensor,
    right_basis: torch.Tensor,
) -> torch.Tensor:
    """Map real Tucker coefficients to Theta0 + U C W V."""

    ranks = (
        int(left_basis.shape[1]),
        int(middle_basis.shape[1]),
        int(right_basis.shape[1]),
    )
    complex_count = int(np.prod(ranks))
    if parameter.ndim != 1 or parameter.numel() != 2 * complex_count:
        raise ValueError("reduced three-site coefficient vector has the wrong shape")
    coefficient = torch.complex(
        parameter[:complex_count],
        parameter[complex_count:],
    ).reshape(ranks)
    update = torch.einsum(
        "ai,bj,ck,ijk->abc",
        left_basis,
        middle_basis,
        right_basis,
        coefficient,
    )
    left_bond, right_bond, first_physical, middle_physical, right_physical = (
        baseline_core.shape
    )
    expected = (
        left_bond * first_physical,
        middle_physical,
        right_physical * right_bond,
    )
    if update.shape != expected:
        raise ValueError("reduced update does not match the active three-site core")
    update_core = update.reshape(
        left_bond,
        first_physical,
        middle_physical,
        right_physical,
        right_bond,
    ).permute(0, 4, 1, 2, 3)
    return baseline_core + update_core


def reduced_three_site_raw_function_factory(
    model: torch.nn.Module,
    baseline_core: torch.Tensor,
    left_basis: torch.Tensor,
    middle_basis: torch.Tensor,
    right_basis: torch.Tensor,
    dataset: dict[str, Any],
) -> Callable[[slice], Callable[[torch.Tensor], torch.Tensor]]:
    def factory(selection: slice) -> Callable[[torch.Tensor], torch.Tensor]:
        values = dataset["values"][selection]
        derivatives = dataset["derivatives"][selection]
        log_omega = dataset["log_omega"][selection]

        def raw(parameter: torch.Tensor) -> torch.Tensor:
            core = real_coefficients_to_three_site_core(
                parameter,
                baseline_core,
                left_basis,
                middle_basis,
                right_basis,
            )
            metric = functional_call(
                model,
                {"three_site_core": core},
                (values, derivatives),
                strict=False,
            )
            return training_log_volume(metric, "cholesky") - log_omega

        return raw

    return factory


def set_reduced_parameter_(
    model: torch.nn.Module,
    parameter: torch.Tensor,
    baseline_core: torch.Tensor,
    left_basis: torch.Tensor,
    middle_basis: torch.Tensor,
    right_basis: torch.Tensor,
) -> None:
    core = real_coefficients_to_three_site_core(
        parameter,
        baseline_core,
        left_basis,
        middle_basis,
        right_basis,
    )
    with torch.no_grad():
        model.three_site_core.copy_(core)


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_report = args.output_report.expanduser().resolve()
    status_path = output_report.with_suffix(output_report.suffix + ".status.json")
    indices_path = output_report.with_suffix(output_report.suffix + ".indices.npz")
    for path in (output_report, status_path, indices_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {path}")
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(status_path, {"state": "running", "phase": "loading"})

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    complex_dtype = torch.complex128
    real_dtype = torch.float64

    model_path = args.model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    train_pullbacks_path = pullbacks_dir / "train_pullbacks.npy"
    validation_pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    for path in (
        model_path,
        dataset_path,
        train_pullbacks_path,
        validation_pullbacks_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "quintic-positive-tensor-network-v1":
        raise ValueError("three-site audit requires a quintic TN artifact")
    if payload.get("architecture") != "shared_local_dictionary":
        raise ValueError("three-site audit requires a shared dictionary")
    source_degree = int(payload.get("source_degree", 1))
    if source_degree != 1:
        raise ValueError("Fermat orbit augmentation requires an O(1) source")
    source_bond_dimension = int(payload["bond_dimension"])
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    source_model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=complex_dtype)
    source_model.requires_grad_(False).eval()
    if args.three_site_start + 2 >= source_model.site_count:
        raise ValueError("selected three-site block lies outside the chain")

    data = np.load(dataset_path, allow_pickle=False)
    train_pullbacks = np.load(train_pullbacks_path, mmap_mode="r")
    validation_pullbacks = np.load(validation_pullbacks_path, mmap_mode="r")
    windows = {
        "subspace_fit": (
            "train",
            args.subspace_fit_train_start,
            args.subspace_fit_size,
        ),
        "subspace_selection": (
            "validation",
            args.subspace_selection_validation_start,
            args.subspace_selection_size,
        ),
        "coefficient_fit": (
            "train",
            args.coefficient_fit_train_start,
            args.coefficient_fit_size,
        ),
        "coefficient_selection": (
            "validation",
            args.coefficient_selection_validation_start,
            args.coefficient_selection_size,
        ),
    }
    indices: dict[str, np.ndarray] = {}
    for name, (domain, start, count) in windows.items():
        available = len(data["X_train"] if domain == "train" else data["X_val"])
        stop = start + count
        if stop > available:
            raise ValueError(f"{name} window exceeds {domain} data")
        indices[name] = np.arange(start, stop, dtype=np.int64)
    np.savez_compressed(indices_path, **indices)

    def make_split(name: str) -> dict[str, Any]:
        domain = windows[name][0]
        selected = indices[name]
        if domain == "train":
            x = data["X_train"]
            y = data["y_train"]
            pullbacks = train_pullbacks
        else:
            x = data["X_val"]
            y = data["y_val"]
            pullbacks = validation_pullbacks
        return tensor_split(
            np.asarray(x[selected], dtype=np.float32),
            np.asarray(pullbacks[selected]),
            np.asarray(y[selected], dtype=np.float64),
            source_degree=source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )

    canonical = copy.deepcopy(source_model)
    canonical.mixed_canonicalize_coefficient_block_(args.three_site_start, 3)
    triple_scale = normalize_three_site_supercore_scale_(
        canonical,
        args.three_site_start,
    )

    write_json(status_path, {"state": "running", "phase": "subspace_discovery"})
    subspace_fit = make_split("subspace_fit")
    subspace_selection = make_split("subspace_selection")
    subspace_generator = torch.Generator(device=device)
    subspace_generator.manual_seed(args.seed + 1009)
    subspace_actions = fixed_fermat_actions_torch(
        args.subspace_group_samples,
        generator=subspace_generator,
        complex_dtype=complex_dtype,
        device=device,
    )
    fit_analysis = three_site_normal_gradient_analysis(
        canonical,
        orbit_augment_dataset(subspace_fit, subspace_actions),
        start=args.three_site_start,
        source_bond_dimension=source_bond_dimension,
        chunk_size=args.operator_chunk_size,
        score_ranks=(args.subspace_rank,),
    )
    selection_analysis = three_site_normal_gradient_analysis(
        canonical,
        orbit_augment_dataset(subspace_selection, subspace_actions),
        start=args.three_site_start,
        source_bond_dimension=source_bond_dimension,
        chunk_size=args.operator_chunk_size,
        score_ranks=(args.subspace_rank,),
    )

    mode_bases: dict[str, torch.Tensor] = {}
    mode_spectra: dict[str, list[float]] = {}
    mode_nulls: dict[str, dict[str, Any]] = {}
    mode_prefixes: dict[str, int] = {}
    for mode_index, mode in enumerate(("left", "middle", "right")):
        fit_basis = fit_analysis[f"{mode}_basis"][:, : args.subspace_rank]
        selection_basis = selection_analysis[f"{mode}_basis"][
            :, : args.subspace_rank
        ]
        consensus, spectrum = consensus_principal_basis(
            fit_basis,
            selection_basis,
            args.subspace_rank,
        )
        null = random_subspace_ordered_null(
            int(consensus.shape[0]),
            args.subspace_rank,
            draws=args.null_draws,
            quantile=args.null_quantile,
            seed=args.seed + 4001 + mode_index,
        )
        prefix = min(
            args.maximum_mode_rank,
            significant_prefix(spectrum, null["ordered_quantiles"]),
        )
        if prefix <= 0:
            raise RuntimeError(f"no stable {mode} mode exceeds the random null")
        mode_bases[mode] = consensus[:, :prefix].detach()
        mode_spectra[mode] = spectrum
        mode_nulls[mode] = null
        mode_prefixes[mode] = prefix
    del fit_analysis, selection_analysis, subspace_fit, subspace_selection
    release_cuda()

    optimization_model = copy.deepcopy(canonical)
    baseline_core = (
        optimization_model.activate_three_site_coefficient_core_(
            args.three_site_start
        )
        .detach()
        .clone()
    )
    optimization_model.three_site_core.requires_grad_(False)
    optimization_model.requires_grad_(False).eval()
    complex_coefficient_count = int(np.prod(list(mode_prefixes.values())))
    theta = torch.zeros(
        2 * complex_coefficient_count,
        dtype=real_dtype,
        device=device,
    )

    coefficient_generator = torch.Generator(device=device)
    coefficient_generator.manual_seed(args.seed + 2009)
    coefficient_actions = fixed_fermat_actions_torch(
        args.optimization_group_samples,
        generator=coefficient_generator,
        complex_dtype=complex_dtype,
        device=device,
    )
    coefficient_fit = make_split("coefficient_fit")
    coefficient_selection = make_split("coefficient_selection")
    coefficient_fit_objective = orbit_augment_dataset(
        coefficient_fit,
        coefficient_actions,
    )
    coefficient_selection_objective = orbit_augment_dataset(
        coefficient_selection,
        coefficient_actions,
    )

    def jacobian_progress(label: str) -> Callable[[int, int], None]:
        def progress(index: int, count: int) -> None:
            if index == 1 or index == count or index % max(1, count // 20) == 0:
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": f"coefficient_{label}_jacobian",
                        "iteration": index,
                        "count": count,
                    },
                )
                if not args.quiet:
                    print(f"{label} jacobian={index}/{count}", flush=True)

        return progress

    fit_factory = reduced_three_site_raw_function_factory(
        optimization_model,
        baseline_core,
        mode_bases["left"],
        mode_bases["middle"],
        mode_bases["right"],
        coefficient_fit_objective,
    )
    selection_factory = reduced_three_site_raw_function_factory(
        optimization_model,
        baseline_core,
        mode_bases["left"],
        mode_bases["middle"],
        mode_bases["right"],
        coefficient_selection_objective,
    )
    fit_residual, fit_jacobian, fit_linearization = (
        explicit_native_residual_jacobian(
            theta,
            coefficient_fit_objective,
            fit_factory,
            chunk_size=args.operator_chunk_size,
            progress=jacobian_progress("fit"),
        )
    )
    selection_residual, selection_jacobian, selection_linearization = (
        explicit_native_residual_jacobian(
            theta,
            coefficient_selection_objective,
            selection_factory,
            chunk_size=args.operator_chunk_size,
            progress=jacobian_progress("selection"),
        )
    )
    curvature = torch.transpose(fit_jacobian, 0, 1) @ fit_jacobian
    gradient = torch.transpose(fit_jacobian, 0, 1) @ fit_residual
    curvature_scale = float(
        torch.trace(curvature) / max(1, curvature.shape[0])
    )
    if not np.isfinite(curvature_scale) or curvature_scale <= 0:
        raise FloatingPointError("reduced GN curvature has no finite scale")
    identity = torch.eye(
        curvature.shape[0],
        dtype=curvature.dtype,
        device=curvature.device,
    )
    fit_e2 = float(torch.dot(fit_residual, fit_residual))
    selection_e2 = float(torch.dot(selection_residual, selection_residual))
    rows: list[dict[str, Any]] = []
    deltas: list[torch.Tensor] = []
    for ridge_factor in args.ridge_factors:
        ridge = float(ridge_factor * curvature_scale)
        delta_base = torch.linalg.solve(
            curvature + ridge * identity,
            -gradient,
        )
        for step_scale in args.step_scales:
            delta = float(step_scale) * delta_base
            predicted_fit = float(
                torch.dot(
                    fit_residual + fit_jacobian @ delta,
                    fit_residual + fit_jacobian @ delta,
                )
            )
            predicted_selection = float(
                torch.dot(
                    selection_residual + selection_jacobian @ delta,
                    selection_residual + selection_jacobian @ delta,
                )
            )
            fit_capture = 1.0 - predicted_fit / fit_e2
            selection_capture = 1.0 - predicted_selection / selection_e2
            rows.append(
                {
                    "ridge_factor": float(ridge_factor),
                    "ridge": ridge,
                    "step_scale": float(step_scale),
                    "parameter_rms": float(torch.sqrt(torch.mean(delta**2))),
                    "predicted_fit_e2": predicted_fit,
                    "predicted_selection_e2": predicted_selection,
                    "predicted_fit_capture": fit_capture,
                    "predicted_selection_capture": selection_capture,
                    "passes_prediction_floor": bool(
                        fit_capture > 0
                        and selection_capture >= args.minimum_selection_capture
                    ),
                    "shortlisted": False,
                    "actual": None,
                    "eligible": False,
                }
            )
            deltas.append(delta.detach().clone())

    predicted_eligible = [
        index for index, row in enumerate(rows) if row["passes_prediction_floor"]
    ]
    shortlisted = sorted(
        predicted_eligible,
        key=lambda index: (
            rows[index]["predicted_selection_e2"],
            rows[index]["predicted_fit_e2"],
            index,
        ),
    )[: args.nonlinear_shortlist]
    for index in shortlisted:
        rows[index]["shortlisted"] = True

    coefficient_baseline = {
        "base": model_summary(
            optimization_model,
            coefficient_selection,
            batch_size=args.eval_batch_size,
        ),
        "objective": model_summary(
            optimization_model,
            coefficient_selection_objective,
            batch_size=args.eval_batch_size,
        ),
    }
    for index in shortlisted:
        set_reduced_parameter_(
            optimization_model,
            deltas[index],
            baseline_core,
            mode_bases["left"],
            mode_bases["middle"],
            mode_bases["right"],
        )
        actual = {
            "base": model_summary(
                optimization_model,
                coefficient_selection,
                batch_size=args.eval_batch_size,
            ),
            "objective": model_summary(
                optimization_model,
                coefficient_selection_objective,
                batch_size=args.eval_batch_size,
            ),
        }
        rows[index]["actual"] = actual
        base_capture = (
            1.0 - actual["base"]["e2"] / coefficient_baseline["base"]["e2"]
        )
        objective_capture = (
            1.0
            - actual["objective"]["e2"]
            / coefficient_baseline["objective"]["e2"]
        )
        rows[index]["actual_base_e2_capture"] = base_capture
        rows[index]["actual_objective_e2_capture"] = objective_capture
        rows[index]["eligible"] = bool(
            base_capture >= args.minimum_selection_capture
            and objective_capture >= args.minimum_selection_capture
            and actual["base"]["sigma"] <= coefficient_baseline["base"]["sigma"]
            and actual["objective"]["sigma"]
            <= coefficient_baseline["objective"]["sigma"]
            and actual["base"]["minimum_metric_eigenvalue"] > 0
            and actual["objective"]["minimum_metric_eigenvalue"] > 0
        )
        set_reduced_parameter_(
            optimization_model,
            theta,
            baseline_core,
            mode_bases["left"],
            mode_bases["middle"],
            mode_bases["right"],
        )
    eligible = [index for index, row in enumerate(rows) if row["eligible"]]
    selected_index = (
        min(
            eligible,
            key=lambda index: (
                rows[index]["actual"]["objective"]["e2"],
                rows[index]["actual"]["base"]["e2"],
                index,
            ),
        )
        if eligible
        else None
    )
    best_predicted_selection_capture = max(
        (row["predicted_selection_capture"] for row in rows),
        default=float("-inf"),
    )
    report = {
        "schema": "quintic-tn-three-site-reduced-reachability-v1",
        "scientific_scope": {
            "purpose": (
                "Test whether a cross-sample-stable three-site Tucker support "
                "contains a native-E2 direction large enough to justify an "
                "independent nonlinear gate."
            ),
            "parameterization": (
                "The unrestricted three-site core is used only for support "
                "discovery. Optimization occurs in the adaptive Tucker "
                "coefficient tensor."
            ),
            "stop_rule": (
                "No gate is opened unless predicted selection E2 capture is "
                f"at least {args.minimum_selection_capture:.3%}; actual base "
                "and orbit selection captures must also clear that floor."
            ),
            "claim_limit": (
                "Passing this audit only nominates a candidate for a new "
                "independent gate; it does not commit a model."
            ),
        },
        "configuration": {
            **{
                key: value
                for key, value in vars(args).items()
                if not isinstance(value, Path)
            },
            "model": str(model_path),
            "source_run_dir": str(source_dir),
            "pullbacks_dir": str(pullbacks_dir),
            "output_report": str(output_report),
            "device": str(device),
        },
        "source": {
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "train_pullbacks": str(train_pullbacks_path),
            "train_pullbacks_sha256": sha256_file(train_pullbacks_path),
            "validation_pullbacks": str(validation_pullbacks_path),
            "validation_pullbacks_sha256": sha256_file(
                validation_pullbacks_path
            ),
            "indices": str(indices_path),
            "indices_sha256": sha256_file(indices_path),
        },
        "triple_scale": triple_scale,
        "subspace": {
            "principal_cosine_squares": mode_spectra,
            "random_nulls": mode_nulls,
            "significant_prefixes": mode_prefixes,
            "complex_coefficient_parameters": complex_coefficient_count,
            "real_coefficient_parameters": int(theta.numel()),
        },
        "linearization": {
            "fit": fit_linearization,
            "selection": selection_linearization,
            "curvature_scale": curvature_scale,
        },
        "coefficient_selection": {
            "baseline": coefficient_baseline,
            "rows": rows,
            "best_predicted_selection_capture": (
                best_predicted_selection_capture
            ),
            "selected_index": selected_index,
            "qualifies_for_independent_gate": selected_index is not None,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(
        status_path,
        {
            "state": "complete",
            "phase": "complete",
            "three_site_start": args.three_site_start,
            "best_predicted_selection_capture": (
                best_predicted_selection_capture
            ),
            "selected_index": selected_index,
            "qualifies_for_independent_gate": selected_index is not None,
            "wall_seconds": report["wall_seconds"],
        },
    )
    print(
        json.dumps(
            {
                "three_site_start": args.three_site_start,
                "mode_ranks": mode_prefixes,
                "real_coefficient_parameters": int(theta.numel()),
                "best_predicted_selection_capture": (
                    best_predicted_selection_capture
                ),
                "selected_index": selected_index,
                "qualifies_for_independent_gate": selected_index is not None,
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
