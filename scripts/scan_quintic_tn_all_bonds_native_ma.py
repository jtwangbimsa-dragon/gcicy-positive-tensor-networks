#!/usr/bin/env python3
"""Rank all TN bonds by stable native-MA normal-gradient Schmidt scores."""

from __future__ import annotations

import argparse
import copy
import gc
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

from gcicy_metric.pipeline import (  # noqa: E402
    positive_tensor_network_from_artifact_payload,
)
from scripts.refine_quintic_hard_symmetry_channel_native_ma import (  # noqa: E402
    model_summary,
)
from scripts.refine_quintic_tn_rank_growth_native_ma_als import (  # noqa: E402
    maximum_relative_metric_difference,
    normal_gradient_schmidt_analysis,
    orbit_augment_dataset,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    fixed_fermat_actions_torch,
    tensor_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--fit-train-start", type=int, default=400_000)
    parser.add_argument("--fit-size", type=int, default=64)
    parser.add_argument("--selection-validation-start", type=int, default=25_000)
    parser.add_argument("--selection-size", type=int, default=64)
    parser.add_argument("--preservation-size", type=int, default=8)
    parser.add_argument("--group-samples", type=int, default=2)
    parser.add_argument("--operator-chunk-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--shortlist-size", type=int, default=6)
    parser.add_argument("--score-ranks", type=int, nargs="+", default=(4, 8))
    parser.add_argument("--primary-rank", type=int, default=4)
    parser.add_argument("--bonds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--search-precision",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument("--seed", type=int, default=202607275)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.fit_size,
        args.selection_size,
        args.preservation_size,
        args.group_samples,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.shortlist_size,
        args.primary_rank,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sample, batch, shortlist, and rank sizes must be positive")
    if args.fit_train_start < 0 or args.selection_validation_start < 0:
        raise ValueError("dataset starts cannot be negative")
    if not args.score_ranks or any(rank <= 0 for rank in args.score_ranks):
        raise ValueError("score ranks must be nonempty and positive")
    if any(rank not in (4, 8) for rank in args.score_ranks):
        raise ValueError("this scanner currently reports rank-4 and rank-8 scores")
    if args.primary_rank not in args.score_ranks:
        raise ValueError("primary rank must be one of the score ranks")
    if args.bonds is not None and len(set(args.bonds)) != len(args.bonds):
        raise ValueError("bond list contains duplicates")


def normalize_pair_supercore_scale_(
    model: torch.nn.Module,
    bond_index: int,
    *,
    target_norm: float = 1.0,
) -> dict[str, float]:
    """Fix the metric-invariant scale by normalizing one merged pair core."""

    if target_norm <= 0 or not np.isfinite(target_norm):
        raise ValueError("target pair norm must be finite and positive")
    if bond_index < 0 or bond_index + 1 >= model.site_count:
        raise ValueError("pair scale bond lies outside the chain")
    left = model.coefficient_cores[bond_index]
    right = model.coefficient_cores[bond_index + 1]
    merged = torch.einsum("lmq,mrs->lrqs", left, right)
    observed = torch.linalg.vector_norm(merged)
    if not bool(torch.isfinite(observed)) or float(observed) <= 0:
        raise FloatingPointError("merged pair core has no finite norm")
    scale = target_norm / float(observed)
    with torch.no_grad():
        left.mul_(scale)
        model.positive_floor *= scale**2
    normalized = torch.linalg.vector_norm(
        torch.einsum("lmq,mrs->lrqs", left, right)
    )
    return {
        "observed_pair_norm": float(observed),
        "scale": float(scale),
        "normalized_pair_norm": float(normalized),
    }


def truncated_normal_gradient(
    factors: dict[str, torch.Tensor],
    rank: int,
) -> torch.Tensor:
    """Reconstruct the rank-r normal-gradient matrix from Schmidt factors."""

    singular = factors["singular_values"]
    stop = min(int(rank), int(singular.numel()))
    if stop <= 0:
        raise ValueError("truncated normal gradient requires positive rank")
    return (
        factors["left_singular_vectors"][:, :stop]
        * singular[:stop][None, :]
    ) @ factors["right_adjoint"][:stop]


def real_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the real inner-product cosine used by the complex optimizer."""

    if left.shape != right.shape:
        raise ValueError("gradient matrices must have matching shapes")
    numerator = torch.real(torch.sum(torch.conj(left) * right))
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    tiny = torch.finfo(denominator.dtype).tiny
    return float(numerator / torch.clamp(denominator, min=tiny))


def principal_subspace_overlap(
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
    rank: int,
) -> float:
    """Return the mean squared cosine between two rank-r column spaces."""

    if left_basis.ndim != 2 or right_basis.ndim != 2:
        raise ValueError("subspace bases must be matrices")
    if left_basis.shape[0] != right_basis.shape[0]:
        raise ValueError("subspace bases must share their ambient dimension")
    stop = min(int(rank), int(left_basis.shape[1]), int(right_basis.shape[1]))
    if stop <= 0:
        raise ValueError("subspace overlap requires positive rank")
    overlap = (
        torch.conj(torch.transpose(left_basis[:, :stop], 0, 1))
        @ right_basis[:, :stop]
    )
    value = torch.sum(torch.abs(overlap) ** 2) / stop
    return float(torch.clamp(value.real, min=0.0, max=1.0))


def principal_cosine_squares(
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
    rank: int,
) -> list[float]:
    """Return ordered squared canonical correlations between two subspaces."""

    if left_basis.ndim != 2 or right_basis.ndim != 2:
        raise ValueError("subspace bases must be matrices")
    if left_basis.shape[0] != right_basis.shape[0]:
        raise ValueError("subspace bases must share their ambient dimension")
    stop = min(int(rank), int(left_basis.shape[1]), int(right_basis.shape[1]))
    if stop <= 0:
        raise ValueError("principal cosines require positive rank")
    overlap = (
        torch.conj(torch.transpose(left_basis[:, :stop], 0, 1))
        @ right_basis[:, :stop]
    )
    singular = torch.linalg.svdvals(overlap)
    return [
        float(value)
        for value in torch.square(singular).real.detach().cpu()
    ]


def schmidt_subspace_overlaps(
    fit_factors: dict[str, torch.Tensor],
    selection_factors: dict[str, torch.Tensor],
    rank: int,
) -> dict[str, float]:
    """Compare the left and right Schmidt spaces of two gradient estimates."""

    left = principal_subspace_overlap(
        fit_factors["left_singular_vectors"],
        selection_factors["left_singular_vectors"],
        rank,
    )
    fit_right = torch.conj(
        torch.transpose(fit_factors["right_adjoint"], 0, 1)
    )
    selection_right = torch.conj(
        torch.transpose(selection_factors["right_adjoint"], 0, 1)
    )
    right = principal_subspace_overlap(fit_right, selection_right, rank)
    return {
        "left": left,
        "right": right,
        "two_sided_geometric_mean": math.sqrt(left * right),
        "left_principal_cosine_squares": principal_cosine_squares(
            fit_factors["left_singular_vectors"],
            selection_factors["left_singular_vectors"],
            rank,
        ),
        "right_principal_cosine_squares": principal_cosine_squares(
            fit_right,
            selection_right,
            rank,
        ),
    }


def compact_factors(
    analysis: dict[str, Any],
    maximum_rank: int,
) -> dict[str, torch.Tensor]:
    singular = analysis["singular_values"]
    stop = min(int(maximum_rank), int(singular.numel()))
    return {
        "left_singular_vectors": (
            analysis["left_singular_vectors"][:, :stop].detach().cpu()
        ),
        "singular_values": singular[:stop].detach().cpu(),
        "right_adjoint": analysis["right_adjoint"][:stop].detach().cpu(),
    }


def normalized_rank_score(
    diagnostics: dict[str, Any],
    rank: int,
) -> float:
    score = diagnostics[f"normal_gradient_score_rank{rank}"]
    return float(
        score / max(diagnostics["fit_native_e2"], np.finfo(float).tiny)
    )


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
    complex_dtype = (
        torch.complex64
        if args.search_precision == "complex64"
        else torch.complex128
    )
    real_dtype = (
        torch.float32 if complex_dtype == torch.complex64 else torch.float64
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

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
        raise ValueError("scanner requires a quintic TN artifact")
    if payload.get("architecture") != "shared_local_dictionary":
        raise ValueError("scanner requires a shared local dictionary")
    source_degree = int(payload.get("source_degree", 1))
    if source_degree != 1:
        raise ValueError("Fermat orbit scanning currently requires an O(1) source")
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

    all_bonds = list(range(source_model.site_count - 1))
    bonds = all_bonds if args.bonds is None else list(args.bonds)
    if any(bond not in all_bonds for bond in bonds):
        raise ValueError("requested bond lies outside the tensor-network chain")
    maximum_rank = max(args.score_ranks)

    data = np.load(dataset_path, allow_pickle=False)
    fit_stop = args.fit_train_start + args.fit_size
    selection_stop = args.selection_validation_start + args.selection_size
    if fit_stop > len(data["X_train"]):
        raise ValueError("fit window exceeds X_train")
    if selection_stop > len(data["X_val"]):
        raise ValueError("selection window exceeds X_val")
    fit_indices = np.arange(args.fit_train_start, fit_stop, dtype=np.int64)
    selection_indices = np.arange(
        args.selection_validation_start,
        selection_stop,
        dtype=np.int64,
    )
    np.savez_compressed(
        indices_path,
        fit_train_indices=fit_indices,
        selection_validation_indices=selection_indices,
    )
    train_pullbacks = np.load(train_pullbacks_path, mmap_mode="r")
    validation_pullbacks = np.load(validation_pullbacks_path, mmap_mode="r")

    def make_split(
        *,
        domain: str,
        indices: np.ndarray,
    ) -> dict[str, Any]:
        if domain == "train":
            x = data["X_train"]
            y = data["y_train"]
            pullbacks = train_pullbacks
        elif domain == "validation":
            x = data["X_val"]
            y = data["y_val"]
            pullbacks = validation_pullbacks
        else:
            raise ValueError("unknown data domain")
        return tensor_split(
            np.asarray(x[indices], dtype=np.float32),
            np.asarray(pullbacks[indices]),
            np.asarray(y[indices], dtype=np.float64),
            source_degree=source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )

    fit = make_split(domain="train", indices=fit_indices)
    selection = make_split(domain="validation", indices=selection_indices)
    preservation = {
        **selection,
        "count": min(args.preservation_size, selection["count"]),
        "values": selection["values"][: args.preservation_size],
        "derivatives": selection["derivatives"][: args.preservation_size],
        "weights": selection["weights"][: args.preservation_size],
        "weights_numpy": selection["weights_numpy"][: args.preservation_size],
        "log_omega": selection["log_omega"][: args.preservation_size],
    }
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.seed + 1009)
    actions = fixed_fermat_actions_torch(
        args.group_samples,
        generator=action_generator,
        complex_dtype=complex_dtype,
        device=device,
    )
    fit_objective = orbit_augment_dataset(fit, actions)
    selection_objective = orbit_augment_dataset(selection, actions)
    baseline = {
        "fit_objective": model_summary(
            source_model,
            fit_objective,
            batch_size=args.eval_batch_size,
        ),
        "selection_objective": model_summary(
            source_model,
            selection_objective,
            batch_size=args.eval_batch_size,
        ),
    }

    stage_one: list[dict[str, Any]] = []
    fit_factors: dict[int, dict[str, torch.Tensor]] = {}
    preservation_tolerance = (
        5.0e-5 if complex_dtype == torch.complex64 else 2.0e-10
    )
    for scan_index, bond in enumerate(bonds, start=1):
        bond_started = time.perf_counter()
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "fit_scan",
                "bond": bond,
                "iteration": scan_index,
                "count": len(bonds),
            },
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        canonical = copy.deepcopy(source_model)
        canonical.mixed_canonicalize_coefficient_pair_(bond)
        scale = normalize_pair_supercore_scale_(canonical, bond)
        relative_change = maximum_relative_metric_difference(
            source_model,
            canonical,
            preservation,
            batch_size=args.eval_batch_size,
        )
        if relative_change > preservation_tolerance:
            raise RuntimeError(
                f"bond {bond} gauge normalization changed the metric by "
                f"{relative_change:.3e}"
            )
        left_active = int(canonical.coefficient_cores[bond].shape[0])
        right_active = int(canonical.coefficient_cores[bond + 1].shape[1])
        analysis = normal_gradient_schmidt_analysis(
            canonical,
            fit_objective,
            bond_index=bond,
            source_bond_dimension=source_bond_dimension,
            left_active=left_active,
            right_active=right_active,
            chunk_size=args.operator_chunk_size,
        )
        diagnostics = analysis["diagnostics"]
        fit_factors[bond] = compact_factors(analysis, maximum_rank)
        scores = {
            f"rank{rank}": {
                "absolute": diagnostics[f"normal_gradient_score_rank{rank}"],
                "over_native_e2": normalized_rank_score(diagnostics, rank),
                "normal_energy_fraction": diagnostics[
                    f"normal_gradient_energy_fraction_rank{rank}"
                ],
            }
            for rank in args.score_ranks
        }
        row = {
            "bond": bond,
            "left_active": left_active,
            "right_active": right_active,
            "pair_scale": scale,
            "gauge_metric_relative_change": relative_change,
            "fit": diagnostics,
            "scores": scores,
            "wall_seconds": time.perf_counter() - bond_started,
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        }
        stage_one.append(row)
        if not args.quiet:
            print(
                f"fit bond={bond:02d} "
                f"rank{args.primary_rank}_score="
                f"{scores[f'rank{args.primary_rank}']['over_native_e2']:.6e} "
                f"normal_fraction="
                f"{diagnostics['normal_gradient_fraction_of_full']:.3f} "
                f"wall={row['wall_seconds']:.1f}s",
                flush=True,
            )
        del analysis, canonical
        release_cuda()

    fit_order = sorted(
        stage_one,
        key=lambda row: (
            row["scores"][f"rank{args.primary_rank}"]["over_native_e2"],
            -row["bond"],
        ),
        reverse=True,
    )
    shortlisted_bonds = [
        row["bond"] for row in fit_order[: min(args.shortlist_size, len(fit_order))]
    ]

    stage_two: list[dict[str, Any]] = []
    for selection_index, bond in enumerate(shortlisted_bonds, start=1):
        bond_started = time.perf_counter()
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "selection_rescan",
                "bond": bond,
                "iteration": selection_index,
                "count": len(shortlisted_bonds),
            },
        )
        canonical = copy.deepcopy(source_model)
        canonical.mixed_canonicalize_coefficient_pair_(bond)
        normalize_pair_supercore_scale_(canonical, bond)
        left_active = int(canonical.coefficient_cores[bond].shape[0])
        right_active = int(canonical.coefficient_cores[bond + 1].shape[1])
        analysis = normal_gradient_schmidt_analysis(
            canonical,
            selection_objective,
            bond_index=bond,
            source_bond_dimension=source_bond_dimension,
            left_active=left_active,
            right_active=right_active,
            chunk_size=args.operator_chunk_size,
        )
        diagnostics = analysis["diagnostics"]
        selection_factors = compact_factors(analysis, maximum_rank)
        fit_row = next(row for row in stage_one if row["bond"] == bond)
        agreements = {}
        composite_scores = {}
        subspace_overlaps = {}
        subspace_scores = {}
        selection_scores = {}
        for rank in args.score_ranks:
            fit_matrix = truncated_normal_gradient(fit_factors[bond], rank)
            selection_matrix = truncated_normal_gradient(selection_factors, rank)
            agreement = real_cosine(fit_matrix, selection_matrix)
            fit_score = fit_row["scores"][f"rank{rank}"]["over_native_e2"]
            selection_score = normalized_rank_score(diagnostics, rank)
            composite = (
                math.sqrt(max(fit_score, 0.0) * max(selection_score, 0.0))
                * max(agreement, 0.0)
            )
            overlap = schmidt_subspace_overlaps(
                fit_factors[bond],
                selection_factors,
                rank,
            )
            subspace_score = (
                math.sqrt(max(fit_score, 0.0) * max(selection_score, 0.0))
                * overlap["two_sided_geometric_mean"]
            )
            agreements[f"rank{rank}"] = agreement
            subspace_overlaps[f"rank{rank}"] = overlap
            selection_scores[f"rank{rank}"] = {
                "absolute": diagnostics[f"normal_gradient_score_rank{rank}"],
                "over_native_e2": selection_score,
                "normal_energy_fraction": diagnostics[
                    f"normal_gradient_energy_fraction_rank{rank}"
                ],
            }
            composite_scores[f"rank{rank}"] = composite
            subspace_scores[f"rank{rank}"] = subspace_score
        row = {
            "bond": bond,
            "fit_rank": shortlisted_bonds.index(bond) + 1,
            "selection": diagnostics,
            "selection_scores": selection_scores,
            "fit_selection_direction_cosine": agreements,
            "fit_selection_schmidt_subspace_overlap": subspace_overlaps,
            "stable_composite_scores": composite_scores,
            "stable_subspace_scores": subspace_scores,
            "wall_seconds": time.perf_counter() - bond_started,
        }
        stage_two.append(row)
        if not args.quiet:
            print(
                f"selection bond={bond:02d} "
                f"rank{args.primary_rank}_cos="
                f"{agreements[f'rank{args.primary_rank}']:.3f} "
                f"subspace="
                f"{subspace_overlaps[f'rank{args.primary_rank}']['two_sided_geometric_mean']:.3f} "
                f"stable_score="
                f"{subspace_scores[f'rank{args.primary_rank}']:.6e} "
                f"wall={row['wall_seconds']:.1f}s",
                flush=True,
            )
        del analysis, canonical, selection_factors
        release_cuda()

    stable_order = sorted(
        stage_two,
        key=lambda row: (
            row["stable_subspace_scores"][f"rank{args.primary_rank}"],
            -row["bond"],
        ),
        reverse=True,
    )
    report = {
        "schema": "quintic-tn-all-bond-native-ma-scan-v2",
        "scientific_scope": {
            "purpose": (
                "Rank candidate TN bonds by the native-E2 normal-gradient "
                "Schmidt energy that is stable across independent point sets."
            ),
            "gauge_control": (
                "Every pair is put in mixed-canonical gauge and its merged "
                "supercore is normalized to unit Frobenius norm without "
                "changing the metric."
            ),
            "selection_control": (
                "All bonds are ranked on fit points. Only the frozen fit "
                "shortlist is rescored on independent validation points."
            ),
            "stability_score": (
                "The primary ranking uses the geometric mean of fit/selection "
                "rank scores multiplied by the geometric mean of the left and "
                "right Schmidt-subspace overlaps. Signed full-gradient cosine "
                "is retained as a stricter diagnostic."
            ),
            "claim_limit": (
                "This is a linearized bond-priority calibration, not evidence "
                "that any finite nonlinear rank-growth update will pass gates."
            ),
        },
        "configuration": {
            "model": str(model_path),
            "source_run_dir": str(source_dir),
            "pullbacks_dir": str(pullbacks_dir),
            "output_report": str(output_report),
            "fit_train_start": args.fit_train_start,
            "fit_size": args.fit_size,
            "selection_validation_start": args.selection_validation_start,
            "selection_size": args.selection_size,
            "preservation_size": args.preservation_size,
            "group_samples": args.group_samples,
            "operator_chunk_size": args.operator_chunk_size,
            "eval_batch_size": args.eval_batch_size,
            "shortlist_size": args.shortlist_size,
            "score_ranks": list(args.score_ranks),
            "primary_rank": args.primary_rank,
            "bonds": bonds,
            "search_precision": args.search_precision,
            "seed": args.seed,
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
        "model": {
            "site_count": int(source_model.site_count),
            "bond_dimension": source_bond_dimension,
            "physical_dictionary_rank": int(source_model.physical_dictionary_rank),
        },
        "baseline": baseline,
        "fit_scan": stage_one,
        "fit_ranking": [row["bond"] for row in fit_order],
        "selection_shortlist": shortlisted_bonds,
        "selection_rescan": stage_two,
        "stable_ranking": [row["bond"] for row in stable_order],
        "recommended_bond": stable_order[0]["bond"] if stable_order else None,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(
        status_path,
        {
            "state": "complete",
            "phase": "complete",
            "recommended_bond": report["recommended_bond"],
            "wall_seconds": report["wall_seconds"],
        },
    )
    print(
        json.dumps(
            {
                "output_report": str(output_report),
                "fit_ranking": report["fit_ranking"],
                "selection_shortlist": shortlisted_bonds,
                "stable_ranking": report["stable_ranking"],
                "recommended_bond": report["recommended_bond"],
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
