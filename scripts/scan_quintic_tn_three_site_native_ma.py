#!/usr/bin/env python3
"""Rank three-site TN blocks by cross-sample-stable native-MA directions."""

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
    orbit_augment_dataset,
    three_site_normal_gradient_analysis,
)
from scripts.scan_quintic_tn_all_bonds_native_ma import (  # noqa: E402
    principal_cosine_squares,
    principal_subspace_overlap,
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
    parser.add_argument("--fit-train-start", type=int, default=430_000)
    parser.add_argument("--fit-size", type=int, default=64)
    parser.add_argument("--selection-validation-start", type=int, default=80_000)
    parser.add_argument("--selection-size", type=int, default=64)
    parser.add_argument("--preservation-size", type=int, default=8)
    parser.add_argument("--group-samples", type=int, default=2)
    parser.add_argument("--operator-chunk-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--shortlist-size", type=int, default=4)
    parser.add_argument("--score-ranks", type=int, nargs="+", default=(2, 4))
    parser.add_argument("--primary-rank", type=int, default=4)
    parser.add_argument("--triples", type=int, nargs="*", default=None)
    parser.add_argument(
        "--search-precision",
        choices=("complex64", "complex128"),
        default="complex128",
    )
    parser.add_argument("--seed", type=int, default=202607281)
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
    if args.primary_rank not in args.score_ranks:
        raise ValueError("primary rank must be one of the score ranks")
    if args.triples is not None and len(set(args.triples)) != len(args.triples):
        raise ValueError("three-site block list contains duplicates")


def normalize_three_site_supercore_scale_(
    model: torch.nn.Module,
    start: int,
    *,
    target_norm: float = 1.0,
) -> dict[str, float]:
    """Fix the metric-invariant scale using one merged three-site core."""

    if target_norm <= 0 or not np.isfinite(target_norm):
        raise ValueError("target triple norm must be finite and positive")
    if start < 0 or start + 2 >= model.site_count:
        raise ValueError("three-site block lies outside the chain")
    left = model.coefficient_cores[start]
    middle = model.coefficient_cores[start + 1]
    right = model.coefficient_cores[start + 2]
    merged = torch.einsum("lmx,mny,nrz->lrxyz", left, middle, right)
    observed = torch.linalg.vector_norm(merged)
    if not bool(torch.isfinite(observed)) or float(observed) <= 0:
        raise FloatingPointError("merged three-site core has no finite norm")
    scale = target_norm / float(observed)
    with torch.no_grad():
        left.mul_(scale)
        model.positive_floor *= scale**2
    normalized = torch.linalg.vector_norm(
        torch.einsum("lmx,mny,nrz->lrxyz", left, middle, right)
    )
    return {
        "observed_triple_norm": float(observed),
        "scale": float(scale),
        "normalized_triple_norm": float(normalized),
    }


def compact_mode_bases(
    analysis: dict[str, Any],
    maximum_rank: int,
) -> dict[str, torch.Tensor]:
    return {
        name: analysis[name][:, :maximum_rank].detach().cpu()
        for name in ("left_basis", "middle_basis", "right_basis")
    }


def three_mode_subspace_overlap(
    fit: dict[str, torch.Tensor],
    selection: dict[str, torch.Tensor],
    rank: int,
) -> dict[str, Any]:
    overlaps = {
        mode: principal_subspace_overlap(
            fit[f"{mode}_basis"],
            selection[f"{mode}_basis"],
            rank,
        )
        for mode in ("left", "middle", "right")
    }
    overlaps["three_sided_geometric_mean"] = (
        overlaps["left"] * overlaps["middle"] * overlaps["right"]
    ) ** (1.0 / 3.0)
    overlaps["principal_cosine_squares"] = {
        mode: principal_cosine_squares(
            fit[f"{mode}_basis"],
            selection[f"{mode}_basis"],
            rank,
        )
        for mode in ("left", "middle", "right")
    }
    return overlaps


def normalized_tucker_score(
    diagnostics: dict[str, Any],
    rank: int,
) -> float:
    return float(
        diagnostics["tucker_scores"][f"rank{rank}"]
        / max(diagnostics["fit_native_e2"], np.finfo(float).tiny)
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

    all_triples = list(range(source_model.site_count - 2))
    triples = all_triples if args.triples is None else list(args.triples)
    if any(start not in all_triples for start in triples):
        raise ValueError("requested three-site block lies outside the chain")
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

    def make_split(*, domain: str, indices: np.ndarray) -> dict[str, Any]:
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
    preservation_count = min(args.preservation_size, selection["count"])
    preservation = {
        **selection,
        "count": preservation_count,
        "values": selection["values"][:preservation_count],
        "derivatives": selection["derivatives"][:preservation_count],
        "weights": selection["weights"][:preservation_count],
        "weights_numpy": selection["weights_numpy"][:preservation_count],
        "log_omega": selection["log_omega"][:preservation_count],
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

    preservation_tolerance = (
        5.0e-5 if complex_dtype == torch.complex64 else 3.0e-10
    )
    stage_one: list[dict[str, Any]] = []
    fit_bases: dict[int, dict[str, torch.Tensor]] = {}
    for scan_index, start in enumerate(triples, start=1):
        block_started = time.perf_counter()
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "fit_scan",
                "three_site_start": start,
                "iteration": scan_index,
                "count": len(triples),
            },
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        canonical = copy.deepcopy(source_model)
        canonical.mixed_canonicalize_coefficient_block_(start, 3)
        scale = normalize_three_site_supercore_scale_(canonical, start)
        relative_change = maximum_relative_metric_difference(
            source_model,
            canonical,
            preservation,
            batch_size=args.eval_batch_size,
        )
        if relative_change > preservation_tolerance:
            raise RuntimeError(
                f"triple {start} gauge normalization changed the metric by "
                f"{relative_change:.3e}"
            )
        analysis = three_site_normal_gradient_analysis(
            canonical,
            fit_objective,
            start=start,
            source_bond_dimension=source_bond_dimension,
            chunk_size=args.operator_chunk_size,
            score_ranks=tuple(args.score_ranks),
        )
        diagnostics = analysis["diagnostics"]
        fit_bases[start] = compact_mode_bases(analysis, maximum_rank)
        scores = {
            f"rank{rank}": {
                "absolute": diagnostics["tucker_scores"][f"rank{rank}"],
                "over_native_e2": normalized_tucker_score(diagnostics, rank),
                "double_normal_energy_fraction": diagnostics[
                    "tucker_normal_energy_fractions"
                ][f"rank{rank}"],
            }
            for rank in args.score_ranks
        }
        row = {
            "three_site_start": start,
            "triple_scale": scale,
            "gauge_metric_relative_change": relative_change,
            "fit": diagnostics,
            "scores": scores,
            "wall_seconds": time.perf_counter() - block_started,
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        }
        stage_one.append(row)
        if not args.quiet:
            print(
                f"fit triple={start:02d} "
                f"rank{args.primary_rank}_score="
                f"{scores[f'rank{args.primary_rank}']['over_native_e2']:.6e} "
                f"double_normal_fraction="
                f"{diagnostics['double_normal_fraction_of_full']:.3f} "
                f"wall={row['wall_seconds']:.1f}s",
                flush=True,
            )
        del analysis, canonical
        release_cuda()

    fit_order = sorted(
        stage_one,
        key=lambda row: (
            row["scores"][f"rank{args.primary_rank}"]["over_native_e2"],
            -row["three_site_start"],
        ),
        reverse=True,
    )
    shortlisted = [
        row["three_site_start"]
        for row in fit_order[: min(args.shortlist_size, len(fit_order))]
    ]

    stage_two: list[dict[str, Any]] = []
    for selection_index, start in enumerate(shortlisted, start=1):
        block_started = time.perf_counter()
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "selection_rescan",
                "three_site_start": start,
                "iteration": selection_index,
                "count": len(shortlisted),
            },
        )
        canonical = copy.deepcopy(source_model)
        canonical.mixed_canonicalize_coefficient_block_(start, 3)
        normalize_three_site_supercore_scale_(canonical, start)
        analysis = three_site_normal_gradient_analysis(
            canonical,
            selection_objective,
            start=start,
            source_bond_dimension=source_bond_dimension,
            chunk_size=args.operator_chunk_size,
            score_ranks=tuple(args.score_ranks),
        )
        diagnostics = analysis["diagnostics"]
        selection_bases = compact_mode_bases(analysis, maximum_rank)
        fit_row = next(
            row for row in stage_one if row["three_site_start"] == start
        )
        overlaps: dict[str, Any] = {}
        selection_scores: dict[str, Any] = {}
        stable_scores: dict[str, float] = {}
        for rank in args.score_ranks:
            overlap = three_mode_subspace_overlap(
                fit_bases[start],
                selection_bases,
                rank,
            )
            fit_score = fit_row["scores"][f"rank{rank}"]["over_native_e2"]
            selection_score = normalized_tucker_score(diagnostics, rank)
            stable = (
                math.sqrt(max(fit_score, 0.0) * max(selection_score, 0.0))
                * overlap["three_sided_geometric_mean"]
            )
            overlaps[f"rank{rank}"] = overlap
            selection_scores[f"rank{rank}"] = {
                "absolute": diagnostics["tucker_scores"][f"rank{rank}"],
                "over_native_e2": selection_score,
                "double_normal_energy_fraction": diagnostics[
                    "tucker_normal_energy_fractions"
                ][f"rank{rank}"],
            }
            stable_scores[f"rank{rank}"] = stable
        row = {
            "three_site_start": start,
            "fit_rank": shortlisted.index(start) + 1,
            "selection": diagnostics,
            "selection_scores": selection_scores,
            "fit_selection_mode_subspace_overlap": overlaps,
            "stable_subspace_scores": stable_scores,
            "wall_seconds": time.perf_counter() - block_started,
        }
        stage_two.append(row)
        if not args.quiet:
            primary_overlap = overlaps[f"rank{args.primary_rank}"]
            print(
                f"selection triple={start:02d} "
                f"left={primary_overlap['left']:.3f} "
                f"middle={primary_overlap['middle']:.3f} "
                f"right={primary_overlap['right']:.3f} "
                f"stable_score="
                f"{stable_scores[f'rank{args.primary_rank}']:.6e} "
                f"wall={row['wall_seconds']:.1f}s",
                flush=True,
            )
        del analysis, canonical, selection_bases
        release_cuda()

    stable_order = sorted(
        stage_two,
        key=lambda row: (
            row["stable_subspace_scores"][f"rank{args.primary_rank}"],
            -row["three_site_start"],
        ),
        reverse=True,
    )
    report = {
        "schema": "quintic-tn-three-site-native-ma-scan-v1",
        "scientific_scope": {
            "purpose": (
                "Detect residual directions that require simultaneous new "
                "Schmidt support across both internal cuts of a three-site block."
            ),
            "projection": (
                "The native-E2 gradient of an unrestricted three-site core is "
                "projected outside the inherited left and right Schmidt supports."
            ),
            "compression": (
                "Only the three Tucker mode spaces are retained. A later optimizer "
                "may fit at most a 4x4x4 complex coefficient tensor; the full "
                "three-site supercore is never proposed as a trainable production model."
            ),
            "claim_limit": (
                "This scanner establishes cross-sample support stability only, "
                "not a finite nonlinear improvement."
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
            "triples": triples,
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
            "physical_dictionary_rank": int(
                source_model.physical_dictionary_rank
            ),
        },
        "baseline": baseline,
        "fit_scan": stage_one,
        "fit_order": [row["three_site_start"] for row in fit_order],
        "selection_shortlist": shortlisted,
        "selection_rescan": stage_two,
        "stable_order": [row["three_site_start"] for row in stable_order],
        "recommended_three_site_start": (
            None if not stable_order else stable_order[0]["three_site_start"]
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(
        status_path,
        {
            "state": "complete",
            "phase": "complete",
            "recommended_three_site_start": report[
                "recommended_three_site_start"
            ],
            "wall_seconds": report["wall_seconds"],
        },
    )
    print(
        json.dumps(
            {
                "output_report": str(output_report),
                "fit_order": report["fit_order"],
                "stable_order": report["stable_order"],
                "recommended_three_site_start": report[
                    "recommended_three_site_start"
                ],
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
