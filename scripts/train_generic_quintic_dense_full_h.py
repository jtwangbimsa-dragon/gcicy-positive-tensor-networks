#!/usr/bin/env python3
"""Train a complete quotient-basis dense full-H on the generic quintic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.fermat_quintic import fubini_study_h  # noqa: E402
from gcicy_metric.generic_quintic import (  # noqa: E402
    GENERIC_QUINTIC_COEFFICIENTS,
    GENERIC_QUINTIC_EXPONENTS,
    section_count,
)
from gcicy_metric.pipeline.hypersurface_section_ring import (  # noqa: E402
    HypersurfaceSectionRing,
)
from scripts.refine_generic_quintic_compiled_tree_native_gn import (  # noqa: E402
    load_disjoint_splits,
    load_excluded_indices,
)
from scripts.train_generic_quintic_low_degree_teacher import (  # noqa: E402
    evaluate,
    make_dataset,
    native_components,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-h", type=Path)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--train-points", type=Path, required=True)
    parser.add_argument("--train-pullbacks", type=Path, required=True)
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
    )
    parser.add_argument("--maximum-steps", type=int, default=4_000)
    parser.add_argument("--minimum-steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--training-batch-size", type=int, default=2_048)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience-evaluations", type=int, default=10)
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument("--sigma-weight", type=float, default=0.0)
    parser.add_argument("--condition-weight", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--selection-size", type=int, default=20_000)
    parser.add_argument("--confirmation-size", type=int, default=50_000)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--metric-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=202607478)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    return parser.parse_args()


def generic_section_ring() -> HypersurfaceSectionRing:
    return HypersurfaceSectionRing(
        GENERIC_QUINTIC_EXPONENTS,
        GENERIC_QUINTIC_COEFFICIENTS,
        pivot_coordinate=0,
    )


def normalize_hermitian(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.complex128)
    result = 0.5 * (result + result.conj().T)
    eigenvalues = np.linalg.eigvalsh(result)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
        raise ValueError("the initial Hermitian matrix must be positive")
    return result * (len(result) / np.trace(result).real)


def quotient_fubini_study_h(
    ring: HypersurfaceSectionRing,
    degree: int,
) -> np.ndarray:
    ambient = ring.ambient_exponents(degree)
    reduction = ring.reduction_matrix(degree).toarray()
    matrix = reduction.conj().T @ fubini_study_h(ambient) @ reduction
    return normalize_hermitian(matrix)


def product_lift_h(
    ring: HypersurfaceSectionRing,
    left_degree: int,
    right_degree: int,
    left_h: np.ndarray,
    right_h: np.ndarray,
) -> np.ndarray:
    """Represent the exact product of two positive algebraic functions."""

    multiplication = ring.multiplication_map(left_degree, right_degree)
    left = normalize_hermitian(left_h)
    right = normalize_hermitian(right_h)
    if left.shape != (multiplication.left_count,) * 2:
        raise ValueError("left H dimension does not match its quotient basis")
    if right.shape != (multiplication.right_count,) * 2:
        raise ValueError("right H dimension does not match its quotient basis")
    product_map = multiplication.matrix.T.toarray()
    coefficients = product_map.reshape(
        multiplication.left_count,
        multiplication.right_count,
        multiplication.output_count,
    )
    transformed = np.einsum(
        "ac,cdq->adq",
        left,
        coefficients,
        optimize=True,
    )
    transformed = np.einsum(
        "bd,adq->abq",
        right,
        transformed,
        optimize=True,
    )
    matrix = product_map.conj().T @ transformed.reshape(
        multiplication.left_count * multiplication.right_count,
        multiplication.output_count,
    )
    return normalize_hermitian(matrix)


def power_lift_h(
    ring: HypersurfaceSectionRing,
    source_degree: int,
    source_h: np.ndarray,
    target_degree: int,
) -> np.ndarray:
    if target_degree % source_degree:
        raise ValueError("target degree must be a multiple of source degree")
    power = target_degree // source_degree
    if power < 1:
        raise ValueError("target degree cannot be below the source degree")
    result = normalize_hermitian(source_h)
    result_degree = source_degree
    for _ in range(1, power):
        result = product_lift_h(
            ring,
            result_degree,
            source_degree,
            result,
            source_h,
        )
        result_degree += source_degree
    return result


def load_initial_h(
    path: Path | None,
    *,
    ring: HypersurfaceSectionRing,
    degree: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if path is None:
        return quotient_fubini_study_h(ring, degree), {
            "kind": "quotient_fubini_study",
            "source_degree": degree,
            "power": 1,
        }
    source_path = path.expanduser().resolve()
    artifact = np.load(source_path, allow_pickle=False)
    source_degree = int(artifact["degree"])
    source_h = np.asarray(artifact["global_h_matrix"], dtype=np.complex128)
    source_exponents = np.asarray(artifact["exponents"], dtype=np.int64)
    if not np.array_equal(
        source_exponents,
        ring.quotient_exponents(source_degree),
    ):
        raise ValueError("initial H uses a different quotient basis")
    lifted = power_lift_h(
        ring,
        source_degree,
        source_h,
        degree,
    )
    return lifted, {
        "kind": "exact_power_lift",
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_degree": source_degree,
        "power": degree // source_degree,
    }


class CompactPositiveFullH(torch.nn.Module):
    """Dense positive Hermitian matrix with exactly N^2 real coordinates."""

    def __init__(
        self,
        initial_h: np.ndarray,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        initial = normalize_hermitian(initial_h)
        factor = np.linalg.cholesky(initial)
        count = len(factor)
        indices = np.tril_indices(count, k=-1)
        real_dtype = (
            torch.float32 if dtype == torch.complex64 else torch.float64
        )
        self.diagonal_log = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    np.diag(factor).real,
                    dtype=real_dtype,
                    device=device,
                )
            )
        )
        strict = factor[indices]
        self.strict_real = torch.nn.Parameter(
            torch.tensor(strict.real, dtype=real_dtype, device=device)
        )
        self.strict_imag = torch.nn.Parameter(
            torch.tensor(strict.imag, dtype=real_dtype, device=device)
        )
        self.register_buffer(
            "row_indices",
            torch.tensor(indices[0], dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "column_indices",
            torch.tensor(indices[1], dtype=torch.int64, device=device),
        )
        self.count = count
        self.complex_dtype = dtype

    def factor(self) -> torch.Tensor:
        strict = torch.complex(self.strict_real, self.strict_imag).to(
            self.complex_dtype
        )
        result = torch.zeros(
            (self.count, self.count),
            dtype=self.complex_dtype,
            device=strict.device,
        )
        result = result.index_put(
            (self.row_indices, self.column_indices),
            strict,
        )
        return result + torch.diag(torch.exp(self.diagonal_log)).to(
            self.complex_dtype
        )

    def forward(self) -> torch.Tensor:
        factor = self.factor()
        matrix = factor @ torch.conj(factor.T)
        return matrix * (self.count / torch.real(torch.trace(matrix)))


def active_real_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
    )


def state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def relative_gain(start: float, final: float) -> float:
    return (start - final) / start


def main() -> None:
    args = parse_args()
    positive = (
        args.degree,
        args.maximum_steps,
        args.minimum_steps,
        args.learning_rate,
        args.minimum_learning_rate,
        args.training_batch_size,
        args.eval_every,
        args.patience_evaluations,
        args.gradient_clip_norm,
        args.train_size,
        args.selection_size,
        args.confirmation_size,
        args.feature_batch_size,
        args.metric_chunk_size,
        args.threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("dense full-H sizes must be positive")
    if args.minimum_steps > args.maximum_steps:
        raise ValueError("minimum steps cannot exceed maximum steps")
    if not 0 <= args.minimum_relative_improvement < 1:
        raise ValueError("minimum relative improvement must lie in [0, 1)")
    if args.sigma_weight < 0 or args.condition_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    device = torch.device(args.device)
    dtype = (
        torch.complex64
        if args.precision == "complex64"
        else torch.complex128
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a dense full-H run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})
    started = time.perf_counter()

    try:
        ring = generic_section_ring()
        exponents = ring.quotient_exponents(args.degree)
        if len(exponents) != section_count(args.degree):
            raise RuntimeError("quotient section count is inconsistent")
        initial_h, initialization = load_initial_h(
            args.initial_h,
            ring=ring,
            degree=args.degree,
        )
        normalization = 1.0 / (math.pi * args.degree)
        model = CompactPositiveFullH(
            initial_h,
            dtype=dtype,
            device=device,
        )
        parameter_count = active_real_parameter_count(model)
        expected_parameters = len(exponents) ** 2
        if parameter_count != expected_parameters:
            raise RuntimeError("compact Cholesky parameter count is incorrect")

        specifications = {
            "fit": (
                args.train_points,
                args.train_pullbacks,
                args.train_size,
            ),
            "selection": (
                args.selection_points,
                args.selection_pullbacks,
                args.selection_size,
            ),
        }
        exclusions = load_excluded_indices(args.exclude_indices_file)
        arrays, indices = load_disjoint_splits(
            specifications,
            seed=args.seed,
            exclusions=exclusions,
        )
        np.savez_compressed(output_dir / "data_indices.npz", **indices)
        write_json(status_path, {"state": "running", "phase": "features"})
        training = make_dataset(
            *arrays["fit"],
            exponents,
            feature_batch_size=args.feature_batch_size,
            dtype=dtype,
            device=device,
        )
        selection = make_dataset(
            *arrays["selection"],
            exponents,
            feature_batch_size=args.feature_batch_size,
            dtype=dtype,
            device=device,
        )

        initial_selection = evaluate(
            initial_h,
            selection,
            normalization=normalization,
            chunk_size=args.metric_chunk_size,
            dtype=dtype,
            device=device,
        )
        initial_statistics = initial_selection["statistics"]
        best_score = float(
            initial_statistics["weighted_rms_abs_residual"]
        )
        best_state = state_to_cpu(model)
        best_step = 0
        material_reference = best_score
        stale_evaluations = 0
        history: list[dict[str, Any]] = [
            {
                "step": 0,
                "learning_rate": args.learning_rate,
                "sigma": initial_statistics["sigma_official_formula"],
                "chi": best_score,
                **initial_selection["tail"],
            }
        ]
        write_json(output_dir / "history.json", {"rows": history})
        print(
            f"degree={args.degree} sections={len(exponents)} "
            f"real_parameters={parameter_count} "
            f"initial_sigma={history[0]['sigma']:.8e} "
            f"initial_chi={history[0]['chi']:.8e}",
            flush=True,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.maximum_steps,
            eta_min=args.minimum_learning_rate,
        )
        write_json(status_path, {"state": "running", "phase": "training"})
        stopped_early = False
        completed_steps = 0
        for step in range(1, args.maximum_steps + 1):
            batch_indices = rng.choice(
                training["count"],
                size=min(args.training_batch_size, training["count"]),
                replace=False,
            )
            selected = torch.as_tensor(
                batch_indices,
                dtype=torch.int64,
                device=device,
            )
            active = {
                "count": len(batch_indices),
                "values": training["values"].index_select(0, selected),
                "derivatives": training["derivatives"].index_select(
                    0,
                    selected,
                ),
                "weights": training["weights"].index_select(0, selected),
                "weights_numpy": training["weights_numpy"][batch_indices],
                "log_omega": training["log_omega"].index_select(
                    0,
                    selected,
                ),
            }
            optimizer.zero_grad(set_to_none=True)
            matrix = model()
            components = native_components(
                matrix,
                active,
                normalization=normalization,
                chunk_size=args.metric_chunk_size,
            )
            loss = components["e2"] + args.sigma_weight * components["sigma"]
            condition_penalty = torch.zeros((), device=device)
            if args.condition_weight:
                eigenvalues = torch.linalg.eigvalsh(matrix)
                condition_penalty = torch.square(
                    torch.log(eigenvalues[-1] / eigenvalues[0])
                )
                loss = loss + args.condition_weight * condition_penalty
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()
            completed_steps = step
            if step % args.eval_every and step != args.maximum_steps:
                continue

            candidate_h = model().detach().cpu().numpy()
            candidate = evaluate(
                candidate_h,
                selection,
                normalization=normalization,
                chunk_size=args.metric_chunk_size,
                dtype=dtype,
                device=device,
            )
            statistics = candidate["statistics"]
            score = float(statistics["weighted_rms_abs_residual"])
            row = {
                "step": step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_e2": float(components["e2"].detach().cpu()),
                "train_sigma": float(components["sigma"].detach().cpu()),
                "condition_penalty": float(condition_penalty.detach().cpu()),
                "gradient_norm": float(gradient_norm),
                "sigma": statistics["sigma_official_formula"],
                "chi": score,
                **candidate["tail"],
            }
            history.append(row)
            if score < best_score:
                best_score = score
                best_step = step
                best_state = state_to_cpu(model)
            if score <= material_reference * (
                1.0 - args.minimum_relative_improvement
            ):
                material_reference = score
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            write_json(output_dir / "history.json", {"rows": history})
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "training",
                    "step": step,
                    "best_step": best_step,
                    "best_chi": best_score,
                    "stale_evaluations": stale_evaluations,
                },
            )
            print(
                f"step={step} val_sigma={row['sigma']:.8e} "
                f"val_chi={row['chi']:.8e} best={best_score:.8e}",
                flush=True,
            )
            if (
                step >= args.minimum_steps
                and stale_evaluations >= args.patience_evaluations
            ):
                stopped_early = True
                break

        model.load_state_dict(best_state)
        best_h = model().detach().cpu().numpy().astype(np.complex128)
        best_selection = evaluate(
            best_h,
            selection,
            normalization=normalization,
            chunk_size=args.metric_chunk_size,
            dtype=dtype,
            device=device,
        )
        del selection
        del training
        if device.type == "cuda":
            torch.cuda.empty_cache()

        write_json(status_path, {"state": "running", "phase": "confirmation"})
        confirmation_arrays, confirmation_indices = load_disjoint_splits(
            {
                "confirmation": (
                    args.confirmation_points,
                    args.confirmation_pullbacks,
                    args.confirmation_size,
                )
            },
            seed=args.seed + 10_000,
            exclusions=exclusions,
        )
        with np.load(output_dir / "data_indices.npz") as existing:
            index_payload = {
                key: np.asarray(existing[key], dtype=np.int64)
                for key in existing.files
            }
        index_payload["confirmation"] = confirmation_indices["confirmation"]
        np.savez_compressed(output_dir / "data_indices.npz", **index_payload)
        confirmation = make_dataset(
            *confirmation_arrays["confirmation"],
            exponents,
            feature_batch_size=args.feature_batch_size,
            dtype=dtype,
            device=device,
        )
        confirmation_result = evaluate(
            best_h,
            confirmation,
            normalization=normalization,
            chunk_size=args.metric_chunk_size,
            dtype=dtype,
            device=device,
        )

        artifact_path = output_dir / "full_h.npz"
        np.savez_compressed(
            artifact_path,
            global_h_matrix=best_h,
            degree=np.asarray(args.degree, dtype=np.int64),
            exponents=exponents,
            normalization=np.asarray(normalization, dtype=np.float64),
        )
        checkpoint_path = output_dir / "checkpoint.pt"
        torch.save(
            {
                "schema": "generic-quintic-dense-full-h-v1",
                "state_dict": best_state,
                "configuration": {
                    **vars(args),
                    "output_dir": str(output_dir),
                    "exponents": exponents,
                    "normalization": normalization,
                },
                "initialization": initialization,
                "best_step": best_step,
            },
            checkpoint_path,
        )
        report = {
            "schema": "generic-quintic-dense-full-h-v1",
            "configuration": {
                **vars(args),
                "output_dir": str(output_dir),
            },
            "model": {
                "degree": args.degree,
                "section_count": len(exponents),
                "trainable_real_parameter_count": parameter_count,
                "initialization": initialization,
                "best_step": best_step,
                "completed_steps": completed_steps,
                "stopped_early": stopped_early,
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            },
            "counts": {
                "training": args.train_size,
                "selection": args.selection_size,
                "confirmation": args.confirmation_size,
            },
            "initial_selection": initial_selection,
            "best_selection": best_selection,
            "confirmation": confirmation_result,
            "relative_selection_gain": {
                "sigma": relative_gain(
                    initial_statistics["sigma_official_formula"],
                    best_selection["statistics"]["sigma_official_formula"],
                ),
                "chi": relative_gain(
                    initial_statistics["weighted_rms_abs_residual"],
                    best_selection["statistics"][
                        "weighted_rms_abs_residual"
                    ],
                ),
            },
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "best_step": best_step,
                "confirmation_sigma": confirmation_result["statistics"][
                    "sigma_official_formula"
                ],
                "confirmation_chi": confirmation_result["statistics"][
                    "weighted_rms_abs_residual"
                ],
                "wall_seconds": report["wall_seconds"],
            },
        )
        print(
            json.dumps(
                {
                    "state": "complete",
                    "best_step": best_step,
                    "completed_steps": completed_steps,
                    "selection_sigma": best_selection["statistics"][
                        "sigma_official_formula"
                    ],
                    "confirmation_sigma": confirmation_result["statistics"][
                        "sigma_official_formula"
                    ],
                    "confirmation_chi": confirmation_result["statistics"][
                        "weighted_rms_abs_residual"
                    ],
                    "wall_seconds": report["wall_seconds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
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
