#!/usr/bin/env python3
"""Rotate newly added sibling subspaces with teacher-free native-E2 GN/LM.

The original teacher-compiled rows and parent block remain exactly fixed.
Only the newly added child rows are rotated in their orthogonal complements,
together with the parent couplings that touch at least one new row.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
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
from gcicy_metric.pipeline.positive_multiplication_tree import (  # noqa: E402
    PositiveMultiplicationTreeMetric,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    lanczos_tridiagonal,
    real_inner,
    ridge_coefficients_from_lanczos,
    vector_norm,
)
from scripts.evaluate_generic_quintic_h4_architecture_arms import (  # noqa: E402
    build_checkpoint_model,
    infer_architecture,
)
from scripts.refine_generic_quintic_compiled_tree_native_gn import (  # noqa: E402
    load_disjoint_splits,
    load_excluded_indices,
)
from scripts.train_generic_quintic_compiled_tree import (  # noqa: E402
    metric_row,
    sha256_file,
    whiten_dataset,
)
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    make_dataset,
    tail_guard,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--rank-proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--fit-points", type=Path, required=True)
    parser.add_argument("--fit-pullbacks", type=Path, required=True)
    parser.add_argument("--selection-points", type=Path, required=True)
    parser.add_argument("--selection-pullbacks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-indices-file",
        type=Path,
        nargs="*",
        default=(),
    )
    parser.add_argument("--fit-size", type=int, default=512)
    parser.add_argument("--selection-size", type=int, default=1024)
    parser.add_argument(
        "--rotation-directions",
        type=int,
        default=8,
        help="orthogonal-complement tangent directions per sibling child",
    )
    parser.add_argument("--lanczos-steps", type=int, default=8)
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
        default=(0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5),
    )
    parser.add_argument("--operator-chunk-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--maximum-relative-coordinate-step",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument("--seed", type=int, default=202607442)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


@dataclass(frozen=True)
class SiblingRotationParameterization:
    """Gauge-fixed coordinates for new child rows and parent couplings."""

    child_names: tuple[str, str]
    parent_name: str
    child_baselines: tuple[torch.Tensor, torch.Tensor]
    child_old_counts: tuple[int, int]
    child_target_norms: tuple[torch.Tensor, torch.Tensor]
    parent_baseline: torch.Tensor
    parent_indices: torch.Tensor
    counts: tuple[int, int, int]
    theta: torch.Tensor

    @classmethod
    def from_models(
        cls,
        baseline: PositiveMultiplicationTreeMetric,
        proposal: PositiveMultiplicationTreeMetric,
        *,
        parent: int,
        sibling_edges: tuple[int, int],
    ) -> "SiblingRotationParameterization":
        child_names = tuple(
            f"internal_tensors.{edge - proposal.leaf_count}"
            for edge in sibling_edges
        )
        parent_name = f"internal_tensors.{parent - proposal.leaf_count}"
        proposal_parameters = dict(proposal.named_parameters())
        child_baselines = tuple(
            proposal_parameters[name].detach().clone()
            for name in child_names
        )
        child_old_counts = tuple(
            int(baseline.edge_dimensions[edge]) for edge in sibling_edges
        )
        child_rows = tuple(
            tensor[old_count:].reshape(tensor.shape[0] - old_count, -1)
            for tensor, old_count in zip(
                child_baselines,
                child_old_counts,
                strict=True,
            )
        )
        if any(len(rows) == 0 for rows in child_rows):
            raise ValueError("rank proposal contains no new sibling rows")
        child_target_norms = tuple(
            torch.linalg.vector_norm(rows, dim=1).detach()
            for rows in child_rows
        )
        parent_baseline = proposal_parameters[parent_name].detach().clone()
        left_old, right_old = child_old_counts
        parent_mask = torch.ones(
            parent_baseline.shape,
            dtype=torch.bool,
            device=parent_baseline.device,
        )
        parent_mask[:, :left_old, :right_old] = False
        parent_indices = torch.nonzero(
            parent_mask.reshape(-1),
            as_tuple=False,
        ).reshape(-1)
        rows = [
            child_rows[0].reshape(-1),
            child_rows[1].reshape(-1),
            parent_baseline.reshape(-1).index_select(0, parent_indices),
        ]
        counts = tuple(int(row.numel()) for row in rows)
        return cls(
            child_names=(child_names[0], child_names[1]),
            parent_name=parent_name,
            child_baselines=(
                child_baselines[0],
                child_baselines[1],
            ),
            child_old_counts=(
                child_old_counts[0],
                child_old_counts[1],
            ),
            child_target_norms=(
                child_target_norms[0],
                child_target_norms[1],
            ),
            parent_baseline=parent_baseline,
            parent_indices=parent_indices,
            counts=(counts[0], counts[1], counts[2]),
            theta=torch.cat(rows),
        )

    @staticmethod
    def _orthogonal_rows(
        fixed_rows: torch.Tensor,
        raw_rows: torch.Tensor,
        target_norms: torch.Tensor,
    ) -> torch.Tensor:
        fixed_matrix = fixed_rows.reshape(len(fixed_rows), -1)
        raw_matrix = raw_rows.reshape(len(raw_rows), -1)
        basis, _ = torch.linalg.qr(
            torch.transpose(fixed_matrix, 0, 1),
            mode="reduced",
        )
        rows = []
        for index in range(len(raw_matrix)):
            candidate = raw_matrix[index]
            candidate = candidate - basis @ (
                torch.conj(torch.transpose(basis, 0, 1)) @ candidate
            )
            norm = torch.linalg.vector_norm(candidate)
            epsilon = torch.finfo(torch.real(candidate).dtype).eps
            normalized = (
                target_norms[index]
                * candidate
                / torch.clamp(norm, min=epsilon)
            )
            rows.append(normalized)
            basis = torch.cat(
                (
                    basis,
                    (normalized / target_norms[index])[:, None],
                ),
                dim=1,
            )
        return torch.stack(rows).reshape(raw_rows.shape)

    def unpack(self, vector: torch.Tensor) -> dict[str, torch.Tensor]:
        if vector.ndim != 1 or vector.numel() != sum(self.counts):
            raise ValueError("sibling-rotation coordinate vector has the wrong shape")
        offset = 0
        result: dict[str, torch.Tensor] = {}
        for (
            name,
            baseline,
            old_count,
            target_norms,
            count,
        ) in zip(
            self.child_names,
            self.child_baselines,
            self.child_old_counts,
            self.child_target_norms,
            self.counts[:2],
            strict=True,
        ):
            new_shape = (baseline.shape[0] - old_count,) + baseline.shape[1:]
            raw_rows = vector[offset : offset + count].reshape(new_shape)
            offset += count
            rotated = self._orthogonal_rows(
                baseline[:old_count],
                raw_rows,
                target_norms,
            )
            result[name] = torch.cat(
                (baseline[:old_count], rotated),
                dim=0,
            )
        parent_coordinates = vector[offset:]
        parent_flat = self.parent_baseline.reshape(-1).clone()
        parent_flat = parent_flat.scatter(
            0,
            self.parent_indices,
            parent_coordinates,
        )
        result[self.parent_name] = parent_flat.reshape(
            self.parent_baseline.shape
        )
        return result


@dataclass(frozen=True)
class SketchedSiblingRotationParameterization:
    """Low-dimensional tangent rotations of the newly added child rows."""

    child_names: tuple[str, str]
    parent_name: str
    child_baselines: tuple[torch.Tensor, torch.Tensor]
    child_old_counts: tuple[int, int]
    child_target_norms: tuple[torch.Tensor, torch.Tensor]
    tangent_bases: tuple[torch.Tensor, torch.Tensor]
    parent_baseline: torch.Tensor
    parent_indices: torch.Tensor
    counts: tuple[int, int, int]
    theta: torch.Tensor

    @classmethod
    def from_models(
        cls,
        baseline: PositiveMultiplicationTreeMetric,
        proposal: PositiveMultiplicationTreeMetric,
        *,
        parent: int,
        sibling_edges: tuple[int, int],
        directions: int,
        seed: int,
    ) -> "SketchedSiblingRotationParameterization":
        if directions <= 0:
            raise ValueError("rotation sketch must contain positive directions")
        full = SiblingRotationParameterization.from_models(
            baseline,
            proposal,
            parent=parent,
            sibling_edges=sibling_edges,
        )
        rng = np.random.default_rng(seed)
        tangent_bases = []
        coefficient_counts = []
        coefficient_rows = []
        for child_index, (tensor, old_count) in enumerate(
            zip(
                full.child_baselines,
                full.child_old_counts,
                strict=True,
            )
        ):
            rows = tensor.reshape(tensor.shape[0], -1)
            ambient = rows.shape[1]
            if directions > ambient - tensor.shape[0]:
                raise ValueError("rotation sketch exceeds the orthogonal complement")
            normalized = rows / torch.linalg.vector_norm(rows, dim=1)[:, None]
            random_values = rng.normal(size=(ambient, directions)) + 1j * rng.normal(
                size=(ambient, directions)
            )
            random_tensor = torch.tensor(
                random_values,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            random_tensor = random_tensor - torch.conj(
                torch.transpose(normalized, 0, 1)
            ) @ (normalized @ random_tensor)
            tangent, _ = torch.linalg.qr(random_tensor, mode="reduced")
            tangent_basis = torch.transpose(tangent, 0, 1).detach()
            tangent_bases.append(tangent_basis)
            new_count = tensor.shape[0] - old_count
            coefficient_count = new_count * directions
            coefficient_counts.append(coefficient_count)
            coefficient_rows.append(
                torch.zeros(
                    coefficient_count,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            )
        parent_coordinates = full.parent_baseline.reshape(-1).index_select(
            0,
            full.parent_indices,
        )
        return cls(
            child_names=full.child_names,
            parent_name=full.parent_name,
            child_baselines=full.child_baselines,
            child_old_counts=full.child_old_counts,
            child_target_norms=full.child_target_norms,
            tangent_bases=(tangent_bases[0], tangent_bases[1]),
            parent_baseline=full.parent_baseline,
            parent_indices=full.parent_indices,
            counts=(
                coefficient_counts[0],
                coefficient_counts[1],
                int(parent_coordinates.numel()),
            ),
            theta=torch.cat(
                (
                    coefficient_rows[0],
                    coefficient_rows[1],
                    parent_coordinates,
                )
            ),
        )

    def unpack(self, vector: torch.Tensor) -> dict[str, torch.Tensor]:
        if vector.ndim != 1 or vector.numel() != sum(self.counts):
            raise ValueError("sketched sibling coordinates have the wrong shape")
        result: dict[str, torch.Tensor] = {}
        offset = 0
        for (
            name,
            baseline,
            old_count,
            target_norms,
            tangent_basis,
            count,
        ) in zip(
            self.child_names,
            self.child_baselines,
            self.child_old_counts,
            self.child_target_norms,
            self.tangent_bases,
            self.counts[:2],
            strict=True,
        ):
            new_count = baseline.shape[0] - old_count
            direction_count = tangent_basis.shape[0]
            coefficients = vector[offset : offset + count].reshape(
                new_count,
                direction_count,
            )
            offset += count
            baseline_new = baseline[old_count:].reshape(new_count, -1)
            rotated = baseline_new + coefficients @ tangent_basis
            rotated = (
                target_norms[:, None]
                * rotated
                / torch.clamp(
                    torch.linalg.vector_norm(rotated, dim=1)[:, None],
                    min=torch.finfo(torch.real(rotated).dtype).eps,
                )
            )
            result[name] = torch.cat(
                (
                    baseline[:old_count],
                    rotated.reshape(baseline[old_count:].shape),
                ),
                dim=0,
            )
        parent_flat = self.parent_baseline.reshape(-1).clone()
        parent_flat = parent_flat.scatter(
            0,
            self.parent_indices,
            vector[offset:],
        )
        result[self.parent_name] = parent_flat.reshape(
            self.parent_baseline.shape
        )
        return result


def copy_coordinates_(
    model: PositiveMultiplicationTreeMetric,
    parameterization: Any,
    coordinates: torch.Tensor,
) -> None:
    values = parameterization.unpack(coordinates)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            parameters[name].copy_(value)


def serializable(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }


def main() -> None:
    args = parse_args()
    if (
        min(
            args.fit_size,
            args.selection_size,
            args.rotation_directions,
            args.lanczos_steps,
            args.operator_chunk_size,
            args.feature_batch_size,
            args.eval_batch_size,
            args.threads,
        )
        <= 0
        or not args.ridge_factors
        or any(value <= 0 for value in args.ridge_factors)
        or not args.line_search_alphas
        or any(value <= 0 or value > 1 for value in args.line_search_alphas)
        or args.maximum_relative_coordinate_step <= 0
    ):
        raise ValueError("invalid sibling-rotation configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a sibling-rotation run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    started = time.perf_counter()

    try:
        baseline_path = args.baseline_checkpoint.expanduser().resolve()
        proposal_path = args.rank_proposal_checkpoint.expanduser().resolve()
        baseline_payload = torch.load(
            baseline_path,
            map_location="cpu",
            weights_only=False,
        )
        proposal_payload = torch.load(
            proposal_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            infer_architecture(baseline_payload) != "compiled-tree"
            or infer_architecture(proposal_payload) != "compiled-tree"
        ):
            raise ValueError("rotation requires compiled-tree checkpoints")
        if bool(
            baseline_payload.get("teacher_runtime_dependency", False)
            or proposal_payload.get("teacher_runtime_dependency", False)
        ):
            raise ValueError("rotation checkpoints must be teacher-independent")
        metadata = proposal_payload.get("native_sibling_rank_enrichment")
        if not isinstance(metadata, dict):
            raise ValueError("rank proposal lacks sibling-enrichment metadata")
        parent = int(metadata["parent"])
        sibling_edges = tuple(int(value) for value in metadata["sibling_edges"])
        if len(sibling_edges) != 2:
            raise ValueError("rank proposal does not contain two sibling edges")
        configuration = baseline_payload["configuration"]
        exponents = np.asarray(configuration["exponents"], dtype=np.int64)
        whitening = np.asarray(configuration["whitening"], dtype=np.complex128)

        baseline_model = build_checkpoint_model(
            baseline_payload,
            device=device,
        ).to(dtype=torch.complex64)
        proposal_model = build_checkpoint_model(
            proposal_payload,
            device=device,
        ).to(dtype=torch.complex64)
        if not isinstance(
            baseline_model,
            PositiveMultiplicationTreeMetric,
        ) or not isinstance(proposal_model, PositiveMultiplicationTreeMetric):
            raise TypeError("checkpoint did not reconstruct a multiplication tree")
        parameterization = SketchedSiblingRotationParameterization.from_models(
            baseline_model,
            proposal_model,
            parent=parent,
            sibling_edges=(sibling_edges[0], sibling_edges[1]),
            directions=args.rotation_directions,
            seed=args.seed + 100,
        )
        theta = parameterization.theta.detach()

        excluded = load_excluded_indices(args.exclude_indices_file)
        arrays, indices = load_disjoint_splits(
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
            },
            seed=args.seed,
            exclusions=excluded,
        )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **indices)
        datasets = {
            name: whiten_dataset(
                make_dataset(
                    split,
                    exponents,
                    feature_batch_size=args.feature_batch_size,
                    complex_dtype=torch.complex64,
                    device=device,
                ),
                whitening,
            )
            for name, split in arrays.items()
        }
        baseline_selection, _, baseline_tail = metric_row(
            baseline_model,
            datasets["selection"],
            chunk_size=args.eval_batch_size,
        )
        proposal_selection, _, proposal_tail = metric_row(
            proposal_model,
            datasets["selection"],
            chunk_size=args.eval_batch_size,
        )
        fit_operator = MatrixFreeNormalizedE2Jacobian(
            proposal_model,
            parameterization.unpack,
            theta,
            datasets["fit"],
            chunk_size=args.operator_chunk_size,
        )
        selection_operator = MatrixFreeNormalizedE2Jacobian(
            proposal_model,
            parameterization.unpack,
            theta,
            datasets["selection"],
            chunk_size=args.operator_chunk_size,
        )
        residual = fit_operator.residual()
        fit_e2 = float(real_inner(residual, residual))
        selection_e2 = selection_operator.native_e2
        gradient = fit_operator.vjp(residual)
        gradient_norm = float(vector_norm(gradient))
        rayleigh_scale = gradient_norm**2 / max(
            fit_e2,
            np.finfo(np.float64).tiny,
        )
        if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
            raise FloatingPointError("sibling rotation has no finite native direction")

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

        parameter_vjps: list[torch.Tensor] = []

        def normal_and_cache(cotangent: torch.Tensor) -> torch.Tensor:
            parameter_vjp = fit_operator.vjp(cotangent)
            parameter_vjps.append(parameter_vjp)
            return fit_operator.jvp(parameter_vjp)

        lanczos = lanczos_tridiagonal(
            normal_and_cache,
            residual,
            steps=args.lanczos_steps,
            callback=progress,
        )
        if len(parameter_vjps) != lanczos.tridiagonal.shape[0]:
            raise RuntimeError("Lanczos/VJP cache dimension mismatch")
        parameter_vjp_basis = torch.stack(parameter_vjps, dim=1)
        del parameter_vjps
        theta_scale = max(
            float(vector_norm(theta)),
            np.finfo(np.float64).tiny,
        )
        rows: list[dict[str, Any]] = []
        for ridge_factor in args.ridge_factors:
            ridge = float(ridge_factor * rayleigh_scale)
            coefficients = ridge_coefficients_from_lanczos(lanczos, ridge)
            base_delta = -(
                parameter_vjp_basis
                @ coefficients.to(dtype=parameter_vjp_basis.dtype)
            )
            for alpha in args.line_search_alphas:
                coordinates = (theta + float(alpha) * base_delta).detach()
                relative_step = float(
                    vector_norm(coordinates - theta)
                ) / theta_scale
                row: dict[str, Any] = {
                    "ridge_factor": float(ridge_factor),
                    "ridge": ridge,
                    "alpha": float(alpha),
                    "relative_coordinate_step": relative_step,
                    "eligible": False,
                    "failure": None,
                    "_coordinates": coordinates,
                }
                if relative_step > args.maximum_relative_coordinate_step:
                    row["failure"] = "relative_coordinate_step"
                    rows.append(row)
                    continue
                try:
                    fit_candidate = fit_operator.residual(coordinates)
                    selection_candidate = selection_operator.residual(
                        coordinates
                    )
                    row["actual_fit_e2"] = float(
                        real_inner(fit_candidate, fit_candidate)
                    )
                    row["actual_selection_e2"] = float(
                        real_inner(
                            selection_candidate,
                            selection_candidate,
                        )
                    )
                    row["actual_fit_capture"] = (
                        1.0 - row["actual_fit_e2"] / fit_e2
                    )
                    row["actual_selection_capture"] = (
                        1.0 - row["actual_selection_e2"] / selection_e2
                    )
                except (RuntimeError, FloatingPointError) as error:
                    row["failure"] = f"{type(error).__name__}: {error}"
                rows.append(row)

        viable = [
            row
            for row in rows
            if row.get("actual_fit_capture", 0.0) > 0
            and row.get("actual_selection_capture", 0.0) > 0
        ]
        shortlisted = sorted(
            viable,
            key=lambda row: (
                row["actual_selection_e2"],
                row["actual_fit_e2"],
            ),
        )[:4]
        proposal_state = copy.deepcopy(proposal_model.state_dict())
        for row in shortlisted:
            try:
                copy_coordinates_(
                    proposal_model,
                    parameterization,
                    row["_coordinates"],
                )
                statistics, _, tail = metric_row(
                    proposal_model,
                    datasets["selection"],
                    chunk_size=args.eval_batch_size,
                )
                row["selection_statistics"] = statistics
                row["selection_tail"] = tail
                row["eligible"] = bool(
                    statistics["sigma_official_formula"]
                    < baseline_selection["sigma_official_formula"]
                    and statistics["weighted_rms_abs_residual"]
                    < baseline_selection["weighted_rms_abs_residual"]
                    and tail_guard(
                        tail,
                        baseline_tail,
                        relative_degradation=(
                            args.maximum_selection_tail_relative_degradation
                        ),
                    )
                )
            except (RuntimeError, FloatingPointError) as error:
                row["failure"] = f"{type(error).__name__}: {error}"
            finally:
                proposal_model.load_state_dict(proposal_state)
        eligible = [row for row in shortlisted if row["eligible"]]
        selected = (
            min(
                eligible,
                key=lambda row: (
                    row["actual_selection_e2"],
                    row["selection_statistics"]["sigma_official_formula"],
                ),
            )
            if eligible
            else None
        )

        saved_proposal = None
        if selected is not None:
            high_model = build_checkpoint_model(
                proposal_payload,
                device=device,
            )
            high_baseline = build_checkpoint_model(
                baseline_payload,
                device=device,
            )
            high_parameterization = (
                SketchedSiblingRotationParameterization.from_models(
                    high_baseline,
                    high_model,
                    parent=parent,
                    sibling_edges=(sibling_edges[0], sibling_edges[1]),
                    directions=args.rotation_directions,
                    seed=args.seed + 100,
                )
            )
            copy_coordinates_(
                high_model,
                high_parameterization,
                selected["_coordinates"].to(
                    device=device,
                    dtype=high_parameterization.theta.dtype,
                ),
            )
            output_payload = copy.deepcopy(proposal_payload)
            output_payload["state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in high_model.state_dict().items()
            }
            output_payload["selection_only"] = True
            output_payload["confirmation_passes"] = False
            output_payload["teacher_runtime_dependency"] = False
            output_payload["native_sibling_subspace_rotation"] = {
                "parent": parent,
                "sibling_edges": sibling_edges,
                "old_teacher_rows_frozen": True,
                "old_parent_block_frozen": True,
                "normalization_derivative_included": True,
            }
            saved_proposal = output_dir / "selection_proposal.pt"
            torch.save(output_payload, saved_proposal)

        report = {
            "schema": "generic-quintic-sibling-subspace-rotation-v1",
            "teacher_role": "absent_after_round_zero",
            "teacher_runtime_dependency": False,
            "configuration": {
                **vars(args),
                "baseline_checkpoint": str(baseline_path),
                "rank_proposal_checkpoint": str(proposal_path),
                "output_dir": str(output_dir),
            },
            "parameterization": {
                "parent": parent,
                "sibling_edges": sibling_edges,
                "complex_coordinate_count": int(theta.numel()),
                "real_coordinate_count": int(2 * theta.numel()),
                "old_teacher_rows_frozen": True,
                "old_parent_block_frozen": True,
                "new_rows_tangent_to_current_row_complement": True,
                "rotation_directions_per_child": args.rotation_directions,
                "rotation_seed": args.seed + 100,
            },
            "linearization": {
                "fit_native_e2": fit_e2,
                "selection_native_e2": selection_e2,
                "gradient_norm": gradient_norm,
                "rayleigh_scale": rayleigh_scale,
                "lanczos_dimension": int(lanczos.tridiagonal.shape[0]),
            },
            "selection": {
                "baseline": baseline_selection,
                "baseline_tail": baseline_tail,
                "rank_proposal": proposal_selection,
                "rank_proposal_tail": proposal_tail,
                "candidates": [serializable(row) for row in rows],
                "selected": (
                    None if selected is None else serializable(selected)
                ),
            },
            "confirmation": {
                "opened": False,
                "passes": False,
            },
            "data": {
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
            },
            "proposal_checkpoint": (
                None if saved_proposal is None else str(saved_proposal)
            ),
            "proposal_checkpoint_sha256": (
                None
                if saved_proposal is None
                else sha256_file(saved_proposal)
            ),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "selection_complete",
                "selection_found_candidate": selected is not None,
                "confirmation_opened": False,
                "proposal_checkpoint": (
                    None if saved_proposal is None else str(saved_proposal)
                ),
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
