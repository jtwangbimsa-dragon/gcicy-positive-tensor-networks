#!/usr/bin/env python3
"""Refine compact multiplication-tree root coefficients with native MA E2.

This is the direct-tree control for the projected-native Fermat experiment.
The left/right Schmidt subspaces and the positive baseline factor are frozen.
Only the root Schmidt coefficients are optimized.  Candidate metrics are
formed as ``H = Reynolds(B^* B)``, so positivity and exact Fermat symmetry are
structural rather than penalty terms.
"""

from __future__ import annotations

import argparse
import hashlib
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

from gcicy_metric.pipeline.fermat_full_h import (
    FermatReynoldsBlockProjector,
    FermatSymmetricFullH,
)
from scripts.audit_quintic_full_h_oracle_multiplication_tree import (
    exact_product_right_inverse,
    quotient_multiplication_map,
)
from scripts.build_quintic_compact_phase_sector_purification_residual import (
    reconstruct_schmidt_from_blocks,
    reconstruct_section_factor_from_schmidt,
)
from scripts.train_quintic_full_h_gn_matched import make_dense_split
from scripts.train_quintic_full_h_same_points import (
    ratio_statistics,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-model", type=Path, required=True)
    parser.add_argument("--reference-h", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument(
        "--pool-points",
        type=Path,
        help="optional generated points NPZ with X/weights/omega_squared",
    )
    parser.add_argument(
        "--pool-pullbacks",
        type=Path,
        help="pullback NPY paired with --pool-points",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-indices-file",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "NPZ files whose train/selection/confirmation indices are excluded "
            "from this round"
        ),
    )
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--selection-size", type=int, default=512)
    parser.add_argument("--confirmation-size", type=int, default=2048)
    parser.add_argument(
        "--freeze-first-coordinates",
        type=int,
        default=0,
        help="keep an initial prefix of root channels exactly fixed",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=(
            "log-relative",
            "additive",
            "complex-additive",
            "vector-rotation",
        ),
        default="log-relative",
        help=(
            "log-relative rescales existing singular values; additive uses "
            "stored per-channel coordinate scales and supports an exact zero "
            "root; complex-additive also trains each channel's relative "
            "phase; vector-rotation trains one orthogonal left/right angle per "
            "channel"
        ),
    )
    parser.add_argument("--rotation-seed", type=int, default=202607382)
    parser.add_argument(
        "--rotation-directions",
        type=int,
        default=1,
        help=(
            "number of mutually orthogonal left and right tangent directions "
            "per Schmidt channel in vector-rotation mode"
        ),
    )
    parser.add_argument(
        "--rotation-sketch-dimension",
        type=int,
        default=0,
        help=(
            "optional orthonormal random sketch dimension for the complete "
            "vector-rotation tangent space; zero uses every tangent coordinate"
        ),
    )
    parser.add_argument(
        "--rotation-sketch-seed",
        type=int,
        default=202607393,
    )
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--finite-difference-step", type=float, default=2.0e-4)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(1000.0, 100.0, 10.0, 1.0),
    )
    parser.add_argument(
        "--spectral-relative-cutoffs",
        type=float,
        nargs="*",
        default=(),
        help=(
            "optional truncated-GN eigenvalue cutoffs relative to the largest "
            "Gram eigenvalue"
        ),
    )
    parser.add_argument(
        "--line-search-alphas",
        type=float,
        nargs="+",
        default=(0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--maximum-rms-log-multiplier", type=float, default=0.1)
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=5.0e-3,
    )
    parser.add_argument(
        "--maximum-confirmation-tail-relative-degradation",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=202607351)
    parser.add_argument(
        "--diagnostic-split-channels",
        type=int,
        default=0,
        help=(
            "report separate and incremental Jacobian ranks before/after this "
            "root-channel boundary"
        ),
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="stop after the finite-difference Jacobian and rank/capture audit",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_size,
        args.selection_size,
        args.confirmation_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.finite_difference_step,
        args.maximum_rms_log_multiplier,
        args.rotation_directions,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sample, batch, difference, and trust values must be positive")
    if (args.pool_points is None) != (args.pool_pullbacks is None):
        raise ValueError("pool points and pool pullbacks must be supplied together")
    if args.freeze_first_coordinates < 0:
        raise ValueError("frozen coordinate count must be non-negative")
    if args.rotation_sketch_dimension < 0:
        raise ValueError("rotation sketch dimension cannot be negative")
    if args.diagnostic_split_channels < 0:
        raise ValueError("diagnostic split must be non-negative")
    if not args.ridge_factors or any(value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be positive")
    if any(
        value <= 0 or value > 1
        for value in args.spectral_relative_cutoffs
    ):
        raise ValueError("spectral relative cutoffs must lie in (0,1]")
    if not args.line_search_alphas or any(
        value <= 0 or value > 1 for value in args.line_search_alphas
    ):
        raise ValueError("line-search alphas must lie in (0,1]")
    if (
        args.maximum_selection_tail_relative_degradation < 0
        or args.maximum_confirmation_tail_relative_degradation < 0
    ):
        raise ValueError("tail degradation allowances must be non-negative")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tail_summary(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "q999": float(statistics["abs_residual_weighted_quantiles"]["q0.9990"]),
        "cvar99": float(statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]),
        "maximum": float(statistics["abs_residual_weighted_quantiles"]["q1.0000"]),
    }


def absorb_complex_root_coefficients(
    left: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Store complex root weights as real magnitudes and left-vector phases."""

    values = np.asarray(coefficients, dtype=np.complex128)
    magnitude = np.abs(values)
    phase = np.ones_like(values)
    nonzero = magnitude > np.finfo(np.float64).tiny
    phase[nonzero] = values[nonzero] / magnitude[nonzero]
    return np.asarray(left, dtype=np.complex128) * phase[None, :], magnitude


def random_orthogonal_column_directions(
    basis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return orthonormal columns orthogonal to the supplied column space."""

    return random_orthogonal_column_direction_bank(
        basis,
        rng,
        direction_count=1,
    )[:, :, 0]


def random_orthogonal_column_direction_bank(
    basis: np.ndarray,
    rng: np.random.Generator,
    *,
    direction_count: int,
) -> np.ndarray:
    """Return independent tangent directions for every basis column.

    The result has shape ``(ambient, rank, direction_count)``. All tangent
    columns are mutually orthonormal and orthogonal to the inherited basis.
    """

    vectors = np.asarray(basis, dtype=np.complex128)
    if vectors.ndim != 2:
        raise ValueError("rotation basis must be a matrix")
    if direction_count <= 0:
        raise ValueError("rotation direction count must be positive")
    ambient_dimension, rank = vectors.shape
    if rank == 0:
        return np.empty(
            (ambient_dimension, 0, direction_count),
            dtype=np.complex128,
        )
    tangent_count = rank * direction_count
    if ambient_dimension - rank < tangent_count:
        raise ValueError(
            "rotation audit has too few complement dimensions for the "
            "requested tangent bank"
        )
    basis_gram = np.conj(vectors).T @ vectors
    if not np.allclose(
        basis_gram,
        np.eye(rank),
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise ValueError("Schmidt vectors must be orthonormal before rotation")

    directions = np.empty(
        (ambient_dimension, tangent_count),
        dtype=np.complex128,
    )
    accepted = 0
    while accepted < tangent_count:
        candidate = rng.normal(size=ambient_dimension) + 1j * rng.normal(
            size=ambient_dimension
        )
        candidate = candidate - vectors @ (np.conj(vectors).T @ candidate)
        if accepted:
            previous = directions[:, :accepted]
            candidate = candidate - previous @ (
                np.conj(previous).T @ candidate
            )
        norm = np.linalg.norm(candidate)
        if norm <= 100.0 * np.finfo(np.float64).eps:
            continue
        directions[:, accepted] = candidate / norm
        accepted += 1
    return directions.reshape(ambient_dimension, rank, direction_count)


def geodesic_column_rotation(
    basis: np.ndarray,
    tangent_bank: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Rotate each basis column in its orthonormal tangent subspace."""

    vectors = np.asarray(basis, dtype=np.complex128)
    tangents = np.asarray(tangent_bank, dtype=np.complex128)
    values = np.asarray(coordinates, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("rotation basis must be a matrix")
    if tangents.ndim != 3 or tangents.shape[:2] != vectors.shape:
        raise ValueError("rotation tangent bank is not aligned with the basis")
    if values.shape != (vectors.shape[1], tangents.shape[2]):
        raise ValueError("rotation coordinates have the wrong shape")
    radii = np.linalg.norm(values, axis=1)
    sine_over_radius = np.ones_like(radii)
    nonzero = radii > np.sqrt(np.finfo(np.float64).eps)
    sine_over_radius[nonzero] = np.sin(radii[nonzero]) / radii[nonzero]
    tangent = np.einsum("irk,rk->ir", tangents, values, optimize=True)
    return (
        vectors * np.cos(radii)[None, :]
        + tangent * sine_over_radius[None, :]
    )


def tail_guard(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    relative_degradation: float,
) -> bool:
    limit = 1.0 + relative_degradation
    return bool(
        candidate["q999"] <= limit * baseline["q999"]
        and candidate["cvar99"] <= limit * baseline["cvar99"]
        and candidate["maximum"] <= limit * baseline["maximum"]
    )


class CompactRootCoefficientMetric:
    """Exact positive metric generated by fixed Schmidt vectors and root weights."""

    def __init__(
        self,
        compact_path: Path,
        reference_h_path: Path,
        *,
        device: torch.device,
        coordinate_mode: str,
        rotation_seed: int,
        rotation_directions: int = 1,
    ) -> None:
        payload = np.load(compact_path, allow_pickle=False)
        self.payload = {key: np.asarray(payload[key]).copy() for key in payload.files}
        self.degree = int(self.payload["degree"])
        self.exponents = np.asarray(self.payload["exponents"], dtype=np.int64)
        self.split = tuple(int(value) for value in self.payload["split"])
        self.baseline_factor = np.asarray(
            self.payload["baseline_factor"],
            dtype=np.complex128,
        )
        self.device = device
        self.coordinate_mode = coordinate_mode
        self.rotation_directions = int(rotation_directions)
        if self.rotation_directions <= 0:
            raise ValueError("rotation direction count must be positive")
        if self.coordinate_mode not in {
            "log-relative",
            "additive",
            "complex-additive",
            "vector-rotation",
        }:
            raise ValueError("unsupported root coordinate mode")
        if self.coordinate_mode == "vector-rotation":
            self.coordinate_width = 2 * self.rotation_directions
        elif self.coordinate_mode == "complex-additive":
            self.coordinate_width = 2
        else:
            self.coordinate_width = 1
        self.normalization = 1.0 / (math.pi * self.degree)
        self.blocks = []
        self.coordinate_slices = []
        rotation_rng = np.random.default_rng(rotation_seed)
        offset = 0
        for index in range(int(self.payload["sector_count"])):
            prefix = f"sector_{index:03d}"
            singular = np.asarray(
                self.payload[f"{prefix}_singular"],
                dtype=np.float64,
            )
            coordinate_scale = np.asarray(
                self.payload.get(
                    f"{prefix}_coordinate_scale",
                    np.ones_like(singular),
                ),
                dtype=np.float64,
            )
            if coordinate_scale.shape != singular.shape or np.any(
                ~np.isfinite(coordinate_scale) | (coordinate_scale <= 0)
            ):
                raise ValueError("root coordinate scales must be finite and positive")
            coordinate_count = self.coordinate_width * len(singular)
            self.coordinate_slices.append(slice(offset, offset + coordinate_count))
            offset += coordinate_count
            block = {
                "prefix": prefix,
                "row_indices": np.asarray(
                    self.payload[f"{prefix}_row_indices"],
                    dtype=np.int64,
                ),
                "column_indices": np.asarray(
                    self.payload[f"{prefix}_column_indices"],
                    dtype=np.int64,
                ),
                "left": np.asarray(
                    self.payload[f"{prefix}_left"],
                    dtype=np.complex128,
                ),
                "singular": singular,
                "coordinate_scale": coordinate_scale,
                "right": np.asarray(
                    self.payload[f"{prefix}_right"],
                    dtype=np.complex128,
                ),
            }
            if self.coordinate_mode == "vector-rotation":
                block["left_rotation_bank"] = (
                    random_orthogonal_column_direction_bank(
                        block["left"],
                        rotation_rng,
                        direction_count=self.rotation_directions,
                    )
                )
                right_columns = np.conj(block["right"]).T
                block["right_rotation_column_bank"] = (
                    random_orthogonal_column_direction_bank(
                        right_columns,
                        rotation_rng,
                        direction_count=self.rotation_directions,
                    )
                )
            self.blocks.append(block)
        self.channel_count = sum(len(block["singular"]) for block in self.blocks)
        self.parameter_count = offset
        if self.channel_count != int(self.payload["rank"]):
            raise RuntimeError("compact root rank and singular-value count disagree")

        left, right, output, _ = quotient_multiplication_map(*self.split)
        if not np.array_equal(output, self.exponents):
            raise RuntimeError("compact root output basis is inconsistent")
        self.left_count = len(left)
        self.right_count = len(right)
        self.right_inverse = exact_product_right_inverse(
            left,
            right,
            output,
            mode="canonical-exact",
        )
        self.schmidt_shape = (self.left_count**2, self.right_count**2)

        reference_payload = np.load(reference_h_path, allow_pickle=False)
        self.reference_h = np.asarray(
            reference_payload["global_h_matrix"],
            dtype=np.complex128,
        )
        if self.reference_h.shape != (len(self.exponents), len(self.exponents)):
            raise ValueError("reference H has the wrong section dimension")
        self.target_trace = float(np.trace(self.reference_h).real)
        self.projector = FermatReynoldsBlockProjector(
            self.exponents,
            conjugation_invariant=True,
        )
        baseline_blocks = self.h_blocks(np.zeros(self.parameter_count))
        reconstructed = self.materialize_blocks(baseline_blocks)
        self.reconstruction_relative_error = float(
            np.linalg.norm(reconstructed - self.reference_h)
            / np.linalg.norm(self.reference_h)
        )
        self.reconstruction_maximum_error = float(
            np.max(np.abs(reconstructed - self.reference_h))
        )
        if (
            self.reconstruction_relative_error > 5.0e-12
            or self.reconstruction_maximum_error > 5.0e-11
        ):
            raise RuntimeError(
                "compact root does not reproduce its frozen H artifact: "
                f"relative={self.reconstruction_relative_error:.3e}, "
                f"maximum={self.reconstruction_maximum_error:.3e}"
            )
        self.reference_blocks = self.to_torch_blocks(baseline_blocks)

    def singular_values(self, coordinates: np.ndarray) -> list[np.ndarray]:
        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (self.parameter_count,):
            raise ValueError("root coordinate vector has the wrong shape")
        if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 50:
            raise FloatingPointError("root log multipliers are nonfinite or excessive")
        if self.coordinate_mode == "log-relative":
            return [
                block["singular"] * np.exp(values[coordinate_slice])
                for block, coordinate_slice in zip(
                    self.blocks,
                    self.coordinate_slices,
                )
            ]
        if self.coordinate_mode == "complex-additive":
            return [
                block["singular"].astype(np.complex128)
                + block["coordinate_scale"]
                * (
                    values[coordinate_slice].reshape(-1, 2)[:, 0]
                    + 1j * values[coordinate_slice].reshape(-1, 2)[:, 1]
                )
                for block, coordinate_slice in zip(
                    self.blocks,
                    self.coordinate_slices,
                )
            ]
        if self.coordinate_mode == "vector-rotation":
            return [block["singular"].copy() for block in self.blocks]
        return [
            block["singular"]
            + block["coordinate_scale"] * values[coordinate_slice]
            for block, coordinate_slice in zip(self.blocks, self.coordinate_slices)
        ]

    def root_components(
        self,
        coordinates: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        singular_values = self.singular_values(coordinates)
        if self.coordinate_mode != "vector-rotation":
            return [
                (block["left"], singular, block["right"])
                for block, singular in zip(self.blocks, singular_values)
            ]

        values = np.asarray(coordinates, dtype=np.float64)
        components = []
        for block, singular, coordinate_slice in zip(
            self.blocks,
            singular_values,
            self.coordinate_slices,
        ):
            angles = values[coordinate_slice].reshape(
                -1,
                2,
                self.rotation_directions,
            )
            left = geodesic_column_rotation(
                block["left"],
                block["left_rotation_bank"],
                angles[:, 0, :],
            )
            right_columns = geodesic_column_rotation(
                np.conj(block["right"]).T,
                block["right_rotation_column_bank"],
                angles[:, 1, :],
            )
            right = np.conj(right_columns).T
            components.append((left, singular, right))
        return components

    def section_factor(self, coordinates: np.ndarray) -> np.ndarray:
        candidate_blocks = []
        for block, (left, singular, right) in zip(
            self.blocks,
            self.root_components(coordinates),
        ):
            candidate_blocks.append(
                {
                    "row_indices": block["row_indices"],
                    "column_indices": block["column_indices"],
                    "left": left,
                    "singular": singular,
                    "right": right,
                }
            )
        schmidt = reconstruct_schmidt_from_blocks(
            candidate_blocks,
            shape=self.schmidt_shape,
        )
        correction = reconstruct_section_factor_from_schmidt(
            schmidt,
            self.right_inverse,
            left_count=self.left_count,
            right_count=self.right_count,
        )
        return self.baseline_factor + correction

    def h_blocks(
        self,
        coordinates: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        factor = self.section_factor(coordinates)
        raw_h = factor.conj().T @ factor
        blocks = self.projector.project_blocks(raw_h)
        observed_trace = sum(
            float(np.real(np.trace(values, axis1=1, axis2=2)).sum())
            for _, values in blocks
        )
        scale = self.target_trace / observed_trace
        return tuple((indices, values * scale) for indices, values in blocks)

    def materialize_blocks(
        self,
        blocks: tuple[tuple[np.ndarray, np.ndarray], ...],
    ) -> np.ndarray:
        result = np.zeros_like(self.reference_h)
        for indices, values in blocks:
            for block_indices, block in zip(indices, values):
                result[np.ix_(block_indices, block_indices)] = block
        return result

    def h_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        return self.materialize_blocks(self.h_blocks(coordinates))

    def to_torch_blocks(
        self,
        blocks: tuple[tuple[np.ndarray, np.ndarray], ...],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (
                torch.tensor(indices, dtype=torch.long, device=self.device),
                torch.tensor(values, dtype=torch.complex128, device=self.device),
            )
            for indices, values in blocks
        )

    def raw_and_minimum(
        self,
        coordinates: np.ndarray,
        dataset: dict[str, Any],
        *,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = self.to_torch_blocks(self.h_blocks(coordinates))
        raw_rows = []
        minimum_rows = []
        with torch.no_grad():
            for start in range(0, dataset["count"], chunk_size):
                stop = min(start + chunk_size, dataset["count"])
                section_values = dataset["values"][start:stop]
                section_derivatives = dataset["derivatives"][start:stop]
                reference_norm = FermatSymmetricFullH._quadratic_norm(
                    section_values,
                    self.reference_blocks,
                )
                inverse_scale = torch.rsqrt(reference_norm)
                values = section_values * inverse_scale[:, None]
                derivatives = section_derivatives * inverse_scale[:, None, None]
                denominator, first, gradient = (
                    FermatSymmetricFullH._quadratic_jets(
                        values,
                        derivatives,
                        blocks,
                    )
                )
                metric = first / denominator[:, None, None]
                metric = metric - (
                    torch.conj(gradient)[:, :, None]
                    * gradient[:, None, :]
                    / denominator[:, None, None] ** 2
                )
                metric = self.normalization * 0.5 * (
                    metric + torch.conj(torch.transpose(metric, 1, 2))
                )
                eigenvalues = torch.linalg.eigvalsh(metric)
                if not bool(torch.all(torch.isfinite(eigenvalues))):
                    raise FloatingPointError("candidate metric has nonfinite eigenvalues")
                raw_rows.append(
                    torch.sum(torch.log(eigenvalues), dim=1)
                    - dataset["log_omega"][start:stop]
                )
                minimum_rows.append(torch.min(eigenvalues, dim=1).values)
        return torch.cat(raw_rows), torch.cat(minimum_rows)

    def residual(
        self,
        coordinates: np.ndarray,
        dataset: dict[str, Any],
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        raw, minimum = self.raw_and_minimum(
            coordinates,
            dataset,
            chunk_size=chunk_size,
        )
        if not bool(torch.all(minimum > 0)):
            raise FloatingPointError("candidate metric is not positive")
        maximum = torch.max(raw)
        log_mean = maximum + torch.log(
            torch.sum(dataset["weights"] * torch.exp(raw - maximum))
        )
        ratio = torch.exp(raw - log_mean)
        return torch.sqrt(dataset["weights"]) * (ratio - 1.0)

    def statistics(
        self,
        coordinates: np.ndarray,
        dataset: dict[str, Any],
        *,
        chunk_size: int,
    ) -> tuple[dict[str, Any], np.ndarray]:
        raw, minimum = self.raw_and_minimum(
            coordinates,
            dataset,
            chunk_size=chunk_size,
        )
        return ratio_statistics(
            raw.cpu().numpy(),
            dataset["weights_numpy"],
            minimum.cpu().numpy(),
        )

    def save_compact(self, path: Path, coordinates: np.ndarray) -> None:
        payload = dict(self.payload)
        components = self.root_components(coordinates)
        for block, (left, singular, right) in zip(self.blocks, components):
            if self.coordinate_mode == "complex-additive":
                stored_left, magnitude = absorb_complex_root_coefficients(
                    left,
                    singular,
                )
                payload[f"{block['prefix']}_left"] = stored_left
                payload[f"{block['prefix']}_singular"] = magnitude
            elif self.coordinate_mode == "vector-rotation":
                payload[f"{block['prefix']}_left"] = left
                payload[f"{block['prefix']}_singular"] = singular
                payload[f"{block['prefix']}_right"] = right
            else:
                payload[f"{block['prefix']}_singular"] = singular
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)


def load_split_arrays(
    source_dir: Path,
    pullbacks_dir: Path,
    *,
    sizes: tuple[int, int, int],
    seed: int,
    excluded_indices: np.ndarray,
    pool_points_path: Path | None,
    pool_pullbacks_path: Path | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[np.ndarray]]:
    if pool_points_path is None:
        data = np.load(
            source_dir / "training_data" / "dataset.npz",
            allow_pickle=False,
        )
        x_values = np.asarray(data["X_train"], dtype=np.float32)
        labels = np.asarray(data["y_train"], dtype=np.float64)
        pullbacks = np.load(
            pullbacks_dir / "train_pullbacks.npy",
            mmap_mode="r",
        )
    else:
        if pool_pullbacks_path is None:
            raise RuntimeError("generated pool pullbacks are missing")
        points = np.load(pool_points_path, allow_pickle=False)
        x_values = np.asarray(points["X"], dtype=np.float32)
        labels = np.column_stack(
            (
                np.asarray(points["weights"], dtype=np.float64),
                np.asarray(points["omega_squared"], dtype=np.float64),
            )
        )
        pullbacks = np.load(pool_pullbacks_path, mmap_mode="r")
    if len(x_values) != len(labels) or len(x_values) != len(pullbacks):
        raise RuntimeError("training point, label, and pullback counts disagree")
    requested = sum(sizes)
    excluded = np.unique(np.asarray(excluded_indices, dtype=np.int64))
    if np.any((excluded < 0) | (excluded >= len(x_values))):
        raise ValueError("excluded source-pool index is out of range")
    available = np.ones(len(x_values), dtype=bool)
    available[excluded] = False
    eligible = np.flatnonzero(available)
    if requested > len(eligible):
        raise ValueError("requested split sizes exceed the source training pool")
    indices = np.random.default_rng(seed).permutation(eligible)[:requested]
    rows = []
    index_rows = []
    start = 0
    for size in sizes:
        selected = np.asarray(indices[start : start + size], dtype=np.int64)
        rows.append(
            (
                np.asarray(x_values[selected]),
                np.asarray(labels[selected]),
                np.asarray(pullbacks[selected]),
            )
        )
        index_rows.append(selected)
        start += size
    return rows, index_rows


def paired_improvement(
    baseline_ratio: np.ndarray,
    candidate_ratio: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    mass = np.asarray(weights, dtype=np.float64)
    mass = mass / np.sum(mass)
    baseline = np.asarray(baseline_ratio, dtype=np.float64) - 1.0
    candidate = np.asarray(candidate_ratio, dtype=np.float64) - 1.0

    def summary(values: np.ndarray) -> dict[str, float]:
        mean = float(np.sum(mass * values))
        variance = float(np.sum(mass * np.square(values - mean)))
        effective_count = float(1.0 / np.sum(np.square(mass)))
        standard_error = math.sqrt(max(variance, 0.0) / effective_count)
        return {
            "mean": mean,
            "standard_error": standard_error,
            "ci95_low": mean - 1.96 * standard_error,
            "ci95_high": mean + 1.96 * standard_error,
            "effective_count": effective_count,
        }

    return {
        "e2": summary(np.square(baseline) - np.square(candidate)),
        "sigma": summary(np.abs(baseline) - np.abs(candidate)),
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
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        compact_path = args.compact_model.expanduser().resolve()
        reference_h_path = args.reference_h.expanduser().resolve()
        source_dir = args.source_run_dir.expanduser().resolve()
        pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
        pool_points_path = (
            None
            if args.pool_points is None
            else args.pool_points.expanduser().resolve()
        )
        pool_pullbacks_path = (
            None
            if args.pool_pullbacks is None
            else args.pool_pullbacks.expanduser().resolve()
        )
        exclusion_paths = [
            path.expanduser().resolve() for path in args.exclude_indices_file
        ]
        excluded_rows = []
        for path in exclusion_paths:
            payload = np.load(path, allow_pickle=False)
            for key in (
                "train_indices",
                "selection_indices",
                "confirmation_indices",
                "validation_indices",
            ):
                if key in payload.files:
                    excluded_rows.append(
                        np.asarray(payload[key], dtype=np.int64).reshape(-1)
                    )
        excluded_indices = (
            np.unique(np.concatenate(excluded_rows))
            if excluded_rows
            else np.empty(0, dtype=np.int64)
        )

        compile_started = time.perf_counter()
        model = CompactRootCoefficientMetric(
            compact_path,
            reference_h_path,
            device=device,
            coordinate_mode=args.coordinate_mode,
            rotation_seed=args.rotation_seed,
            rotation_directions=args.rotation_directions,
        )
        if args.freeze_first_coordinates >= model.channel_count:
            raise ValueError(
                "frozen coordinate prefix must leave at least one active channel"
            )
        if args.diagnostic_split_channels > model.channel_count:
            raise ValueError("diagnostic split exceeds the compact root rank")
        active_start = (
            args.freeze_first_coordinates * model.coordinate_width
        )
        full_active_parameter_count = model.parameter_count - active_start
        coordinate_sketch = None
        if args.rotation_sketch_dimension:
            if model.coordinate_mode != "vector-rotation":
                raise ValueError(
                    "rotation sketches require vector-rotation coordinates"
                )
            if args.rotation_sketch_dimension > full_active_parameter_count:
                raise ValueError(
                    "rotation sketch exceeds the complete active tangent space"
                )
            if args.diagnostic_split_channels:
                raise ValueError(
                    "channel-prefix rank diagnostics are undefined after a "
                    "global rotation sketch"
                )
            sketch_rng = np.random.default_rng(args.rotation_sketch_seed)
            raw_sketch = sketch_rng.normal(
                size=(
                    full_active_parameter_count,
                    args.rotation_sketch_dimension,
                )
            )
            coordinate_sketch, triangular = np.linalg.qr(
                raw_sketch,
                mode="reduced",
            )
            if np.min(np.abs(np.diag(triangular))) <= 1.0e-12:
                raise FloatingPointError(
                    "rotation tangent sketch lost numerical rank"
                )
            active_parameter_count = int(args.rotation_sketch_dimension)
        else:
            active_parameter_count = full_active_parameter_count

        def expand_coordinates(active: np.ndarray) -> np.ndarray:
            values = np.asarray(active, dtype=np.float64)
            if values.shape != (active_parameter_count,):
                raise ValueError("active root coordinate vector has the wrong shape")
            full = np.zeros(model.parameter_count, dtype=np.float64)
            full[active_start:] = (
                values
                if coordinate_sketch is None
                else coordinate_sketch @ values
            )
            return full

        compile_seconds = time.perf_counter() - compile_started
        if not args.quiet:
            print(
                f"compiled root rank={model.channel_count}, "
                f"real coordinates={model.parameter_count} in "
                f"{compile_seconds:.2f}s; reconstruction relative error="
                f"{model.reconstruction_relative_error:.3e}",
                flush=True,
            )

        split_arrays, split_indices = load_split_arrays(
            source_dir,
            pullbacks_dir,
            sizes=(
                args.train_size,
                args.selection_size,
                args.confirmation_size,
            ),
            seed=args.seed,
            excluded_indices=excluded_indices,
            pool_points_path=pool_points_path,
            pool_pullbacks_path=pool_pullbacks_path,
        )
        datasets = [
            make_dense_split(
                arrays,
                model.exponents,
                feature_batch_size=args.feature_batch_size,
                device=device,
            )
            for arrays in split_arrays
        ]
        train, selection, confirmation = datasets
        np.savez_compressed(
            output_dir / "optimization_indices.npz",
            train_indices=split_indices[0],
            selection_indices=split_indices[1],
            confirmation_indices=split_indices[2],
        )

        theta = np.zeros(active_parameter_count, dtype=np.float64)
        baseline_train_residual = model.residual(
            expand_coordinates(theta),
            train,
            chunk_size=args.eval_batch_size,
        )
        baseline_train_e2 = float(torch.dot(
            baseline_train_residual,
            baseline_train_residual,
        ))
        baseline_selection, _ = model.statistics(
            expand_coordinates(theta),
            selection,
            chunk_size=args.eval_batch_size,
        )
        baseline_selection_tail = tail_summary(baseline_selection)

        write_json(
            status_path,
            {
                "state": "running",
                "phase": "finite_difference_jacobian",
                "completed": 0,
                "total": active_parameter_count,
            },
        )
        jacobian = torch.empty(
            (args.train_size, active_parameter_count),
            dtype=torch.float64,
            device=device,
        )
        for index in range(active_parameter_count):
            positive = theta.copy()
            negative = theta.copy()
            positive[index] += args.finite_difference_step
            negative[index] -= args.finite_difference_step
            jacobian[:, index] = (
                model.residual(
                    expand_coordinates(positive),
                    train,
                    chunk_size=args.eval_batch_size,
                )
                - model.residual(
                    expand_coordinates(negative),
                    train,
                    chunk_size=args.eval_batch_size,
                )
            ) / (2.0 * args.finite_difference_step)
            if (index + 1) % 8 == 0 or index + 1 == active_parameter_count:
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "finite_difference_jacobian",
                        "completed": index + 1,
                        "total": active_parameter_count,
                    },
                )
                if not args.quiet:
                    print(
                        f"finite_difference={index + 1}/{active_parameter_count}",
                        flush=True,
                    )

        gradient = torch.transpose(jacobian, 0, 1) @ baseline_train_residual
        gram = torch.transpose(jacobian, 0, 1) @ jacobian
        gram = 0.5 * (gram + torch.transpose(gram, 0, 1))
        gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(gram)
        largest_gram_eigenvalue = float(torch.max(gram_eigenvalues))
        rank_tolerance = (
            np.finfo(float).eps
            * max(gram.shape)
            * max(largest_gram_eigenvalue, np.finfo(float).tiny)
        )
        positive_eigenvalue_mask = gram_eigenvalues > rank_tolerance
        numerical_rank = int(torch.count_nonzero(positive_eigenvalue_mask))
        positive_eigenvalues = gram_eigenvalues[positive_eigenvalue_mask]
        gram_condition_number = (
            float(torch.max(positive_eigenvalues) / torch.min(positive_eigenvalues))
            if numerical_rank
            else math.inf
        )
        gradient_eigenbasis = (
            torch.transpose(gram_eigenvectors, 0, 1) @ gradient
        )
        maximum_linear_capture = float(
            torch.sum(
                gradient_eigenbasis[positive_eigenvalue_mask] ** 2
                / positive_eigenvalues
            )
            / max(baseline_train_e2, np.finfo(float).tiny)
        )

        split_diagnostics: dict[str, Any] | None = None
        split_global_coordinate = (
            args.diagnostic_split_channels * model.coordinate_width
        )
        split_active_coordinate = split_global_coordinate - active_start
        if 0 < split_active_coordinate < active_parameter_count:
            old_jacobian = jacobian[:, :split_active_coordinate]
            new_jacobian = jacobian[:, split_active_coordinate:]

            def jacobian_rank(values: torch.Tensor) -> int:
                subgram = torch.transpose(values, 0, 1) @ values
                subgram = 0.5 * (subgram + torch.transpose(subgram, 0, 1))
                eigenvalues = torch.linalg.eigvalsh(subgram)
                maximum = float(torch.max(eigenvalues))
                tolerance = (
                    np.finfo(float).eps
                    * max(subgram.shape)
                    * max(maximum, np.finfo(float).tiny)
                )
                return int(torch.count_nonzero(eigenvalues > tolerance))

            old_rank = jacobian_rank(old_jacobian)
            new_rank = jacobian_rank(new_jacobian)
            split_diagnostics = {
                "split_channel": args.diagnostic_split_channels,
                "split_active_coordinate": split_active_coordinate,
                "old_rank": old_rank,
                "new_rank": new_rank,
                "joint_rank": numerical_rank,
                "incremental_rank_over_old": numerical_rank - old_rank,
            }

        np.savez_compressed(
            output_dir / "fit_jacobian_diagnostics.npz",
            gram=gram.cpu().numpy(),
            gram_eigenvalues=gram_eigenvalues.cpu().numpy(),
            gradient=gradient.cpu().numpy(),
            baseline_residual=baseline_train_residual.cpu().numpy(),
            rank_tolerance=np.asarray(rank_tolerance, dtype=np.float64),
        )
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        rayleigh_scale = gradient_norm**2 / max(
            baseline_train_e2,
            np.finfo(float).tiny,
        )
        if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
            raise FloatingPointError("root coefficients have no finite E2 gradient")

        if args.diagnostics_only:
            audit_report = {
                "schema": "quintic-compact-root-jacobian-audit-v1",
                "configuration": {
                    key: value
                    for key, value in vars(args).items()
                    if not isinstance(value, Path)
                    and key != "exclude_indices_file"
                },
                "paths": {
                    "compact_model": str(compact_path),
                    "compact_model_sha256": sha256_file(compact_path),
                    "reference_h": str(reference_h_path),
                    "reference_h_sha256": sha256_file(reference_h_path),
                    "pool_points": (
                        None
                        if pool_points_path is None
                        else str(pool_points_path)
                    ),
                    "pool_pullbacks": (
                        None
                        if pool_pullbacks_path is None
                        else str(pool_pullbacks_path)
                    ),
                    "fit_diagnostics": str(
                        output_dir / "fit_jacobian_diagnostics.npz"
                    ),
                },
                "root": {
                    "degree": model.degree,
                    "split": list(model.split),
                    "rank": model.channel_count,
                    "active_real_parameter_count": active_parameter_count,
                    "frozen_root_channel_count": args.freeze_first_coordinates,
                    "real_coordinates_per_channel": model.coordinate_width,
                    "rotation_directions_per_side": (
                        model.rotation_directions
                        if model.coordinate_mode == "vector-rotation"
                        else None
                    ),
                    "reconstruction_relative_error": (
                        model.reconstruction_relative_error
                    ),
                    "compile_seconds": compile_seconds,
                },
                "fit": {
                    "n_points": args.train_size,
                    "baseline_e2": baseline_train_e2,
                    "gradient_norm": gradient_norm,
                    "rayleigh_scale": rayleigh_scale,
                    "gram_eigenvalue_minimum": float(
                        torch.min(gram_eigenvalues)
                    ),
                    "gram_eigenvalue_maximum": largest_gram_eigenvalue,
                    "gram_rank_tolerance": rank_tolerance,
                    "gram_numerical_rank": numerical_rank,
                    "gram_condition_number_on_numerical_range": (
                        gram_condition_number
                    ),
                    "maximum_linear_residual_capture": maximum_linear_capture,
                    "split_rank_diagnostics": split_diagnostics,
                },
                "timing_seconds": time.perf_counter() - started,
            }
            write_json(output_dir / "report.json", audit_report)
            write_json(
                status_path,
                {
                    "state": "complete",
                    "phase": "diagnostics_complete",
                    "timing_seconds": time.perf_counter() - started,
                },
            )
            print(json.dumps(audit_report, indent=2, default=str), flush=True)
            return

        candidates = []
        identity = torch.eye(
            active_parameter_count,
            dtype=torch.float64,
            device=device,
        )

        def evaluate_direction(
            delta: torch.Tensor,
            *,
            solver_metadata: dict[str, Any],
        ) -> None:
            predicted = baseline_train_residual + jacobian @ delta
            predicted_capture = 1.0 - float(torch.dot(predicted, predicted)) / max(
                baseline_train_e2,
                np.finfo(float).tiny,
            )
            delta_numpy = delta.cpu().numpy()
            for alpha in args.line_search_alphas:
                candidate = theta + alpha * delta_numpy
                rms_step = float(
                    np.linalg.norm(alpha * delta_numpy)
                    / math.sqrt(active_parameter_count)
                )
                train_residual = model.residual(
                    expand_coordinates(candidate),
                    train,
                    chunk_size=args.eval_batch_size,
                )
                train_e2 = float(torch.dot(train_residual, train_residual))
                selection_statistics, _ = model.statistics(
                    expand_coordinates(candidate),
                    selection,
                    chunk_size=args.eval_batch_size,
                )
                observed_tail = tail_summary(selection_statistics)
                eligible = bool(
                    train_e2 < baseline_train_e2
                    and selection_statistics["weighted_rms_abs_residual"]
                    < baseline_selection["weighted_rms_abs_residual"]
                    and rms_step <= args.maximum_rms_log_multiplier
                    and tail_guard(
                        observed_tail,
                        baseline_selection_tail,
                        relative_degradation=(
                            args.maximum_selection_tail_relative_degradation
                        ),
                    )
                )
                candidates.append(
                    {
                        **solver_metadata,
                        "alpha": float(alpha),
                        "rms_log_multiplier": rms_step,
                        "predicted_capture_at_alpha_1": predicted_capture,
                        "train_e2": train_e2,
                        "selection": selection_statistics,
                        "selection_tail": observed_tail,
                        "eligible": eligible,
                        "coordinates": candidate,
                    }
                )
                if not args.quiet:
                    solver_label = solver_metadata["solver"]
                    print(
                        f"candidate={len(candidates)} solver={solver_label} "
                        f"alpha={alpha:g} train_e2={train_e2:.6e} "
                        f"selection_chi="
                        f"{selection_statistics['weighted_rms_abs_residual']:.6e} "
                        f"eligible={eligible}",
                        flush=True,
                    )

        for factor in args.ridge_factors:
            ridge = factor * rayleigh_scale
            delta = torch.linalg.solve(gram + ridge * identity, -gradient)
            evaluate_direction(
                delta,
                solver_metadata={
                    "solver": "ridge",
                    "ridge_factor": float(factor),
                    "ridge": float(ridge),
                    "spectral_relative_cutoff": None,
                    "retained_modes": numerical_rank,
                },
            )

        for cutoff in args.spectral_relative_cutoffs:
            retained = gram_eigenvalues > cutoff * largest_gram_eigenvalue
            retained_count = int(torch.count_nonzero(retained))
            if retained_count == 0:
                continue
            coefficients = (
                -gradient_eigenbasis[retained] / gram_eigenvalues[retained]
            )
            delta = gram_eigenvectors[:, retained] @ coefficients
            evaluate_direction(
                delta,
                solver_metadata={
                    "solver": "truncated_spectral",
                    "ridge_factor": None,
                    "ridge": None,
                    "spectral_relative_cutoff": float(cutoff),
                    "retained_modes": retained_count,
                },
            )

        eligible = [row for row in candidates if row["eligible"]]
        selected = (
            min(
                eligible,
                key=lambda row: row["selection"]["weighted_rms_abs_residual"],
            )
            if eligible
            else None
        )
        confirmation_result = None
        accepted = False
        if selected is not None:
            baseline_confirmation, baseline_ratio = model.statistics(
                expand_coordinates(theta),
                confirmation,
                chunk_size=args.eval_batch_size,
            )
            candidate_confirmation, candidate_ratio = model.statistics(
                expand_coordinates(selected["coordinates"]),
                confirmation,
                chunk_size=args.eval_batch_size,
            )
            paired = paired_improvement(
                baseline_ratio,
                candidate_ratio,
                confirmation["weights_numpy"],
            )
            baseline_confirmation_tail = tail_summary(baseline_confirmation)
            candidate_confirmation_tail = tail_summary(candidate_confirmation)
            confirmation_tail_passed = tail_guard(
                candidate_confirmation_tail,
                baseline_confirmation_tail,
                relative_degradation=(
                    args.maximum_confirmation_tail_relative_degradation
                ),
            )
            accepted = bool(
                paired["e2"]["ci95_low"] > 0
                and paired["sigma"]["mean"] >= 0
                and confirmation_tail_passed
            )
            confirmation_result = {
                "baseline": baseline_confirmation,
                "candidate": candidate_confirmation,
                "paired_improvement": paired,
                "baseline_tail": baseline_confirmation_tail,
                "candidate_tail": candidate_confirmation_tail,
                "tail_guard_passed": confirmation_tail_passed,
                "accepted": accepted,
            }
            candidate_compact = output_dir / "candidate_compact_root.npz"
            selected_full_coordinates = expand_coordinates(
                selected["coordinates"]
            )
            model.save_compact(candidate_compact, selected_full_coordinates)
            candidate_h = model.h_matrix(selected_full_coordinates)
            np.savez_compressed(
                output_dir / "candidate_h_metric.npz",
                degree=np.asarray(model.degree, dtype=np.int64),
                exponents=model.exponents,
                global_h_matrix=candidate_h,
                source_compact_sha256=np.asarray(sha256_file(compact_path)),
            )
            if accepted:
                model.save_compact(
                    output_dir / "accepted_compact_root.npz",
                    selected_full_coordinates,
                )
                np.savez_compressed(
                    output_dir / "accepted_h_metric.npz",
                    degree=np.asarray(model.degree, dtype=np.int64),
                    exponents=model.exponents,
                    global_h_matrix=candidate_h,
                    source_compact_sha256=np.asarray(sha256_file(compact_path)),
                )

        report = {
            "schema": "quintic-compact-root-native-ma-fd-gn-v1",
            "scientific_scope": {
                "proposal_coordinates": (
                    f"{active_parameter_count} active direct "
                    f"{args.coordinate_mode} "
                    "coordinates of the compact root Schmidt coefficients; "
                    "no full-H coordinate proposal is used"
                ),
                "fixed_structure": (
                    "baseline purification factor and root left/right Schmidt "
                    "subspaces"
                ),
                "objective": (
                    "empirically normalized native Monge--Ampere E2 with the "
                    "normalization differentiated by central finite differences"
                ),
                "positivity": "H=Reynolds(B^*B) for every candidate",
            },
            "configuration": {
                key: value
                for key, value in vars(args).items()
                if not isinstance(value, Path)
                and key != "exclude_indices_file"
            },
            "paths": {
                "compact_model": str(compact_path),
                "compact_model_sha256": sha256_file(compact_path),
                "reference_h": str(reference_h_path),
                "reference_h_sha256": sha256_file(reference_h_path),
                "source_run_dir": str(source_dir),
                "pullbacks_dir": str(pullbacks_dir),
                "output_dir": str(output_dir),
                "exclude_indices_files": [str(path) for path in exclusion_paths],
                "pool_points": (
                    None if pool_points_path is None else str(pool_points_path)
                ),
                "pool_pullbacks": (
                    None
                    if pool_pullbacks_path is None
                    else str(pool_pullbacks_path)
                ),
            },
            "root": {
                "degree": model.degree,
                "split": list(model.split),
                "rank": model.channel_count,
                "direct_real_parameter_count": active_parameter_count,
                "frozen_root_channel_count": args.freeze_first_coordinates,
                "frozen_root_coordinate_count": active_start,
                "real_coordinates_per_channel": model.coordinate_width,
                "rotation_directions_per_side": (
                    model.rotation_directions
                    if model.coordinate_mode == "vector-rotation"
                    else None
                ),
                "reconstruction_relative_error": (
                    model.reconstruction_relative_error
                ),
                "reconstruction_maximum_error": (
                    model.reconstruction_maximum_error
                ),
                "compile_seconds": compile_seconds,
            },
            "indices": {
                "train_count": args.train_size,
                "selection_count": args.selection_size,
                "confirmation_count": args.confirmation_size,
                "excluded_source_pool_count": int(len(excluded_indices)),
            },
            "fit": {
                "baseline_e2": baseline_train_e2,
                "gradient_norm": gradient_norm,
                "rayleigh_scale": rayleigh_scale,
                "gram_eigenvalue_minimum": float(torch.min(gram_eigenvalues)),
                "gram_eigenvalue_maximum": largest_gram_eigenvalue,
                "gram_rank_tolerance": rank_tolerance,
                "gram_numerical_rank": numerical_rank,
                "gram_condition_number_on_numerical_range": (
                    gram_condition_number
                ),
                "maximum_linear_residual_capture": maximum_linear_capture,
                "split_rank_diagnostics": split_diagnostics,
            },
            "selection_baseline": baseline_selection,
            "candidates": [
                {key: value for key, value in row.items() if key != "coordinates"}
                for row in candidates
            ],
            "selected": (
                None
                if selected is None
                else {
                    key: value
                    for key, value in selected.items()
                    if key != "coordinates"
                }
            ),
            "confirmation": confirmation_result,
            "accepted": accepted,
            "timing_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "accepted": accepted,
                "selected": selected is not None,
                "timing_seconds": report["timing_seconds"],
            },
        )
        print(json.dumps({"accepted": accepted, "selected": selected is not None}))
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "timing_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
