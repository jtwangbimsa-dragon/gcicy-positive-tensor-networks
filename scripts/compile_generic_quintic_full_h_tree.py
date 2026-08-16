#!/usr/bin/env python3
"""Compile a generic-quintic full-H metric through the 2+2 product map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.generic_quintic import (  # noqa: E402
    GENERIC_QUINTIC_COEFFICIENTS,
    GENERIC_QUINTIC_EXPONENTS,
)
from gcicy_metric.pipeline.full_h_tree_compiler import (  # noqa: E402
    exact_monomial_right_inverse,
    lift_gauge_diagnostics,
    lift_hermitian_form,
    lift_purification_factor,
    minimum_norm_multiplication_lift,
    operator_schmidt_singular_values,
    purification_schmidt_channels,
    rank_for_squared_weight,
    section_covariance,
)
from gcicy_metric.pipeline.hypersurface_section_ring import (  # noqa: E402
    HypersurfaceSectionRing,
    evaluate_monomials,
)
from gcicy_metric.pipeline.positive_multiplication_tree import (  # noqa: E402
    balanced_binary_topology,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-h", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gram-points", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=202607402)
    parser.add_argument("--pivot-coordinate", type=int, default=0)
    parser.add_argument("--relative-eigenvalue-floor", type=float, default=1.0e-10)
    parser.add_argument(
        "--relative-singular-value-tolerance",
        type=float,
        default=1.0e-11,
    )
    parser.add_argument(
        "--purification-squared-weight-retention",
        type=float,
        default=1.0,
        help=(
            "Fraction of the H4 purification Schmidt weight used to compile "
            "the shared leaf channels. The default retains the complete "
            "numerical rank and therefore reconstructs the teacher."
        ),
    )
    parser.add_argument(
        "--maximum-leaf-rank",
        type=int,
        help="Optional hard cap for the compiled shared-leaf rank.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_points(
    path: Path,
    *,
    maximum_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    if "complex_points" in payload:
        points = np.asarray(payload["complex_points"], dtype=np.complex128)
        weights = np.asarray(payload["weights"], dtype=np.float64)
    elif "X_train" in payload:
        x_values = np.asarray(payload["X_train"], dtype=np.float64)
        if x_values.ndim != 2 or x_values.shape[1] % 2:
            raise ValueError("real-concatenated projective points are malformed")
        coordinate_count = x_values.shape[1] // 2
        points = (
            x_values[:, :coordinate_count]
            + 1j * x_values[:, coordinate_count:]
        )
        labels = np.asarray(payload["y_train"], dtype=np.float64)
        weights = labels[:, 0]
    else:
        raise ValueError("unsupported point-pool schema")
    if len(points) != len(weights) or len(points) == 0:
        raise ValueError("point pool and weights are not aligned")
    if maximum_count <= 0:
        raise ValueError("Gram point count must be positive")
    if maximum_count < len(points):
        indices = np.random.default_rng(seed).choice(
            len(points),
            size=maximum_count,
            replace=False,
        )
        points = points[indices]
        weights = weights[indices]
    if (
        not np.all(np.isfinite(points))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("point pool contains invalid values")
    return points, weights


def positive_factor(hermitian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(hermitian, dtype=np.complex128)
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.min(eigenvalues) <= 0 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("teacher H must be positive definite")
    factor = np.sqrt(eigenvalues)[:, None] * eigenvectors.conj().T
    np.testing.assert_allclose(
        factor.conj().T @ factor,
        matrix,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    return factor, eigenvalues


def spectrum_summary(singular_values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(singular_values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "leading": values[:32].tolist(),
        "rank_for_cumulative_squared_weight": {
            f"{threshold:.12g}": rank_for_squared_weight(values, threshold)
            for threshold in (0.9, 0.99, 0.999, 0.9999, 0.99999)
        },
    }


def main() -> None:
    args = parse_args()
    teacher_path = args.teacher_h.expanduser().resolve()
    points_path = args.points.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a compiler run")
    output_dir.mkdir(parents=True)

    teacher = np.load(teacher_path, allow_pickle=False)
    degree = int(teacher["degree"])
    if degree != 4:
        raise ValueError("the first compiler experiment requires a degree-4 H")
    teacher_h = np.asarray(teacher["global_h_matrix"], dtype=np.complex128)
    teacher_h = 0.5 * (teacher_h + teacher_h.conj().T)
    source_normalization = float(teacher["normalization"])
    teacher_exponents = np.asarray(teacher["exponents"], dtype=np.int64)
    ring = HypersurfaceSectionRing(
        GENERIC_QUINTIC_EXPONENTS,
        GENERIC_QUINTIC_COEFFICIENTS,
        pivot_coordinate=args.pivot_coordinate,
    )
    multiplication = ring.multiplication_map(2, 2)
    if not np.array_equal(teacher_exponents, multiplication.output_exponents):
        raise ValueError("teacher H basis differs from the quotient-ring basis")

    points, weights = load_points(
        points_path,
        maximum_count=args.gram_points,
        seed=args.seed,
    )
    degree2_values = evaluate_monomials(points, multiplication.left_exponents)
    degree4_values = evaluate_monomials(points, multiplication.output_exponents)
    degree2_gram = section_covariance(degree2_values, weights)
    degree4_gram = section_covariance(degree4_values, weights)
    minimum_lift = minimum_norm_multiplication_lift(
        multiplication,
        left_gram=degree2_gram,
        right_gram=degree2_gram,
        output_gram=degree4_gram,
        relative_eigenvalue_floor=args.relative_eigenvalue_floor,
        relative_singular_value_tolerance=args.relative_singular_value_tolerance,
    )
    whitened_teacher_h = (
        minimum_lift.output_whitening.unwhitening.conj().T
        @ teacher_h
        @ minimum_lift.output_whitening.unwhitening
    )
    whitened_teacher_h = 0.5 * (
        whitened_teacher_h + whitened_teacher_h.conj().T
    )
    whitened_trace_scale = float(
        len(whitened_teacher_h) / np.trace(whitened_teacher_h).real
    )
    whitened_teacher_h = whitened_trace_scale * whitened_teacher_h
    whitened_teacher_eigenvalues = np.linalg.eigvalsh(whitened_teacher_h)
    whitened_factor, _ = positive_factor(whitened_teacher_h)
    right_inverse = minimum_lift.raw_right_inverse
    factor, teacher_eigenvalues = positive_factor(teacher_h)
    lifted_factor = lift_purification_factor(factor, right_inverse)
    lifted_h = lift_hermitian_form(teacher_h, right_inverse)
    factor_error = float(
        np.linalg.norm(lifted_factor.conj().T @ lifted_factor - lifted_h)
        / np.linalg.norm(lifted_h)
    )

    products = np.einsum(
        "bi,bj->bij",
        degree2_values,
        degree2_values,
    ).reshape(len(points), -1)
    reconstructed_sections = products @ right_inverse
    section_error = float(
        np.linalg.norm(reconstructed_sections - degree4_values)
        / np.linalg.norm(degree4_values)
    )
    teacher_norm = np.einsum(
        "bi,ij,bj->b",
        degree4_values.conj(),
        teacher_h,
        degree4_values,
        optimize=True,
    ).real
    lifted_norm = np.einsum(
        "bi,ij,bj->b",
        products.conj(),
        lifted_h,
        products,
        optimize=True,
    ).real
    function_error = float(
        np.linalg.norm(lifted_norm - teacher_norm)
        / np.linalg.norm(teacher_norm)
    )

    operator_spectrum = operator_schmidt_singular_values(
        lifted_h,
        left_count=multiplication.left_count,
        right_count=multiplication.right_count,
    )
    factor_tensor = lifted_factor.reshape(
        len(factor),
        multiplication.left_count,
        multiplication.right_count,
    )
    purification_left_spectrum = np.linalg.svd(
        factor_tensor.transpose(1, 0, 2).reshape(
            multiplication.left_count,
            len(factor) * multiplication.right_count,
        ),
        compute_uv=False,
    )
    purification_right_spectrum = np.linalg.svd(
        factor_tensor.transpose(2, 0, 1).reshape(
            multiplication.right_count,
            len(factor) * multiplication.left_count,
        ),
        compute_uv=False,
    )
    compiled_channels = purification_schmidt_channels(
        whitened_factor,
        minimum_lift.whitened_right_inverse,
        minimum_lift.whitened_multiplication,
        left_count=multiplication.left_count,
        right_count=multiplication.right_count,
        section_gram=np.eye(len(whitened_teacher_h), dtype=np.complex128),
        squared_weight_retention=(
            args.purification_squared_weight_retention
        ),
        maximum_rank=args.maximum_leaf_rank,
        relative_singular_value_tolerance=(
            args.relative_singular_value_tolerance
        ),
    )
    topology = balanced_binary_topology(5)
    compiled_edge_dimensions = np.asarray(
        [compiled_channels.retained_rank] * topology.leaf_count
        + [1] * (topology.root - topology.leaf_count),
        dtype=np.int64,
    )
    if (
        args.purification_squared_weight_retention == 1.0
        and args.maximum_leaf_rank is None
        and compiled_channels.hermitian_relative_error > 5.0e-11
    ):
        raise FloatingPointError(
            "full-rank compiled channels do not reconstruct the teacher H"
        )

    gauges = {
        "whitened-minimum-norm": right_inverse,
        "canonical-exact": exact_monomial_right_inverse(
            multiplication,
            mode="canonical",
        ),
        "uniform-exact": exact_monomial_right_inverse(
            multiplication,
            mode="uniform",
        ),
    }
    gauge_report = lift_gauge_diagnostics(
        teacher_h,
        multiplication,
        gauges,
    )
    artifact_path = output_dir / "compiled_h4_2plus2.npz"
    np.savez_compressed(
        artifact_path,
        schema=np.asarray("generic-quintic-full-h-tree-compiler-v2"),
        degree=np.asarray(degree, dtype=np.int64),
        source_normalization=np.asarray(source_normalization, dtype=np.float64),
        target_degree=np.asarray(20, dtype=np.int64),
        power=np.asarray(5, dtype=np.int64),
        teacher_h=teacher_h,
        whitened_teacher_h=whitened_teacher_h,
        teacher_factor=factor,
        degree2_exponents=multiplication.left_exponents,
        degree4_exponents=multiplication.output_exponents,
        multiplication_data=multiplication.matrix.data,
        multiplication_indices=multiplication.matrix.indices,
        multiplication_indptr=multiplication.matrix.indptr,
        multiplication_shape=np.asarray(multiplication.matrix.shape, dtype=np.int64),
        degree2_gram=degree2_gram,
        degree4_gram=degree4_gram,
        degree2_whitening=minimum_lift.left_whitening.whitening,
        degree2_unwhitening=minimum_lift.left_whitening.unwhitening,
        degree4_whitening=minimum_lift.output_whitening.whitening,
        degree4_unwhitening=minimum_lift.output_whitening.unwhitening,
        whitened_multiplication=minimum_lift.whitened_multiplication,
        whitened_right_inverse=minimum_lift.whitened_right_inverse,
        raw_right_inverse=right_inverse,
        lifted_factor=lifted_factor,
        lifted_hermitian=lifted_h,
        operator_schmidt_singular_values=operator_spectrum,
        purification_left_singular_values=purification_left_spectrum,
        purification_right_singular_values=purification_right_spectrum,
        compiled_schmidt_orientation=np.asarray(
            compiled_channels.orientation
        ),
        compiled_schmidt_singular_values=(
            compiled_channels.singular_values
        ),
        compiled_selected_singular_indices=(
            compiled_channels.selected_singular_indices
        ),
        compiled_leaf_tensor=compiled_channels.leaf_tensor,
        compiled_leaf_combiner=compiled_channels.leaf_combiner,
        compiled_reconstructed_factor=(
            compiled_channels.reconstructed_factor
        ),
        compiled_edge_dimensions=compiled_edge_dimensions,
        compiled_topology_children=np.asarray(
            topology.children,
            dtype=np.int64,
        ),
    )
    report = {
        "schema": "generic-quintic-full-h-tree-compiler-v2",
        "source": {
            "teacher_h": str(teacher_path),
            "teacher_h_sha256": sha256_file(teacher_path),
            "points": str(points_path),
            "points_sha256": sha256_file(points_path),
            "gram_point_count": len(points),
        },
        "geometry": {
            "pivot_coordinate": args.pivot_coordinate,
            "split": [2, 2],
            "degree2_section_count": multiplication.left_count,
            "degree4_section_count": multiplication.output_count,
            "target_degree": 20,
            "power": 5,
            "multiplication_shape": list(multiplication.matrix.shape),
            "multiplication_nonzero_count": int(multiplication.matrix.nnz),
        },
        "whitening": {
            "degree2_condition_number": (
                minimum_lift.left_whitening.condition_number
            ),
            "degree4_condition_number": (
                minimum_lift.output_whitening.condition_number
            ),
            "degree2_identity_error": (
                minimum_lift.left_whitening.identity_error()
            ),
            "degree4_identity_error": (
                minimum_lift.output_whitening.identity_error()
            ),
            "multiplication_singular_values": (
                minimum_lift.singular_values.tolist()
            ),
            "multiplication_condition_number": float(
                minimum_lift.singular_values[0]
                / minimum_lift.singular_values[-1]
            ),
        },
        "exactness": {
            "right_inverse_error": minimum_lift.right_inverse_error(),
            "whitened_right_inverse_error": (
                minimum_lift.whitened_right_inverse_error()
            ),
            "section_reconstruction_relative_error": section_error,
            "quadratic_function_relative_error": function_error,
            "purification_factor_relative_error": factor_error,
        },
        "teacher_h": {
            "whitened_trace_scale": whitened_trace_scale,
            "minimum_eigenvalue": float(teacher_eigenvalues[0]),
            "maximum_eigenvalue": float(teacher_eigenvalues[-1]),
            "condition_number": float(
                teacher_eigenvalues[-1] / teacher_eigenvalues[0]
            ),
            "whitened_minimum_eigenvalue": float(
                whitened_teacher_eigenvalues[0]
            ),
            "whitened_maximum_eigenvalue": float(
                whitened_teacher_eigenvalues[-1]
            ),
            "whitened_condition_number": float(
                whitened_teacher_eigenvalues[-1]
                / whitened_teacher_eigenvalues[0]
            ),
        },
        "spectra": {
            "operator": spectrum_summary(operator_spectrum),
            "purification_left": spectrum_summary(purification_left_spectrum),
            "purification_right": spectrum_summary(purification_right_spectrum),
        },
        "compiled_tree": {
            "topology_children": [list(pair) for pair in topology.children],
            "edge_dimensions": compiled_edge_dimensions.tolist(),
            "schmidt_orientation": compiled_channels.orientation,
            "numerical_leaf_rank": compiled_channels.numerical_rank,
            "retained_leaf_rank": compiled_channels.retained_rank,
            "requested_squared_weight_retention": (
                args.purification_squared_weight_retention
            ),
            "retained_squared_weight": (
                compiled_channels.retained_squared_weight
            ),
            "lifted_factor_relative_error": (
                compiled_channels.lifted_factor_relative_error
            ),
            "factor_relative_error": (
                compiled_channels.factor_relative_error
            ),
            "hermitian_relative_error": (
                compiled_channels.hermitian_relative_error
            ),
            "round_zero_activation": (
                "deterministic teacher Schmidt channels; no random dormant "
                "channel activation"
            ),
        },
        "lift_gauge_diagnostics": gauge_report,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "interpretation": {
            "bottom_layer_source": (
                "degree-4 full-H purification Schmidt channels compiled "
                "through whitened mu_2,2"
            ),
            "leaf_rank_source": "teacher purification Schmidt spectrum",
            "upper_tree_initial_rank": 1,
            "upper_tree_rank_source_after_round_zero": (
                "native E2 residual only; the H4 power lift supplies no upper-rank data"
            ),
            "teacher_runtime_dependency_after_compilation": False,
        },
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
