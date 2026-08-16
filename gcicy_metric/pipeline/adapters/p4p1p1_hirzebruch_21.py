"""Adapter for a smooth type-(2,1) presentation of Hirzebruch X3."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...type21_hirzebruch_x3 import (
    FACTOR_DIMENSIONS,
    HirzebruchType21Model,
    HirzebruchType21Point,
    hirzebruch_type21_baseline_metric,
    hirzebruch_type21_global_h_metric,
    hirzebruch_type21_importance_log_weight,
    hirzebruch_type21_monge_ampere_log_error,
    hirzebruch_type21_point_in_chart,
    hirzebruch_type21_restricted_ambient_basis,
    hirzebruch_type21_restricted_fubini_study_h_matrix,
    hirzebruch_type21_section_values_and_jacobian,
    make_hirzebruch_type21_model,
    retract_hirzebruch_type21_intrinsic_step,
    sample_hirzebruch_type21_points,
    sample_hirzebruch_type21_points_with_diagnostics,
)
from ..adapter import (
    Chart,
    GCICYAdapter,
    HMetricArtifact,
    IndependentCoordinates,
    positive_hermitian_projection,
)
from ..schema import GCICYConfiguration


class P4P1P1HirzebruchType21Adapter(GCICYAdapter):
    key = "p4p1p1_type21_hirzebruch_x3"
    version = "1"
    configuration = GCICYConfiguration(
        key=key,
        name="Smooth type-(2,1) control presentation of Hirzebruch X3",
        ambient_dimensions=(4, 1, 1),
        positive_columns=((1, 3, 0), (0, 0, 1)),
        generalized_columns=((4, -1, 1),),
        kahler_line_bundle=(1, 1, 1),
        source=(
            "Equivalent type-(2,1) stabilization of arXiv:1606.07420, "
            "Eq. (1.1), m=3"
        ),
    )

    def make_model(self, seed: int, *, exact: bool) -> HirzebruchType21Model:
        return make_hirzebruch_type21_model(seed, exact=exact)

    def model_metadata(self, model: HirzebruchType21Model) -> dict[str, Any]:
        return {
            "seed": int(model.seed),
            "p1_term_count": int(model.p_tensor.size),
            "generalized_section_parameter_count": int(
                model.q_cubic_coefficients.size
            ),
            "generalized_section_space_dimension": 105,
            "equivalent_type_11_configuration": [[1, 4], [3, -1]],
            "reported_hodge_numbers": [2, 86],
        }

    def sample_points(
        self,
        model: HirzebruchType21Model,
        count: int,
        *,
        seed: int,
    ) -> list[HirzebruchType21Point]:
        return sample_hirzebruch_type21_points(model, count, seed=seed)

    def default_sampling_options(self) -> dict[str, float]:
        return {"root_separation_tolerance": 1.0e-7}

    def sample_points_configured(
        self,
        model: HirzebruchType21Model,
        count: int,
        *,
        seed: int,
        sampling_options: dict[str, Any] | None = None,
    ) -> list[HirzebruchType21Point]:
        options = dict(sampling_options or self.default_sampling_options())
        unknown = set(options) - {"root_separation_tolerance"}
        if unknown:
            raise ValueError(f"unsupported sampling options: {sorted(unknown)}")
        return sample_hirzebruch_type21_points(
            model,
            count,
            seed=seed,
            root_separation_tolerance=float(options["root_separation_tolerance"]),
        )

    def sample_points_with_diagnostics(
        self,
        model: HirzebruchType21Model,
        count: int,
        *,
        seed: int,
    ) -> tuple[list[HirzebruchType21Point], dict[str, object]]:
        return sample_hirzebruch_type21_points_with_diagnostics(
            model,
            count,
            seed=seed,
        )

    def retract_intrinsic_step(
        self,
        model: HirzebruchType21Model,
        center: HirzebruchType21Point,
        intrinsic_step: np.ndarray,
        *,
        residual_tolerance: float = 1e-10,
        maximum_newton_iterations: int = 16,
    ) -> tuple[HirzebruchType21Point, dict[str, Any]]:
        return retract_hirzebruch_type21_intrinsic_step(
            model,
            center,
            intrinsic_step,
            residual_tolerance=residual_tolerance,
            maximum_newton_iterations=maximum_newton_iterations,
        )

    def sampling_cluster_ids(
        self,
        points: Sequence[HirzebruchType21Point],
    ) -> np.ndarray:
        return np.arange(len(points), dtype=np.int64) // 4

    def baseline_metric(self, point: HirzebruchType21Point) -> np.ndarray:
        return hirzebruch_type21_baseline_metric(point)

    def baseline_metrics(
        self,
        points: Sequence[HirzebruchType21Point],
    ) -> np.ndarray:
        if not points:
            return np.empty((0, 3, 3), dtype=np.complex128)
        affine = np.asarray(
            [point.affine_coordinates for point in points],
            dtype=np.complex128,
        )
        tangent = np.asarray(
            [point.tangent_basis for point in points],
            dtype=np.complex128,
        )
        point_count = len(points)
        ambient_dimension = sum(self.configuration.ambient_dimensions)
        if affine.shape != (point_count, ambient_dimension):
            # The direct type-(1,1) adapter retains one fixed auxiliary P1
            # internally, so use the tangent representation's actual size.
            ambient_dimension = tangent.shape[1]
        if affine.shape != (point_count, ambient_dimension):
            raise ValueError("unexpected batched affine-coordinate shape")
        if tangent.shape != (point_count, ambient_dimension, 3):
            raise ValueError("unexpected batched tangent-basis shape")

        ambient = np.zeros(
            (point_count, ambient_dimension, ambient_dimension),
            dtype=np.complex128,
        )
        cursor = 0
        internal_dimensions = (4, 1, 1)
        for dimension in internal_dimensions:
            block = slice(cursor, cursor + dimension)
            values = affine[:, block]
            rho = 1.0 + np.sum(np.abs(values) ** 2, axis=1)
            identity = np.eye(dimension, dtype=np.complex128)[None, :, :]
            numerator = rho[:, None, None] * identity
            numerator -= values[:, :, None] * np.conjugate(values[:, None, :])
            ambient[:, block, block] = numerator / rho[:, None, None] ** 2
            cursor += dimension
        metrics = np.einsum(
            "nai,nab,nbj->nij",
            np.conjugate(tangent),
            ambient,
            tangent,
            optimize=True,
        )
        return 0.5 * (metrics + np.conjugate(np.swapaxes(metrics, 1, 2)))

    def importance_log_weight(self, point: HirzebruchType21Point) -> float:
        return hirzebruch_type21_importance_log_weight(point)

    def monge_ampere_log_error(
        self,
        point: HirzebruchType21Point,
        metric: np.ndarray,
    ) -> float:
        return hirzebruch_type21_monge_ampere_log_error(point, metric)

    def rechart_point(
        self,
        model: HirzebruchType21Model,
        point: HirzebruchType21Point,
        chart: Chart,
        *,
        independent: IndependentCoordinates | None = None,
    ) -> HirzebruchType21Point:
        if len(chart) != 3:
            raise ValueError("the Hirzebruch type-(2,1) adapter requires three chart entries")
        selected_chart = tuple(int(value) for value in chart)
        selected_independent = (
            tuple(int(value) for value in independent)
            if independent is not None
            else None
        )
        return hirzebruch_type21_point_in_chart(
            model,
            point.coordinates,
            selected_chart,  # type: ignore[arg-type]
            independent=selected_independent,  # type: ignore[arg-type]
        )

    def chart_is_available(
        self,
        point: HirzebruchType21Point,
        chart: Chart,
        *,
        minimum: float,
    ) -> bool:
        if len(chart) != 3:
            return False
        selected = [
            values[int(index)]
            for values, index in zip(point.coordinates, chart, strict=True)
        ]
        return min(abs(value) for value in selected) > minimum

    def point_projective_chart(self, point: HirzebruchType21Point) -> Chart:
        return tuple(int(value) for value in point.projective_chart)

    def point_jacobian_min_singular_value(
        self,
        point: HirzebruchType21Point,
    ) -> float:
        return float(point.jacobian_min_singular_value)

    def implicit_coordinate_choices(self):
        # The final affine coordinate belongs to the P1 fixed by p2=0 and
        # cannot be an intrinsic independent coordinate.
        return combinations(range(5), 3)

    def expected_implicit_coordinate_choice_count(self) -> int:
        return 10

    def load_h_artifact(
        self,
        path: Path,
        model: HirzebruchType21Model,
    ) -> HMetricArtifact:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"missing H-metric artifact: {resolved}")
        with np.load(resolved, allow_pickle=False) as payload:
            required = {
                "pipeline_adapter",
                "hirzebruch_type21_model_seed",
                "hirzebruch_type21_p_tensor",
                "hirzebruch_type21_q_cubic_coefficients",
                "hirzebruch_type21_point_equation",
                "hirzebruch_type21_point_section",
                "global_section_degree",
                "global_section_exponents",
                "global_h_matrix",
                "global_section_normalization",
            }
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"artifact is missing fields: {', '.join(missing)}")
            if str(payload["pipeline_adapter"]) != self.key:
                raise ValueError("artifact was produced by a different adapter")
            if int(payload["hirzebruch_type21_model_seed"]) != int(model.seed):
                raise ValueError("artifact model seed does not match the requested model")
            comparisons = (
                ("p", payload["hirzebruch_type21_p_tensor"], model.p_tensor),
                (
                    "q",
                    payload["hirzebruch_type21_q_cubic_coefficients"],
                    model.q_cubic_coefficients,
                ),
                (
                    "point equation",
                    payload["hirzebruch_type21_point_equation"],
                    model.point_equation,
                ),
                (
                    "point section",
                    payload["hirzebruch_type21_point_section"],
                    model.point_section,
                ),
            )
            for label, candidate, expected in comparisons:
                if not np.allclose(candidate, expected):
                    raise ValueError(
                        f"artifact {label} coefficients do not match the model"
                    )
            degree = tuple(int(value) for value in payload["global_section_degree"])
            if len(degree) != len(self.configuration.ambient_dimensions):
                raise ValueError("artifact section degree does not match the ambient factors")
            return HMetricArtifact(
                path=resolved,
                degree=degree,
                section_exponents=np.asarray(
                    payload["global_section_exponents"],
                    dtype=np.int64,
                ).copy(),
                h_matrix=np.asarray(
                    payload["global_h_matrix"],
                    dtype=np.complex128,
                ).copy(),
                normalization=float(payload["global_section_normalization"]),
                metadata={
                    "model_seed": int(payload["hirzebruch_type21_model_seed"]),
                    "exact_integer_coefficients": bool(
                        payload["exact_integer_coefficients"]
                    )
                    if "exact_integer_coefficients" in payload.files
                    else None,
                    "importance_weighted_training": bool(
                        payload["importance_weighted_training"]
                    )
                    if "importance_weighted_training" in payload.files
                    else None,
                },
            )

    def h_metric(
        self,
        point: HirzebruchType21Point,
        artifact: HMetricArtifact,
    ) -> np.ndarray:
        return hirzebruch_type21_global_h_metric(
            point,
            artifact.section_exponents,
            artifact.h_matrix,
            normalization=artifact.normalization,
        )

    def h_metrics(
        self,
        points: Sequence[HirzebruchType21Point],
        artifact: HMetricArtifact,
    ) -> np.ndarray:
        if not points:
            return np.empty((0, 3, 3), dtype=np.complex128)
        evaluated = [
            self.section_values_and_jacobian(point, artifact.section_exponents)
            for point in points
        ]
        values = np.asarray([item[0] for item in evaluated], dtype=np.complex128)
        derivatives = np.asarray([item[1] for item in evaluated], dtype=np.complex128)
        h_matrix = np.asarray(artifact.h_matrix, dtype=np.complex128)
        h_values = np.einsum("ab,nb->na", h_matrix, values, optimize=True)
        denominator = np.real(
            np.einsum("na,na->n", np.conjugate(values), h_values, optimize=True)
        )
        if np.any(denominator <= 0):
            raise FloatingPointError("H-metric denominator is not positive")
        h_derivatives = np.einsum(
            "ab,nbj->naj",
            h_matrix,
            derivatives,
            optimize=True,
        )
        first = np.einsum(
            "nmi,nmj->nij",
            np.conjugate(derivatives),
            h_derivatives,
            optimize=True,
        )
        gradient = np.einsum(
            "nm,nmj->nj",
            np.conjugate(values),
            h_derivatives,
            optimize=True,
        )
        metrics = first / denominator[:, None, None]
        metrics -= (
            np.conjugate(gradient)[:, :, None]
            * gradient[:, None, :]
            / denominator[:, None, None] ** 2
        )
        metrics *= artifact.normalization
        return 0.5 * (metrics + np.conjugate(np.swapaxes(metrics, 1, 2)))

    def restricted_section_basis(
        self,
        points: Sequence[HirzebruchType21Point],
        degree: tuple[int, ...],
    ) -> Any:
        return hirzebruch_type21_restricted_ambient_basis(points, degree)

    def restricted_fubini_study_h_matrix(
        self,
        points: Sequence[HirzebruchType21Point],
        basis: Any,
    ) -> tuple[np.ndarray, float]:
        return hirzebruch_type21_restricted_fubini_study_h_matrix(points, basis)

    def section_values_and_jacobian(
        self,
        point: HirzebruchType21Point,
        exponents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return hirzebruch_type21_section_values_and_jacobian(point, exponents)

    def section_values_and_jacobian_batch(
        self,
        points: Sequence[HirzebruchType21Point],
        exponents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorize monomial jets over points sharing a projective chart."""

        powers = np.asarray(exponents, dtype=np.int64)
        point_count = len(points)
        section_count = len(powers)
        if point_count == 0:
            return (
                np.empty((0, section_count), dtype=np.complex128),
                np.empty((0, section_count, 3), dtype=np.complex128),
            )
        homogeneous_dimension = sum(
            dimension + 1 for dimension in FACTOR_DIMENSIONS
        )
        if powers.ndim != 2 or powers.shape[1] != homogeneous_dimension:
            raise ValueError("unexpected exponent shape")

        values = np.empty((point_count, section_count), dtype=np.complex128)
        derivatives = np.empty(
            (point_count, section_count, 3),
            dtype=np.complex128,
        )
        grouped_indices: dict[tuple[int, ...], list[int]] = {}
        for index, point in enumerate(points):
            chart = tuple(int(value) for value in point.projective_chart)
            grouped_indices.setdefault(chart, []).append(index)

        factor_sizes = tuple(
            dimension + 1 for dimension in FACTOR_DIMENSIONS
        )
        factor_offsets = np.cumsum((0, *factor_sizes))
        for chart, indices in grouped_indices.items():
            affine = np.asarray(
                [points[index].affine_coordinates for index in indices],
                dtype=np.complex128,
            )
            homogeneous = np.empty(
                (len(indices), homogeneous_dimension),
                dtype=np.complex128,
            )
            affine_cursor = 0
            for factor, (start, stop) in enumerate(
                zip(
                    factor_offsets[:-1],
                    factor_offsets[1:],
                    strict=True,
                )
            ):
                size = int(stop - start)
                selected = chart[factor]
                homogeneous[:, int(start) + selected] = 1.0
                active = np.arange(int(start), int(stop)) != int(start) + selected
                homogeneous[:, np.arange(int(start), int(stop))[active]] = (
                    affine[:, affine_cursor : affine_cursor + size - 1]
                )
                affine_cursor += size - 1
            group_values = np.prod(
                homogeneous[:, None, :] ** powers[None, :, :],
                axis=2,
            )
            active_indices = []
            for factor, (start, stop) in enumerate(
                zip(
                    factor_offsets[:-1],
                    factor_offsets[1:],
                    strict=True,
                )
            ):
                active_indices.extend(
                    index
                    for index in range(int(start), int(stop))
                    if index - int(start) != chart[factor]
                )
            ambient_jacobian = np.empty(
                (len(indices), section_count, len(active_indices)),
                dtype=np.complex128,
            )
            for column, homogeneous_index in enumerate(active_indices):
                reduced = powers.copy()
                reduced[:, homogeneous_index] = np.maximum(
                    reduced[:, homogeneous_index] - 1,
                    0,
                )
                ambient_jacobian[:, :, column] = (
                    powers[None, :, homogeneous_index]
                    * np.prod(
                        homogeneous[:, None, :] ** reduced[None, :, :],
                        axis=2,
                    )
                )
            tangent = np.asarray(
                [points[index].tangent_basis for index in indices],
                dtype=np.complex128,
            )
            intrinsic = np.einsum(
                "nsa,naj->nsj",
                ambient_jacobian,
                tangent,
                optimize=True,
            )
            values[indices] = group_values
            derivatives[indices] = intrinsic
        return values, derivatives

    def artifact_model_payload(
        self,
        model: HirzebruchType21Model,
        *,
        exact: bool,
    ) -> dict[str, np.ndarray]:
        return {
            "hirzebruch_type21_model_seed": np.asarray(model.seed),
            "exact_integer_coefficients": np.asarray(exact),
            "hirzebruch_type21_p_tensor": np.asarray(
                model.p_tensor,
                dtype=np.complex128,
            ),
            "hirzebruch_type21_cubic_exponents": np.asarray(
                model.cubic_exponents,
                dtype=np.int64,
            ),
            "hirzebruch_type21_q_cubic_coefficients": np.asarray(
                model.q_cubic_coefficients,
                dtype=np.complex128,
            ),
            "hirzebruch_type21_point_equation": np.asarray(
                model.point_equation,
                dtype=np.complex128,
            ),
            "hirzebruch_type21_point_section": np.asarray(
                model.point_section,
                dtype=np.complex128,
            ),
            "hirzebruch_type21_p1_exponents": np.asarray(
                model.p1_exponents,
                dtype=np.int64,
            ),
            "hirzebruch_type21_p1_coefficients": np.asarray(
                model.p1_coefficients,
                dtype=np.complex128,
            ),
            "hirzebruch_type21_p2_exponents": np.asarray(
                model.p2_exponents,
                dtype=np.int64,
            ),
            "hirzebruch_type21_p2_coefficients": np.asarray(
                model.p2_coefficients,
                dtype=np.complex128,
            ),
        }

    def lift_h_matrix_power(
        self,
        points: Sequence[HirzebruchType21Point],
        source: HMetricArtifact,
        target_basis: Any,
    ) -> tuple[np.ndarray, float]:
        source_power = self.configuration.kahler_power(tuple(source.degree))
        target_power = self.configuration.kahler_power(tuple(target_basis.degree))
        if target_power % source_power != 0:
            raise NotImplementedError(
                "target degree must be an integer power of the source degree"
            )
        power_ratio = target_power // source_power
        if power_ratio not in (1, 2, 3):
            raise NotImplementedError(
                "only same-degree, quadratic, and cubic lifts are implemented"
            )
        source_values = np.asarray(
            [
                self.section_values_and_jacobian(
                    point,
                    source.section_exponents,
                )[0]
                for point in points
            ],
            dtype=np.complex128,
        )
        product_values = source_values
        source_h = positive_hermitian_projection(source.h_matrix)
        product_h = source_h
        for _ in range(power_ratio - 1):
            product_values = np.einsum(
                "ni,nj->nij",
                product_values,
                source_values,
            ).reshape(len(points), -1)
            product_h = np.kron(product_h, source_h)
        target_values = np.asarray(
            [
                self.section_values_and_jacobian(
                    point,
                    target_basis.selected_exponents,
                )[0]
                for point in points
            ],
            dtype=np.complex128,
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            target_values,
            product_values,
            rcond=1e-10,
        )
        relation_error = float(
            np.linalg.norm(target_values @ coefficients - product_values)
            / max(1.0, np.linalg.norm(product_values))
        )
        target_h = np.conjugate(coefficients) @ product_h @ coefficients.T
        return positive_hermitian_projection(target_h), relation_error
