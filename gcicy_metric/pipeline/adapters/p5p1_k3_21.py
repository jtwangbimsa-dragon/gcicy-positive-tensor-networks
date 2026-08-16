"""Adapter for the smooth K3-fibered P5 x P1 type-(2,1) model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...type21_candidate_p5p1_1223 import (
    P5P1Type21Candidate1223Model,
    P5P1Type21Point,
    make_p5p1_type21_candidate_1223_model,
    p5p1_type21_baseline_metric,
    p5p1_type21_global_h_metric,
    p5p1_type21_importance_log_weight,
    p5p1_type21_monge_ampere_log_error,
    p5p1_type21_point_in_chart,
    p5p1_type21_restricted_ambient_basis,
    p5p1_type21_restricted_fubini_study_h_matrix,
    p5p1_type21_section_values_and_jacobian,
    sample_p5p1_type21_points,
    sample_p5p1_type21_points_with_diagnostics,
)
from ...product_projective import choose_product_chart
from ..adapter import (
    Chart,
    GCICYAdapter,
    HMetricArtifact,
    IndependentCoordinates,
    positive_hermitian_projection,
)
from ..schema import GCICYConfiguration
from ..spectrum import ScalarTrialData, product_projective_moment_map_trial_data


class P5P1K3Type21Adapter(GCICYAdapter):
    key = "p5p1_type21_k3_1223"
    version = "1"
    configuration = GCICYConfiguration(
        key=key,
        name="Smooth K3-fibered P5 x P1 type-(2,1) threefold",
        ambient_dimensions=(5, 1),
        positive_columns=((1, 1), (2, 2)),
        generalized_columns=((3, -1),),
        kahler_line_bundle=(1, 1),
        source=(
            "P5 x P1 codimension-(2,1) scan class of arXiv:1507.03235 "
            "and arXiv:2209.10157"
        ),
    )

    def make_model(
        self,
        seed: int,
        *,
        exact: bool,
    ) -> P5P1Type21Candidate1223Model:
        return make_p5p1_type21_candidate_1223_model(seed, exact=exact)

    def model_metadata(
        self,
        model: P5P1Type21Candidate1223Model,
    ) -> dict[str, Any]:
        return {
            "seed": int(model.seed),
            "exact_integer_coefficients": bool(model.exact_coefficients),
            "p1_term_count": int(len(model.p1_coefficients)),
            "p2_term_count": int(len(model.p2_coefficients)),
            "generalized_section_parameter_count_before_koszul_quotient": int(
                len(model.r_a_coefficients) + model.r_b_coefficients.size
            ),
            "generalized_section_space_dimension": 30,
            "euler_characteristic": -100,
            "expected_hodge_numbers": [2, 52],
            "generic_fiber": "complete-intersection K3 of degrees (2,3) in P4",
            "characteristic_zero_smoothness": (
                "proved for seed 20260802 by complete good reduction"
                if model.seed == 20260802 and model.exact_coefficients
                else "not covered by the stored good-reduction certificate"
            ),
        }

    def sample_points(
        self,
        model: P5P1Type21Candidate1223Model,
        count: int,
        *,
        seed: int,
    ) -> list[P5P1Type21Point]:
        return sample_p5p1_type21_points(model, count, seed=seed)

    def sample_points_with_diagnostics(
        self,
        model: P5P1Type21Candidate1223Model,
        count: int,
        *,
        seed: int,
    ) -> tuple[list[P5P1Type21Point], dict[str, object]]:
        return sample_p5p1_type21_points_with_diagnostics(
            model,
            count,
            seed=seed,
        )

    def sampling_cluster_ids(
        self,
        points: Sequence[P5P1Type21Point],
    ) -> np.ndarray:
        return np.arange(len(points), dtype=np.int64) // 6

    def point_storage_payload(
        self,
        points: Sequence[P5P1Type21Point],
    ) -> dict[str, np.ndarray]:
        if not points:
            raise ValueError("point storage payload cannot be empty")
        return {
            "coordinates_x": np.stack([point.x for point in points]),
            "coordinates_y": np.stack([point.y for point in points]),
            "projective_charts": np.asarray(
                [point.projective_chart for point in points],
                dtype=np.int64,
            ),
            "independent_indices": np.asarray(
                [point.independent_indices for point in points],
                dtype=np.int64,
            ),
        }

    def points_from_storage_payload(
        self,
        model: P5P1Type21Candidate1223Model,
        payload: Mapping[str, np.ndarray],
    ) -> list[P5P1Type21Point]:
        coordinate_fields = {"coordinates_x", "coordinates_y"}
        exact_chart_fields = {"projective_charts", "independent_indices"}
        if not coordinate_fields.issubset(payload):
            raise ValueError("type-(2,1) point payload has unexpected fields")
        optional_fields = set(payload) - coordinate_fields
        if optional_fields not in (set(), exact_chart_fields):
            raise ValueError("type-(2,1) point payload has unexpected fields")
        x = np.asarray(payload["coordinates_x"], dtype=np.complex128)
        y = np.asarray(payload["coordinates_y"], dtype=np.complex128)
        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError("stored P5 coordinates have an unexpected shape")
        if y.shape != (len(x), 2):
            raise ValueError("stored P1 coordinates have an unexpected shape")
        charts = (
            np.asarray(payload["projective_charts"], dtype=np.int64)
            if exact_chart_fields.issubset(payload)
            else None
        )
        independent = (
            np.asarray(payload["independent_indices"], dtype=np.int64)
            if exact_chart_fields.issubset(payload)
            else None
        )
        if charts is not None and charts.shape != (len(x), 2):
            raise ValueError("stored projective charts have an unexpected shape")
        if independent is not None and independent.shape != (len(x), 3):
            raise ValueError("stored independent indices have an unexpected shape")
        points = []
        for index in range(len(x)):
            coordinates = (x[index], y[index])
            chart = (
                tuple(int(value) for value in charts[index])
                if charts is not None
                else choose_product_chart(coordinates)
            )
            points.append(
                p5p1_type21_point_in_chart(
                    model,
                    coordinates,
                    chart,  # type: ignore[arg-type]
                    independent=(
                        tuple(int(value) for value in independent[index])
                        if independent is not None
                        else None
                    ),  # type: ignore[arg-type]
                )
            )
        return points

    def points_from_common_pool_payload(
        self,
        model: P5P1Type21Candidate1223Model,
        point_payload: Mapping[str, np.ndarray],
        geometry_payload: Mapping[str, np.ndarray],
    ) -> list[P5P1Type21Point]:
        """Replay frozen points without recomputing their local geometry."""

        del model
        x = np.asarray(point_payload["coordinates_x"], dtype=np.complex128)
        y = np.asarray(point_payload["coordinates_y"], dtype=np.complex128)
        charts = np.asarray(
            geometry_payload["projective_charts"],
            dtype=np.int64,
        )
        independent = np.asarray(
            geometry_payload["independent_indices"],
            dtype=np.int64,
        )
        dependent = np.asarray(
            geometry_payload["dependent_indices"],
            dtype=np.int64,
        )
        affine = np.asarray(
            geometry_payload["affine_coordinates"],
            dtype=np.complex128,
        )
        tangent = np.asarray(
            geometry_payload["tangent_basis"],
            dtype=np.complex128,
        )
        residue = np.asarray(
            geometry_payload["residue_denominator"],
            dtype=np.complex128,
        )
        minimum = np.asarray(
            geometry_payload["jacobian_min_singular_value"],
            dtype=np.float64,
        )
        count = len(x)
        if (
            x.shape != (count, 6)
            or y.shape != (count, 2)
            or charts.shape != (count, 2)
            or independent.shape != (count, 3)
            or dependent.shape != (count, 3)
            or affine.shape != (count, 6)
            or tangent.shape != (count, 6, 3)
            or residue.shape != (count,)
            or minimum.shape != (count,)
        ):
            raise ValueError("type-(2,1) common-pool geometry has invalid shapes")
        return [
            P5P1Type21Point(
                coordinates=(x[index], y[index]),
                projective_chart=tuple(int(value) for value in charts[index]),
                affine_coordinates=affine[index],
                independent_indices=tuple(
                    int(value) for value in independent[index]
                ),
                dependent_indices=tuple(
                    int(value) for value in dependent[index]
                ),
                tangent_basis=tangent[index],
                residue_denominator=complex(residue[index]),
                jacobian_min_singular_value=float(minimum[index]),
            )
            for index in range(count)
        ]

    def baseline_metric(self, point: P5P1Type21Point) -> np.ndarray:
        return p5p1_type21_baseline_metric(point)

    def importance_log_weight(self, point: P5P1Type21Point) -> float:
        return p5p1_type21_importance_log_weight(point)

    def monge_ampere_log_error(
        self,
        point: P5P1Type21Point,
        metric: np.ndarray,
    ) -> float:
        return p5p1_type21_monge_ampere_log_error(point, metric)

    def rechart_point(
        self,
        model: P5P1Type21Candidate1223Model,
        point: P5P1Type21Point,
        chart: Chart,
        *,
        independent: IndependentCoordinates | None = None,
    ) -> P5P1Type21Point:
        if len(chart) != 2:
            raise ValueError("the P5 x P1 adapter requires two chart entries")
        selected_chart = tuple(int(value) for value in chart)
        selected_independent = (
            tuple(int(value) for value in independent)
            if independent is not None
            else None
        )
        return p5p1_type21_point_in_chart(
            model,
            point.coordinates,
            selected_chart,  # type: ignore[arg-type]
            independent=selected_independent,  # type: ignore[arg-type]
        )

    def chart_is_available(
        self,
        point: P5P1Type21Point,
        chart: Chart,
        *,
        minimum: float,
    ) -> bool:
        if len(chart) != 2:
            return False
        selected = [
            values[int(index)]
            for values, index in zip(point.coordinates, chart, strict=True)
        ]
        return min(abs(value) for value in selected) > minimum

    def point_projective_chart(self, point: P5P1Type21Point) -> Chart:
        return tuple(int(value) for value in point.projective_chart)

    def point_jacobian_min_singular_value(
        self,
        point: P5P1Type21Point,
    ) -> float:
        return float(point.jacobian_min_singular_value)

    def load_h_artifact(
        self,
        path: Path,
        model: P5P1Type21Candidate1223Model,
    ) -> HMetricArtifact:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"missing H-metric artifact: {resolved}")
        with np.load(resolved, allow_pickle=False) as payload:
            required = {
                "pipeline_adapter",
                "p5p1_type21_model_seed",
                "p5p1_type21_a_tensor",
                "p5p1_type21_b_tensor",
                "p5p1_type21_r_a_coefficients",
                "p5p1_type21_r_b_coefficients",
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
            if int(payload["p5p1_type21_model_seed"]) != int(model.seed):
                raise ValueError("artifact model seed does not match the requested model")
            comparisons = (
                ("A", payload["p5p1_type21_a_tensor"], model.a_tensor),
                ("B", payload["p5p1_type21_b_tensor"], model.b_tensor),
                (
                    "Ra",
                    payload["p5p1_type21_r_a_coefficients"],
                    model.r_a_coefficients,
                ),
                (
                    "Rb",
                    payload["p5p1_type21_r_b_coefficients"],
                    model.r_b_coefficients,
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
                    "model_seed": int(payload["p5p1_type21_model_seed"]),
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
        point: P5P1Type21Point,
        artifact: HMetricArtifact,
    ) -> np.ndarray:
        return p5p1_type21_global_h_metric(
            point,
            artifact.section_exponents,
            artifact.h_matrix,
            normalization=artifact.normalization,
        )

    def h_metrics(
        self,
        points: Sequence[P5P1Type21Point],
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
        points: Sequence[P5P1Type21Point],
        degree: tuple[int, ...],
    ) -> Any:
        if len(degree) != 2:
            raise ValueError("the P5 x P1 section degree requires two entries")
        selected_degree = (int(degree[0]), int(degree[1]))
        return p5p1_type21_restricted_ambient_basis(points, selected_degree)

    def restricted_fubini_study_h_matrix(
        self,
        points: Sequence[P5P1Type21Point],
        basis: Any,
    ) -> tuple[np.ndarray, float]:
        return p5p1_type21_restricted_fubini_study_h_matrix(points, basis)

    def section_values_and_jacobian(
        self,
        point: P5P1Type21Point,
        exponents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return p5p1_type21_section_values_and_jacobian(point, exponents)

    def scalar_laplacian_trial_data(
        self,
        point: P5P1Type21Point,
        *,
        level: int,
        cubic_feature_count: int = 256,
        cubic_feature_seed: int = 314159,
    ) -> ScalarTrialData:
        return product_projective_moment_map_trial_data(
            point.affine_coordinates,
            point.projective_chart,
            point.tangent_basis,
            self.configuration.ambient_dimensions,
            level=level,
            cubic_feature_count=cubic_feature_count,
            cubic_feature_seed=cubic_feature_seed,
        )

    def artifact_model_payload(
        self,
        model: P5P1Type21Candidate1223Model,
        *,
        exact: bool,
    ) -> dict[str, np.ndarray]:
        return {
            "p5p1_type21_model_seed": np.asarray(model.seed),
            "exact_integer_coefficients": np.asarray(exact),
            "p5p1_type21_a_exponents": np.asarray(
                model.a_exponents,
                dtype=np.int64,
            ),
            "p5p1_type21_a_tensor": np.asarray(
                model.a_tensor,
                dtype=np.complex128,
            ),
            "p5p1_type21_b_exponents": np.asarray(
                model.b_exponents,
                dtype=np.int64,
            ),
            "p5p1_type21_b_tensor": np.asarray(
                model.b_tensor,
                dtype=np.complex128,
            ),
            "p5p1_type21_r_a_exponents": np.asarray(
                model.r_a_exponents,
                dtype=np.int64,
            ),
            "p5p1_type21_r_a_coefficients": np.asarray(
                model.r_a_coefficients,
                dtype=np.complex128,
            ),
            "p5p1_type21_r_b_exponents": np.asarray(
                model.r_b_exponents,
                dtype=np.int64,
            ),
            "p5p1_type21_r_b_coefficients": np.asarray(
                model.r_b_coefficients,
                dtype=np.complex128,
            ),
            "p5p1_type21_p1_exponents": np.asarray(
                model.p1_exponents,
                dtype=np.int64,
            ),
            "p5p1_type21_p1_coefficients": np.asarray(
                model.p1_coefficients,
                dtype=np.complex128,
            ),
            "p5p1_type21_p2_exponents": np.asarray(
                model.p2_exponents,
                dtype=np.int64,
            ),
            "p5p1_type21_p2_coefficients": np.asarray(
                model.p2_coefficients,
                dtype=np.complex128,
            ),
        }

    def lift_h_matrix_power(
        self,
        points: Sequence[P5P1Type21Point],
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
