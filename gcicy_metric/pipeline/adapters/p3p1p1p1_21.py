"""Pipeline adapter for the Table-13 P3 x P1 x P1 x P1 type-(2,1) model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...type21_p3p1p1p1 import (
    Type21Model,
    Type21Point,
    make_type21_model,
    sample_type21_points,
    type21_baseline_metric,
    type21_global_h_metric,
    type21_importance_log_weight,
    type21_monge_ampere_log_error,
    type21_point_in_chart,
    type21_restricted_ambient_basis,
    type21_restricted_fubini_study_h_matrix,
    type21_section_values_and_jacobian,
)
from ..adapter import (
    Chart,
    GCICYAdapter,
    HMetricArtifact,
    IndependentCoordinates,
    positive_hermitian_projection,
)
from ..schema import GCICYConfiguration


class P3P1P1P1Type21Adapter(GCICYAdapter):
    key = "p3p1p1p1_type21_hodge_9_13"
    version = "1"
    configuration = GCICYConfiguration(
        key=key,
        name="P3 x P1 x P1 x P1 type-(2,1) gCICY with reported Hodge pair (9,13)",
        ambient_dimensions=(3, 1, 1, 1),
        positive_columns=((2, 0, 0, 1), (0, 3, 1, 1)),
        generalized_columns=((2, -1, 1, 0),),
        kahler_line_bundle=(1, 1, 1, 1),
        source="arXiv:1507.03235, Table 13, second (9,13) configuration",
    )

    def make_model(self, seed: int, *, exact: bool) -> Type21Model:
        return make_type21_model(seed, exact=exact)

    def model_metadata(self, model: Type21Model) -> dict[str, Any]:
        return {
            "seed": int(model.seed),
            "a_term_count": int(len(model.a_coefficients)),
            "b_term_count": int(len(model.b_coefficients)),
            "c_term_count": int(model.c_tensor.size),
            "d_term_count": int(model.d_tensor.size),
            "generalized_section_space_dimension": 3,
            "reported_hodge_numbers": [9, 13],
        }

    def sample_points(
        self,
        model: Type21Model,
        count: int,
        *,
        seed: int,
    ) -> list[Type21Point]:
        return sample_type21_points(model, count, seed=seed)

    def baseline_metric(self, point: Type21Point) -> np.ndarray:
        return type21_baseline_metric(point)

    def importance_log_weight(self, point: Type21Point) -> float:
        return type21_importance_log_weight(point)

    def monge_ampere_log_error(
        self,
        point: Type21Point,
        metric: np.ndarray,
    ) -> float:
        return type21_monge_ampere_log_error(point, metric)

    def rechart_point(
        self,
        model: Type21Model,
        point: Type21Point,
        chart: Chart,
        *,
        independent: IndependentCoordinates | None = None,
    ) -> Type21Point:
        if len(chart) != 4:
            raise ValueError("the type-(2,1) adapter requires a four-entry chart")
        selected_chart = tuple(int(value) for value in chart)
        selected_independent = (
            tuple(int(value) for value in independent)
            if independent is not None
            else None
        )
        return type21_point_in_chart(
            model,
            point.coordinates,
            selected_chart,  # type: ignore[arg-type]
            independent=selected_independent,  # type: ignore[arg-type]
        )

    def chart_is_available(
        self,
        point: Type21Point,
        chart: Chart,
        *,
        minimum: float,
    ) -> bool:
        if len(chart) != 4:
            return False
        selected = [
            values[int(index)]
            for values, index in zip(point.coordinates, chart, strict=True)
        ]
        return min(abs(value) for value in selected) > minimum

    def point_projective_chart(self, point: Type21Point) -> Chart:
        return tuple(int(value) for value in point.projective_chart)

    def point_jacobian_min_singular_value(self, point: Type21Point) -> float:
        return float(point.jacobian_min_singular_value)

    def load_h_artifact(self, path: Path, model: Type21Model) -> HMetricArtifact:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"missing H-metric artifact: {resolved}")
        with np.load(resolved, allow_pickle=False) as payload:
            required = {
                "pipeline_adapter",
                "type21_model_seed",
                "type21_a_coefficients",
                "type21_b_coefficients",
                "type21_c_tensor",
                "type21_d_tensor",
                "type21_q_moments",
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
            if int(payload["type21_model_seed"]) != int(model.seed):
                raise ValueError("artifact model seed does not match the requested model")
            comparisons = (
                ("a", payload["type21_a_coefficients"], model.a_coefficients),
                ("b", payload["type21_b_coefficients"], model.b_coefficients),
                ("c", payload["type21_c_tensor"], model.c_tensor),
                ("d", payload["type21_d_tensor"], model.d_tensor),
                ("q", payload["type21_q_moments"], model.q_moments),
            )
            for label, candidate, expected in comparisons:
                if not np.allclose(candidate, expected):
                    raise ValueError(f"artifact {label} coefficients do not match the model")
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
                    "model_seed": int(payload["type21_model_seed"]),
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
        point: Type21Point,
        artifact: HMetricArtifact,
    ) -> np.ndarray:
        return type21_global_h_metric(
            point,
            artifact.section_exponents,
            artifact.h_matrix,
            normalization=artifact.normalization,
        )

    def restricted_section_basis(
        self,
        points: Sequence[Type21Point],
        degree: tuple[int, ...],
    ) -> Any:
        return type21_restricted_ambient_basis(points, degree)

    def restricted_fubini_study_h_matrix(
        self,
        points: Sequence[Type21Point],
        basis: Any,
    ) -> tuple[np.ndarray, float]:
        return type21_restricted_fubini_study_h_matrix(points, basis)

    def section_values_and_jacobian(
        self,
        point: Type21Point,
        exponents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return type21_section_values_and_jacobian(point, exponents)

    def artifact_model_payload(
        self,
        model: Type21Model,
        *,
        exact: bool,
    ) -> dict[str, np.ndarray]:
        return {
            "type21_model_seed": np.asarray(model.seed),
            "exact_integer_coefficients": np.asarray(exact),
            "type21_x_quadratic_exponents": np.asarray(
                model.x_quadratic_exponents,
                dtype=np.int64,
            ),
            "type21_a_coefficients": np.asarray(
                model.a_coefficients,
                dtype=np.complex128,
            ),
            "type21_b_coefficients": np.asarray(
                model.b_coefficients,
                dtype=np.complex128,
            ),
            "type21_c_tensor": np.asarray(model.c_tensor, dtype=np.complex128),
            "type21_d_tensor": np.asarray(model.d_tensor, dtype=np.complex128),
            "type21_q_moments": np.asarray(model.q_moments, dtype=np.complex128),
            "type21_p1_exponents": np.asarray(model.p1_exponents, dtype=np.int64),
            "type21_p1_coefficients": np.asarray(
                model.p1_coefficients,
                dtype=np.complex128,
            ),
            "type21_p2_exponents": np.asarray(model.p2_exponents, dtype=np.int64),
            "type21_p2_coefficients": np.asarray(
                model.p2_coefficients,
                dtype=np.complex128,
            ),
        }

    def lift_h_matrix_power(
        self,
        points: Sequence[Type21Point],
        source: HMetricArtifact,
        target_basis: Any,
    ) -> tuple[np.ndarray, float]:
        if tuple(source.degree) != tuple(target_basis.degree):
            raise NotImplementedError(
                "power lifting for this type-(2,1) adapter will be added after the k=1 gate"
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
            source_values,
            rcond=1e-10,
        )
        relation_error = float(
            np.linalg.norm(target_values @ coefficients - source_values)
            / max(1.0, np.linalg.norm(source_values))
        )
        target_h = np.conjugate(coefficients) @ source.h_matrix @ coefficients.T
        return positive_hermitian_projection(target_h), relation_error
