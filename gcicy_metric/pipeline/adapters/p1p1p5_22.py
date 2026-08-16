"""Adapter for the existing P1 x P1 x P5 type-(2,2) gCICY model."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...generic_model import (
    generic_baseline_metric,
    generic_global_h_metric,
    generic_importance_log_weight,
    generic_lift_h_matrix_power,
    generic_monge_ampere_log_error,
    generic_point_in_chart,
    generic_point_in_chart,
    generic_restricted_ambient_basis,
    generic_restricted_fubini_study_h_matrix,
    generic_section_values_and_jacobian,
    make_exact_generic_model,
    make_generic_model,
    sample_generic_gcicy_points,
    sample_generic_gcicy_points_with_diagnostics,
)
from ..adapter import (
    Chart,
    GCICYAdapter,
    HMetricArtifact,
    IndependentCoordinates,
    positive_hermitian_projection,
)
from ..schema import GCICYConfiguration


class P1P1P5Type22Adapter(GCICYAdapter):
    key = "p1p1p5_type22"
    version = "1"
    configuration = GCICYConfiguration(
        key=key,
        name="P1 x P1 x P5 generalized complete intersection",
        ambient_dimensions=(1, 1, 5),
        positive_columns=((1, 1, 3), (1, 1, 1)),
        generalized_columns=((-1, 1, 1), (1, -1, 1)),
        kahler_line_bundle=(1, 1, 1),
        source="arXiv:1507.03235, Eq. (1.2)",
    )

    def make_model(self, seed: int, *, exact: bool) -> Any:
        return make_exact_generic_model(seed) if exact else make_generic_model(seed)

    def model_metadata(self, model: Any) -> dict[str, Any]:
        return {
            "seed": int(model.seed),
            "p1_term_count": int(len(model.p1_coefficients)),
            "p2_term_count": int(len(model.p2_coefficients)),
        }

    def sample_points(self, model: Any, count: int, *, seed: int) -> list[Any]:
        return sample_generic_gcicy_points(model, count, seed=seed)

    def default_sampling_options(self) -> dict[str, float]:
        return {"root_separation_tolerance": 1.0e-7}

    def sample_points_configured(
        self,
        model: Any,
        count: int,
        *,
        seed: int,
        sampling_options: dict[str, Any] | None = None,
    ) -> list[Any]:
        options = dict(sampling_options or self.default_sampling_options())
        unknown = set(options) - {"root_separation_tolerance"}
        if unknown:
            raise ValueError(f"unsupported sampling options: {sorted(unknown)}")
        return sample_generic_gcicy_points(
            model,
            count,
            seed=seed,
            root_separation_tolerance=float(options["root_separation_tolerance"]),
        )

    def sample_points_with_diagnostics(
        self,
        model: Any,
        count: int,
        *,
        seed: int,
    ) -> tuple[list[Any], dict[str, object]]:
        return sample_generic_gcicy_points_with_diagnostics(
            model,
            count,
            seed=seed,
        )

    def sampling_cluster_ids(self, points: list[Any]) -> np.ndarray:
        return np.arange(len(points), dtype=np.int64) // 3

    def baseline_metric(self, point: Any) -> np.ndarray:
        return generic_baseline_metric(point)

    def importance_log_weight(self, point: Any) -> float:
        return generic_importance_log_weight(point)

    def monge_ampere_log_error(self, point: Any, metric: np.ndarray) -> float:
        return generic_monge_ampere_log_error(point, metric)

    def rechart_point(
        self,
        model: Any,
        point: Any,
        chart: Chart,
        *,
        independent: IndependentCoordinates | None = None,
    ) -> Any:
        if len(chart) != 3:
            raise ValueError("the P1 x P1 x P5 adapter requires a three-entry chart")
        return generic_point_in_chart(
            model,
            point.x,
            point.y,
            point.z,
            tuple(int(value) for value in chart),
            independent=independent,
        )

    def chart_is_available(self, point: Any, chart: Chart, *, minimum: float) -> bool:
        selected = (point.x[chart[0]], point.y[chart[1]], point.z[chart[2]])
        return min(abs(value) for value in selected) > minimum

    def point_projective_chart(self, point: Any) -> Chart:
        return tuple(int(value) for value in point.projective_chart)

    def point_jacobian_min_singular_value(self, point: Any) -> float:
        return float(point.jacobian_min_singular_value)

    def point_storage_payload(self, points: Sequence[Any]) -> dict[str, np.ndarray]:
        return {
            "coordinates_x": np.asarray(
                [point.x for point in points], dtype=np.complex128
            ),
            "coordinates_y": np.asarray(
                [point.y for point in points], dtype=np.complex128
            ),
            "coordinates_z": np.asarray(
                [point.z for point in points], dtype=np.complex128
            ),
            "projective_charts": np.asarray(
                [point.projective_chart for point in points], dtype=np.int64
            ),
            "independent_indices": np.asarray(
                [point.independent_indices for point in points], dtype=np.int64
            ),
        }

    def points_from_storage_payload(
        self,
        model: Any,
        payload: Mapping[str, np.ndarray],
    ) -> list[Any]:
        required = {
            "coordinates_x",
            "coordinates_y",
            "coordinates_z",
            "projective_charts",
            "independent_indices",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"point payload is missing fields: {', '.join(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in required}
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("point payload arrays have inconsistent lengths")
        return [
            generic_point_in_chart(
                model,
                x,
                y,
                z,
                tuple(int(value) for value in chart),
                independent=tuple(int(value) for value in independent),
            )
            for x, y, z, chart, independent in zip(
                arrays["coordinates_x"],
                arrays["coordinates_y"],
                arrays["coordinates_z"],
                arrays["projective_charts"],
                arrays["independent_indices"],
                strict=True,
            )
        ]

    def load_h_artifact(self, path: Path, model: Any) -> HMetricArtifact:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"missing H-metric artifact: {resolved}")
        with np.load(resolved, allow_pickle=False) as payload:
            required = {
                "generic_model_seed",
                "p1_coefficients",
                "p2_tensor",
                "global_section_degree",
                "global_section_exponents",
                "global_h_matrix",
                "global_section_normalization",
            }
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"artifact is missing fields: {', '.join(missing)}")
            if int(payload["generic_model_seed"]) != int(model.seed):
                raise ValueError("artifact model seed does not match the requested model")
            if not np.allclose(payload["p1_coefficients"], model.p1_coefficients):
                raise ValueError("artifact p1 coefficients do not match the explicit model")
            if not np.allclose(payload["p2_tensor"], model.p2_tensor):
                raise ValueError("artifact p2 coefficients do not match the explicit model")

            degree = tuple(int(value) for value in payload["global_section_degree"])
            if len(degree) != len(self.configuration.ambient_dimensions):
                raise ValueError("artifact section degree does not match the ambient factors")
            metadata = {
                "model_seed": int(payload["generic_model_seed"]),
                "exact_integer_coefficients": bool(payload["exact_integer_coefficients"])
                if "exact_integer_coefficients" in payload.files
                else None,
                "importance_weighted_training": bool(payload["importance_weighted_training"])
                if "importance_weighted_training" in payload.files
                else None,
            }
            return HMetricArtifact(
                path=resolved,
                degree=degree,
                section_exponents=np.asarray(payload["global_section_exponents"], dtype=np.int64).copy(),
                h_matrix=np.asarray(payload["global_h_matrix"], dtype=np.complex128).copy(),
                normalization=float(payload["global_section_normalization"]),
                metadata=metadata,
            )

    def h_metric(self, point: Any, artifact: HMetricArtifact) -> np.ndarray:
        return generic_global_h_metric(
            point,
            artifact.section_exponents,
            artifact.h_matrix,
            normalization=artifact.normalization,
        )

    def restricted_section_basis(self, points: list[Any], degree: tuple[int, ...]) -> Any:
        return generic_restricted_ambient_basis(points, tuple(int(value) for value in degree))

    def restricted_fubini_study_h_matrix(
        self, points: list[Any], basis: Any
    ) -> tuple[np.ndarray, float]:
        return generic_restricted_fubini_study_h_matrix(points, basis)

    def section_values_and_jacobian(
        self, point: Any, exponents: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return generic_section_values_and_jacobian(point, exponents)

    def artifact_model_payload(self, model: Any, *, exact: bool) -> dict[str, np.ndarray]:
        return {
            "generic_model_seed": np.asarray(model.seed),
            "exact_integer_coefficients": np.asarray(exact),
            "p1_exponents": np.asarray(model.p1_exponents, dtype=np.int64),
            "p1_coefficients": np.asarray(model.p1_coefficients, dtype=np.complex128),
            "p2_exponents": np.asarray(model.p2_exponents, dtype=np.int64),
            "p2_coefficients": np.asarray(model.p2_coefficients, dtype=np.complex128),
            "p2_tensor": np.asarray(model.p2_tensor, dtype=np.complex128),
        }

    def lift_h_matrix_power(
        self,
        points: list[Any],
        source: HMetricArtifact,
        target_basis: Any,
    ) -> tuple[np.ndarray, float]:
        if tuple(source.degree) == tuple(target_basis.degree):
            source_values = np.asarray(
                [
                    self.section_values_and_jacobian(point, source.section_exponents)[0]
                    for point in points
                ],
                dtype=np.complex128,
            )
            target_values = np.asarray(
                [
                    self.section_values_and_jacobian(point, target_basis.selected_exponents)[0]
                    for point in points
                ],
                dtype=np.complex128,
            )
            coefficients, _, _, _ = np.linalg.lstsq(target_values, source_values, rcond=1e-10)
            relation_error = float(
                np.linalg.norm(target_values @ coefficients - source_values)
                / max(1.0, np.linalg.norm(source_values))
            )
            target_h = np.conjugate(coefficients) @ source.h_matrix @ coefficients.T
            return positive_hermitian_projection(target_h), relation_error
        return generic_lift_h_matrix_power(
            points,
            source.section_exponents,
            positive_hermitian_projection(source.h_matrix),
            target_basis,
        )
