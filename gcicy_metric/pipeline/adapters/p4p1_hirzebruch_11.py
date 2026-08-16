"""Direct type-(1,1) adapter for the smooth Hirzebruch X3 model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from ...type21_hirzebruch_x3 import (
    HirzebruchType21Model,
    HirzebruchType21Point,
    hirzebruch_type21_point_in_chart,
    hirzebruch_type21_product_fubini_study_distance,
    hirzebruch_type21_restricted_ambient_basis,
    make_hirzebruch_type21_model,
    sample_hirzebruch_type21_points,
    sample_hirzebruch_type21_points_with_diagnostics,
    sample_hirzebruch_type21_point_neighborhood,
)
from ...product_projective import choose_product_chart
from ..adapter import Chart, IndependentCoordinates
from ..schema import GCICYConfiguration
from ..spectrum import ScalarTrialData, product_projective_moment_map_trial_data
from .p4p1p1_hirzebruch_21 import P4P1P1HirzebruchType21Adapter


def hirzebruch_x3_type11_riemann_roch_dimension(power: int) -> int:
    return hirzebruch_type11_riemann_roch_dimension(power, base_degree=3)


def hirzebruch_type11_riemann_roch_dimension(
    power: int,
    *,
    base_degree: int,
) -> int:
    if power <= 0:
        raise ValueError("power must be positive")
    numerator = (14 + 3 * base_degree) * power**3
    numerator += (34 + 3 * base_degree) * power
    if numerator % 6:
        raise ArithmeticError("Riemann--Roch dimension is not integral")
    return numerator // 6


class P4P1HirzebruchType11Adapter(P4P1P1HirzebruchType21Adapter):
    """Expose X3 in its direct [P4|1,4; P1|3,-1] presentation."""

    key = "p4p1_type11_hirzebruch_x3"
    version = "1"
    configuration = GCICYConfiguration(
        key=key,
        name="Smooth direct type-(1,1) Hirzebruch X3",
        ambient_dimensions=(4, 1),
        positive_columns=((1, 3),),
        generalized_columns=((4, -1),),
        kahler_line_bundle=(1, 1),
        source="arXiv:1606.07420, Eq. (1), m=3",
    )

    def model_metadata(self, model: HirzebruchType21Model) -> dict[str, Any]:
        metadata = super().model_metadata(model)
        metadata.update(
            {
                "direct_type": "(1,1)",
                "direct_configuration": [[1, 4], [3, -1]],
                "polarization": [1, 1],
                "polarization_cube": 23,
                "c2_polarization": 86,
                "riemann_roch_section_dimensions_k1_to_k4": [
                    hirzebruch_x3_type11_riemann_roch_dimension(power)
                    for power in range(1, 5)
                ],
                "k1_ambient_restriction_is_complete": False,
                "ambient_restriction_complete_from_power": 2,
                "sampling_linear_subspace_method": "analytic_isotropic_projection",
            }
        )
        return metadata

    def sample_points(
        self,
        model: HirzebruchType21Model,
        count: int,
        *,
        seed: int,
    ) -> list[HirzebruchType21Point]:
        return sample_hirzebruch_type21_points(
            model,
            count,
            seed=seed,
            safe_projection=True,
        )

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
            safe_projection=True,
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
            safe_projection=True,
        )

    def sampling_cluster_ids(
        self,
        points: Sequence[HirzebruchType21Point],
    ) -> np.ndarray:
        return np.arange(len(points), dtype=np.int64) // 4

    def point_distance(
        self,
        left: HirzebruchType21Point,
        right: HirzebruchType21Point,
    ) -> float:
        return hirzebruch_type21_product_fubini_study_distance(left, right)

    def sample_local_neighborhood(
        self,
        model: HirzebruchType21Model,
        center: HirzebruchType21Point,
        count: int,
        *,
        seed: int,
        radius: float,
    ) -> tuple[list[HirzebruchType21Point], dict[str, Any]]:
        return sample_hirzebruch_type21_point_neighborhood(
            model,
            center,
            count,
            seed=seed,
            radius=radius,
        )

    def point_storage_payload(
        self,
        points: Sequence[HirzebruchType21Point],
    ) -> dict[str, np.ndarray]:
        if not points:
            raise ValueError("point storage payload cannot be empty")
        return {
            "coordinates_x": np.stack([point.x for point in points]),
            "coordinates_y": np.stack([point.y for point in points]),
            "coordinates_z": np.stack([point.z for point in points]),
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
        model: HirzebruchType21Model,
        payload: Mapping[str, np.ndarray],
    ) -> list[HirzebruchType21Point]:
        coordinate_fields = {"coordinates_x", "coordinates_y", "coordinates_z"}
        exact_chart_fields = {"projective_charts", "independent_indices"}
        if not coordinate_fields.issubset(payload):
            raise ValueError("type-(1,1) point payload has unexpected fields")
        optional_fields = set(payload) - coordinate_fields
        if optional_fields not in (set(), exact_chart_fields):
            raise ValueError("type-(1,1) point payload has unexpected fields")
        x = np.asarray(payload["coordinates_x"], dtype=np.complex128)
        y = np.asarray(payload["coordinates_y"], dtype=np.complex128)
        z = np.asarray(payload["coordinates_z"], dtype=np.complex128)
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError("stored P4 coordinates have an unexpected shape")
        if y.shape != (len(x), 2) or z.shape != (len(x), 2):
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
        if charts is not None and charts.shape != (len(x), 3):
            raise ValueError("stored projective charts have an unexpected shape")
        if independent is not None and independent.shape != (len(x), 3):
            raise ValueError("stored independent indices have an unexpected shape")
        points = []
        for index in range(len(x)):
            coordinates = (x[index], y[index], z[index])
            chart = (
                tuple(int(value) for value in charts[index])
                if charts is not None
                else choose_product_chart(coordinates)
            )
            points.append(
                hirzebruch_type21_point_in_chart(
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
        model: HirzebruchType21Model,
        point_payload: Mapping[str, np.ndarray],
        geometry_payload: Mapping[str, np.ndarray],
    ) -> list[HirzebruchType21Point]:
        """Replay frozen points without recomputing their local geometry."""

        del model
        x = np.asarray(point_payload["coordinates_x"], dtype=np.complex128)
        y = np.asarray(point_payload["coordinates_y"], dtype=np.complex128)
        z = np.asarray(point_payload["coordinates_z"], dtype=np.complex128)
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
            x.shape != (count, 5)
            or y.shape != (count, 2)
            or z.shape != (count, 2)
            or charts.shape != (count, 3)
            or independent.shape != (count, 3)
            or dependent.shape != (count, 3)
            or affine.shape != (count, 6)
            or tangent.shape != (count, 6, 3)
            or residue.shape != (count,)
            or minimum.shape != (count,)
        ):
            raise ValueError("type-(1,1) common-pool geometry has invalid shapes")
        return [
            HirzebruchType21Point(
                coordinates=(x[index], y[index], z[index]),
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

    def rechart_point(
        self,
        model: HirzebruchType21Model,
        point: HirzebruchType21Point,
        chart: Chart,
        *,
        independent: IndependentCoordinates | None = None,
    ) -> HirzebruchType21Point:
        if len(chart) != 2:
            raise ValueError("the direct P4 x P1 adapter requires two chart entries")
        stabilized_chart = (
            int(chart[0]),
            int(chart[1]),
            int(point.projective_chart[2]),
        )
        selected_independent = (
            tuple(int(value) for value in independent)
            if independent is not None
            else None
        )
        return hirzebruch_type21_point_in_chart(
            model,
            point.coordinates,
            stabilized_chart,
            independent=selected_independent,  # type: ignore[arg-type]
        )

    def chart_is_available(
        self,
        point: HirzebruchType21Point,
        chart: Chart,
        *,
        minimum: float,
    ) -> bool:
        if len(chart) != 2:
            return False
        selected = (
            point.coordinates[0][int(chart[0])],
            point.coordinates[1][int(chart[1])],
        )
        return min(abs(value) for value in selected) > minimum

    def point_projective_chart(self, point: HirzebruchType21Point) -> Chart:
        return tuple(int(value) for value in point.projective_chart[:2])

    def restricted_section_basis(
        self,
        points: Sequence[HirzebruchType21Point],
        degree: tuple[int, ...],
    ) -> Any:
        if len(degree) != 2:
            raise ValueError("the direct P4 x P1 section degree requires two entries")
        direct_degree = (int(degree[0]), int(degree[1]))
        stabilized = hirzebruch_type21_restricted_ambient_basis(
            points,
            (*direct_degree, 0),
        )
        return replace(stabilized, degree=direct_degree)

    def scalar_laplacian_trial_data(
        self,
        point: HirzebruchType21Point,
        *,
        level: int,
        cubic_feature_count: int = 256,
        cubic_feature_seed: int = 314159,
    ) -> ScalarTrialData:
        return product_projective_moment_map_trial_data(
            point.affine_coordinates[:5],
            point.projective_chart[:2],
            point.tangent_basis[:5],
            self.configuration.ambient_dimensions,
            level=level,
            cubic_feature_count=cubic_feature_count,
            cubic_feature_seed=cubic_feature_seed,
        )


class P4P1HirzebruchM4Type11Adapter(P4P1HirzebruchType11Adapter):
    """Out-of-sample m=4 member with a degree -2 generalized section."""

    key = "p4p1_type11_hirzebruch_m4"
    version = "1"
    configuration = GCICYConfiguration(
        key=key,
        name="Hirzebruch m=4 type-(1,1) gCICY",
        ambient_dimensions=(4, 1),
        positive_columns=((1, 4),),
        generalized_columns=((4, -2),),
        kahler_line_bundle=(1, 1),
        source="arXiv:1606.07420 Hirzebruch sequence",
    )

    def make_model(self, seed: int, *, exact: bool) -> HirzebruchType21Model:
        return make_hirzebruch_type21_model(
            seed,
            exact=exact,
            base_degree=4,
        )

    def model_metadata(self, model: HirzebruchType21Model) -> dict[str, Any]:
        return {
            "seed": int(model.seed),
            "direct_type": "(1,1)",
            "direct_configuration": [[1, 4], [4, -2]],
            "polarization": [1, 1],
            "polarization_cube": 26,
            "c2_polarization": 92,
            "riemann_roch_section_dimensions_k1_to_k4": [
                hirzebruch_type11_riemann_roch_dimension(
                    power,
                    base_degree=4,
                )
                for power in range(1, 5)
            ],
            "generalized_section_construction": (
                "kernel of the Cech obstruction map generated by Koszul syzygies"
            ),
            "generalized_section_space_dimension": 105,
            "ambient_restriction_complete_from_power": 3,
            "sampling_linear_subspace_method": "analytic_isotropic_projection",
            "out_of_sample_geometry": True,
        }
