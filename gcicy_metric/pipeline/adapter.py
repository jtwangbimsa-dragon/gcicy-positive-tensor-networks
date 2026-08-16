"""Abstract geometry boundary for configuration-driven gCICY computations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import GCICYConfiguration


Array = np.ndarray
Chart = tuple[int, ...]
IndependentCoordinates = tuple[int, ...]
H_POSITIVE_RELATIVE_FLOOR = 1e-12
H_NEGATIVE_RELATIVE_TOLERANCE = 1e-6


def positive_hermitian_projection(
    matrix: Array,
    *,
    relative_floor: float = H_POSITIVE_RELATIVE_FLOOR,
    negative_relative_tolerance: float = H_NEGATIVE_RELATIVE_TOLERANCE,
) -> Array:
    """Remove roundoff-scale negative modes from a positive Hermitian form."""

    candidate = np.asarray(matrix, dtype=np.complex128)
    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError("Hermitian projection requires a square matrix")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("Hermitian projection requires a finite matrix")
    if relative_floor <= 0 or negative_relative_tolerance <= 0:
        raise ValueError("Hermitian projection tolerances must be positive")
    candidate = 0.5 * (candidate + candidate.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(candidate)
    scale = max(1.0, float(eigenvalues[-1]))
    if eigenvalues[-1] <= 0 or eigenvalues[0] < -negative_relative_tolerance * scale:
        raise FloatingPointError("H-matrix has a non-roundoff negative eigenvalue")
    clipped = np.maximum(eigenvalues, relative_floor * scale)
    projected = (eigenvectors * clipped[None, :]) @ eigenvectors.conjugate().T
    return 0.5 * (projected + projected.conjugate().T)


@dataclass(frozen=True)
class HMetricArtifact:
    """Configuration-validated global-section H-metric data."""

    path: Path
    degree: tuple[int, ...]
    section_exponents: Array
    h_matrix: Array
    normalization: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        exponents = np.asarray(self.section_exponents)
        h_matrix = np.asarray(self.h_matrix)
        if exponents.ndim != 2 or len(exponents) == 0:
            raise ValueError("section exponents must be a non-empty matrix")
        if h_matrix.shape != (len(exponents), len(exponents)):
            raise ValueError("H-matrix shape does not match the section basis")
        if not np.isfinite(self.normalization) or self.normalization <= 0:
            raise ValueError("metric normalization must be finite and positive")
        if not np.all(np.isfinite(h_matrix)):
            raise ValueError("H-matrix must be finite")
        hermitian_error = float(np.max(np.abs(h_matrix - h_matrix.conjugate().T)))
        if hermitian_error > 1e-8:
            raise ValueError(f"H-matrix is not Hermitian (error {hermitian_error:.3e})")

    @property
    def key(self) -> str:
        return self.path.stem

    @property
    def section_count(self) -> int:
        return len(self.section_exponents)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "degree": list(self.degree),
            "section_count": self.section_count,
            "normalization": self.normalization,
            "metadata": self.metadata,
        }


class GCICYAdapter(ABC):
    """Interface between configuration-specific geometry and generic audits."""

    key: str
    version: str = "1"
    configuration: GCICYConfiguration

    @abstractmethod
    def make_model(self, seed: int, *, exact: bool) -> Any:
        """Construct explicit defining sections for one configuration member."""

    @abstractmethod
    def model_metadata(self, model: Any) -> dict[str, Any]:
        """Return JSON-serializable identity data for the explicit model."""

    @abstractmethod
    def sample_points(self, model: Any, count: int, *, seed: int) -> list[Any]:
        """Sample points together with exact local-coordinate data."""

    def default_sampling_options(self) -> dict[str, Any]:
        """Return JSON-safe numerical controls used by the sampler."""

        return {}

    def sample_points_configured(
        self,
        model: Any,
        count: int,
        *,
        seed: int,
        sampling_options: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """Sample with explicit, reproducibly recorded numerical controls."""

        options = dict(sampling_options or self.default_sampling_options())
        if options:
            raise ValueError(
                f"adapter {self.key} does not support sampling options "
                f"{sorted(options)}"
            )
        return self.sample_points(model, count, seed=seed)

    def sample_points_with_diagnostics(
        self,
        model: Any,
        count: int,
        *,
        seed: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Sample points and return JSON-safe sampler diagnostics.

        Adapters whose sampler emits correlated all-root fibres should override
        this method together with ``sampling_cluster_ids``.  The default treats
        every point as an independent sampling cluster.
        """

        points = self.sample_points(model, count, seed=seed)
        return points, {
            "requested_points": int(count),
            "returned_points": int(len(points)),
            "attempted_clusters": int(len(points)),
            "accepted_clusters": int(len(points)),
            "expected_points_per_cluster": 1,
            "returned_complete_clusters": int(len(points)),
            "truncated_final_cluster": False,
            "all_returned_clusters_complete": True,
        }

    def sampling_cluster_ids(self, points: Sequence[Any]) -> Array:
        """Return one integer id per independent sampler cluster."""

        return np.arange(len(points), dtype=np.int64)

    def point_distance(self, left: Any, right: Any) -> float:
        """Return a chart-invariant distance used to separate active centers."""

        raise NotImplementedError(
            f"adapter {self.key} does not implement point-distance evaluation"
        )

    def sample_local_neighborhood(
        self,
        model: Any,
        center: Any,
        count: int,
        *,
        seed: int,
        radius: float,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Sample an audited on-manifold neighborhood for active-set training."""

        raise NotImplementedError(
            f"adapter {self.key} does not implement local-neighborhood sampling"
        )

    def retract_intrinsic_step(
        self,
        model: Any,
        center: Any,
        intrinsic_step: Array,
        *,
        residual_tolerance: float = 1e-10,
        maximum_newton_iterations: int = 16,
    ) -> tuple[Any, dict[str, Any]]:
        """Retract one prescribed complex tangent step back to the geometry."""

        raise NotImplementedError(
            f"adapter {self.key} does not implement intrinsic retraction"
        )

    def point_storage_payload(self, points: Sequence[Any]) -> dict[str, Array]:
        """Serialize points without Python pickles for an active-set artifact."""

        raise NotImplementedError(
            f"adapter {self.key} does not implement point serialization"
        )

    def points_from_storage_payload(
        self,
        model: Any,
        payload: Mapping[str, Array],
    ) -> list[Any]:
        """Reconstruct exact point objects from an active-set artifact."""

        raise NotImplementedError(
            f"adapter {self.key} does not implement point deserialization"
        )

    def points_from_common_pool_payload(
        self,
        model: Any,
        point_payload: Mapping[str, Array],
        geometry_payload: Mapping[str, Array],
    ) -> list[Any]:
        """Reconstruct points, optionally using stored local geometry."""

        del geometry_payload
        return self.points_from_storage_payload(model, point_payload)

    @abstractmethod
    def baseline_metric(self, point: Any) -> Array:
        """Evaluate the reference Kahler metric in the point's local coordinates."""

    @abstractmethod
    def importance_log_weight(self, point: Any) -> float:
        """Return log(target volume / proposal volume), up to one constant."""

    @abstractmethod
    def monge_ampere_log_error(self, point: Any, metric: Array) -> float:
        """Return log det(g) - log |Omega|^2 in matching local coordinates."""

    @abstractmethod
    def rechart_point(
        self,
        model: Any,
        point: Any,
        chart: Chart,
        *,
        independent: IndependentCoordinates | None = None,
    ) -> Any:
        """Reconstruct the same geometric point in another local chart."""

    @abstractmethod
    def chart_is_available(self, point: Any, chart: Chart, *, minimum: float) -> bool:
        """Whether the selected homogeneous coordinates define this chart."""

    @abstractmethod
    def point_projective_chart(self, point: Any) -> Chart:
        """Return the point's current ambient projective chart."""

    @abstractmethod
    def point_jacobian_min_singular_value(self, point: Any) -> float:
        """Return the smallest defining-Jacobian singular value at a point."""

    @abstractmethod
    def load_h_artifact(self, path: Path, model: Any) -> HMetricArtifact:
        """Load an H-metric only after checking its explicit-model identity."""

    @abstractmethod
    def h_metric(self, point: Any, artifact: HMetricArtifact) -> Array:
        """Evaluate one validated H-metric artifact at one point."""

    @abstractmethod
    def restricted_section_basis(self, points: Sequence[Any], degree: tuple[int, ...]) -> Any:
        """Build a numerically independent restricted ambient section basis."""

    @abstractmethod
    def restricted_fubini_study_h_matrix(
        self, points: Sequence[Any], basis: Any
    ) -> tuple[Array, float]:
        """Return a positive reference H-matrix and its relation error."""

    @abstractmethod
    def section_values_and_jacobian(
        self, point: Any, exponents: Array
    ) -> tuple[Array, Array]:
        """Evaluate sections and intrinsic holomorphic derivatives."""

    def section_values_and_jacobian_batch(
        self,
        points: Sequence[Any],
        exponents: Array,
    ) -> tuple[Array, Array]:
        """Evaluate sections and derivatives on a batch of points."""

        rows = [
            self.section_values_and_jacobian(point, exponents)
            for point in points
        ]
        if not rows:
            powers = np.asarray(exponents)
            return (
                np.empty((0, len(powers)), dtype=np.complex128),
                np.empty(
                    (0, len(powers), self.configuration.complex_dimension),
                    dtype=np.complex128,
                ),
            )
        values, derivatives = zip(*rows, strict=True)
        return (
            np.asarray(values, dtype=np.complex128),
            np.asarray(derivatives, dtype=np.complex128),
        )

    @abstractmethod
    def artifact_model_payload(self, model: Any, *, exact: bool) -> dict[str, Array]:
        """Return configuration-specific fields needed to identify a saved artifact."""

    def lift_h_matrix_power(
        self,
        points: Sequence[Any],
        source: HMetricArtifact,
        target_basis: Any,
    ) -> tuple[Array, float]:
        raise NotImplementedError(f"adapter {self.key} does not implement H-matrix power lifting")

    def projective_charts(self) -> Iterable[Chart]:
        return product(*(range(dimension + 1) for dimension in self.configuration.ambient_dimensions))

    def implicit_coordinate_choices(self) -> Iterable[IndependentCoordinates]:
        return combinations(
            range(self.configuration.ambient_affine_dimension),
            self.configuration.complex_dimension,
        )

    def expected_projective_chart_count(self) -> int:
        """Return the number of projective charts the adapter expects to cover."""

        return self.configuration.projective_chart_count

    def expected_implicit_coordinate_choice_count(self) -> int:
        """Return the number of structurally admissible implicit choices."""

        return self.configuration.implicit_coordinate_choice_count

    def baseline_metrics(self, points: Sequence[Any]) -> Array:
        return np.asarray([self.baseline_metric(point) for point in points], dtype=np.complex128)

    def h_metrics(self, points: Sequence[Any], artifact: HMetricArtifact) -> Array:
        return np.asarray(
            [self.h_metric(point, artifact) for point in points],
            dtype=np.complex128,
        )

    def importance_weights(self, points: Sequence[Any]) -> Array:
        log_weights = np.asarray([self.importance_log_weight(point) for point in points], dtype=float)
        if log_weights.size == 0 or not np.all(np.isfinite(log_weights)):
            raise ValueError("importance log weights must be a non-empty finite array")
        weights = np.exp(log_weights - float(np.max(log_weights)))
        mean = float(np.mean(weights))
        if not np.isfinite(mean) or mean <= 0:
            raise FloatingPointError("importance weights could not be normalized")
        return weights / mean

    def residual_values(self, points: Sequence[Any], metrics: Array) -> Array:
        if len(points) != len(metrics):
            raise ValueError("metric count must match point count")
        values = []
        for point, metric in zip(points, metrics, strict=True):
            try:
                values.append(self.monge_ampere_log_error(point, metric))
            except FloatingPointError:
                values.append(float("nan"))
        return np.asarray(values, dtype=float)

    def holomorphic_volume_log_density(self, point: Any) -> float:
        """Recover log |Omega|^2 using the reference metric and MA residual."""

        metric = np.asarray(self.baseline_metric(point), dtype=np.complex128)
        eigenvalues = np.linalg.eigvalsh(metric)
        if eigenvalues[0] <= 0:
            raise FloatingPointError("baseline metric is not positive definite")
        return float(
            np.sum(np.log(eigenvalues)) - self.monge_ampere_log_error(point, metric)
        )

    def scalar_laplacian_trial_data(
        self,
        point: Any,
        *,
        level: int,
        cubic_feature_count: int = 256,
        cubic_feature_seed: int = 314159,
    ) -> Any:
        """Return global real scalar features and their holomorphic gradients."""

        raise NotImplementedError(
            f"adapter {self.key} does not implement scalar-Laplacian trial functions"
        )
