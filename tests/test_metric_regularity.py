from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from gcicy_metric.pipeline import (
    HMetricArtifact,
    estimate_h_metric_residual_gradient,
    finite_distance_residual_slopes,
    get_adapter,
    reference_orthonormal_complex_directions,
)


class MetricRegularityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = get_adapter("p4p1_type11_hirzebruch_x3")
        cls.model = cls.adapter.make_model(20260731, exact=True)
        basis_points = cls.adapter.sample_points(cls.model, 64, seed=17721)
        basis = cls.adapter.restricted_section_basis(basis_points, (1, 1))
        h_matrix, relation_error = cls.adapter.restricted_fubini_study_h_matrix(
            basis_points,
            basis,
        )
        if relation_error > 1e-10:
            raise AssertionError("test reference H-matrix did not reproduce FS")
        cls.artifact = HMetricArtifact(
            path=Path("<regularity-test>"),
            degree=(1, 1),
            section_exponents=basis.selected_exponents,
            h_matrix=h_matrix,
            normalization=1.0,
        )
        cls.point = cls.adapter.sample_points(cls.model, 1, seed=17722)[0]

    def test_intrinsic_retraction_is_on_manifold_and_distance_calibrated(self) -> None:
        reference_metric = self.adapter.baseline_metric(self.point)
        direction = reference_orthonormal_complex_directions(reference_metric)[:, 0]
        step = 1e-3
        candidate, diagnostics = self.adapter.retract_intrinsic_step(
            self.model,
            self.point,
            step * direction,
        )
        self.assertLessEqual(diagnostics["relative_equation_residual"], 1e-10)
        self.assertGreater(
            float(np.min(np.linalg.eigvalsh(self.adapter.baseline_metric(candidate)))),
            0.0,
        )
        self.assertAlmostEqual(
            diagnostics["product_fubini_study_distance"] / step,
            1.0,
            delta=2e-3,
        )

    def test_two_scale_gradient_has_small_discretization_error(self) -> None:
        estimate = estimate_h_metric_residual_gradient(
            self.adapter,
            self.model,
            self.point,
            self.artifact,
            coarse_step=1e-3,
        )
        self.assertTrue(np.all(np.isfinite(estimate.richardson_components)))
        self.assertGreater(estimate.richardson_norm, 0.0)
        self.assertLess(
            estimate.richardson_error_norm / estimate.richardson_norm,
            1e-3,
        )
        self.assertLess(estimate.maximum_retraction_residual, 1e-10)
        self.assertEqual(
            estimate.to_dict()["interpretation"],
            "empirical_local_derivative_not_global_bound",
        )

    def test_gradient_norm_is_projective_chart_invariant(self) -> None:
        reference = estimate_h_metric_residual_gradient(
            self.adapter,
            self.model,
            self.point,
            self.artifact,
            coarse_step=1e-3,
        )
        alternate_chart = next(
            chart
            for chart in self.adapter.projective_charts()
            if chart != self.adapter.point_projective_chart(self.point)
        )
        recharts = self.adapter.rechart_point(
            self.model,
            self.point,
            alternate_chart,
        )
        candidate = estimate_h_metric_residual_gradient(
            self.adapter,
            self.model,
            recharts,
            self.artifact,
            coarse_step=1e-3,
        )
        self.assertAlmostEqual(
            reference.richardson_norm,
            candidate.richardson_norm,
            delta=1e-7 * reference.richardson_norm,
        )

    def test_local_secant_slopes_use_center_labels(self) -> None:
        neighbors, _ = self.adapter.sample_local_neighborhood(
            self.model,
            self.point,
            4,
            seed=17723,
            radius=0.01,
        )
        center_residual = np.asarray(
            [
                self.adapter.monge_ampere_log_error(
                    self.point,
                    self.adapter.h_metric(self.point, self.artifact),
                )
            ]
        )
        residuals = self.adapter.residual_values(
            neighbors,
            self.adapter.h_metrics(neighbors, self.artifact),
        )
        slopes, distances = finite_distance_residual_slopes(
            self.adapter,
            [self.point],
            neighbors,
            np.zeros(len(neighbors), dtype=np.int64),
            center_residual,
            residuals,
        )
        self.assertTrue(np.all(np.isfinite(slopes)))
        self.assertTrue(np.all(slopes >= 0.0))
        self.assertTrue(np.all(distances > 0.0))


if __name__ == "__main__":
    unittest.main()
