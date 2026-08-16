from __future__ import annotations

import unittest

import numpy as np

from gcicy_metric.pipeline.positive_tensor_network import (
    PositiveTensorNetworkCoherentSumMetric,
    PositiveTensorNetworkDirectSumMetric,
    PositiveTensorNetworkMetric,
    PositiveTensorNetworkPositiveSumMetric,
    activate_one_sided_repeated_shared_dictionary_bridges,
    activate_repeated_shared_dictionary_bridges,
    anchored_orthonormal_physical_dictionary,
    canonical_matrix_unit_dictionary,
    compress_cores_to_shared_local_dictionary,
    expand_shared_local_dictionary_rank,
    pair_adjacent_dense_cores_to_symmetric_square,
    repeat_shared_dictionary_coefficient_cores,
    resize_shared_dictionary_coefficient_cores,
    symmetric_square_reference_h,
)


class PositiveTensorNetworkMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        self.torch = torch
        torch.manual_seed(20260717)
        rng = np.random.default_rng(20260717)
        self.batch = 7
        self.section_count = 3
        self.coordinate_count = 2
        factor = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        self.reference_h = factor @ factor.conjugate().T + np.eye(3)
        values = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
        derivatives = rng.normal(size=(7, 3, 2)) + 1j * rng.normal(size=(7, 3, 2))
        self.values = torch.tensor(values, dtype=torch.complex128)
        self.derivatives = torch.tensor(derivatives, dtype=torch.complex128)
        self.model = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=2,
            bond_dimension=2,
            target_normalization=0.25,
            positive_floor=0.1,
            initialization_noise=0.03,
            seed=20260718,
            dtype=torch.complex128,
        )

    def dense_product_features(self):
        values = self.values
        derivatives = self.derivatives
        product_values = self.torch.einsum("ni,nj->nij", values, values).reshape(
            self.batch, -1
        )
        product_derivatives = []
        for coordinate in range(self.coordinate_count):
            derivative = self.torch.einsum(
                "ni,nj->nij", derivatives[:, :, coordinate], values
            ) + self.torch.einsum(
                "ni,nj->nij", values, derivatives[:, :, coordinate]
            )
            product_derivatives.append(derivative.reshape(self.batch, -1))
        return product_values, self.torch.stack(product_derivatives, dim=2)

    def test_native_metric_matches_materialized_dense_h(self) -> None:
        dense_h = self.model.materialize_m2_h(maximum_section_count=4)
        product_values, product_derivatives = self.dense_product_features()
        h_values = self.torch.einsum("ab,nb->na", dense_h, product_values)
        denominator = self.torch.real(
            self.torch.einsum("na,na->n", self.torch.conj(product_values), h_values)
        )
        h_derivatives = self.torch.einsum(
            "ab,nbj->naj", dense_h, product_derivatives
        )
        first = self.torch.einsum(
            "nmi,nmj->nij",
            self.torch.conj(product_derivatives),
            h_derivatives,
        )
        gradient = self.torch.einsum(
            "nm,nmj->nj", self.torch.conj(product_values), h_derivatives
        )
        expected = first / denominator[:, None, None]
        expected -= (
            self.torch.conj(gradient)[:, :, None]
            * gradient[:, None, :]
            / denominator[:, None, None] ** 2
        )
        expected *= 0.25
        expected = 0.5 * (expected + self.torch.conj(expected.transpose(1, 2)))
        actual = self.model(self.values, self.derivatives)
        self.torch.testing.assert_close(actual, expected, rtol=1.0e-11, atol=1.0e-11)

    def test_zero_coherent_gate_recovers_leading_branch_exactly(self) -> None:
        floor = 0.07
        leading = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            positive_floor=floor,
            initialization_noise=0.04,
            seed=202607181,
            dtype=self.torch.complex128,
        )
        atom = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            positive_floor=0.0,
            initialization_noise=0.06,
            seed=202607182,
            dtype=self.torch.complex128,
        )
        coherent = PositiveTensorNetworkCoherentSumMetric(
            (leading, atom),
            positive_floor=floor,
            initial_new_gates=(0.0j,),
        )
        expected_potential, expected_metric = leading.potential_and_metric(
            self.values, self.derivatives
        )
        actual_potential, actual_metric = coherent.potential_and_metric(
            self.values, self.derivatives
        )
        self.torch.testing.assert_close(
            actual_potential, expected_potential, rtol=2.0e-11, atol=2.0e-11
        )
        self.torch.testing.assert_close(
            actual_metric, expected_metric, rtol=2.0e-10, atol=2.0e-10
        )

    def test_coherent_sum_matches_explicit_block_mps(self) -> None:
        floor = 0.03
        gate = 0.27 - 0.19j
        branches = tuple(
            PositiveTensorNetworkMetric(
                self.reference_h,
                site_count=3,
                bond_dimension=2,
                target_normalization=1.0 / 6.0,
                positive_floor=0.0,
                initialization_noise=0.05,
                seed=seed,
                dtype=self.torch.complex128,
            )
            for seed in (202607183, 202607184)
        )
        coherent = PositiveTensorNetworkCoherentSumMetric(
            branches,
            positive_floor=floor,
            initial_new_gates=(gate,),
        )
        combined = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=4,
            target_normalization=1.0 / 6.0,
            positive_floor=floor,
            initialization_noise=0.0,
            seed=202607185,
            dtype=self.torch.complex128,
        )
        with self.torch.no_grad():
            combined.cores[0].zero_()
            combined.cores[0][:, :2].copy_(branches[0].cores[0])
            combined.cores[0][:, 2:].copy_(gate * branches[1].cores[0])
            combined.cores[1].zero_()
            combined.cores[1][:2, :2].copy_(branches[0].cores[1])
            combined.cores[1][2:, 2:].copy_(branches[1].cores[1])
            combined.cores[2].zero_()
            combined.cores[2][:2].copy_(branches[0].cores[2])
            combined.cores[2][2:].copy_(branches[1].cores[2])
        coherent_potential, coherent_metric = coherent.potential_and_metric(
            self.values, self.derivatives
        )
        explicit_potential, explicit_metric = combined.potential_and_metric(
            self.values, self.derivatives
        )
        self.torch.testing.assert_close(
            coherent_potential,
            explicit_potential,
            rtol=3.0e-11,
            atol=3.0e-11,
        )
        self.torch.testing.assert_close(
            coherent_metric, explicit_metric, rtol=3.0e-10, atol=3.0e-10
        )

    def test_zero_gate_activates_gate_but_not_new_atom_cores(self) -> None:
        branches = tuple(
            PositiveTensorNetworkMetric(
                self.reference_h,
                site_count=3,
                bond_dimension=2,
                target_normalization=1.0 / 6.0,
                positive_floor=0.0,
                initialization_noise=0.08,
                seed=seed,
                dtype=self.torch.complex128,
            )
            for seed in (202607186, 202607187)
        )
        coherent = PositiveTensorNetworkCoherentSumMetric(
            branches,
            positive_floor=0.02,
            initial_new_gates=(0.0j,),
        )
        loss = self.torch.mean(
            self.torch.abs(coherent(self.values, self.derivatives)) ** 2
        )
        loss.backward()
        self.assertGreater(
            float(self.torch.linalg.vector_norm(coherent.new_gates.grad)),
            1.0e-12,
        )
        for core in branches[1].cores:
            self.assertIsNotNone(core.grad)
            self.assertEqual(float(self.torch.linalg.vector_norm(core.grad)), 0.0)

    def test_positive_sum_matches_same_source_direct_sum(self) -> None:
        floor = 0.025
        gate = 0.13 - 0.21j

        def make_branches(leading_floor: float):
            return (
                PositiveTensorNetworkMetric(
                    self.reference_h,
                    site_count=3,
                    bond_dimension=2,
                    target_normalization=1.0 / 6.0,
                    positive_floor=leading_floor,
                    initialization_noise=0.05,
                    seed=202607188,
                    dtype=self.torch.complex128,
                ),
                PositiveTensorNetworkMetric(
                    self.reference_h,
                    site_count=3,
                    bond_dimension=2,
                    target_normalization=1.0 / 6.0,
                    positive_floor=0.0,
                    initialization_noise=0.07,
                    seed=202607189,
                    dtype=self.torch.complex128,
                ),
            )

        positive = PositiveTensorNetworkPositiveSumMetric(
            make_branches(0.0),
            positive_floor=floor,
            initial_new_gates=(gate,),
        )
        direct = PositiveTensorNetworkDirectSumMetric(
            make_branches(floor),
            branch_total_degrees=(3, 3),
            branch_weights=(1.0, abs(gate) ** 2),
        )
        positive_potential, positive_metric = positive.potential_and_metric(
            self.values, self.derivatives
        )
        direct_potential, direct_metric = direct.potential_and_metric(
            (
                (self.values, self.derivatives),
                (self.values, self.derivatives),
            )
        )
        self.torch.testing.assert_close(
            positive_potential, direct_potential, rtol=3.0e-11, atol=3.0e-11
        )
        self.torch.testing.assert_close(
            positive_metric, direct_metric, rtol=3.0e-10, atol=3.0e-10
        )

    def test_zero_positive_sum_gate_has_dead_first_derivative(self) -> None:
        branches = tuple(
            PositiveTensorNetworkMetric(
                self.reference_h,
                site_count=3,
                bond_dimension=2,
                target_normalization=1.0 / 6.0,
                positive_floor=0.0,
                initialization_noise=0.08,
                seed=seed,
                dtype=self.torch.complex128,
            )
            for seed in (202607190, 202607191)
        )
        positive = PositiveTensorNetworkPositiveSumMetric(
            branches,
            positive_floor=0.02,
            initial_new_gates=(0.0j,),
        )
        loss = self.torch.mean(
            self.torch.abs(positive(self.values, self.derivatives)) ** 2
        )
        loss.backward()
        self.assertEqual(
            float(self.torch.linalg.vector_norm(positive.new_gates.grad)), 0.0
        )
        for core in branches[1].cores:
            self.assertIsNotNone(core.grad)
            self.assertEqual(float(self.torch.linalg.vector_norm(core.grad)), 0.0)

    def test_streaming_moments_match_explicit_derivative_states_at_three_sites(
        self,
    ) -> None:
        model = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            positive_floor=0.0,
            initialization_noise=0.03,
            seed=20260719,
            dtype=self.torch.complex128,
        )
        base_sites = model._state_sites(self.values)
        derivative_states = [
            [
                model._state_sites(
                    self.values,
                    replacement=(site, self.derivatives[:, :, coordinate]),
                )
                for site in range(model.site_count)
            ]
            for coordinate in range(self.coordinate_count)
        ]
        expected_norm = self.torch.real(model._mps_overlap(base_sites, base_sites))
        expected_gradient = self.torch.zeros(
            (self.batch, self.coordinate_count), dtype=self.torch.complex128
        )
        expected_mixed = self.torch.zeros(
            (self.batch, self.coordinate_count, self.coordinate_count),
            dtype=self.torch.complex128,
        )
        for holomorphic in range(self.coordinate_count):
            for ket_state in derivative_states[holomorphic]:
                expected_gradient[:, holomorphic] += model._mps_overlap(
                    base_sites, ket_state
                )
        for antiholomorphic in range(self.coordinate_count):
            for holomorphic in range(self.coordinate_count):
                for bra_state in derivative_states[antiholomorphic]:
                    for ket_state in derivative_states[holomorphic]:
                        expected_mixed[:, antiholomorphic, holomorphic] += (
                            model._mps_overlap(bra_state, ket_state)
                        )

        actual = model.feature_moments(self.values, self.derivatives)
        self.torch.testing.assert_close(
            actual.norm, expected_norm, rtol=1.0e-11, atol=1.0e-11
        )
        self.torch.testing.assert_close(
            actual.holomorphic_gradient,
            expected_gradient,
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        self.torch.testing.assert_close(
            actual.mixed_hessian,
            expected_mixed,
            rtol=1.0e-11,
            atol=1.0e-11,
        )

    def test_vectorized_transfer_matches_scalar_values_and_gradients(self) -> None:
        dictionary = anchored_orthonormal_physical_dictionary(
            self.reference_h,
            4,
            seed=20260729,
        )
        vectorized = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=4,
            bond_dimension=2,
            target_normalization=0.125,
            positive_floor=0.1,
            initialization_noise=0.03,
            physical_dictionary=dictionary,
            trainable_physical_dictionary=True,
            transfer_implementation="vectorized",
            seed=20260730,
            dtype=self.torch.complex128,
        )
        scalar = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=4,
            bond_dimension=2,
            target_normalization=0.125,
            positive_floor=0.1,
            initialization_noise=0.0,
            physical_dictionary=dictionary,
            trainable_physical_dictionary=True,
            transfer_implementation="scalar",
            seed=20260731,
            dtype=self.torch.complex128,
        )
        scalar.load_state_dict(vectorized.state_dict())

        vectorized_potential, vectorized_metric = vectorized.potential_and_metric(
            self.values,
            self.derivatives,
        )
        scalar_potential, scalar_metric = scalar.potential_and_metric(
            self.values,
            self.derivatives,
        )
        self.torch.testing.assert_close(
            vectorized_potential,
            scalar_potential,
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        self.torch.testing.assert_close(
            vectorized_metric,
            scalar_metric,
            rtol=1.0e-11,
            atol=1.0e-11,
        )

        vectorized_loss = self.torch.mean(self.torch.abs(vectorized_metric) ** 2)
        scalar_loss = self.torch.mean(self.torch.abs(scalar_metric) ** 2)
        vectorized_loss.backward()
        scalar_loss.backward()
        for vectorized_parameter, scalar_parameter in zip(
            vectorized.parameters(), scalar.parameters(), strict=True
        ):
            self.torch.testing.assert_close(
                vectorized_parameter.grad,
                scalar_parameter.grad,
                rtol=2.0e-10,
                atol=2.0e-10,
            )

    def test_two_site_supercore_activation_and_exact_split_preserve_metric(self) -> None:
        dictionary = anchored_orthonormal_physical_dictionary(
            self.reference_h,
            4,
            seed=202607301,
        )
        model = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=4,
            bond_dimension=3,
            target_normalization=0.125,
            positive_floor=0.1,
            initialization_noise=0.03,
            physical_dictionary=dictionary,
            transfer_implementation="vectorized",
            seed=202607302,
            dtype=self.torch.complex128,
        )
        expected_potential, expected_metric = model.potential_and_metric(
            self.values,
            self.derivatives,
        )
        model.mixed_canonicalize_coefficient_pair_(1)
        canonical_potential, canonical_metric = model.potential_and_metric(
            self.values,
            self.derivatives,
        )
        self.torch.testing.assert_close(
            canonical_potential, expected_potential, rtol=2.0e-11, atol=2.0e-11
        )
        self.torch.testing.assert_close(
            canonical_metric, expected_metric, rtol=2.0e-10, atol=2.0e-10
        )

        model.requires_grad_(False)
        supercore = model.activate_two_site_coefficient_core_(1)
        supercore.requires_grad_(True)
        active_potential, active_metric = model.potential_and_metric(
            self.values,
            self.derivatives,
        )
        self.torch.testing.assert_close(
            active_potential, canonical_potential, rtol=2.0e-11, atol=2.0e-11
        )
        self.torch.testing.assert_close(
            active_metric, canonical_metric, rtol=2.0e-10, atol=2.0e-10
        )
        self.torch.mean(self.torch.abs(active_metric) ** 2).backward()
        self.assertIsNotNone(supercore.grad)
        self.assertGreater(float(self.torch.linalg.vector_norm(supercore.grad)), 1.0e-12)

        left, right, split = model.split_two_site_coefficient_core(
            relative_singular_value_cutoff=1.0e-12,
            direction="right",
        )
        self.assertEqual(split["retained_rank"], 3)
        self.assertLess(split["discarded_relative_weight"], 1.0e-24)
        model.commit_two_site_coefficient_split_(left, right)
        final_potential, final_metric = model.potential_and_metric(
            self.values,
            self.derivatives,
        )
        self.torch.testing.assert_close(
            final_potential, canonical_potential, rtol=2.0e-11, atol=2.0e-11
        )
        self.torch.testing.assert_close(
            final_metric, canonical_metric, rtol=2.0e-10, atol=2.0e-10
        )
        self.assertNotIn("two_site_core", model.state_dict())

    def test_active_two_site_vectorized_transfer_matches_scalar(self) -> None:
        dictionary = anchored_orthonormal_physical_dictionary(
            self.reference_h,
            4,
            seed=202607303,
        )
        vectorized = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=4,
            bond_dimension=3,
            target_normalization=0.125,
            positive_floor=0.1,
            initialization_noise=0.03,
            physical_dictionary=dictionary,
            transfer_implementation="vectorized",
            seed=202607304,
            dtype=self.torch.complex128,
        )
        scalar = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=4,
            bond_dimension=3,
            target_normalization=0.125,
            positive_floor=0.1,
            initialization_noise=0.0,
            physical_dictionary=dictionary,
            transfer_implementation="scalar",
            seed=202607305,
            dtype=self.torch.complex128,
        )
        scalar.load_state_dict(vectorized.state_dict())
        for model in (vectorized, scalar):
            model.mixed_canonicalize_coefficient_pair_(1)
            model.requires_grad_(False)
            model.activate_two_site_coefficient_core_(1).requires_grad_(True)

        vectorized_potential, vectorized_metric = vectorized.potential_and_metric(
            self.values,
            self.derivatives,
        )
        scalar_potential, scalar_metric = scalar.potential_and_metric(
            self.values,
            self.derivatives,
        )
        self.torch.testing.assert_close(
            vectorized_potential, scalar_potential, rtol=2.0e-11, atol=2.0e-11
        )
        self.torch.testing.assert_close(
            vectorized_metric, scalar_metric, rtol=2.0e-10, atol=2.0e-10
        )
        self.torch.mean(self.torch.abs(vectorized_metric) ** 2).backward()
        self.torch.mean(self.torch.abs(scalar_metric) ** 2).backward()
        self.torch.testing.assert_close(
            vectorized.two_site_core.grad,
            scalar.two_site_core.grad,
            rtol=3.0e-10,
            atol=3.0e-10,
        )

    def test_three_site_supercore_preserves_metric_and_transfer_gradients(self) -> None:
        dictionary = anchored_orthonormal_physical_dictionary(
            self.reference_h,
            4,
            seed=202607309,
        )
        vectorized = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=5,
            bond_dimension=3,
            target_normalization=0.1,
            positive_floor=0.1,
            initialization_noise=0.03,
            physical_dictionary=dictionary,
            transfer_implementation="vectorized",
            seed=202607310,
            dtype=self.torch.complex128,
        )
        scalar = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=5,
            bond_dimension=3,
            target_normalization=0.1,
            positive_floor=0.1,
            initialization_noise=0.0,
            physical_dictionary=dictionary,
            transfer_implementation="scalar",
            seed=202607311,
            dtype=self.torch.complex128,
        )
        scalar.load_state_dict(vectorized.state_dict())
        expected_potential, expected_metric = vectorized.potential_and_metric(
            self.values,
            self.derivatives,
        )
        for model in (vectorized, scalar):
            model.mixed_canonicalize_coefficient_block_(1, 3)
            model.requires_grad_(False)
            model.activate_three_site_coefficient_core_(1).requires_grad_(True)

        vectorized_potential, vectorized_metric = vectorized.potential_and_metric(
            self.values,
            self.derivatives,
        )
        scalar_potential, scalar_metric = scalar.potential_and_metric(
            self.values,
            self.derivatives,
        )
        self.torch.testing.assert_close(
            vectorized_potential,
            expected_potential,
            rtol=3.0e-11,
            atol=3.0e-11,
        )
        self.torch.testing.assert_close(
            vectorized_metric,
            expected_metric,
            rtol=3.0e-10,
            atol=3.0e-10,
        )
        self.torch.testing.assert_close(
            scalar_potential,
            vectorized_potential,
            rtol=3.0e-11,
            atol=3.0e-11,
        )
        self.torch.testing.assert_close(
            scalar_metric,
            vectorized_metric,
            rtol=3.0e-10,
            atol=3.0e-10,
        )
        self.torch.mean(self.torch.abs(vectorized_metric) ** 2).backward()
        self.torch.mean(self.torch.abs(scalar_metric) ** 2).backward()
        self.torch.testing.assert_close(
            vectorized.three_site_core.grad,
            scalar.three_site_core.grad,
            rtol=4.0e-10,
            atol=4.0e-10,
        )
        vectorized.clear_three_site_coefficient_core_()
        self.assertNotIn("three_site_core", vectorized.state_dict())

    def test_two_site_split_detects_rank_growth_before_truncation(self) -> None:
        dictionary = anchored_orthonormal_physical_dictionary(
            self.reference_h,
            4,
            seed=202607306,
        )
        model = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=4,
            bond_dimension=3,
            target_normalization=0.125,
            positive_floor=0.1,
            initialization_noise=0.0,
            physical_dictionary=dictionary,
            seed=202607307,
            dtype=self.torch.complex128,
        )
        model.mixed_canonicalize_coefficient_pair_(1)
        model.activate_two_site_coefficient_core_(1)
        generator = self.torch.Generator().manual_seed(202607308)
        real = self.torch.randn(
            model.two_site_core.shape,
            generator=generator,
            dtype=self.torch.float64,
        )
        imaginary = self.torch.randn(
            model.two_site_core.shape,
            generator=generator,
            dtype=self.torch.float64,
        )
        candidate = model.two_site_core + 1.0e-2 * (real + 1j * imaginary)
        _, _, split = model.split_two_site_coefficient_core(
            candidate,
            relative_singular_value_cutoff=1.0e-12,
        )
        self.assertGreater(split["numerical_rank"], split["bond_capacity"])
        self.assertEqual(split["retained_rank"], split["bond_capacity"])
        self.assertGreater(split["discarded_relative_weight"], 0.0)

    def test_potential_has_projective_scaling_law(self) -> None:
        scale = 1.7 - 0.4j
        original = self.model.potential(self.values)
        scaled = self.model.potential(scale * self.values)
        expected_shift = (
            self.model.site_count
            * self.model.target_normalization
            * np.log(abs(scale) ** 2)
        )
        self.torch.testing.assert_close(
            scaled - original,
            self.torch.full_like(original, expected_shift),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_metric_is_invariant_under_constant_frame_rescaling(self) -> None:
        scale = -0.6 + 1.3j
        original = self.model(self.values, self.derivatives)
        scaled = self.model(scale * self.values, scale * self.derivatives)
        self.torch.testing.assert_close(scaled, original, rtol=1.0e-11, atol=1.0e-11)

    def test_parameter_count_is_linear_in_site_count_at_fixed_bond(self) -> None:
        model_two = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=2,
            bond_dimension=2,
            target_normalization=0.25,
            dtype=self.torch.complex128,
        )
        model_three = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            dtype=self.torch.complex128,
        )
        expected_two_complex = 2 * 2 * self.section_count**2
        expected_three_complex = (2 + 4 + 2) * self.section_count**2
        self.assertEqual(model_two.trainable_real_parameter_count, 2 * expected_two_complex)
        self.assertEqual(
            model_three.trainable_real_parameter_count,
            2 * expected_three_complex,
        )

    def test_metric_loss_backpropagates_to_every_core(self) -> None:
        models = [
            self.model,
            PositiveTensorNetworkMetric(
                self.reference_h,
                site_count=3,
                bond_dimension=2,
                target_normalization=1.0 / 6.0,
                positive_floor=0.1,
                initialization_noise=0.03,
                seed=20260720,
                dtype=self.torch.complex128,
            ),
        ]
        for model in models:
            metric = model(self.values, self.derivatives)
            loss = self.torch.mean(self.torch.abs(metric) ** 2)
            loss.backward()
            for core in model.cores:
                self.assertIsNotNone(core.grad)
                self.assertTrue(bool(self.torch.all(self.torch.isfinite(core.grad))))
                self.assertGreater(
                    float(self.torch.linalg.vector_norm(core.grad)), 0.0
                )

    def test_full_rank_shared_dictionary_matches_dense_cores(self) -> None:
        dense = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            positive_floor=0.1,
            initialization_noise=0.03,
            seed=20260721,
            dtype=self.torch.complex128,
        )
        dense_cores = tuple(
            core.detach().cpu().numpy() for core in dense.materialized_cores()
        )
        total_local_matrices = sum(core.shape[0] * core.shape[1] for core in dense_cores)
        compression = compress_cores_to_shared_local_dictionary(
            dense_cores,
            rank=min(total_local_matrices, self.section_count**2),
        )
        dictionary = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            positive_floor=0.1,
            initialization_noise=0.0,
            physical_dictionary=compression.physical_dictionary,
            dtype=self.torch.complex128,
        )
        with self.torch.no_grad():
            for target, source in zip(
                dictionary.coefficient_cores,
                compression.coefficient_cores,
                strict=True,
            ):
                target.copy_(self.torch.tensor(source, dtype=self.torch.complex128))
        self.assertLess(compression.relative_frobenius_error, 1.0e-7)
        actual_potential, actual_metric = dictionary.potential_and_metric(
            self.values,
            self.derivatives,
        )
        expected_potential, expected_metric = dense.potential_and_metric(
            self.values,
            self.derivatives,
        )
        self.torch.testing.assert_close(
            actual_potential, expected_potential, rtol=1.0e-11, atol=1.0e-11
        )
        self.torch.testing.assert_close(
            actual_metric, expected_metric, rtol=1.0e-10, atol=1.0e-10
        )

    def test_fixed_canonical_dictionary_fast_path_matches_generic_gradients(
        self,
    ) -> None:
        model = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=2,
            output_dimension=4,
            target_normalization=1.0 / 6.0,
            positive_floor=0.1,
            initialization_noise=0.03,
            physical_dictionary=canonical_matrix_unit_dictionary(4, 3),
            trainable_physical_dictionary=False,
            transfer_implementation="vectorized",
            seed=20260743,
            dtype=self.torch.complex128,
        )
        self.assertTrue(model.fixed_canonical_matrix_units)

        model.fixed_canonical_matrix_units = False
        expected_potential, expected_metric = model.potential_and_metric(
            self.values,
            self.derivatives,
        )
        self.torch.mean(self.torch.abs(expected_metric) ** 2).backward()
        expected_gradients = [
            parameter.grad.detach().clone() for parameter in model.parameters()
        ]
        for parameter in model.parameters():
            parameter.grad = None

        model.fixed_canonical_matrix_units = True
        actual_potential, actual_metric = model.potential_and_metric(
            self.values,
            self.derivatives,
        )
        self.torch.mean(self.torch.abs(actual_metric) ** 2).backward()

        self.torch.testing.assert_close(
            actual_potential,
            expected_potential,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        self.torch.testing.assert_close(
            actual_metric,
            expected_metric,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        for parameter, expected_gradient in zip(
            model.parameters(),
            expected_gradients,
            strict=True,
        ):
            self.torch.testing.assert_close(
                parameter.grad,
                expected_gradient,
                rtol=2.0e-11,
                atol=2.0e-11,
            )

    def test_symmetric_square_pairing_preserves_metric_with_rectangular_output(
        self,
    ) -> None:
        source = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=4,
            bond_dimension=2,
            target_normalization=0.25,
            positive_floor=0.1,
            initialization_noise=0.03,
            seed=20260734,
            dtype=self.torch.complex128,
        )
        paired_cores = pair_adjacent_dense_cores_to_symmetric_square(
            tuple(
                core.detach().cpu().numpy() for core in source.materialized_cores()
            )
        )
        maximum_rank = min(
            sum(core.shape[0] * core.shape[1] for core in paired_cores),
            paired_cores[0].shape[2] * paired_cores[0].shape[3],
        )
        compression = compress_cores_to_shared_local_dictionary(
            paired_cores,
            maximum_rank,
        )
        paired_reference = symmetric_square_reference_h(self.reference_h)
        target = PositiveTensorNetworkMetric(
            paired_reference,
            site_count=2,
            bond_dimension=2,
            target_normalization=0.25,
            output_dimension=self.section_count**2,
            positive_floor=0.1,
            initialization_noise=0.0,
            physical_dictionary=compression.physical_dictionary,
            dtype=self.torch.complex128,
        )
        with self.torch.no_grad():
            for target_core, source_core in zip(
                target.coefficient_cores,
                compression.coefficient_cores,
                strict=True,
            ):
                target_core.copy_(
                    self.torch.tensor(source_core, dtype=self.torch.complex128)
                )

        values = self.values.detach().cpu().numpy()
        derivatives = self.derivatives.detach().cpu().numpy()
        paired_values = []
        paired_derivatives = []
        for left in range(self.section_count):
            for right in range(left, self.section_count):
                normalization = 1.0 if left == right else np.sqrt(2.0)
                paired_values.append(
                    normalization * values[:, left] * values[:, right]
                )
                paired_derivatives.append(
                    normalization
                    * (
                        derivatives[:, left, :] * values[:, right, None]
                        + values[:, left, None] * derivatives[:, right, :]
                    )
                )
        paired_values_tensor = self.torch.tensor(
            np.stack(paired_values, axis=1), dtype=self.torch.complex128
        )
        paired_derivatives_tensor = self.torch.tensor(
            np.stack(paired_derivatives, axis=1), dtype=self.torch.complex128
        )

        expected_potential, expected_metric = source.potential_and_metric(
            self.values,
            self.derivatives,
        )
        actual_potential, actual_metric = target.potential_and_metric(
            paired_values_tensor,
            paired_derivatives_tensor,
        )
        self.assertEqual(target.output_dimension, self.section_count**2)
        self.assertEqual(
            tuple(compression.physical_dictionary.shape[1:]),
            (self.section_count**2, self.section_count * (self.section_count + 1) // 2),
        )
        self.assertLess(compression.relative_frobenius_error, 1.0e-12)
        self.torch.testing.assert_close(
            actual_potential,
            expected_potential,
            rtol=2.0e-11,
            atol=2.0e-11,
        )
        self.torch.testing.assert_close(
            actual_metric,
            expected_metric,
            rtol=2.0e-10,
            atol=2.0e-10,
        )

    def test_fixed_dictionary_parameter_count_and_gradients(self) -> None:
        rng = np.random.default_rng(20260722)
        physical_dictionary = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(
            size=(2, 3, 3)
        )
        fixed = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            physical_dictionary=physical_dictionary,
            dtype=self.torch.complex128,
        )
        trainable = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            physical_dictionary=physical_dictionary,
            trainable_physical_dictionary=True,
            dtype=self.torch.complex128,
        )
        coefficient_complex_count = (2 + 4 + 2) * 2
        dictionary_complex_count = 2 * self.section_count**2
        self.assertEqual(
            fixed.trainable_real_parameter_count,
            2 * coefficient_complex_count,
        )
        self.assertEqual(
            trainable.trainable_real_parameter_count,
            2 * (coefficient_complex_count + dictionary_complex_count),
        )
        before = tuple(core.detach().clone() for core in trainable.materialized_cores())
        trainable.orthonormalize_physical_dictionary_()
        after = trainable.materialized_cores()
        for expected, actual in zip(before, after, strict=True):
            self.torch.testing.assert_close(
                actual,
                expected,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        flattened = trainable.physical_dictionary.reshape(2, -1)
        row_gram = flattened @ self.torch.conj(flattened.T)
        self.torch.testing.assert_close(
            row_gram,
            self.torch.eye(2, dtype=self.torch.complex128),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        loss = self.torch.mean(self.torch.abs(fixed(self.values, self.derivatives)) ** 2)
        loss.backward()
        for core in fixed.coefficient_cores:
            self.assertIsNotNone(core.grad)
            self.assertTrue(bool(self.torch.all(self.torch.isfinite(core.grad))))
        self.assertFalse(fixed.physical_dictionary.requires_grad)

    def test_nonzero_core_noise_activates_dormant_bond_gradients(self) -> None:
        dormant = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=4,
            bond_dimension=2,
            target_normalization=1.0 / 8.0,
            initialization_noise=0.0,
            seed=20260726,
            dtype=self.torch.complex128,
        )
        dormant_loss = self.torch.mean(
            self.torch.abs(dormant(self.values, self.derivatives)) ** 2
        )
        dormant_loss.backward()
        self.assertEqual(
            float(self.torch.linalg.vector_norm(dormant.cores[0].grad[0, 1])),
            0.0,
        )
        self.assertEqual(
            float(self.torch.linalg.vector_norm(dormant.cores[1].grad[1, 0])),
            0.0,
        )

        active = PositiveTensorNetworkMetric(
            self.reference_h,
            site_count=4,
            bond_dimension=2,
            target_normalization=1.0 / 8.0,
            initialization_noise=1.0e-2,
            seed=20260726,
            dtype=self.torch.complex128,
        )
        active_loss = self.torch.mean(
            self.torch.abs(active(self.values, self.derivatives)) ** 2
        )
        active_loss.backward()
        self.assertGreater(
            float(self.torch.linalg.vector_norm(active.cores[0].grad[0, 1])),
            1.0e-12,
        )
        self.assertGreater(
            float(self.torch.linalg.vector_norm(active.cores[1].grad[1, 0])),
            1.0e-12,
        )

    def test_shared_dictionary_site_resize_preserves_boundaries_and_interpolates(self) -> None:
        left = np.arange(6, dtype=float).reshape(1, 2, 3).astype(np.complex128)
        interior_left = np.ones((2, 2, 3), dtype=np.complex128)
        interior_right = 4.0 * np.ones((2, 2, 3), dtype=np.complex128)
        right = np.arange(6, dtype=float).reshape(2, 1, 3).astype(np.complex128)
        resized = resize_shared_dictionary_coefficient_cores(
            (left, interior_left, interior_right, right),
            target_site_count=6,
        )
        self.assertEqual(len(resized), 6)
        np.testing.assert_array_equal(resized[0], left)
        np.testing.assert_array_equal(resized[-1], right)
        np.testing.assert_allclose(resized[1], interior_left)
        np.testing.assert_allclose(resized[2], 2.0 * np.ones((2, 2, 3)))
        np.testing.assert_allclose(resized[3], 3.0 * np.ones((2, 2, 3)))
        np.testing.assert_allclose(resized[4], interior_right)

    def test_shared_dictionary_site_repetition_preserves_metric(self) -> None:
        dictionary = anchored_orthonormal_physical_dictionary(
            np.eye(self.section_count),
            self.section_count**2,
            seed=20260728,
        )
        source = PositiveTensorNetworkMetric(
            np.eye(self.section_count),
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 3.0,
            positive_floor=0.0,
            initialization_noise=3.0e-2,
            physical_dictionary=dictionary,
            trainable_physical_dictionary=True,
            seed=20260729,
            dtype=self.torch.complex128,
        )
        repeated_coefficients = repeat_shared_dictionary_coefficient_cores(
            tuple(
                core.detach().cpu().numpy() for core in source.coefficient_cores
            ),
            2,
        )
        repeated = PositiveTensorNetworkMetric(
            np.eye(self.section_count),
            site_count=6,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            positive_floor=0.0,
            initialization_noise=0.0,
            physical_dictionary=source.physical_dictionary.detach().cpu().numpy(),
            trainable_physical_dictionary=True,
            seed=20260730,
            dtype=self.torch.complex128,
        )
        with self.torch.no_grad():
            for target, value in zip(
                repeated.coefficient_cores,
                repeated_coefficients,
                strict=True,
            ):
                target.copy_(self.torch.tensor(value, dtype=self.torch.complex128))

        source_moments = source.feature_moments(self.values, self.derivatives)
        repeated_moments = repeated.feature_moments(self.values, self.derivatives)
        self.torch.testing.assert_close(
            repeated_moments.norm,
            source_moments.norm**2,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        self.torch.testing.assert_close(
            repeated(self.values, self.derivatives),
            source(self.values, self.derivatives),
            rtol=2.0e-11,
            atol=2.0e-11,
        )

    def test_high_degree_metric_is_stable_in_complex64(self) -> None:
        values = self.values.to(dtype=self.torch.complex64) * 30.0
        derivatives = self.derivatives.to(dtype=self.torch.complex64) * 30.0
        model = PositiveTensorNetworkMetric(
            np.eye(self.section_count),
            site_count=40,
            bond_dimension=2,
            target_normalization=1.0 / 40.0,
            positive_floor=1.0e-4,
            initialization_noise=0.0,
            seed=20260731,
            dtype=self.torch.complex64,
        )
        metric = model(values, derivatives)
        self.assertTrue(bool(self.torch.all(self.torch.isfinite(metric))))
        reference = PositiveTensorNetworkMetric(
            np.eye(self.section_count),
            site_count=4,
            bond_dimension=2,
            target_normalization=1.0 / 4.0,
            positive_floor=1.0e-4,
            initialization_noise=0.0,
            seed=20260731,
            dtype=self.torch.complex64,
        )
        self.torch.testing.assert_close(
            metric,
            reference(values, derivatives),
            rtol=2.0e-4,
            atol=2.0e-4,
        )

    def test_repeated_bridge_activation_is_deterministic_and_local(self) -> None:
        rng = np.random.default_rng(20260732)
        source = (
            rng.normal(size=(1, 3, 4)).astype(np.complex128),
            rng.normal(size=(3, 3, 4)).astype(np.complex128),
            rng.normal(size=(3, 1, 4)).astype(np.complex128),
        )
        repeated = repeat_shared_dictionary_coefficient_cores(source, 2)
        activated = activate_repeated_shared_dictionary_bridges(
            repeated,
            source_site_count=3,
            relative_noise=1.0e-2,
            seed=20260733,
        )
        repeated_again = activate_repeated_shared_dictionary_bridges(
            repeated,
            source_site_count=3,
            relative_noise=1.0e-2,
            seed=20260733,
        )
        for first, second in zip(activated, repeated_again, strict=True):
            np.testing.assert_array_equal(first, second)
        for index in (0, 1, 4, 5):
            np.testing.assert_array_equal(activated[index], repeated[index])
        np.testing.assert_array_equal(activated[2][:, 0, :], repeated[2][:, 0, :])
        np.testing.assert_array_equal(activated[3][0, :, :], repeated[3][0, :, :])
        self.assertGreater(np.linalg.norm(activated[2][:, 1:, :]), 0.0)
        self.assertGreater(np.linalg.norm(activated[3][1:, :, :]), 0.0)

    def test_one_sided_repeated_bridge_activation_preserves_metric_and_gradient(
        self,
    ) -> None:
        dictionary = anchored_orthonormal_physical_dictionary(
            np.eye(self.section_count),
            self.section_count**2,
            seed=20260734,
        )
        source = PositiveTensorNetworkMetric(
            np.eye(self.section_count),
            site_count=3,
            bond_dimension=2,
            target_normalization=1.0 / 3.0,
            positive_floor=0.0,
            initialization_noise=3.0e-2,
            physical_dictionary=dictionary,
            seed=20260735,
            dtype=self.torch.complex128,
        )
        repeated_coefficients = repeat_shared_dictionary_coefficient_cores(
            tuple(
                core.detach().cpu().numpy() for core in source.coefficient_cores
            ),
            2,
        )
        activated_coefficients = (
            activate_one_sided_repeated_shared_dictionary_bridges(
                repeated_coefficients,
                source_site_count=3,
                relative_scale=1.0e-2,
                seed=20260736,
            )
        )
        repeated = PositiveTensorNetworkMetric(
            np.eye(self.section_count),
            site_count=6,
            bond_dimension=2,
            target_normalization=1.0 / 6.0,
            positive_floor=0.0,
            initialization_noise=0.0,
            physical_dictionary=dictionary,
            seed=20260737,
            dtype=self.torch.complex128,
        )
        with self.torch.no_grad():
            for target, value in zip(
                repeated.coefficient_cores,
                activated_coefficients,
                strict=True,
            ):
                target.copy_(self.torch.tensor(value, dtype=self.torch.complex128))

        self.torch.testing.assert_close(
            repeated(self.values, self.derivatives),
            source(self.values, self.derivatives),
            rtol=2.0e-11,
            atol=2.0e-11,
        )
        repeated(self.values, self.derivatives).real.sum().backward()
        incoming_gradient = repeated.coefficient_cores[3].grad[1:, :, :]
        self.assertGreater(float(self.torch.linalg.vector_norm(incoming_gradient)), 0.0)

    def test_dictionary_rank_expansion_preserves_dense_cores(self) -> None:
        rng = np.random.default_rng(20260723)
        dictionary = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(
            size=(3, 4, 4)
        )
        coefficients = (
            rng.normal(size=(1, 2, 3)) + 1j * rng.normal(size=(1, 2, 3)),
            rng.normal(size=(2, 2, 3)) + 1j * rng.normal(size=(2, 2, 3)),
            rng.normal(size=(2, 1, 3)) + 1j * rng.normal(size=(2, 1, 3)),
        )
        expansion = expand_shared_local_dictionary_rank(
            dictionary,
            coefficients,
            7,
            seed=20260724,
        )
        self.assertEqual(expansion.physical_dictionary.shape, (7, 4, 4))
        self.assertLess(expansion.relative_core_error, 1.0e-12)
        self.assertLess(expansion.row_orthonormality_error, 1.0e-12)
        flattened = expansion.physical_dictionary.reshape(7, -1)
        np.testing.assert_allclose(
            flattened @ flattened.conjugate().T,
            np.eye(7),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        for source, expanded in zip(
            coefficients,
            expansion.coefficient_cores,
            strict=True,
        ):
            expected = np.einsum("lrq,qpi->lrpi", source, dictionary)
            actual = np.einsum(
                "lrq,qpi->lrpi",
                expanded,
                expansion.physical_dictionary,
            )
            np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_anchored_dictionary_contains_normalized_anchor(self) -> None:
        rng = np.random.default_rng(20260725)
        anchor = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        dictionary = anchored_orthonormal_physical_dictionary(
            anchor,
            6,
            seed=20260726,
        )
        np.testing.assert_allclose(
            dictionary[0],
            anchor / np.linalg.norm(anchor),
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        flattened = dictionary.reshape(6, -1)
        np.testing.assert_allclose(
            flattened @ flattened.conjugate().T,
            np.eye(6),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
