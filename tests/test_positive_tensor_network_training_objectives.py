from __future__ import annotations

import unittest


class PositiveTensorNetworkTrainingObjectiveTests(unittest.TestCase):
    def test_inherited_bond_mask_freezes_only_the_old_subnetwork(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        from scripts.train_quintic_positive_tensor_network_same_points import (
            inherited_bond_gradient_mask,
        )

        first = torch.ones((1, 24, 25), dtype=torch.complex128)
        interior = torch.ones((24, 24, 25), dtype=torch.complex128)
        last = torch.ones((24, 1, 25), dtype=torch.complex128)
        first_mask = inherited_bond_gradient_mask(first, 12)
        interior_mask = inherited_bond_gradient_mask(interior, 12)
        last_mask = inherited_bond_gradient_mask(last, 12)

        self.assertEqual(int(torch.count_nonzero(first_mask)), 12 * 25)
        self.assertEqual(
            int(torch.count_nonzero(interior_mask)),
            (24 * 24 - 12 * 12) * 25,
        )
        self.assertEqual(int(torch.count_nonzero(last_mask)), 12 * 25)
        self.assertTrue(bool(torch.all(interior_mask[:12, :12] == 0)))
        self.assertTrue(bool(torch.all(interior_mask[12:, :] == 1)))
        self.assertTrue(bool(torch.all(interior_mask[:, 12:] == 1)))

    def test_inherited_bond_hook_blocks_old_block_gradients(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        from scripts.train_quintic_positive_tensor_network_same_points import (
            inherited_bond_gradient_mask,
        )

        parameter = torch.nn.Parameter(
            torch.ones((4, 4, 2), dtype=torch.complex128)
        )
        mask = inherited_bond_gradient_mask(parameter, 2)
        parameter.register_hook(lambda gradient: gradient * mask)
        torch.real(torch.sum(parameter)).backward()

        self.assertTrue(bool(torch.all(parameter.grad[:2, :2] == 0)))
        self.assertTrue(bool(torch.all(parameter.grad[2:, :] == 1)))
        self.assertTrue(bool(torch.all(parameter.grad[:, 2:] == 1)))

    def test_ma_point_losses_support_squared_and_absolute_objectives(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        from scripts.train_quintic_positive_tensor_network_same_points import (
            ma_point_losses_torch,
        )

        ratio = torch.tensor([0.8, 1.0, 1.3], dtype=torch.float64)
        residual = ratio - 1.0
        torch.testing.assert_close(
            ma_point_losses_torch(ratio, kind="squared"),
            torch.square(residual),
        )
        torch.testing.assert_close(
            ma_point_losses_torch(ratio, kind="absolute"),
            torch.abs(residual),
        )

    def test_absolute_ratio_tail_loss_matches_squared_ma_residual(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        from scripts.train_quintic_positive_tensor_network_same_points import (
            tail_point_losses_torch,
        )

        log_ratio = torch.tensor([-0.2, 0.0, 0.3], dtype=torch.float64)
        ratio = torch.exp(log_ratio)
        actual = tail_point_losses_torch(
            log_ratio,
            ratio,
            kind="absolute_ratio",
            ratio_threshold=1.5,
            smooth_temperature=0.05,
        )
        torch.testing.assert_close(actual, torch.square(ratio - 1.0))

    def test_continuation_can_introduce_or_remove_teacher(self) -> None:
        from scripts.train_type11_positive_tensor_network import (
            validate_initial_model_compatibility,
        )

        payload = {
            "adapter": "test_adapter",
            "site_count": 4,
            "precision": "complex128",
            "source_artifact_sha256": "source",
            "teacher_artifact_sha256": None,
            "target_degree": [4, 4],
            "target_normalization": 0.25,
        }
        transition = validate_initial_model_compatibility(
            payload,
            adapter_key="test_adapter",
            site_count=4,
            precision="complex128",
            source_artifact_sha256="source",
            teacher_artifact_sha256="energy-teacher",
            target_degree=(4, 4),
            target_normalization=0.25,
        )
        self.assertEqual(transition, "teacher_introduced")

        payload["teacher_artifact_sha256"] = "energy-teacher"
        transition = validate_initial_model_compatibility(
            payload,
            adapter_key="test_adapter",
            site_count=4,
            precision="complex128",
            source_artifact_sha256="source",
            teacher_artifact_sha256=None,
            target_degree=(4, 4),
            target_normalization=0.25,
        )
        self.assertEqual(transition, "teacher_removed")

    def test_continuation_rejects_different_non_null_teacher(self) -> None:
        from scripts.train_type11_positive_tensor_network import (
            validate_initial_model_compatibility,
        )

        payload = {
            "adapter": "test_adapter",
            "site_count": 4,
            "precision": "complex128",
            "source_artifact_sha256": "source",
            "teacher_artifact_sha256": "teacher-a",
            "target_degree": [4, 4],
            "target_normalization": 0.25,
        }
        with self.assertRaisesRegex(ValueError, "different non-null teachers"):
            validate_initial_model_compatibility(
                payload,
                adapter_key="test_adapter",
                site_count=4,
                precision="complex128",
                source_artifact_sha256="source",
                teacher_artifact_sha256="teacher-b",
                target_degree=(4, 4),
                target_normalization=0.25,
            )

    def test_teacher_free_components_match_direct_geometric_losses(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        from scripts.train_type11_positive_tensor_network import (
            distillation_components,
        )

        log_eigenvalues = torch.tensor(
            [
                [0.1, -0.2, 0.3],
                [0.4, 0.2, -0.1],
                [-0.3, 0.1, 0.2],
                [0.0, 0.5, -0.2],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        metric = torch.diag_embed(torch.exp(log_eigenvalues)).to(torch.complex128)
        potential = torch.zeros(4, dtype=torch.float64)
        log_omega = torch.tensor([0.2, -0.1, 0.0, 0.3], dtype=torch.float64)
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        fixed_log_kappa = torch.tensor(0.15, dtype=torch.float64)

        (
            potential_loss,
            metric_loss,
            log_energy_loss,
            ma_loss,
            tail_loss,
            model_log_eta,
        ) = distillation_components(
            potential,
            metric,
            teacher_potential=None,
            teacher_inverse_cholesky=None,
            log_omega=log_omega,
            weights=weights,
            fixed_log_kappa=fixed_log_kappa,
            tail_fraction=0.5,
            tail_ratio_threshold=1.1,
            tail_smooth_temperature=0.05,
        )

        expected_log_eta = torch.sum(log_eigenvalues, dim=1) - log_omega
        expected_log_ratio = expected_log_eta - fixed_log_kappa
        expected_log_energy = torch.sum(weights * expected_log_ratio**2)
        expected_ma = torch.sum(weights * (torch.exp(expected_log_ratio) - 1.0) ** 2)
        torch.testing.assert_close(model_log_eta, expected_log_eta)
        torch.testing.assert_close(log_energy_loss, expected_log_energy)
        torch.testing.assert_close(ma_loss, expected_ma)
        self.assertEqual(float(potential_loss), 0.0)
        self.assertEqual(float(metric_loss), 0.0)
        self.assertGreaterEqual(float(tail_loss), 0.0)

        (log_energy_loss + ma_loss + tail_loss).backward()
        self.assertIsNotNone(log_eigenvalues.grad)
        self.assertTrue(bool(torch.all(torch.isfinite(log_eigenvalues.grad))))


if __name__ == "__main__":
    unittest.main()
