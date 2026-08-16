from __future__ import annotations

import numpy as np
import torch

from gcicy_metric.pipeline.exact_lift_tree import dense_h_potential_and_metric
from gcicy_metric.pipeline.positive_multiplication_tree import (
    PositiveMultiplicationTreeMetric,
    balanced_binary_topology,
    edge_parent_gradient_scores,
    expand_multiplication_tree_bonds,
    five_leaf_two_three_topology,
    reassociate_five_leaf_tree_to_two_three,
)


def positive_h(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    matrix = factor.conj().T @ factor + np.eye(size)
    return matrix * (size / np.trace(matrix).real)


def section_jets(
    *,
    batch: int,
    sections: int,
    coordinates: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(batch, sections)) + 1j * rng.normal(
        size=(batch, sections)
    )
    derivatives = rng.normal(size=(batch, sections, coordinates)) + 1j * rng.normal(
        size=(batch, sections, coordinates)
    )
    return (
        torch.tensor(values, dtype=torch.complex128),
        torch.tensor(derivatives, dtype=torch.complex128),
    )


def test_balanced_topology_is_connected_and_postordered() -> None:
    topology = balanced_binary_topology(5)
    assert topology.children == ((0, 1), (2, 3), (5, 6), (7, 4))
    assert topology.root == 8
    parent_map = topology.parent_map()
    assert set(parent_map) == set(range(8))
    for parent_index, (left, right) in enumerate(
        topology.children,
        start=topology.leaf_count,
    ):
        assert left < parent_index
        assert right < parent_index


def test_five_leaf_two_three_topology_is_connected_and_postordered() -> None:
    topology = five_leaf_two_three_topology()
    assert topology.children == ((0, 1), (2, 3), (6, 4), (5, 7))
    assert topology.root == 8
    assert set(topology.parent_map()) == set(range(8))


def test_five_leaf_reassociation_preserves_the_complete_metric() -> None:
    model = PositiveMultiplicationTreeMetric(
        positive_h(4, 202607296),
        leaf_count=5,
        bond_dimension=3,
        source_normalization=0.047,
        positive_floor=1.0e-7,
        shared_leaf=True,
    )
    generator = torch.Generator().manual_seed(202607297)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    generator=generator,
                )
            )
    values, derivatives = section_jets(
        batch=19,
        sections=4,
        coordinates=3,
        seed=202607298,
    )
    before_potential, before_metric = model.potential_and_metric(
        values,
        derivatives,
    )
    reassociated = reassociate_five_leaf_tree_to_two_three(model)
    after_potential, after_metric = reassociated.potential_and_metric(
        values,
        derivatives,
    )
    assert reassociated.topology == five_leaf_two_three_topology()
    torch.testing.assert_close(
        after_potential,
        before_potential,
        rtol=4.0e-12,
        atol=4.0e-12,
    )
    torch.testing.assert_close(
        after_metric,
        before_metric,
        rtol=8.0e-12,
        atol=8.0e-12,
    )


def test_rank_one_tree_exactly_reproduces_dense_source_metric() -> None:
    reference_h = positive_h(4, 202607284)
    source_normalization = 0.071
    model = PositiveMultiplicationTreeMetric(
        reference_h,
        leaf_count=5,
        bond_dimension=3,
        source_normalization=source_normalization,
        positive_floor=1.0e-5,
        shared_leaf=True,
    )
    values, derivatives = section_jets(
        batch=11,
        sections=4,
        coordinates=3,
        seed=202607285,
    )
    expected_potential, expected_metric = dense_h_potential_and_metric(
        reference_h,
        values,
        derivatives,
        normalization=source_normalization,
    )
    observed_potential, observed_metric = model.potential_and_metric(
        values,
        derivatives,
    )
    torch.testing.assert_close(
        observed_potential,
        expected_potential,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        observed_metric,
        expected_metric,
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_dormant_channel_activation_preserves_metric_and_exposes_root_gradient() -> None:
    reference_h = positive_h(4, 202607286)
    model = PositiveMultiplicationTreeMetric(
        reference_h,
        leaf_count=5,
        bond_dimension=3,
        source_normalization=0.043,
        positive_floor=1.0e-8,
        shared_leaf=True,
    )
    values, derivatives = section_jets(
        batch=13,
        sections=4,
        coordinates=2,
        seed=202607287,
    )
    before_potential, before_metric = model.potential_and_metric(values, derivatives)
    model.activate_dormant_output_channels_(
        relative_scale=2.0e-2,
        seed=202607288,
    )
    after_potential, after_metric = model.potential_and_metric(values, derivatives)
    torch.testing.assert_close(
        after_potential,
        before_potential,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        after_metric,
        before_metric,
        rtol=3.0e-12,
        atol=3.0e-12,
    )

    model.set_trainable_stage_("root")
    model.zero_grad(set_to_none=True)
    moments = model.moments(values, derivatives)
    torch.mean(torch.log(moments.norm)).backward()
    root_gradient = model.internal_tensors[-1].grad
    assert root_gradient is not None
    assert float(torch.linalg.vector_norm(root_gradient[:, 1:, :])) > 1.0e-10
    assert float(torch.linalg.vector_norm(root_gradient[:, :, 1:])) > 1.0e-10


def test_shared_tree_parameter_count_is_independent_of_leaf_count_at_bottom() -> None:
    reference_h = positive_h(4, 202607289)
    model = PositiveMultiplicationTreeMetric(
        reference_h,
        leaf_count=5,
        bond_dimension=3,
        source_normalization=0.05,
        positive_floor=0.0,
        shared_leaf=True,
    )
    expected_complex = 3 * 4 * 4 + 3 * (3**3) + 3**2
    assert model.trainable_real_parameter_count == 2 * expected_complex


def test_nested_rank_expansion_preserves_metric_and_exposes_new_edge_gradients() -> None:
    reference_h = positive_h(3, 202607290)
    model = PositiveMultiplicationTreeMetric(
        reference_h,
        leaf_count=5,
        bond_dimension=2,
        source_normalization=0.061,
        positive_floor=1.0e-7,
        shared_leaf=True,
    )
    model.activate_dormant_output_channels_(
        relative_scale=1.0e-2,
        seed=202607291,
    )
    values, derivatives = section_jets(
        batch=17,
        sections=3,
        coordinates=2,
        seed=202607292,
    )
    before_potential, before_metric = model.potential_and_metric(values, derivatives)
    expanded = expand_multiplication_tree_bonds(
        model,
        4,
        relative_activation_scale=2.0e-2,
        seed=202607293,
    )
    after_potential, after_metric = expanded.potential_and_metric(
        values,
        derivatives,
    )
    torch.testing.assert_close(
        after_potential,
        before_potential,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    torch.testing.assert_close(
        after_metric,
        before_metric,
        rtol=4.0e-12,
        atol=4.0e-12,
    )
    assert (
        expanded.trainable_real_parameter_count
        > model.trainable_real_parameter_count
    )

    expanded.set_trainable_stage_("internal")
    expanded.zero_grad(set_to_none=True)
    torch.mean(torch.log(expanded.moments(values, derivatives).norm)).backward()
    scores = edge_parent_gradient_scores(
        expanded,
        channel_starts=model.edge_dimensions,
    )
    assert set(scores) == set(range(expanded.topology.root))
    assert max(scores.values()) > 1.0e-10


def test_nested_rank_expansion_can_seed_orthogonal_output_channels() -> None:
    model = PositiveMultiplicationTreeMetric(
        positive_h(3, 202607294),
        leaf_count=5,
        bond_dimension=(2, 2, 2, 2, 2, 1, 1, 1),
        source_normalization=0.061,
        positive_floor=0.0,
        shared_leaf=True,
    )
    expanded = expand_multiplication_tree_bonds(
        model,
        (2, 2, 2, 2, 2, 2, 2, 1),
        relative_activation_scale=1.0,
        seed=202607295,
        orthogonalize_new_outputs=True,
    )
    for internal_index in (0, 1):
        rows = (
            expanded.internal_tensors[internal_index]
            .detach()
            .cpu()
            .numpy()
            .reshape(2, -1)
        )
        overlap = np.vdot(rows[0], rows[1])
        assert abs(overlap) < 2.0e-12 * np.linalg.norm(rows[0]) * np.linalg.norm(
            rows[1]
        )
        assert np.isclose(
            np.linalg.norm(rows[0]),
            np.linalg.norm(rows[1]),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
