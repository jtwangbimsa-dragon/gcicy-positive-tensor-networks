from __future__ import annotations

import numpy as np

from scripts.audit_positive_tensor_network_blocking import (
    block_core_pair,
    normalized_symmetric_square_embedding,
    project_block_output_to_symmetric,
)


def test_symmetric_square_embedding_is_isometric() -> None:
    embedding = normalized_symmetric_square_embedding(5).reshape(25, 15)
    np.testing.assert_allclose(
        embedding.conj().T @ embedding,
        np.eye(15),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_rectangular_block_exactly_reproduces_two_site_output() -> None:
    generator = np.random.default_rng(20260719)
    first = generator.normal(size=(1, 3, 5, 5))
    first = first + 1j * generator.normal(size=first.shape)
    second = generator.normal(size=(3, 1, 5, 5))
    second = second + 1j * generator.normal(size=second.shape)
    point = generator.normal(size=5) + 1j * generator.normal(size=5)
    embedding = normalized_symmetric_square_embedding(5)
    blocked = block_core_pair(first, second, embedding)

    first_output = np.einsum("lbpi,i->lbp", first, point)
    second_output = np.einsum("brqj,j->brq", second, point)
    expected = np.einsum("lbp,brq->lrpq", first_output, second_output)
    degree_two = np.einsum("ija,i,j->a", np.conj(embedding), point, point)
    observed = np.einsum("lrpqa,a->lrpq", blocked, degree_two)

    np.testing.assert_allclose(observed, expected, rtol=2.0e-13, atol=2.0e-13)


def test_symmetric_output_projection_removes_only_antisymmetric_component() -> None:
    generator = np.random.default_rng(20260720)
    block = generator.normal(size=(2, 3, 5, 5, 15))
    block = block + 1j * generator.normal(size=block.shape)
    embedding = normalized_symmetric_square_embedding(5)
    projected = project_block_output_to_symmetric(block, embedding)
    reconstructed = np.einsum("pqa,lrab->lrpqb", embedding, projected)
    residual = block - reconstructed

    residual_symmetric_coordinates = np.einsum(
        "pqa,lrpqb->lrab",
        np.conj(embedding),
        residual,
    )
    np.testing.assert_allclose(
        residual_symmetric_coordinates,
        np.zeros_like(residual_symmetric_coordinates),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert np.linalg.norm(projected) <= np.linalg.norm(block) * (1.0 + 1.0e-14)
