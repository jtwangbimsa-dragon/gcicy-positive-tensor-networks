"""Linear root-supercore residuals for positive multiplication trees."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .positive_multiplication_tree import (
    PositiveMultiplicationTreeMetric,
    TreeMoments,
    _TreeMessage,
    five_leaf_two_three_topology,
)


@dataclass(frozen=True)
class RootResidualSVD:
    left: torch.Tensor
    singular_values: torch.Tensor
    right_h: torch.Tensor

    @property
    def rank(self) -> int:
        return int(len(self.singular_values))

    def retained_fraction(self, rank: int) -> float:
        if not 0 <= rank <= self.rank:
            raise ValueError("retained root-residual rank is outside the SVD")
        squared = torch.square(self.singular_values)
        denominator = float(torch.sum(squared))
        if denominator == 0:
            return 1.0
        return float(torch.sum(squared[:rank])) / denominator


def root_residual_svd(matrix: torch.Tensor) -> RootResidualSVD:
    if matrix.ndim != 2:
        raise ValueError("root residual must be a matrix")
    left, singular_values, right_h = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    return RootResidualSVD(left, singular_values, right_h)


def root_residual_matrix(
    left_basis: torch.Tensor,
    core: torch.Tensor,
    right_basis: torch.Tensor,
) -> torch.Tensor:
    """Lift a reduced active-subspace core into the complete root matrix."""

    if left_basis.ndim != 3 or right_basis.ndim != 3 or core.ndim != 2:
        raise ValueError("root residual factors have incompatible dimensions")
    if core.shape != (len(left_basis), len(right_basis)):
        raise ValueError("root residual core does not match its bases")
    left = left_basis.reshape(len(left_basis), -1)
    right = right_basis.reshape(len(right_basis), -1)
    return torch.transpose(left, 0, 1) @ core @ right


def root_supercore_matrix(
    tree: PositiveMultiplicationTreeMetric,
) -> torch.Tensor:
    """Return the implicit complete-basis coefficient matrix at the root."""

    if tree.topology != five_leaf_two_three_topology():
        raise ValueError("root supercore matrix requires the 2+3 topology")
    root_left, root_right = tree.topology.children[-1]
    left = tree.internal_tensors[root_left - tree.leaf_count].reshape(
        tree.edge_dimensions[root_left],
        -1,
    )
    right = tree.internal_tensors[root_right - tree.leaf_count].reshape(
        tree.edge_dimensions[root_right],
        -1,
    )
    root = tree.internal_tensors[-1][0]
    return torch.transpose(left, 0, 1) @ root @ right


class RootResidualSupercoreMetric(torch.nn.Module):
    """Add a linearly active residual at a five-leaf ``2+3`` root.

    The frozen base tree has root children ``node 5`` and ``node 7``.  New
    output rows are appended to those two child tensors, and ``residual_core``
    couples only the new rows.  Consequently ``residual_core == 0`` exactly
    recovers the base tree, while the derivative with respect to the core is
    generally nonzero.
    """

    def __init__(
        self,
        base_tree: PositiveMultiplicationTreeMetric,
        *,
        left_basis: torch.Tensor,
        right_basis: torch.Tensor,
        initial_core: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if base_tree.topology != five_leaf_two_three_topology():
            raise ValueError("root residual requires the five-leaf 2+3 topology")
        base_tree.freeze_all_()
        self.base_tree = base_tree
        root_left, root_right = base_tree.topology.children[-1]
        if root_left < base_tree.leaf_count or root_right < base_tree.leaf_count:
            raise ValueError("both root children must be internal tree nodes")
        left_tensor = base_tree.internal_tensors[root_left - base_tree.leaf_count]
        right_tensor = base_tree.internal_tensors[
            root_right - base_tree.leaf_count
        ]
        expected_left = tuple(left_tensor.shape[1:])
        expected_right = tuple(right_tensor.shape[1:])
        if (
            left_basis.ndim != 3
            or tuple(left_basis.shape[1:]) != expected_left
            or right_basis.ndim != 3
            or tuple(right_basis.shape[1:]) != expected_right
        ):
            raise ValueError("root-residual bases do not match child input spaces")
        if (
            left_basis.dtype != left_tensor.dtype
            or right_basis.dtype != right_tensor.dtype
            or left_basis.device != left_tensor.device
            or right_basis.device != right_tensor.device
        ):
            raise ValueError("root-residual bases must match the base tree")
        self.register_buffer("left_basis", left_basis.detach().clone())
        self.register_buffer("right_basis", right_basis.detach().clone())
        core_shape = (len(left_basis), len(right_basis))
        if initial_core is None:
            core = torch.zeros(
                core_shape,
                dtype=left_tensor.dtype,
                device=left_tensor.device,
            )
        else:
            if (
                initial_core.shape != core_shape
                or initial_core.dtype != left_tensor.dtype
                or initial_core.device != left_tensor.device
            ):
                raise ValueError("initial root-residual core has the wrong contract")
            core = initial_core.detach().clone()
        self.residual_core = torch.nn.Parameter(core)

    @classmethod
    def complete_basis(
        cls,
        base_tree: PositiveMultiplicationTreeMetric,
    ) -> "RootResidualSupercoreMetric":
        """Open the complete child-product basis at the selected root."""

        if base_tree.topology != five_leaf_two_three_topology():
            raise ValueError("complete root basis requires the 2+3 topology")
        root_left, root_right = base_tree.topology.children[-1]
        left_tensor = base_tree.internal_tensors[root_left - base_tree.leaf_count]
        right_tensor = base_tree.internal_tensors[
            root_right - base_tree.leaf_count
        ]

        def identity_basis(tensor: torch.Tensor) -> torch.Tensor:
            rows = int(tensor.shape[1] * tensor.shape[2])
            return torch.eye(
                rows,
                dtype=tensor.dtype,
                device=tensor.device,
            ).reshape(rows, tensor.shape[1], tensor.shape[2])

        return cls(
            base_tree,
            left_basis=identity_basis(left_tensor),
            right_basis=identity_basis(right_tensor),
        )

    @property
    def trainable_real_parameter_count(self) -> int:
        return int(2 * self.residual_core.numel())

    def complete_residual_matrix(self) -> torch.Tensor:
        return root_residual_matrix(
            self.left_basis,
            self.residual_core,
            self.right_basis,
        )

    @staticmethod
    def _expanded_root_tensor(
        root: torch.Tensor,
        residual_core: torch.Tensor,
    ) -> torch.Tensor:
        left_old = root.shape[1]
        right_old = root.shape[2]
        left_new, right_new = residual_core.shape
        top = torch.cat(
            (
                root,
                root.new_zeros((1, left_old, right_new)),
            ),
            dim=2,
        )
        bottom = torch.cat(
            (
                root.new_zeros((1, left_new, right_old)),
                residual_core[None],
            ),
            dim=2,
        )
        return torch.cat((top, bottom), dim=1)

    def learned_moments(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> TreeMoments:
        base = self.base_tree
        if base.shared_leaf:
            shared = base._leaf_message(
                base.shared_leaf_tensor,
                section_values,
                section_derivatives,
            )
            messages: dict[int, _TreeMessage] = {
                leaf: shared for leaf in range(base.leaf_count)
            }
        else:
            messages = {
                leaf: base._leaf_message(
                    base.leaf_tensor(leaf),
                    section_values,
                    section_derivatives,
                )
                for leaf in range(base.leaf_count)
            }

        root_left, root_right = base.topology.children[-1]
        for internal_index, (left, right) in enumerate(base.topology.children):
            node = base.leaf_count + internal_index
            tensor = base.internal_tensors[internal_index]
            if node == root_left:
                tensor = torch.cat((tensor, self.left_basis), dim=0)
            elif node == root_right:
                tensor = torch.cat((tensor, self.right_basis), dim=0)
            elif node == base.topology.root:
                tensor = self._expanded_root_tensor(
                    tensor,
                    self.residual_core,
                )
            messages[node] = base._internal_message(
                tensor,
                messages[left],
                messages[right],
            )
        root = messages[base.topology.root]
        return TreeMoments(
            norm=torch.real(root.base[:, 0, 0]),
            holomorphic_gradient=root.holomorphic[:, :, 0, 0],
            mixed_hessian=root.mixed[:, :, :, 0, 0],
        )

    def moments(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> TreeMoments:
        base = self.base_tree
        learned = self.learned_moments(section_values, section_derivatives)
        if base.positive_floor == 0:
            return learned
        reference = base.reference_moments(section_values, section_derivatives)
        learned_weight = 1.0 - base.positive_floor
        return TreeMoments(
            norm=(
                learned_weight * learned.norm
                + base.positive_floor * reference.norm
            ),
            holomorphic_gradient=(
                learned_weight * learned.holomorphic_gradient
                + base.positive_floor * reference.holomorphic_gradient
            ),
            mixed_hessian=(
                learned_weight * learned.mixed_hessian
                + base.positive_floor * reference.mixed_hessian
            ),
        )

    def potential_and_metric(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moments = self.moments(section_values, section_derivatives)
        if not bool(torch.all(torch.isfinite(moments.norm) & (moments.norm > 0))):
            raise FloatingPointError("root-residual tree norm is not positive")
        denominator = moments.norm
        metric = moments.mixed_hessian / denominator[:, None, None]
        metric = metric - (
            torch.conj(moments.holomorphic_gradient)[:, :, None]
            * moments.holomorphic_gradient[:, None, :]
            / torch.square(denominator[:, None, None])
        )
        metric = self.base_tree.target_normalization * metric
        metric = 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        potential = self.base_tree.target_normalization * torch.log(denominator)
        return potential, metric

    def forward(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> torch.Tensor:
        return self.potential_and_metric(section_values, section_derivatives)[1]


def commit_root_residual_svd(
    base_tree: PositiveMultiplicationTreeMetric,
    residual_matrix: torch.Tensor,
    *,
    rank: int,
) -> PositiveMultiplicationTreeMetric:
    """Absorb a truncated root residual into ordinary tree bond channels."""

    if base_tree.topology != five_leaf_two_three_topology():
        raise ValueError("root residual commit requires the 2+3 topology")
    root_left, root_right = base_tree.topology.children[-1]
    left_tensor = base_tree.internal_tensors[root_left - base_tree.leaf_count]
    right_tensor = base_tree.internal_tensors[root_right - base_tree.leaf_count]
    left_count = int(left_tensor.shape[1] * left_tensor.shape[2])
    right_count = int(right_tensor.shape[1] * right_tensor.shape[2])
    if residual_matrix.shape != (left_count, right_count):
        raise ValueError("root residual matrix has the wrong complete-basis shape")
    decomposition = root_residual_svd(residual_matrix)
    if not 0 <= rank <= decomposition.rank:
        raise ValueError("requested root residual rank is outside the SVD")

    source = tuple(base_tree.edge_dimensions)
    target = list(source)
    target[root_left] += rank
    target[root_right] += rank
    committed = PositiveMultiplicationTreeMetric(
        base_tree.reference_h.detach().cpu().numpy(),
        leaf_count=base_tree.leaf_count,
        bond_dimension=tuple(target),
        source_normalization=base_tree.source_normalization,
        output_dimension=base_tree.output_dimension,
        positive_floor=base_tree.positive_floor,
        shared_leaf=base_tree.shared_leaf,
        topology=base_tree.topology,
        dtype=base_tree.reference_h.dtype,
        device=base_tree.reference_h.device,
    )

    with torch.no_grad():
        if base_tree.shared_leaf:
            committed.shared_leaf_tensor.copy_(base_tree.shared_leaf_tensor)
        else:
            for old_leaf, new_leaf in zip(
                base_tree.leaf_tensors,
                committed.leaf_tensors,
                strict=True,
            ):
                new_leaf.copy_(old_leaf)
        for internal_index, (old_tensor, new_tensor) in enumerate(
            zip(
                base_tree.internal_tensors,
                committed.internal_tensors,
                strict=True,
            )
        ):
            new_tensor.zero_()
            old_parent, old_left, old_right = old_tensor.shape
            new_tensor[:old_parent, :old_left, :old_right].copy_(old_tensor)

        if rank:
            left_rows = decomposition.left[:, :rank].T.reshape(
                rank,
                left_tensor.shape[1],
                left_tensor.shape[2],
            )
            right_rows = decomposition.right_h[:rank].reshape(
                rank,
                right_tensor.shape[1],
                right_tensor.shape[2],
            )
            committed.internal_tensors[
                root_left - base_tree.leaf_count
            ][left_tensor.shape[0] :].copy_(left_rows)
            committed.internal_tensors[
                root_right - base_tree.leaf_count
            ][right_tensor.shape[0] :].copy_(right_rows)
            root = committed.internal_tensors[-1]
            for channel in range(rank):
                root[
                    0,
                    left_tensor.shape[0] + channel,
                    right_tensor.shape[0] + channel,
                ] = decomposition.singular_values[channel]
    committed.train(base_tree.training)
    return committed


def complete_root_residual_shape(
    base_tree: PositiveMultiplicationTreeMetric,
) -> tuple[int, int]:
    if base_tree.topology != five_leaf_two_three_topology():
        raise ValueError("complete root shape requires the 2+3 topology")
    root_left, root_right = base_tree.topology.children[-1]
    left = base_tree.internal_tensors[root_left - base_tree.leaf_count]
    right = base_tree.internal_tensors[root_right - base_tree.leaf_count]
    return (
        int(left.shape[1] * left.shape[2]),
        int(right.shape[1] * right.shape[2]),
    )
