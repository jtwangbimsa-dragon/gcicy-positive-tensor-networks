"""Positive balanced-tree metrics initialized by an exact algebraic power lift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .positive_tensor_network import rectangular_reference_factor


@dataclass(frozen=True)
class BinaryTreeTopology:
    """Postordered full binary tree with leaves numbered first."""

    leaf_count: int
    children: tuple[tuple[int, int], ...]

    @property
    def node_count(self) -> int:
        return self.leaf_count + len(self.children)

    @property
    def root(self) -> int:
        return self.node_count - 1

    def parent_map(self) -> dict[int, tuple[int, int]]:
        result: dict[int, tuple[int, int]] = {}
        for internal_index, (left, right) in enumerate(self.children):
            parent = self.leaf_count + internal_index
            if left in result or right in result:
                raise ValueError("tree node has more than one parent")
            result[left] = (parent, 0)
            result[right] = (parent, 1)
        if set(result) != set(range(self.root)):
            raise ValueError("tree is not connected")
        return result


def balanced_binary_topology(leaf_count: int) -> BinaryTreeTopology:
    """Build a deterministic approximately balanced postordered tree."""

    if leaf_count < 2:
        raise ValueError("a multiplication tree needs at least two leaves")
    active = list(range(leaf_count))
    children: list[tuple[int, int]] = []
    next_node = leaf_count
    while len(active) > 1:
        following: list[int] = []
        for start in range(0, len(active) - 1, 2):
            children.append((active[start], active[start + 1]))
            following.append(next_node)
            next_node += 1
        if len(active) % 2:
            following.append(active[-1])
        active = following
    topology = BinaryTreeTopology(
        leaf_count=leaf_count,
        children=tuple(children),
    )
    topology.parent_map()
    return topology


def five_leaf_two_three_topology() -> BinaryTreeTopology:
    """Return the five-leaf topology with a two-versus-three root split.

    For equal-degree leaves this changes the default ``4+1`` leaf split into
    the more balanced ``2+3`` split:

    ``(0, 1) | ((2, 3), 4)``.
    """

    topology = BinaryTreeTopology(
        leaf_count=5,
        children=((0, 1), (2, 3), (6, 4), (5, 7)),
    )
    topology.parent_map()
    return topology


@dataclass(frozen=True)
class TreeMoments:
    norm: torch.Tensor
    holomorphic_gradient: torch.Tensor
    mixed_hessian: torch.Tensor


@dataclass
class _TreeMessage:
    base: torch.Tensor
    holomorphic: torch.Tensor
    antiholomorphic: torch.Tensor
    mixed: torch.Tensor


def _positive_factor(matrix: np.ndarray, output_dimension: int) -> np.ndarray:
    return rectangular_reference_factor(matrix, output_dimension)


class PositiveMultiplicationTreeMetric(torch.nn.Module):
    """Purified tree-state metric with an exact low-degree power-lift origin.

    Every physical leaf receives the same degree-``b`` section vector.  The
    initial shared leaf map is a purification factor of ``H_b`` and every
    virtual edge starts in channel zero.  Therefore the initial tree norm is
    exactly ``(s^dagger H_b s)**leaf_count``.
    """

    def __init__(
        self,
        reference_h: np.ndarray,
        *,
        leaf_count: int,
        bond_dimension: int | Sequence[int],
        source_normalization: float,
        output_dimension: int | None = None,
        positive_floor: float = 1.0e-8,
        shared_leaf: bool = True,
        topology: BinaryTreeTopology | None = None,
        dtype: torch.dtype = torch.complex128,
        device: Any = None,
    ) -> None:
        super().__init__()
        if dtype not in {torch.complex64, torch.complex128}:
            raise ValueError("tree dtype must be complex64 or complex128")
        reference = np.asarray(reference_h, dtype=np.complex128)
        reference = 0.5 * (reference + reference.conj().T)
        eigenvalues = np.linalg.eigvalsh(reference)
        if (
            reference.ndim != 2
            or reference.shape[0] != reference.shape[1]
            or not np.all(np.isfinite(eigenvalues))
            or eigenvalues[0] <= 0
        ):
            raise ValueError("reference H must be finite and positive definite")
        if source_normalization <= 0 or not np.isfinite(source_normalization):
            raise ValueError("source normalization must be finite and positive")
        if not 0 <= positive_floor < 1:
            raise ValueError("positive floor must lie in [0, 1)")
        self.topology = topology or balanced_binary_topology(leaf_count)
        if self.topology.leaf_count != leaf_count:
            raise ValueError("tree topology and requested leaf count disagree")
        self.topology.parent_map()
        self.leaf_count = int(leaf_count)
        self.section_count = int(len(reference))
        self.output_dimension = int(output_dimension or self.section_count)
        if self.output_dimension < self.section_count:
            raise ValueError("output dimension must cover the reference rank")
        self.source_normalization = float(source_normalization)
        self.target_normalization = float(source_normalization / leaf_count)
        self.positive_floor = float(positive_floor)
        self.shared_leaf = bool(shared_leaf)

        nonroot_count = self.topology.root
        if isinstance(bond_dimension, (int, np.integer)):
            if int(bond_dimension) <= 0:
                raise ValueError("bond dimension must be positive")
            edge_dimensions = (int(bond_dimension),) * nonroot_count
        else:
            edge_dimensions = tuple(int(value) for value in bond_dimension)
            if len(edge_dimensions) != nonroot_count or any(
                value <= 0 for value in edge_dimensions
            ):
                raise ValueError("one positive dimension is required per nonroot edge")
        self.edge_dimensions = edge_dimensions

        factor = _positive_factor(reference, self.output_dimension)
        self.register_buffer(
            "reference_h",
            torch.tensor(reference, dtype=dtype, device=device),
        )
        self.register_buffer(
            "reference_factor",
            torch.tensor(factor, dtype=dtype, device=device),
        )
        if self.shared_leaf:
            initial = np.zeros(
                (
                    edge_dimensions[0],
                    self.output_dimension,
                    self.section_count,
                ),
                dtype=np.complex128,
            )
            initial[0] = factor
            self.shared_leaf_tensor = torch.nn.Parameter(
                torch.tensor(initial, dtype=dtype, device=device)
            )
            self.leaf_tensors = torch.nn.ParameterList()
        else:
            self.register_parameter("shared_leaf_tensor", None)
            self.leaf_tensors = torch.nn.ParameterList()
            for leaf in range(self.leaf_count):
                initial = np.zeros(
                    (
                        edge_dimensions[leaf],
                        self.output_dimension,
                        self.section_count,
                    ),
                    dtype=np.complex128,
                )
                initial[0] = factor
                self.leaf_tensors.append(
                    torch.nn.Parameter(
                        torch.tensor(initial, dtype=dtype, device=device)
                    )
                )

        self.internal_tensors = torch.nn.ParameterList()
        for internal_index, (left, right) in enumerate(self.topology.children):
            node = self.leaf_count + internal_index
            parent_dimension = 1 if node == self.topology.root else edge_dimensions[node]
            initial = np.zeros(
                (
                    parent_dimension,
                    edge_dimensions[left],
                    edge_dimensions[right],
                ),
                dtype=np.complex128,
            )
            initial[0, 0, 0] = 1.0
            self.internal_tensors.append(
                torch.nn.Parameter(
                    torch.tensor(initial, dtype=dtype, device=device)
                )
            )

    @property
    def trainable_real_parameter_count(self) -> int:
        return int(2 * sum(parameter.numel() for parameter in self.parameters()))

    def leaf_tensor(self, leaf: int) -> torch.Tensor:
        if not 0 <= leaf < self.leaf_count:
            raise ValueError("leaf index is outside the tree")
        if self.shared_leaf:
            expected = self.edge_dimensions[leaf]
            if self.shared_leaf_tensor.shape[0] != expected:
                raise ValueError(
                    "shared leaves require equal dimensions on all leaf edges"
                )
            return self.shared_leaf_tensor
        return self.leaf_tensors[leaf]

    @staticmethod
    def _leaf_message(
        tensor: torch.Tensor,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> _TreeMessage:
        state = torch.einsum("api,ni->nap", tensor, section_values)
        derivative = torch.einsum(
            "api,nij->najp",
            tensor,
            section_derivatives,
        ).permute(0, 2, 1, 3)
        base = torch.einsum("nap,nbp->nab", torch.conj(state), state)
        holomorphic = torch.einsum(
            "nap,njbp->njab",
            torch.conj(state),
            derivative,
        )
        antiholomorphic = torch.einsum(
            "niap,nbp->niab",
            torch.conj(derivative),
            state,
        )
        mixed = torch.einsum(
            "niap,njbp->nijab",
            torch.conj(derivative),
            derivative,
        )
        return _TreeMessage(base, holomorphic, antiholomorphic, mixed)

    @staticmethod
    def _contract_pair(
        tensor: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "ABC,nBb,nCc,abc->nAa",
            torch.conj(tensor),
            left,
            right,
            tensor,
        )

    @classmethod
    def _internal_message(
        cls,
        tensor: torch.Tensor,
        left: _TreeMessage,
        right: _TreeMessage,
    ) -> _TreeMessage:
        conjugate = torch.conj(tensor)
        base = torch.einsum(
            "ABC,nBb,nCc,abc->nAa",
            conjugate,
            left.base,
            right.base,
            tensor,
        )
        holomorphic = torch.einsum(
            "ABC,njBb,nCc,abc->njAa",
            conjugate,
            left.holomorphic,
            right.base,
            tensor,
        ) + torch.einsum(
            "ABC,nBb,njCc,abc->njAa",
            conjugate,
            left.base,
            right.holomorphic,
            tensor,
        )
        antiholomorphic = torch.einsum(
            "ABC,niBb,nCc,abc->niAa",
            conjugate,
            left.antiholomorphic,
            right.base,
            tensor,
        ) + torch.einsum(
            "ABC,nBb,niCc,abc->niAa",
            conjugate,
            left.base,
            right.antiholomorphic,
            tensor,
        )
        mixed = (
            torch.einsum(
                "ABC,nijBb,nCc,abc->nijAa",
                conjugate,
                left.mixed,
                right.base,
                tensor,
            )
            + torch.einsum(
                "ABC,niBb,njCc,abc->nijAa",
                conjugate,
                left.antiholomorphic,
                right.holomorphic,
                tensor,
            )
            + torch.einsum(
                "ABC,njBb,niCc,abc->nijAa",
                conjugate,
                left.holomorphic,
                right.antiholomorphic,
                tensor,
            )
            + torch.einsum(
                "ABC,nBb,nijCc,abc->nijAa",
                conjugate,
                left.base,
                right.mixed,
                tensor,
            )
        )
        return _TreeMessage(base, holomorphic, antiholomorphic, mixed)

    def learned_moments(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> TreeMoments:
        if (
            section_values.ndim != 2
            or section_values.shape[1] != self.section_count
            or section_derivatives.ndim != 3
            or section_derivatives.shape[:2] != section_values.shape
        ):
            raise ValueError("tree section values and derivatives are misaligned")
        if self.shared_leaf:
            shared_message = self._leaf_message(
                self.shared_leaf_tensor,
                section_values,
                section_derivatives,
            )
            messages: dict[int, _TreeMessage] = {
                leaf: shared_message for leaf in range(self.leaf_count)
            }
        else:
            messages = {
                leaf: self._leaf_message(
                    self.leaf_tensor(leaf),
                    section_values,
                    section_derivatives,
                )
                for leaf in range(self.leaf_count)
            }
        for internal_index, (left, right) in enumerate(self.topology.children):
            node = self.leaf_count + internal_index
            messages[node] = self._internal_message(
                self.internal_tensors[internal_index],
                messages[left],
                messages[right],
            )
        root = messages[self.topology.root]
        return TreeMoments(
            norm=torch.real(root.base[:, 0, 0]),
            holomorphic_gradient=root.holomorphic[:, :, 0, 0],
            mixed_hessian=root.mixed[:, :, :, 0, 0],
        )

    def reference_moments(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> TreeMoments:
        state = torch.einsum(
            "pi,ni->np",
            self.reference_factor,
            section_values,
        )
        derivative = torch.einsum(
            "pi,nij->npj",
            self.reference_factor,
            section_derivatives,
        )
        local_norm = torch.real(
            torch.einsum("np,np->n", torch.conj(state), state)
        )
        local_gradient = torch.einsum(
            "np,npj->nj",
            torch.conj(state),
            derivative,
        )
        local_mixed = torch.einsum(
            "npi,npj->nij",
            torch.conj(derivative),
            derivative,
        )
        power = self.leaf_count
        norm = torch.pow(local_norm, power)
        gradient = (
            power
            * torch.pow(local_norm, power - 1)[:, None]
            * local_gradient
        )
        mixed = (
            power
            * torch.pow(local_norm, power - 1)[:, None, None]
            * local_mixed
        )
        if power > 1:
            mixed = mixed + (
                power
                * (power - 1)
                * torch.pow(local_norm, power - 2)[:, None, None]
                * torch.conj(local_gradient)[:, :, None]
                * local_gradient[:, None, :]
            )
        return TreeMoments(norm, gradient, mixed)

    def moments(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> TreeMoments:
        learned = self.learned_moments(section_values, section_derivatives)
        if self.positive_floor == 0:
            return learned
        reference = self.reference_moments(
            section_values,
            section_derivatives,
        )
        learned_weight = 1.0 - self.positive_floor
        return TreeMoments(
            norm=(
                learned_weight * learned.norm
                + self.positive_floor * reference.norm
            ),
            holomorphic_gradient=(
                learned_weight * learned.holomorphic_gradient
                + self.positive_floor * reference.holomorphic_gradient
            ),
            mixed_hessian=(
                learned_weight * learned.mixed_hessian
                + self.positive_floor * reference.mixed_hessian
            ),
        )

    def potential_and_metric(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moments = self.moments(section_values, section_derivatives)
        if not bool(torch.all(torch.isfinite(moments.norm) & (moments.norm > 0))):
            raise FloatingPointError("multiplication-tree norm is not positive")
        denominator = moments.norm
        metric = moments.mixed_hessian / denominator[:, None, None]
        metric = metric - (
            torch.conj(moments.holomorphic_gradient)[:, :, None]
            * moments.holomorphic_gradient[:, None, :]
            / torch.square(denominator[:, None, None])
        )
        metric = self.target_normalization * metric
        metric = 0.5 * (
            metric + torch.conj(torch.transpose(metric, 1, 2))
        )
        potential = self.target_normalization * torch.log(denominator)
        return potential, metric

    def forward(
        self,
        section_values: torch.Tensor,
        section_derivatives: torch.Tensor,
    ) -> torch.Tensor:
        """Return the metric for functional JVP/VJP optimization."""

        return self.potential_and_metric(
            section_values,
            section_derivatives,
        )[1]

    def freeze_all_(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def set_trainable_stage_(self, stage: str) -> None:
        """Select root, all internal nodes, leaves, or the complete tree."""

        if stage not in {"root", "internal", "leaves", "all"}:
            raise ValueError("unknown multiplication-tree training stage")
        self.freeze_all_()
        if stage in {"root", "internal", "all"}:
            selected = (
                self.internal_tensors[-1:]
                if stage == "root"
                else self.internal_tensors
            )
            for parameter in selected:
                parameter.requires_grad_(True)
        if stage in {"leaves", "all"}:
            if self.shared_leaf:
                self.shared_leaf_tensor.requires_grad_(True)
            else:
                for parameter in self.leaf_tensors:
                    parameter.requires_grad_(True)

    def activate_dormant_output_channels_(
        self,
        *,
        relative_scale: float,
        seed: int,
    ) -> None:
        """Seed child-side dormant channels while preserving the exact root state."""

        if relative_scale <= 0 or not np.isfinite(relative_scale):
            raise ValueError("dormant-channel scale must be finite and positive")
        rng = np.random.default_rng(seed)

        def fill_dormant(parameter: torch.Tensor) -> None:
            if parameter.shape[0] <= 1:
                return
            values = parameter.detach().cpu().numpy()
            if np.any(values[1:] != 0):
                raise ValueError("dormant output channels are already active")
            active_norm = float(np.linalg.norm(values[0]))
            entry_scale = (
                relative_scale
                * active_norm
                / np.sqrt(max(values[1:].size, 1))
            )
            noise = rng.normal(size=values[1:].shape) + 1j * rng.normal(
                size=values[1:].shape
            )
            values[1:] = entry_scale * noise / np.sqrt(2.0)
            parameter.copy_(
                torch.tensor(
                    values,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )

        with torch.no_grad():
            if self.shared_leaf:
                fill_dormant(self.shared_leaf_tensor)
            else:
                for tensor in self.leaf_tensors:
                    fill_dormant(tensor)
            for internal_index, tensor in enumerate(self.internal_tensors[:-1]):
                fill_dormant(tensor)


def build_teacher_compiled_multiplication_tree(
    reference_h: np.ndarray,
    *,
    leaf_tensor: np.ndarray,
    leaf_combiner: np.ndarray,
    leaf_count: int,
    source_normalization: float,
    positive_floor: float = 1.0e-8,
    topology: BinaryTreeTopology | None = None,
    dtype: torch.dtype = torch.complex128,
    device: Any = None,
) -> PositiveMultiplicationTreeMetric:
    """Build a power tree whose active channels come from a teacher SVD.

    ``leaf_combiner @ leaf_tensor`` must be a purification factor of
    ``reference_h`` (exactly or to the compiler's registered tolerance).
    Each physical leaf receives the same compiled channel dictionary.  The
    deterministic internal tensors contract those channels back to the local
    factor, so the round-zero tree represents its exact power without random
    dormant-channel activation.
    """

    reference = np.asarray(reference_h, dtype=np.complex128)
    channels = np.asarray(leaf_tensor, dtype=np.complex128)
    combiner = np.asarray(leaf_combiner, dtype=np.complex128)
    if channels.ndim != 3 or len(channels) == 0:
        raise ValueError("compiled leaf tensor must have shape (rank, output, section)")
    if combiner.shape != (len(channels),):
        raise ValueError("one compiled combiner coefficient is required per channel")
    if channels.shape[2] != len(reference):
        raise ValueError("compiled leaf sections do not match the reference H")
    if leaf_count < 2:
        raise ValueError("a compiled multiplication tree needs at least two leaves")
    resolved_topology = topology or balanced_binary_topology(leaf_count)
    if resolved_topology.leaf_count != leaf_count:
        raise ValueError("compiled topology and leaf count disagree")
    resolved_topology.parent_map()

    edge_dimensions = [len(channels)] * leaf_count
    edge_dimensions.extend(
        1 for _ in range(leaf_count, resolved_topology.root)
    )
    model = PositiveMultiplicationTreeMetric(
        reference,
        leaf_count=leaf_count,
        bond_dimension=tuple(edge_dimensions),
        source_normalization=source_normalization,
        output_dimension=channels.shape[1],
        positive_floor=positive_floor,
        shared_leaf=True,
        topology=resolved_topology,
        dtype=dtype,
        device=device,
    )

    selectors: dict[int, np.ndarray] = {
        leaf: combiner for leaf in range(leaf_count)
    }
    with torch.no_grad():
        model.shared_leaf_tensor.copy_(
            torch.tensor(channels, dtype=dtype, device=device)
        )
        for internal_index, (left, right) in enumerate(
            resolved_topology.children
        ):
            node = leaf_count + internal_index
            tensor = model.internal_tensors[internal_index]
            tensor.zero_()
            product_selector = np.outer(selectors[left], selectors[right])
            if tensor.shape[1:] != product_selector.shape:
                raise ValueError("compiled selectors do not match the tree topology")
            tensor[0].copy_(
                torch.tensor(product_selector, dtype=dtype, device=device)
            )
            if node != resolved_topology.root:
                selector = np.zeros(
                    model.edge_dimensions[node],
                    dtype=np.complex128,
                )
                selector[0] = 1.0
                selectors[node] = selector
    return model


def expand_multiplication_tree_bonds(
    model: PositiveMultiplicationTreeMetric,
    target_edge_dimensions: int | Sequence[int],
    *,
    relative_activation_scale: float,
    seed: int,
    orthogonalize_new_outputs: bool = False,
) -> PositiveMultiplicationTreeMetric:
    """Embed a tree in larger edge spaces without changing its represented metric."""

    if isinstance(target_edge_dimensions, (int, np.integer)):
        target = (int(target_edge_dimensions),) * model.topology.root
    else:
        target = tuple(int(value) for value in target_edge_dimensions)
    source = model.edge_dimensions
    if (
        len(target) != len(source)
        or any(new < old for new, old in zip(target, source, strict=True))
        or not any(new > old for new, old in zip(target, source, strict=True))
    ):
        raise ValueError("target edge spaces must form a strict nested expansion")
    if relative_activation_scale <= 0 or not np.isfinite(
        relative_activation_scale
    ):
        raise ValueError("rank activation scale must be finite and positive")
    if model.shared_leaf and len(set(target[: model.leaf_count])) != 1:
        raise ValueError("shared leaves require equal expanded leaf dimensions")

    dtype = model.reference_h.dtype
    device = model.reference_h.device
    expanded = PositiveMultiplicationTreeMetric(
        model.reference_h.detach().cpu().numpy(),
        leaf_count=model.leaf_count,
        bond_dimension=target,
        source_normalization=model.source_normalization,
        output_dimension=model.output_dimension,
        positive_floor=model.positive_floor,
        shared_leaf=model.shared_leaf,
        topology=model.topology,
        dtype=dtype,
        device=device,
    )
    rng = np.random.default_rng(seed)

    def seed_new_outputs(
        parameter: torch.Tensor,
        source_output: int,
    ) -> None:
        if source_output == parameter.shape[0]:
            return
        values = parameter.detach().cpu().numpy()
        old_values = values[:source_output]
        active_norm = float(np.linalg.norm(old_values))
        new_values = values[source_output:]
        if orthogonalize_new_outputs:
            old_rows = old_values.reshape(source_output, -1)
            _, singular_values, right_vectors_h = np.linalg.svd(
                old_rows,
                full_matrices=False,
            )
            tolerance = (
                max(old_rows.shape)
                * np.finfo(np.float64).eps
                * max(float(singular_values[0]), 1.0)
            )
            rank = int(np.count_nonzero(singular_values > tolerance))
            basis = [
                row.copy()
                for row in right_vectors_h[:rank]
            ]
            target_norm = (
                relative_activation_scale
                * active_norm
                / np.sqrt(max(source_output, 1))
            )
            flat_new = new_values.reshape(len(new_values), -1)
            for row_index in range(len(flat_new)):
                candidate = rng.normal(size=flat_new.shape[1]) + 1j * rng.normal(
                    size=flat_new.shape[1]
                )
                for vector in basis:
                    candidate = candidate - np.vdot(vector, candidate) * vector
                norm = float(np.linalg.norm(candidate))
                if norm <= 100.0 * np.finfo(np.float64).eps:
                    raise ValueError(
                        "expanded output space has no orthogonal activation direction"
                    )
                candidate = candidate / norm
                flat_new[row_index] = target_norm * candidate
                basis.append(candidate)
            parameter.copy_(
                torch.tensor(
                    values,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )
            return
        entry_scale = (
            relative_activation_scale
            * active_norm
            / np.sqrt(max(new_values.size, 1))
        )
        noise = rng.normal(size=new_values.shape) + 1j * rng.normal(
            size=new_values.shape
        )
        values[source_output:] = entry_scale * noise / np.sqrt(2.0)
        parameter.copy_(
            torch.tensor(
                values,
                dtype=parameter.dtype,
                device=parameter.device,
            )
        )

    with torch.no_grad():
        if model.shared_leaf:
            expanded.shared_leaf_tensor.zero_()
            expanded.shared_leaf_tensor[: source[0]].copy_(
                model.shared_leaf_tensor
            )
            seed_new_outputs(expanded.shared_leaf_tensor, source[0])
        else:
            for leaf, (old_tensor, new_tensor) in enumerate(
                zip(model.leaf_tensors, expanded.leaf_tensors, strict=True)
            ):
                new_tensor.zero_()
                new_tensor[: source[leaf]].copy_(old_tensor)
                seed_new_outputs(new_tensor, source[leaf])

        for internal_index, (old_tensor, new_tensor) in enumerate(
            zip(model.internal_tensors, expanded.internal_tensors, strict=True)
        ):
            node = model.leaf_count + internal_index
            left, right = model.topology.children[internal_index]
            old_parent = 1 if node == model.topology.root else source[node]
            new_tensor.zero_()
            new_tensor[
                :old_parent,
                : source[left],
                : source[right],
            ].copy_(old_tensor)
            if node != model.topology.root:
                seed_new_outputs(new_tensor, old_parent)
    return expanded


def reassociate_five_leaf_tree_to_two_three(
    model: PositiveMultiplicationTreeMetric,
) -> PositiveMultiplicationTreeMetric:
    """Exactly re-associate the default five-leaf tree at its upper nodes.

    The source topology is

    ``((0, 1), (2, 3)) | 4``,

    and the returned topology is

    ``(0, 1) | ((2, 3), 4)``.

    The top two source tensors are first contracted into one three-leg
    coefficient tensor and then factorized across the new root cut.  Since
    the left side of that matrix has dimension ``edge_dimensions[5]``, the
    returned edge-7 dimension is sufficient for an exact factorization.
    """

    expected = balanced_binary_topology(5)
    if model.topology != expected:
        raise ValueError(
            "five-leaf re-association requires the default 4+1 topology"
        )
    source = tuple(model.edge_dimensions)
    new_edge_seven = source[5]
    target_dimensions = (*source[:7], new_edge_seven)
    dtype = model.reference_h.dtype
    device = model.reference_h.device
    reassociated = PositiveMultiplicationTreeMetric(
        model.reference_h.detach().cpu().numpy(),
        leaf_count=5,
        bond_dimension=target_dimensions,
        source_normalization=model.source_normalization,
        output_dimension=model.output_dimension,
        positive_floor=model.positive_floor,
        shared_leaf=model.shared_leaf,
        topology=five_leaf_two_three_topology(),
        dtype=dtype,
        device=device,
    )

    with torch.no_grad():
        if model.shared_leaf:
            reassociated.shared_leaf_tensor.copy_(model.shared_leaf_tensor)
        else:
            for source_leaf, target_leaf in zip(
                model.leaf_tensors,
                reassociated.leaf_tensors,
                strict=True,
            ):
                target_leaf.copy_(source_leaf)
        reassociated.internal_tensors[0].copy_(model.internal_tensors[0])
        reassociated.internal_tensors[1].copy_(model.internal_tensors[1])

        upper = torch.einsum(
            "qk,qij->ijk",
            model.internal_tensors[3][0],
            model.internal_tensors[2],
        )
        matrix = upper.reshape(source[5], source[6] * source[4])
        left, singular_values, right_h = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )
        rank = len(singular_values)
        if rank > new_edge_seven:
            raise RuntimeError("new five-leaf root edge cannot preserve the tree")
        square_root = torch.sqrt(singular_values)
        reassociated.internal_tensors[2].zero_()
        reassociated.internal_tensors[3].zero_()
        reassociated.internal_tensors[3][0, :, :rank].copy_(
            left[:, :rank] * square_root[:rank][None, :]
        )
        reassociated.internal_tensors[2][:rank].copy_(
            (
                square_root[:rank, None]
                * right_h[:rank]
            ).reshape(rank, source[6], source[4])
        )
    reassociated.train(model.training)
    return reassociated


def edge_parent_gradient_scores(
    model: PositiveMultiplicationTreeMetric,
    *,
    channel_starts: Sequence[int] | None = None,
) -> dict[int, float]:
    """Score tree edges by the native-loss gradient entering their channels."""

    parents = model.topology.parent_map()
    if channel_starts is None:
        starts = (0,) * model.topology.root
    else:
        starts = tuple(int(value) for value in channel_starts)
        if len(starts) != model.topology.root:
            raise ValueError("one channel start is required per nonroot edge")
    scores: dict[int, float] = {}
    for child in range(model.topology.root):
        parent, side = parents[child]
        parent_tensor = model.internal_tensors[parent - model.leaf_count]
        gradient = parent_tensor.grad
        if gradient is None:
            scores[child] = 0.0
            continue
        start = starts[child]
        if not 0 <= start < model.edge_dimensions[child]:
            raise ValueError("edge gradient channel range is empty")
        selected = (
            gradient[:, start:, :]
            if side == 0
            else gradient[:, :, start:]
        )
        scores[child] = float(torch.linalg.vector_norm(selected))
    return scores
