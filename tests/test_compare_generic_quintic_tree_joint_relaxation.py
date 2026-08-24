from argparse import Namespace

import numpy as np
import pytest
import torch

from scripts.compare_generic_quintic_tree_joint_relaxation import (
    assert_batch_plan_unchanged,
    assert_frozen_parameters_unchanged,
    batch_plan_digest,
    configure_trainable_scope,
    named_real_parameter_count,
    parameter_audit,
    role_training_policies,
    total_parameter_parity_audit,
)


class _ToyTree(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_leaf_tensor = torch.nn.Parameter(
            torch.ones(2, dtype=torch.complex64)
        )
        self.internal_tensors = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(3, dtype=torch.complex64))]
        )


def _initial_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def test_role_policies_preserve_legacy_defaults_and_allow_independent_overrides():
    legacy = Namespace(
        scheduler="cosine",
        control_scheduler=None,
        candidate_scheduler=None,
        control_trainable_scope="all",
        candidate_trainable_scope="all",
    )
    assert role_training_policies(legacy) == {
        "control": {"trainable_scope": "all", "scheduler": "cosine"},
        "candidate": {"trainable_scope": "all", "scheduler": "cosine"},
    }

    independent = Namespace(
        scheduler="cosine",
        control_scheduler="cosine",
        candidate_scheduler="constant",
        control_trainable_scope="internal",
        candidate_trainable_scope="all",
    )
    assert role_training_policies(independent) == {
        "control": {"trainable_scope": "internal", "scheduler": "cosine"},
        "candidate": {"trainable_scope": "all", "scheduler": "constant"},
    }


def test_internal_scope_freezes_leaf_and_strict_audit_rejects_any_leaf_change():
    model = _ToyTree()
    initial = _initial_state(model)
    trainable = configure_trainable_scope(model, "internal")
    assert trainable == ["internal_tensors.0"]
    assert named_real_parameter_count(model, set(trainable)) == 6
    assert model.shared_leaf_tensor.requires_grad is False

    with torch.no_grad():
        model.internal_tensors[0][0] += 0.25
    audit = parameter_audit(model, initial, trainable_names=set(trainable))
    assert audit["all_frozen_parameters_exactly_unchanged"] is True
    assert_frozen_parameters_unchanged(audit)

    with torch.no_grad():
        model.shared_leaf_tensor[0] += 1.0e-7
    audit = parameter_audit(model, initial, trainable_names=set(trainable))
    assert audit["all_frozen_parameters_exactly_unchanged"] is False
    with pytest.raises(RuntimeError, match="shared_leaf_tensor"):
        assert_frozen_parameters_unchanged(audit)


def test_batch_plan_and_total_parameter_audits_fail_closed():
    reference = np.arange(12, dtype=np.int64).reshape(3, 4)
    digest = batch_plan_digest(reference)
    assert_batch_plan_unchanged(reference.copy(), reference, digest)

    changed = reference.copy()
    changed[1, 2] += 1
    with pytest.raises(RuntimeError, match="same fixed batch plan"):
        assert_batch_plan_unchanged(changed, reference, digest)

    assert total_parameter_parity_audit(
        control=10,
        candidate=10,
        required=True,
    )["equal"]
    with pytest.raises(ValueError, match="equal total parameters"):
        total_parameter_parity_audit(
            control=10,
            candidate=12,
            required=True,
        )
