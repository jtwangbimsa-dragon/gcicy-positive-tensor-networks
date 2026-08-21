#!/usr/bin/env python3
"""Run deterministic tests for the independent post-v1 experiment control plane."""

from __future__ import annotations

import subprocess
import sys


TESTS = (
    "tests/test_type11_positive_tensor_network_checkpoint.py",
    "tests/test_gcicy_tn_plateau_stage.py",
    "tests/test_gcicy_tn_study_arm.py",
    "tests/test_experiment_workflow.py",
    "tests/test_quintic_positive_tensor_network_checkpoint.py",
    "tests/test_quintic_positive_tensor_network_equivalence.py",
    "tests/test_quintic_tn_plateau_stage.py",
    "tests/test_quintic_tn_study_arm.py",
    "tests/test_cast_positive_tensor_network_precision.py",
    "tests/test_audit_quintic_tn_scaling_preflight.py",
    "tests/test_quintic_experiment_manifest.py",
)


def main() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "-rs", *TESTS])


if __name__ == "__main__":
    raise SystemExit(main())
