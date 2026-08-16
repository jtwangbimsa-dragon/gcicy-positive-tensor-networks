#!/usr/bin/env python3
"""Run the deterministic mathematical-kernel and pipeline audit tests."""

from __future__ import annotations

import subprocess
import sys


TESTS = (
    "tests/test_positive_tensor_network.py",
    "tests/test_positive_tensor_network_blocking.py",
    "tests/test_positive_tensor_network_training_objectives.py",
    "tests/test_gcicy_tail_statistics.py",
    "tests/test_algebraic_metric_power_lift.py",
    "tests/test_metric_regularity.py",
    "tests/test_positive_multiplication_tree.py",
    "tests/test_type11_reference_whitened_full_h.py",
)


def main() -> int:
    command = [sys.executable, "-m", "pytest", "-q", "-rs", *TESTS]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
