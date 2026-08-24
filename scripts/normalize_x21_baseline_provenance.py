#!/usr/bin/env python3
"""Normalize raw X21 baseline/recovery artifacts into fixed certificates."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.x21_baseline_provenance import main  # noqa: E402


if __name__ == "__main__":
    main()
