from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_quintic_tn_scaling_preflight.py"
SPEC = importlib.util.spec_from_file_location("quintic_scaling_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resolve_inputs_accepts_published_report_and_null_blind(tmp_path: Path) -> None:
    run_dir = tmp_path / "arm"
    run_dir.mkdir()
    source = tmp_path / "source"
    pullbacks = tmp_path / "pullbacks"
    model = run_dir / "final_model.pt"
    report = run_dir / "final_training_report.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {
                    "source_run_dir": str(source),
                    "blind_reference_run_dir": None,
                    "pullbacks_dir": str(pullbacks),
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        run_dir=run_dir,
        run_report=report,
        model=model,
        source_run_dir=None,
        blind_reference_run_dir=None,
        pullbacks_dir=None,
        output=None,
    )
    paths = MODULE.resolve_inputs(args)
    assert paths["report"] == report.resolve()
    assert paths["model"] == model.resolve()
    assert paths["source"] == source.resolve()
    assert paths["blind"] == source.resolve()
    assert paths["pullbacks"] == pullbacks.resolve()
    assert paths["output"] == (run_dir / "scaling_preflight_audit.json").resolve()


def test_validation_input_hashes_exclude_blind_material(tmp_path: Path) -> None:
    source = tmp_path / "source"
    pullbacks = tmp_path / "pullbacks"
    (source / "training_data").mkdir(parents=True)
    pullbacks.mkdir()
    files = {
        source / "training_data" / "dataset.npz": b"dataset",
        pullbacks / "report.json": b"registry",
        pullbacks / "validation_pullbacks.npy": b"validation",
    }
    for path, content in files.items():
        path.write_bytes(content)
    paths = {"source": source, "pullbacks": pullbacks, "blind": tmp_path / "absent"}
    hashes = MODULE.audited_input_hashes(paths, "validation")
    assert hashes == {
        "dataset": hashlib.sha256(b"dataset").hexdigest(),
        "pullback_report": hashlib.sha256(b"registry").hexdigest(),
        "validation_pullbacks": hashlib.sha256(b"validation").hexdigest(),
    }
