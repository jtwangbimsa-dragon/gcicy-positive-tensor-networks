#!/usr/bin/env python3
"""Audit two quintic positive-TN artifacts on frozen validation geometry.

The audit is deliberately development-only: it reads ``X_val`` and the
registered validation pullbacks, never a blind point or blind result.  The
JSON report is atomically replaced only after all comparisons have completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "quintic-positive-tensor-network-common-point-equivalence-v1"
MODEL_SCHEMA = "quintic-positive-tensor-network-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--absolute-potential-tolerance", type=float, default=1e-6)
    parser.add_argument("--absolute-metric-tolerance", type=float, default=1e-6)
    parser.add_argument("--relative-metric-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--allow-different-site-counts",
        action="store_true",
        help="explicitly permit a site-transfer comparison",
    )
    parser.add_argument(
        "--allow-different-precisions",
        action="store_true",
        help="explicitly permit native-dtype comparison across precisions",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in (
        "points",
        "chunk_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "absolute_potential_tolerance",
        "absolute_metric_tolerance",
        "relative_metric_tolerance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def validate_model_pair(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    allow_different_site_counts: bool,
    allow_different_precisions: bool,
) -> None:
    for label, payload in (("A", payload_a), ("B", payload_b)):
        if not isinstance(payload, dict) or payload.get("schema") != MODEL_SCHEMA:
            raise ValueError(f"model {label} is not a quintic positive-TN artifact")
        if payload.get("architecture") != "shared_local_dictionary":
            raise ValueError(f"model {label} is not shared-local-dictionary")
        if payload.get("precision") not in {"complex64", "complex128"}:
            raise ValueError(f"model {label} has an unsupported precision")
    if int(payload_a["source_degree"]) != int(payload_b["source_degree"]):
        raise ValueError("models use different source degrees")
    if (
        int(payload_a["site_count"]) != int(payload_b["site_count"])
        and not allow_different_site_counts
    ):
        raise ValueError(
            "models use different site counts; pass --allow-different-site-counts"
        )
    if (
        str(payload_a["precision"]) != str(payload_b["precision"])
        and not allow_different_precisions
    ):
        raise ValueError(
            "models use different precisions; pass --allow-different-precisions"
        )


def comparison_statistics(
    potentials_a: np.ndarray,
    potentials_b: np.ndarray,
    metrics_a: np.ndarray,
    metrics_b: np.ndarray,
) -> dict[str, float]:
    potentials_a = np.asarray(potentials_a, dtype=np.float64)
    potentials_b = np.asarray(potentials_b, dtype=np.float64)
    metrics_a = np.asarray(metrics_a, dtype=np.complex128)
    metrics_b = np.asarray(metrics_b, dtype=np.complex128)
    if potentials_a.shape != potentials_b.shape:
        raise ValueError("potential arrays have different shapes")
    if metrics_a.shape != metrics_b.shape or metrics_a.shape[0] != len(potentials_a):
        raise ValueError("metric arrays have incompatible shapes")
    if not all(
        np.all(np.isfinite(value))
        for value in (
            potentials_a,
            potentials_b,
            metrics_a.real,
            metrics_a.imag,
            metrics_b.real,
            metrics_b.imag,
        )
    ):
        raise FloatingPointError("nonfinite potential or metric in audit")
    metric_difference = metrics_b - metrics_a
    absolute_rows = np.linalg.norm(
        metric_difference.reshape(len(metrics_a), -1), axis=1
    )
    denominator = np.maximum(
        np.linalg.norm(metrics_a.reshape(len(metrics_a), -1), axis=1),
        np.finfo(np.float64).tiny,
    )
    return {
        "maximum_absolute_potential_difference": float(
            np.max(np.abs(potentials_b - potentials_a), initial=0.0)
        ),
        "maximum_absolute_metric_frobenius_difference": float(
            np.max(absolute_rows, initial=0.0)
        ),
        "maximum_relative_metric_frobenius_difference": float(
            np.max(absolute_rows / denominator, initial=0.0)
        ),
    }


def construction_claims(
    payload_a: dict[str, Any], payload_b: dict[str, Any], model_a_sha256: str
) -> dict[str, Any]:
    """Separate algebraic initialization evidence from the sampled metric gate."""

    learned_norm_exact = False
    mechanism = None
    if payload_b.get("site_transfer_source_model_sha256") == model_a_sha256 and str(
        payload_b.get("site_transfer_rule", "")
    ).startswith("exact learned-norm MPS repetition"):
        learned_norm_exact = True
        mechanism = "repeated learned-norm MPS with function-preserving bridges"
    expansion = payload_b.get("bond_expansion")
    if (
        isinstance(expansion, dict)
        and expansion.get("source_model_sha256") == model_a_sha256
        and float(expansion.get("initialization_noise", math.nan)) == 0.0
    ):
        learned_norm_exact = True
        mechanism = "zero-noise nested bond embedding"
    return {
        "learned_norm_exact_by_construction": learned_norm_exact,
        "learned_norm_construction_mechanism": mechanism,
        "full_metric_audited_with_positive_floor": True,
        "full_metric_claim": (
            "finite frozen-validation audit only; the positive-floor term means "
            "learned-norm exactness is not asserted as global full-model identity"
        ),
    }


def _registered_validation_inputs(
    source_dir: Path, pullbacks_dir: Path
) -> tuple[Path, Path, Path, dict[str, str]]:
    dataset = source_dir / "training_data" / "dataset.npz"
    validation_pullbacks = pullbacks_dir / "validation_pullbacks.npy"
    pullback_report_path = pullbacks_dir / "report.json"
    for path in (dataset, validation_pullbacks, pullback_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_sha = sha256_file(dataset)
    pullback_sha = sha256_file(validation_pullbacks)
    report = json.loads(pullback_report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("pullback report must be a JSON object")
    source_hashes = report.get("source_sha256")
    output_hashes = report.get("output_sha256")
    if (
        not isinstance(source_hashes, dict)
        or source_hashes.get("dataset") != dataset_sha
    ):
        raise ValueError("validation pullbacks are not registered to this dataset")
    if (
        not isinstance(output_hashes, dict)
        or output_hashes.get("validation") != pullback_sha
    ):
        raise ValueError("validation pullback hash does not match its report")
    return (
        dataset,
        validation_pullbacks,
        pullback_report_path,
        {
            "dataset": dataset_sha,
            "validation_pullbacks": pullback_sha,
            "pullback_report": sha256_file(pullback_report_path),
        },
    )


def _evaluate(
    payload: dict[str, Any],
    x_values: np.ndarray,
    pullbacks: np.ndarray,
    *,
    chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload
    from scripts.train_quintic_positive_tensor_network_same_points import (
        source_features,
    )

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    precision = str(payload["precision"])
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    reference_h_value = payload["state_dict"]["reference_h"]
    if hasattr(reference_h_value, "detach"):
        reference_h_value = reference_h_value.detach().cpu().numpy()
    model = positive_tensor_network_from_artifact_payload(
        np.asarray(reference_h_value, dtype=np.complex128),
        payload,
        device=torch_device,
        trainable_physical_dictionary=False,
    )
    model.eval()
    potentials: list[np.ndarray] = []
    metrics: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_values), chunk_size):
            stop = min(start + chunk_size, len(x_values))
            values, derivatives = source_features(
                x_values[start:stop],
                pullbacks[start:stop],
                source_degree=int(payload["source_degree"]),
                complex_dtype=complex_dtype,
                device=torch_device,
            )
            potential, metric = model.potential_and_metric(values, derivatives)
            potentials.append(potential.detach().cpu().numpy().astype(np.float64))
            metrics.append(metric.detach().cpu().numpy().astype(np.complex128))
    return np.concatenate(potentials), np.concatenate(metrics)


def run(args: argparse.Namespace) -> int:
    import torch

    model_a = args.model_a.expanduser().resolve()
    model_b = args.model_b.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    for path in (model_a, model_b):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload_a = torch.load(model_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(model_b, map_location="cpu", weights_only=False)
    validate_model_pair(
        payload_a,
        payload_b,
        allow_different_site_counts=args.allow_different_site_counts,
        allow_different_precisions=args.allow_different_precisions,
    )
    dataset_path, pullbacks_path, pullback_report, input_hashes = (
        _registered_validation_inputs(source_dir, pullbacks_dir)
    )
    with np.load(dataset_path, allow_pickle=False) as dataset:
        if "X_val" not in dataset:
            raise ValueError("dataset has no X_val array")
        available = len(dataset["X_val"])
        point_count = min(args.points, available)
        if point_count <= 0:
            raise ValueError("validation set is empty")
        x_values = np.asarray(dataset["X_val"][:point_count], dtype=np.float32)
    pullback_array = np.load(pullbacks_path, mmap_mode="r")
    if len(pullback_array) < point_count:
        raise ValueError("validation pullbacks are shorter than X_val")
    pullbacks = np.asarray(pullback_array[:point_count])
    potential_a, metric_a = _evaluate(
        payload_a, x_values, pullbacks, chunk_size=args.chunk_size, device=args.device
    )
    potential_b, metric_b = _evaluate(
        payload_b, x_values, pullbacks, chunk_size=args.chunk_size, device=args.device
    )
    statistics = comparison_statistics(potential_a, potential_b, metric_a, metric_b)
    gates = {
        "absolute_potential": (
            statistics["maximum_absolute_potential_difference"]
            <= args.absolute_potential_tolerance
        ),
        "absolute_metric": (
            statistics["maximum_absolute_metric_frobenius_difference"]
            <= args.absolute_metric_tolerance
        ),
        "relative_metric": (
            statistics["maximum_relative_metric_frobenius_difference"]
            <= args.relative_metric_tolerance
        ),
    }
    success = all(gates.values())
    model_a_sha = sha256_file(model_a)
    model_b_sha = sha256_file(model_b)
    report = {
        "schema": SCHEMA,
        "status": "equivalent" if success else "not_equivalent",
        "success": success,
        "model_a": str(model_a),
        "model_a_sha256": model_a_sha,
        "model_b": str(model_b),
        "model_b_sha256": model_b_sha,
        "model_a_site_count": int(payload_a["site_count"]),
        "model_b_site_count": int(payload_b["site_count"]),
        "model_a_bond_dimension": int(payload_a["bond_dimension"]),
        "model_b_bond_dimension": int(payload_b["bond_dimension"]),
        "model_a_precision": str(payload_a["precision"]),
        "model_b_precision": str(payload_b["precision"]),
        "source_degree": int(payload_a["source_degree"]),
        "allow_different_site_counts": bool(args.allow_different_site_counts),
        "allow_different_precisions": bool(args.allow_different_precisions),
        "source_run_dir": str(source_dir),
        "pullbacks_dir": str(pullbacks_dir),
        "pullback_report": str(pullback_report),
        "input_sha256": input_hashes,
        "points": point_count,
        "chunk_size": int(args.chunk_size),
        "device": args.device,
        "tolerances": {
            "absolute_potential": float(args.absolute_potential_tolerance),
            "absolute_metric": float(args.absolute_metric_tolerance),
            "relative_metric": float(args.relative_metric_tolerance),
        },
        "statistics": statistics,
        "gates": gates,
        "equivalence_claims": construction_claims(payload_a, payload_b, model_a_sha),
    }
    write_json_atomic(args.out.expanduser().resolve(), report)
    print(json.dumps({"success": success, **statistics}, sort_keys=True), flush=True)
    return 0 if success else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"quintic equivalence audit failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
