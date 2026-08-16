#!/usr/bin/env python3
"""Audit a saved quintic H matrix on the common 200k cymetric blind points."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_quintic_full_h_same_points import (
    exponent_compositions,
    prepare_features,
    ratio_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--h-artifact", type=Path, required=True)
    parser.add_argument("--matrix-key", default="symmetrized_global_h_matrix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--test-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_spectrum(matrix: np.ndarray) -> dict[str, float]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
    return {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "relative_hermiticity_error": float(
            np.linalg.norm(matrix - matrix.conj().T) / np.linalg.norm(matrix)
        ),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source_dir = args.source_run_dir.resolve()
    pullbacks_dir = (
        args.pullbacks_dir.resolve()
        if args.pullbacks_dir is not None
        else source_dir / "full_h_common_geometry"
    )
    h_artifact_path = args.h_artifact.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})

    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU")
        device = torch.device(args.device)
        blind_path = source_dir / "blind_points.npz"
        pullbacks_path = pullbacks_dir / "blind_pullbacks.npy"
        pullback_report_path = pullbacks_dir / "report.json"
        cymetric_tail_path = source_dir / "blind_test_tail_arrays.npz"
        for path in (
            h_artifact_path,
            blind_path,
            pullbacks_path,
            pullback_report_path,
            cymetric_tail_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        pullback_report = json.loads(pullback_report_path.read_text(encoding="utf-8"))
        blind_hash = sha256_file(blind_path)
        if pullback_report.get("source_sha256", {}).get("blind_points") != blind_hash:
            raise RuntimeError("blind pullbacks do not match the source blind points")
        registered_blind_hash = pullback_report.get("output_sha256", {}).get("blind")
        pullbacks_hash = sha256_file(pullbacks_path)
        if registered_blind_hash != pullbacks_hash:
            raise RuntimeError("blind pullback hash does not match its report")

        h_artifact = np.load(h_artifact_path, allow_pickle=False)
        if args.matrix_key not in h_artifact.files:
            raise KeyError(f"{args.matrix_key} not found in {h_artifact_path}")
        matrix = np.asarray(h_artifact[args.matrix_key], dtype=np.complex64)
        matrix = 0.5 * (matrix + matrix.conj().T)
        exponents = (
            np.asarray(h_artifact["exponents"], dtype=np.int64)
            if "exponents" in h_artifact.files
            else exponent_compositions(args.degree, 5)
        )
        if matrix.shape != (len(exponents), len(exponents)):
            raise RuntimeError("H shape does not match the degree-k section count")

        blind = np.load(blind_path, allow_pickle=False)
        x_values = np.asarray(blind["X"], dtype=np.float32)
        weights = np.asarray(blind["weights"], dtype=np.float64)
        omega = np.asarray(blind["omega_squared"], dtype=np.float64)
        pullbacks = np.load(pullbacks_path, mmap_mode="r")
        if len(pullbacks) != len(x_values):
            raise RuntimeError("blind pullback count does not match blind points")

        matrix_t = torch.tensor(matrix, dtype=torch.complex64, device=device)
        normalization = 1.0 / (math.pi * args.degree)
        raw = np.empty(len(x_values), dtype=np.float64)
        minimum_eigenvalues = np.empty(len(x_values), dtype=np.float64)
        write_json(status_path, {"state": "running", "phase": "blind_audit"})
        for start in range(0, len(x_values), args.test_batch_size):
            stop = min(start + args.test_batch_size, len(x_values))
            values, derivatives = prepare_features(
                x_values[start:stop],
                pullbacks[start:stop],
                exponents,
                args.feature_batch_size,
            )
            values_t = torch.tensor(values, dtype=torch.complex64, device=device)
            derivatives_t = torch.tensor(
                derivatives, dtype=torch.complex64, device=device
            )
            with torch.no_grad():
                h_values = torch.einsum("ab,nb->na", matrix_t, values_t)
                denominator = torch.real(
                    torch.einsum("na,na->n", torch.conj(values_t), h_values)
                )
                h_derivatives = torch.einsum(
                    "ab,nbj->naj", matrix_t, derivatives_t
                )
                first = torch.einsum(
                    "nmi,nmj->nji", torch.conj(derivatives_t), h_derivatives
                )
                gradient = torch.einsum(
                    "nm,nmj->nj", torch.conj(values_t), h_derivatives
                )
                metric = first / denominator[:, None, None]
                metric = metric - (
                    gradient[:, :, None]
                    * torch.conj(gradient)[:, None, :]
                    / denominator[:, None, None] ** 2
                )
                metric = normalization * 0.5 * (
                    metric + torch.conj(torch.transpose(metric, 1, 2))
                )
                eigenvalues = torch.linalg.eigvalsh(metric)
                block_raw = torch.log(
                    torch.clamp(eigenvalues, min=1.0e-12)
                ).sum(dim=1)
                block_raw = block_raw - torch.log(
                    torch.tensor(omega[start:stop], dtype=torch.float32, device=device)
                )
            raw[start:stop] = block_raw.cpu().numpy()
            minimum_eigenvalues[start:stop] = torch.min(
                eigenvalues, dim=1
            ).values.cpu().numpy()

        statistics, ratio = ratio_statistics(raw, weights, minimum_eigenvalues)
        source_arrays = np.load(cymetric_tail_path, allow_pickle=False)
        phi_ratio = np.asarray(source_arrays["normalized_ratio"], dtype=np.float64)
        worst_indices = np.argsort(np.abs(1.0 - ratio))[-64:][::-1]
        arrays_path = output_dir / "blind_test_tail_arrays.npz"
        np.savez_compressed(
            arrays_path,
            weights=weights,
            omega_squared=omega,
            raw_log_ratio=raw,
            normalized_ratio=ratio,
            min_eigenvalue=minimum_eigenvalues,
            cymetric_phi_normalized_ratio=phi_ratio,
        )
        worst_path = output_dir / "worst_points.npz"
        np.savez_compressed(
            worst_path,
            indices=worst_indices,
            X=x_values[worst_indices],
            weights=weights[worst_indices],
            normalized_ratio=ratio[worst_indices],
            cymetric_phi_normalized_ratio=phi_ratio[worst_indices],
            min_eigenvalue=minimum_eigenvalues[worst_indices],
        )

        report = {
            "schema": "quintic-saved-h-common-blind-audit-v1",
            "configuration": {
                "source_run_dir": source_dir,
                "pullbacks_dir": pullbacks_dir,
                "h_artifact": h_artifact_path,
                "matrix_key": args.matrix_key,
                "degree": args.degree,
                "blind_count": len(x_values),
            },
            "scientific_scope": {
                "geometry": "Fermat quintic X_5 in P^4",
                "claim_limit": (
                    "This is a finite common-point blind audit, not a global "
                    "sup-norm certificate."
                ),
            },
            "h_spectrum": matrix_spectrum(matrix.astype(np.complex128)),
            "blind": {
                "audited_h": statistics,
                "cymetric_phi_source": json.loads(
                    (source_dir / "report.json").read_text()
                )["trained_phi_model"],
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
            },
            "timing_seconds": {"wall_total": time.perf_counter() - started},
            "artifacts_sha256": {
                "script": sha256_file(Path(__file__).resolve()),
                "h_artifact": sha256_file(h_artifact_path),
                "blind_points": blind_hash,
                "blind_pullbacks": pullbacks_hash,
                "pullback_report": sha256_file(pullback_report_path),
                "tail_arrays": sha256_file(arrays_path),
                "worst_points": sha256_file(worst_path),
            },
        }
        report_path = output_dir / "report.json"
        write_json(report_path, report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report_sha256": sha256_file(report_path),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps(json_value(report), indent=2, sort_keys=True), flush=True)
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "exception",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
