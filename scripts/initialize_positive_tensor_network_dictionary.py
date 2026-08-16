#!/usr/bin/env python3
"""Initialize a trainable H-aligned shared-dictionary tensor network."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    PositiveTensorNetworkMetric,
    anchored_orthonormal_physical_dictionary,
    get_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--sampling-cluster-size", type=int, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--site-count", type=int, required=True)
    parser.add_argument("--bond-dimension", type=int, required=True)
    parser.add_argument("--dictionary-rank", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--positive-floor", type=float, default=1.0e-4)
    parser.add_argument(
        "--initialization-noise",
        type=float,
        default=0.0,
        help="complex core noise used to break dormant bond and site symmetries",
    )
    parser.add_argument(
        "--maximum-relative-reference-core-perturbation",
        type=float,
        default=1.0e-12,
        help="hard gate on the largest initialized-core deviation from the H anchor",
    )
    parser.add_argument(
        "--precision",
        choices=("complex64", "complex128"),
        default="complex128",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_square_root(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    value = 0.5 * (value + value.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues[0] <= 0 or not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("source H must be positive definite")
    root = (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conjugate().T
    return 0.5 * (root + root.conjugate().T)


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for tensor-network initialization") from exc

    args = parse_args()
    if (
        args.site_count <= 0
        or args.bond_dimension <= 0
        or args.dictionary_rank <= 0
        or args.sampling_cluster_size <= 0
        or args.initialization_noise < 0
        or args.maximum_relative_reference_core_perturbation < 0
    ):
        raise SystemExit("site count, bond, rank, and cluster size must be positive")
    adapter = get_adapter(args.adapter)
    geometry = adapter.make_model(args.model_seed, exact=True)
    source_path = args.source_artifact.expanduser().resolve()
    source = adapter.load_h_artifact(source_path, geometry)
    source_hash = sha256_file(source_path)
    square_root = positive_square_root(source.h_matrix)
    dictionary = anchored_orthonormal_physical_dictionary(
        square_root,
        args.dictionary_rank,
        seed=args.seed,
    )
    dtype = torch.complex64 if args.precision == "complex64" else torch.complex128
    target_normalization = float(source.normalization) / args.site_count
    model = PositiveTensorNetworkMetric(
        source.h_matrix,
        site_count=args.site_count,
        bond_dimension=args.bond_dimension,
        target_normalization=target_normalization,
        positive_floor=args.positive_floor,
        initialization_noise=args.initialization_noise,
        physical_dictionary=dictionary,
        trainable_physical_dictionary=True,
        seed=args.seed,
        dtype=dtype,
        device="cpu",
    )
    materialized = tuple(
        core.detach().cpu().numpy() for core in model.materialized_cores()
    )
    reconstruction_errors = []
    for site, core in enumerate(materialized):
        expected = np.zeros_like(core)
        expected[0, 0] = square_root
        reconstruction_errors.append(
            float(
                np.linalg.norm(core - expected)
                / max(np.linalg.norm(expected), np.finfo(float).tiny)
            )
        )
    maximum_reconstruction_error = max(reconstruction_errors)
    if (
        maximum_reconstruction_error
        > args.maximum_relative_reference_core_perturbation
    ):
        raise FloatingPointError(
            "initialized cores exceed the registered reference perturbation gate: "
            f"observed={maximum_reconstruction_error:.6e}, "
            "maximum="
            f"{args.maximum_relative_reference_core_perturbation:.6e}"
        )
    flattened = dictionary.reshape(args.dictionary_rank, -1)
    row_orthonormality_error = float(
        np.linalg.norm(
            flattened @ flattened.conjugate().T - np.eye(args.dictionary_rank),
            ord=2,
        )
    )
    target_degree = [int(value) * args.site_count for value in source.degree]
    payload = {
        "schema": "type11-positive-tensor-network-v1",
        "adapter": adapter.key,
        "model_seed": int(args.model_seed),
        "sampling_cluster_size": int(args.sampling_cluster_size),
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "source_artifact": str(source_path),
        "source_artifact_sha256": source_hash,
        "teacher_artifact": None,
        "teacher_artifact_sha256": None,
        "training_mode": "teacher_free_h_aligned_dictionary_initialization",
        "source_degree": list(source.degree),
        "target_degree": target_degree,
        "source_section_exponents": source.section_exponents,
        "site_count": int(args.site_count),
        "bond_dimension": int(args.bond_dimension),
        "architecture": model.architecture,
        "physical_dictionary_rank": model.physical_dictionary_rank,
        "trainable_physical_dictionary": True,
        "physical_dictionary_gauge": "row_orthonormal",
        "target_normalization": target_normalization,
        "positive_floor": float(args.positive_floor),
        "precision": args.precision,
        "fixed_log_kappa": None,
        "fixed_log_kappa_source": "requires_training_pool_estimation",
        "dictionary_initialization_seed": int(args.seed),
        "initialization_noise": float(args.initialization_noise),
        "maximum_relative_reference_core_perturbation": float(
            args.maximum_relative_reference_core_perturbation
        ),
        "observed_maximum_relative_reference_core_perturbation": (
            maximum_reconstruction_error
        ),
        "dictionary_initialization_rule": (
            "normalized source-H square-root anchor plus deterministic "
            "orthogonal completion and registered complex core perturbation"
        ),
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(out_path)
    summary = {
        "schema": "positive-tensor-network-h-aligned-dictionary-initialization-v1",
        "adapter": adapter.key,
        "model_seed": int(args.model_seed),
        "source_artifact": str(source_path),
        "source_artifact_sha256": source_hash,
        "model": str(out_path),
        "model_sha256": sha256_file(out_path),
        "source_section_count": int(len(source.h_matrix)),
        "site_count": int(args.site_count),
        "target_degree": target_degree,
        "bond_dimension": int(args.bond_dimension),
        "physical_dictionary_rank": int(args.dictionary_rank),
        "trainable_real_parameter_count": int(model.trainable_real_parameter_count),
        "seed": int(args.seed),
        "initialization_noise": float(args.initialization_noise),
        "maximum_reference_core_reconstruction_error": (
            maximum_reconstruction_error
        ),
        "maximum_relative_reference_core_perturbation_gate": float(
            args.maximum_relative_reference_core_perturbation
        ),
        "row_orthonormality_error": row_orthonormality_error,
        "teacher_artifact_sha256": None,
    }
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temporary.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_temporary.replace(summary_path)
    print(
        f"adapter={adapter.key} sites={args.site_count} "
        f"rank={args.dictionary_rank} "
        f"parameters={model.trainable_real_parameter_count} "
        f"reference_error={maximum_reconstruction_error:.3e}",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
