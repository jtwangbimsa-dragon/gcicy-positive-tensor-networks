#!/usr/bin/env python3
"""Pair an O(1) quintic TN into an O(2) TN with rectangular purification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    PositiveTensorNetworkMetric,
    compress_cores_to_shared_local_dictionary,
    pair_adjacent_dense_cores_to_symmetric_square,
    positive_tensor_network_from_artifact_payload,
    symmetric_square_reference_h,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    veronese_source_features_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dictionary-rank", type=int, default=375)
    parser.add_argument("--verification-points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=202607271)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def source_dense_cores(payload: dict[str, Any]) -> tuple[np.ndarray, ...]:
    state = payload["state_dict"]
    dictionary = np.asarray(
        state["physical_dictionary"].detach().cpu(), dtype=np.complex128
    )
    site_count = int(payload["site_count"])
    coefficients = tuple(
        np.asarray(
            state[f"coefficient_cores.{index}"].detach().cpu(),
            dtype=np.complex128,
        )
        for index in range(site_count)
    )
    return tuple(
        np.einsum("lrq,qpi->lrpi", coefficient, dictionary)
        for coefficient in coefficients
    )


def verification_data(
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    points = generator.normal(size=(count, 5))
    points = points + 1j * generator.normal(size=(count, 5))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    derivatives = generator.normal(size=(count, 5, 3))
    derivatives = derivatives + 1j * generator.normal(size=(count, 5, 3))
    return points, derivatives


def compare_models(
    source: torch.nn.Module,
    target: torch.nn.Module,
    *,
    count: int,
    seed: int,
) -> dict[str, float]:
    points, derivatives = verification_data(count, seed)
    target_values, target_derivatives = veronese_source_features_numpy(
        points,
        derivatives,
        source_degree=2,
    )
    dtype = source.reference_h.dtype
    with torch.no_grad():
        source_potential, source_metric = source.potential_and_metric(
            torch.tensor(points, dtype=dtype),
            torch.tensor(derivatives, dtype=dtype),
        )
        target_potential, target_metric = target.potential_and_metric(
            torch.tensor(target_values, dtype=dtype),
            torch.tensor(target_derivatives, dtype=dtype),
        )
    potential_difference = torch.abs(target_potential - source_potential)
    metric_difference = torch.linalg.matrix_norm(target_metric - source_metric)
    metric_scale = torch.clamp(torch.linalg.matrix_norm(source_metric), min=1.0e-30)
    relative_metric = metric_difference / metric_scale
    return {
        "point_count": count,
        "maximum_potential_absolute_difference": float(torch.max(potential_difference)),
        "maximum_metric_relative_frobenius_difference": float(
            torch.max(relative_metric)
        ),
        "mean_metric_relative_frobenius_difference": float(
            torch.mean(relative_metric)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.dictionary_rank <= 0 or args.verification_points <= 0:
        raise ValueError("rank and verification count must be positive")
    source_path = args.source_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})

    try:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        expected = {
            "schema": "quintic-positive-tensor-network-v1",
            "architecture": "shared_local_dictionary",
            "source_degree": 1,
        }
        observed = {key: payload.get(key, 1) for key in expected}
        if observed != expected:
            raise ValueError(f"source model is not an O(1) shared TN: {observed}")
        source_site_count = int(payload["site_count"])
        if source_site_count % 2:
            raise ValueError("source site count must be even")
        source_reference = np.asarray(
            payload["state_dict"]["reference_h"].detach().cpu(),
            dtype=np.complex128,
        )
        if source_reference.shape != (5, 5):
            raise ValueError("quintic O(1) source must have five sections")

        paired_cores = pair_adjacent_dense_cores_to_symmetric_square(
            source_dense_cores(payload)
        )
        output_dimension = paired_cores[0].shape[2]
        target_section_count = paired_cores[0].shape[3]
        local_map_dimension = output_dimension * target_section_count
        if args.dictionary_rank > local_map_dimension:
            raise ValueError(
                f"dictionary rank exceeds rectangular dimension {local_map_dimension}"
            )
        compression = compress_cores_to_shared_local_dictionary(
            paired_cores,
            args.dictionary_rank,
        )
        target_reference = symmetric_square_reference_h(source_reference)
        if target_reference.shape != (target_section_count, target_section_count):
            raise RuntimeError("paired reference dimension mismatch")

        precision = str(payload["precision"])
        dtype = torch.complex64 if precision == "complex64" else torch.complex128
        target = PositiveTensorNetworkMetric(
            target_reference,
            site_count=source_site_count // 2,
            bond_dimension=int(payload["bond_dimension"]),
            target_normalization=float(payload["target_normalization"]),
            output_dimension=output_dimension,
            positive_floor=float(payload["positive_floor"]),
            initialization_noise=0.0,
            physical_dictionary=compression.physical_dictionary,
            trainable_physical_dictionary=True,
            transfer_implementation=str(
                payload.get("transfer_implementation", "vectorized")
            ),
            seed=args.seed,
            dtype=dtype,
            device="cpu",
        )
        with torch.no_grad():
            for target_core, source_core in zip(
                target.coefficient_cores,
                compression.coefficient_cores,
                strict=True,
            ):
                target_core.copy_(torch.tensor(source_core, dtype=dtype))

        source = positive_tensor_network_from_artifact_payload(
            source_reference,
            payload,
            device="cpu",
        )
        verification = compare_models(
            source,
            target,
            count=args.verification_points,
            seed=args.seed + 1,
        )
        local_output_ranks = []
        local_rank15_errors = []
        for core in paired_cores:
            matrix = core.transpose(2, 0, 1, 3).reshape(output_dimension, -1)
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            local_output_ranks.append(
                int(np.sum(singular_values > singular_values[0] * 1.0e-10))
            )
            local_rank15_errors.append(
                float(
                    np.sqrt(
                        np.sum(np.square(singular_values[target_section_count:]))
                        / np.sum(np.square(singular_values))
                    )
                )
            )

        state = {
            key: value.detach().cpu() for key, value in target.state_dict().items()
        }
        total_degree = int(payload.get("total_degree", source_site_count))
        converted_payload = {
            "schema": "quintic-positive-tensor-network-v1",
            "geometry": payload.get(
                "geometry", "Fermat quintic hypersurface X_5 in P^4"
            ),
            "state_dict": state,
            "source_degree": 2,
            "total_degree": total_degree,
            "site_count": source_site_count // 2,
            "bond_dimension": int(payload["bond_dimension"]),
            "physical_dictionary_rank": args.dictionary_rank,
            "output_dimension": output_dimension,
            "architecture": "shared_local_dictionary",
            "transfer_implementation": str(
                payload.get("transfer_implementation", "vectorized")
            ),
            "trainable_physical_dictionary": False,
            "physical_dictionary_gauge": "fixed_row_orthonormal_basis",
            "target_normalization": float(payload["target_normalization"]),
            "positive_floor": float(payload["positive_floor"]),
            "precision": precision,
            "fixed_log_kappa": payload.get("fixed_log_kappa"),
            "fixed_log_kappa_source": payload.get("fixed_log_kappa_source"),
            "conversion": {
                "kind": "exact-adjacent-pairing-with-shared-dictionary-compression",
                "source_model": str(source_path),
                "source_model_sha256": sha256_file(source_path),
                "relative_core_frobenius_error": (
                    compression.relative_frobenius_error
                ),
            },
        }
        model_path = output_dir / "best_tensor_network.pt"
        torch.save(converted_payload, model_path)
        total_parameters = int(target.trainable_real_parameter_count)
        active_parameters = 2 * sum(
            parameter.numel() for parameter in target.coefficient_cores
        )
        report = {
            "schema": "quintic-o1-to-o2-rectangular-pairing-v1",
            "source_model": str(source_path),
            "source_model_sha256": sha256_file(source_path),
            "target_model": str(model_path),
            "target_model_sha256": sha256_file(model_path),
            "source_site_count": source_site_count,
            "target_site_count": source_site_count // 2,
            "source_section_count": 5,
            "target_section_count": target_section_count,
            "purification_output_dimension": output_dimension,
            "local_rectangular_map_dimension": local_map_dimension,
            "dictionary_rank": args.dictionary_rank,
            "total_real_parameter_count": total_parameters,
            "active_real_parameter_count_with_fixed_dictionary": active_parameters,
            "relative_core_frobenius_error": compression.relative_frobenius_error,
            "retained_core_energy_fraction": compression.retained_energy_fraction,
            "local_output_ranks": local_output_ranks,
            "local_rank15_relative_errors": local_rank15_errors,
            "verification": verification,
        }
        report_path = output_dir / "report.json"
        write_json(report_path, report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report_sha256": sha256_file(report_path),
            },
        )
        print(json.dumps(json_value(report), indent=2, sort_keys=True))
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "exception",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
