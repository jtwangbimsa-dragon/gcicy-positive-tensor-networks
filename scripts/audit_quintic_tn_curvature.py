#!/usr/bin/env python3
"""Audit Ricci curvature of a positive TN metric on fixed Fermat points."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from gcicy_metric.quintic_curvature import curvature_at_fermat_point  # noqa: E402


PROBABILITIES = np.array([0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--model-tail-arrays", type=Path, required=True)
    parser.add_argument("--blind-points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=5000)
    parser.add_argument("--fibre-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=202607193)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--chart-invariance-points",
        type=int,
        default=0,
        help="number of sampled projective points to re-evaluate in all five affine patches",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--vectorized-hessian", action="store_true")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        raise ValueError("weighted quantiles require finite positive-weight observations")
    selected_values = values[finite]
    selected_weights = weights[finite]
    order = np.argsort(selected_values, kind="mergesort")
    selected_values = selected_values[order]
    cumulative = np.cumsum(selected_weights[order])
    cumulative /= cumulative[-1]
    quantiles = np.interp(
        PROBABILITIES,
        cumulative,
        selected_values,
        left=selected_values[0],
        right=selected_values[-1],
    )
    return {
        f"q{probability:.3f}": float(value)
        for probability, value in zip(PROBABILITIES, quantiles, strict=True)
    }


def select_whole_fibres(
    point_count: int, sample_points: int, fibre_size: int, seed: int
) -> np.ndarray:
    if fibre_size <= 0 or point_count % fibre_size:
        raise ValueError("the blind point count must be divisible by the fibre size")
    if sample_points <= 0 or sample_points % fibre_size:
        raise ValueError("sample-points must be a positive multiple of fibre-size")
    if sample_points > point_count:
        raise ValueError("sample-points exceeds the blind pool")
    generator = np.random.default_rng(seed)
    fibre_count = point_count // fibre_size
    selected_fibres = np.sort(
        generator.choice(fibre_count, sample_points // fibre_size, replace=False)
    )
    offsets = np.arange(fibre_size, dtype=np.int64)
    return (selected_fibres[:, None] * fibre_size + offsets[None, :]).reshape(-1)


def cluster_standard_error(values: np.ndarray, fibre_size: int) -> float:
    clusters = values.reshape(-1, fibre_size).mean(axis=1)
    if len(clusters) < 2:
        return float("nan")
    return float(np.std(clusters, ddof=1) / math.sqrt(len(clusters)))


def chart_invariance_audit(
    model: torch.nn.Module,
    points: np.ndarray,
    indices: np.ndarray,
    *,
    requested_points: int,
    vectorize: bool,
) -> dict[str, Any] | None:
    if requested_points == 0:
        return None
    if requested_points < 0:
        raise ValueError("chart-invariance-points must be non-negative")

    records: list[dict[str, Any]] = []
    for point_index in indices:
        point = points[point_index]
        if np.min(np.abs(point)) < 0.1:
            continue
        patch_results = []
        valid = True
        for patch_index in range(5):
            representative = point / point[patch_index]
            representative[patch_index] = 1.0 + 0.0j
            if np.count_nonzero(
                np.isclose(representative, 1.0 + 0.0j, atol=2.0e-6)
            ) != 1:
                valid = False
                break
            result = curvature_at_fermat_point(
                model,
                representative,
                vectorize=vectorize,
            )
            patch_results.append(
                {
                    "patch_index": patch_index,
                    "dependent_index": result["chart"].dependent_index,
                    "ricci_scalar": result["ricci_scalar"],
                    "ricci_tensor_norm": result["ricci_tensor_norm"],
                    "corrected_quintic_residual": result[
                        "corrected_quintic_residual"
                    ],
                    "tangent_residual": result["tangent_residual"],
                }
            )
        if not valid:
            continue

        scalars = np.array(
            [result["ricci_scalar"] for result in patch_results], dtype=np.float64
        )
        norms = np.array(
            [result["ricci_tensor_norm"] for result in patch_results],
            dtype=np.float64,
        )
        records.append(
            {
                "blind_point_index": int(point_index),
                "patches": patch_results,
                "ricci_scalar_absolute_spread": float(np.ptp(scalars)),
                "ricci_scalar_relative_spread": float(
                    np.ptp(scalars) / max(np.max(np.abs(scalars)), 1.0e-12)
                ),
                "ricci_tensor_norm_absolute_spread": float(np.ptp(norms)),
                "ricci_tensor_norm_relative_spread": float(
                    np.ptp(norms) / max(np.max(np.abs(norms)), 1.0e-12)
                ),
            }
        )
        if len(records) == requested_points:
            break

    if len(records) != requested_points:
        raise RuntimeError(
            f"only {len(records)} points support the requested {requested_points}-point "
            "all-patch curvature audit"
        )
    return {
        "sample_points": len(records),
        "patches_per_point": 5,
        "maximum_ricci_scalar_absolute_spread": max(
            record["ricci_scalar_absolute_spread"] for record in records
        ),
        "maximum_ricci_scalar_relative_spread": max(
            record["ricci_scalar_relative_spread"] for record in records
        ),
        "maximum_ricci_tensor_norm_absolute_spread": max(
            record["ricci_tensor_norm_absolute_spread"] for record in records
        ),
        "maximum_ricci_tensor_norm_relative_spread": max(
            record["ricci_tensor_norm_relative_spread"] for record in records
        ),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})

    try:
        paths = {
            "model": args.model.expanduser().resolve(),
            "model_report": args.model_report.expanduser().resolve(),
            "model_tail_arrays": args.model_tail_arrays.expanduser().resolve(),
            "blind_points": args.blind_points.expanduser().resolve(),
        }
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        if args.progress_every <= 0:
            raise ValueError("progress-every must be positive")

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        payload = torch.load(paths["model"], map_location="cpu", weights_only=False)
        model = positive_tensor_network_from_artifact_payload(
            np.eye(5, dtype=np.complex128), payload, device=device
        )
        model = model.to(dtype=torch.complex128, device=device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        blind = np.load(paths["blind_points"], allow_pickle=False)
        points = np.asarray(blind["complex_points"], dtype=np.complex128)
        weights = np.asarray(blind["weights"], dtype=np.float64)
        omega_squared = np.asarray(blind["omega_squared"], dtype=np.float64)
        tail = np.load(paths["model_tail_arrays"], allow_pickle=False)
        saved_raw = np.asarray(tail["raw_log_volume_ratio"], dtype=np.float64)
        saved_weights = np.asarray(tail["weights"], dtype=np.float64)
        saved_omega = np.asarray(tail["omega_squared"], dtype=np.float64)
        if not (
            len(points)
            == len(weights)
            == len(omega_squared)
            == len(saved_raw)
            == len(saved_weights)
            == len(saved_omega)
        ):
            raise RuntimeError("blind points and saved TN tail arrays are not aligned")
        if not (
            np.array_equal(weights, saved_weights)
            and np.array_equal(omega_squared, saved_omega)
        ):
            raise RuntimeError("blind integration labels differ from the saved TN audit")

        indices = select_whole_fibres(
            len(points), args.sample_points, args.fibre_size, args.seed
        )
        sample_count = len(indices)
        determinants = np.empty(sample_count, dtype=np.float64)
        minimum_eigenvalues = np.empty(sample_count, dtype=np.float64)
        maximum_eigenvalues = np.empty(sample_count, dtype=np.float64)
        ricci_scalars = np.empty(sample_count, dtype=np.float64)
        ricci_norms = np.empty(sample_count, dtype=np.float64)
        input_residuals = np.empty(sample_count, dtype=np.float64)
        corrected_residuals = np.empty(sample_count, dtype=np.float64)
        tangent_residuals = np.empty(sample_count, dtype=np.float64)
        patch_indices = np.empty(sample_count, dtype=np.int64)
        dependent_indices = np.empty(sample_count, dtype=np.int64)

        write_json(
            status_path,
            {
                "state": "running",
                "phase": "curvature",
                "sample_points": sample_count,
                "completed": 0,
            },
        )
        curvature_started = time.perf_counter()
        for row, point_index in enumerate(indices):
            result = curvature_at_fermat_point(
                model,
                points[point_index],
                vectorize=args.vectorized_hessian,
            )
            chart = result["chart"]
            determinants[row] = result["determinant"]
            minimum_eigenvalues[row] = result["minimum_eigenvalue"]
            maximum_eigenvalues[row] = result["maximum_eigenvalue"]
            ricci_scalars[row] = result["ricci_scalar"]
            ricci_norms[row] = result["ricci_tensor_norm"]
            input_residuals[row] = chart.input_quintic_residual
            corrected_residuals[row] = result["corrected_quintic_residual"]
            tangent_residuals[row] = result["tangent_residual"]
            patch_indices[row] = chart.patch_index
            dependent_indices[row] = chart.dependent_index
            completed = row + 1
            if completed % args.progress_every == 0 or completed == sample_count:
                elapsed = time.perf_counter() - curvature_started
                print(
                    f"curvature={completed}/{sample_count} "
                    f"seconds_per_point={elapsed / completed:.3f}",
                    flush=True,
                )
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "curvature",
                        "sample_points": sample_count,
                        "completed": completed,
                        "seconds_per_point": elapsed / completed,
                    },
                )

        sample_weights = weights[indices]
        sample_omega = omega_squared[indices]
        det_over_omega = determinants / sample_omega
        saved_log_ratio = saved_raw[indices]
        recomputed_log_ratio = np.log(det_over_omega)
        log_ratio_difference = recomputed_log_ratio - saved_log_ratio

        full_volume_omega = float(np.mean(weights))
        full_det_over_omega = np.exp(saved_raw)
        full_volume_k = float(np.mean(full_det_over_omega * weights))
        sample_volume_k = float(np.mean(det_over_omega * sample_weights))
        metric_measure_weights = det_over_omega * sample_weights
        absolute_ricci = np.abs(ricci_scalars)
        chart_invariance = chart_invariance_audit(
            model,
            points,
            indices,
            requested_points=args.chart_invariance_points,
            vectorize=args.vectorized_hessian,
        )
        numerator_terms = metric_measure_weights * absolute_ricci
        numerator = float(np.mean(numerator_terms))
        ricci_measure = full_volume_k ** (1.0 / 3.0) * numerator / full_volume_omega
        numerator_standard_error = cluster_standard_error(
            numerator_terms, args.fibre_size
        )
        ricci_measure_standard_error = (
            full_volume_k ** (1.0 / 3.0)
            * numerator_standard_error
            / full_volume_omega
        )
        metric_weight_sum = float(np.sum(metric_measure_weights))

        arrays_path = output_dir / "curvature_arrays.npz"
        np.savez_compressed(
            arrays_path,
            indices=indices,
            determinant=determinants,
            min_eigenvalue=minimum_eigenvalues,
            max_eigenvalue=maximum_eigenvalues,
            ricci_scalar=ricci_scalars,
            ricci_tensor_norm=ricci_norms,
            det_over_omega=det_over_omega,
            integration_weight=sample_weights,
            metric_measure_weight=metric_measure_weights,
            input_quintic_residual=input_residuals,
            corrected_quintic_residual=corrected_residuals,
            tangent_residual=tangent_residuals,
            patch_index=patch_indices,
            dependent_index=dependent_indices,
            saved_log_det_over_omega=saved_log_ratio,
            recomputed_log_det_over_omega=recomputed_log_ratio,
        )

        report = {
            "schema": "quintic-positive-tensor-network-curvature-audit-v1",
            "scientific_scope": {
                "geometry": "Fermat quintic hypersurface X_5 in P^4",
                "metric": "positive tensor-network Kahler metric",
                "curvature_method": (
                    "exact PyTorch automatic differentiation in the cymetric affine/"
                    "implicit chart; no finite differences"
                ),
                "ricci_measure_definition": (
                    "Vol_K^(1/3)/Vol_Omega * integral_X dVol_K |R|"
                ),
                "claim_limit": (
                    "The curvature integral and tails are Monte Carlo estimates on "
                    "independently selected whole fibres, not global sup-norm certificates."
                ),
            },
            "configuration": vars(args),
            "model": {
                "site_count_k": int(payload["site_count"]),
                "bond_dimension_D": int(payload["bond_dimension"]),
                "dictionary_rank_q": int(payload["physical_dictionary_rank"]),
                "trainable_real_parameter_count": int(model.trainable_real_parameter_count),
                "source_report": json.loads(
                    paths["model_report"].read_text(encoding="utf-8")
                )["schema"],
            },
            "sampling": {
                "blind_pool_points": len(points),
                "sample_points": sample_count,
                "sample_fibres": sample_count // args.fibre_size,
                "fibre_size": args.fibre_size,
                "seed": args.seed,
                "whole_fibre_cluster_standard_error": True,
            },
            "chart_validation": {
                "maximum_input_quintic_residual": float(np.max(input_residuals)),
                "maximum_corrected_quintic_residual": float(
                    np.max(corrected_residuals)
                ),
                "maximum_tangent_residual": float(np.max(tangent_residuals)),
                "maximum_abs_saved_log_ratio_difference": float(
                    np.max(np.abs(log_ratio_difference))
                ),
                "rms_saved_log_ratio_difference": float(
                    np.sqrt(np.mean(log_ratio_difference**2))
                ),
                "patch_counts": {
                    str(index): int(np.count_nonzero(patch_indices == index))
                    for index in range(5)
                },
                "dependent_counts": {
                    str(index): int(np.count_nonzero(dependent_indices == index))
                    for index in range(5)
                },
            },
            "chart_invariance": chart_invariance,
            "volume_estimates": {
                "full_blind_volume_omega": full_volume_omega,
                "full_blind_volume_k": full_volume_k,
                "curvature_sample_volume_k": sample_volume_k,
                "sample_relative_volume_k_difference": (
                    sample_volume_k / full_volume_k - 1.0
                ),
            },
            "curvature": {
                "ricci_measure": ricci_measure,
                "ricci_measure_cluster_standard_error": ricci_measure_standard_error,
                "metric_volume_weighted_mean_abs_ricci_scalar": float(
                    np.sum(metric_measure_weights * absolute_ricci) / metric_weight_sum
                ),
                "metric_volume_weighted_rms_ricci_scalar": float(
                    np.sqrt(
                        np.sum(metric_measure_weights * ricci_scalars**2)
                        / metric_weight_sum
                    )
                ),
                "abs_ricci_scalar_metric_volume_quantiles": weighted_quantiles(
                    absolute_ricci, metric_measure_weights
                ),
                "ricci_tensor_norm_metric_volume_quantiles": weighted_quantiles(
                    ricci_norms, metric_measure_weights
                ),
                "metric_minimum_eigenvalue_quantiles": weighted_quantiles(
                    minimum_eigenvalues, metric_measure_weights
                ),
                "metric_condition_number_quantiles": weighted_quantiles(
                    maximum_eigenvalues / minimum_eigenvalues,
                    metric_measure_weights,
                ),
            },
            "provenance": {
                **{name + "_sha256": sha256_file(path) for name, path in paths.items()},
                "curvature_arrays_sha256": sha256_file(arrays_path),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
            },
            "timing_seconds": {
                "curvature": time.perf_counter() - curvature_started,
                "wall_total": time.perf_counter() - started,
            },
        }
        report_path = output_dir / "report.json"
        write_json(report_path, report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report": str(report_path),
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
