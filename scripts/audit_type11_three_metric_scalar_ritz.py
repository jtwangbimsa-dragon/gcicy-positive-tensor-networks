#!/usr/bin/env python3
"""Compare X11 scalar Ritz observables for full-H, residual-Phi, and TN metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    cluster_delete_group_jackknife_scalar_ritz,
    estimate_scalar_laplacian_ritz,
    get_adapter,
    metric_volume_weights,
    paired_metric_distortion,
    positive_tensor_network_from_artifact_payload,
)
from scripts.audit_type11_positive_tensor_network_scalar_ritz import (  # noqa: E402
    _aggregate_metric_level_rows,
    _sample_fresh_points,
    _trial_arrays,
    summarize_pairwise_shifts,
)
from scripts.run_cymetric_phi_gcicy_type11 import point_arrays  # noqa: E402
from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    section_values_and_jacobian_arrays,
)


METRIC_KEYS = ("full_h2", "residual_phi", "tensor_network")


def parse_integer_list(value: str) -> tuple[int, ...]:
    entries = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not entries:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-h-artifact", type=Path, required=True)
    parser.add_argument("--phi-base-h-artifact", type=Path, required=True)
    parser.add_argument("--phi-model", type=Path, required=True)
    parser.add_argument(
        "--phi-dtype",
        choices=("auto", "float32", "float64"),
        default="auto",
    )
    parser.add_argument("--tn-source-artifact", type=Path, required=True)
    parser.add_argument("--tn-model", type=Path, required=True)
    parser.add_argument("--adapter", default="p4p1_type11_hirzebruch_x3")
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--sampling-cluster-size", type=int, default=4)
    parser.add_argument(
        "--seeds",
        type=parse_integer_list,
        default=(86511, 86512, 86513, 86514),
    )
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--metric-chunk-size", type=int, default=256)
    parser.add_argument("--trial-levels", type=parse_integer_list, default=(2, 3))
    parser.add_argument("--eigenvalue-count", type=int, default=3)
    parser.add_argument("--mass-relative-threshold", type=float, default=1e-9)
    parser.add_argument("--cubic-feature-count", type=int, default=256)
    parser.add_argument("--cubic-feature-seed", type=int, default=314159)
    parser.add_argument("--cluster-jackknife-groups", type=int, default=16)
    parser.add_argument("--cluster-jackknife-level", type=int, default=3)
    parser.add_argument("--registered-relative-shift", type=float, default=0.1)
    parser.add_argument("--minimum-point-ess-per-rank", type=float, default=3.0)
    parser.add_argument("--minimum-fibre-ess-per-rank", type=float, default=5.0)
    parser.add_argument(
        "--maximum-jackknife-relative-standard-error",
        type=float,
        default=0.08,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResidualPhiMetricEvaluator:
    """Reconstruct the saved projective-invariant X11 residual-Phi metric."""

    def __init__(self, payload: dict[str, Any], *, device: Any, dtype_name: str):
        import torch
        from torch.func import hessian, vmap

        network = payload["network"]
        if int(network["input_dimension"]) != 29:
            raise ValueError("residual-Phi model must use the registered 29 features")
        resolved_dtype = (
            str(payload.get("dtype", "float32"))
            if dtype_name == "auto"
            else dtype_name
        )
        self.real_dtype = (
            torch.float64 if resolved_dtype == "float64" else torch.float32
        )
        self.complex_dtype = (
            torch.complex128 if resolved_dtype == "float64" else torch.complex64
        )
        self.dtype_name = resolved_dtype
        self.device = device
        activation_name = str(network["activation"])
        activation_type = {
            "gelu": torch.nn.GELU,
            "silu": torch.nn.SiLU,
            "tanh": torch.nn.Tanh,
        }[activation_name]

        layers: list[Any] = []
        width = 29
        for _ in range(int(network["hidden_layers"])):
            layers.append(torch.nn.Linear(width, int(network["hidden_width"])))
            layers.append(activation_type())
            width = int(network["hidden_width"])
        layers.append(torch.nn.Linear(width, 1, bias=False))
        self.potential = torch.nn.Sequential(*layers).to(
            device=device,
            dtype=self.real_dtype,
        )
        state = {
            (
                key.removeprefix("net.")
                if key.startswith("net.")
                else key
            ): value
            for key, value in payload["state_dict"].items()
        }
        self.potential.load_state_dict(state)
        self.potential.eval()
        self.potential_scale = float(network["potential_scale"])

        def insertion_maps(size: int) -> tuple[Any, Any]:
            maps = torch.zeros(
                (size, size, size - 1),
                dtype=self.real_dtype,
                device=device,
            )
            for fixed_index in range(size):
                cursor = 0
                for index in range(size):
                    if index == fixed_index:
                        continue
                    maps[fixed_index, index, cursor] = 1
                    cursor += 1
            return maps, torch.eye(size, dtype=self.real_dtype, device=device)

        x_maps, x_fixed = insertion_maps(5)
        y_maps, y_fixed = insertion_maps(2)

        def homogeneous_block(
            active_real: Any,
            active_imag: Any,
            maps: Any,
            fixed_vectors: Any,
            selector: Any,
        ) -> tuple[Any, Any]:
            insertion = torch.einsum("c,cij->ij", selector, maps)
            fixed = torch.einsum("c,ci->i", selector, fixed_vectors)
            return insertion @ active_real + fixed, insertion @ active_imag

        def density_features(real: Any, imag: Any) -> Any:
            denominator = torch.clamp(
                torch.sum(real**2 + imag**2),
                min=1e-12,
            )
            features = [
                (real[index] ** 2 + imag[index] ** 2) / denominator
                for index in range(len(real))
            ]
            for left in range(len(real)):
                for right in range(left + 1, len(real)):
                    features.append(
                        (
                            real[left] * real[right]
                            + imag[left] * imag[right]
                        )
                        / denominator
                    )
                    features.append(
                        (
                            imag[left] * real[right]
                            - real[left] * imag[right]
                        )
                        / denominator
                    )
            return torch.stack(features)

        def projective_features(coords: Any, chart_selector: Any) -> Any:
            product_selector = chart_selector.reshape(5, 2)
            x_selector = torch.sum(product_selector, dim=1)
            y_selector = torch.sum(product_selector, dim=0)
            real = coords[:6]
            imag = coords[6:]
            x_real, x_imag = homogeneous_block(
                real[:4],
                imag[:4],
                x_maps,
                x_fixed,
                x_selector,
            )
            y_real, y_imag = homogeneous_block(
                real[4:5],
                imag[4:5],
                y_maps,
                y_fixed,
                y_selector,
            )
            return torch.cat(
                (
                    density_features(x_real, x_imag),
                    density_features(y_real, y_imag),
                )
            )

        def single_potential(coords: Any, chart_selector: Any) -> Any:
            return self.potential_scale * self.potential(
                projective_features(coords, chart_selector)
            ).squeeze(-1)

        self.batched_hessian = vmap(
            hessian(single_potential, argnums=0),
            in_dims=(0, 0),
        )

    def evaluate(
        self,
        points: list[Any],
        base_metrics: np.ndarray,
        *,
        chunk_size: int,
    ) -> np.ndarray:
        import torch

        arrays = point_arrays(points)
        output = []
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            coords = torch.as_tensor(
                arrays["coords"][start:stop],
                dtype=self.real_dtype,
                device=self.device,
            )
            charts = torch.as_tensor(
                arrays["charts"][start:stop],
                dtype=torch.int64,
                device=self.device,
            )
            tangent = torch.as_tensor(
                arrays["tangent"][start:stop],
                dtype=self.complex_dtype,
                device=self.device,
            )
            selectors = torch.nn.functional.one_hot(
                charts,
                num_classes=10,
            ).to(dtype=self.real_dtype)
            with torch.enable_grad():
                real_hessian = self.batched_hessian(coords, selectors)
            h_xx = real_hessian[:, :6, :6]
            h_xy = real_hessian[:, :6, 6:]
            h_yx = real_hessian[:, 6:, :6]
            h_yy = real_hessian[:, 6:, 6:]
            ambient = 0.25 * (h_xx + h_yy).to(self.complex_dtype)
            ambient -= 0.25j * (h_xy - h_yx).to(self.complex_dtype)
            correction = torch.einsum(
                "nai,nab,nbj->nij",
                torch.conj(tangent),
                ambient,
                tangent,
            )
            correction = 0.5 * (
                correction + torch.conj(correction.transpose(1, 2))
            )
            base = torch.as_tensor(
                base_metrics[start:stop],
                dtype=self.complex_dtype,
                device=self.device,
            )
            metric = base + correction
            metric = 0.5 * (
                metric + torch.conj(metric.transpose(1, 2))
            )
            output.append(metric.detach().cpu().numpy())
        return np.concatenate(output).astype(np.complex128, copy=False)


def evaluate_tn(
    model: Any,
    source: Any,
    adapter: Any,
    points: list[Any],
    *,
    device: Any,
    complex_dtype: Any,
    chunk_size: int,
) -> np.ndarray:
    import torch

    values, derivatives = section_values_and_jacobian_arrays(
        adapter,
        points,
        source.section_exponents,
    )
    rows = []
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            values_batch = torch.as_tensor(
                values[start:stop],
                dtype=complex_dtype,
                device=device,
            )
            derivatives_batch = torch.as_tensor(
                derivatives[start:stop],
                dtype=complex_dtype,
                device=device,
            )
            _, metric = model.potential_and_metric(
                values_batch,
                derivatives_batch,
            )
            rows.append(metric.detach().cpu().numpy())
    return np.concatenate(rows).astype(np.complex128, copy=False)


def main() -> None:
    import torch

    args = parse_args()
    levels = tuple(int(value) for value in args.trial_levels)
    if tuple(sorted(levels)) != levels or args.cluster_jackknife_level not in levels:
        raise SystemExit("trial levels must be sorted and include the jackknife level")
    if args.points <= 0 or args.points % args.sampling_cluster_size:
        raise SystemExit("points must be divisible by the sampling cluster size")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = time.perf_counter()

    paths = {
        "full_h2": args.full_h_artifact.expanduser().resolve(),
        "phi_base_h2": args.phi_base_h_artifact.expanduser().resolve(),
        "phi_model": args.phi_model.expanduser().resolve(),
        "tn_source_h2": args.tn_source_artifact.expanduser().resolve(),
        "tn_model": args.tn_model.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    adapter = get_adapter(args.adapter)
    geometry = adapter.make_model(args.model_seed, exact=True)
    full_h = adapter.load_h_artifact(paths["full_h2"], geometry)
    phi_base_h = adapter.load_h_artifact(paths["phi_base_h2"], geometry)
    tn_source = adapter.load_h_artifact(paths["tn_source_h2"], geometry)

    tn_payload = torch.load(paths["tn_model"], map_location="cpu", weights_only=False)
    tn_precision = str(tn_payload["precision"])
    tn_dtype = torch.complex64 if tn_precision == "complex64" else torch.complex128
    tn_model = positive_tensor_network_from_artifact_payload(
        tn_source.h_matrix,
        tn_payload,
        device=device,
    )
    tn_model.eval()

    phi_payload = torch.load(paths["phi_model"], map_location="cpu", weights_only=False)
    expected_phi_hash = phi_payload.get("h_artifact_sha256")
    if expected_phi_hash is not None and expected_phi_hash != sha256(paths["phi_base_h2"]):
        raise ValueError("Phi checkpoint and base-H artifact hashes disagree")
    phi_evaluator = ResidualPhiMetricEvaluator(
        phi_payload,
        device=device,
        dtype_name=args.phi_dtype,
    )

    dataset_rows: list[dict[str, Any]] = []
    for dataset_index, seed in enumerate(args.seeds):
        dataset_started = time.perf_counter()
        points, cluster_ids, dataset = _sample_fresh_points(
            adapter,
            geometry,
            count=args.points,
            seed=seed,
            workers=args.workers,
            cluster_size=args.sampling_cluster_size,
        )
        print(f"dataset={dataset_index}: evaluating three metrics", flush=True)
        full_h_metrics = np.asarray(
            adapter.h_metrics(points, full_h),
            dtype=np.complex128,
        )
        phi_base_metrics = np.asarray(
            adapter.h_metrics(points, phi_base_h),
            dtype=np.complex128,
        )
        metrics = {
            "full_h2": full_h_metrics,
            "residual_phi": phi_evaluator.evaluate(
                points,
                phi_base_metrics,
                chunk_size=args.metric_chunk_size,
            ),
            "tensor_network": evaluate_tn(
                tn_model,
                tn_source,
                adapter,
                points,
                device=device,
                complex_dtype=tn_dtype,
                chunk_size=args.metric_chunk_size,
            ),
        }
        values, gradients, feature_counts = _trial_arrays(
            adapter,
            points,
            level=max(levels),
            cubic_count=args.cubic_feature_count,
            cubic_seed=args.cubic_feature_seed,
        )
        omega_weights = np.asarray(adapter.importance_weights(points), dtype=float)
        omega_weights /= np.sum(omega_weights)
        metric_reports: dict[str, Any] = {}
        for key in METRIC_KEYS:
            print(f"dataset={dataset_index}: Ritz metric={key}", flush=True)
            metric_weights, _ = metric_volume_weights(
                adapter,
                points,
                metrics[key],
            )
            trial_reports = {}
            for level in levels:
                feature_count = feature_counts[level]
                estimate = estimate_scalar_laplacian_ritz(
                    values[:, :feature_count],
                    gradients[:, :feature_count],
                    metrics[key],
                    metric_weights,
                    eigenvalue_count=args.eigenvalue_count,
                    mass_relative_threshold=args.mass_relative_threshold,
                    cluster_ids=cluster_ids,
                )
                if level == args.cluster_jackknife_level:
                    estimate["cluster_delete_group_jackknife"] = (
                        cluster_delete_group_jackknife_scalar_ritz(
                            values[:, :feature_count],
                            gradients[:, :feature_count],
                            metrics[key],
                            metric_weights,
                            cluster_ids,
                            eigenvalue_count=args.eigenvalue_count,
                            mass_relative_threshold=args.mass_relative_threshold,
                            group_count=args.cluster_jackknife_groups,
                        )
                    )
                trial_reports[str(level)] = estimate
            metric_reports[key] = {
                "metric_volume_effective_sample_size": float(
                    1.0 / np.sum(metric_weights**2)
                ),
                "trial_levels": trial_reports,
            }
        dataset_rows.append(
            {
                "dataset": dataset,
                "fibre_cluster_count": int(len(np.unique(cluster_ids))),
                "metrics": metric_reports,
                "metric_distortions": {
                    "full_h2_to_residual_phi": paired_metric_distortion(
                        metrics["full_h2"],
                        metrics["residual_phi"],
                        omega_weights,
                    ),
                    "full_h2_to_tensor_network": paired_metric_distortion(
                        metrics["full_h2"],
                        metrics["tensor_network"],
                        omega_weights,
                    ),
                    "residual_phi_to_tensor_network": paired_metric_distortion(
                        metrics["residual_phi"],
                        metrics["tensor_network"],
                        omega_weights,
                    ),
                },
                "runtime_seconds": float(time.perf_counter() - dataset_started),
            }
        )

    summaries = {
        key: {
            "trial_levels": {
                str(level): _aggregate_metric_level_rows(
                    dataset_rows,
                    key,
                    level,
                )
                for level in levels
            }
        }
        for key in METRIC_KEYS
    }
    comparisons = [
        summarize_pairwise_shifts(
            dataset_rows,
            candidate_key=candidate,
            reference_key="full_h2",
            level=level,
            registered_relative_shift=args.registered_relative_shift,
        )
        for candidate in ("residual_phi", "tensor_network")
        for level in levels
    ]
    finite_positive = all(
        np.all(np.asarray(report["eigenvalues"], dtype=float) > 0)
        for row in dataset_rows
        for key in METRIC_KEYS
        for report in row["metrics"][key]["trial_levels"].values()
    )
    point_ess_gate = all(
        summaries[key]["trial_levels"][str(level)][
            "minimum_point_ess_per_retained_rank"
        ]
        >= args.minimum_point_ess_per_rank
        for key in METRIC_KEYS
        for level in levels
    )
    fibre_ess_gate = all(
        summaries[key]["trial_levels"][str(level)][
            "minimum_fibre_ess_per_retained_rank"
        ]
        >= args.minimum_fibre_ess_per_rank
        for key in METRIC_KEYS
        for level in levels
    )
    jackknife_gate = all(
        summaries[key]["trial_levels"][str(args.cluster_jackknife_level)][
            "maximum_cluster_jackknife_relative_standard_error"
        ]
        <= args.maximum_jackknife_relative_standard_error
        for key in METRIC_KEYS
    )
    positive_metrics = all(
        summaries[key]["trial_levels"][str(level)]["minimum_metric_eigenvalue"] > 0
        for key in METRIC_KEYS
        for level in levels
    )
    report = {
        "schema": "type11-three-metric-scalar-ritz-v1",
        "artifacts": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "adapter": adapter.key,
        "model_seed": args.model_seed,
        "device": str(device),
        "phi_dtype": phi_evaluator.dtype_name,
        "seeds": list(args.seeds),
        "points_per_seed": args.points,
        "trial_levels": list(levels),
        "eigenvalue_count": args.eigenvalue_count,
        "mass_relative_threshold": args.mass_relative_threshold,
        "cubic_feature_count": args.cubic_feature_count,
        "cubic_feature_seed": args.cubic_feature_seed,
        "datasets": dataset_rows,
        "metric_summaries": summaries,
        "paired_ritz_comparisons": comparisons,
        "gates": {
            "finite_positive_spectra": bool(finite_positive),
            "positive_metrics": bool(positive_metrics),
            "point_ess_per_retained_rank": bool(point_ess_gate),
            "fibre_ess_per_retained_rank": bool(fibre_ess_gate),
            "cluster_jackknife_relative_standard_error": bool(jackknife_gate),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    report["success"] = bool(all(report["gates"].values()))
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"success={report['success']}", flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
