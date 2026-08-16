#!/usr/bin/env python3
"""Evaluate H4 teacher, compiled tree, direct TN, and residual phi on one blind set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.exact_lift_tree import (  # noqa: E402
    build_exact_power_lift_tree,
)
from gcicy_metric.pipeline.positive_multiplication_tree import (  # noqa: E402
    BinaryTreeTopology,
    PositiveMultiplicationTreeMetric,
)
from gcicy_metric.pipeline.projective_residual_phi import (  # noqa: E402
    ProjectiveResidualPhiMetric,
)
from scripts.train_generic_quintic_compiled_tree import (  # noqa: E402
    add_projective_inputs,
    metric_row,
    whiten_dataset,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    load_pool_arrays,
    make_dataset,
    paired_improvement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-tree", type=Path, required=True)
    parser.add_argument("--direct-tn", type=Path, required=True)
    parser.add_argument("--residual-phi", type=Path, required=True)
    parser.add_argument("--direct-tn-time-matched", type=Path)
    parser.add_argument("--residual-phi-time-matched", type=Path)
    parser.add_argument("--blind-points", type=Path, required=True)
    parser.add_argument("--blind-pullbacks", type=Path, required=True)
    parser.add_argument("--blind-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=202607407)
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--tn-eval-batch-size", type=int, default=1024)
    parser.add_argument("--phi-eval-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_architecture(payload: dict[str, Any]) -> str:
    configuration = payload.get("configuration", {})
    architecture = configuration.get("architecture")
    if architecture in {"compiled-tree", "direct-tn", "residual-phi"}:
        return architecture
    schema = str(payload.get("schema", ""))
    if "compiled-positive-tree" in schema:
        return "compiled-tree"
    if "direct-positive-tn" in schema:
        return "direct-tn"
    if "residual-phi" in schema:
        return "residual-phi"
    raise ValueError(f"cannot infer checkpoint architecture from {schema!r}")


def checkpoint_contract(
    payload: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], str]:
    configuration = dict(payload["configuration"])
    state = payload["state_dict"]
    reference_h = state["reference_h"]
    if hasattr(reference_h, "detach"):
        reference_h = reference_h.detach().cpu().numpy()
    return (
        np.asarray(reference_h, dtype=np.complex128),
        configuration,
        infer_architecture(payload),
    )


def residual_phi_hyperparameters(
    payload: dict[str, Any],
) -> tuple[int, int, float]:
    configuration = payload["configuration"]
    state = payload["state_dict"]
    weight_keys = sorted(
        (
            key
            for key in state
            if key.startswith("network.") and key.endswith(".weight")
        ),
        key=lambda key: int(key.split(".")[1]),
    )
    if len(weight_keys) < 2:
        raise ValueError("residual-phi checkpoint has an invalid network state")
    inferred_width = int(state[weight_keys[0]].shape[0])
    inferred_layers = len(weight_keys) - 1
    return (
        int(configuration.get("phi_hidden_width", inferred_width)),
        int(configuration.get("phi_hidden_layers", inferred_layers)),
        float(configuration.get("phi_potential_scale", 1.0)),
    )


def build_checkpoint_model(
    payload: dict[str, Any],
    *,
    device: torch.device,
) -> torch.nn.Module:
    reference_h, configuration, architecture = checkpoint_contract(payload)
    precision = str(configuration["precision"])
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    power = int(configuration["leaf_count"])
    if architecture == "compiled-tree":
        bond_dimension = configuration.get(
            "edge_dimensions",
            configuration["bond_dimension"],
        )
    else:
        bond_dimension = configuration.get(
            "bond_dimensions",
            configuration["bond_dimension"],
        )
    if not isinstance(bond_dimension, (int, np.integer)):
        bond_dimension = tuple(int(value) for value in bond_dimension)
    normalization = float(configuration["source_normalization"])
    positive_floor = float(configuration["positive_floor"])
    if architecture == "compiled-tree":
        topology_children = configuration.get("topology_children")
        topology = (
            None
            if topology_children is None
            else BinaryTreeTopology(
                leaf_count=power,
                children=tuple(
                    tuple(int(node) for node in pair)
                    for pair in topology_children
                ),
            )
        )
        model = PositiveMultiplicationTreeMetric(
            reference_h,
            leaf_count=power,
            bond_dimension=bond_dimension,
            source_normalization=normalization,
            positive_floor=positive_floor,
            shared_leaf=True,
            topology=topology,
            dtype=dtype,
            device=device,
        )
    elif architecture == "direct-tn":
        model = build_exact_power_lift_tree(
            reference_h,
            source_normalization=normalization,
            power=power,
            bond_dimension=bond_dimension,
            positive_floor=positive_floor,
            precision=precision,
            device=device,
        )
    else:
        hidden_width, hidden_layers, potential_scale = (
            residual_phi_hyperparameters(payload)
        )
        model = ProjectiveResidualPhiMetric(
            reference_h,
            normalization=normalization,
            ambient_dimension=5,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            potential_scale=potential_scale,
            complex_dtype=dtype,
            device=device,
        )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if min(
        args.blind_size,
        args.feature_batch_size,
        args.tn_eval_batch_size,
        args.phi_eval_batch_size,
        args.threads,
    ) <= 0:
        raise ValueError("blind evaluation sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a blind evaluation")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()

    paths = {
        "compiled_tree": args.compiled_tree.expanduser().resolve(),
        "direct_tn_same_updates": args.direct_tn.expanduser().resolve(),
        "residual_phi_same_updates": args.residual_phi.expanduser().resolve(),
    }
    if args.direct_tn_time_matched is not None:
        paths["direct_tn_time_matched"] = (
            args.direct_tn_time_matched.expanduser().resolve()
        )
    if args.residual_phi_time_matched is not None:
        paths["residual_phi_time_matched"] = (
            args.residual_phi_time_matched.expanduser().resolve()
        )
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    contracts = {
        name: checkpoint_contract(payload)
        for name, payload in payloads.items()
    }
    first_configuration = contracts["compiled_tree"][1]
    exponents = np.asarray(first_configuration["exponents"], dtype=np.int64)
    whitening = np.asarray(
        first_configuration["whitening"],
        dtype=np.complex128,
    )
    normalization = float(first_configuration["source_normalization"])
    power = int(first_configuration["leaf_count"])
    precision = str(first_configuration["precision"])
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    reference_h = contracts["compiled_tree"][0]
    expected_architectures = {
        "compiled_tree": "compiled-tree",
        "direct_tn_same_updates": "direct-tn",
        "direct_tn_time_matched": "direct-tn",
        "residual_phi_same_updates": "residual-phi",
        "residual_phi_time_matched": "residual-phi",
    }
    for name, (candidate_h, configuration, architecture) in contracts.items():
        if architecture != expected_architectures[name]:
            raise ValueError(f"checkpoint {name} has architecture {architecture}")
        if (
            int(configuration["leaf_count"]) != power
            or str(configuration["precision"]) != precision
            or not np.isclose(
                float(configuration["source_normalization"]),
                normalization,
            )
            or not np.array_equal(
                np.asarray(configuration["exponents"], dtype=np.int64),
                exponents,
            )
            or not np.allclose(
                np.asarray(configuration["whitening"], dtype=np.complex128),
                whitening,
                rtol=2.0e-7,
                atol=2.0e-8,
            )
            or not np.allclose(
                candidate_h,
                reference_h,
                rtol=2.0e-7,
                atol=2.0e-8,
            )
        ):
            raise ValueError(f"checkpoint {name} does not share the H4 contract")

    arrays, blind_indices = load_pool_arrays(
        args.blind_points,
        args.blind_pullbacks,
        size=args.blind_size,
        seed=args.seed,
    )
    dataset = whiten_dataset(
        make_dataset(
            arrays,
            exponents,
            feature_batch_size=args.feature_batch_size,
            complex_dtype=dtype,
            device=device,
        ),
        whitening,
    )
    projective_dataset = add_projective_inputs(
        dataset,
        arrays,
        dtype=dtype,
        device=device,
    )

    teacher = build_exact_power_lift_tree(
        reference_h,
        source_normalization=normalization,
        power=power,
        bond_dimension=1,
        positive_floor=float(first_configuration["positive_floor"]),
        precision=precision,
        device=device,
    )
    rows: dict[str, Any] = {}
    ratios: dict[str, np.ndarray] = {}
    teacher_statistics, ratios["teacher_h4"], teacher_tail = metric_row(
        teacher,
        dataset,
        chunk_size=args.tn_eval_batch_size,
    )
    rows["teacher_h4"] = {
        "statistics": teacher_statistics,
        "tail": teacher_tail,
        "trainable_real_parameter_count": 0,
    }
    for name, payload in payloads.items():
        model = build_checkpoint_model(payload, device=device)
        architecture = contracts[name][2]
        active_dataset = (
            projective_dataset
            if architecture == "residual-phi"
            else dataset
        )
        chunk_size = (
            args.phi_eval_batch_size
            if architecture == "residual-phi"
            else args.tn_eval_batch_size
        )
        model_statistics, ratios[name], model_tail = metric_row(
            model,
            active_dataset,
            chunk_size=chunk_size,
        )
        rows[name] = {
            "statistics": model_statistics,
            "tail": model_tail,
            "trainable_real_parameter_count": (
                model.trainable_real_parameter_count
            ),
        }

    weights = dataset["weights_numpy"]
    paired: dict[str, Any] = {}
    names = tuple(rows)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            paired[f"{left}_minus_{right}"] = paired_improvement(
                ratios[left],
                ratios[right],
                weights,
            )
    ratios_path = output_dir / "blind_ratios.npz"
    np.savez_compressed(
        ratios_path,
        indices=blind_indices,
        weights=weights,
        **ratios,
    )
    report = {
        "schema": "generic-quintic-h4-architecture-blind-comparison-v1",
        "shared_contract": {
            "teacher": "same_whitened_generic_quintic_H4",
            "target_degree": 4 * power,
            "blind_size": args.blind_size,
            "blind_seed": args.seed,
            "precision": precision,
        },
        "arms": rows,
        "paired_improvement": paired,
        "checkpoint_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "ratios_artifact": str(ratios_path),
        "ratios_artifact_sha256": sha256_file(ratios_path),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
