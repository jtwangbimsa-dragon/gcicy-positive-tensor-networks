#!/usr/bin/env python3
"""Initialize an incoherent positive two-atom quintic TN control."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import PositiveTensorNetworkPositiveSumMetric  # noqa: E402
from scripts.audit_quintic_tn_scaling_preflight import (  # noqa: E402
    build_model,
    cast_artifact_precision,
    load_numpy_split,
    make_tensor_split,
    resolve_inputs,
)
from scripts.train_quintic_full_h_same_points import sha256_file, write_json  # noqa: E402
from scripts.train_quintic_tn_coherent_gate import validate_branch_payloads  # noqa: E402
from scripts.train_quintic_tn_one_site_lm import (  # noqa: E402
    direct_empirical_statistics,
    random_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--atom-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--blind-reference-run-dir", type=Path)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--initial-gate-real", type=float, default=0.05)
    parser.add_argument("--initial-gate-imaginary", type=float, default=0.0)
    parser.add_argument("--validation-size", type=int, default=20000)
    parser.add_argument("--blind-limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--canonical-center", type=int)
    parser.add_argument("--seed", type=int, default=202607219)
    parser.add_argument("--precision", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-blind", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(args.validation_size, args.eval_batch_size) <= 0 or args.blind_limit < 0:
        raise ValueError("sample and batch sizes must be valid")
    gate = complex(args.initial_gate_real, args.initial_gate_imaginary)
    if not np.isfinite(abs(gate)) or abs(gate) == 0:
        raise ValueError("positive-sum initialization requires a finite nonzero gate")


def statistics_with_gate(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    gate: complex,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    saved = model.new_gates.detach().clone()
    model.set_new_gates_((gate,))
    result = direct_empirical_statistics(model, dataset, chunk_size=chunk_size)
    model.set_new_gates_(saved)
    return result


def tail_summary(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "q999": float(statistics["abs_residual_weighted_quantiles"]["q0.9990"]),
        "cvar99": float(
            statistics["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum": float(
            statistics["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})
    try:
        paths = resolve_inputs(args)
        atom_path = args.atom_model.expanduser().resolve()
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        leading_payload = torch.load(
            paths["model"], map_location="cpu", weights_only=False
        )
        atom_payload = torch.load(atom_path, map_location="cpu", weights_only=False)
        validate_branch_payloads(leading_payload, atom_payload)
        leading_payload = cast_artifact_precision(leading_payload, args.precision)
        atom_payload = cast_artifact_precision(atom_payload, args.precision)
        leading = build_model(leading_payload, device)
        atom = build_model(atom_payload, device)
        center = (
            leading.site_count // 2
            if args.canonical_center is None
            else args.canonical_center
        )
        if center < 0 or center >= leading.site_count:
            raise ValueError("canonical center is outside the branch chain")
        leading.requires_grad_(False)
        atom.requires_grad_(False)
        leading.mixed_canonicalize_coefficient_cores_(center)
        leading_scale = leading.normalize_coefficient_chain_scale_(site_index=center)
        atom.mixed_canonicalize_coefficient_cores_(center)
        atom_scale = atom.normalize_coefficient_chain_scale_(site_index=center)
        floor = float(leading.positive_floor)
        initial_gate = complex(args.initial_gate_real, args.initial_gate_imaginary)
        model = PositiveTensorNetworkPositiveSumMetric(
            (leading, atom),
            positive_floor=floor,
            initial_new_gates=(initial_gate,),
        )

        validation_arrays, validation_indices = random_subset(
            load_numpy_split(paths, "validation", 0),
            args.validation_size,
            seed=args.seed,
        )
        source_degree = int(leading_payload.get("source_degree", 1))
        validation = make_tensor_split(
            *validation_arrays,
            source_degree=source_degree,
            precision=args.precision,
            device=device,
        )
        indices_path = output_dir / "initialization_indices.npz"
        np.savez_compressed(indices_path, validation_indices=validation_indices)
        validation_leading = statistics_with_gate(
            model, validation, 0.0j, chunk_size=args.eval_batch_size
        )
        validation_initial = direct_empirical_statistics(
            model, validation, chunk_size=args.eval_batch_size
        )

        blind_leading = None
        blind_initial = None
        if not args.skip_blind:
            blind_arrays = load_numpy_split(paths, "blind", args.blind_limit)
            blind = make_tensor_split(
                *blind_arrays,
                source_degree=source_degree,
                precision=args.precision,
                device=device,
            )
            blind_leading = statistics_with_gate(
                model, blind, 0.0j, chunk_size=args.eval_batch_size
            )
            blind_initial = direct_empirical_statistics(
                model, blind, chunk_size=args.eval_batch_size
            )

        artifact = {
            "schema": "quintic-positive-tensor-network-positive-sum-v1",
            "geometry": leading_payload.get("geometry"),
            "source_degree": source_degree,
            "total_degree": int(leading_payload["total_degree"]),
            "site_count": int(leading_payload["site_count"]),
            "target_normalization": float(leading_payload["target_normalization"]),
            "positive_floor": floor,
            "precision": args.precision,
            "fixed_log_kappa": float(leading_payload["fixed_log_kappa"]),
            "branch_states": [
                {
                    key: value.detach().cpu()
                    for key, value in branch.state_dict().items()
                }
                for branch in model.branches
            ],
            "branch_metadata": [
                {
                    "source_model": str(path),
                    "source_model_sha256": sha256_file(path),
                    "bond_dimension": int(payload["bond_dimension"]),
                    "architecture": payload["architecture"],
                    "physical_dictionary_rank": payload.get("physical_dictionary_rank"),
                    "output_dimension": payload.get("output_dimension", 5),
                }
                for path, payload in (
                    (paths["model"], leading_payload),
                    (atom_path, atom_payload),
                )
            ],
            "new_gates": model.new_gates.detach().cpu(),
            "leading_scale": leading_scale,
            "atom_scale": atom_scale,
            "validation_tail_reference": {
                "validation_indices": np.asarray(validation_indices, dtype=np.int64),
                "leading": tail_summary(validation_leading),
                "initial": tail_summary(validation_initial),
                "selection_seed": args.seed,
            },
        }
        model_path = output_dir / "coherent_model.pt"
        torch.save(artifact, model_path)
        parameter_count = sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
        )
        report = {
            "schema": "quintic-tn-positive-sum-initialization-v1",
            "scientific_scope": {
                "purpose": "incoherent positive-branch control",
                "formula": "F=||Psi0||^2+|gamma|^2||Psi1||^2+epsilon*Fref",
                "zero_gate_nestedness": True,
            },
            "configuration": {
                **{
                    key: value
                    for key, value in vars(args).items()
                    if not isinstance(value, Path)
                },
                "run_dir": str(args.run_dir),
                "atom_model": str(atom_path),
                "output_dir": str(output_dir),
                "device": str(device),
                "canonical_center": center,
            },
            "architecture": {
                "branch_count": 2,
                "bond_dimension_per_branch": int(leading_payload["bond_dimension"]),
                "stored_real_parameters_excluding_fixed_dictionaries": parameter_count,
                "initial_gate": {
                    "real": float(torch.real(model.new_gates[0])),
                    "imaginary": float(torch.imag(model.new_gates[0])),
                    "magnitude": float(torch.abs(model.new_gates[0])),
                    "positive_branch_weight": float(torch.abs(model.new_gates[0]) ** 2),
                },
            },
            "source": {
                "leading_model_sha256": sha256_file(paths["model"]),
                "atom_model_sha256": sha256_file(atom_path),
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
            },
            "validation": {
                "leading": validation_leading,
                "initial": validation_initial,
            },
            "benchmark": {
                "leading": blind_leading,
                "initial": blind_initial,
            },
            "artifacts": {
                "model": str(model_path),
                "model_sha256": sha256_file(model_path),
            },
            "timing_seconds": time.perf_counter() - started,
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
            },
        )
        print(
            json.dumps(
                {
                    "validation_leading_sigma": validation_leading[
                        "sigma_official_formula"
                    ],
                    "validation_initial_sigma": validation_initial[
                        "sigma_official_formula"
                    ],
                    "report": str(report_path),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
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
