#!/usr/bin/env python3
"""Build a checked, common-schema summary of the three gCICY TN results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


CASES = (
    {
        "type": "(1,1)",
        "geometry": "X11_m3",
        "configuration": "P4 x P1 [3,2; 1,1 | -1,1] (direct Hirzebruch form)",
        "model": "outputs/pipeline/type11_tensor_network_scaling_k6_d5_production_20260717/model_d5_k6_production.pt",
        "training": "outputs/pipeline/type11_tensor_network_scaling_k6_d5_production_20260717/model_d5_k6_production_summary.json",
        "blind": "outputs/pipeline/type11_tensor_network_scaling_k6_d5_production_20260717/model_d5_k6_production_blind.json",
        "source": "outputs/pipeline/type11_tensor_network_scaling_k2_20260717/nu_balanced_k2_n32768_c128.npz",
        "certificate_degree": [2, 2, 0],
        "fibre_size": 4,
        "dictionary_mode": "complete local operator space",
    },
    {
        "type": "(2,1)",
        "geometry": "X21",
        "configuration": "P5 x P1 [1,2|3; 1,2|-1]",
        "model": "outputs/pipeline/type21_tensor_network_q22_degree_ladder_20260719/q22_k8_teacher_free.pt",
        "training": "outputs/pipeline/type21_tensor_network_q22_degree_ladder_20260719/q22_k8_teacher_free_summary.json",
        "blind": "outputs/pipeline/type21_tensor_network_q22_degree_ladder_20260719/q22_k8_teacher_free_blind83523_n65532.json",
        "source": "outputs/pipeline/type21_k1_energy_highstat_pointb64_20260718/k1_energy_highstat.npz",
        "certificate_degree": [1, 1],
        "fibre_size": 6,
        "dictionary_mode": "intentionally truncated learned operator dictionary",
    },
    {
        "type": "(2,2)",
        "geometry": "X22",
        "configuration": "P1 x P1 x P5 generalized complete intersection",
        "model": "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/active_q60_k6_degree_control.pt",
        "training": "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/active_q60_k6_degree_control_summary.json",
        "blind": "outputs/pipeline/type22_tensor_network_symmetry_breaking_20260719/active_q60_k6_degree_control_blind83713_n65532.json",
        "source": "outputs/gcicy_generic_global_h_metric_weighted.npz",
        "certificate_degree": [1, 1, 1],
        "fibre_size": 3,
        "dictionary_mode": "intentionally truncated learned operator dictionary",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs/gcicy_tn_publication_summary_20260719.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "docs/gcicy_tn_publication_summary_20260719.md",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def certificate_entry(
    certificates: dict[str, Any], geometry: str, degree: list[int]
) -> dict[str, Any]:
    geometry_entry = next(
        entry for entry in certificates["certificates"] if entry["geometry"] == geometry
    )
    return next(entry for entry in geometry_entry["degrees"] if entry["degree"] == degree)


def dictionary_audit(payload: dict[str, Any], source_dimension: int) -> dict[str, Any]:
    state = payload["state_dict"]
    rank = payload.get("physical_dictionary_rank")
    complete_rank = source_dimension**2
    if rank is None:
        core_shapes = [
            tuple(value.shape)
            for key, value in state.items()
            if key.startswith("cores.")
        ]
        if not core_shapes or any(shape[-2:] != (source_dimension, source_dimension) for shape in core_shapes):
            raise RuntimeError("dense local cores do not span the full source operator shape")
        return {
            "architecture": "dense_local_operator",
            "rank": complete_rank,
            "complete_rank": complete_rank,
            "fraction_of_complete_operator_space": 1.0,
            "is_complete": True,
            "row_gram_maximum_error": 0.0,
        }

    dictionary = state["physical_dictionary"].detach().cpu().numpy()
    if dictionary.shape != (rank, source_dimension, source_dimension):
        raise RuntimeError("stored physical dictionary has an unexpected shape")
    flat = dictionary.reshape(rank, -1)
    gram = flat @ np.conj(flat.T)
    gram_error = float(np.max(np.abs(gram - np.eye(rank))))
    if gram_error > 1.0e-10:
        raise RuntimeError(f"physical dictionary gauge check failed: {gram_error}")
    return {
        "architecture": payload["architecture"],
        "rank": int(rank),
        "complete_rank": complete_rank,
        "fraction_of_complete_operator_space": float(rank / complete_rank),
        "is_complete": bool(rank == complete_rank),
        "row_gram_maximum_error": gram_error,
    }


def parameter_count(payload: dict[str, Any], source_dimension: int) -> int:
    sites = int(payload["site_count"])
    bond = int(payload["bond_dimension"])
    rank = payload.get("physical_dictionary_rank")
    transfer_blocks = 2 * bond + max(sites - 2, 0) * bond**2
    if rank is None:
        return 2 * source_dimension**2 * transfer_blocks
    return 2 * int(rank) * (source_dimension**2 + transfer_blocks)


def build_case(
    specification: dict[str, Any], certificates: dict[str, Any]
) -> dict[str, Any]:
    paths = {
        key: ROOT / specification[key]
        for key in ("model", "training", "blind", "source")
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    training = read_json(paths["training"])
    blind = read_json(paths["blind"])
    payload = torch.load(paths["model"], map_location="cpu", weights_only=False)
    source = np.load(paths["source"], allow_pickle=False)
    section_exponents = np.asarray(payload["source_section_exponents"])
    source_dimension = int(len(section_exponents))
    source_artifact_dimension = int(len(source["global_section_exponents"]))
    relation_error = float(source["basis_relation_error"])
    certificate = certificate_entry(
        certificates,
        specification["geometry"],
        specification["certificate_degree"],
    )
    if not (
        source_dimension
        == source_artifact_dimension
        == certificate["riemann_roch_target"]
        == certificate["finite_field_rank"]
    ):
        raise RuntimeError(f"section-space completeness mismatch for {specification['type']}")
    if relation_error > 1.0e-12:
        raise RuntimeError("source section relation error exceeds tolerance")

    dictionary = dictionary_audit(payload, source_dimension)
    expected_parameters = parameter_count(payload, source_dimension)
    reported_parameters = int(blind["trainable_real_parameter_count"])
    if expected_parameters != reported_parameters:
        raise RuntimeError(
            f"parameter formula mismatch: {expected_parameters} != {reported_parameters}"
        )

    model_hash = sha256_file(paths["model"])
    if model_hash != blind["model_sha256"] or model_hash != training["model_sha256"]:
        raise RuntimeError("model hash disagreement across training and blind reports")
    if payload["source_artifact_sha256"] != sha256_file(paths["source"]):
        raise RuntimeError("source artifact hash disagreement")

    fibre_size = int(specification["fibre_size"])
    for split in (training["train"], training["validation"]):
        if split["points"] % fibre_size:
            raise RuntimeError("training or validation split cuts a sampling fibre")
        if any(shard["points"] % fibre_size for shard in split["shards"]):
            raise RuntimeError("a training or validation shard cuts a sampling fibre")
    if blind["points"] % fibre_size or any(
        shard["points"] % fibre_size for shard in blind["point_shards"]
    ):
        raise RuntimeError("blind evidence cuts a sampling fibre")
    seeds = (training["train"]["seed"], training["validation"]["seed"], blind["seed"])
    if len(set(seeds)) != 3:
        raise RuntimeError("train, validation, and blind seeds must be distinct")

    metrics = blind["metrics"]["compressed_ma_errors"]
    if metrics["normalized_ratio_above_3_point_count"] != 0:
        raise RuntimeError("accepted gCICY artifact has a sampled ratio above three")
    minimum_eigenvalue = float(blind["metrics"]["minimum_metric_eigenvalue"])
    if minimum_eigenvalue <= 0:
        raise RuntimeError("accepted gCICY artifact has a nonpositive sampled metric")

    return {
        "type": specification["type"],
        "geometry": specification["geometry"],
        "configuration": specification["configuration"],
        "target_multidegree": payload["target_degree"],
        "site_count": int(payload["site_count"]),
        "bond_dimension": int(payload["bond_dimension"]),
        "trainable_real_parameter_count": reported_parameters,
        "parameter_formula_verified": True,
        "training_mode": training.get("training_mode", "energy-guided geometric"),
        "best_epoch": int(training["best_epoch"]),
        "runtime_seconds": float(training["runtime_seconds"]),
        "sample_counts": {
            "train": int(training["train"]["points"]),
            "validation": int(training["validation"]["points"]),
            "blind": int(blind["points"]),
        },
        "seeds": {
            "train": int(seeds[0]),
            "validation": int(seeds[1]),
            "blind": int(seeds[2]),
        },
        "complete_fibre_sampling": {
            "verified_from_split_and_shard_divisibility": True,
            "roots_per_fibre": fibre_size,
            "blind_fibres": int(blind["points"] // fibre_size),
        },
        "source_section_space": {
            "degree": specification["certificate_degree"],
            "dimension": source_dimension,
            "riemann_roch_target": certificate["riemann_roch_target"],
            "finite_field_rank": certificate["finite_field_rank"],
            "basis_relation_error": relation_error,
            "complete": True,
            "certificate_geometry": specification["geometry"],
        },
        "physical_dictionary": {
            **dictionary,
            "selection": specification["dictionary_mode"],
        },
        "blind_metrics": {
            "sigma": float(metrics["sigma"]),
            "chi": float(metrics["sqrt_squared_energy"]),
            "absolute_log_ratio_q999": float(metrics["positive_log_ratio_q999"]),
            "absolute_log_ratio_cvar_1pct": float(
                metrics["positive_log_ratio_cvar_1pct"]
            ),
            "normalized_ratio_min": float(metrics["normalized_ratio_min"]),
            "normalized_ratio_max": float(metrics["normalized_ratio_max"]),
            "weighted_mass_ratio_above_3": float(
                metrics["normalized_ratio_above_3_weighted_mass"]
            ),
            "minimum_metric_eigenvalue": minimum_eigenvalue,
            "importance_effective_sample_size": float(
                metrics["importance_effective_sample_size"]
            ),
        },
        "structural_guarantees": {
            "global_kahler": True,
            "global_positive_definite": True,
            "fixed_kahler_class": True,
            "blind_tail_claim_is_statistical_not_sup_norm": True,
        },
        "evidence": {
            "model": specification["model"],
            "model_sha256": model_hash,
            "training_report": specification["training"],
            "training_report_sha256": sha256_file(paths["training"]),
            "blind_report": specification["blind"],
            "blind_report_sha256": sha256_file(paths["blind"]),
            "source_artifact": specification["source"],
            "source_artifact_sha256": sha256_file(paths["source"]),
        },
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# gCICY tensor-network publication summary",
        "",
        "All rows use independent train, validation, and blind seeds. Section-space",
        "completeness is checked against the finite-field/Riemann--Roch certificates;",
        "tail statements remain finite-sample statements.",
        "",
        "| type | target degree | q / d^2 | parameters | train / val / blind | sigma | chi | q99.9 | CVaR1% | min / max r | min eig |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        metric = case["blind_metrics"]
        dictionary = case["physical_dictionary"]
        counts = case["sample_counts"]
        lines.append(
            "| {type} | {degree} | {rank}/{complete} | {parameters:,} | "
            "{train:,}/{validation:,}/{blind:,} | {sigma:.5f} | {chi:.5f} | "
            "{q999:.5f} | {cvar:.5f} | {minimum:.3f}/{maximum:.3f} | {mineig:.5f} |".format(
                type=case["type"],
                degree="x".join(str(value) for value in case["target_multidegree"]),
                rank=dictionary["rank"],
                complete=dictionary["complete_rank"],
                parameters=case["trainable_real_parameter_count"],
                train=counts["train"],
                validation=counts["validation"],
                blind=counts["blind"],
                sigma=metric["sigma"],
                chi=metric["chi"],
                q999=metric["absolute_log_ratio_q999"],
                cvar=metric["absolute_log_ratio_cvar_1pct"],
                minimum=metric["normalized_ratio_min"],
                maximum=metric["normalized_ratio_max"],
                mineig=metric["minimum_metric_eigenvalue"],
            )
        )
    lines.extend(
        [
            "",
            "The `(1,1)` arm uses the complete local `45^2` operator space. The",
            "`(2,1)` and `(2,2)` arms use intentionally truncated learned dictionaries",
            "inside complete 11- and 17-section source spaces; their ranks are reported",
            "explicitly and are not described as complete dictionaries.",
            "",
            "Every stored metric is structurally positive and Kahler. The sampled minimum",
            "eigenvalues and zero observed mass above `r=3` are independent blind checks,",
            "not proofs of a global residual supremum bound.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    certificates_path = ROOT / "outputs/section_restriction_rank_certificates.json"
    certificates = read_json(certificates_path)
    if not certificates["all_certificates_passed"]:
        raise RuntimeError("section restriction certificate suite is not passing")
    cases = [build_case(case, certificates) for case in CASES]
    payload = {
        "schema": "gcicy-positive-tensor-network-publication-summary-v1",
        "scientific_scope": {
            "claim": "finite-sample transfer of a positive linear-in-site-count tensor-network metric family across three sequential gCICY types",
            "claim_limit": "not a global sup-norm certificate and not an asymptotic approximation theorem",
        },
        "all_checks_passed": True,
        "section_certificate": {
            "path": "outputs/section_restriction_rank_certificates.json",
            "sha256": sha256_file(certificates_path),
        },
        "cases": cases,
    }
    write_json(args.output_json.resolve(), payload)
    output_markdown = args.output_markdown.resolve()
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
