#!/usr/bin/env python3
"""Run fast, nested, native-E2 rank growth for a generic-quintic tree."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GrowthProposal:
    edge: int
    source_dimension: int
    target_dimension: int
    structural_maximum: int
    new_output_real_parameters: int


@dataclass(frozen=True)
class SharedLeafGrowthProposal:
    source_dimension: int
    target_dimension: int
    new_output_real_parameters: int


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    target_edges: tuple[tuple[int, int], ...]
    proposal: dict[str, Any]


@dataclass(frozen=True)
class TreeShape:
    leaf_count: int
    edge_dimensions: tuple[int, ...]
    topology_children: tuple[tuple[int, int], ...]
    shared_leaf: bool
    shared_leaf_tensor_shape: tuple[int, int, int] | None

    @property
    def root(self) -> int:
        return self.leaf_count + len(self.topology_children) - 1


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def encode(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"cannot JSON encode {type(value).__name__}")

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=encode,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--train-points", type=Path, required=True)
    parser.add_argument("--train-pullbacks", type=Path, required=True)
    parser.add_argument("--selection-points", type=Path, required=True)
    parser.add_argument("--selection-pullbacks", type=Path, required=True)
    parser.add_argument("--confirmation-points", type=Path, required=True)
    parser.add_argument("--confirmation-pullbacks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-indices-file",
        type=Path,
        nargs="*",
        default=(),
    )
    parser.add_argument("--real-parameter-limit", type=int, default=10_000)
    parser.add_argument("--maximum-rounds", type=int, default=8)
    parser.add_argument("--target-sigma", type=float, default=0.01)
    parser.add_argument(
        "--maximum-shared-leaf-dimension",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--shared-leaf-growth",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sweeps-per-round", type=int, default=2)
    parser.add_argument("--epochs-per-block", type=int, default=200)
    parser.add_argument("--selection-eval-every", type=int, default=10)
    parser.add_argument("--early-stopping-evaluations", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--leaf-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--selection-size", type=int, default=2_900)
    parser.add_argument("--base-batch-size", type=int, default=4_096)
    parser.add_argument("--minimum-batch-size", type=int, default=512)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--train-chunk-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--minimum-relative-sigma-gain",
        type=float,
        default=2.0e-3,
    )
    parser.add_argument(
        "--minimum-relative-chi-gain",
        type=float,
        default=2.0e-3,
    )
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.02,
    )
    parser.add_argument("--seed", type=int, default=202607473)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.real_parameter_limit,
        args.maximum_rounds,
        args.target_sigma,
        args.maximum_shared_leaf_dimension,
        args.sweeps_per_round,
        args.epochs_per_block,
        args.selection_eval_every,
        args.early_stopping_evaluations,
        args.learning_rate,
        args.leaf_learning_rate,
        args.train_size,
        args.selection_size,
        args.base_batch_size,
        args.minimum_batch_size,
        args.gradient_clip_norm,
        args.train_chunk_size,
        args.feature_batch_size,
        args.eval_batch_size,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("workflow sizes and optimization values must be positive")
    if not 0 <= args.minimum_relative_sigma_gain < 1:
        raise ValueError("minimum sigma gain must lie in [0, 1)")
    if not 0 <= args.minimum_relative_chi_gain < 1:
        raise ValueError("minimum chi gain must lie in [0, 1)")
    if args.maximum_selection_tail_relative_degradation < 0:
        raise ValueError("tail degradation allowance must be nonnegative")


def inspect_tree_shape(checkpoint: Path) -> TreeShape:
    code = "\n".join(
        (
            "import json, sys, torch",
            "payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)",
            "configuration = payload['configuration']",
            "architecture = configuration.get('architecture')",
            "assert architecture == 'compiled-tree', architecture",
            "dimensions = configuration.get('edge_dimensions', configuration['bond_dimension'])",
            "leaf_count = int(configuration['leaf_count'])",
            "if isinstance(dimensions, int):",
            "    dimensions = [dimensions] * (2 * leaf_count - 2)",
            "children = configuration.get('topology_children')",
            "assert children is not None",
            "shared_leaf = bool(configuration.get('shared_leaf', False))",
            "leaf_shape = None",
            "if shared_leaf:",
            "    leaf_shape = list(payload['state_dict']['shared_leaf_tensor'].shape)",
            "print(json.dumps({'leaf_count': leaf_count, 'edge_dimensions': list(dimensions), 'topology_children': children, 'shared_leaf': shared_leaf, 'shared_leaf_tensor_shape': leaf_shape}))",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(checkpoint)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return TreeShape(
        leaf_count=int(payload["leaf_count"]),
        edge_dimensions=tuple(
            int(value) for value in payload["edge_dimensions"]
        ),
        topology_children=tuple(
            (int(left), int(right))
            for left, right in payload["topology_children"]
        ),
        shared_leaf=bool(payload["shared_leaf"]),
        shared_leaf_tensor_shape=(
            None
            if payload["shared_leaf_tensor_shape"] is None
            else tuple(
                int(value)
                for value in payload["shared_leaf_tensor_shape"]
            )
        ),
    )


def internal_growth_proposals(
    tree: TreeShape,
    *,
    real_parameter_limit: int,
) -> list[GrowthProposal]:
    proposals = []
    for edge in range(tree.leaf_count, tree.root):
        left, right = tree.topology_children[edge - tree.leaf_count]
        row_size = tree.edge_dimensions[left] * tree.edge_dimensions[right]
        structural_maximum = row_size
        source = tree.edge_dimensions[edge]
        if source >= structural_maximum:
            continue
        maximum_increment = real_parameter_limit // (2 * row_size)
        if maximum_increment <= 0:
            continue
        target = min(structural_maximum, source + maximum_increment)
        proposals.append(
            GrowthProposal(
                edge=edge,
                source_dimension=source,
                target_dimension=target,
                structural_maximum=structural_maximum,
                new_output_real_parameters=2 * (target - source) * row_size,
            )
        )
    return proposals


def shared_leaf_growth_proposal(
    tree: TreeShape,
    *,
    real_parameter_limit: int,
    maximum_dimension: int,
) -> SharedLeafGrowthProposal | None:
    if not tree.shared_leaf or tree.shared_leaf_tensor_shape is None:
        return None
    source_dimensions = set(tree.edge_dimensions[: tree.leaf_count])
    if len(source_dimensions) != 1:
        raise ValueError("shared leaf edge dimensions are inconsistent")
    source = next(iter(source_dimensions))
    if source >= maximum_dimension:
        return None
    _, output_dimension, section_count = tree.shared_leaf_tensor_shape
    row_real_parameters = 2 * output_dimension * section_count
    if row_real_parameters > real_parameter_limit:
        return None
    return SharedLeafGrowthProposal(
        source_dimension=source,
        target_dimension=source + 1,
        new_output_real_parameters=row_real_parameters,
    )


def adaptive_batch_size(
    *,
    base_batch_size: int,
    minimum_batch_size: int,
    largest_internal_dimension: int,
) -> int:
    raw = base_batch_size * 25.0 / max(largest_internal_dimension, 25)
    power = 2 ** int(round(math.log2(max(raw, minimum_batch_size))))
    return max(minimum_batch_size, min(base_batch_size, power))


def relative_gain(start: float, final: float) -> float:
    return (start - final) / start


def run_candidate(
    *,
    args: argparse.Namespace,
    trainer: Path,
    current_checkpoint: Path,
    output_dir: Path,
    status_path: Path,
    round_index: int,
    candidate_index: int,
    tree: TreeShape,
    candidate: CandidateSpec,
) -> dict[str, Any]:
    candidate_dir = output_dir / f"round_{round_index:02d}_{candidate.label}"
    target_dimensions = list(tree.edge_dimensions)
    for edge, dimension in candidate.target_edges:
        target_dimensions[edge] = dimension
    largest_dimension = max(target_dimensions[tree.leaf_count :])
    batch_size = adaptive_batch_size(
        base_batch_size=args.base_batch_size,
        minimum_batch_size=args.minimum_batch_size,
        largest_internal_dimension=largest_dimension,
    )
    command = [
        sys.executable,
        str(trainer),
        "--initial-checkpoint",
        str(current_checkpoint),
        "--train-points",
        str(args.train_points),
        "--train-pullbacks",
        str(args.train_pullbacks),
        "--selection-points",
        str(args.selection_points),
        "--selection-pullbacks",
        str(args.selection_pullbacks),
        "--confirmation-points",
        str(args.confirmation_points),
        "--confirmation-pullbacks",
        str(args.confirmation_pullbacks),
        "--output-dir",
        str(candidate_dir),
        "--real-parameter-limit",
        str(args.real_parameter_limit),
    ]
    for edge, dimension in candidate.target_edges:
        command.extend(("--target-edge", f"{edge}:{dimension}"))
    command.extend(
        (
            "--rank-activation-scale",
            "1",
            "--orthogonalize-new-outputs",
            "--expansion-only",
            "--development-only",
            "--defer-block-acceptance",
            "--sweeps",
            str(args.sweeps_per_round),
            "--epochs-per-block",
            str(args.epochs_per_block),
            "--selection-eval-every",
            str(args.selection_eval_every),
            "--early-stopping-evaluations",
            str(args.early_stopping_evaluations),
            "--internal-learning-rate",
            str(args.learning_rate),
            "--leaf-learning-rate",
            str(args.leaf_learning_rate),
            "--train-size",
            str(args.train_size),
            "--selection-size",
            str(args.selection_size),
            "--confirmation-size",
            "1",
            "--stochastic-batch-size",
            str(batch_size),
            "--gradient-clip-norm",
            str(args.gradient_clip_norm),
            "--backward-mode",
            "graph",
            "--maximum-selection-tail-relative-degradation",
            str(args.maximum_selection_tail_relative_degradation),
            "--train-chunk-size",
            str(min(args.train_chunk_size, batch_size)),
            "--feature-batch-size",
            str(args.feature_batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--device",
            args.device,
            "--seed",
            str(args.seed + 100 * round_index),
        )
    )
    if args.exclude_indices_file:
        command.append("--exclude-indices-file")
        command.extend(str(path) for path in args.exclude_indices_file)
    write_json(
        status_path,
        {
            "state": "running",
            "phase": "candidate_training",
            "round": round_index,
            "proposal": candidate.proposal,
            "batch_size": batch_size,
        },
    )
    subprocess.run(command, cwd=ROOT, check=True)
    report_path = candidate_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    initial = report["initial_selection"]
    final = report["final_selection"]
    sigma_gain = relative_gain(
        initial["sigma_official_formula"],
        final["sigma_official_formula"],
    )
    chi_gain = relative_gain(
        initial["weighted_rms_abs_residual"],
        final["weighted_rms_abs_residual"],
    )
    passes = bool(
        report["accepted_blocks"] > 0
        and sigma_gain >= args.minimum_relative_sigma_gain
        and chi_gain >= args.minimum_relative_chi_gain
    )
    return {
        "proposal": candidate.proposal,
        "target_edges": candidate.target_edges,
        "batch_size": batch_size,
        "sigma_gain": sigma_gain,
        "chi_gain": chi_gain,
        "score": min(sigma_gain, chi_gain),
        "passes": passes,
        "report": str(report_path),
        "checkpoint": report["checkpoint"],
        "initial_selection": initial,
        "final_selection": final,
        "wall_seconds": report["wall_seconds"],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite an adaptive workflow")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})
    started = time.perf_counter()

    try:
        current_checkpoint = args.initial_checkpoint.expanduser().resolve()
        history: list[dict[str, Any]] = []
        stop_reason = "maximum_rounds"
        trainer = ROOT / "scripts/train_generic_quintic_adaptive_direct_blocks.py"

        for round_index in range(1, args.maximum_rounds + 1):
            tree = inspect_tree_shape(current_checkpoint)
            source_dimensions = tree.edge_dimensions
            internal_proposals = internal_growth_proposals(
                tree,
                real_parameter_limit=args.real_parameter_limit,
            )
            internal_candidates = [
                CandidateSpec(
                    label=(
                        f"edge_{proposal.edge}_to_"
                        f"{proposal.target_dimension}"
                    ),
                    target_edges=(
                        (proposal.edge, proposal.target_dimension),
                    ),
                    proposal={
                        "kind": "internal_edge",
                        **asdict(proposal),
                    },
                )
                for proposal in internal_proposals
            ]
            candidate_rows = [
                run_candidate(
                    args=args,
                    trainer=trainer,
                    current_checkpoint=current_checkpoint,
                    output_dir=output_dir,
                    status_path=status_path,
                    round_index=round_index,
                    candidate_index=candidate_index,
                    tree=tree,
                    candidate=candidate,
                )
                for candidate_index, candidate in enumerate(
                    internal_candidates
                )
            ]
            passing = [row for row in candidate_rows if row["passes"]]
            leaf_attempted = False
            leaf_proposal = None
            if (
                not passing
                and args.shared_leaf_growth
            ):
                leaf_proposal = shared_leaf_growth_proposal(
                    tree,
                    real_parameter_limit=args.real_parameter_limit,
                    maximum_dimension=args.maximum_shared_leaf_dimension,
                )
                if leaf_proposal is not None:
                    leaf_attempted = True
                    leaf_candidate = CandidateSpec(
                        label=(
                            "shared_leaf_to_"
                            f"{leaf_proposal.target_dimension}"
                        ),
                        target_edges=tuple(
                            (edge, leaf_proposal.target_dimension)
                            for edge in range(tree.leaf_count)
                        ),
                        proposal={
                            "kind": "shared_leaf",
                            **asdict(leaf_proposal),
                        },
                    )
                    leaf_row = run_candidate(
                        args=args,
                        trainer=trainer,
                        current_checkpoint=current_checkpoint,
                        output_dir=output_dir,
                        status_path=status_path,
                        round_index=round_index,
                        candidate_index=len(candidate_rows),
                        tree=tree,
                        candidate=leaf_candidate,
                    )
                    candidate_rows.append(leaf_row)
                    if leaf_row["passes"]:
                        passing = [leaf_row]
            if not passing:
                history.append(
                    {
                        "round": round_index,
                        "source_edge_dimensions": source_dimensions,
                        "candidates": candidate_rows,
                        "accepted": None,
                    }
                )
                if leaf_attempted:
                    stop_reason = "internal_and_leaf_gain_below_threshold"
                elif not internal_proposals and leaf_proposal is None:
                    stop_reason = "adaptive_structural_capacity_exhausted"
                else:
                    stop_reason = "marginal_gain_below_threshold"
                break
            accepted = max(passing, key=lambda row: row["score"])
            current_checkpoint = Path(accepted["checkpoint"]).resolve()
            history.append(
                {
                    "round": round_index,
                    "source_edge_dimensions": source_dimensions,
                    "candidates": candidate_rows,
                    "accepted": accepted,
                }
            )
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "round_complete",
                    "round": round_index,
                    "accepted": accepted,
                    "current_checkpoint": str(current_checkpoint),
                },
            )
            if (
                accepted["final_selection"]["sigma_official_formula"]
                <= args.target_sigma
            ):
                stop_reason = "target_sigma_reached_on_development"
                break

        best_checkpoint = output_dir / "best_development_checkpoint.pt"
        shutil.copy2(current_checkpoint, best_checkpoint)
        report = {
            "schema": "generic-quintic-fast-adaptive-workflow-v1",
            "teacher_role": "absent",
            "configuration": {
                **vars(args),
                "initial_checkpoint": str(
                    args.initial_checkpoint.expanduser().resolve()
                ),
                "output_dir": str(output_dir),
            },
            "history": history,
            "stop_reason": stop_reason,
            "accepted_rounds": sum(
                row["accepted"] is not None for row in history
            ),
            "best_checkpoint": str(best_checkpoint),
            "final_confirmation": "not_opened",
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "workflow_report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "stop_reason": stop_reason,
                "accepted_rounds": report["accepted_rounds"],
                "best_checkpoint": str(best_checkpoint),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(status_path.read_text(encoding="utf-8"), flush=True)
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
