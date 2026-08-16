"""Generic positive-Hermitian global-section metric trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .adapter import (
    GCICYAdapter,
    HMetricArtifact,
    H_POSITIVE_RELATIVE_FLOOR,
    positive_hermitian_projection,
)
from .audit import normalized_volume_ratios, standard_errors
from .active_set import file_sha256, load_active_point_pool
from .common_point_pool import CommonPointPool, load_common_point_pool
from .parallel_sampling import sample_points_parallel
from .risk import (
    smooth_upper_log_ratio_excess_torch,
    weighted_cvar_numpy,
    weighted_cvar_selected_linearization_torch,
    weighted_cvar_torch,
    weighted_log_mean_exp_torch,
)


def _relative_spd_factor_log_statistics(
    reference_factor: np.ndarray,
    candidate_factor: np.ndarray,
) -> dict[str, float]:
    """Measure a centered SPD displacement without forming ill-conditioned H."""

    reference = np.asarray(reference_factor, dtype=np.complex128)
    candidate = np.asarray(candidate_factor, dtype=np.complex128)
    if (
        reference.ndim != 2
        or reference.shape[0] != reference.shape[1]
        or candidate.shape != reference.shape
    ):
        raise ValueError("SPD diagnostic factors must be equally sized square matrices")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("SPD diagnostic factors must be finite")
    relative_factor = np.linalg.solve(reference, candidate)
    singular_values = np.linalg.svd(relative_factor, compute_uv=False)
    if float(singular_values[-1]) <= 0:
        raise FloatingPointError("SPD diagnostic factor is singular")
    log_eigenvalues = 2.0 * np.log(singular_values)
    centered = log_eigenvalues - np.mean(log_eigenvalues)
    return {
        "log_eigenvalue_span": float(np.ptp(centered)),
        "max_abs_centered_log_eigenvalue": float(np.max(np.abs(centered))),
    }


@dataclass(frozen=True)
class TrainingRequest:
    model_seed: int
    exact_model: bool
    degree: tuple[int, ...]
    device: str = "auto"
    precision: str = "complex64"
    importance_weighted: bool = True
    basis_points: int = 512
    train_points: int = 2048
    validation_points: int = 1024
    check_points: int = 512
    export_points: int = 1024
    epochs: int = 1200
    learning_rate: float = 2e-3
    eval_every: int = 100
    patience_evaluations: int = 6
    validation_weight: float = 0.75
    drift_weight: float = 1e-5
    centered_log_variance_loss_weight: float = 1.0
    metric_barrier_weight: float = 200.0
    relative_log_spectrum_loss_weight: float = 0.0
    h_parameterization: str = "full_cholesky"
    low_rank: int = 16
    low_rank_epsilon: float = 1e-4
    low_rank_factor_weight: float = 0.0
    low_rank_seed: int = 3204
    sigma_loss_weight: float = 0.0
    volume_ratio_l2_loss_weight: float = 0.0
    groupwise_objective: bool = False
    group_normalization_mode: str = "per_group"
    group_optimizer_step_mode: str = "accumulate_all_groups"
    group_shuffle_seed: int | None = None
    clusters_per_optimizer_step: int = 0
    cluster_minibatch_loss_reduction: str = "mean"
    cluster_minibatch_uniform_loss: bool = False
    cluster_tail_replay_batches_per_epoch: int = 0
    points_per_optimizer_step: int = 0
    point_minibatch_loss_reduction: str = "mean"
    point_minibatch_uniform_loss: bool = False
    point_upper_tail_replay_points_per_fraction: int = 0
    gradient_clip_norm: float | None = None
    record_spd_step_every: int = 0
    maximum_spd_log_step_radius: float | None = None
    spd_step_projection_bisections: int = 12
    group_volume_ratio_l2_loss_weight: float = 0.0
    group_volume_ratio_l2_mean_fraction: float = 0.5
    group_volume_ratio_l2_smooth_max_temperature: float = 0.1
    group_volume_ratio_cvar_loss_weight: float = 0.0
    group_volume_ratio_cvar_tail_fraction: float = 0.01
    group_positive_log_ratio_cvar_loss_weight: float = 0.0
    group_positive_log_ratio_cvar_tail_fraction: float = 0.01
    global_volume_ratio_cvar_loss_weight: float = 0.0
    global_volume_ratio_cvar_tail_fractions: tuple[float, ...] = ()
    global_volume_ratio_cvar_tail_weights: tuple[float, ...] = ()
    global_upper_log_ratio_cvar_loss_weight: float = 0.0
    global_upper_log_ratio_cvar_tail_fractions: tuple[float, ...] = ()
    global_upper_log_ratio_cvar_tail_weights: tuple[float, ...] = ()
    global_upper_log_ratio_threshold: float = 3.0
    global_upper_log_ratio_smooth_temperature: float = 0.1
    active_set_path: Path | None = None
    active_set_loss_weight: float = 0.0
    active_set_log_ratio_threshold: float = float(np.log(3.0))
    active_set_smooth_max_temperature: float = 0.1
    active_set_region_balanced: bool = False
    selection_sigma_weight: float = 0.0
    selection_volume_ratio_l2_weight: float = 0.0
    selection_volume_ratio_l2_max_weight: float = 0.0
    selection_positive_log_ratio_q999_weight: float = 0.0
    selection_positive_log_ratio_cvar_weight: float = 0.0
    selection_ratio_above_3_weighted_mass_weight: float = 0.0
    maximum_check_volume_ratio_l2: float | None = None
    torch_seed: int = 3200
    basis_seed: int = 3201
    train_seed: int = 3202
    validation_seed: int = 3203
    train_batches: tuple[tuple[int, int], ...] = ()
    validation_batches: tuple[tuple[int, int], ...] = ()
    checkpoint_batches: tuple[tuple[int, int], ...] = ()
    checkpoint_policy: str = "best_score"
    checkpoint_required_consecutive: int = 1
    ineligible_checkpoint_action: str = "continue"
    ineligible_checkpoint_learning_rate_factor: float = 0.5
    maximum_ineligible_checkpoint_rollbacks: int = 8
    maximum_checkpoint_sigma: float | None = None
    maximum_checkpoint_volume_ratio_l2: float | None = None
    maximum_checkpoint_positive_log_ratio_q999: float | None = None
    maximum_checkpoint_positive_log_ratio_cvar: float | None = None
    maximum_checkpoint_ratio_above_3_weighted_mass: float | None = None
    check_seeds: tuple[int, ...] = (3301, 3302, 3303, 3304)
    check_batches: tuple[tuple[int, int], ...] = ()
    export_seed: int = 3401
    initial_artifact: Path | None = None
    train_common_pool: Path | None = None
    selection_common_pool: Path | None = None
    sampling_workers: int = 1
    sampling_cluster_size: int = 1
    sampling_backend: str = "process"
    training_sampling_mode: str = "complete_fibres"
    stream_training_groups: bool = False

    def __post_init__(self) -> None:
        if (self.train_common_pool is None) != (
            self.selection_common_pool is None
        ):
            raise ValueError(
                "train_common_pool and selection_common_pool must be provided together"
            )
        if self.train_common_pool is not None:
            if any(
                (
                    self.train_batches,
                    self.validation_batches,
                    self.checkpoint_batches,
                    self.check_batches,
                )
            ):
                raise ValueError(
                    "common point pools cannot be combined with explicit data batches"
                )
            if self.training_sampling_mode != "complete_fibres":
                raise ValueError(
                    "common point pools require complete_fibres sampling mode"
                )
            if self.stream_training_groups:
                raise ValueError(
                    "common point pools are cached and cannot stream training groups"
                )
            if self.train_common_pool.expanduser().resolve() == (
                self.selection_common_pool.expanduser().resolve()
            ):
                raise ValueError("training and selection common pools must be distinct")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.precision not in {"complex64", "complex128"}:
            raise ValueError("precision must be complex64 or complex128")
        if self.h_parameterization not in {
            "full_cholesky",
            "reference_whitened_cholesky",
            "reference_eigen_diagonal",
            "reference_low_rank",
        }:
            raise ValueError(
                "h_parameterization must be full_cholesky, "
                "reference_whitened_cholesky, reference_eigen_diagonal, "
                "or reference_low_rank"
            )
        if self.group_normalization_mode not in {
            "per_group",
            "fixed_initial_training_pool",
        }:
            raise ValueError(
                "group_normalization_mode must be per_group or "
                "fixed_initial_training_pool"
            )
        if self.group_optimizer_step_mode not in {
            "accumulate_all_groups",
            "shuffled_step_per_group",
            "shuffled_cluster_minibatch",
            "shuffled_point_minibatch",
        }:
            raise ValueError(
                "group_optimizer_step_mode must be accumulate_all_groups, "
                "shuffled_step_per_group, shuffled_cluster_minibatch, or "
                "shuffled_point_minibatch"
            )
        if self.group_shuffle_seed is not None and self.group_shuffle_seed < 0:
            raise ValueError("group_shuffle_seed cannot be negative")
        if self.clusters_per_optimizer_step < 0:
            raise ValueError("clusters_per_optimizer_step cannot be negative")
        if self.cluster_tail_replay_batches_per_epoch < 0:
            raise ValueError("cluster_tail_replay_batches_per_epoch cannot be negative")
        if self.points_per_optimizer_step < 0:
            raise ValueError("points_per_optimizer_step cannot be negative")
        if self.point_upper_tail_replay_points_per_fraction < 0:
            raise ValueError(
                "point_upper_tail_replay_points_per_fraction cannot be negative"
            )
        if self.cluster_minibatch_loss_reduction not in {"mean", "sum"}:
            raise ValueError(
                "cluster_minibatch_loss_reduction must be mean or sum"
            )
        if self.point_minibatch_loss_reduction not in {"mean", "sum"}:
            raise ValueError("point_minibatch_loss_reduction must be mean or sum")
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be positive")
        if self.record_spd_step_every < 0:
            raise ValueError("record_spd_step_every cannot be negative")
        if self.maximum_spd_log_step_radius is not None and (
            not np.isfinite(self.maximum_spd_log_step_radius)
            or self.maximum_spd_log_step_radius <= 0
        ):
            raise ValueError("maximum_spd_log_step_radius must be positive")
        if self.spd_step_projection_bisections <= 0:
            raise ValueError("spd_step_projection_bisections must be positive")
        if self.low_rank <= 0 or self.low_rank_epsilon <= 0:
            raise ValueError("low_rank and low_rank_epsilon must be positive")
        if (
            min(
                self.basis_points,
                self.train_points,
                self.validation_points,
                self.check_points,
                self.export_points,
            )
            <= 0
        ):
            raise ValueError("all training point counts must be positive")
        if self.epochs < 0:
            raise ValueError("epochs cannot be negative")
        if self.learning_rate <= 0 or self.eval_every <= 0:
            raise ValueError("learning rate and eval_every must be positive")
        if self.patience_evaluations < 0:
            raise ValueError("patience_evaluations cannot be negative")
        if (
            min(
                self.validation_weight,
                self.drift_weight,
                self.centered_log_variance_loss_weight,
                self.metric_barrier_weight,
                self.relative_log_spectrum_loss_weight,
                self.low_rank_factor_weight,
                self.sigma_loss_weight,
                self.volume_ratio_l2_loss_weight,
                self.group_volume_ratio_l2_loss_weight,
                self.group_volume_ratio_cvar_loss_weight,
                self.group_positive_log_ratio_cvar_loss_weight,
                self.global_volume_ratio_cvar_loss_weight,
                self.global_upper_log_ratio_cvar_loss_weight,
                self.active_set_loss_weight,
                self.selection_sigma_weight,
                self.selection_volume_ratio_l2_weight,
                self.selection_volume_ratio_l2_max_weight,
                self.selection_positive_log_ratio_q999_weight,
                self.selection_positive_log_ratio_cvar_weight,
                self.selection_ratio_above_3_weighted_mass_weight,
            )
            < 0
        ):
            raise ValueError("loss and selection weights cannot be negative")
        if not 0.0 <= self.group_volume_ratio_l2_mean_fraction <= 1.0:
            raise ValueError("group L2 mean fraction must lie in [0, 1]")
        if not 0.0 < self.group_volume_ratio_cvar_tail_fraction <= 1.0:
            raise ValueError("group CVaR tail fraction must lie in (0, 1]")
        if not 0.0 < self.group_positive_log_ratio_cvar_tail_fraction <= 1.0:
            raise ValueError("group positive-log CVaR tail fraction must lie in (0, 1]")
        if bool(self.global_volume_ratio_cvar_loss_weight) != bool(
            self.global_volume_ratio_cvar_tail_fractions
        ):
            raise ValueError(
                "global CVaR weight and tail fractions must be enabled together"
            )
        if self.global_volume_ratio_cvar_tail_fractions:
            if len(self.global_volume_ratio_cvar_tail_fractions) != len(
                self.global_volume_ratio_cvar_tail_weights
            ):
                raise ValueError(
                    "global CVaR tail fractions and weights must have equal length"
                )
            if any(
                not np.isfinite(value) or not 0.0 < value <= 1.0
                for value in self.global_volume_ratio_cvar_tail_fractions
            ):
                raise ValueError("global CVaR tail fractions must lie in (0, 1]")
            if any(
                not np.isfinite(value) or value <= 0
                for value in self.global_volume_ratio_cvar_tail_weights
            ):
                raise ValueError("global CVaR tail weights must be positive")
        if bool(self.global_upper_log_ratio_cvar_loss_weight) != bool(
            self.global_upper_log_ratio_cvar_tail_fractions
        ):
            raise ValueError(
                "global upper-log CVaR weight and tail fractions must be enabled together"
            )
        if self.global_upper_log_ratio_cvar_tail_fractions:
            if len(self.global_upper_log_ratio_cvar_tail_fractions) != len(
                self.global_upper_log_ratio_cvar_tail_weights
            ):
                raise ValueError(
                    "global upper-log CVaR tail fractions and weights must have equal length"
                )
            if any(
                not np.isfinite(value) or not 0.0 < value <= 1.0
                for value in self.global_upper_log_ratio_cvar_tail_fractions
            ):
                raise ValueError(
                    "global upper-log CVaR tail fractions must lie in (0, 1]"
                )
            if any(
                not np.isfinite(value) or value <= 0
                for value in self.global_upper_log_ratio_cvar_tail_weights
            ):
                raise ValueError("global upper-log CVaR tail weights must be positive")
        if (
            not np.isfinite(self.global_upper_log_ratio_threshold)
            or self.global_upper_log_ratio_threshold <= 1.0
        ):
            raise ValueError("global upper-log ratio threshold must be greater than one")
        if (
            not np.isfinite(self.global_upper_log_ratio_smooth_temperature)
            or self.global_upper_log_ratio_smooth_temperature <= 0.0
        ):
            raise ValueError("global upper-log smooth temperature must be positive")
        if (
            not np.isfinite(self.group_volume_ratio_l2_smooth_max_temperature)
            or self.group_volume_ratio_l2_smooth_max_temperature <= 0
        ):
            raise ValueError("group L2 smooth-max temperature must be positive")
        if (
            not np.isfinite(self.active_set_log_ratio_threshold)
            or self.active_set_log_ratio_threshold <= 0
        ):
            raise ValueError("active-set log-ratio threshold must be positive")
        if (
            not np.isfinite(self.active_set_smooth_max_temperature)
            or self.active_set_smooth_max_temperature <= 0
        ):
            raise ValueError("active-set smooth-max temperature must be positive")
        if self.checkpoint_policy not in {
            "best_score",
            "first_consecutive_eligible",
            "best_score_exploratory",
        }:
            raise ValueError("checkpoint_policy is not supported")
        if self.checkpoint_required_consecutive <= 0:
            raise ValueError("checkpoint_required_consecutive must be positive")
        if self.ineligible_checkpoint_action not in {
            "continue",
            "stop",
            "rollback_reduce_lr",
        }:
            raise ValueError("ineligible_checkpoint_action is not supported")
        if (
            not np.isfinite(self.ineligible_checkpoint_learning_rate_factor)
            or not 0.0 < self.ineligible_checkpoint_learning_rate_factor < 1.0
        ):
            raise ValueError(
                "ineligible checkpoint learning-rate factor must lie in (0, 1)"
            )
        if self.maximum_ineligible_checkpoint_rollbacks <= 0:
            raise ValueError(
                "maximum_ineligible_checkpoint_rollbacks must be positive"
            )
        for name, value in (
            ("maximum_checkpoint_sigma", self.maximum_checkpoint_sigma),
            (
                "maximum_checkpoint_volume_ratio_l2",
                self.maximum_checkpoint_volume_ratio_l2,
            ),
            (
                "maximum_checkpoint_positive_log_ratio_q999",
                self.maximum_checkpoint_positive_log_ratio_q999,
            ),
            (
                "maximum_checkpoint_positive_log_ratio_cvar",
                self.maximum_checkpoint_positive_log_ratio_cvar,
            ),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive")
        if self.maximum_checkpoint_ratio_above_3_weighted_mass is not None and (
            not np.isfinite(self.maximum_checkpoint_ratio_above_3_weighted_mass)
            or not 0.0 <= self.maximum_checkpoint_ratio_above_3_weighted_mass <= 1.0
        ):
            raise ValueError(
                "maximum_checkpoint_ratio_above_3_weighted_mass must lie in [0, 1]"
            )
        if self.maximum_check_volume_ratio_l2 is not None and (
            not np.isfinite(self.maximum_check_volume_ratio_l2)
            or self.maximum_check_volume_ratio_l2 <= 0
        ):
            raise ValueError("maximum_check_volume_ratio_l2 must be positive")
        if self.check_batches:
            if any(len(batch) != 2 for batch in self.check_batches):
                raise ValueError(
                    "check_batches entries must be (seed, point_count) pairs"
                )
            check_seeds = [int(batch[0]) for batch in self.check_batches]
            check_counts = [int(batch[1]) for batch in self.check_batches]
            if len(set(check_seeds)) != len(check_seeds):
                raise ValueError("check_batches seeds must be distinct")
            if any(count <= 0 for count in check_counts):
                raise ValueError("check_batches point counts must be positive")
        elif not self.check_seeds or len(set(self.check_seeds)) != len(
            self.check_seeds
        ):
            raise ValueError("check_seeds must be non-empty and distinct")
        for name, batches in (
            ("train_batches", self.train_batches),
            ("validation_batches", self.validation_batches),
            ("checkpoint_batches", self.checkpoint_batches),
        ):
            if any(len(batch) != 2 for batch in batches):
                raise ValueError(f"{name} entries must be (seed, point_count) pairs")
            seeds = [int(batch[0]) for batch in batches]
            counts = [int(batch[1]) for batch in batches]
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{name} seeds must be distinct")
            if any(count <= 0 for count in counts):
                raise ValueError(f"{name} point counts must be positive")
        effective_train_seeds = {int(seed) for seed, _ in self.train_batches} or {
            int(self.train_seed)
        }
        effective_validation_seeds = (
            set()
            if self.groupwise_objective
            else (
                {int(seed) for seed, _ in self.validation_batches}
                or {int(self.validation_seed)}
            )
        )
        if effective_train_seeds & effective_validation_seeds:
            raise ValueError("training and validation seeds must be disjoint")
        checkpoint_seeds = {int(seed) for seed, _ in self.checkpoint_batches}
        if checkpoint_seeds & effective_train_seeds:
            raise ValueError("training and checkpoint seeds must be disjoint")
        if checkpoint_seeds & effective_validation_seeds:
            raise ValueError("validation and checkpoint seeds must be disjoint")
        if self.checkpoint_batches and not self.groupwise_objective:
            raise ValueError("checkpoint_batches require groupwise_objective")
        if self.group_volume_ratio_cvar_loss_weight and not self.groupwise_objective:
            raise ValueError("group CVaR loss requires groupwise_objective")
        if (
            self.group_positive_log_ratio_cvar_loss_weight
            and not self.groupwise_objective
        ):
            raise ValueError(
                "group positive-log CVaR loss requires groupwise_objective"
            )
        if self.global_volume_ratio_cvar_loss_weight and not self.groupwise_objective:
            raise ValueError("global CVaR loss requires groupwise_objective")
        if (
            self.global_upper_log_ratio_cvar_loss_weight
            and not self.groupwise_objective
        ):
            raise ValueError("global upper-log CVaR loss requires groupwise_objective")
        if (
            self.global_volume_ratio_cvar_loss_weight
            and self.group_normalization_mode != "fixed_initial_training_pool"
        ):
            raise ValueError("global CVaR loss requires fixed training-pool kappa")
        if (
            self.global_upper_log_ratio_cvar_loss_weight
            and self.group_normalization_mode != "fixed_initial_training_pool"
        ):
            raise ValueError(
                "global upper-log CVaR loss requires fixed training-pool kappa"
            )
        if (self.active_set_path is None) != (self.active_set_loss_weight == 0):
            raise ValueError(
                "active_set_path and a positive active_set_loss_weight are required together"
            )
        if self.active_set_path is not None and not self.groupwise_objective:
            raise ValueError("active-set training requires groupwise_objective")
        if self.active_set_region_balanced and self.active_set_path is None:
            raise ValueError("region-balanced active-set loss requires active_set_path")
        if self.groupwise_objective:
            minibatch_mode = self.group_optimizer_step_mode in {
                "shuffled_cluster_minibatch",
                "shuffled_point_minibatch",
            }
            minimum_training_batches = 1 if minibatch_mode else 2
            available_training_batches = (
                1 if self.train_common_pool is not None else len(self.train_batches)
            )
            if available_training_batches < minimum_training_batches:
                raise ValueError(
                    "groupwise_objective requires at least "
                    f"{minimum_training_batches} train_batches for this optimizer mode"
                )
            if (
                self.selection_common_pool is None
                and not self.checkpoint_batches
            ):
                raise ValueError("groupwise_objective requires checkpoint_batches")
            if self.validation_batches or self.validation_weight != 0:
                raise ValueError(
                    "groupwise_objective requires empty validation_batches and "
                    "validation_weight=0"
                )
            if self.volume_ratio_l2_loss_weight != 0:
                raise ValueError(
                    "groupwise_objective cannot also use the pooled L2 loss"
                )
            if self.checkpoint_policy not in {
                "first_consecutive_eligible",
                "best_score_exploratory",
            }:
                raise ValueError(
                    "groupwise_objective requires first_consecutive_eligible "
                    "checkpoints or explicit best_score_exploratory selection"
                )
            if (
                self.checkpoint_policy == "first_consecutive_eligible"
                and self.checkpoint_required_consecutive < 2
            ):
                raise ValueError(
                    "groupwise_objective requires at least two consecutive checkpoint passes"
                )
            if (
                self.checkpoint_policy == "best_score_exploratory"
                and self.selection_volume_ratio_l2_max_weight <= 0
            ):
                raise ValueError(
                    "best_score_exploratory requires a positive maximum-L2 "
                    "selection weight"
                )
            if self.maximum_checkpoint_sigma is None:
                raise ValueError(
                    "groupwise_objective requires maximum_checkpoint_sigma"
                )
            if self.maximum_checkpoint_volume_ratio_l2 is None:
                raise ValueError(
                    "groupwise_objective requires maximum_checkpoint_volume_ratio_l2"
                )
        if self.sampling_workers <= 0 or self.sampling_cluster_size <= 0:
            raise ValueError(
                "sampling_workers and sampling_cluster_size must be positive"
            )
        if self.sampling_backend not in {"process", "thread"}:
            raise ValueError("sampling_backend must be process or thread")
        if self.training_sampling_mode not in {
            "complete_fibres",
            "one_random_root_per_fibre",
        }:
            raise ValueError(
                "training_sampling_mode must be complete_fibres or "
                "one_random_root_per_fibre"
            )
        if self.training_sampling_mode == "one_random_root_per_fibre":
            if self.group_optimizer_step_mode != "shuffled_point_minibatch":
                raise ValueError(
                    "one_random_root_per_fibre requires shuffled_point_minibatch"
                )
            if self.sampling_cluster_size != 4:
                raise ValueError(
                    "one_random_root_per_fibre requires sampling_cluster_size=4"
                )
        if self.stream_training_groups and not self.groupwise_objective:
            raise ValueError("stream_training_groups requires groupwise_objective")
        if (
            self.group_optimizer_step_mode != "accumulate_all_groups"
            and not self.groupwise_objective
        ):
            raise ValueError(
                "shuffled per-group optimizer steps require groupwise_objective"
            )
        if (
            self.group_normalization_mode != "per_group"
            and not self.groupwise_objective
        ):
            raise ValueError("fixed group normalization requires groupwise_objective")
        cluster_minibatch = (
            self.group_optimizer_step_mode == "shuffled_cluster_minibatch"
        )
        point_minibatch = self.group_optimizer_step_mode == "shuffled_point_minibatch"
        minibatch = cluster_minibatch or point_minibatch
        if self.relative_log_spectrum_loss_weight and (
            not minibatch
            or self.h_parameterization != "reference_whitened_cholesky"
        ):
            raise ValueError(
                "relative log-spectrum loss requires reference-whitened "
                "mini-batch training"
            )
        if minibatch:
            if self.group_normalization_mode != "fixed_initial_training_pool":
                raise ValueError(
                    "shuffled mini-batches require fixed_initial_training_pool normalization"
                )
            if self.stream_training_groups:
                raise ValueError("shuffled mini-batches require cached training groups")
            if self.sigma_loss_weight <= 0:
                raise ValueError("shuffled mini-batches require positive sigma_loss_weight")
            if any(
                (
                    self.group_volume_ratio_l2_loss_weight,
                    self.group_volume_ratio_cvar_loss_weight,
                    self.group_positive_log_ratio_cvar_loss_weight,
                    self.active_set_loss_weight,
                )
            ) or self.active_set_path is not None:
                raise ValueError(
                    "shuffled mini-batches do not mix in group-tail or active-set objectives"
                )
        if cluster_minibatch:
            if self.clusters_per_optimizer_step <= 0:
                raise ValueError(
                    "shuffled_cluster_minibatch requires positive "
                    "clusters_per_optimizer_step"
                )
            if (
                self.cluster_tail_replay_batches_per_epoch
                and not self.global_volume_ratio_cvar_loss_weight
            ):
                raise ValueError(
                    "cluster tail replay requires global volume-ratio CVaR"
                )
        elif self.clusters_per_optimizer_step:
            raise ValueError(
                "clusters_per_optimizer_step requires shuffled_cluster_minibatch"
            )
        elif self.cluster_minibatch_uniform_loss:
            raise ValueError(
                "cluster_minibatch_uniform_loss requires shuffled_cluster_minibatch"
            )
        elif self.cluster_minibatch_loss_reduction != "mean":
            raise ValueError(
                "cluster_minibatch_loss_reduction requires "
                "shuffled_cluster_minibatch"
            )
        elif self.cluster_tail_replay_batches_per_epoch:
            raise ValueError(
                "cluster_tail_replay_batches_per_epoch requires "
                "shuffled_cluster_minibatch"
            )
        if point_minibatch:
            if self.points_per_optimizer_step <= 0:
                raise ValueError(
                    "shuffled_point_minibatch requires positive points_per_optimizer_step"
                )
            if (
                self.point_upper_tail_replay_points_per_fraction
                and not self.global_upper_log_ratio_cvar_loss_weight
            ):
                raise ValueError(
                    "point upper-tail replay requires global upper-log CVaR"
                )
        elif self.points_per_optimizer_step:
            raise ValueError(
                "points_per_optimizer_step requires shuffled_point_minibatch"
            )
        elif self.point_minibatch_uniform_loss:
            raise ValueError(
                "point_minibatch_uniform_loss requires shuffled_point_minibatch"
            )
        elif self.point_minibatch_loss_reduction != "mean":
            raise ValueError(
                "point_minibatch_loss_reduction requires shuffled_point_minibatch"
            )
        elif self.point_upper_tail_replay_points_per_fraction:
            raise ValueError(
                "point upper-tail replay requires shuffled_point_minibatch"
            )
        if self.sampling_workers > 1:
            counts = [self.basis_points, self.export_points]
            if self.train_common_pool is None:
                if self.train_batches:
                    counts.extend(count for _, count in self.train_batches)
                else:
                    counts.append(self.train_points)
                if self.groupwise_objective:
                    pass
                elif self.validation_batches:
                    counts.extend(count for _, count in self.validation_batches)
                else:
                    counts.append(self.validation_points)
                counts.extend(count for _, count in self.checkpoint_batches)
                if self.check_batches:
                    counts.extend(count for _, count in self.check_batches)
                else:
                    counts.append(self.check_points)
            if any(count % self.sampling_cluster_size for count in counts):
                raise ValueError(
                    "parallel sampling point counts must be divisible by sampling_cluster_size"
                )
            if (
                min(count // self.sampling_cluster_size for count in counts)
                < self.sampling_workers
            ):
                raise ValueError(
                    "parallel sampling requires at least one complete cluster per worker"
                )


def _metric_stats(
    adapter: GCICYAdapter,
    points: list[Any],
    metrics: np.ndarray,
    importance_weights: np.ndarray,
    *,
    weighted: bool,
) -> dict[str, float]:
    weights = importance_weights if weighted else np.ones(len(points), dtype=float)
    residuals = adapter.residual_values(points, metrics)
    stats = standard_errors(residuals, weights)
    normalized_ratio, _ = normalized_volume_ratios(residuals, weights)
    cluster_ids = np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
    if cluster_ids.shape != (len(points),):
        raise ValueError("sampling cluster ids must have one entry per point")
    failure = normalized_ratio > 3.0
    stats["sampling_cluster_count"] = int(len(np.unique(cluster_ids)))
    stats["normalized_ratio_above_3_cluster_count"] = int(
        len(np.unique(cluster_ids[failure]))
    )
    stats["min_metric_eigenvalue"] = float(
        np.min(np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128)))
    )
    return stats


def _stats_gate(baseline: dict[str, float], candidate: dict[str, float]) -> bool:
    return bool(
        np.isfinite(candidate["weighted_centered_log_ma_rms"])
        and candidate["min_metric_eigenvalue"] > 0
        and candidate["weighted_centered_log_ma_rms"]
        < baseline["weighted_centered_log_ma_rms"]
    )


def train_h_metric(
    adapter: GCICYAdapter,
    request: TrainingRequest,
    *,
    artifact_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Train and serialize one global-section H-metric through an adapter."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for H-metric training") from exc

    started = time.perf_counter()
    kahler_power = adapter.configuration.kahler_power(request.degree)
    normalization = 1.0 / kahler_power
    if request.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(request.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    real_dtype = torch.float32 if request.precision == "complex64" else torch.float64
    complex_dtype = (
        torch.complex64 if request.precision == "complex64" else torch.complex128
    )
    prepared_complex_dtype = (
        np.complex64 if request.precision == "complex64" else np.complex128
    )
    torch.manual_seed(request.torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(request.torch_seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = adapter.make_model(request.model_seed, exact=request.exact_model)
    sampling_started = time.perf_counter()
    sampling_shards: dict[str, list[dict[str, int]]] = {}
    basis_started = time.perf_counter()
    print(
        f"preparing basis: points={request.basis_points}, seed={request.basis_seed}",
        flush=True,
    )
    if request.sampling_workers > 1:
        basis_points, sampling_shards["basis"] = sample_points_parallel(
            adapter,
            model_seed=request.model_seed,
            exact_model=request.exact_model,
            count=request.basis_points,
            seed=request.basis_seed,
            workers=request.sampling_workers,
            cluster_size=request.sampling_cluster_size,
            backend=request.sampling_backend,
        )
    else:
        basis_points = adapter.sample_points(
            model,
            request.basis_points,
            seed=request.basis_seed,
        )
    basis = adapter.restricted_section_basis(basis_points, request.degree)
    initial_h, relation_error = adapter.restricted_fubini_study_h_matrix(
        basis_points, basis
    )
    initialization = "ambient_fubini_study"
    source_artifact: HMetricArtifact | None = None
    if request.initial_artifact is not None:
        source_artifact = adapter.load_h_artifact(request.initial_artifact, model)
        initial_h, relation_error = adapter.lift_h_matrix_power(
            basis_points,
            source_artifact,
            basis,
        )
        initialization = f"lifted_power_of:{source_artifact.path}"
    exponents = np.asarray(basis.selected_exponents, dtype=np.int64)
    print(
        f"prepared basis: ambient={len(basis.ambient_exponents)}, "
        f"restricted={basis.numerical_rank}, relation_error={relation_error:.3e}, "
        f"seconds={time.perf_counter() - basis_started:.1f}",
        flush=True,
    )
    initial_h = np.asarray(initial_h, dtype=np.complex128)
    initial_h = 0.5 * (initial_h + initial_h.conjugate().T)
    initial_h *= len(initial_h) / float(np.trace(initial_h).real)
    initial_cholesky = np.linalg.cholesky(initial_h)
    if request.h_parameterization == "reference_low_rank" and request.low_rank >= len(
        initial_h
    ):
        raise ValueError("low_rank must be smaller than the section count")

    def prepare(
        points: list[Any],
        *,
        retain_evaluation_data: bool = True,
        unique_sampling_clusters: bool = False,
        importance_weights: np.ndarray | None = None,
        sampling_cluster_ids: np.ndarray | None = None,
        log_omega: np.ndarray | None = None,
        baseline_metrics: np.ndarray | None = None,
    ) -> dict[str, Any]:
        values, derivatives = adapter.section_values_and_jacobian_batch(
            points,
            exponents,
        )
        weights = (
            adapter.importance_weights(points)
            if importance_weights is None
            else np.asarray(importance_weights, dtype=np.float64)
        )
        if weights.shape != (len(points),):
            raise ValueError("importance weights must have one entry per point")
        if sampling_cluster_ids is not None:
            cluster_ids = np.asarray(sampling_cluster_ids, dtype=np.int64)
        elif unique_sampling_clusters:
            cluster_ids = np.arange(len(points), dtype=np.int64)
        else:
            cluster_ids = np.asarray(
                adapter.sampling_cluster_ids(points),
                dtype=np.int64,
            )
        if cluster_ids.shape != (len(points),):
            raise ValueError("sampling cluster ids must have one entry per point")
        prepared_log_omega = (
            np.asarray(
                [
                    adapter.holomorphic_volume_log_density(point)
                    for point in points
                ],
                dtype=float,
            )
            if log_omega is None
            else np.asarray(log_omega, dtype=np.float64)
        )
        if prepared_log_omega.shape != (len(points),):
            raise ValueError("log Omega density must have one entry per point")
        dataset = {
            "values": np.asarray(values, dtype=prepared_complex_dtype),
            "derivatives": np.asarray(derivatives, dtype=prepared_complex_dtype),
            "log_omega": prepared_log_omega,
            "importance_weights": weights,
            "sampling_cluster_ids": cluster_ids,
        }
        if retain_evaluation_data:
            dataset["points"] = points
            dataset["baseline_metrics"] = (
                adapter.baseline_metrics(points)
                if baseline_metrics is None
                else np.asarray(baseline_metrics, dtype=np.complex128)
            )
        return dataset

    def prepare_common_pool(
        pool: CommonPointPool,
        *,
        retain_evaluation_data: bool = True,
    ) -> dict[str, Any]:
        stage_started = time.perf_counter()
        split = str(pool.metadata["split"])
        print(
            f"preparing frozen {split} pool: points={len(pool.points)}, "
            f"path={pool.path}",
            flush=True,
        )
        dataset = prepare(
            pool.points,
            retain_evaluation_data=retain_evaluation_data,
            importance_weights=pool.importance_weights,
            sampling_cluster_ids=pool.sampling_cluster_ids,
            log_omega=pool.holomorphic_volume_log_density,
            baseline_metrics=pool.baseline_metrics,
        )
        print(
            f"prepared frozen {split} pool: points={len(pool.points)}, "
            f"seconds={time.perf_counter() - stage_started:.1f}",
            flush=True,
        )
        return dataset

    def sample_stage(label: str, count: int, seed: int) -> list[Any]:
        independent_training_fibres = (
            label.startswith("train")
            and request.training_sampling_mode == "one_random_root_per_fibre"
        )
        generated_count = 4 * count if independent_training_fibres else count
        if request.sampling_workers > 1:
            points, shard_metadata = sample_points_parallel(
                adapter,
                model_seed=request.model_seed,
                exact_model=request.exact_model,
                count=generated_count,
                seed=seed,
                workers=request.sampling_workers,
                cluster_size=request.sampling_cluster_size,
                backend=request.sampling_backend,
            )
            sampling_shards[label] = shard_metadata
        else:
            points = adapter.sample_points(model, generated_count, seed=seed)
        if independent_training_fibres:
            if len(points) != generated_count or generated_count % 4:
                raise RuntimeError(
                    "independent-fibre training did not receive complete four-root fibres"
                )
            points = points[::4]
            if len(points) != count:
                raise RuntimeError(
                    "independent-fibre training returned the wrong point count"
                )
        return points

    def sample_and_prepare(
        label: str,
        count: int,
        seed: int,
        *,
        retain_evaluation_data: bool = True,
    ) -> dict[str, Any]:
        stage_started = time.perf_counter()
        print(f"preparing {label}: points={count}, seed={seed}", flush=True)
        points = sample_stage(label, count, seed)
        independent_training_fibres = (
            label.startswith("train")
            and request.training_sampling_mode == "one_random_root_per_fibre"
        )
        dataset = prepare(
            points,
            retain_evaluation_data=retain_evaluation_data,
            unique_sampling_clusters=independent_training_fibres,
        )
        print(
            f"prepared {label}: points={count}, "
            f"seconds={time.perf_counter() - stage_started:.1f}",
            flush=True,
        )
        return dataset

    def sample_batches_and_prepare(
        label: str, batches: tuple[tuple[int, int], ...]
    ) -> dict[str, Any]:
        if len(batches) == 1:
            seed, count = batches[0]
            return sample_and_prepare(label, count, seed)
        stage_started = time.perf_counter()
        total = sum(count for _, count in batches)
        print(
            f"preparing {label}: batches={len(batches)}, total_points={total}",
            flush=True,
        )
        points = []
        for seed, count in batches:
            batch_label = f"{label} seed{seed}"
            print(
                f"preparing {batch_label}: points={count}, seed={seed}",
                flush=True,
            )
            points.extend(sample_stage(batch_label, count, seed))
        dataset = prepare(points)
        print(
            f"prepared {label}: batches={len(batches)}, points={total}, "
            f"seconds={time.perf_counter() - stage_started:.1f}",
            flush=True,
        )
        return dataset

    train_common_pool = None
    selection_common_pool = None
    common_pool_records: dict[str, dict[str, Any]] = {}
    if request.train_common_pool is not None:
        train_common_pool = load_common_point_pool(
            request.train_common_pool,
            adapter,
            model,
            expected_model_seed=request.model_seed,
            expected_exact_model=request.exact_model,
            expected_split="train",
        )
        selection_common_pool = load_common_point_pool(
            request.selection_common_pool,
            adapter,
            model,
            expected_model_seed=request.model_seed,
            expected_exact_model=request.exact_model,
            expected_split="selection",
        )
        if len(train_common_pool.points) != request.train_points:
            raise ValueError("training common-pool point count does not match request")
        if len(selection_common_pool.points) != request.validation_points:
            raise ValueError("selection common-pool point count does not match request")
        for split, pool in (
            ("train", train_common_pool),
            ("selection", selection_common_pool),
        ):
            if int(pool.metadata["cluster_size"]) != request.sampling_cluster_size:
                raise ValueError(
                    f"{split} common-pool cluster size does not match request"
                )
            common_pool_records[split] = {
                "path": str(pool.path),
                "sha256": file_sha256(pool.path),
                "point_count": len(pool.points),
                "sampling_seed": int(pool.metadata["sampling_seed"]),
                "cluster_size": int(pool.metadata["cluster_size"]),
                "cluster_count": int(pool.metadata["cluster_count"]),
            }

    train_batches = (
        ()
        if train_common_pool is not None
        else request.train_batches
        or ((request.train_seed, request.train_points),)
    )
    validation_batches = (
        ()
        if request.groupwise_objective or selection_common_pool is not None
        else request.validation_batches
        or ((request.validation_seed, request.validation_points),)
    )
    if request.groupwise_objective:
        if train_common_pool is not None:
            train_groups = [
                (
                    "frozen_common_train",
                    prepare_common_pool(train_common_pool),
                )
            ]
        elif request.group_optimizer_step_mode == "shuffled_point_minibatch":
            # Point mini-batches represent one global i.i.d. training pool. Pool
            # seed shards before shuffling so a batch is not constrained to a
            # single shard (or to complete fibres within that shard).
            train_groups = [
                (
                    "pooled_training_points",
                    sample_batches_and_prepare("train pool", train_batches),
                )
            ]
        else:
            train_groups = [
                (
                    f"seed{seed}",
                    sample_and_prepare(
                        f"train seed{seed}",
                        count,
                        seed,
                        retain_evaluation_data=not request.stream_training_groups,
                    ),
                )
                for seed, count in train_batches
            ]
        train = None
        validation = None
        checkpoint_sets = (
            [
                (
                    "frozen_common_selection",
                    prepare_common_pool(selection_common_pool),
                )
            ]
            if selection_common_pool is not None
            else [
                (
                    f"seed{seed}",
                    sample_and_prepare(f"checkpoint seed{seed}", count, seed),
                )
                for seed, count in request.checkpoint_batches
            ]
        )
        check_batches: tuple[tuple[int, int], ...] = ()
        checks: list[tuple[str, dict[str, Any]]] = []
    else:
        train_groups = []
        if train_common_pool is not None and selection_common_pool is not None:
            train = prepare_common_pool(train_common_pool)
            validation = prepare_common_pool(selection_common_pool)
        else:
            train = sample_batches_and_prepare("train", train_batches)
            validation = sample_batches_and_prepare("validation", validation_batches)
        checkpoint_sets = []
        if selection_common_pool is not None:
            check_batches = ()
            checks = [("frozen_common_selection", validation)]
        else:
            check_batches = request.check_batches or tuple(
                (seed, request.check_points) for seed in request.check_seeds
            )
            checks = [
                (
                    f"seed{seed}",
                    sample_and_prepare(f"check seed{seed}", count, seed),
                )
                for seed, count in check_batches
            ]
    active_pool = None
    active_set = None
    active_region_indices: list[np.ndarray] = []
    if request.active_set_path is not None:
        active_started = time.perf_counter()
        print(f"preparing active set: {request.active_set_path}", flush=True)
        active_pool = load_active_point_pool(
            request.active_set_path,
            adapter,
            model,
            expected_model_seed=request.model_seed,
            expected_exact_model=request.exact_model,
        )
        active_set = prepare(
            active_pool.points,
            retain_evaluation_data=False,
        )
        active_region_indices = [
            np.flatnonzero(active_pool.center_ids == center_id)
            for center_id in np.unique(active_pool.center_ids)
        ]
        if any(len(indices) == 0 for indices in active_region_indices):
            raise RuntimeError(
                "active-set region construction produced an empty region"
            )
        print(
            f"prepared active set: points={len(active_pool.points)}, "
            f"regions={len(active_region_indices)}, "
            f"seconds={time.perf_counter() - active_started:.1f}",
            flush=True,
        )
    selection_sets = checkpoint_sets or checks
    can_reuse_selection_for_export = bool(
        selection_common_pool is not None
        and selection_sets
        and request.export_points == len(selection_common_pool.points)
        and request.export_seed
        == int(selection_common_pool.metadata["sampling_seed"])
    )
    if can_reuse_selection_for_export:
        print(
            "reusing frozen selection pool for export: "
            f"points={request.export_points}, seed={request.export_seed}",
            flush=True,
        )
        export = selection_sets[0][1]
    else:
        export = sample_and_prepare("export", request.export_points, request.export_seed)
    sampling_seconds = time.perf_counter() - sampling_started

    initial_h_t = torch.tensor(initial_h, dtype=complex_dtype, device=device)
    if request.h_parameterization == "full_cholesky":
        optimization_coordinate_system = "absolute_cholesky"
        spd_step_diagnostic_representation = "absolute_cholesky_factor"
        parameterization_real_dimension = len(initial_h) ** 2
        scale_invariant_h_dimension_upper_bound = len(initial_h) ** 2 - 1
        diagonal_log = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    np.diag(initial_cholesky).real,
                    dtype=real_dtype,
                    device=device,
                )
            )
        )
        lower_real = torch.nn.Parameter(
            torch.tensor(initial_cholesky.real, dtype=real_dtype, device=device)
        )
        lower_imag = torch.nn.Parameter(
            torch.tensor(initial_cholesky.imag, dtype=real_dtype, device=device)
        )
        lower_mask = torch.tril(torch.ones_like(lower_real), diagonal=-1)
        trainable_parameters = [diagonal_log, lower_real, lower_imag]

        def current_spd_diagnostic_factor():
            lower = lower_mask * lower_real + 1j * lower_mask * lower_imag
            return lower.to(complex_dtype) + torch.diag(torch.exp(diagonal_log)).to(
                complex_dtype
            )

        def current_h_matrix():
            cholesky = current_spd_diagnostic_factor()
            h_matrix = cholesky @ torch.conj(cholesky.T)
            return h_matrix * (len(initial_h) / torch.real(torch.trace(h_matrix)))

        def parameterization_penalty():
            return torch.zeros((), dtype=real_dtype, device=device)

        def drift_penalty(h_matrix):
            return torch.mean(torch.abs(h_matrix - initial_h_t) ** 2)

    elif request.h_parameterization == "reference_whitened_cholesky":
        optimization_coordinate_system = "initial_h_relative_cholesky"
        spd_step_diagnostic_representation = "initial_h_relative_cholesky_factor"
        parameterization_real_dimension = len(initial_h) ** 2
        scale_invariant_h_dimension_upper_bound = len(initial_h) ** 2 - 1
        diagonal_log = torch.nn.Parameter(
            torch.zeros(len(initial_h), dtype=real_dtype, device=device)
        )
        lower_real = torch.nn.Parameter(
            torch.zeros(
                (len(initial_h), len(initial_h)),
                dtype=real_dtype,
                device=device,
            )
        )
        lower_imag = torch.nn.Parameter(torch.zeros_like(lower_real))
        lower_mask = torch.tril(torch.ones_like(lower_real), diagonal=-1)
        initial_cholesky_t = torch.tensor(
            initial_cholesky, dtype=complex_dtype, device=device
        )
        identity_t = torch.eye(len(initial_h), dtype=complex_dtype, device=device)
        trainable_parameters = [diagonal_log, lower_real, lower_imag]

        def relative_cholesky_factor():
            lower = lower_mask * lower_real + 1j * lower_mask * lower_imag
            return lower.to(complex_dtype) + torch.diag(torch.exp(diagonal_log)).to(
                complex_dtype
            )

        def current_spd_diagnostic_factor():
            return relative_cholesky_factor()

        def relative_h_matrix():
            factor = relative_cholesky_factor()
            relative_h = factor @ torch.conj(factor.T)
            return relative_h * (len(initial_h) / torch.real(torch.trace(relative_h)))

        def current_h_matrix():
            factor = relative_cholesky_factor()
            transformed = initial_cholesky_t @ factor
            h_matrix = transformed @ torch.conj(transformed.T)
            return h_matrix * (len(initial_h) / torch.real(torch.trace(h_matrix)))

        def parameterization_penalty():
            return torch.zeros((), dtype=real_dtype, device=device)

        def drift_penalty(_h_matrix):
            return torch.mean(torch.abs(relative_h_matrix() - identity_t) ** 2)

    elif request.h_parameterization == "reference_eigen_diagonal":
        optimization_coordinate_system = "initial_h_eigen_diagonal_exponential"
        spd_step_diagnostic_representation = "initial_h_eigen_diagonal_relative_factor"
        parameterization_real_dimension = len(initial_h) - 1
        scale_invariant_h_dimension_upper_bound = len(initial_h) - 1
        initial_eigenvalues, initial_eigenvectors = np.linalg.eigh(initial_h)
        if float(initial_eigenvalues[0]) <= 0:
            raise FloatingPointError("initial H eigenbasis is not positive definite")
        trace_free_frame = np.zeros(
            (len(initial_h), len(initial_h) - 1), dtype=np.float64
        )
        for column in range(len(initial_h) - 1):
            denominator = np.sqrt((column + 1) * (column + 2))
            trace_free_frame[: column + 1, column] = 1.0 / denominator
            trace_free_frame[column + 1, column] = -(column + 1) / denominator
        relative_log_coefficients = torch.nn.Parameter(
            torch.zeros(len(initial_h) - 1, dtype=real_dtype, device=device)
        )
        trace_free_frame_t = torch.tensor(
            trace_free_frame, dtype=real_dtype, device=device
        )
        initial_eigenvalues_t = torch.tensor(
            initial_eigenvalues, dtype=real_dtype, device=device
        )
        initial_eigenvectors_t = torch.tensor(
            initial_eigenvectors, dtype=complex_dtype, device=device
        )
        trainable_parameters = [relative_log_coefficients]

        def relative_log_eigenvalues():
            return trace_free_frame_t @ relative_log_coefficients

        def current_spd_diagnostic_factor():
            return torch.diag(
                torch.exp(0.5 * relative_log_eigenvalues()).to(complex_dtype)
            )

        def current_h_matrix():
            scales = torch.sqrt(initial_eigenvalues_t) * torch.exp(
                0.5 * relative_log_eigenvalues()
            )
            factor = initial_eigenvectors_t @ torch.diag(scales.to(complex_dtype))
            h_matrix = factor @ torch.conj(factor.T)
            return h_matrix * (len(initial_h) / torch.real(torch.trace(h_matrix)))

        def parameterization_penalty():
            return torch.zeros((), dtype=real_dtype, device=device)

        def drift_penalty(_h_matrix):
            relative_eigenvalues = torch.exp(relative_log_eigenvalues())
            return torch.mean((relative_eigenvalues - 1.0) ** 2)

    else:
        optimization_coordinate_system = "initial_h_relative_low_rank"
        spd_step_diagnostic_representation = "dense_h_matrix"
        parameterization_real_dimension = 4 * len(initial_h) * request.low_rank
        scale_invariant_h_dimension_upper_bound = min(
            len(initial_h) ** 2 - 1,
            parameterization_real_dimension,
        )
        rng = np.random.default_rng(request.low_rank_seed)
        frame = rng.normal(size=(len(initial_h), request.low_rank)) + 1j * rng.normal(
            size=(len(initial_h), request.low_rank)
        )
        initial_v, _ = np.linalg.qr(frame)
        initial_v = np.asarray(initial_v[:, : request.low_rank], dtype=np.complex128)
        u_real = torch.nn.Parameter(
            torch.zeros(
                (len(initial_h), request.low_rank),
                dtype=real_dtype,
                device=device,
            )
        )
        u_imag = torch.nn.Parameter(torch.zeros_like(u_real))
        v_real = torch.nn.Parameter(
            torch.tensor(initial_v.real, dtype=real_dtype, device=device)
        )
        v_imag = torch.nn.Parameter(
            torch.tensor(initial_v.imag, dtype=real_dtype, device=device)
        )
        initial_v_t = torch.tensor(initial_v, dtype=complex_dtype, device=device)
        initial_cholesky_t = torch.tensor(
            initial_cholesky, dtype=complex_dtype, device=device
        )
        identity_t = torch.eye(len(initial_h), dtype=complex_dtype, device=device)
        trainable_parameters = [u_real, u_imag, v_real, v_imag]

        def low_rank_factors():
            u = u_real.to(complex_dtype) + 1j * u_imag.to(complex_dtype)
            v = v_real.to(complex_dtype) + 1j * v_imag.to(complex_dtype)
            return u, v

        def current_h_matrix():
            u, v = low_rank_factors()
            transform = identity_t + u @ torch.conj(v.T)
            transformed = initial_cholesky_t @ transform
            h_matrix = transformed @ torch.conj(transformed.T)
            h_matrix = h_matrix + request.low_rank_epsilon * initial_h_t
            return h_matrix * (len(initial_h) / torch.real(torch.trace(h_matrix)))

        def parameterization_penalty():
            u, v = low_rank_factors()
            return torch.mean(torch.abs(u) ** 2) + torch.mean(
                torch.abs(v - initial_v_t) ** 2
            )

        def drift_penalty(h_matrix):
            return torch.mean(torch.abs(h_matrix - initial_h_t) ** 2)

    if request.h_parameterization == "reference_whitened_cholesky":

        def relative_log_spectrum_penalty():
            eigenvalues = torch.linalg.eigvalsh(relative_h_matrix())
            log_eigenvalues = torch.log(torch.clamp(eigenvalues, min=1.0e-8))
            centered = log_eigenvalues - torch.mean(log_eigenvalues)
            return torch.mean(torch.square(centered))

    else:

        def relative_log_spectrum_penalty():
            return torch.zeros((), dtype=real_dtype, device=device)

    def relative_spd_log_statistics(
        reference_h: np.ndarray,
        candidate_h: np.ndarray,
    ) -> dict[str, float]:
        reference = positive_hermitian_projection(reference_h)
        candidate = positive_hermitian_projection(candidate_h)
        reference *= len(reference) / float(np.trace(reference).real)
        candidate *= len(candidate) / float(np.trace(candidate).real)
        cholesky = np.linalg.cholesky(reference)
        left = np.linalg.solve(cholesky, candidate)
        relative = np.linalg.solve(cholesky.conjugate(), left.T).T
        relative = 0.5 * (relative + relative.conjugate().T)
        eigenvalues = np.linalg.eigvalsh(relative)
        scale = float(np.max(np.abs(eigenvalues)))
        floor = max(np.finfo(np.float64).tiny, scale * 1.0e-13)
        if float(eigenvalues[0]) < -scale * 1.0e-8:
            raise FloatingPointError("relative SPD diagnostic found a negative mode")
        log_eigenvalues = np.log(np.maximum(eigenvalues, floor))
        centered = log_eigenvalues - np.mean(log_eigenvalues)
        return {
            "log_eigenvalue_span": float(np.ptp(centered)),
            "max_abs_centered_log_eigenvalue": float(np.max(np.abs(centered))),
        }

    def current_h_numpy() -> np.ndarray:
        with torch.no_grad():
            return np.asarray(
                current_h_matrix().detach().cpu().numpy(),
                dtype=np.complex128,
            )

    use_factor_spd_diagnostic = request.h_parameterization in {
        "full_cholesky",
        "reference_whitened_cholesky",
        "reference_eigen_diagonal",
    }

    def current_spd_diagnostic_state_numpy() -> np.ndarray:
        if not use_factor_spd_diagnostic:
            return current_h_numpy()
        with torch.no_grad():
            return np.asarray(
                current_spd_diagnostic_factor().detach().cpu().numpy(),
                dtype=np.complex128,
            )

    def spd_diagnostic_statistics(
        reference_state: np.ndarray,
        candidate_state: np.ndarray,
    ) -> dict[str, float]:
        if use_factor_spd_diagnostic:
            return _relative_spd_factor_log_statistics(
                reference_state,
                candidate_state,
            )
        return relative_spd_log_statistics(reference_state, candidate_state)

    initial_spd_diagnostic_state = current_spd_diagnostic_state_numpy()

    optimizer = torch.optim.Adam(trainable_parameters, lr=request.learning_rate)

    def snapshot_training_state() -> tuple[list[Any], dict[str, Any]]:
        return (
            [parameter.detach().clone() for parameter in trainable_parameters],
            copy.deepcopy(optimizer.state_dict()),
        )

    def restore_training_state(
        state: tuple[list[Any], dict[str, Any]],
        *,
        learning_rate: float,
    ) -> None:
        parameter_values, optimizer_state = state
        with torch.no_grad():
            for parameter, value in zip(
                trainable_parameters,
                parameter_values,
                strict=True,
            ):
                parameter.copy_(value)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    effective_group_shuffle_seed = (
        request.torch_seed
        if request.group_shuffle_seed is None
        else request.group_shuffle_seed
    )
    group_order_rng = np.random.default_rng(effective_group_shuffle_seed)
    optimizer_step_count = 0
    interval_optimizer_step_count = 0
    interval_preclip_gradient_norms: list[float] = []
    interval_clipped_optimizer_steps = 0
    interval_spd_step_log_spans: list[float] = []
    interval_spd_step_log_radii: list[float] = []
    interval_unprojected_spd_step_log_radii: list[float] = []
    interval_spd_step_projection_scales: list[float] = []
    interval_projected_spd_steps = 0
    ineligible_checkpoint_count = 0
    checkpoint_rollback_count = 0

    def take_optimizer_step() -> None:
        nonlocal optimizer_step_count
        nonlocal interval_optimizer_step_count
        nonlocal interval_clipped_optimizer_steps
        nonlocal interval_projected_spd_steps
        enforce_spd_step_radius = request.maximum_spd_log_step_radius is not None
        should_record_spd_step = bool(
            enforce_spd_step_radius
            or (
                request.record_spd_step_every
                and (optimizer_step_count + 1) % request.record_spd_step_every == 0
            )
        )
        before_spd_state = (
            current_spd_diagnostic_state_numpy() if should_record_spd_step else None
        )
        before_parameters = (
            [parameter.detach().clone() for parameter in trainable_parameters]
            if enforce_spd_step_radius
            else None
        )
        if request.gradient_clip_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=request.gradient_clip_norm,
            )
            gradient_norm_value = float(gradient_norm.detach().cpu())
            interval_preclip_gradient_norms.append(gradient_norm_value)
            if gradient_norm_value > request.gradient_clip_norm:
                interval_clipped_optimizer_steps += 1
        optimizer.step()
        optimizer_step_count += 1
        interval_optimizer_step_count += 1
        if before_spd_state is not None:
            unprojected_statistics = spd_diagnostic_statistics(
                before_spd_state,
                current_spd_diagnostic_state_numpy(),
            )
            interval_unprojected_spd_step_log_radii.append(
                unprojected_statistics["max_abs_centered_log_eigenvalue"]
            )
            step_statistics = unprojected_statistics
            if enforce_spd_step_radius:
                if before_parameters is None:
                    raise RuntimeError(
                        "SPD step projection lost its parameter snapshot"
                    )
                after_parameters = [
                    parameter.detach().clone() for parameter in trainable_parameters
                ]

                def set_interpolated_parameters(scale: float) -> None:
                    with torch.no_grad():
                        for parameter, before, after in zip(
                            trainable_parameters,
                            before_parameters,
                            after_parameters,
                            strict=True,
                        ):
                            parameter.copy_(before + scale * (after - before))

                projection_scale = 1.0
                if (
                    unprojected_statistics["max_abs_centered_log_eigenvalue"]
                    > request.maximum_spd_log_step_radius
                ):
                    lower = 0.0
                    upper = 1.0
                    step_statistics = {
                        "log_eigenvalue_span": 0.0,
                        "max_abs_centered_log_eigenvalue": 0.0,
                    }
                    for _ in range(request.spd_step_projection_bisections):
                        midpoint = 0.5 * (lower + upper)
                        set_interpolated_parameters(midpoint)
                        midpoint_statistics = spd_diagnostic_statistics(
                            before_spd_state,
                            current_spd_diagnostic_state_numpy(),
                        )
                        if (
                            midpoint_statistics["max_abs_centered_log_eigenvalue"]
                            <= request.maximum_spd_log_step_radius
                        ):
                            lower = midpoint
                            step_statistics = midpoint_statistics
                        else:
                            upper = midpoint
                    projection_scale = lower
                    set_interpolated_parameters(projection_scale)
                    interval_projected_spd_steps += 1
                interval_spd_step_projection_scales.append(projection_scale)
            interval_spd_step_log_spans.append(step_statistics["log_eigenvalue_span"])
            interval_spd_step_log_radii.append(
                step_statistics["max_abs_centered_log_eigenvalue"]
            )

    def optimizer_history_fields() -> dict[str, float | int | None]:
        return {
            "optimizer_step_count": optimizer_step_count,
            "optimizer_steps_since_previous_evaluation": (
                interval_optimizer_step_count
            ),
            "mean_preclip_gradient_norm_since_previous_evaluation": (
                float(np.mean(interval_preclip_gradient_norms))
                if interval_preclip_gradient_norms
                else None
            ),
            "maximum_preclip_gradient_norm_since_previous_evaluation": (
                float(np.max(interval_preclip_gradient_norms))
                if interval_preclip_gradient_norms
                else None
            ),
            "clipped_optimizer_steps_since_previous_evaluation": (
                interval_clipped_optimizer_steps
                if request.gradient_clip_norm is not None
                else None
            ),
            "recorded_spd_steps_since_previous_evaluation": len(
                interval_spd_step_log_spans
            ),
            "mean_spd_log_step_span_since_previous_evaluation": (
                float(np.mean(interval_spd_step_log_spans))
                if interval_spd_step_log_spans
                else None
            ),
            "maximum_spd_log_step_span_since_previous_evaluation": (
                float(np.max(interval_spd_step_log_spans))
                if interval_spd_step_log_spans
                else None
            ),
            "maximum_spd_log_step_radius_since_previous_evaluation": (
                float(np.max(interval_spd_step_log_radii))
                if interval_spd_step_log_radii
                else None
            ),
            "maximum_unprojected_spd_log_step_radius_since_previous_evaluation": (
                float(np.max(interval_unprojected_spd_step_log_radii))
                if interval_unprojected_spd_step_log_radii
                else None
            ),
            "projected_spd_steps_since_previous_evaluation": (
                interval_projected_spd_steps
                if request.maximum_spd_log_step_radius is not None
                else None
            ),
            "minimum_spd_step_projection_scale_since_previous_evaluation": (
                float(np.min(interval_spd_step_projection_scales))
                if interval_spd_step_projection_scales
                else None
            ),
        }

    def reset_optimizer_history_interval() -> None:
        nonlocal interval_optimizer_step_count
        nonlocal interval_clipped_optimizer_steps
        nonlocal interval_projected_spd_steps
        interval_optimizer_step_count = 0
        interval_clipped_optimizer_steps = 0
        interval_projected_spd_steps = 0
        interval_preclip_gradient_norms.clear()
        interval_spd_step_log_spans.clear()
        interval_spd_step_log_radii.clear()
        interval_unprojected_spd_step_log_radii.clear()
        interval_spd_step_projection_scales.clear()

    torch_cache: dict[int, tuple[Any, Any, Any, Any]] = {}
    streamed_training_group_ids = (
        {id(dataset) for _, dataset in train_groups}
        if request.stream_training_groups
        else set()
    )
    if request.stream_training_groups and active_set is not None:
        streamed_training_group_ids.add(id(active_set))

    def torch_tensors(dataset: dict[str, Any]):
        return (
            torch.tensor(dataset["values"], dtype=complex_dtype, device=device),
            torch.tensor(dataset["derivatives"], dtype=complex_dtype, device=device),
            torch.tensor(dataset["log_omega"], dtype=real_dtype, device=device),
            torch.tensor(
                dataset["importance_weights"], dtype=real_dtype, device=device
            ),
        )

    def as_torch(dataset: dict[str, Any]):
        key = id(dataset)
        if key in streamed_training_group_ids:
            return torch_tensors(dataset)
        if key not in torch_cache:
            torch_cache[key] = torch_tensors(dataset)
        return torch_cache[key]

    def raw_and_barrier_tensors(
        values,
        derivatives,
        log_omega,
        importance_weights,
        h_matrix,
    ):
        h_values = torch.einsum("ab,nb->na", h_matrix, values)
        denominator = torch.real(torch.einsum("na,na->n", torch.conj(values), h_values))
        h_derivatives = torch.einsum("ab,nbj->naj", h_matrix, derivatives)
        first = torch.einsum("nmi,nmj->nij", torch.conj(derivatives), h_derivatives)
        gradient = torch.einsum("nm,nmj->nj", torch.conj(values), h_derivatives)
        metric = first / denominator[:, None, None]
        metric -= (
            torch.conj(gradient)[:, :, None]
            * gradient[:, None, :]
            / denominator[:, None, None] ** 2
        )
        metric = (
            normalization * 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))
        )
        eigenvalues = torch.linalg.eigvalsh(metric)
        raw = torch.log(torch.clamp(eigenvalues, min=1e-12)).sum(dim=1) - log_omega
        scale = torch.clamp(
            torch.mean(torch.abs(eigenvalues), dim=1, keepdim=True), min=1e-14
        )
        relative_eigenvalues = eigenvalues / scale
        barrier = (
            torch.nn.functional.softplus((1e-8 - relative_eigenvalues) * 80.0).mean()
            / 80.0
        )
        weights = (
            importance_weights
            if request.importance_weighted
            else torch.ones_like(importance_weights)
        )
        return raw, barrier, weights

    def raw_and_barrier(dataset: dict[str, Any], h_matrix):
        return raw_and_barrier_tensors(*as_torch(dataset), h_matrix)

    training_cluster_members: list[list[np.ndarray]] = []
    training_point_cluster_indices: list[np.ndarray] = []
    training_cluster_count = 0
    training_point_count = sum(len(dataset["log_omega"]) for _, dataset in train_groups)
    training_sampling_cluster_count = sum(
        len(np.unique(dataset["sampling_cluster_ids"]))
        for _, dataset in train_groups
    )
    if request.group_optimizer_step_mode == "shuffled_cluster_minibatch":
        for name, dataset in train_groups:
            cluster_ids = np.asarray(dataset["sampling_cluster_ids"], dtype=np.int64)
            members = [
                np.flatnonzero(cluster_ids == cluster_id)
                for cluster_id in np.unique(cluster_ids)
            ]
            cluster_sizes = np.asarray([len(indices) for indices in members])
            if np.any(cluster_sizes != request.sampling_cluster_size):
                raise ValueError(
                    f"training group {name} contains incomplete sampling clusters: "
                    f"observed sizes={sorted(set(cluster_sizes.tolist()))}, "
                    f"expected={request.sampling_cluster_size}"
                )
            training_cluster_members.append(members)
            point_cluster_indices = np.empty(len(cluster_ids), dtype=np.int64)
            for member_index, indices in enumerate(members):
                point_cluster_indices[indices] = member_index
            training_point_cluster_indices.append(point_cluster_indices)
            training_cluster_count += len(members)
        print(
            "prepared cluster mini-batches: "
            f"clusters={training_cluster_count}, "
            f"clusters_per_step={request.clusters_per_optimizer_step}, "
            f"nominal_points_per_step="
            f"{request.clusters_per_optimizer_step * request.sampling_cluster_size}",
            flush=True,
        )
    elif request.group_optimizer_step_mode == "shuffled_point_minibatch":
        print(
            "prepared point mini-batches: "
            f"points={training_point_count}, "
            f"points_per_step={request.points_per_optimizer_step}",
            flush=True,
        )

    def weighted_mean(values, weights):
        return torch.sum(weights * values) / torch.sum(weights)

    def log_weighted_mean_exp(values, weights):
        return weighted_log_mean_exp_torch(values, weights)

    fixed_group_log_normalization = None
    if request.group_normalization_mode == "fixed_initial_training_pool":
        with torch.no_grad():
            log_numerators = []
            log_denominators = []
            for _, dataset in train_groups:
                initial_raw, _, initial_weights = raw_and_barrier(
                    dataset,
                    initial_h_t,
                )
                positive_weights = torch.clamp(
                    initial_weights,
                    min=torch.finfo(initial_weights.dtype).tiny,
                )
                log_numerators.append(
                    torch.logsumexp(
                        torch.log(positive_weights) + initial_raw,
                        dim=0,
                    )
                )
                log_denominators.append(torch.log(torch.sum(positive_weights)))
            fixed_group_log_normalization = (
                torch.logsumexp(torch.stack(log_numerators), dim=0)
                - torch.logsumexp(torch.stack(log_denominators), dim=0)
            ).detach()
        print(
            "fixed initial-training-pool log(kappa)="
            f"{float(fixed_group_log_normalization.cpu()):.12e}",
            flush=True,
        )

    def objective_log_normalization(raw, weights):
        if fixed_group_log_normalization is not None:
            return fixed_group_log_normalization
        return log_weighted_mean_exp(raw, weights)

    def normalized_sigma(raw, weights, *, clamp_exponent: bool):
        log_mean_ratio = objective_log_normalization(raw, weights)
        centered = raw - log_mean_ratio
        if clamp_exponent:
            centered = torch.clamp(centered, min=-30.0, max=30.0)
        normalized_ratio = torch.exp(centered)
        return weighted_mean(torch.abs(1.0 - normalized_ratio), weights)

    def normalized_volume_ratio_l2(raw, weights, *, clamp_exponent: bool):
        log_mean_ratio = objective_log_normalization(raw, weights)
        centered = raw - log_mean_ratio
        if clamp_exponent:
            # This guard is retained only for reproducing the pre-v4 pooled objective.
            centered = torch.clamp(centered, min=-20.0, max=20.0)
        normalized_ratio = torch.exp(centered)
        mean_square = weighted_mean((1.0 - normalized_ratio) ** 2, weights)
        return torch.sqrt(mean_square + torch.finfo(weights.dtype).eps)

    def weighted_cvar(values, weights, *, tail_fraction: float):
        return weighted_cvar_torch(
            values,
            weights,
            tail_fraction=tail_fraction,
        )

    def normalized_volume_ratio_cvar(raw, weights, *, tail_fraction: float):
        """Weighted empirical CVaR of |1 - eta| over the upper tail."""
        log_mean_ratio = objective_log_normalization(raw, weights)
        absolute_error = torch.abs(1.0 - torch.exp(raw - log_mean_ratio))
        return weighted_cvar(absolute_error, weights, tail_fraction=tail_fraction)

    def normalized_positive_log_ratio_cvar(raw, weights, *, tail_fraction: float):
        """Weighted CVaR of max(log(eta / mean(eta)), 0)."""
        log_mean_ratio = objective_log_normalization(raw, weights)
        positive_log_ratio = torch.relu(raw - log_mean_ratio)
        return weighted_cvar(
            positive_log_ratio,
            weights,
            tail_fraction=tail_fraction,
        )

    def group_objective_components(dataset: dict[str, Any], h_matrix):
        raw, barrier, weights = raw_and_barrier(dataset, h_matrix)
        empirical_log_mean_ratio = log_weighted_mean_exp(raw, weights)
        log_center = (
            objective_log_normalization(raw, weights)
            if fixed_group_log_normalization is not None
            else weighted_mean(raw, weights)
        )
        centered_log_variance = weighted_mean((raw - log_center) ** 2, weights)
        sigma = normalized_sigma(raw, weights, clamp_exponent=False)
        volume_ratio_l2 = normalized_volume_ratio_l2(
            raw,
            weights,
            clamp_exponent=False,
        )
        if request.group_volume_ratio_cvar_loss_weight:
            volume_ratio_cvar = normalized_volume_ratio_cvar(
                raw,
                weights,
                tail_fraction=request.group_volume_ratio_cvar_tail_fraction,
            )
        else:
            volume_ratio_cvar = torch.zeros((), dtype=raw.dtype, device=raw.device)
        if request.group_positive_log_ratio_cvar_loss_weight:
            positive_log_ratio_cvar = normalized_positive_log_ratio_cvar(
                raw,
                weights,
                tail_fraction=request.group_positive_log_ratio_cvar_tail_fraction,
            )
        else:
            positive_log_ratio_cvar = torch.zeros(
                (), dtype=raw.dtype, device=raw.device
            )
        positive_weights = torch.clamp(
            weights,
            min=torch.finfo(weights.dtype).tiny,
        )
        if request.global_volume_ratio_cvar_loss_weight:
            global_volume_ratio_losses = torch.abs(
                1.0 - torch.exp(raw - fixed_group_log_normalization)
            )
        else:
            global_volume_ratio_losses = None
        if request.global_upper_log_ratio_cvar_loss_weight:
            global_upper_log_ratio_losses = smooth_upper_log_ratio_excess_torch(
                raw - fixed_group_log_normalization,
                ratio_threshold=request.global_upper_log_ratio_threshold,
                smooth_temperature=(
                    request.global_upper_log_ratio_smooth_temperature
                ),
            )
        else:
            global_upper_log_ratio_losses = None
        base = (
            request.centered_log_variance_loss_weight * centered_log_variance
            + request.sigma_loss_weight * sigma
        )
        return {
            "base": base,
            "centered_log_variance": centered_log_variance,
            "sigma": sigma,
            "volume_ratio_l2": volume_ratio_l2,
            "volume_ratio_cvar": volume_ratio_cvar,
            "positive_log_ratio_cvar": positive_log_ratio_cvar,
            "log_mean_ratio": empirical_log_mean_ratio,
            "log_weight_sum": torch.log(torch.sum(positive_weights)),
            "global_volume_ratio_losses": global_volume_ratio_losses,
            "global_upper_log_ratio_losses": global_upper_log_ratio_losses,
            "global_volume_ratio_weights": positive_weights,
            "barrier": barrier,
        }

    active_region_index_tensors = [
        torch.tensor(indices, dtype=torch.int64, device=device)
        for indices in active_region_indices
    ]

    def active_set_objective(h_matrix, reference_log_normalization):
        if active_set is None:
            zero = torch.zeros((), dtype=real_dtype, device=device)
            return zero, zero, zero, zero
        raw, barrier, _ = raw_and_barrier(active_set, h_matrix)
        temperature = request.active_set_smooth_max_temperature
        excess = (
            torch.nn.functional.softplus(
                (
                    raw
                    - reference_log_normalization
                    - request.active_set_log_ratio_threshold
                )
                / temperature
            )
            * temperature
        )
        if request.active_set_region_balanced:
            region_risks = torch.stack(
                [
                    temperature
                    * (
                        torch.logsumexp(excess[indices] / temperature, dim=0)
                        - np.log(float(len(indices)))
                    )
                    for indices in active_region_index_tensors
                ]
            )
            smooth_max = temperature * (
                torch.logsumexp(region_risks / temperature, dim=0)
                - np.log(float(len(region_risks)))
            )
            maximum_region_risk = torch.max(region_risks)
        else:
            smooth_max = temperature * (
                torch.logsumexp(excess / temperature, dim=0)
                - np.log(float(len(excess)))
            )
            maximum_region_risk = torch.zeros_like(smooth_max)
        maximum_log_ratio = torch.max(raw - reference_log_normalization)
        return smooth_max, barrier, maximum_log_ratio, maximum_region_risk

    def as_hermitian_numpy(h_matrix: np.ndarray) -> np.ndarray:
        matrix = positive_hermitian_projection(h_matrix)
        return matrix * (len(matrix) / float(np.trace(matrix).real))

    def evaluate(dataset: dict[str, Any], h_matrix: np.ndarray) -> tuple[dict, dict]:
        baseline = _metric_stats(
            adapter,
            dataset["points"],
            dataset["baseline_metrics"],
            dataset["importance_weights"],
            weighted=request.importance_weighted,
        )
        artifact = HMetricArtifact(
            path=Path("<in-memory>"),
            degree=request.degree,
            section_exponents=exponents,
            h_matrix=as_hermitian_numpy(h_matrix),
            normalization=normalization,
        )
        candidate_metrics = adapter.h_metrics(dataset["points"], artifact)
        candidate = _metric_stats(
            adapter,
            dataset["points"],
            candidate_metrics,
            dataset["importance_weights"],
            weighted=request.importance_weighted,
        )
        return baseline, candidate

    def selection_row_gate(baseline: dict, candidate: dict) -> bool:
        metric_gate_passed = _stats_gate(baseline, candidate)
        if request.groupwise_objective:
            sigma_gate_passed = bool(
                candidate["sigma"] <= request.maximum_checkpoint_sigma
            )
            l2_gate_passed = bool(
                candidate["sqrt_squared_energy"]
                <= request.maximum_checkpoint_volume_ratio_l2
            )
            q999_gate_passed = bool(
                request.maximum_checkpoint_positive_log_ratio_q999 is None
                or candidate["positive_log_ratio_q999"]
                <= request.maximum_checkpoint_positive_log_ratio_q999
            )
            cvar_gate_passed = bool(
                request.maximum_checkpoint_positive_log_ratio_cvar is None
                or candidate["positive_log_ratio_cvar_1pct"]
                <= request.maximum_checkpoint_positive_log_ratio_cvar
            )
            failure_mass_gate_passed = bool(
                request.maximum_checkpoint_ratio_above_3_weighted_mass is None
                or candidate["normalized_ratio_above_3_weighted_mass"]
                <= request.maximum_checkpoint_ratio_above_3_weighted_mass
            )
        else:
            sigma_gate_passed = True
            l2_gate_passed = bool(
                request.maximum_check_volume_ratio_l2 is None
                or candidate["sqrt_squared_energy"]
                <= request.maximum_check_volume_ratio_l2
            )
            q999_gate_passed = True
            cvar_gate_passed = True
            failure_mass_gate_passed = True
        return bool(
            metric_gate_passed
            and sigma_gate_passed
            and l2_gate_passed
            and q999_gate_passed
            and cvar_gate_passed
            and failure_mass_gate_passed
        )

    def probe_group_objective() -> tuple[list[dict[str, float]], Any, float]:
        if not request.groupwise_objective:
            return [], None, float("nan")
        with torch.no_grad():
            probe_h = current_h_matrix()
            components = [
                group_objective_components(dataset, probe_h)
                for _, dataset in train_groups
            ]
            chi_values = torch.stack(
                [component["volume_ratio_l2"] for component in components]
            )
            temperature = request.group_volume_ratio_l2_smooth_max_temperature
            chi_smooth_weights = torch.softmax(chi_values / temperature, dim=0)
            mean_fraction = request.group_volume_ratio_l2_mean_fraction
            risk = mean_fraction * torch.mean(chi_values) + (1.0 - mean_fraction) * (
                temperature * torch.logsumexp(chi_values / temperature, dim=0)
            )
            cvar_values = torch.stack(
                [component["volume_ratio_cvar"] for component in components]
            )
            cvar_smooth_weights = torch.softmax(cvar_values / temperature, dim=0)
            cvar_risk = mean_fraction * torch.mean(cvar_values) + (
                1.0 - mean_fraction
            ) * (temperature * torch.logsumexp(cvar_values / temperature, dim=0))
            positive_log_cvar_values = torch.stack(
                [component["positive_log_ratio_cvar"] for component in components]
            )
            positive_log_cvar_smooth_weights = torch.softmax(
                positive_log_cvar_values / temperature,
                dim=0,
            )
            positive_log_cvar_risk = mean_fraction * torch.mean(
                positive_log_cvar_values
            ) + (1.0 - mean_fraction) * (
                temperature
                * torch.logsumexp(positive_log_cvar_values / temperature, dim=0)
            )
            def probe_global_cvar(
                loss_key: str,
                tail_fractions: tuple[float, ...],
                tail_weights: tuple[float, ...],
            ) -> tuple[float, list[dict[str, float]], list[list[Any]], float]:
                global_losses = np.asarray(
                    np.concatenate(
                        [
                            component[loss_key].detach().cpu().numpy()
                            for component in components
                        ]
                    ),
                    dtype=np.float64,
                )
                global_weights = np.asarray(
                    np.concatenate(
                        [
                            component["global_volume_ratio_weights"]
                            .detach()
                            .cpu()
                            .numpy()
                            for component in components
                        ]
                    ),
                    dtype=np.float64,
                )
                configured_mixture_weights = np.asarray(
                    tail_weights,
                    dtype=np.float64,
                )
                mixture_weights = configured_mixture_weights / np.sum(
                    configured_mixture_weights
                )
                global_cvar_weight_sum = float(np.sum(global_weights))
                global_cvar_risk_value = 0.0
                global_cvar_audits: list[dict[str, float]] = []
                global_cvar_group_selections: list[list[Any]] = []
                for tail_fraction, mixture_weight in zip(
                    tail_fractions,
                    mixture_weights,
                    strict=True,
                ):
                    audit = weighted_cvar_numpy(
                        global_losses,
                        global_weights,
                        tail_fraction=tail_fraction,
                    )
                    boundary_selection_fraction = (
                        audit.selected_boundary_mass / audit.boundary_mass
                    )
                    selected_fractions = (global_losses > audit.var_threshold).astype(
                        np.float64
                    ) + boundary_selection_fraction * (
                        global_losses == audit.var_threshold
                    ).astype(
                        np.float64
                    )
                    selected_mass = float(
                        np.sum(global_weights * selected_fractions)
                        / global_cvar_weight_sum
                    )
                    if not np.isclose(
                        selected_mass,
                        tail_fraction,
                        rtol=0.0,
                        atol=5.0e-7,
                    ):
                        raise RuntimeError(
                            "global CVaR tail selection does not have the "
                            "requested weighted mass"
                        )
                    offset = 0
                    group_selections = []
                    for component in components:
                        count = len(component[loss_key])
                        group_selections.append(
                            torch.tensor(
                                selected_fractions[offset : offset + count],
                                dtype=real_dtype,
                                device=device,
                            )
                        )
                        offset += count
                    if offset != len(selected_fractions):
                        raise RuntimeError("global CVaR group split lost samples")
                    global_cvar_group_selections.append(group_selections)
                    global_cvar_risk_value += mixture_weight * audit.value
                    global_cvar_audits.append(
                        {
                            "tail_fraction": float(tail_fraction),
                            "mixture_weight": float(mixture_weight),
                            "value": audit.value,
                            "var_threshold": audit.var_threshold,
                            "selected_tail_mass": audit.selected_tail_mass,
                            "strict_tail_mass": audit.strict_tail_mass,
                            "boundary_mass": audit.boundary_mass,
                            "selected_boundary_mass": audit.selected_boundary_mass,
                            "boundary_selection_fraction": (
                                boundary_selection_fraction
                            ),
                        }
                    )
                return (
                    global_cvar_risk_value,
                    global_cvar_audits,
                    global_cvar_group_selections,
                    global_cvar_weight_sum,
                )

            global_cvar_audits: list[dict[str, float]] = []
            global_cvar_group_selections: list[list[Any]] = []
            global_cvar_risk_value = 0.0
            global_cvar_weight_sum = 0.0
            if request.global_volume_ratio_cvar_loss_weight:
                (
                    global_cvar_risk_value,
                    global_cvar_audits,
                    global_cvar_group_selections,
                    global_cvar_weight_sum,
                ) = probe_global_cvar(
                    "global_volume_ratio_losses",
                    request.global_volume_ratio_cvar_tail_fractions,
                    request.global_volume_ratio_cvar_tail_weights,
                )
            global_upper_log_cvar_audits: list[dict[str, float]] = []
            global_upper_log_cvar_group_selections: list[list[Any]] = []
            global_upper_log_cvar_risk_value = 0.0
            global_upper_log_cvar_weight_sum = 0.0
            if request.global_upper_log_ratio_cvar_loss_weight:
                (
                    global_upper_log_cvar_risk_value,
                    global_upper_log_cvar_audits,
                    global_upper_log_cvar_group_selections,
                    global_upper_log_cvar_weight_sum,
                ) = probe_global_cvar(
                    "global_upper_log_ratio_losses",
                    request.global_upper_log_ratio_cvar_tail_fractions,
                    request.global_upper_log_ratio_cvar_tail_weights,
                )
            log_numerators = torch.stack(
                [
                    component["log_mean_ratio"] + component["log_weight_sum"]
                    for component in components
                ]
            )
            log_denominators = torch.stack(
                [component["log_weight_sum"] for component in components]
            )
            reference_log_normalization = (
                fixed_group_log_normalization
                if fixed_group_log_normalization is not None
                else torch.logsumexp(log_numerators, dim=0)
                - torch.logsumexp(log_denominators, dim=0)
            )
            (
                active_risk,
                active_barrier,
                active_maximum_log_ratio,
                active_maximum_region_risk,
            ) = active_set_objective(probe_h, reference_log_normalization)
            base_mean = torch.mean(
                torch.stack([component["base"] for component in components])
            )
            barrier_mean = torch.mean(
                torch.stack([component["barrier"] for component in components])
            )
            drift = drift_penalty(probe_h)
            objective = (
                base_mean
                + request.group_volume_ratio_l2_loss_weight * risk
                + request.group_volume_ratio_cvar_loss_weight * cvar_risk
                + request.group_positive_log_ratio_cvar_loss_weight
                * positive_log_cvar_risk
                + request.global_volume_ratio_cvar_loss_weight * global_cvar_risk_value
                + request.global_upper_log_ratio_cvar_loss_weight
                * global_upper_log_cvar_risk_value
                + request.active_set_loss_weight * active_risk
                + request.metric_barrier_weight * barrier_mean
                + request.metric_barrier_weight * active_barrier
                + request.drift_weight * drift
                + request.low_rank_factor_weight * parameterization_penalty()
                + request.relative_log_spectrum_loss_weight
                * relative_log_spectrum_penalty()
            )
            rows = []
            for (name, _), component in zip(train_groups, components, strict=True):
                rows.append(
                    {
                        "name": name,
                        "base": float(component["base"].cpu()),
                        "centered_log_variance": float(
                            component["centered_log_variance"].cpu()
                        ),
                        "sigma": float(component["sigma"].cpu()),
                        "volume_ratio_l2": float(component["volume_ratio_l2"].cpu()),
                        "volume_ratio_cvar": float(
                            component["volume_ratio_cvar"].cpu()
                        ),
                        "positive_log_ratio_cvar": float(
                            component["positive_log_ratio_cvar"].cpu()
                        ),
                        "barrier": float(component["barrier"].cpu()),
                    }
                )
        return (
            rows,
            {
                "volume_ratio_l2": chi_smooth_weights,
                "volume_ratio_cvar": cvar_smooth_weights,
                "positive_log_ratio_cvar": positive_log_cvar_smooth_weights,
                "global_volume_ratio_cvar_risk": global_cvar_risk_value,
                "global_volume_ratio_cvar_audits": global_cvar_audits,
                "global_volume_ratio_cvar_group_selections": (
                    global_cvar_group_selections
                ),
                "global_volume_ratio_weight_sum": global_cvar_weight_sum,
                "global_upper_log_ratio_cvar_risk": (
                    global_upper_log_cvar_risk_value
                ),
                "global_upper_log_ratio_cvar_audits": (
                    global_upper_log_cvar_audits
                ),
                "global_upper_log_ratio_cvar_group_selections": (
                    global_upper_log_cvar_group_selections
                ),
                "global_upper_log_ratio_weight_sum": (
                    global_upper_log_cvar_weight_sum
                ),
                "reference_log_normalization": reference_log_normalization,
                "active_set_smooth_max": active_risk,
                "active_set_maximum_log_ratio": active_maximum_log_ratio,
                "active_set_maximum_region_risk": active_maximum_region_risk,
            },
            float(objective.cpu()),
        )

    def global_volume_ratio_cvar_gradient_term(
        component: dict[str, Any],
        risk_data: dict[str, Any],
        *,
        group_index: int,
        group_multiplier: float,
    ):
        if not request.global_volume_ratio_cvar_loss_weight:
            return torch.zeros((), dtype=real_dtype, device=device)
        losses = component["global_volume_ratio_losses"]
        weights = component["global_volume_ratio_weights"]
        weight_sum = risk_data["global_volume_ratio_weight_sum"]
        risk_gradient_term = torch.zeros((), dtype=real_dtype, device=device)
        for tail_index, audit in enumerate(
            risk_data["global_volume_ratio_cvar_audits"]
        ):
            selected_fractions = risk_data["global_volume_ratio_cvar_group_selections"][
                tail_index
            ][group_index]
            risk_gradient_term = risk_gradient_term + (
                audit["mixture_weight"]
                * weighted_cvar_selected_linearization_torch(
                    losses,
                    weights,
                    selected_fractions,
                    tail_fraction=audit["tail_fraction"],
                    total_weight=weight_sum,
                )
            )
        return (
            request.global_volume_ratio_cvar_loss_weight
            * group_multiplier
            * risk_gradient_term
        )

    def global_upper_log_ratio_cvar_gradient_term(
        component: dict[str, Any],
        risk_data: dict[str, Any],
        *,
        group_index: int,
        group_multiplier: float,
    ):
        if not request.global_upper_log_ratio_cvar_loss_weight:
            return torch.zeros((), dtype=real_dtype, device=device)
        losses = component["global_upper_log_ratio_losses"]
        weights = component["global_volume_ratio_weights"]
        weight_sum = risk_data["global_upper_log_ratio_weight_sum"]
        risk_gradient_term = torch.zeros((), dtype=real_dtype, device=device)
        for tail_index, audit in enumerate(
            risk_data["global_upper_log_ratio_cvar_audits"]
        ):
            selected_fractions = risk_data[
                "global_upper_log_ratio_cvar_group_selections"
            ][tail_index][group_index]
            risk_gradient_term = risk_gradient_term + (
                audit["mixture_weight"]
                * weighted_cvar_selected_linearization_torch(
                    losses,
                    weights,
                    selected_fractions,
                    tail_fraction=audit["tail_fraction"],
                    total_weight=weight_sum,
                )
            )
        return (
            request.global_upper_log_ratio_cvar_loss_weight
            * group_multiplier
            * risk_gradient_term
        )

    def group_history_fields(
        rows: list[dict[str, float]],
        risk_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not rows:
            return {
                "mean_training_group_sigma": None,
                "maximum_training_group_sigma": None,
                "mean_training_group_volume_ratio_l2": None,
                "maximum_training_group_volume_ratio_l2": None,
                "mean_training_group_volume_ratio_cvar": None,
                "maximum_training_group_volume_ratio_cvar": None,
                "mean_training_group_positive_log_ratio_cvar": None,
                "maximum_training_group_positive_log_ratio_cvar": None,
                "global_training_volume_ratio_cvar_risk": None,
                "global_training_volume_ratio_cvar_tail_audits": None,
                "global_training_upper_log_ratio_cvar_risk": None,
                "global_training_upper_log_ratio_cvar_tail_audits": None,
                "group_l2_risk_maximum_weight": None,
                "group_l2_risk_effective_group_count": None,
                "group_cvar_risk_maximum_weight": None,
                "group_cvar_risk_effective_group_count": None,
                "group_positive_log_cvar_risk_maximum_weight": None,
                "group_positive_log_cvar_risk_effective_group_count": None,
                "active_set_smooth_max": None,
                "active_set_maximum_log_ratio": None,
                "active_set_maximum_region_risk": None,
            }

        def group_risk_weight_fields(
            key: str,
            prefix: str,
        ) -> dict[str, float | None]:
            if risk_data is None:
                return {
                    f"{prefix}_maximum_weight": None,
                    f"{prefix}_effective_group_count": None,
                }
            weights = np.asarray(
                risk_data[key].detach().cpu().numpy(),
                dtype=np.float64,
            )
            return {
                f"{prefix}_maximum_weight": float(np.max(weights)),
                f"{prefix}_effective_group_count": float(1.0 / np.sum(weights**2)),
            }

        return {
            "mean_training_group_sigma": float(np.mean([row["sigma"] for row in rows])),
            "maximum_training_group_sigma": float(
                np.max([row["sigma"] for row in rows])
            ),
            "mean_training_group_volume_ratio_l2": float(
                np.mean([row["volume_ratio_l2"] for row in rows])
            ),
            "maximum_training_group_volume_ratio_l2": float(
                np.max([row["volume_ratio_l2"] for row in rows])
            ),
            "mean_training_group_volume_ratio_cvar": float(
                np.mean([row["volume_ratio_cvar"] for row in rows])
            ),
            "maximum_training_group_volume_ratio_cvar": float(
                np.max([row["volume_ratio_cvar"] for row in rows])
            ),
            "mean_training_group_positive_log_ratio_cvar": float(
                np.mean([row["positive_log_ratio_cvar"] for row in rows])
            ),
            "maximum_training_group_positive_log_ratio_cvar": float(
                np.max([row["positive_log_ratio_cvar"] for row in rows])
            ),
            "global_training_volume_ratio_cvar_risk": (
                float(risk_data["global_volume_ratio_cvar_risk"])
                if risk_data is not None
                and request.global_volume_ratio_cvar_loss_weight
                else None
            ),
            "global_training_volume_ratio_cvar_tail_audits": (
                risk_data["global_volume_ratio_cvar_audits"]
                if risk_data is not None
                and request.global_volume_ratio_cvar_loss_weight
                else None
            ),
            "global_training_upper_log_ratio_cvar_risk": (
                float(risk_data["global_upper_log_ratio_cvar_risk"])
                if risk_data is not None
                and request.global_upper_log_ratio_cvar_loss_weight
                else None
            ),
            "global_training_upper_log_ratio_cvar_tail_audits": (
                risk_data["global_upper_log_ratio_cvar_audits"]
                if risk_data is not None
                and request.global_upper_log_ratio_cvar_loss_weight
                else None
            ),
            **group_risk_weight_fields("volume_ratio_l2", "group_l2_risk"),
            **group_risk_weight_fields("volume_ratio_cvar", "group_cvar_risk"),
            **group_risk_weight_fields(
                "positive_log_ratio_cvar",
                "group_positive_log_cvar_risk",
            ),
            "active_set_smooth_max": (
                float(risk_data["active_set_smooth_max"].cpu())
                if risk_data is not None and active_set is not None
                else None
            ),
            "active_set_maximum_log_ratio": (
                float(risk_data["active_set_maximum_log_ratio"].cpu())
                if risk_data is not None and active_set is not None
                else None
            ),
            "active_set_maximum_region_risk": (
                float(risk_data["active_set_maximum_region_risk"].cpu())
                if risk_data is not None
                and active_set is not None
                and request.active_set_region_balanced
                else None
            ),
        }

    def relative_to_initial_history_fields(_h_matrix: np.ndarray) -> dict[str, float]:
        statistics = spd_diagnostic_statistics(
            initial_spd_diagnostic_state,
            current_spd_diagnostic_state_numpy(),
        )
        return {
            "relative_to_initial_log_eigenvalue_span": statistics[
                "log_eigenvalue_span"
            ],
            "relative_to_initial_max_abs_centered_log_eigenvalue": statistics[
                "max_abs_centered_log_eigenvalue"
            ],
        }

    initial_rows = []
    for name, dataset in selection_sets:
        baseline, candidate = evaluate(dataset, initial_h)
        initial_rows.append(
            (name, baseline, candidate, selection_row_gate(baseline, candidate))
        )
    best_h = as_hermitian_numpy(initial_h)

    def selection_statistics(
        rows: list[tuple],
    ) -> tuple[float, float, float, float, float, float, float, float]:
        mean_log_rms = float(
            np.mean([row[2]["weighted_centered_log_ma_rms"] for row in rows])
        )
        mean_sigma = float(np.mean([row[2]["sigma"] for row in rows]))
        mean_volume_ratio_l2 = float(
            np.mean([row[2]["sqrt_squared_energy"] for row in rows])
        )
        maximum_volume_ratio_l2 = float(
            np.max([row[2]["sqrt_squared_energy"] for row in rows])
        )
        maximum_positive_log_ratio_q999 = float(
            np.max([row[2]["positive_log_ratio_q999"] for row in rows])
        )
        maximum_positive_log_ratio_cvar = float(
            np.max([row[2]["positive_log_ratio_cvar_1pct"] for row in rows])
        )
        maximum_ratio_above_3_weighted_mass = float(
            np.max([row[2]["normalized_ratio_above_3_weighted_mass"] for row in rows])
        )
        return (
            mean_log_rms
            + request.selection_sigma_weight * mean_sigma
            + request.selection_volume_ratio_l2_weight * mean_volume_ratio_l2
            + request.selection_volume_ratio_l2_max_weight * maximum_volume_ratio_l2
            + request.selection_positive_log_ratio_q999_weight
            * maximum_positive_log_ratio_q999
            + request.selection_positive_log_ratio_cvar_weight
            * maximum_positive_log_ratio_cvar
            + request.selection_ratio_above_3_weighted_mass_weight
            * maximum_ratio_above_3_weighted_mass,
            mean_log_rms,
            mean_sigma,
            mean_volume_ratio_l2,
            maximum_volume_ratio_l2,
            maximum_positive_log_ratio_q999,
            maximum_positive_log_ratio_cvar,
            maximum_ratio_above_3_weighted_mass,
        )

    (
        initial_score,
        initial_mean_log_rms,
        initial_mean_sigma,
        initial_mean_volume_ratio_l2,
        initial_maximum_volume_ratio_l2,
        initial_maximum_positive_log_ratio_q999,
        initial_maximum_positive_log_ratio_cvar,
        initial_maximum_ratio_above_3_weighted_mass,
    ) = selection_statistics(initial_rows)
    initial_all_passed = bool(all(row[3] for row in initial_rows))
    (
        initial_training_group_rows,
        initial_risk_data,
        initial_group_loss,
    ) = probe_group_objective()
    score_selection_policies = {"best_score", "best_score_exploratory"}
    best_passed = bool(
        initial_all_passed and request.checkpoint_policy in score_selection_policies
    )
    best_score = initial_score if best_passed else float("inf")
    best_epoch = 0
    selected_training_group_rows = initial_training_group_rows if best_passed else []
    consecutive_eligible = 0
    pending_h: np.ndarray | None = None
    pending_epoch: int | None = None
    pending_score: float | None = None
    pending_training_group_rows: list[dict[str, float]] = []
    pending_history_index: int | None = None
    if request.checkpoint_policy == "first_consecutive_eligible" and initial_all_passed:
        consecutive_eligible = 1
        pending_h = as_hermitian_numpy(initial_h)
        pending_epoch = 0
        pending_score = initial_score
        pending_training_group_rows = initial_training_group_rows
    history = [
        {
            "epoch": 0,
            "loss": initial_group_loss if request.groupwise_objective else None,
            "accepted": best_passed,
            "passed_internal_gates": initial_all_passed,
            "checkpoint_eligible": initial_all_passed,
            "checkpoint_confirmed": best_passed,
            "consecutive_eligible_evaluations": consecutive_eligible,
            "ineligible_checkpoint_action_taken": "none",
            "rolled_back_after_evaluation": False,
            "learning_rate_before_action": request.learning_rate,
            "learning_rate_after_action": request.learning_rate,
            "cumulative_ineligible_checkpoints": 0,
            "cumulative_checkpoint_rollbacks": 0,
            "selection_score": initial_score,
            "mean_check_log_rms": initial_mean_log_rms,
            "mean_check_sigma": initial_mean_sigma,
            "mean_check_volume_ratio_l2": initial_mean_volume_ratio_l2,
            "maximum_check_volume_ratio_l2": initial_maximum_volume_ratio_l2,
            "maximum_check_positive_log_ratio_q999": (
                initial_maximum_positive_log_ratio_q999
            ),
            "maximum_check_positive_log_ratio_cvar": (
                initial_maximum_positive_log_ratio_cvar
            ),
            "maximum_check_ratio_above_3_weighted_mass": (
                initial_maximum_ratio_above_3_weighted_mass
            ),
            **optimizer_history_fields(),
            **relative_to_initial_history_fields(initial_h),
            **group_history_fields(initial_training_group_rows, initial_risk_data),
        }
    ]
    reset_optimizer_history_interval()
    if pending_epoch == 0:
        pending_history_index = 0
    print(
        f"adapter={adapter.key}, initialization={initialization}, device={device}, "
        f"precision={request.precision}, sections={len(exponents)}, "
        f"mean_check_log_rms={initial_mean_log_rms:.6e}, "
        f"mean_check_sigma={initial_mean_sigma:.6e}, "
        f"mean_check_volume_ratio_l2={initial_mean_volume_ratio_l2:.6e}, "
        f"maximum_check_volume_ratio_l2={initial_maximum_volume_ratio_l2:.6e}, "
        f"maximum_check_positive_log_ratio_q999="
        f"{initial_maximum_positive_log_ratio_q999:.6e}, "
        f"maximum_check_positive_log_ratio_cvar="
        f"{initial_maximum_positive_log_ratio_cvar:.6e}, "
        f"maximum_check_ratio_above_3_weighted_mass="
        f"{initial_maximum_ratio_above_3_weighted_mass:.6e}, "
        f"selection_score={initial_score:.6e}, accepted={best_passed}",
        flush=True,
    )

    optimization_started = time.perf_counter()
    stale_evaluations = 0
    confirmation_epoch: int | None = None
    last_eligible_training_state = (
        snapshot_training_state() if initial_all_passed else None
    )
    termination_reason = "completed_requested_epochs"
    for epoch in range(1, request.epochs + 1):
        if request.groupwise_objective:
            _, risk_weights, loss_value = probe_group_objective()
            group_count = len(train_groups)
            mean_fraction = request.group_volume_ratio_l2_mean_fraction
            if request.group_optimizer_step_mode == "shuffled_cluster_minibatch":
                batch_descriptors: list[tuple[int, np.ndarray]] = []
                for group_index, members in enumerate(training_cluster_members):
                    cluster_order = group_order_rng.permutation(len(members))
                    for start in range(
                        0,
                        len(cluster_order),
                        request.clusters_per_optimizer_step,
                    ):
                        selected_clusters = cluster_order[
                            start : start + request.clusters_per_optimizer_step
                        ]
                        point_indices = np.concatenate(
                            [members[int(index)] for index in selected_clusters]
                        )
                        batch_descriptors.append((group_index, point_indices))
                if request.cluster_tail_replay_batches_per_epoch:
                    tail_cluster_pool: list[tuple[int, int]] = []
                    for tail_group_selections in risk_weights[
                        "global_volume_ratio_cvar_group_selections"
                    ]:
                        for group_index, selected_fractions in enumerate(
                            tail_group_selections
                        ):
                            selected_points = np.flatnonzero(
                                selected_fractions.detach().cpu().numpy() > 0
                            )
                            selected_clusters = np.unique(
                                training_point_cluster_indices[group_index][
                                    selected_points
                                ]
                            )
                            tail_cluster_pool.extend(
                                (group_index, int(cluster_index))
                                for cluster_index in selected_clusters
                            )
                    if not tail_cluster_pool:
                        raise RuntimeError(
                            "cluster tail replay found no CVaR-selected fibres"
                        )
                    tail_order = group_order_rng.permutation(len(tail_cluster_pool))
                    for replay_index in range(
                        request.cluster_tail_replay_batches_per_epoch
                    ):
                        group_index, target_cluster = tail_cluster_pool[
                            int(tail_order[replay_index % len(tail_order)])
                        ]
                        members = training_cluster_members[group_index]
                        fill_count = request.clusters_per_optimizer_step - 1
                        if fill_count:
                            available = np.delete(
                                np.arange(len(members), dtype=np.int64),
                                target_cluster,
                            )
                            fill_clusters = group_order_rng.choice(
                                available,
                                size=fill_count,
                                replace=fill_count > len(available),
                            )
                            selected_clusters = np.concatenate(
                                (
                                    np.asarray([target_cluster], dtype=np.int64),
                                    np.asarray(fill_clusters, dtype=np.int64),
                                )
                            )
                        else:
                            selected_clusters = np.asarray(
                                [target_cluster], dtype=np.int64
                            )
                        point_indices = np.concatenate(
                            [members[int(index)] for index in selected_clusters]
                        )
                        batch_descriptors.append((group_index, point_indices))
                batch_order = group_order_rng.permutation(len(batch_descriptors))
                batch_mean_losses = []
                for descriptor_value in batch_order:
                    group_index, point_indices = batch_descriptors[
                        int(descriptor_value)
                    ]
                    _, dataset = train_groups[group_index]
                    source_tensors = as_torch(dataset)
                    point_indices_t = torch.tensor(
                        point_indices,
                        dtype=torch.int64,
                        device=device,
                    )
                    batch_tensors = tuple(
                        values.index_select(0, point_indices_t)
                        for values in source_tensors
                    )
                    optimizer.zero_grad(set_to_none=True)
                    h_matrix = current_h_matrix()
                    raw, barrier, importance_weights = raw_and_barrier_tensors(
                        *batch_tensors,
                        h_matrix,
                    )
                    loss_weights = (
                        torch.ones_like(importance_weights)
                        if request.cluster_minibatch_uniform_loss
                        else importance_weights
                    )
                    loss_weights = loss_weights / torch.mean(loss_weights)
                    centered = raw - fixed_group_log_normalization
                    absolute_error = torch.abs(1.0 - torch.exp(centered))
                    weighted_point_losses = loss_weights * absolute_error
                    sigma_term = (
                        torch.sum(weighted_point_losses)
                        if request.cluster_minibatch_loss_reduction == "sum"
                        else torch.mean(weighted_point_losses)
                    )
                    centered_log_variance = weighted_mean(
                        torch.square(centered),
                        loss_weights,
                    )
                    group_loss = (
                        request.centered_log_variance_loss_weight
                        * centered_log_variance
                        + request.sigma_loss_weight * sigma_term
                        + request.metric_barrier_weight * barrier
                    )
                    global_tail_step_multiplier = float(len(batch_descriptors))
                    if request.cluster_minibatch_loss_reduction == "sum":
                        global_tail_step_multiplier *= len(point_indices)
                    if request.global_volume_ratio_cvar_loss_weight:
                        global_losses = torch.abs(
                            1.0 - torch.exp(raw - fixed_group_log_normalization)
                        )
                        global_cvar_term = torch.zeros(
                            (), dtype=real_dtype, device=device
                        )
                        for tail_index, audit in enumerate(
                            risk_weights["global_volume_ratio_cvar_audits"]
                        ):
                            selected_fractions = risk_weights[
                                "global_volume_ratio_cvar_group_selections"
                            ][tail_index][group_index].index_select(
                                0, point_indices_t
                            )
                            global_cvar_term = global_cvar_term + (
                                audit["mixture_weight"]
                                * weighted_cvar_selected_linearization_torch(
                                    global_losses,
                                    importance_weights,
                                    selected_fractions,
                                    tail_fraction=audit["tail_fraction"],
                                    total_weight=risk_weights[
                                        "global_volume_ratio_weight_sum"
                                    ],
                                )
                            )
                        group_loss = (
                            group_loss
                            + request.global_volume_ratio_cvar_loss_weight
                            * global_tail_step_multiplier
                            * global_cvar_term
                        )
                    if request.global_upper_log_ratio_cvar_loss_weight:
                        global_upper_log_losses = (
                            smooth_upper_log_ratio_excess_torch(
                                centered,
                                ratio_threshold=(
                                    request.global_upper_log_ratio_threshold
                                ),
                                smooth_temperature=(
                                    request.global_upper_log_ratio_smooth_temperature
                                ),
                            )
                        )
                        global_upper_log_cvar_term = torch.zeros(
                            (), dtype=real_dtype, device=device
                        )
                        for tail_index, audit in enumerate(
                            risk_weights["global_upper_log_ratio_cvar_audits"]
                        ):
                            selected_fractions = risk_weights[
                                "global_upper_log_ratio_cvar_group_selections"
                            ][tail_index][group_index].index_select(
                                0, point_indices_t
                            )
                            global_upper_log_cvar_term = (
                                global_upper_log_cvar_term
                                + audit["mixture_weight"]
                                * weighted_cvar_selected_linearization_torch(
                                    global_upper_log_losses,
                                    importance_weights,
                                    selected_fractions,
                                    tail_fraction=audit["tail_fraction"],
                                    total_weight=risk_weights[
                                        "global_upper_log_ratio_weight_sum"
                                    ],
                                )
                            )
                        group_loss = (
                            group_loss
                            + request.global_upper_log_ratio_cvar_loss_weight
                            * global_tail_step_multiplier
                            * global_upper_log_cvar_term
                        )
                    if request.drift_weight:
                        group_loss = group_loss + request.drift_weight * drift_penalty(
                            h_matrix
                        )
                    if request.relative_log_spectrum_loss_weight:
                        group_loss = (
                            group_loss
                            + request.relative_log_spectrum_loss_weight
                            * relative_log_spectrum_penalty()
                        )
                    if request.low_rank_factor_weight:
                        group_loss = (
                            group_loss
                            + request.low_rank_factor_weight
                            * parameterization_penalty()
                        )
                    if not torch.isfinite(group_loss):
                        raise FloatingPointError(
                            "cluster mini-batch produced a nonfinite loss"
                        )
                    group_loss.backward()
                    take_optimizer_step()
                    batch_mean_losses.append(
                        float(torch.mean(weighted_point_losses).detach().cpu())
                    )
                loss_value = float(np.mean(batch_mean_losses))
            elif request.group_optimizer_step_mode == "shuffled_point_minibatch":
                batch_descriptors: list[tuple[int, np.ndarray]] = []
                for group_index, (_, dataset) in enumerate(train_groups):
                    point_order = group_order_rng.permutation(len(dataset["log_omega"]))
                    for start in range(
                        0,
                        len(point_order),
                        request.points_per_optimizer_step,
                    ):
                        batch_descriptors.append(
                            (
                                group_index,
                                point_order[
                                    start : start + request.points_per_optimizer_step
                                ],
                            )
                        )
                upper_tail_replay_probabilities: list[np.ndarray] = []
                if request.point_upper_tail_replay_points_per_fraction:
                    if len(train_groups) != 1:
                        raise RuntimeError(
                            "global point replay requires one pooled training group"
                        )
                    replay_dataset = train_groups[0][1]
                    replay_base_weights = (
                        np.asarray(
                            replay_dataset["importance_weights"], dtype=np.float64
                        )
                        if request.importance_weighted
                        else np.ones(len(replay_dataset["log_omega"]), dtype=np.float64)
                    )
                    for tail_group_selections in risk_weights[
                        "global_upper_log_ratio_cvar_group_selections"
                    ]:
                        if len(tail_group_selections) != 1:
                            raise RuntimeError(
                                "global point replay tail selection is not pooled"
                            )
                        selected = (
                            tail_group_selections[0].detach().cpu().numpy()
                        )
                        replay_mass = replay_base_weights * selected
                        replay_mass_sum = float(np.sum(replay_mass))
                        if not np.isfinite(replay_mass_sum) or replay_mass_sum <= 0:
                            raise FloatingPointError(
                                "upper-tail replay has no positive selected mass"
                            )
                        upper_tail_replay_probabilities.append(
                            replay_mass / replay_mass_sum
                        )
                batch_order = group_order_rng.permutation(len(batch_descriptors))
                batch_mean_losses = []
                for descriptor_value in batch_order:
                    group_index, point_indices = batch_descriptors[
                        int(descriptor_value)
                    ]
                    _, dataset = train_groups[group_index]
                    source_tensors = as_torch(dataset)
                    point_indices_t = torch.tensor(
                        point_indices,
                        dtype=torch.int64,
                        device=device,
                    )
                    batch_tensors = tuple(
                        values.index_select(0, point_indices_t)
                        for values in source_tensors
                    )
                    optimizer.zero_grad(set_to_none=True)
                    h_matrix = current_h_matrix()
                    raw, barrier, importance_weights = raw_and_barrier_tensors(
                        *batch_tensors,
                        h_matrix,
                    )
                    loss_weights = (
                        torch.ones_like(importance_weights)
                        if request.point_minibatch_uniform_loss
                        else importance_weights
                    )
                    loss_weights = loss_weights / torch.mean(loss_weights)
                    centered = raw - fixed_group_log_normalization
                    absolute_error = torch.abs(1.0 - torch.exp(centered))
                    weighted_point_losses = loss_weights * absolute_error
                    sigma_term = (
                        torch.sum(weighted_point_losses)
                        if request.point_minibatch_loss_reduction == "sum"
                        else torch.mean(weighted_point_losses)
                    )
                    centered_log_variance = weighted_mean(
                        torch.square(centered),
                        loss_weights,
                    )
                    group_loss = (
                        request.centered_log_variance_loss_weight
                        * centered_log_variance
                        + request.sigma_loss_weight * sigma_term
                        + request.metric_barrier_weight * barrier
                    )
                    global_tail_step_multiplier = float(len(batch_descriptors))
                    if request.point_minibatch_loss_reduction == "sum":
                        global_tail_step_multiplier *= len(point_indices)
                    if request.global_volume_ratio_cvar_loss_weight:
                        global_losses = torch.abs(
                            1.0 - torch.exp(raw - fixed_group_log_normalization)
                        )
                        global_cvar_term = torch.zeros(
                            (), dtype=real_dtype, device=device
                        )
                        for tail_index, audit in enumerate(
                            risk_weights["global_volume_ratio_cvar_audits"]
                        ):
                            selected_fractions = risk_weights[
                                "global_volume_ratio_cvar_group_selections"
                            ][tail_index][group_index].index_select(
                                0, point_indices_t
                            )
                            global_cvar_term = global_cvar_term + (
                                audit["mixture_weight"]
                                * weighted_cvar_selected_linearization_torch(
                                    global_losses,
                                    importance_weights,
                                    selected_fractions,
                                    tail_fraction=audit["tail_fraction"],
                                    total_weight=risk_weights[
                                        "global_volume_ratio_weight_sum"
                                    ],
                                )
                            )
                        group_loss = (
                            group_loss
                            + request.global_volume_ratio_cvar_loss_weight
                            * global_tail_step_multiplier
                            * global_cvar_term
                        )
                    if (
                        request.global_upper_log_ratio_cvar_loss_weight
                        and not request.point_upper_tail_replay_points_per_fraction
                    ):
                        global_upper_log_losses = (
                            smooth_upper_log_ratio_excess_torch(
                                centered,
                                ratio_threshold=(
                                    request.global_upper_log_ratio_threshold
                                ),
                                smooth_temperature=(
                                    request.global_upper_log_ratio_smooth_temperature
                                ),
                            )
                        )
                        global_upper_log_cvar_term = torch.zeros(
                            (), dtype=real_dtype, device=device
                        )
                        for tail_index, audit in enumerate(
                            risk_weights["global_upper_log_ratio_cvar_audits"]
                        ):
                            selected_fractions = risk_weights[
                                "global_upper_log_ratio_cvar_group_selections"
                            ][tail_index][group_index].index_select(
                                0, point_indices_t
                            )
                            global_upper_log_cvar_term = (
                                global_upper_log_cvar_term
                                + audit["mixture_weight"]
                                * weighted_cvar_selected_linearization_torch(
                                    global_upper_log_losses,
                                    importance_weights,
                                    selected_fractions,
                                    tail_fraction=audit["tail_fraction"],
                                    total_weight=risk_weights[
                                        "global_upper_log_ratio_weight_sum"
                                    ],
                                )
                            )
                        group_loss = (
                            group_loss
                            + request.global_upper_log_ratio_cvar_loss_weight
                            * global_tail_step_multiplier
                            * global_upper_log_cvar_term
                        )
                    elif request.point_upper_tail_replay_points_per_fraction:
                        replay_source_tensors = as_torch(train_groups[0][1])
                        replay_upper_log_cvar_term = torch.zeros(
                            (), dtype=real_dtype, device=device
                        )
                        replay_count = (
                            request.point_upper_tail_replay_points_per_fraction
                        )
                        for tail_index, audit in enumerate(
                            risk_weights["global_upper_log_ratio_cvar_audits"]
                        ):
                            replay_indices = group_order_rng.choice(
                                len(upper_tail_replay_probabilities[tail_index]),
                                size=replay_count,
                                replace=True,
                                p=upper_tail_replay_probabilities[tail_index],
                            )
                            replay_indices_t = torch.tensor(
                                replay_indices,
                                dtype=torch.int64,
                                device=device,
                            )
                            replay_tensors = tuple(
                                values.index_select(0, replay_indices_t)
                                for values in replay_source_tensors
                            )
                            replay_raw, _, _ = raw_and_barrier_tensors(
                                *replay_tensors,
                                h_matrix,
                            )
                            replay_losses = smooth_upper_log_ratio_excess_torch(
                                replay_raw - fixed_group_log_normalization,
                                ratio_threshold=(
                                    request.global_upper_log_ratio_threshold
                                ),
                                smooth_temperature=(
                                    request.global_upper_log_ratio_smooth_temperature
                                ),
                            )
                            replay_upper_log_cvar_term = (
                                replay_upper_log_cvar_term
                                + audit["mixture_weight"]
                                * torch.mean(replay_losses)
                            )
                        group_loss = (
                            group_loss
                            + request.global_upper_log_ratio_cvar_loss_weight
                            * replay_upper_log_cvar_term
                        )
                    if request.drift_weight:
                        group_loss = group_loss + request.drift_weight * drift_penalty(
                            h_matrix
                        )
                    if request.relative_log_spectrum_loss_weight:
                        group_loss = (
                            group_loss
                            + request.relative_log_spectrum_loss_weight
                            * relative_log_spectrum_penalty()
                        )
                    if request.low_rank_factor_weight:
                        group_loss = (
                            group_loss
                            + request.low_rank_factor_weight
                            * parameterization_penalty()
                        )
                    if not torch.isfinite(group_loss):
                        raise FloatingPointError(
                            "point mini-batch produced a nonfinite loss"
                        )
                    group_loss.backward()
                    take_optimizer_step()
                    batch_mean_losses.append(
                        float(torch.mean(weighted_point_losses).detach().cpu())
                    )
                loss_value = float(np.mean(batch_mean_losses))
            elif request.group_optimizer_step_mode == "shuffled_step_per_group":
                group_order = group_order_rng.permutation(group_count)
                for index_value in group_order:
                    index = int(index_value)
                    _, dataset = train_groups[index]
                    optimizer.zero_grad(set_to_none=True)
                    h_matrix = current_h_matrix()
                    component = group_objective_components(dataset, h_matrix)
                    chi_coefficient = request.group_volume_ratio_l2_loss_weight * (
                        mean_fraction
                        + (1.0 - mean_fraction)
                        * group_count
                        * risk_weights["volume_ratio_l2"][index]
                    )
                    cvar_coefficient = request.group_volume_ratio_cvar_loss_weight * (
                        mean_fraction
                        + (1.0 - mean_fraction)
                        * group_count
                        * risk_weights["volume_ratio_cvar"][index]
                    )
                    positive_log_cvar_coefficient = (
                        request.group_positive_log_ratio_cvar_loss_weight
                        * (
                            mean_fraction
                            + (1.0 - mean_fraction)
                            * group_count
                            * risk_weights["positive_log_ratio_cvar"][index]
                        )
                    )
                    group_loss = (
                        component["base"]
                        + chi_coefficient * component["volume_ratio_l2"]
                        + cvar_coefficient * component["volume_ratio_cvar"]
                        + positive_log_cvar_coefficient
                        * component["positive_log_ratio_cvar"]
                        + global_volume_ratio_cvar_gradient_term(
                            component,
                            risk_weights,
                            group_index=index,
                            group_multiplier=float(group_count),
                        )
                        + global_upper_log_ratio_cvar_gradient_term(
                            component,
                            risk_weights,
                            group_index=index,
                            group_multiplier=float(group_count),
                        )
                        + request.metric_barrier_weight * component["barrier"]
                    )
                    if active_set is not None:
                        active_risk, active_barrier, _, _ = active_set_objective(
                            h_matrix,
                            risk_weights["reference_log_normalization"],
                        )
                        group_loss = (
                            group_loss
                            + request.active_set_loss_weight * active_risk
                            + request.metric_barrier_weight * active_barrier
                        )
                    if request.drift_weight:
                        drift = drift_penalty(h_matrix)
                        group_loss = group_loss + request.drift_weight * drift
                    if request.low_rank_factor_weight:
                        group_loss = (
                            group_loss
                            + request.low_rank_factor_weight
                            * parameterization_penalty()
                        )
                    group_loss.backward()
                    take_optimizer_step()
                    if request.stream_training_groups:
                        del component, group_loss, h_matrix
            else:
                optimizer.zero_grad(set_to_none=True)
                for index, (_, dataset) in enumerate(train_groups):
                    h_matrix = current_h_matrix()
                    component = group_objective_components(dataset, h_matrix)
                    chi_coefficient = request.group_volume_ratio_l2_loss_weight * (
                        mean_fraction / group_count
                        + (1.0 - mean_fraction) * risk_weights["volume_ratio_l2"][index]
                    )
                    cvar_coefficient = request.group_volume_ratio_cvar_loss_weight * (
                        mean_fraction / group_count
                        + (1.0 - mean_fraction)
                        * risk_weights["volume_ratio_cvar"][index]
                    )
                    positive_log_cvar_coefficient = (
                        request.group_positive_log_ratio_cvar_loss_weight
                        * (
                            mean_fraction / group_count
                            + (1.0 - mean_fraction)
                            * risk_weights["positive_log_ratio_cvar"][index]
                        )
                    )
                    group_loss = (
                        component["base"] / group_count
                        + chi_coefficient * component["volume_ratio_l2"]
                        + cvar_coefficient * component["volume_ratio_cvar"]
                        + positive_log_cvar_coefficient
                        * component["positive_log_ratio_cvar"]
                        + global_volume_ratio_cvar_gradient_term(
                            component,
                            risk_weights,
                            group_index=index,
                            group_multiplier=1.0,
                        )
                        + global_upper_log_ratio_cvar_gradient_term(
                            component,
                            risk_weights,
                            group_index=index,
                            group_multiplier=1.0,
                        )
                        + request.metric_barrier_weight
                        * component["barrier"]
                        / group_count
                    )
                    group_loss.backward()
                    if request.stream_training_groups:
                        del component, group_loss, h_matrix
                if active_set is not None:
                    active_h = current_h_matrix()
                    active_risk, active_barrier, _, _ = active_set_objective(
                        active_h,
                        risk_weights["reference_log_normalization"],
                    )
                    (
                        request.active_set_loss_weight * active_risk
                        + request.metric_barrier_weight * active_barrier
                    ).backward()
                    if request.stream_training_groups:
                        del active_h, active_risk, active_barrier
                if request.drift_weight:
                    drift_h = current_h_matrix()
                    drift = drift_penalty(drift_h)
                    (request.drift_weight * drift).backward()
                if request.low_rank_factor_weight:
                    (
                        request.low_rank_factor_weight * parameterization_penalty()
                    ).backward()
                take_optimizer_step()
        else:
            optimizer.zero_grad(set_to_none=True)
            h_matrix = current_h_matrix()
            train_raw, train_barrier, train_weights = raw_and_barrier(train, h_matrix)
            validation_raw, validation_barrier, validation_weights = raw_and_barrier(
                validation, h_matrix
            )
            train_mean = weighted_mean(train_raw, train_weights)
            validation_mean = weighted_mean(validation_raw, validation_weights)
            shared_mean = (train_mean + request.validation_weight * validation_mean) / (
                1.0 + request.validation_weight
            )
            train_loss = weighted_mean((train_raw - shared_mean) ** 2, train_weights)
            validation_loss = weighted_mean(
                (validation_raw - shared_mean) ** 2,
                validation_weights,
            )
            train_sigma_loss = normalized_sigma(
                train_raw,
                train_weights,
                clamp_exponent=True,
            )
            validation_sigma_loss = normalized_sigma(
                validation_raw,
                validation_weights,
                clamp_exponent=True,
            )
            train_volume_ratio_l2_loss = normalized_volume_ratio_l2(
                train_raw,
                train_weights,
                clamp_exponent=True,
            )
            validation_volume_ratio_l2_loss = normalized_volume_ratio_l2(
                validation_raw,
                validation_weights,
                clamp_exponent=True,
            )
            drift = drift_penalty(h_matrix)
            loss = (
                train_loss
                + request.validation_weight * validation_loss
                + request.sigma_loss_weight
                * (train_sigma_loss + request.validation_weight * validation_sigma_loss)
                + request.volume_ratio_l2_loss_weight
                * (
                    train_volume_ratio_l2_loss
                    + request.validation_weight * validation_volume_ratio_l2_loss
                )
                + request.metric_barrier_weight
                * (train_barrier + validation_barrier)
                + request.drift_weight * drift
                + request.low_rank_factor_weight * parameterization_penalty()
            )
            loss.backward()
            loss_value = float(loss.detach().cpu())
            take_optimizer_step()

        if epoch % request.eval_every != 0 and epoch != request.epochs:
            continue
        (
            evaluated_training_group_rows,
            evaluated_risk_data,
            _,
        ) = probe_group_objective()
        candidate_h = as_hermitian_numpy(current_h_matrix().detach().cpu().numpy())
        rows = []
        for name, dataset in selection_sets:
            baseline, candidate = evaluate(dataset, candidate_h)
            rows.append(
                (name, baseline, candidate, selection_row_gate(baseline, candidate))
            )
        (
            score,
            mean_check_log_rms,
            mean_check_sigma,
            mean_check_volume_ratio_l2,
            maximum_check_volume_ratio_l2,
            maximum_check_positive_log_ratio_q999,
            maximum_check_positive_log_ratio_cvar,
            maximum_check_ratio_above_3_weighted_mass,
        ) = selection_statistics(rows)
        all_passed = bool(all(row[3] for row in rows))
        checkpoint_confirmed = False
        if request.checkpoint_policy == "first_consecutive_eligible":
            accepted = False
            if all_passed:
                if consecutive_eligible == 0:
                    pending_h = candidate_h.copy()
                    pending_epoch = epoch
                    pending_score = score
                    pending_training_group_rows = evaluated_training_group_rows
                    pending_history_index = len(history)
                consecutive_eligible += 1
                if consecutive_eligible >= request.checkpoint_required_consecutive:
                    if (
                        pending_h is None
                        or pending_epoch is None
                        or pending_score is None
                    ):
                        raise RuntimeError(
                            "eligible checkpoint sequence lost its first entry"
                        )
                    best_h = pending_h.copy()
                    best_epoch = pending_epoch
                    best_score = pending_score
                    best_passed = True
                    selected_training_group_rows = pending_training_group_rows
                    checkpoint_confirmed = True
                    confirmation_epoch = epoch
            else:
                consecutive_eligible = 0
                pending_h = None
                pending_epoch = None
                pending_score = None
                pending_training_group_rows = []
                pending_history_index = None
            stale_evaluations = 0 if all_passed else stale_evaluations + 1
        else:
            accepted = bool(all_passed and score < best_score)
            if accepted:
                best_h = candidate_h.copy()
                best_score = score
                best_passed = True
                best_epoch = epoch
                selected_training_group_rows = evaluated_training_group_rows
                stale_evaluations = 0
            else:
                stale_evaluations += 1
        learning_rate_before_action = float(optimizer.param_groups[0]["lr"])
        learning_rate_after_action = learning_rate_before_action
        ineligible_action_taken = "none"
        rollback_after_evaluation = False
        stop_after_checkpoint_action = False
        checkpoint_action_termination_reason: str | None = None
        if not all_passed:
            ineligible_checkpoint_count += 1
            if request.ineligible_checkpoint_action == "continue":
                ineligible_action_taken = "continue"
            elif request.ineligible_checkpoint_action == "stop":
                ineligible_action_taken = "stop"
                stop_after_checkpoint_action = True
                checkpoint_action_termination_reason = "ineligible_checkpoint_stop"
            elif last_eligible_training_state is None:
                ineligible_action_taken = "stop_no_eligible_checkpoint"
                stop_after_checkpoint_action = True
                checkpoint_action_termination_reason = (
                    "ineligible_checkpoint_without_rollback_state"
                )
            else:
                ineligible_action_taken = "rollback_reduce_lr"
                rollback_after_evaluation = True
                checkpoint_rollback_count += 1
                learning_rate_after_action *= (
                    request.ineligible_checkpoint_learning_rate_factor
                )
                if (
                    checkpoint_rollback_count
                    >= request.maximum_ineligible_checkpoint_rollbacks
                ):
                    stop_after_checkpoint_action = True
                    checkpoint_action_termination_reason = (
                        "maximum_ineligible_checkpoint_rollbacks"
                    )
        history_row = {
            "epoch": epoch,
            "loss": loss_value,
            "accepted": accepted,
            "passed_internal_gates": all_passed,
            "checkpoint_eligible": all_passed,
            "checkpoint_confirmed": checkpoint_confirmed,
            "consecutive_eligible_evaluations": consecutive_eligible,
            "ineligible_checkpoint_action_taken": ineligible_action_taken,
            "rolled_back_after_evaluation": rollback_after_evaluation,
            "learning_rate_before_action": learning_rate_before_action,
            "learning_rate_after_action": learning_rate_after_action,
            "cumulative_ineligible_checkpoints": ineligible_checkpoint_count,
            "cumulative_checkpoint_rollbacks": checkpoint_rollback_count,
            "selection_score": score,
            "mean_check_log_rms": mean_check_log_rms,
            "mean_check_sigma": mean_check_sigma,
            "mean_check_volume_ratio_l2": mean_check_volume_ratio_l2,
            "maximum_check_volume_ratio_l2": maximum_check_volume_ratio_l2,
            "maximum_check_positive_log_ratio_q999": (
                maximum_check_positive_log_ratio_q999
            ),
            "maximum_check_positive_log_ratio_cvar": (
                maximum_check_positive_log_ratio_cvar
            ),
            "maximum_check_ratio_above_3_weighted_mass": (
                maximum_check_ratio_above_3_weighted_mass
            ),
            **optimizer_history_fields(),
            **relative_to_initial_history_fields(candidate_h),
            **group_history_fields(evaluated_training_group_rows, evaluated_risk_data),
        }
        history.append(history_row)
        reset_optimizer_history_interval()
        if checkpoint_confirmed:
            if pending_history_index is None:
                raise RuntimeError("confirmed checkpoint has no history entry")
            history[pending_history_index]["accepted"] = True
            history[pending_history_index]["checkpoint_confirmed"] = True
            history[pending_history_index]["confirmed_at_epoch"] = epoch
        print(
            f"epoch {epoch}: loss={loss_value:.6e}, "
            f"mean_check_log_rms={mean_check_log_rms:.6e}, "
            f"mean_check_sigma={mean_check_sigma:.6e}, "
            f"mean_check_volume_ratio_l2={mean_check_volume_ratio_l2:.6e}, "
            f"maximum_check_volume_ratio_l2={maximum_check_volume_ratio_l2:.6e}, "
            f"maximum_check_positive_log_ratio_q999="
            f"{maximum_check_positive_log_ratio_q999:.6e}, "
            f"maximum_check_positive_log_ratio_cvar="
            f"{maximum_check_positive_log_ratio_cvar:.6e}, "
            f"maximum_check_ratio_above_3_weighted_mass="
            f"{maximum_check_ratio_above_3_weighted_mass:.6e}, "
            f"selection_score={score:.6e}, eligible={all_passed}, "
            f"consecutive={consecutive_eligible}, accepted={accepted}, "
            f"confirmed={checkpoint_confirmed}, "
            f"gate_action={ineligible_action_taken}",
            flush=True,
        )
        if all_passed:
            last_eligible_training_state = snapshot_training_state()
        elif rollback_after_evaluation:
            if last_eligible_training_state is None:
                raise RuntimeError("checkpoint rollback state unexpectedly missing")
            restore_training_state(
                last_eligible_training_state,
                learning_rate=learning_rate_after_action,
            )
        if checkpoint_confirmed:
            print(
                f"selected first eligible checkpoint at epoch {best_epoch}; "
                f"confirmed at epoch {epoch}",
                flush=True,
            )
            termination_reason = "checkpoint_confirmed"
            break
        if stop_after_checkpoint_action:
            termination_reason = str(checkpoint_action_termination_reason)
            print(
                f"stopping after checkpoint gate action: {termination_reason}",
                flush=True,
            )
            break
        if (
            request.patience_evaluations > 0
            and stale_evaluations >= request.patience_evaluations
        ):
            print(
                f"early stopping after {stale_evaluations} stale evaluations",
                flush=True,
            )
            termination_reason = "patience_exhausted"
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimization_seconds = time.perf_counter() - optimization_started

    export_baseline, export_candidate = evaluate(export, best_h)
    export_artifact = HMetricArtifact(
        path=artifact_path.resolve(),
        degree=request.degree,
        section_exponents=exponents,
        h_matrix=best_h,
        normalization=normalization,
    )
    export_metrics = adapter.h_metrics(export["points"], export_artifact)
    selection_summary = {}
    for name, dataset in selection_sets:
        baseline, candidate = evaluate(dataset, best_h)
        metric_gate_passed = _stats_gate(baseline, candidate)
        if request.groupwise_objective:
            sigma_gate_passed = bool(
                candidate["sigma"] <= request.maximum_checkpoint_sigma
            )
            volume_ratio_l2_gate_passed = bool(
                candidate["sqrt_squared_energy"]
                <= request.maximum_checkpoint_volume_ratio_l2
            )
            positive_log_ratio_q999_gate_passed = bool(
                request.maximum_checkpoint_positive_log_ratio_q999 is None
                or candidate["positive_log_ratio_q999"]
                <= request.maximum_checkpoint_positive_log_ratio_q999
            )
            positive_log_ratio_cvar_gate_passed = bool(
                request.maximum_checkpoint_positive_log_ratio_cvar is None
                or candidate["positive_log_ratio_cvar_1pct"]
                <= request.maximum_checkpoint_positive_log_ratio_cvar
            )
            ratio_above_3_weighted_mass_gate_passed = bool(
                request.maximum_checkpoint_ratio_above_3_weighted_mass is None
                or candidate["normalized_ratio_above_3_weighted_mass"]
                <= request.maximum_checkpoint_ratio_above_3_weighted_mass
            )
        else:
            sigma_gate_passed = True
            volume_ratio_l2_gate_passed = bool(
                request.maximum_check_volume_ratio_l2 is None
                or candidate["sqrt_squared_energy"]
                <= request.maximum_check_volume_ratio_l2
            )
            positive_log_ratio_q999_gate_passed = True
            positive_log_ratio_cvar_gate_passed = True
            ratio_above_3_weighted_mass_gate_passed = True
        selection_summary[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "metric_gate_passed": metric_gate_passed,
            "sigma_gate_passed": sigma_gate_passed,
            "volume_ratio_l2_gate_passed": volume_ratio_l2_gate_passed,
            "positive_log_ratio_q999_gate_passed": (
                positive_log_ratio_q999_gate_passed
            ),
            "positive_log_ratio_cvar_gate_passed": (
                positive_log_ratio_cvar_gate_passed
            ),
            "ratio_above_3_weighted_mass_gate_passed": (
                ratio_above_3_weighted_mass_gate_passed
            ),
            "passed": bool(
                metric_gate_passed
                and sigma_gate_passed
                and volume_ratio_l2_gate_passed
                and positive_log_ratio_q999_gate_passed
                and positive_log_ratio_cvar_gate_passed
                and ratio_above_3_weighted_mass_gate_passed
            ),
        }
    check_summary = {} if request.groupwise_objective else selection_summary
    checkpoint_summary = selection_summary if request.groupwise_objective else {}

    artifact_path = artifact_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selected_history_row = next(
        row for row in history if int(row["epoch"]) == int(best_epoch)
    )
    selected_relative_h_update = {
        "relative_to_initial_log_eigenvalue_span": float(
            selected_history_row["relative_to_initial_log_eigenvalue_span"]
        ),
        "relative_to_initial_max_abs_centered_log_eigenvalue": float(
            selected_history_row["relative_to_initial_max_abs_centered_log_eigenvalue"]
        ),
    }
    payload = {
        **adapter.artifact_model_payload(model, exact=request.exact_model),
        "pipeline_schema_version": np.asarray(1),
        "pipeline_adapter": np.asarray(adapter.key),
        "importance_weighted_training": np.asarray(request.importance_weighted),
        "global_section_degree": np.asarray(request.degree, dtype=np.int64),
        "global_section_exponents": exponents,
        "global_h_matrix": best_h,
        "global_h_positive_relative_floor": np.asarray(
            H_POSITIVE_RELATIVE_FLOOR,
            dtype=np.float64,
        ),
        "global_section_normalization": np.asarray(normalization),
        "basis_selected_indices": np.asarray(basis.selected_indices, dtype=np.int64),
        "basis_relation_error": np.asarray(relation_error),
        "h_parameterization": np.asarray(request.h_parameterization),
        "h_optimization_coordinate_system": np.asarray(optimization_coordinate_system),
        "spd_step_diagnostic_representation": np.asarray(
            spd_step_diagnostic_representation
        ),
        "parameterization_real_dimension": np.asarray(
            parameterization_real_dimension, dtype=np.int64
        ),
        "scale_invariant_h_dimension_upper_bound": np.asarray(
            scale_invariant_h_dimension_upper_bound, dtype=np.int64
        ),
        "group_normalization_mode": np.asarray(request.group_normalization_mode),
        "group_optimizer_step_mode": np.asarray(request.group_optimizer_step_mode),
        "training_sampling_mode": np.asarray(request.training_sampling_mode),
        "clusters_per_optimizer_step": np.asarray(
            request.clusters_per_optimizer_step,
            dtype=np.int64,
        ),
        "cluster_minibatch_loss_reduction": np.asarray(
            request.cluster_minibatch_loss_reduction
        ),
        "cluster_minibatch_uniform_loss": np.asarray(
            request.cluster_minibatch_uniform_loss
        ),
        "cluster_tail_replay_batches_per_epoch": np.asarray(
            request.cluster_tail_replay_batches_per_epoch,
            dtype=np.int64,
        ),
        "points_per_optimizer_step": np.asarray(
            request.points_per_optimizer_step,
            dtype=np.int64,
        ),
        "point_minibatch_loss_reduction": np.asarray(
            request.point_minibatch_loss_reduction
        ),
        "point_minibatch_uniform_loss": np.asarray(
            request.point_minibatch_uniform_loss
        ),
        "point_upper_tail_replay_points_per_fraction": np.asarray(
            request.point_upper_tail_replay_points_per_fraction,
            dtype=np.int64,
        ),
        "point_minibatch_pooling": np.asarray(
            (
                "all_training_batches"
                if request.group_optimizer_step_mode == "shuffled_point_minibatch"
                else "not_applicable"
            )
        ),
        "centered_log_variance_loss_weight": np.asarray(
            request.centered_log_variance_loss_weight,
            dtype=np.float64,
        ),
        "metric_barrier_weight": np.asarray(
            request.metric_barrier_weight,
            dtype=np.float64,
        ),
        "relative_log_spectrum_loss_weight": np.asarray(
            request.relative_log_spectrum_loss_weight,
            dtype=np.float64,
        ),
        "group_shuffle_seed": np.asarray(
            effective_group_shuffle_seed,
            dtype=np.int64,
        ),
        "gradient_clip_norm": np.asarray(
            (
                request.gradient_clip_norm
                if request.gradient_clip_norm is not None
                else np.nan
            ),
            dtype=np.float64,
        ),
        "optimizer_step_count": np.asarray(optimizer_step_count, dtype=np.int64),
        "ineligible_checkpoint_action": np.asarray(
            request.ineligible_checkpoint_action
        ),
        "ineligible_checkpoint_count": np.asarray(
            ineligible_checkpoint_count, dtype=np.int64
        ),
        "checkpoint_rollback_count": np.asarray(
            checkpoint_rollback_count, dtype=np.int64
        ),
        "optimization_termination_reason": np.asarray(termination_reason),
        "record_spd_step_every": np.asarray(
            request.record_spd_step_every,
            dtype=np.int64,
        ),
        "maximum_spd_log_step_radius": np.asarray(
            (
                request.maximum_spd_log_step_radius
                if request.maximum_spd_log_step_radius is not None
                else np.nan
            ),
            dtype=np.float64,
        ),
        "spd_step_projection_bisections": np.asarray(
            request.spd_step_projection_bisections,
            dtype=np.int64,
        ),
        "selected_relative_to_initial_log_eigenvalue_span": np.asarray(
            selected_relative_h_update["relative_to_initial_log_eigenvalue_span"],
            dtype=np.float64,
        ),
        "fixed_group_log_kappa": np.asarray(
            (
                float(fixed_group_log_normalization.cpu())
                if fixed_group_log_normalization is not None
                else np.nan
            ),
            dtype=np.float64,
        ),
        "low_rank": np.asarray(
            request.low_rank
            if request.h_parameterization == "reference_low_rank"
            else 0
        ),
        "low_rank_epsilon": np.asarray(request.low_rank_epsilon),
        "group_volume_ratio_cvar_loss_weight": np.asarray(
            request.group_volume_ratio_cvar_loss_weight
        ),
        "group_volume_ratio_cvar_tail_fraction": np.asarray(
            request.group_volume_ratio_cvar_tail_fraction
        ),
        "group_positive_log_ratio_cvar_loss_weight": np.asarray(
            request.group_positive_log_ratio_cvar_loss_weight
        ),
        "group_positive_log_ratio_cvar_tail_fraction": np.asarray(
            request.group_positive_log_ratio_cvar_tail_fraction
        ),
        "global_volume_ratio_cvar_loss_weight": np.asarray(
            request.global_volume_ratio_cvar_loss_weight
        ),
        "global_volume_ratio_cvar_tail_fractions": np.asarray(
            request.global_volume_ratio_cvar_tail_fractions,
            dtype=np.float64,
        ),
        "global_volume_ratio_cvar_tail_weights": np.asarray(
            request.global_volume_ratio_cvar_tail_weights,
            dtype=np.float64,
        ),
        "global_upper_log_ratio_cvar_loss_weight": np.asarray(
            request.global_upper_log_ratio_cvar_loss_weight
        ),
        "global_upper_log_ratio_cvar_tail_fractions": np.asarray(
            request.global_upper_log_ratio_cvar_tail_fractions,
            dtype=np.float64,
        ),
        "global_upper_log_ratio_cvar_tail_weights": np.asarray(
            request.global_upper_log_ratio_cvar_tail_weights,
            dtype=np.float64,
        ),
        "global_upper_log_ratio_threshold": np.asarray(
            request.global_upper_log_ratio_threshold,
            dtype=np.float64,
        ),
        "global_upper_log_ratio_smooth_temperature": np.asarray(
            request.global_upper_log_ratio_smooth_temperature,
            dtype=np.float64,
        ),
        "active_set_loss_weight": np.asarray(request.active_set_loss_weight),
        "active_set_log_ratio_threshold": np.asarray(
            request.active_set_log_ratio_threshold
        ),
        "active_set_smooth_max_temperature": np.asarray(
            request.active_set_smooth_max_temperature
        ),
        "active_set_region_balanced": np.asarray(request.active_set_region_balanced),
        "active_set_sha256": np.asarray(
            file_sha256(request.active_set_path)
            if request.active_set_path is not None
            else ""
        ),
        "train_common_pool_sha256": np.asarray(
            common_pool_records.get("train", {}).get("sha256", "")
        ),
        "selection_common_pool_sha256": np.asarray(
            common_pool_records.get("selection", {}).get("sha256", "")
        ),
        "baseline_log_ma": adapter.residual_values(
            export["points"], export["baseline_metrics"]
        ),
        "corrected_log_ma": adapter.residual_values(export["points"], export_metrics),
        "importance_weights": export["importance_weights"],
    }
    np.savez_compressed(artifact_path, **payload)

    exploratory_selection = request.checkpoint_policy == "best_score_exploratory"
    success = bool(
        not exploratory_selection
        and best_passed
        and _stats_gate(export_baseline, export_candidate)
        and export_candidate["min_metric_eigenvalue"] > 0
    )
    exploratory_selection_completed = bool(
        best_passed
        and _stats_gate(export_baseline, export_candidate)
        and export_candidate["min_metric_eigenvalue"] > 0
    )
    device_memory = None
    if device.type == "cuda":
        allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        device_memory = {
            "peak_allocated_bytes": allocated_bytes,
            "peak_allocated_mib": allocated_bytes / 2**20,
            "peak_reserved_bytes": reserved_bytes,
            "peak_reserved_mib": reserved_bytes / 2**20,
        }
    summary = {
        "schema_version": 1,
        "description": "Generic positive-Hermitian global-section H-metric training.",
        "adapter": {"key": adapter.key, "version": adapter.version},
        "configuration": adapter.configuration.to_dict(),
        "model": adapter.model_metadata(model),
        "request": {
            **asdict(request),
            "degree": list(request.degree),
            "train_batches": [
                {"seed": int(seed), "points": int(count)}
                for seed, count in train_batches
            ],
            "validation_batches": [
                {"seed": int(seed), "points": int(count)}
                for seed, count in validation_batches
            ],
            "checkpoint_batches": [
                {"seed": int(seed), "points": int(count)}
                for seed, count in request.checkpoint_batches
            ],
            "check_seeds": list(request.check_seeds),
            "check_batches": [
                {"seed": int(seed), "points": int(count)}
                for seed, count in check_batches
            ],
            "initial_artifact": (
                str(request.initial_artifact.resolve())
                if request.initial_artifact is not None
                else None
            ),
            "active_set_path": (
                str(request.active_set_path.resolve())
                if request.active_set_path is not None
                else None
            ),
            "train_common_pool": (
                str(request.train_common_pool.expanduser().resolve())
                if request.train_common_pool is not None
                else None
            ),
            "selection_common_pool": (
                str(request.selection_common_pool.expanduser().resolve())
                if request.selection_common_pool is not None
                else None
            ),
        },
        "common_point_pools": common_pool_records or None,
        "parallel_sampling": {
            "workers": request.sampling_workers,
            "cluster_size": request.sampling_cluster_size,
            "backend": request.sampling_backend,
            "training_sampling_mode": request.training_sampling_mode,
            "training_points_are_independent_fibres": (
                request.training_sampling_mode == "one_random_root_per_fibre"
            ),
            "seed_derivation": "numpy.random.SeedSequence(base_seed).spawn(active_workers)",
            "shards": sampling_shards,
        },
        "device": str(device),
        "device_memory": device_memory,
        "normalization": normalization,
        "training_group_normalization": {
            "mode": request.group_normalization_mode,
            "fixed_log_kappa": (
                float(fixed_group_log_normalization.cpu())
                if fixed_group_log_normalization is not None
                else None
            ),
            "source": (
                "initial H over the complete frozen training pool"
                if fixed_group_log_normalization is not None
                else "current H within each training group"
            ),
        },
        "optimization": {
            "optimizer": "Adam",
            "h_coordinate_system": optimization_coordinate_system,
            "spd_step_diagnostic_representation": (spd_step_diagnostic_representation),
            "parameterization_real_dimension": parameterization_real_dimension,
            "scale_invariant_h_dimension_upper_bound": (
                scale_invariant_h_dimension_upper_bound
            ),
            "group_optimizer_step_mode": request.group_optimizer_step_mode,
            "training_sampling_mode": request.training_sampling_mode,
            "group_shuffle_seed": effective_group_shuffle_seed,
            "clusters_per_optimizer_step": request.clusters_per_optimizer_step,
            "cluster_minibatch_loss_reduction": (
                request.cluster_minibatch_loss_reduction
            ),
            "cluster_minibatch_uniform_loss": (
                request.cluster_minibatch_uniform_loss
            ),
            "cluster_tail_replay_batches_per_epoch": (
                request.cluster_tail_replay_batches_per_epoch
            ),
            "points_per_optimizer_step": request.points_per_optimizer_step,
            "point_minibatch_loss_reduction": (
                request.point_minibatch_loss_reduction
            ),
            "point_minibatch_uniform_loss": (
                request.point_minibatch_uniform_loss
            ),
            "point_upper_tail_replay_points_per_fraction": (
                request.point_upper_tail_replay_points_per_fraction
            ),
            "point_minibatch_pooling": (
                "all_training_batches"
                if request.group_optimizer_step_mode == "shuffled_point_minibatch"
                else "not_applicable"
            ),
            "centered_log_variance_loss_weight": (
                request.centered_log_variance_loss_weight
            ),
            "metric_barrier_weight": request.metric_barrier_weight,
            "relative_log_spectrum_loss_weight": (
                request.relative_log_spectrum_loss_weight
            ),
            "training_cluster_count": training_cluster_count,
            "training_point_count": training_point_count,
            "training_sampling_cluster_count": training_sampling_cluster_count,
            "nominal_points_per_optimizer_step": (
                request.points_per_optimizer_step
                if request.group_optimizer_step_mode == "shuffled_point_minibatch"
                else (
                    request.clusters_per_optimizer_step
                    * request.sampling_cluster_size
                    if request.clusters_per_optimizer_step
                    else None
                )
            ),
            "gradient_clip_norm": request.gradient_clip_norm,
            "record_spd_step_every": request.record_spd_step_every,
            "maximum_spd_log_step_radius": request.maximum_spd_log_step_radius,
            "spd_step_projection_bisections": (request.spd_step_projection_bisections),
            "global_volume_ratio_cvar": {
                "loss_weight": request.global_volume_ratio_cvar_loss_weight,
                "tail_fractions": list(request.global_volume_ratio_cvar_tail_fractions),
                "configured_tail_weights": list(
                    request.global_volume_ratio_cvar_tail_weights
                ),
                "tail_weights_are_normalized_to_unit_sum": True,
                "threshold_update": "once_per_training_sweep",
                "gradient_estimator": (
                    "fixed_weighted_var_tail_selection_with_fractional_boundary"
                ),
                "minibatch_gradient_rescaling": (
                    "number_of_sweep_batches_times_batch_size_for_sum_reduction"
                ),
            },
            "global_upper_log_ratio_cvar": {
                "loss_weight": request.global_upper_log_ratio_cvar_loss_weight,
                "tail_fractions": list(
                    request.global_upper_log_ratio_cvar_tail_fractions
                ),
                "configured_tail_weights": list(
                    request.global_upper_log_ratio_cvar_tail_weights
                ),
                "ratio_threshold": request.global_upper_log_ratio_threshold,
                "smooth_temperature": (
                    request.global_upper_log_ratio_smooth_temperature
                ),
                "tail_weights_are_normalized_to_unit_sum": True,
                "threshold_update": "once_per_training_sweep",
                "gradient_estimator": (
                    "stratified_importance_resampling_per_optimizer_step"
                    if request.point_upper_tail_replay_points_per_fraction
                    else "fixed_weighted_var_tail_selection_with_fractional_boundary"
                ),
                "minibatch_gradient_rescaling": (
                    "none_for_stratified_tail_replay"
                    if request.point_upper_tail_replay_points_per_fraction
                    else "number_of_sweep_batches_times_batch_size_for_sum_reduction"
                ),
                "point_replay_points_per_tail_fraction": (
                    request.point_upper_tail_replay_points_per_fraction
                ),
            },
            "optimizer_step_count": optimizer_step_count,
            "ineligible_checkpoint_action": request.ineligible_checkpoint_action,
            "ineligible_checkpoint_learning_rate_factor": (
                request.ineligible_checkpoint_learning_rate_factor
            ),
            "maximum_ineligible_checkpoint_rollbacks": (
                request.maximum_ineligible_checkpoint_rollbacks
            ),
            "ineligible_checkpoint_count": ineligible_checkpoint_count,
            "checkpoint_rollback_count": checkpoint_rollback_count,
            "termination_reason": termination_reason,
            "selected_relative_h_update": selected_relative_h_update,
        },
        "kahler_power": kahler_power,
        "initialization": initialization,
        "ambient_section_count": int(len(basis.ambient_exponents)),
        "restricted_section_count": int(basis.numerical_rank),
        "basis_relation_error": float(relation_error),
        "best_epoch": best_epoch,
        "passed_internal_gates": best_passed,
        "checkpoint_confirmation": {
            "policy": request.checkpoint_policy,
            "required_consecutive": request.checkpoint_required_consecutive,
            "confirmed": bool(best_passed and not exploratory_selection),
            "selected_epoch": best_epoch if best_passed else None,
            "confirmation_epoch": confirmation_epoch,
        },
        "exploratory_selection": {
            "enabled": exploratory_selection,
            "completed": exploratory_selection_completed,
            "selected_epoch": best_epoch if best_passed else None,
            "scientific_release_eligible": False if exploratory_selection else None,
        },
        "export_baseline": export_baseline,
        "export_candidate": export_candidate,
        "checks": check_summary,
        "checkpoint_validation": checkpoint_summary,
        "initial_training_groups": initial_training_group_rows,
        "selected_training_groups": selected_training_group_rows,
        "active_set": (
            {
                "path": str(active_pool.path),
                "sha256": file_sha256(active_pool.path),
                "point_count": len(active_pool.points),
                "region_count": len(active_region_indices),
                "region_balanced_objective": request.active_set_region_balanced,
                "reported_monte_carlo_sample": False,
                "metadata": active_pool.metadata,
            }
            if active_pool is not None
            else None
        ),
        "history": history,
        "artifact": str(artifact_path),
        "success": success,
        "runtime_seconds": {
            "sampling_and_preparation": sampling_seconds,
            "optimization": optimization_seconds,
            "total": time.perf_counter() - started,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"success={success}", flush=True)
    print(f"wrote {artifact_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    return summary
