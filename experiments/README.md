# Independent post-v1 GPU experiments

This directory is the control plane for new experiments.  It does not contain
or regenerate the arXiv v1 source, Zenodo v1.0.0 archive, paper tables, or any
historical `outputs/pipeline` result.

Start with [BASELINE_V1.md](BASELINE_V1.md), then read the registered
[STUDY_PROTOCOL.md](STUDY_PROTOCOL.md).  The runnable X21 campaign is
[x21_kd_progressive_joint_paths_v1.json](manifests/x21_kd_progressive_joint_paths_v1.json).

The lower-cost Fermat-quintic extension is registered separately in
[QUINTIC_STUDY_PROTOCOL.md](QUINTIC_STUDY_PROTOCOL.md), with runnable manifest
[quintic_kd_progressive_joint_paths_v1.json](manifests/quintic_kd_progressive_joint_paths_v1.json).
It starts from the frozen historical `k=20,q=25,D=6` checkpoint, treats the
old quintic blind pool as forbidden input, and studies the new `k=40,D=14`
target.  `D=16` is resource-preflight-only until a separate extension is
registered.

The focused X21 complete-dictionary capacity experiment is registered in
[x21_architecture_capacity_v1.json](manifests/x21_architecture_capacity_v1.json).
It compares the full `k={20,24}` by `D={8,14}` surface with three paired seeds,
uses complex64 for optimization, and requires a zero-update complex128 replay
for every endpoint.  Its 109-job DAG has exactly twelve metric result rows plus
one create-only deterministic D16 adjudication artifact.  It is independent of
the broader 279-job X21 campaign.

## Focused X21 architecture-capacity quick start

Use a fresh run root.  The preflight selector includes only the preparation,
precision-roundtrip calibration, `k=24,D=14` initialization/equivalence, and a
full-pool one-epoch allocation test:

```bash
X21_FROZEN_ROOT="/path/to/frozen/gcicy-workspace"
X21_CAPACITY_RUN_ROOT="/scratch/gcicy/post-v1/x21-architecture-capacity-v1"

python scripts/run_gcicy_tn_gpu_workflow.py validate \
  --manifest experiments/manifests/x21_architecture_capacity_v1.json \
  --frozen-root "$X21_FROZEN_ROOT" \
  --run-root "$X21_CAPACITY_RUN_ROOT" \
  --check-cuda --gpu 0

python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/x21_architecture_capacity_v1.json \
  --frozen-root "$X21_FROZEN_ROOT" \
  --run-root "$X21_CAPACITY_RUN_ROOT" \
  --phase resource-preflight-d14 --gpu 0
```

Only after the registered parameter-count, positivity, and 20/21.5 GiB
allocated/reserved memory gates pass should the capacity grid and precision
replays run:

```bash
python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/x21_architecture_capacity_v1.json \
  --frozen-root "$X21_FROZEN_ROOT" \
  --run-root "$X21_CAPACITY_RUN_ROOT" \
  --phase capacity-grid --gpu 0

python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/x21_architecture_capacity_v1.json \
  --frozen-root "$X21_FROZEN_ROOT" \
  --run-root "$X21_CAPACITY_RUN_ROOT" \
  --phase precision-replay --gpu 0

python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/x21_architecture_capacity_v1.json \
  --frozen-root "$X21_FROZEN_ROOT" \
  --run-root "$X21_CAPACITY_RUN_ROOT" \
  --phase promotion-decision --gpu 0
```

The estimand is the endpoint contrast under one uniform primary/precision
validation-plateau stopping rule.  It is not a fixed-update pure capacity
effect: the adjudication records and hashes the actual round, epoch, optimizer
update, validation-evaluation, and trainer-runtime exposure for every seed.
The q0.999 and CVaR gates use the ratio of the three-seed arithmetic means;
paired sigma wins use strict per-seed inequality, so a tie is not a win.
Missing, malformed, non-finite, wrong-seed, or hash-inconsistent evidence makes
the adjudication job fail without creating a decision.  A complete negative
promotion decision is a valid successful base-workflow outcome.

`D=16` accuracy training is deliberately absent from this base DAG.  A
separate hash-frozen accuracy extension and run root may be created only after
all twelve complex128 rows are strictly positive and `k=24,D=14` satisfies the
registered paired rule over `k=24,D=8`.

## X21 `k=24,D=16` resource-only preflight

Resource feasibility can be measured independently, before any D16 accuracy
sweep, with
[x21_k24_d16_resource_preflight_v1.json](manifests/x21_k24_d16_resource_preflight_v1.json).
It requires a hash-valid successful `resource-preflight-k24-d14` state from the
base campaign, uses a new run root, and performs exactly one full-pool
complex64 epoch at logical batch 1024 with fixed complete q121 and
`P=1,370,688`.

```bash
X21_D16_PREFLIGHT_RUN_ROOT="/scratch/gcicy/post-v1/x21-k24-d16-resource-preflight-v1"

python scripts/run_gcicy_tn_gpu_workflow.py validate \
  --manifest experiments/manifests/x21_k24_d16_resource_preflight_v1.json \
  --frozen-root "$X21_FROZEN_ROOT" \
  --run-root "$X21_D16_PREFLIGHT_RUN_ROOT" \
  --set BASE_CAPACITY_RUN_ROOT="$X21_CAPACITY_RUN_ROOT" \
  --check-cuda --gpu 0

python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/x21_k24_d16_resource_preflight_v1.json \
  --frozen-root "$X21_FROZEN_ROOT" \
  --run-root "$X21_D16_PREFLIGHT_RUN_ROOT" \
  --set BASE_CAPACITY_RUN_ROOT="$X21_CAPACITY_RUN_ROOT" \
  --phase resource-certification --gpu 0
```

The terminal certificate records allocated/reserved memory, trainer runtime,
exit code, positivity, and source hashes.  Its gates are allocated memory at
most 22.5 GiB and reserved memory at most 23.5 GiB, with all reported numbers
finite and the validation metric strictly positive.  OOM is a failed resource
preflight, not permission to lower the registered batch size.  Passing this
certificate establishes feasibility only; it does not authorize D16 accuracy
training unless the separate base promotion decision is true.

## Constrained structure Auto Research

The capacity manifests above keep the chain representation fixed.  The
independent [architecture Auto Research protocol](ARCHITECTURE_AUTO_RESEARCH_V1.md)
searches the higher-leverage quintic structure changes instead: exact tree
re-association, nested internal-edge/shared-leaf rank growth, and registered
root residual ranks.  It freezes search indices, enforces matched three-seed
controls and parameter accounting, permits only typed mutations, and closes
after one durably claimed shadow evaluation.  The controller adjudicates
hash-bound evidence from the existing numerical workers; it does not execute
agent-authored shell commands or alter a trainer during a campaign.

## Quintic campaign quick start

Use a new run root; do not reuse the X21 run root or any historical output
directory.  On the GPU host, validate hashes and CUDA first:

```bash
QUINTIC_FROZEN_ROOT="/path/to/frozen/gcicy-workspace"
QUINTIC_RUN_ROOT="/scratch/gcicy/post-v1/quintic-kd-progressive-joint-paths-v1"

python scripts/run_gcicy_tn_gpu_workflow.py validate \
  --manifest experiments/manifests/quintic_kd_progressive_joint_paths_v1.json \
  --frozen-root "$QUINTIC_FROZEN_ROOT" \
  --run-root "$QUINTIC_RUN_ROOT" \
  --check-cuda --gpu 0

python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/quintic_kd_progressive_joint_paths_v1.json \
  --frozen-root "$QUINTIC_FROZEN_ROOT" \
  --run-root "$QUINTIC_RUN_ROOT" \
  --phase resource-preflight-d14 --gpu 0
```

Only after both registered D14 memory gates pass, start or resume the main
mechanism phase; the identical command is the recovery command after a host or
CUDA-process interruption:

```bash
python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/quintic_kd_progressive_joint_paths_v1.json \
  --frozen-root "$QUINTIC_FROZEN_ROOT" \
  --run-root "$QUINTIC_RUN_ROOT" \
  --phase main-mechanism --gpu 0

python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/quintic_kd_progressive_joint_paths_v1.json \
  --frozen-root "$QUINTIC_FROZEN_ROOT" \
  --run-root "$QUINTIC_RUN_ROOT" \
  --phase precision-replay --gpu 0
```

The optional `conditional-d16-preflight` phase performs no D16 accuracy fit.
All quintic fits are development-only complex64 training; the precision phase
creates hash-bound, zero-update complex128 twins and reevaluates the same
10,000 validation rows.  It does not claim that the complex64 and complex128
optimizer trajectories are equivalent.

## X21 scientific design

The X21 campaign uses the frozen X21 source metric and the same immutable train,
selection, and development-confirmation pools as the v1 development studies.
The historical blind pools are not referenced.

The experiment factors are separated as follows:

| Factor | Matched comparison |
| --- | --- |
| (k,D) capacity | (k=8,12,16,20), (D=8,10,12), three optimizer seeds, common function-preserving (k=4,D=5) origin |
| Progressive initialization | trained (D=8\to12) versus a saved-kappa endpoint continuation, and trained (k=8\to16) versus a re-estimated-kappa endpoint continuation |
| Parameter scope | all cores versus all cores + shared q=121 dictionary; reference (H) and positive floor fixed in both arms |
| Optimizer path | registered standard `2e-4 -> 3e-5` plateaus versus a matched gentle path from the same epoch-zero model |

At the targeted (k=16,D=12) endpoint, two standard endpoint continuations
control both extra training and the distinct normalization semantics of the
progressive arms: D expansion retains saved kappa, while k growth re-estimates
kappa after the degree change.  A cores-only 1.5e-5 continuation likewise
controls delayed dictionary unfreezing.  The full registered DAG contains 279
jobs and publishes 60 long-form result rows when complete.

Every capacity cell uses exact learned-norm site repetition from (k=4),
one-sided function-preserving bond activation, and an epoch-zero numerical
potential/metric equivalence gate.  Each learning-rate stage must reach a
validation plateau; an epoch cap starts a new round and is not itself an
accepted endpoint.

Replicate numbers are paired across the whole capacity surface: a given
replicate uses the same transfer, minibatch-permutation, and audit seeds for
every (k,D) cell.  Thus differences across cells do not also encode an
unrelated optimizer seed change.

The development pool may rank or diagnose paths, but it cannot support a new
final claim.  After all choices are fixed, promotion must be a separate
create-only manifest containing the selected model hashes.  Only then may a
new single-use blind pool be generated.

## Data and branch isolation

Use three distinct roots:

- `REPO_ROOT`: this Git experiment branch;
- `FROZEN_ROOT`: the existing workspace or a read-only Zenodo v1.0.0 mount;
- `RUN_ROOT`: a new external directory, preferably local SSD scratch on the GPU
  host.

The runner refuses a `RUN_ROOT` that overlaps any protected release/results
path.  On first execution it also refuses a non-empty directory without its
own `.gcicy-experiment-root` sentinel.  It then writes an immutable plan lock,
Git identity, frozen-input hashes, per-job states, logs, and output hashes.
Changing the manifest requires a new run root.

## Validate and inspect

From the experiment branch:

```bash
GCICY_FROZEN_INPUT_ROOT="/path/to/frozen/gcicy-workspace"
GCICY_EXPERIMENT_ROOT="/scratch/gcicy/post-v1/x21-kd-progressive-joint-paths-v1"

python scripts/run_gcicy_tn_gpu_workflow.py validate \
  --manifest experiments/manifests/x21_kd_progressive_joint_paths_v1.json \
  --frozen-root "$GCICY_FROZEN_INPUT_ROOT" \
  --run-root "$GCICY_EXPERIMENT_ROOT" \
  --check-cuda --gpu 0

python scripts/run_gcicy_tn_gpu_workflow.py plan \
  --manifest experiments/manifests/x21_kd_progressive_joint_paths_v1.json \
  --frozen-root "$GCICY_FROZEN_INPUT_ROOT" \
  --run-root "$GCICY_EXPERIMENT_ROOT"
```

Validation hashes every declared numerical input, checks that the release
commit is an ancestor of the active `exp/` branch, and requires the committed
worktree to be clean.  The exact source identity is locked into the run root
and rechecked before and after every job.  The observed PyTorch, CUDA, GPU
model, capability, and configured slot are locked on first execution as well.

## Run and resume

Run the preparation and maximum-cell allocation preflight first:

```bash
python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/x21_kd_progressive_joint_paths_v1.json \
  --frozen-root "$GCICY_FROZEN_INPUT_ROOT" \
  --run-root "$GCICY_EXPERIMENT_ROOT" \
  --phase resource-preflight --gpu 0
```

The phase selector automatically includes dependencies.  After the preflight
fits the registered GPU profile, start or resume the capacity surface:

```bash
python scripts/run_gcicy_tn_gpu_workflow.py run \
  --manifest experiments/manifests/x21_kd_progressive_joint_paths_v1.json \
  --frozen-root "$GCICY_FROZEN_INPUT_ROOT" \
  --run-root "$GCICY_EXPERIMENT_ROOT" \
  --phase capacity-grid --gpu 0
```

The same command accepts `--phase progressive-paths`, `--phase
parameter-scope`, or `--phase optimizer-paths`; dependencies are included
automatically.

Reissuing the identical command is the supported recovery procedure.  A
content-verified succeeded job is skipped.  An interrupted plateau round
resumes the current model, Adam state, next epoch, RNG states, history,
fixed-κ normalization, best checkpoint, and early-stop counters.  This is
recorded as crash recovery and is distinct from a scientifically planned
continuation between (k,D), parameter scopes, or learning-rate stages.

One campaign-level `flock` prevents concurrent writers from mutating the same
state tree.  The GPU is separately held with an OS `flock` for the entire
subprocess lifetime, so two campaigns cannot silently share the same
configured CUDA slot.  Job state and logs are updated by heartbeat; completion
requires the declared files, hashes, and JSON gates, not a marker file.

## Failure policy

- `SIGSEGV`, `SIGILL`, and `SIGTRAP` may retry the exact registered command
  and resume its checkpoint a bounded number of times; no other exit status is
  retryable in this manifest.
- CUDA OOM fails the resource preflight or job.  The workflow never silently
  lowers precision, (k), (D), or batch size.
- Non-finite loss, positivity failure, input/hash mismatch, or failed
  equivalence is a scientific/permanent failure and must not trigger a new
  protocol under the same plan hash.
- A smaller batch is a different objective here because loss normalization and
  tail risk are minibatch-dependent.  If required, it must be preregistered in
  a new manifest/run root and reported as a different optimizer path.
- Low disk space must pause the host service; checkpoints are never deleted by
  the runner.

For unattended operation, run the command as a host service with the whole
process group stopped together (for example, systemd with
`KillMode=control-group` and automatic restart).  GitHub Actions remains a CPU
schema/test gate and is not the long-lived numerical scheduler.

## Monitor and aggregate

```bash
python scripts/run_gcicy_tn_gpu_workflow.py status \
  --manifest experiments/manifests/x21_kd_progressive_joint_paths_v1.json \
  --frozen-root "$GCICY_FROZEN_INPUT_ROOT" \
  --run-root "$GCICY_EXPERIMENT_ROOT"

python scripts/run_gcicy_tn_gpu_workflow.py summarize \
  --manifest experiments/manifests/x21_kd_progressive_joint_paths_v1.json \
  --frozen-root "$GCICY_FROZEN_INPUT_ROOT" \
  --run-root "$GCICY_EXPERIMENT_ROOT" \
  --allow-incomplete
```

Aggregation reads only content-verified succeeded result jobs.  Each snapshot
is create-only under `aggregates/<plan-sha256>/<timestamp>/` and contains a
long-form JSON table, CSV table, and checksums.  Without `--allow-incomplete`,
any missing required cell blocks the aggregate rather than silently reporting
a favorable subset.

## Resource expectations

The frozen RTX 4090 records provide planning—not allocation guarantees.  At
(D=8), peak allocated memory grew from about 2.89 GB at (k=8) to 8.03 GB
at (k=20).  The previous (k=16,D=12) run used about 16.46 GB.  The
one-epoch (k=20,D=12) preflight therefore loads the full registered train and
selection datasets and exercises forward, backward, and Adam state at the
largest core grid cell before any multi-day sweep begins.  Its model is never
used as a scientific initialization or result.
