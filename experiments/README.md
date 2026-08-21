# Independent post-v1 GPU experiments

This directory is the control plane for new experiments.  It does not contain
or regenerate the arXiv v1 source, Zenodo v1.0.0 archive, paper tables, or any
historical `outputs/pipeline` result.

Start with [BASELINE_V1.md](BASELINE_V1.md), then read the registered
[STUDY_PROTOCOL.md](STUDY_PROTOCOL.md).  The runnable X21 campaign is
[x21_kd_progressive_joint_paths_v1.json](manifests/x21_kd_progressive_joint_paths_v1.json).

## Scientific design

The campaign uses the frozen X21 source metric and the same immutable train,
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
