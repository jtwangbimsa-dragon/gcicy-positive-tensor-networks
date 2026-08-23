# Constrained quintic architecture Auto Research v1

This is a post-v1 development control plane. It does not modify arXiv v1,
Zenodo v1.0.0, release tags, historical results, or any registered X21
campaign. It also does not allow an agent to edit or execute arbitrary model
code. An agent may propose only the typed actions accepted by
`gcicy-quintic-architecture-action-v1`.

## Scientific contract

The initial implementation is quintic-first because the repository already
has exact transport and numerical workers for these operations:

1. exact five-leaf `4+1 -> 2+3` tree re-association;
2. nested internal-edge rank growth;
3. one-row shared-leaf rank growth;
4. rank `{2,4,8,16,25}` root-residual insertion on the `2+3` tree;
5. local activation of new channels;
6. full-parameter relaxation against a shared no-growth control.

Every structural mutation is automatically followed by `local-activate` and
`matched-relax`. A proposal cannot replace those stages, add a command, name a
new trainer, or change precision. Search is complex64. Only a frozen finalist
may receive a zero-optimizer-update complex128 replay.

The numerical adapters remain the existing scripts:

- `train_generic_quintic_adaptive_direct_blocks.py` for nested rank activation;
- `propose_generic_quintic_root_residual.py` and
  `train_generic_quintic_root_active_subspace.py` for root residuals;
- `compare_generic_quintic_tree_joint_relaxation.py` for the matched
  candidate/control relaxation.

The control plane consumes their normalized evidence JSON. It intentionally
does not invoke a shell command itself.

## Data boundary

`init` verifies and locks every search input by SHA-256, then creates one
deterministic `search_indices.json`. If train and evaluation rows share a
population, the materialized index sets are disjoint. All actions and evidence
must carry the locked index hash; a per-round data seed is not permitted.

No shadow path or hash is registered during search. After a three-seed
candidate has been promoted and frozen, the control plane requires its
zero-update complex128 replay. It then durably claims exactly one shadow
pool/evaluator contract before evaluation, accepts evidence only for that
claim, and closes the campaign whether that shadow gate passes or fails.

There is no final-blind command. A final claim requires a separate create-only
manifest, a different run root, and a new single-use pool after the candidate
and protocol hashes have been frozen.

## Promotion gate

The default search gate requires:

- exactly three registered paired seeds;
- the same frozen search indices for every candidate;
- the registered complex64 precision;
- exact-transport potential and metric audits within tolerance;
- the registered local-activation budget;
- identical registered full-relaxation budgets for candidate and control;
- the identical minibatch plan for candidate and control;
- reported before/after parameter counts, with rank-growth deltas equal to the
  registered action and the complete v1 action adding at most 10,000 real
  parameters;
- one byte-identical no-growth control evidence record per seed across all
  candidates (checkpoint, metrics, budget, minibatch plan, and parameter
  count);
- positive sampled metrics and zero nonpositive samples;
- bounded q999 and CVaR degradation;
- sigma and chi improvement in at least two seeds;
- positive paired sigma and E2 confidence bounds in all three seeds;
- median relative sigma and chi gains of at least 0.2%;
- no seed regression.

Among passing candidates the round winner is selected lexicographically by
median sigma gain, median chi gain, added real parameters, and candidate ID.
The single shadow gate is stricter: q999 and CVaR must not degrade.

## Durable state

Each campaign run root contains:

```text
.gcicy-architecture-auto-research-root
protocol.lock.json
search_indices.json
ledger.json
rounds/
  round_001/
    candidates/<candidate-id>/action.json
    candidates/<candidate-id>/evidence.json
    adjudication.json
frozen_candidate.json
complex128_replay.json
shadow_claim.json
shadow_evidence.json
shadow_adjudication.json
```

All ledger transitions are serialized with an OS file lock and published by
atomic replace. Immutable artifacts are written before their ledger transition
with create-or-verify semantics, so a restart after either half of the
transition can safely resume without overwriting evidence. Events form a
SHA-256 chain and the complete state has its own integrity hash, so accidental
editing fails closed. Reissuing an identical completed operation remains
idempotent even after later freeze/replay/shadow transitions; conflicting or
hash-modified action, replay, adjudication, or shadow evidence is rejected.
The `status` command performs the same full artifact audit before returning the
ledger, rather than trusting file presence alone.

## CLI

```bash
python scripts/run_quintic_architecture_auto_research.py init \
  --protocol /path/to/protocol.json \
  --run-root /scratch/quintic-architecture-auto-v1

python scripts/run_quintic_architecture_auto_research.py register-action \
  --run-root /scratch/quintic-architecture-auto-v1 \
  --action /path/to/action.json

python scripts/run_quintic_architecture_auto_research.py record-evidence \
  --run-root /scratch/quintic-architecture-auto-v1 \
  --evidence /path/to/candidate-a-evidence.json

python scripts/run_quintic_architecture_auto_research.py adjudicate-round \
  --run-root /scratch/quintic-architecture-auto-v1 \
  --round 1 \
  --evidence /path/to/candidate-a-evidence.json \
  --evidence /path/to/candidate-b-evidence.json

python scripts/run_quintic_architecture_auto_research.py freeze \
  --run-root /scratch/quintic-architecture-auto-v1

python scripts/run_quintic_architecture_auto_research.py record-replay \
  --run-root /scratch/quintic-architecture-auto-v1 \
  --evidence /path/to/complex128-replay.json

python scripts/run_quintic_architecture_auto_research.py claim-shadow \
  --run-root /scratch/quintic-architecture-auto-v1 \
  --claim /path/to/shadow-claim.json

python scripts/run_quintic_architecture_auto_research.py record-shadow \
  --run-root /scratch/quintic-architecture-auto-v1 \
  --evidence /path/to/single-shadow-evidence.json
```

The checked-in JSON Schema is
`experiments/schemas/gcicy-quintic-architecture-action-v1.schema.json`.
Runtime validation is stricter than JSON Schema where a constraint relates two
fields, such as `source_dimension < target_dimension <= structural_maximum` or
shared-leaf growth being exactly one row.

## Executable Round 1 bridge

The first numerical bridge is intentionally smaller than the general action
language. It accepts only internal edge 6 growth `25 -> 39` from the pinned
leaf-rank10 checkpoint, whose exact structural increment is 9,800 real
parameters. The 10,000 limit is a **total-action cap in bridge v1**. The
historical adaptive worker's option with the same numeric value is a per-block
optimizer limit; it does not authorize larger actions such as `25 -> 75`
(35,000 new real parameters). A wider capacity envelope must be an explicit
v2 protocol.

The bridge applies the preregistered deterministic `partition_seed` to shuffle
the locked 35,000 native indices, then maps 30,000 rows to fit and 5,000 rows
to selection. This avoids turning the controller's canonical sorted storage
order into a row-number-biased split. The separately locked 5,000 indices
address the development-evaluation pool. That pool is evaluated only after
both matched-relax arms finish and supplies paired confidence intervals only;
it cannot affect optimization, early stopping, checkpoint selection, or the
worker winner. Historical confirmation paths are rejected.

The staged CLI is create-only and resumable across completed seed/stage pairs.
On a fresh campaign, controller initialization and fixed-action registration
must precede the shown `prepare` command; the one-command `execute` path below
performs those steps automatically:

```bash
python scripts/run_quintic_architecture_round1_bridge.py prepare \
  --campaign-run-root /scratch/quintic-architecture-auto-v1 \
  --candidate-id edge6-rank39 \
  --output-root /scratch/quintic-edge6-round1 \
  --baseline-checkpoint 202608231=/path/to/pinned-parent.pt \
  --baseline-checkpoint 202608232=/path/to/pinned-parent.pt \
  --baseline-checkpoint 202608233=/path/to/pinned-parent.pt

python scripts/run_quintic_architecture_round1_bridge.py run \
  --bridge-root /scratch/quintic-edge6-round1

python scripts/run_quintic_architecture_round1_bridge.py normalize \
  --bridge-root /scratch/quintic-edge6-round1
```

For an unattended GPU service, the fixed `execute` command performs protocol
initialization, registers the sole edge6 action, prepares or resumes all three
seeds, normalizes the evidence, records it, and adjudicates Round 1:

```bash
python scripts/run_quintic_architecture_round1_bridge.py execute \
  --protocol experiments/protocols/generic_quintic_architecture_round1_edge6_v1.json \
  --wait-for-user-unit preceding-gpu-campaign.service \
  --campaign-run-root /scratch/quintic-architecture-auto-v1 \
  --output-root /scratch/quintic-edge6-round1 \
  --baseline-checkpoint 202608231=/path/to/pinned-parent.pt \
  --baseline-checkpoint 202608232=/path/to/pinned-parent.pt \
  --baseline-checkpoint 202608233=/path/to/pinned-parent.pt
```

When a predecessor GPU campaign is named, `execute` waits for that exact user
service and proceeds only after systemd reports `Result=success` and exit status
zero. Immediately before launching a CUDA worker it also rejects any remaining
compute process reported by `nvidia-smi`. A failed, missing, or uninspectable
predecessor aborts before Round 1. These guards prevent two campaigns from
silently sharing the same GPU.

An exact retry returns an existing adjudication or resumes at the first
unpublished stage. A partially created worker directory is never overwritten;
it remains a fail-closed diagnostic artifact.

`prepare` requires a clean tracked Git worktree and binds the commit, worker
and dependency-source hashes, baseline checkpoint/source-report hashes, every
input-pool hash, canonical index-array value hashes, and per-seed canonical
batch-plan hashes. It also records Python, NumPy, PyTorch/CUDA-build, GPU, and
driver metadata. `run` uses fixed argument construction and `shell=False`;
the action has no command, environment, or worker escape hatch. `normalize`
revalidates worker configuration and raw report hashes and emits controller
search evidence without inventing missing metrics or confidence intervals.

The three promotion seeds in this bridge start from the same pinned upstream
checkpoint. They are independent channel-activation/optimization replicas with
separate optimizer seeds and matched batch plans, not three independently
trained upstream parent models. Any paper-level claim must preserve that
distinction and use a separately frozen final evaluation workflow.
