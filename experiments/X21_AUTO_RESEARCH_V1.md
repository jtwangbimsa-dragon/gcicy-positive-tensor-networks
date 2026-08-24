# X21 evaluator-driven auto research v1

This is an independent post-v1 development controller. It does not modify the
arXiv v1 source, Zenodo 1.0.0 artifact, the failed
`x21-architecture-capacity-v1` run root, or any frozen output tree.
The v1 ledger remains the baseline/recovery compatibility plane. Its
historical development hash is no longer authorized for scientific evidence;
v1 actions can support diagnostic training only. A scientific round must use
the fresh-pool v2 migration described in
[`X21_FRESH_DEVELOPMENT_POOL_V2.md`](X21_FRESH_DEVELOPMENT_POOL_V2.md).

## Two disjoint evidence planes

Operational checkpoint recovery and scientific model selection use different
schemas, sentinels, ledgers, locks, directories, and CLI operations.

The recovery plane accepts only an exact checkpoint continuation certificate:

- source checkpoint, epoch boundary, training-semantics and frozen-input hashes;
- exact implementation match;
- restored optimizer and RNG state;
- clean terminal process exit and hash-bound output artifacts.

Its schema has no metric fields and its ledger has no leaderboard. For the
known X21 interruption, the source is the retained `k20,D14,r2` precision
checkpoint at epoch 36. Recovery attempts, native return codes, and process
failures remain operational provenance. They are never candidate actions,
replicates, improvements, or votes.

After seed 2 has an exact-recovery certificate and seeds 1 and 3 have their own
completion certificates, the scientific plane may lock a three-model baseline
family. All three rows must supply their normalized certificates inline. The
controller recomputes each canonical hash, requires three different
certificate and checkpoint hashes, and checks every certificate against the
protocol-registered source campaign, plan, job ID, job digest, seed and output
checkpoint. The recovery row is additionally bound to the registered recovery
campaign, source model/checkpoint, epoch 36 -> 37 boundary, training-semantics
hash and frozen-input hash. The inline bodies are stripped before publishing
the baseline family. Recovery metrics or attempt history never enter the
leaderboard.

## Scientific contract

The historical locked protocol is
[`x21_auto_research_v1.json`](protocols/x21_auto_research_v1.json). Search uses
the existing immutable X21 source and common pools by their SHA-256 digests.
The development pool is explicitly search-only and has already informed model
selection. A paper-level claim needs a separately frozen finalist and a new
single-use evaluation manifest.

Because that development hash is the old `X21_confirmation_*` artifact, the
controller now rejects every attempt to record v1 scientific evidence. The
fresh-pool manager produces a create-only v2 protocol whose data-contract hash
also binds the new selection-labelled pool's complete provenance. Only that v2
protocol may record or adjudicate the rounds below.

Every round has exactly one candidate and one mutable axis. Control and
candidate use:

- the same three registered parent seeds;
- 2,304 optimizer updates;
- the same point pools and per-seed minibatch plan;
- the same batch size within a round;
- the same gradient clip, scheduler, evaluation size, and 2,000-replicate
  paired fibre-cluster bootstrap;
- complex64 optimization and search evaluation.

The four rounds are fixed:

1. optimizer path: learning rate `3e-5 -> 1.5e-5`;
2. joint scope: fixed dictionary -> trainable shared q=121 dictionary;
3. bond growth: `D=14 -> 16` by the exact one-sided lift;
4. degree/site growth: `k=20 -> 24` by the exact repeat-sites lift.

Round 3 requires a hash-bound resource certificate. Round 4 also requires one
when its parent is D16. The controller recomputes the certificate hash and
binds its round, action kind, candidate-metadata hash, attempted ladder prefix,
selected batch, outcome, and raw preflight-report hash. A bare digest is
rejected. The resource ladder is `1024, 768, 640, 512`; resource
probes are not accuracy jobs. If no registered batch size fits, a create-only
`resource-infeasible` artifact advances the structural round without adding a
leaderboard row. Thus an infeasible D16 does not prevent the k24,D14 question.

## Promotion and stopping

Promotion requires all gates:

- exactly three seeds, distinct per-seed control/candidate checkpoint hashes,
  and the locked data contract;
- identical fixed budgets and minibatch plans for each paired arm;
- exact epoch-zero function/metric transport within tolerance;
- positive sampled metrics and zero nonpositive points;
- q999 and 1% CVaR degradation at most 0.5%;
- sigma and chi improvement in at least two seeds;
- positive paired sigma and chi 95% confidence lower bounds in all three seeds;
- median relative sigma and chi gains of at least 0.2%;
- no sigma or chi regression in any seed.

A passing candidate becomes the next parent family. A rejected candidate leaves
the parent unchanged. Missing, malformed, hash-inconsistent, wrong-seed,
wrong-budget, wrong-order, or recovery-contaminated evidence does not advance
the ledger. The campaign stops after round 4. Only then may a separate workflow
freeze the champion and perform a zero-update complex128 replay.

## Controller CLI

Recovery is initialized and recorded under its own root:

```bash
python scripts/run_x21_auto_research.py init-recovery \
  --run-root /scratch/x21-recovery-v1 \
  --campaign-id x21-k20d14-r2-exact-recovery-20260824

python scripts/run_x21_auto_research.py record-recovery \
  --run-root /scratch/x21-recovery-v1 \
  --evidence /scratch/x21-recovery-v1/normalized_recovery_evidence.json
```

The v1 compatibility ledger may register the three baseline certificates and
checkpoint hashes, but it cannot record scientific evidence:

```bash
python scripts/run_x21_auto_research.py init \
  --run-root /scratch/x21-auto-research-v1 \
  --protocol experiments/protocols/x21_auto_research_v1.json

python scripts/run_x21_auto_research.py register-baseline \
  --run-root /scratch/x21-auto-research-v1 \
  --family /scratch/x21-baseline-family.json
```

After the fresh-pool workflow has emitted and validated its v2 lock, initialize
an independent scientific root and register the same hash-bound baseline:

```bash
python scripts/run_x21_auto_research.py init \
  --run-root /scratch/x21-auto-research-v2 \
  --protocol /scratch/x21-fresh-development-v2/x21_auto_research_v2.lock.json

python scripts/run_x21_auto_research.py register-baseline \
  --run-root /scratch/x21-auto-research-v2 \
  --family /scratch/x21-baseline-family.json

python scripts/run_x21_auto_research.py register-action \
  --run-root /scratch/x21-auto-research-v2 \
  --action /scratch/round-1-action.json

python scripts/run_x21_auto_research.py record-evidence \
  --run-root /scratch/x21-auto-research-v2 \
  --evidence /scratch/round-1-evidence.json

python scripts/run_x21_auto_research.py adjudicate-round \
  --run-root /scratch/x21-auto-research-v2 \
  --round 1
```

The family file and the numerical bridge's parent bindings are generated from
the normalized certificates and actual files; they are not handwritten. The
legacy family field named `checkpoint_sha256` identifies the runnable plateau
model. Recovery checkpoints remain separately hash-bound provenance and are
never passed as `--initial-model`:

```bash
python scripts/normalize_x21_baseline_provenance.py family \
  --protocol /scratch/x21-fresh-development-v2/x21_auto_research_v2.lock.json \
  --certificate 8660001=/scratch/cert-seed1.json \
  --certificate 8660002=/scratch/cert-seed2-recovery.json \
  --certificate 8660003=/scratch/cert-seed3.json \
  --model 8660001=/scratch/seed1/plateau_model.pt \
  --model 8660002=/scratch/seed2/plateau_model.pt \
  --model 8660003=/scratch/seed3/plateau_model.pt \
  --checkpoint 8660001=/scratch/seed1/checkpoint.pt \
  --checkpoint 8660002=/scratch/seed2/checkpoint.pt \
  --checkpoint 8660003=/scratch/seed3/checkpoint.pt \
  --out /scratch/x21-baseline-family.json \
  --bindings-out /scratch/x21-parent-bindings.json
```

## Executable bridge and host probes

Round 1 has an executable matched control/candidate bridge. For each parent,
run three create-only X21 probes and feed their standard receipts to
`certify_host_stability.py`. Each probe performs one complete-pool epoch (192
optimizer updates) plus selection evaluation under the shared GPU lock. It is
diagnostic only and cannot enter the leaderboard.

```bash
python scripts/run_x21_host_gpu_probe.py prepare \
  --protocol /scratch/x21-fresh-development-v2/x21_auto_research_v2.lock.json \
  --source-artifact /frozen/X21/source.npz \
  --train-common-pool /frozen/X21/train.npz \
  --selection-common-pool /frozen/X21/selection.npz \
  --parent-model /scratch/seed1/plateau_model.pt \
  --parent-model-sha256 <SEED1_MODEL_SHA256> \
  --parent-seed 8660001 \
  --probe-id x21-seed1-probe1 \
  --output-root /scratch/x21-host-probes/seed1-probe1

python scripts/run_x21_host_gpu_probe.py run \
  --root /scratch/x21-host-probes/seed1-probe1
```

After three receipts per parent and a passing certificate for every pending
seed, prepare and run the scientific bridge:

```bash
python scripts/run_x21_scientific_bridge.py prepare \
  --campaign-run-root /scratch/x21-auto-research-v2 \
  --output-root /scratch/x21-auto-research-v2-round1 \
  --parent-bindings /scratch/x21-parent-bindings.json \
  --source-artifact /frozen/X21/source.npz \
  --train-common-pool /frozen/X21/train.npz \
  --selection-common-pool /frozen/X21/selection.npz \
  --fresh-development-binding /scratch/x21-fresh-development-v2/fresh_development_binding.json

python scripts/run_x21_scientific_bridge.py run \
  --output-root /scratch/x21-auto-research-v2-round1 \
  --host-stability-certificate 8660001=/scratch/host-seed1.json \
  --host-stability-certificate 8660002=/scratch/host-seed2.json \
  --host-stability-certificate 8660003=/scratch/host-seed3.json

python scripts/run_x21_scientific_bridge.py normalize \
  --output-root /scratch/x21-auto-research-v2-round1
python scripts/run_x21_scientific_bridge.py adjudicate \
  --output-root /scratch/x21-auto-research-v2-round1
```

Host certificates expire after thirty minutes. A resumed `run` accepts newly
issued certificates, appends their authorization entries, preserves all prior
attempt seals, and revalidates the parent, source and host before each child.

Every status operation revalidates the ledger hash chain and all referenced
baseline, action, evidence, adjudication, and resource-skip artifacts. Exact
retries are idempotent; conflicting retries fail closed.

The controller intentionally accepts no trainer command, shell fragment,
environment override, input path, arbitrary metric, or free-form mutation.
The separate executable bridge builds fixed argument vectors for the existing
training, audit, tail, and paired-bootstrap scripts, requires the shared GPU
and host-stability certificates, and publishes evidence only after the v2
fresh binding and every raw artifact hash are reverified. The present
executable bridge covers Round 1; later X21 catalog rounds remain controller
states until equally narrow numerical adapters are reviewed and registered.
