# Quintic architecture Auto Research manager v2

This post-v1 manager turns the executable quintic architecture Round 1 into the
first node of a durable four-round research queue. It does not modify arXiv v1,
Zenodo v1.0.0, release tags, the frozen result root, or the Round 1 bridge. It
does not open historical confirmation or final-blind data.

The implementation boundary is split between the durable queue manager and
reviewed numerical bridges:

- the existing edge-6 `25 -> 39` Round 1 bridge is executable;
- the Round 2 branches remain historical/preregistered records; the completed
  scale-0.25 strict rejection can be imported only through its original
  controller/bridge hash chain and is labelled
  `historical-auto-research-not-manager-launched`;
- the reviewed R3 parameter-scope and R4 optimizer-path adapters are labelled
  `available`; `run-next` emits a create-only execution handoff and the
  separate paired bridge must consume that exact handoff;
- the manager still refuses to synthesize a command from a catalog entry.

No proposal contains a command, script, environment, working directory, or
input/output path. The manager resolves parent checkpoint paths from the
hash-audited Round 1/Round 2 lineage. A paired bridge cannot replace them with
caller-selected parents.

## Registered decision tree

Round 1 is the existing three-seed exact rank expansion:

```text
edge 6: 25 -> 39, 9,800 new real parameters
```

After auditing the controller ledger, bridge plan, normalized evidence,
matched reports, and checkpoint hashes, the manager records exactly one of
three outcomes:

1. `promoted`: the scientific gates passed. Round 2 is the progressive exact
   `39 -> 53` proposal.
2. `scientific-rejected`: workers completed and the evidence was valid, but no
   candidate passed promotion. Round 2 becomes the preregistered conservative
   scale-0.25 `25 -> 39` alternative.
3. `technical-failure`: a worker stage is `failed` or `incomplete-output`.
   The campaign becomes `blocked-technical-failure`; this is never reported as
   evidence against the scientific hypothesis.

Both scientific branches then preregister:

- Round 3: internal-only control versus all-parameter joint training;
- Round 4: all-parameter cosine control versus all-parameter constant-LR path.

Scientific rejection continues to the next catalog hypothesis with the prior
champion. A technical failure stops the queue. The existing R2 scale-0.25
result can be registered as immutable historical evidence only after the
manager recomputes its controller, bridge, action, evidence, adjudication,
data, batch-plan, parent and three-seed bindings. It must be a strict
scientific rejection and is never retroactively called a manager-launched
experiment. R3 and R4 use their own create-only paired
plan/ledger/evidence/adjudication state machine.

## Fixed development contract

Every catalog recipe binds:

- three optimizer seeds `202608231`, `202608232`, `202608233`;
- complex64 search;
- the exact R1 protocol SHA, fixed search-index SHA, four development input
  SHAs, and all three 600-step matched batch-plan file/value SHAs;
- 30,000 fit, 5,000 selection, and 5,000 post-selection development-evaluation
  examples;
- 600 matched-relax optimizer updates per arm at batch size 1,024;
- at most 1,200 candidate-only local-activation updates for rank growth;
- learning rate `3e-6` and gradient clipping at 1.0;
- median relative sigma and chi gains of at least 0.2%;
- at least two improved seeds and positive paired sigma/E2 confidence lower
  bounds in all three seeds;
- no seed regression, positive metrics, and at most 0.5% q999/CVaR
  degradation.

Rank actions add at most 10,000 real parameters. Parameter-scope and optimizer
actions must preserve total parameter count and differ in exactly the named
axis. The catalog validator rejects execution fields and rejects any claim that
an unavailable adapter is executable. Every `status` operation also rehashes
the champion checkpoints, source reports, bridge plan/ledger/evidence and R1
adjudication; a path that still exists but changed content fails closed.

The paired numerical worker supports independent `internal`/`all`
trainable scopes and `cosine`/`constant` schedulers while enforcing an
unchanged batch-plan digest and exact equality of frozen tensors. This removes
the worker-level gap for Rounds 3 and 4. The reviewed paired bridge fixes the
worker argument vector, verifies all source/data/index/batch/parent hashes,
waits on the shared GPU lock, requires a fresh per-parent host-stability
certificate, resumes seed by seed, normalizes raw reports, and applies the
catalog promotion gate. Short-lived certificates authorize attempts rather
than becoming permanent plan inputs: a later run may append a newly issued
certificate while retaining every earlier authorization and attempt seal. It
has no confirmation or blind-data argument.

## Commands

Initialize a manager beside, not inside, either existing Round 1 root:

```bash
python scripts/run_quintic_architecture_multi_round.py init \
  --catalog experiments/protocols/generic_quintic_architecture_multi_round_v2.json \
  --run-root /scratch/quintic-architecture-manager-v2 \
  --round1-campaign-root /scratch/quintic-architecture-auto-r1-controller \
  --round1-bridge-root /scratch/quintic-architecture-auto-r1-workers
```

Audit and synchronize Round 1. Reissuing the command is idempotent:

```bash
python scripts/run_quintic_architecture_multi_round.py sync-r1 \
  --run-root /scratch/quintic-architecture-manager-v2
```

Inspect the hash-audited durable state:

```bash
python scripts/run_quintic_architecture_multi_round.py status \
  --run-root /scratch/quintic-architecture-manager-v2
```

Import the completed scale-0.25 R2 strict rejection. This does not claim that
the v2 manager launched it:

```bash
python scripts/run_quintic_architecture_multi_round.py sync-historical-r2 \
  --run-root /scratch/quintic-architecture-manager-v2 \
  --bridge-root /scratch/quintic-auto-r2-scale025-workers
```

Materialize the hash-bound R3 handoff, then prepare the paired bridge with one
fresh host certificate entry for each registered seed. Parent paths come only
from the handoff:

```bash
python scripts/run_quintic_architecture_multi_round.py run-next \
  --run-root /scratch/quintic-architecture-manager-v2

python scripts/run_quintic_paired_auto_research_bridge.py prepare \
  --manager-root /scratch/quintic-architecture-manager-v2 \
  --execution-handoff /scratch/quintic-architecture-manager-v2/rounds/round_003/execution_handoff.json \
  --output-root /scratch/quintic-auto-r3 \
  --host-certificate 202608231=/scratch/host-1.json \
  --host-certificate 202608232=/scratch/host-2.json \
  --host-certificate 202608233=/scratch/host-3.json

python scripts/run_quintic_paired_auto_research_bridge.py run \
  --root /scratch/quintic-auto-r3 \
  --host-certificate 202608231=/scratch/host-1.json \
  --host-certificate 202608232=/scratch/host-2.json \
  --host-certificate 202608233=/scratch/host-3.json

python scripts/run_quintic_paired_auto_research_bridge.py normalize \
  --root /scratch/quintic-auto-r3
python scripts/run_quintic_paired_auto_research_bridge.py adjudicate \
  --root /scratch/quintic-auto-r3

python scripts/run_quintic_architecture_multi_round.py record-round-result \
  --run-root /scratch/quintic-architecture-manager-v2 \
  --round 3 \
  --bridge-root /scratch/quintic-auto-r3
```

If a certificate expires before the next worker begins, issue a new one and
rerun `run` with the new `SEED=PATH` mapping. Completed attempts are not
repeated and earlier certificate files remain part of the audited ledger.

After a scientific Round 1 decision, the manager writes:

```text
.gcicy-quintic-architecture-multi-round-v2-root
catalog.lock.json
state.json
queue.json
rounds/
  round_001/decision.json
  round_002/proposal.json
```

The resolved Round 2 proposal binds the exact three-seed champion checkpoint
paths and hashes, but sets `execution_authorized` to `false`. Rounds 3 and 4
remain `preregistered-awaiting-parent` until their preceding champion exists.

## Claim boundary

This manager can produce a development champion and research conclusions only.
It has no complex128 optimizer, shadow, confirmation, or blind command. A later
precision replay or new single-use evaluation requires a separate frozen
manifest and run root after all model and protocol hashes are fixed.
