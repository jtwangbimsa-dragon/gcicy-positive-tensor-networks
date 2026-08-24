# Evaluator-driven Auto Research v2

This document defines the control boundary for the post-release gCICY tensor
network experiments.  It does not change the arXiv v1 source, the Zenodo
v1.0.0 deposit, or any historical result directory.  Every execution uses a
new git branch, checkout, and run root.

## Objective

The objective is not to maximize one noisy validation number.  It is to build
an auditable sequence of positive tensor-network metric families whose paired,
fixed-budget development accuracy improves across independent optimizer seeds.
The agent may choose only from a preregistered, typed action catalogue.  It may
not edit the evaluator, data partition, promotion thresholds, or blind-data
policy after seeing a candidate result.

## Two independent lanes

Operational recovery and scientific search are separate state machines.

1. The recovery lane may replay an exact checkpoint after a process, CUDA, or
   host failure.  It records environment identity, input and checkpoint hashes,
   attempts, signals, and health probes.  Recovery output is never eligible for
   a leaderboard or promotion.
2. The scientific lane starts only from a sealed parent family.  Each round
   changes one declared factor, runs a matched no-change control and candidate,
   evaluates the same fixed indices, and either promotes the three-seed family
   or rolls back to the previous champion.

A recovered model can become an input to a later scientific experiment only
after a separate zero-update equivalence audit and a newly registered matched
experiment.  Recovery metrics themselves cannot be imported as scientific
evidence.

## Closed loop

Each campaign repeats the following durable transitions:

1. **Observe:** verify source, environment, frozen inputs, current champion,
   host-health certificate, and remaining compute budget.
2. **Propose:** select the next typed action from the frozen catalogue.  Only
   one scientific factor may differ between control and candidate.
3. **Preregister:** seal the parent hashes, seeds, fixed indices and batch
   plans, update budget, evaluation metrics, thresholds, and expected outputs.
4. **Run:** acquire the shared GPU lock and execute create-only worker attempts.
   A native crash is an operational event, not a poor accuracy result.
5. **Evaluate:** replay endpoints without updates, require finite values and a
   positive metric, then compute paired per-seed and aggregate comparisons.
6. **Promote or roll back:** update the champion only when every hard gate
   passes.  A scientifically rejected candidate leaves the champion unchanged
   and advances to the next hypothesis.  A protocol violation fails closed.

Every state transition and referenced artifact is content-addressed.  Resume
means verifying and continuing an identical transition; it never overwrites a
completed attempt.

## Shared promotion contract

Unless a stricter geometry-specific protocol is sealed, a candidate must meet
all of the following conditions:

- exactly three registered optimizer seeds;
- zero non-finite values and zero nonpositive metric counts;
- positive minimum metric eigenvalue for every control and candidate;
- median relative improvements in both sigma and chi of at least 0.2%;
- at least two of three seeds improve both sigma and chi;
- every registered paired confidence lower bound is positive;
- no seed regresses in the primary metric;
- q999 and one-percent CVaR degrade by at most 0.5%;
- source, checkpoint, data, index, batch-plan, budget, implementation, and
  output hashes agree with the preregistration.

Complex64 is used for search.  A finalist is frozen before a zero-update
complex128 replay.  Historical confirmation and blind pools are not available
to either search controller.  A fresh blind experiment requires a separate
post-freeze manifest and explicit authorization.

## X21 catalogue

The failed 2026-08-23 capacity run remains an immutable operational record.  An
exact epoch-36 replay belongs only to the recovery lane.  Once a valid
three-seed parent family exists, the scientific catalogue is:

1. fixed-update gentle learning-rate path versus the registered standard path;
2. cores-only versus joint dictionary-and-core training;
3. progressive D14 to D16 growth after a non-scientific memory preflight;
4. progressive k20 to k24 growth, with k24/D16 evaluated only after its own
   non-scientific batch-size preflight.

Each scientific comparison uses the same 12 epochs (2,304 optimizer updates at
batch size 1,024), the same point order, and the same development evaluator.
Resource-preflight results determine feasibility only and never rank models.

## Generic-quintic catalogue

The structural catalogue begins from the sealed leaf-rank-10, internal-rank-25
parent family:

1. grow internal edge 6 from 25 to 39 (9,800 new real parameters), then run a
   matched full-model relaxation;
2. if Round 1 promotes, grow the same edge from 39 to 53 using the trained
   parent; otherwise test the preregistered lower activation scale for the same
   25-to-39 action;
3. compare internal-only and all-parameter relaxation from an identical
   champion checkpoint;
4. compare cosine and constant optimization paths with identical trainable
   parameters, learning rate, update count, batches, and evaluator.

All rounds reuse the fixed 30,000-row fit, 5,000-row selection, and separate
5,000-row development-evaluation indices.  Rank-growth rounds permit at most
10,000 new real parameters per action.  A rejection rolls back and continues;
four rounds, a compute cap, or catalogue exhaustion terminates the campaign.
The rank-growth bridge seals `rank_activation_scale` in its content-addressed
plan and passes it explicitly to the worker; the default is 1.0 and the
scientific-rejection fallback is exactly 0.25.

## Host-health gate

Because process-level crashes can otherwise be mistaken for model failures, a
scientific service must consume a recent, content-addressed host-health
certificate.  The strict certificate requires a quiet kernel-log window, no
OOM/MCE/EDAC/NVRM-Xid event, repeated fresh-process imports and checkpoint
loads, independent short GPU update probes, finite outputs, a positive metric,
and a bounded memory peak.  A failed or stale certificate permits diagnostic
work only.  Diagnostic artifacts are labelled and cannot satisfy a promotion
gate.

## Termination and claims

The controller stops on catalogue exhaustion, the registered round or compute
limit, repeated technical failure, or a hard safety violation.  Zero promotions
is a valid `complete-no-promotion` outcome.  Reports distinguish:

- operational recovery;
- exploratory development evidence;
- promoted development champion;
- complex128 replay;
- separately authorized blind confirmation.

Only the last category can support a final held-out accuracy claim.
