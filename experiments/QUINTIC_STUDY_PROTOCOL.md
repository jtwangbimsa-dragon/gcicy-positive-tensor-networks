# Quintic k/D, progressive-initialization, joint-training, and path protocol

Status: post-v1 development protocol; it does not alter or extend any arXiv v1,
Zenodo v1.0.0, or frozen-result claim.

Manifest: `manifests/quintic_kd_progressive_joint_paths_v1.json`.

## Question and boundary

This campaign asks whether the historical Fermat-quintic positive-TN endpoint
at `k=20`, complete `q=25`, and `D=6` benefits from increasing degree and bond
capacity, and whether any improvement depends on the order of exact transfer,
the trainable parameter scope, staged dictionary unfreezing, or a gentler Adam
learning rate.  It is a development campaign.  Its results must not be folded
back into the frozen v1 tables or described as a new blind result.

The earlier `k=40,D=7` result is retained only as a one-replicate workflow
calibration.  It is not a primary endpoint because that cell was already used
during historical development.  The new primary target is `k=40,D=14`, using
the registered capacity ladder `D=6 -> 10 -> 14` and the exact multiplicative
site repeat `k=20 -> 40`.

All model-development fits use the old 90,000 training rows and 10,000
selection rows, together with their frozen pullbacks.  Every trainer invocation
must receive `--skip-blind-audit`.  In particular, it must not load, hash,
summarize, compare against, or copy any of the following historical 200,000
point material:

- `blind_points.npz`;
- `blind_test_tail_arrays.npz`;
- `blind_pullbacks.npy`;
- any historical blind metric embedded in a source report.

The source report and anchor report are neither frozen-manifest inputs nor
development-command inputs.  Architecture provenance comes from the
hash-registered anchor model and dataset/basis/pullback registry.  A development
report must omit or null historical blind comparator fields.

## Frozen origin and pairing

The common model origin is the immutable historical checkpoint
`k20,q25,D6`, torch seed `202607192`, SHA-256
`2d78bfcc86e7a2f076ae2dfa703e015621fc7ae1f4348b7e8415df3c0e0f808f`.
Using one common origin makes the new replicate index an optimizer/transfer RNG
replicate rather than a mixture of different historical checkpoints.

Replicates 1, 2, and 3 use paired torch seeds across every A--J arm.  Within a replicate,
all arms use the same training and selection arrays, the same minibatch seed,
the same transfer seed, the same fixed log kappa, and the same three budget
blocks.  Seeds must never be redrawn after seeing an arm result.

The fixed normalization is
`log(kappa) = -4.0396350923022535`, estimated historically from the FS metric on
the complete 90,000-row training pool.  It is supplied as a command-line value
in every run; no arm is allowed to re-estimate it from its current checkpoint.

## Equal-update budget

One budget block is exactly 50 complete epochs over 90,000 rows at batch size
1024.  The trainer uses the final partial batch, so a block has
`ceil(90000/1024) * 50 = 4,400` Adam updates and 4,500,000 point exposures.
Every A--J endpoint has exactly three blocks:

- 150 epochs;
- 13,200 Adam updates;
- 13,500,000 point exposures;
- three optimizer initializations, one at each registered block boundary.

Each stage has one plateau-wrapper round.  J's first block is a genuinely cold
plateau-wrapper invocation with initialization noise `1e-2`; its two remaining
blocks continue through the study-arm wrapper.  Thus validation determines the
saved checkpoint within a fixed block but cannot shorten an arm's registered
training exposure.  Intermediate path jobs are lineage segments, not extra
fits: their cumulative budgets are recorded from the common anchor.

## Main A--I design and cold secondary control

| arm | capacity path and three equal blocks | trainable scope | estimand role |
|---|---|---|---|
| A | remain at `k20,D6`; three blocks | cores | extra-training control |
| B | exact `k20->40` before block 1; remain `D6` | cores | k-only effect, B-A |
| C | `D6->10` before block 1, `D10->14` before block 2, remain `k20` | cores | D-only effect, C-A |
| D | `k20->40`, block 1; `D6->10`, block 2; `D10->14`, block 3 | cores | k-then-D progressive path |
| E | `D6->10`, block 1; `D10->14`, block 2; `k20->40`, block 3 | cores | D-then-k progressive path |
| F | exact site repeat and both D expansions before block 1 | cores | endpoint-transfer control |
| G | same epoch-zero target as F | cores plus dictionary in all blocks | immediate-joint control |
| H | same epoch-zero target as F | cores, cores, then joint | staged dictionary unfreeze |
| I | same epoch-zero target as G | joint in all blocks | gentle optimizer path |
| J | independently initialized `k40,D14,q25` FS/random target | cores | cold-target secondary control |

Arms G and I differ only in the learning rate recorded inside otherwise
identical stage specifications.  Arm H isolates staged dictionary unfreezing;
it is not described as an LR comparison.  F versus G isolates fixed-dictionary
cores from immediate all-parameter joint training at the same initialization
and update budget.  D versus E measures order dependence; D/F and E/F measure
progressive timing versus endpoint transfer.  F versus J, not D/E versus F,
measures transferred versus cold target initialization.  J is a secondary
initialization control and does not change the preregistered D14 A--I
estimands.

The primary summaries are paired replicate differences in final selection
sigma/score and registered tail statistics.  Report each replicate and the
mean paired difference; three replicates are not enough to justify asymptotic
significance claims.  Positivity failures, nonfinite values, failed transfer
equivalence, or a missing fixed-budget block remain results and are not silently
rerun under a different configuration.

## Transfer and parameter rules

The `k=20 -> 40` transfer is an exact repeated-site representation of the
learned norm.  Bond expansions use function-preserving one-sided activation,
and every transfer step must pass the quintic potential/metric equivalence
audit before training continues.  The physical `q=25` dictionary and positive
floor are fixed in arms A--F.  `cores` means every coefficient core is trained
while the physical dictionary is frozen.  `joint` means every coefficient core
and the physical dictionary are trained; the reference form and positive floor
remain fixed.

Joint training applies the registered dictionary QR/retraction after optimizer
steps.  Adam moments are not transformed into the new QR coordinates.
Consequently F/G/H/I estimate the complete joint-plus-retraction optimizer
path; they do not isolate a coordinate-independent effect of merely adding
dictionary degrees of freedom.

Only Adam is in scope.  LM, ALS, rank-growth selection, different losses,
different batch sizes, or a changed kappa would define a new campaign.

## Precision and resource preflights

Training remains `complex64` to preserve the historical quintic model lineage
and to make the high-D cells feasible on the reference 24 GiB GPU.  Therefore
these fits are labelled `complex64-trained development models`, not equivalent
to complex128 optimization.

Before the D14 main phase, one direct `k40,D14` cores run and one direct joint
run execute one complete epoch at batch 1024.  That epoch must cover forward,
backward, Adam, dictionary QR when joint, validation, and atomic checkpoint
publication.  Reissuing a completed preflight must take the idempotent reuse
path and return the hash-identical publication without performing another
epoch.  True interruption recovery is certified by trainer fault-injection and
resume tests, not by increasing the registered preflight epoch budget.  Peak
allocated memory must not exceed 21,474,836,480 bytes (20 GiB), and peak
reserved memory must not exceed 23,085,449,216 bytes (21.5 GiB).  Successful
cores and joint D14 preflights gate all A--J work.

`k40,D16` is a conditional extension only.  Its cores and joint preflights use
the same one-epoch plus hash-identical reuse rule.  Passing both
memory gates permits
the creation of a separate optional D16 phase/manifest; it does not add D16 to
the D14 estimand, and this manifest contains no D16 accuracy fit.

Every one of the 30 A--J development endpoints is promoted without training to
a hash-bound complex128 twin and replayed on the same 10,000 selection rows.
The replay gates a positive minimum metric eigenvalue, zero nonpositive metrics,
and a bounded c64/c128 raw-log-ratio delta while reporting all registered
bulk/tail values.  The manifest does not infer or gate rank invariance.  The
60-row aggregate must explicitly report every c64/c128 ranking change, and an
independent selection review must freeze that interpretation and the candidate
before promotion.  Replay validates the forward result, not equivalence to a
complex128 optimizer trajectory.

## Result and promotion policy

The workflow contains 113 jobs: 53 calibration/preflight/training jobs and 60
precision-cast/replay jobs.  It exports 30 c64 endpoint rows and 30 c128 replay
rows.  The workflow records final model/report hashes, target `k,D`, ordered stage
names and scopes, selection score, evaluation scope, and CUDA peak memory.
The primary table contains the 27 A--I endpoints (nine arms times three paired
replicates).  The three J rows form a labelled secondary initialization table,
for 30 endpoint rows in the workflow result export.  Calibration, intermediate
lineage segments, cold-initialization jobs, and resource preflights are
excluded.

No new blind pool is created by this manifest.  After the development table,
precision replay, candidate choice, and stopping rule are frozen, a different
create-only promotion manifest must bind:

1. the exact selected c64 model and complex128 twin hashes;
2. the complete development-manifest digest and selection rule;
3. a previously unused blind seed and point count;
4. a blind-pool output path disjoint from every historical pool;
5. a rule forbidding model updates after pool creation.

Blind evaluation requires a later, separately reviewed audit manifest.  The
create-only promotion step may generate and hash the new pool, but it may not
evaluate a candidate on it.  The historical 200,000-point pool is never
eligible for either step.
