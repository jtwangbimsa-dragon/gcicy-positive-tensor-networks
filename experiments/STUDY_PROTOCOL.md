# X21 post-v1 TN study protocol

Status: registered development workflow. This protocol produces new post-v1
development evidence only. It is not an amendment to arXiv v1, Zenodo v1.0.0,
or any frozen result table.

## Question and estimands

The study asks how finite increases in site count (k), bond dimension (D),
initialization path, trainable parameter scope, and Adam learning-rate path
affect X21 positive tensor-network accuracy under fixed data and evaluation.

The primary endpoint is weighted sigma on the immutable development-
confirmation pool. Secondary endpoints are chi, bilateral absolute-log-ratio
Q999 and CVaR1%, minimum sampled metric eigenvalue, and the count of
nonpositive sampled metrics. A completed numerical result must have strictly
positive sampled metric eigenvalues; this gate is validity, not an accuracy
objective.

All comparisons are paired by replicate. For each contrast, report the three
individual paired sigma differences, their mean, standard deviation, and
range. Report chi and tail differences in the same long-form table. With only
three optimizer replicates, do not use asymptotic p-values or promote a sign
based only on the mean. Do not fit or claim an infinite-k or infinite-D scaling
law from this finite grid.

## Capacity surface

The direct surface is the full Cartesian product

- k in {8, 12, 16, 20};
- D in {8, 10, 12};
- optimizer replicate in {1, 2, 3}.

Every cell starts from the same frozen q=22, (k=4,D=5) learned function after
canonical completion to q=121 and a fixed positive floor of 1e-14. Site growth
uses exact learned-norm repetition. Bond growth uses zero embedding with
one-sided activation. The potential/metric equivalence audit must pass before
epoch zero. Replicate seeds, point pools, minibatch order, and audit points are
paired across all cells.

The standard optimizer path is an Adam stage at learning rate 2e-4 followed by
3e-5. Both use batch size 1024, gradient clipping at 2, the registered
teacher-free X21 objective, evaluation every epoch, and validation-plateau
stopping. A 120-epoch round cap is only a restart boundary; it is not an
accepted endpoint.

## Targeted matched contrasts

All targeted contrasts use the (k=16,D=12) endpoint and three paired
replicates unless the source architecture is explicitly shown.

| Question | Treatment | Matched comparator |
| --- | --- | --- |
| D-progressive initialization | trained (16,8) -> function-preserving D=12 -> standard path, retaining saved kappa | trained (16,12) -> a second standard path, retaining saved kappa |
| k-progressive initialization | trained (8,12) -> function-preserving k=16 -> standard path, re-estimating kappa | trained (16,12) -> a second standard path, re-estimating kappa |
| Joint parameter scope from epoch zero | cores + q=121 dictionary, standard path | direct cores-only (16,12), standard path |
| Delayed dictionary unfreeze | trained cores endpoint -> cores + dictionary at 1.5e-5 | same trained endpoint -> cores-only at 1.5e-5 |
| Gentle optimizer path | 7.5e-5 -> 1.5e-5 from the direct epoch-zero model | 2e-4 -> 3e-5 from the same epoch-zero model |

The two endpoint-continuation comparators are essential: each gives its
progressive arm the same number of post-baseline training stages and matches
its kappa handling, so improvement caused merely by extra optimization or by
normalization re-estimation is not attributed to k or D transfer. The
cores-only low-learning-rate continuation plays the analogous role for
delayed dictionary unfreezing.

Bond expansion preserves the saved fixed-kappa normalization, so the
D-progressive arm and its endpoint control begin with `saved_model` kappa.
Site repetition changes degree and intentionally clears that normalization,
so the k-progressive arm and its separate endpoint control re-estimate kappa
from their initial target-degree models. Later stages always continue the
saved normalization.

“All parameters” in this study means all coefficient cores plus the shared
q=121 physical dictionary. The reference Hermitian metric and positive floor
remain fixed. Dictionary QR retraction changes parameter coordinates without
transporting Adam moments; therefore the joint-training contrast includes
that labeled optimization-geometry effect and is not interpreted as a pure
parameter-count effect.

## Data-use boundary

Training, selection, and development-confirmation pools are the frozen X21
common pools identified by hashes in the manifest. Historical blind pools are
never loaded. Development results may rank mechanisms and paths but cannot
support a new final claim.

If a candidate is later promoted, freeze a create-only candidate manifest
containing its model and protocol hashes. Generate a new single-use blind pool
only after that freeze, and evaluate it in a separate run root. Do not modify
or append to this development campaign to perform promotion.

## Execution and failure rules

The registered DAG contains 279 jobs: 252 capacity jobs, 24 targeted-arm jobs,
two preparation jobs, and one largest-cell resource preflight. It publishes 60
result rows (36 direct cells and 24 targeted rows).

Before long runs, the (k=20,D=12) one-epoch preflight must complete a forward
pass, backward pass, Adam update, validation, checkpoint, and CUDA memory
measurement on the full registered pools. OOM is a terminal resource result;
the runner never changes D, k, precision, batch size, or loss weights.

SIGSEGV, SIGILL, and SIGTRAP may retry only the exact command from the last
hash-validated evaluation checkpoint. Model, Adam state, RNG streams, history,
fixed kappa, stopping state, accumulated runtime, and CUDA peak memory are
restored. Nonfinite loss, failed equivalence, failed positivity, hash drift,
source drift, or JSON-gate failure is terminal under the current plan. A
changed protocol requires a new manifest and run root.

The following phases can be run independently after their dependencies:

1. `resource-preflight`;
2. `capacity-grid`;
3. `progressive-paths`;
4. `parameter-scope`;
5. `optimizer-paths`.

Aggregation is create-only and includes only content-verified succeeded jobs.
The complete aggregate is blocked until all 60 registered result rows exist;
partial snapshots must be explicitly labeled incomplete.
