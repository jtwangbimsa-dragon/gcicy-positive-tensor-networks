# X21 fresh development pool for Auto Research v2

This workflow creates a new search-only development pool. It does not modify
the historical X21 common pools, arXiv v1, Zenodo v1.0.0, or any frozen result.
The old 49,152-point SHA-256
`4c4ec82e315351c9a5bebf0b3ba48b611bb8cb4ae433ce8a646e5c00426167cc`
is a historical confirmation artifact and is rejected unconditionally.

The sampling identity is fixed:

- adapter `p5p1_type21_k3_1223`;
- exact model seed `20260802`;
- split `selection`;
- 49,152 points from sampling seed `86206`;
- complete sampling clusters of size 6.

Worker count and process/thread backend are operational choices, but they are
locked before sampling and recorded with the deterministic shard seeds. The
managed output root must be disjoint from both the clean `exp/*` source clone
and the supplied frozen-results root.

## Prepare, then materialize

Preparation is create-only. It hashes the generator and its source
dependencies, records the clean Git commit, and stores the only permitted
`shell=False` child command.

```bash
python scripts/run_x21_fresh_development_pool.py prepare \
  --output-root /absolute/experiment_runs/x21-fresh-development-v2 \
  --frozen-root /absolute/frozen-release-root \
  --base-protocol experiments/protocols/x21_auto_research_v1.json \
  --repository-root /absolute/clean-exp-branch-clone \
  --workers 6 \
  --backend process

python scripts/run_x21_fresh_development_pool.py materialize \
  --output-root /absolute/experiment_runs/x21-fresh-development-v2

python scripts/run_x21_fresh_development_pool.py status \
  --output-root /absolute/experiment_runs/x21-fresh-development-v2
```

A technical child failure is recorded as retryable and never becomes a
scientific rejection. Partial immutable output requires a new managed root;
the controller never deletes or overwrites a pool.

## Published contract

Successful materialization re-hashes the NPZ and its generator manifest and
publishes, in order:

1. `generation_receipt.json`;
2. `fresh_development_binding.json`;
3. `x21_auto_research_v2.lock.json`;
4. `x21_auto_research_protocol_binding_patch_v2.json`.

The v2 protocol replaces only the historical development-pool hash and adds
the exact top-level `development_pool_binding` contract. Its data-contract
digest covers both the ordinary data fields and that binding. A scientific
bridge must call `validate_fresh_development_binding` with the protocol's
expected binding and pool hashes before opening the pool.

The v1 controller remains available for baseline and recovery normalization,
but it is not allowed to record scientific evidence. Evidence requires the v2
fresh selection binding; blind and confirmation inputs remain absent.
