# Post-v1 experiment baseline

This document fixes the starting point for new (k)-, (D)-, initialization-,
parameter-scope-, and optimizer-path experiments.  It is an index of existing
evidence, not a revision of the paper or deposition.

## Immutable release boundary

The new code branch starts at Git commit
`9f95071647aaf89bd520effeb322b16345d63157` (`v1.0.0` in
`pyproject.toml`).  The following local release identities are read-only:

| Object below `$GCICY_FROZEN_INPUT_ROOT` | SHA-256 |
| --- | --- |
| `gcicy_tn_arxiv_source_20260817.zip` | `bbd16a674d93ccd6da91633312610f16cdfaf2414e4a29bcbb58b2f82fc9c50f` |
| `gcicy_tn_paper_latest_20260817.pdf` | `3abdb11eac8cc10b41883b396cf6eeb163e23ccbb84e86b021764ce525c603b0` |
| `outputs/release/gcicy_tn_zenodo_artifacts_v1_0_0.zip` | `587be3188b8d7a68837c3e073b062174a56244a57d7fbf3e8185a82059b608` |
| `outputs/release/gcicy_tn_full_production_code_audit_v1_0_0.zip` | `0416632a36a0608f560d17661d23e52ad31c4560c3e39968789955bc25142af6` |

The versioned data DOI is `10.5281/zenodo.21963526`.  Existing
`arxiv_submission_gcicy_tn_20260817/`, `artifact_release*`,
`gcicy paper/generated_tn/`, `outputs/pipeline/`, and `outputs/release/` paths
must never be workflow output roots.  The original source branch passed all 58
deterministic core tests before any experiment-branch changes.

There are no post-2026-08-17 scientific result files in the inspected
workspace.  Consequently, all numbers below are frozen v1 evidence and the new
campaign begins with no post-v1 result.

## Frozen final accuracy baselines

The primary accuracy statistics are the weighted sigma convention, the square
root of normalized squared Monge--Ampere energy (`chi`), bilateral
|log-volume-ratio| Q999/CVaR, and sampled minimum metric eigenvalue.  A
topological volume or slope check is not a substitute for these local errors.

| Geometry/model | Replicates / common sample | sigma | chi | Status |
| --- | ---: | ---: | ---: | --- |
| X11 TN, (k=6,D=5,P=141750) | 3 / 200000 | `0.014802 ± 0.001378` | `0.022597 ± 0.001614` | frozen final |
| X21 full (H_4), (P=104976) | 1 finite-budget row | `0.016273` | `0.023968` | frozen final comparator |
| X21 TN, (k=8,D=8,P=96800) | 1 finite-budget row | `0.010503` | `0.015236` | frozen final comparator |

For X11, the three-run mean |log ratio| Q999 is `0.204372 ± 0.016496`
and CVaR1% is `0.126628 ± 0.005525`.  The source is
`outputs/pipeline/type11_x11_equal_time_final_20260807/final_summary.json`
(SHA-256
`f6b521993693529289999db0f07a0dc4d658e6fc20a74d7384f6057c725f1a77`).

## Existing finite-range (k,D) evidence

The three-replicate X21 fixed-(D=8) final ladder is non-monotone after
(k=12):

| (k) | Real trainable parameters | mean sigma | mean chi | mean optimizer wall time | max allocated CUDA memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 96800 | `0.008767` | `0.012791` | 2600 s | 2.89 GB |
| 12 | 158752 | `0.005696` | `0.008715` | 5253 s | 4.60 GB |
| 16 | 220704 | `0.006357` | `0.009318` | 5928 s | 6.32 GB |
| 20 | 282656 | `0.006073` | `0.009268` | 6005 s | 8.03 GB |

This is finite-range empirical evidence, not a (k\to\infty) scaling law.
The registered (k=24) precision process suffered two segmentation faults and
one illegal-instruction failure; it produced no accepted checkpoint and is not
a numerical result.  Source:
`outputs/pipeline/type21_fixed_d8_degree_scaling_floor1e14_20260807/final_summary_k8_k20.json`
(SHA-256
`e1719543f26c76c6fe82955acc1c6ce7325133982c325299384d39aa30df26e1`).

The older single-seed X21 development surface demonstrates both capacity and
path dependence:

| Initialization | ((k,D)) | sigma | chi |
| --- | ---: | ---: | ---: |
| common origin | (8,8) | `0.006656` | `0.010161` |
| common origin | (12,8) | `0.005929` | `0.009125` |
| common origin | (16,8) | `0.005778` | `0.008713` |
| common origin | (8,10) | `0.004876` | `0.007717` |
| common origin | (16,10) | `0.004239` | `0.006692` |
| common origin | (8,12) | `0.005373` | `0.008434` |
| common origin | (16,12) | `0.005296` | `0.008103` |
| ((16,10)\to(16,12)) | (16,12) | `0.004075` | `0.006446` |
| ((8,10)\to(16,10)) | (16,10) | `0.004793` | `0.007550` |

Thus continuation can help or hurt: the (D)-continuation improves the
((16,12)) endpoint, whereas the shown (k)-continuation loses to the direct
((16,10)) run.  These rows use an already inspected development pool and are
not fresh blind evidence.  Source:
`outputs/pipeline/type21_kd_plateau_paper_summary_20260802.json` (SHA-256
`48056100094bd88e9d50726a5f81b4520508ce8dcdd0d0de79792dabc2000c36`).

The preregistered equal-update X21 (D=8\to12) control gave only a `1.92%`
mean sigma and `1.10%` mean chi advantage over continuing at (D=8); one of
three pairs reversed and the directional gate failed.  Increasing (D) is
therefore not by itself a reliable explanation of accuracy.

## Existing optimizer-path evidence

At the identical X11 endpoint (k=6,D=13,P=789750), changing only the
capacity/LR path produced:

| Path | development sigma | development chi |
| --- | ---: | ---: |
| direct standard | `0.013644` | `0.022693` |
| direct gentle | `0.009236` | `0.018050` |
| (D:5\to8\to10\to13) | `0.011605` | `0.021932` |
| (D:5\to7\to9\to11\to13) | `0.011160` | `0.019917` |

The direct gentle path improved sigma by about 32% relative to the direct
standard path without changing the final architecture.  Source:
`outputs/pipeline/type11_same_start_tn_path_search_20260805/path_selection.json`
(SHA-256
`7e2b010d53a10d241ae29a8d548428d61e8f6540c0e0a3f45c549651308c7e0a`).

## Parameter-scope terminology

“All parameters” must be stated precisely:

1. **cores-only joint training**: every coefficient core is optimized
   together.  This is already the X21 (k,D) baseline; the q=121 dictionary,
   reference (H), and positive reference floor are fixed.
2. **cores + dictionary joint training**: all coefficient cores and the shared
   physical dictionary are optimized together, with row orthonormalization
   after each optimizer step.  This is the new matched ablation.
3. **reference/base metric training**: not supported by the current positive-TN
   trainer and not part of the new “full-parameter” claim.  The reference
   branch remains fixed to preserve the common positive ansatz and comparison.

Dictionary orthonormalization changes its parameter coordinates without
transporting Adam moments.  Joint-versus-fixed dictionary results therefore
also test this optimization geometry and must remain a labeled path contrast.

## Baseline gaps addressed by the new branch

- no factorial three-seed (k\times D) surface on one immutable set of pools;
- no matched direct versus progressive-(k) versus progressive-(D) endpoint
  comparison;
- no matched cores-only versus cores+dictionary experiment on X21;
- no durable epoch-boundary recovery of model, Adam, RNG, history, and plateau
  state;
- no campaign-level schema, frozen-input hash guard, GPU lock, or verified
  output state machine.

The post-v1 workflow addresses these gaps while keeping all historical blind
sets sealed.  Any future final claim requires a separately frozen candidate
manifest followed by a newly generated single-use blind pool.
