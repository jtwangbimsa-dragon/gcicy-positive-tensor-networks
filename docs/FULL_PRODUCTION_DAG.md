# Public source map

This document identifies the portable Python programs that implement the
principal numerical pipeline. Machine-specific launchers, registered
protocols, frozen artifacts, and claim-reconstruction records are distributed
in the versioned Zenodo archive rather than this source-only repository.

## Geometry and source sections

- Product-projective coordinates, monomials, and Fubini--Study data:
  `gcicy_metric/product_projective.py`.
- Implicit charts, tangent maps, Jacobian minors, and chart changes:
  `gcicy_metric/implicit_atlas.py`.
- Restricted section bases, section jets, and algebraic metrics:
  `gcicy_metric/global_sections.py`.
- Geometry-specific equations, generalized sections, residue densities,
  complete-fibre samplers, and proposal densities:
  `gcicy_metric/type21_hirzebruch_x3.py` (X11),
  `gcicy_metric/type21_candidate_p5p1_1223.py` (X21), and
  `gcicy_metric/generic_model.py` (X22).
- Common adapters:
  `gcicy_metric/pipeline/adapter.py`,
  `gcicy_metric/pipeline/registry.py`, and
  `gcicy_metric/pipeline/adapters/`.

The corresponding exact and numerical checks are driven by
`scripts/certify_topological_targets.py`,
`scripts/certify_section_restriction_ranks.py`,
`scripts/certify_gcicy_source_map_immersion.py`,
`scripts/check_global_sections.py`,
`scripts/check_implicit_atlas.py`, and
`scripts/audit_sampler_fibres.py`.

## Sampling and metric models

- Common-sample generation: `scripts/generate_gcicy_common_point_pool.py`,
  backed by `gcicy_metric/pipeline/common_point_pool.py` and
  `gcicy_metric/pipeline/parallel_sampling.py`.
- Positive tensor network:
  `gcicy_metric/pipeline/positive_tensor_network.py`, trained through
  `scripts/train_type11_positive_tensor_network.py`.
- Exact metric-preserving degree lift:
  `gcicy_metric/pipeline/algebraic_power_lift.py`.
- Low-degree unrestricted Hermitian metrics:
  `gcicy_metric/pipeline/train.py` and
  `scripts/train_gcicy_pipeline.py`.
- X11 degree-two starting metric:
  `scripts/train_type11_nu_balanced.py`.
- X11 residual-potential comparison:
  `scripts/run_cymetric_phi_gcicy_type11.py`.
- X11 degree-six unrestricted metric:
  `scripts/build_type11_h2_cubic_full_h_lift.py`,
  `scripts/train_type11_k6_reference_whitened_full_h.py`, and
  `scripts/materialize_type11_k6_tn_as_full_h.py`.

## Objectives, evaluation, and uncertainty

- Native Monge--Ampere objectives and tail terms:
  `gcicy_metric/pipeline/risk.py`,
  `gcicy_metric/pipeline/tail.py`, and
  `scripts/train_type11_positive_tensor_network.py`.
- Common-sample evaluation:
  `scripts/evaluate_h_metric_common_pool.py` and
  `scripts/audit_type11_positive_tensor_network.py`.
- Paired fibre-cluster bootstrap:
  `scripts/bootstrap_gcicy_metric_comparison.py` and the geometry-specific
  summary programs in `scripts/`.
- Multiplication-map and Schmidt diagnostics:
  `scripts/audit_gcicy_section_multiplication.py`,
  `scripts/audit_gcicy_full_h_lift_schmidt_rank.py`, and
  `scripts/audit_gcicy_full_h_lift_all_cut_schmidt.py`.

The detailed mathematical identities implemented by these layers are given in
`docs/MATHEMATICAL_IMPLEMENTATION_GUIDE.md`. The Zenodo archive records the
exact schedules and artifact hashes used for the paper results.
