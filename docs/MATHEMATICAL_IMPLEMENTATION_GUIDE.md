# Mathematical implementation guide

This guide tells a code auditor which mathematical identity each production
layer is intended to implement.  The task is to check the identity, not merely
that the program executes.

## Geometry, charts and the holomorphic volume form

For a local complete intersection with equations `f=(f_1,...,f_c)`, the code
chooses ambient affine coordinates `(u,z)` so that the dependent-coordinate
Jacobian `J_z = partial f / partial z` is nonsingular.  The tangent inclusion
is

    T = (I, -J_z^{-1} J_u),

up to the ordering convention of the selected coordinates.  Ambient
Fubini--Study forms are pulled back as `T^* G T`.  The Poincare-residue density
contains the inverse squared Jacobian minor, and must transform with the same
coordinate Jacobian as `det(g)`.  Audit
`implicit_atlas.py`, the three geometry modules, and the adapter methods
`baseline_metric`, `holomorphic_volume_log_density`, `rechart_point`, and
`section_values_and_jacobian` together.

The generalized gCICY sections are represented by regular local formulae.
Their overlap differences must vanish modulo the earlier defining equations,
and their first derivatives on X must obey the induced equation-change law.
The relevant implementations are the `q_polynomial_data`,
`local_polynomial_data`, and section-evaluation functions in the X11 and X22
geometry modules.  The algebraic identities quoted in the paper should be
checked symbolically or at high precision, independently of the stored paper
numbers.

## Complete-fibre sampling and importance weights

The sampler first draws base coordinates with the product Fubini--Study
measure.  It then draws a Haar-random complementary projective subspace in the
fibre ambient space and solves the restricted polynomial system.  Every root
of an accepted transverse intersection is retained as one cluster.  The
expected cluster sizes are 4, 6 and 3 for X11, X21 and X22.

The Crofton proposal form on X is the wedge of the base Fubini--Study forms
with the required fibre Fubini--Study power.  In intrinsic dimension three its
local density is extracted as the appropriate mixed coefficient of

    det(t_x G_x + t_y G_y [+ t_z G_z]).

The importance weight is proportional to

    w(x) = rho_Omega(x) / rho_proposal(x).

Only its normalized value is used.  Audit the root-completeness tests,
proposal-density coefficient, residue density, logarithmic weight, cluster
identifier, and coordinate covariance in all three geometry modules.  X11
and X22 reject a complete fibre when two normalized projective roots have
chordal separation at most `1e-7`; common-pool generation passes this value
through the adapter and stores it under `sampling.sampling_options`, so a
regenerated sample records the numerical gate that produced it.

## Section bases and unrestricted Hermitian metrics

For a restricted section vector `s_k(x)` and positive Hermitian matrix `H`,

    F_H(x) = s_k(x)^dagger H s_k(x),
    K_H(x) = (1/k) log F_H(x),
    g_H = partial dbar K_H.

`global_sections.py` and the adapter batch methods evaluate section values and
first derivatives and use the standard log-quadratic-form Hessian.  The
restriction basis must have the claimed rank, and basis changes must leave the
metric invariant.  The generic full-H optimizer used for X21 is implemented
in `pipeline/train.py`; the X11 H6 optimizer is the separate production script
`train_type11_k6_reference_whitened_full_h.py`.

The X11 degree-two start uses the balanced T-step in
`train_type11_nu_balanced.py`.  An auditor should check the importance-weighted
T-map, trace normalization, whitening/unwhitening, positivity floor, and the
convergence criterion.

## Positive tensor-network ansatz

Let `V=H^0(X,L^b)`, `d=dim(V)`, and let the product feature be formed from `m`
copies of the degree-b section vector.  At each site the physical operator
index has size `d^2`; an optional dictionary of size `q` maps this operator
index to the coefficient-core index.  Contracting the coefficient cores over
the virtual bonds defines a factor `B_theta`.  The production ansatz is

    F_theta(x) = ||B_theta v(x)||^2 + epsilon_ref F_0(x)^m,
    K_theta(x) = (1/(m b)) log F_theta(x).

The squared norm makes the trainable contribution positive semidefinite and
the positive reference term makes the total form strictly positive.  Audit:

1. core shapes, physical-index ordering and parameter counts;
2. shared-dictionary contraction;
3. value, first-derivative and mixed second-derivative contractions;
4. equivalence under bond/site/dictionary embeddings;
5. positivity and the `1/(mb)` normalization;
6. batch and scalar implementations against finite differences and dense
   materialization where feasible.

The reusable implementation is `pipeline/positive_tensor_network.py`; the
model actually trained for all three paper geometries is driven by
`scripts/train_type11_positive_tensor_network.py`.

## Exact degree continuation

If `F_b` defines the starting metric and `k=mb`, then

    F_k = F_b^m,
    (1/k) log F_k = (1/b) log F_b.

The potential and metric must therefore agree to numerical precision before
new channels are trained.  Audit `pipeline/algebraic_power_lift.py`, adapter
`lift_h_matrix_power` methods, and the site-repeat/resize/dictionary conversion
scripts.  Function-preserving bond growth must preserve the old contraction
exactly while making the intended new channels trainable.

## Native Monge--Ampere objective

At sampled points define

    ell_i = log det(g_i) - log rho_Omega,i,
    r_i = exp(ell_i) / sum_j w_j exp(ell_j),

with the matching convention for normalized weights.  The native quadratic
term is the weighted mean of `(r_i-1)^2`; the centered-log term and upper-tail
terms use the same fixed sample weights.  The reported production runs first
compute the normalization on the complete training sample and then hold that
scalar fixed during parameter updates.  Audit the log-sum-exp stabilization,
this frozen-normalization convention, minibatch weight normalization,
positivity gates and the distinction between training normalization and
fresh-sample evaluation.  The production scripts renormalize importance
weights inside each minibatch; this is a self-normalized stochastic
approximation to the complete-sample weighted objective rather than its
unbiased importance-sampling estimator.

## Residual-potential arm

The reported X11 residual model uses

    K = K_H2 + phi_theta.

It is implemented directly in
`scripts/run_cymetric_phi_gcicy_type11.py`.  In particular, the conversion of
the real-coordinate Hessian of `phi` to the ambient complex
`partial dbar phi`, its pullback through the tangent map, and the Hermitian
index orientation must be audited.  The generic helper
`pipeline/projective_residual_phi.py` did not generate the reported X11
checkpoints and must not be substituted for this production implementation.

## X11 full-H6 arm

The multiplication map sends cubic products of degree-two sections to the
degree-six restricted section space.  The code constructs the exact power
lift, whitens the degree-six basis, and parameterizes the relative positive
matrix by a compact complex Cholesky factor.  It evaluates
`log(s^dagger H s)` and its metric derivatives from cached section jets.  Audit
the multiplication orientation, pullback/pushforward convention, whitening,
compact factor unpacking, positivity, and TN-to-H6 materialization.  The
production implementation is
`scripts/train_type11_k6_reference_whitened_full_h.py`, not
`pipeline/factorized_h.py`.

## Final statistics

All reported final comparisons use frozen models on common points.  Weights
are renormalized within each evaluated array.  Bulk, quantile and CVaR
statistics are paired, and bootstrap resampling is by complete fibre cluster,
not by individual point.  Audit `pipeline/risk.py`, `pipeline/tail.py`,
`bootstrap_gcicy_metric_comparison.py`, and each summarizer for consistent
definitions, tails, signs and denominators.
