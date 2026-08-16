"""Global patch data for the explicit P1 x P1 x P5 gCICY example.

The generalized configuration is

    [ P1 | 1 1 -1  1 ]
    [ P1 | 1 1  1 -1 ]
    [ P5 | 3 1  1  1 ]

The first two columns define an intermediate complete intersection M.  The
last two columns are global sections on M, represented by rational functions
in ambient homogeneous coordinates.  Following arXiv:1507.03235, Eq. (1.4),

    q1 = d1/x0 = -d0/x1,
    q2 = c1/y0 = -c0/y1,

where p2 = x0*d0 + x1*d1 = y0*c0 + y1*c1.  Each projective affine chart uses
the expression whose denominator is the selected nonzero homogeneous
coordinate, so all local defining equations are polynomial.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .atlas import Chart, affine_chart_coordinates, homogeneous_coordinates, selected_coordinate_abs
from .simple_patch import Array, p1_rest, random_parameters


P1_DEGREE = (1, 1, 3)
P2_DEGREE = (1, 1, 1)
Q1_DEGREE = (-1, 1, 1)
Q2_DEGREE = (1, -1, 1)


@dataclass(frozen=True)
class GlobalGeometryDiagnostic:
    """Numerical consistency summary for the global patch description."""

    intermediate_points: int
    gcicy_points: int
    charts_tested: int
    charts_with_usable_points: int
    max_intermediate_residual: float
    max_rational_representation_error: float
    max_section_transition_error: float
    max_gcicy_relative_equation_residual: float
    min_gcicy_jacobian_singular_value: float
    min_boundary_jacobian_singular_value: float
    prototype_is_globally_smooth: bool


def _validate_homogeneous(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
    x_value = np.asarray(x, dtype=np.complex128).reshape(-1)
    y_value = np.asarray(y, dtype=np.complex128).reshape(-1)
    z_value = np.asarray(z, dtype=np.complex128).reshape(-1)
    if x_value.shape != (2,) or y_value.shape != (2,) or z_value.shape != (6,):
        raise ValueError(
            "Expected homogeneous coordinate shapes (2,), (2,), and (6,), "
            f"got {x_value.shape}, {y_value.shape}, and {z_value.shape}."
        )
    return x_value, y_value, z_value


def p2_coefficients(x: Array, y: Array, z: Array) -> tuple[complex, complex, complex, complex]:
    """Return (c0, c1, d0, d1) in the two decompositions of p2."""

    x_value, y_value, z_value = _validate_homogeneous(x, y, z)
    x0, x1 = x_value
    y0, y1 = y_value
    _, z1, z2, z3, z4, _ = z_value
    c0 = x0 * z1 + x1 * z2
    c1 = x0 * z3 + x1 * z4
    d0 = y0 * z1 + y1 * z3
    d1 = y0 * z2 + y1 * z4
    return c0, c1, d0, d1


def p2_homogeneous(x: Array, y: Array, z: Array) -> complex:
    """Evaluate the homogeneous (1,1,1) intermediate equation p2."""

    x_value, y_value, z_value = _validate_homogeneous(x, y, z)
    c0, c1, _, _ = p2_coefficients(x_value, y_value, z_value)
    return y_value[0] * c0 + y_value[1] * c1


def p1_homogeneous(x: Array, y: Array, z: Array) -> complex:
    """Evaluate the homogeneous (1,1,3) equation used by the local model."""

    x_value, y_value, z_value = _validate_homogeneous(x, y, z)
    x0, x1 = x_value
    y0, y1 = y_value
    z0, z1, z2, z3, z4, z5 = z_value
    value = x0 * y0 * z0**2 * z5
    value += 0.03 * x0 * y0 * z0**3
    value += 0.20 * x0 * y0 * (z1**3 + 0.70 * z2**3 - 0.40 * z3**3 + 0.50 * z4**3)
    value += 0.11 * x1 * y0 * z0 * z2 * z3
    value -= 0.09 * x0 * y1 * z0 * z1 * z4
    value += 0.07 * x1 * y1 * z0 * z2 * z4
    value += 0.05 * (x1 * y0 + x0 * y1) * z0 * z1 * z2
    return value


def normalize_homogeneous(x: Array, y: Array, z: Array, chart: Chart) -> tuple[Array, Array, Array]:
    """Normalize homogeneous representatives so the chart coordinates equal one."""

    x_value, y_value, z_value = _validate_homogeneous(x, y, z)
    selected = (x_value[chart[0]], y_value[chart[1]], z_value[chart[2]])
    if min(abs(value) for value in selected) < 1e-14:
        raise FloatingPointError("The requested projective chart is not defined at this point.")
    return x_value / selected[0], y_value / selected[1], z_value / selected[2]


def homogeneous_from_affine(chart_coords: Array, chart: Chart) -> tuple[Array, Array, Array]:
    """Reconstruct normalized homogeneous coordinates from seven affine coordinates."""

    coords = np.asarray(chart_coords, dtype=np.complex128).reshape(-1)
    if coords.shape != (7,):
        raise ValueError(f"Expected seven affine coordinates, got {coords.shape}.")

    x = np.empty(2, dtype=np.complex128)
    y = np.empty(2, dtype=np.complex128)
    z = np.empty(6, dtype=np.complex128)
    x[chart[0]] = 1.0
    y[chart[1]] = 1.0
    z[chart[2]] = 1.0
    x[np.arange(2) != chart[0]] = coords[:1]
    y[np.arange(2) != chart[1]] = coords[1:2]
    z[np.arange(6) != chart[2]] = coords[2:]
    return x, y, z


def q1_rational(x: Array, y: Array, z: Array, denominator_index: int) -> complex:
    """Evaluate q1 using d1/x0 or -d0/x1."""

    x_value, _, _ = _validate_homogeneous(x, y, z)
    _, _, d0, d1 = p2_coefficients(x, y, z)
    if denominator_index == 0:
        if abs(x_value[0]) < 1e-14:
            raise FloatingPointError("x0 is too small for the d1/x0 representation.")
        return d1 / x_value[0]
    if denominator_index == 1:
        if abs(x_value[1]) < 1e-14:
            raise FloatingPointError("x1 is too small for the -d0/x1 representation.")
        return -d0 / x_value[1]
    raise ValueError("denominator_index must be 0 or 1.")


def q2_rational(x: Array, y: Array, z: Array, denominator_index: int) -> complex:
    """Evaluate q2 using c1/y0 or -c0/y1."""

    _, y_value, _ = _validate_homogeneous(x, y, z)
    c0, c1, _, _ = p2_coefficients(x, y, z)
    if denominator_index == 0:
        if abs(y_value[0]) < 1e-14:
            raise FloatingPointError("y0 is too small for the c1/y0 representation.")
        return c1 / y_value[0]
    if denominator_index == 1:
        if abs(y_value[1]) < 1e-14:
            raise FloatingPointError("y1 is too small for the -c0/y1 representation.")
        return -c0 / y_value[1]
    raise ValueError("denominator_index must be 0 or 1.")


def local_defining_equations_from_homogeneous(x: Array, y: Array, z: Array, chart: Chart) -> Array:
    """Return the four polynomial local equations in a projective affine chart."""

    x_local, y_local, z_local = normalize_homogeneous(x, y, z, chart)
    return np.asarray(
        [
            p1_homogeneous(x_local, y_local, z_local),
            p2_homogeneous(x_local, y_local, z_local),
            q1_rational(x_local, y_local, z_local, chart[0]),
            q2_rational(x_local, y_local, z_local, chart[1]),
        ],
        dtype=np.complex128,
    )


def local_defining_equations(chart_coords: Array, chart: Chart) -> Array:
    """Return local defining equations from seven affine chart coordinates."""

    return local_defining_equations_from_homogeneous(*homogeneous_from_affine(chart_coords, chart), chart)


def local_equation_scales_from_homogeneous(x: Array, y: Array, z: Array, chart: Chart) -> Array:
    """Return absolute term-sum scales for the four local equations."""

    x_local, y_local, z_local = normalize_homogeneous(x, y, z, chart)
    x0, x1 = x_local
    y0, y1 = y_local
    z0, z1, z2, z3, z4, z5 = z_local
    p1_terms = [
        x0 * y0 * z0**2 * z5,
        0.03 * x0 * y0 * z0**3,
        0.20 * x0 * y0 * z1**3,
        0.14 * x0 * y0 * z2**3,
        -0.08 * x0 * y0 * z3**3,
        0.10 * x0 * y0 * z4**3,
        0.11 * x1 * y0 * z0 * z2 * z3,
        -0.09 * x0 * y1 * z0 * z1 * z4,
        0.07 * x1 * y1 * z0 * z2 * z4,
        0.05 * x1 * y0 * z0 * z1 * z2,
        0.05 * x0 * y1 * z0 * z1 * z2,
    ]
    p2_terms = [x0 * y0 * z1, x1 * y0 * z2, x0 * y1 * z3, x1 * y1 * z4]
    q1_terms = [y0 * z2, y1 * z4] if chart[0] == 0 else [y0 * z1, y1 * z3]
    q2_terms = [x0 * z3, x1 * z4] if chart[1] == 0 else [x0 * z1, x1 * z2]
    return np.asarray(
        [
            max(1.0, sum(abs(term) for term in p1_terms)),
            max(1.0, sum(abs(term) for term in p2_terms)),
            max(1.0, sum(abs(term) for term in q1_terms)),
            max(1.0, sum(abs(term) for term in q2_terms)),
        ],
        dtype=float,
    )


def local_equation_jacobian(chart_coords: Array, chart: Chart, step: float = 1e-6) -> Array:
    """Numerically differentiate the four holomorphic local equations."""

    coords = np.asarray(chart_coords, dtype=np.complex128).reshape(-1)
    jacobian = np.empty((4, 7), dtype=np.complex128)
    for column in range(7):
        direction = np.zeros(7, dtype=np.complex128)
        direction[column] = step
        jacobian[:, column] = (
            local_defining_equations(coords + direction, chart)
            - local_defining_equations(coords - direction, chart)
        ) / (2.0 * step)
    return jacobian


def section_transition_multiplier(
    x: Array,
    y: Array,
    z: Array,
    from_chart: Chart,
    to_chart: Chart,
    degree: tuple[int, int, int],
) -> complex:
    """Return the local-section multiplier from one projective chart to another."""

    x_value, y_value, z_value = _validate_homogeneous(x, y, z)
    ratios = (
        x_value[from_chart[0]] / x_value[to_chart[0]],
        y_value[from_chart[1]] / y_value[to_chart[1]],
        z_value[from_chart[2]] / z_value[to_chart[2]],
    )
    return complex(np.prod([ratio**power for ratio, power in zip(ratios, degree, strict=True)]))


def sample_intermediate_reference_patch(n_points: int, seed: int = 0, scale: float = 0.6):
    """Sample points on M={p1=p2=0} in the x0=y0=z0=1 patch."""

    rng = np.random.default_rng(seed)
    output: list[tuple[Array, Array, Array]] = []
    for _ in range(n_points):
        a, b, w2, w3, w4 = scale * (rng.normal(size=5) + 1j * rng.normal(size=5))
        w1 = -(a * w2 + b * w3 + a * b * w4)
        w5 = -p1_rest(a, b, w1, w2, w3, w4)
        output.append(
            (
                np.asarray([1.0, a], dtype=np.complex128),
                np.asarray([1.0, b], dtype=np.complex128),
                np.asarray([1.0, w1, w2, w3, w4, w5], dtype=np.complex128),
            )
        )
    return output


def reference_parameters_from_homogeneous(x: Array, y: Array, z: Array) -> Array:
    """Convert a point with x0*y0*z0*z4 nonzero to reference (s,t,r)."""

    x_value, y_value, z_value = _validate_homogeneous(x, y, z)
    if min(abs(x_value[0]), abs(y_value[0]), abs(z_value[0]), abs(z_value[4])) < 1e-14:
        raise FloatingPointError("The point is outside the (s,t,r) reference parameter patch.")
    return np.asarray([z_value[2] / z_value[0], z_value[3] / z_value[0], z_value[4] / z_value[0]])


def random_projective_coverage_points(
    n_points: int,
    seed: int = 0,
    *,
    min_solve_coefficient: float = 1e-8,
) -> tuple[Array, Array, Array, Array]:
    """Sample a broad projective coverage set on the special explicit X.

    This is a coverage sampler, not a claim of uniform sampling in the
    Calabi-Yau or induced Fubini-Study measure.  It samples homogeneous x and
    y, together with a projective fibre coordinate [z0:lambda], sets

        (z1,z2,z3,z4) = lambda*(x1*y1,-x0*y1,-x1*y0,x0*y0),

    which solves p2=q1=q2 globally, and then solves the equation p1=0 for z5.
    The returned reference parameters are useful for evaluating existing
    metric code on a much wider set than the Gaussian local sampler.
    """

    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if min_solve_coefficient <= 0:
        raise ValueError("min_solve_coefficient must be positive.")
    rng = np.random.default_rng(seed)
    x_rows: list[Array] = []
    y_rows: list[Array] = []
    z_rows: list[Array] = []
    params_rows: list[Array] = []

    while len(params_rows) < n_points:
        x = rng.normal(size=2) + 1j * rng.normal(size=2)
        y = rng.normal(size=2) + 1j * rng.normal(size=2)
        fibre = rng.normal(size=2) + 1j * rng.normal(size=2)
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        fibre /= np.linalg.norm(fibre)
        z0, lam = fibre
        coefficient = x[0] * y[0] * z0**2
        if abs(coefficient) <= min_solve_coefficient:
            continue
        z = np.asarray(
            [
                z0,
                lam * x[1] * y[1],
                -lam * x[0] * y[1],
                -lam * x[1] * y[0],
                lam * x[0] * y[0],
                0.0,
            ],
            dtype=np.complex128,
        )
        z[5] = -p1_homogeneous(x, y, z) / coefficient
        z /= np.linalg.norm(z)
        try:
            params = reference_parameters_from_homogeneous(x, y, z)
        except FloatingPointError:
            continue
        if not np.all(np.isfinite(params)):
            continue
        x_rows.append(x)
        y_rows.append(y)
        z_rows.append(z)
        params_rows.append(params)

    return (
        np.asarray(params_rows, dtype=np.complex128),
        np.asarray(x_rows, dtype=np.complex128),
        np.asarray(y_rows, dtype=np.complex128),
        np.asarray(z_rows, dtype=np.complex128),
    )


def global_geometry_diagnostic(
    *,
    intermediate_points: int = 128,
    gcicy_points: int = 64,
    seed: int = 17,
    min_selected: float = 1e-5,
) -> GlobalGeometryDiagnostic:
    """Check rational representations, transitions, equations, and smoothness."""

    reference_chart: Chart = (0, 0, 0)
    intermediate = sample_intermediate_reference_patch(intermediate_points, seed=seed)
    max_intermediate_residual = 0.0
    max_representation_error = 0.0
    max_transition_error = 0.0

    for x, y, z in intermediate:
        max_intermediate_residual = max(
            max_intermediate_residual,
            abs(p1_homogeneous(x, y, z)),
            abs(p2_homogeneous(x, y, z)),
        )
        if min(abs(x[0]), abs(x[1])) > min_selected:
            max_representation_error = max(
                max_representation_error,
                abs(q1_rational(x, y, z, 0) - q1_rational(x, y, z, 1)),
            )
        if min(abs(y[0]), abs(y[1])) > min_selected:
            max_representation_error = max(
                max_representation_error,
                abs(q2_rational(x, y, z, 0) - q2_rational(x, y, z, 1)),
            )

        reference_values = local_defining_equations_from_homogeneous(x, y, z, reference_chart)
        for chart in product(range(2), range(2), range(6)):
            if min(abs(x[chart[0]]), abs(y[chart[1]]), abs(z[chart[2]])) <= min_selected:
                continue
            candidate = local_defining_equations_from_homogeneous(x, y, z, chart)
            expected_q1 = section_transition_multiplier(x, y, z, reference_chart, chart, Q1_DEGREE) * reference_values[2]
            expected_q2 = section_transition_multiplier(x, y, z, reference_chart, chart, Q2_DEGREE) * reference_values[3]
            scale_q1 = max(1.0, abs(expected_q1), abs(candidate[2]))
            scale_q2 = max(1.0, abs(expected_q2), abs(candidate[3]))
            max_transition_error = max(
                max_transition_error,
                abs(candidate[2] - expected_q1) / scale_q1,
                abs(candidate[3] - expected_q2) / scale_q2,
            )

    params = random_parameters(gcicy_points, seed=seed + 1, scale=0.35)
    charts_with_points: set[Chart] = set()
    max_relative_equation_residual = 0.0
    min_jacobian_singular_value = float("inf")
    for point in params:
        x, y, z = homogeneous_coordinates(point)
        for chart in product(range(2), range(2), range(6)):
            if selected_coordinate_abs(point, chart) <= min_selected:
                continue
            charts_with_points.add(chart)
            coords = affine_chart_coordinates(point, chart)
            equations = local_defining_equations(coords, chart)
            scales = local_equation_scales_from_homogeneous(x, y, z, chart)
            max_relative_equation_residual = max(
                max_relative_equation_residual,
                float(np.max(np.abs(equations) / scales)),
            )
            singular_values = np.linalg.svd(local_equation_jacobian(coords, chart), compute_uv=False)
            min_jacobian_singular_value = min(min_jacobian_singular_value, float(singular_values[-1]))

    boundary_min_singular_value = float("inf")
    boundary_chart: Chart = (0, 0, 5)
    for a, b in ((0.0, 0.0), (0.3 + 0.2j, -0.4j), (1.0, 1.0)):
        boundary_coords = np.asarray([a, b, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        boundary_equations = local_defining_equations(boundary_coords, boundary_chart)
        if np.max(np.abs(boundary_equations)) > 1e-12:
            continue
        singular_values = np.linalg.svd(local_equation_jacobian(boundary_coords, boundary_chart), compute_uv=False)
        boundary_min_singular_value = min(boundary_min_singular_value, float(singular_values[-1]))

    return GlobalGeometryDiagnostic(
        intermediate_points=intermediate_points,
        gcicy_points=gcicy_points,
        charts_tested=24,
        charts_with_usable_points=len(charts_with_points),
        max_intermediate_residual=max_intermediate_residual,
        max_rational_representation_error=max_representation_error,
        max_section_transition_error=max_transition_error,
        max_gcicy_relative_equation_residual=max_relative_equation_residual,
        min_gcicy_jacobian_singular_value=min_jacobian_singular_value,
        min_boundary_jacobian_singular_value=boundary_min_singular_value,
        prototype_is_globally_smooth=bool(boundary_min_singular_value > 1e-7),
    )


def diagnostic_to_dict(diagnostic: GlobalGeometryDiagnostic) -> dict[str, float | int]:
    """Serialize a global-geometry diagnostic."""

    return {
        "intermediate_points": diagnostic.intermediate_points,
        "gcicy_points": diagnostic.gcicy_points,
        "charts_tested": diagnostic.charts_tested,
        "charts_with_usable_points": diagnostic.charts_with_usable_points,
        "max_intermediate_residual": diagnostic.max_intermediate_residual,
        "max_rational_representation_error": diagnostic.max_rational_representation_error,
        "max_section_transition_error": diagnostic.max_section_transition_error,
        "max_gcicy_relative_equation_residual": diagnostic.max_gcicy_relative_equation_residual,
        "min_gcicy_jacobian_singular_value": diagnostic.min_gcicy_jacobian_singular_value,
        "min_boundary_jacobian_singular_value": diagnostic.min_boundary_jacobian_singular_value,
        "prototype_is_globally_smooth": diagnostic.prototype_is_globally_smooth,
    }
