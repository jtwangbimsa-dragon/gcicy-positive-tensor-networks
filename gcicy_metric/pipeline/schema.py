"""Configuration-matrix schema for reusable gCICY metric pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, prod


Degree = tuple[int, ...]


@dataclass(frozen=True)
class GCICYConfiguration:
    """Static data that can be validated directly from a gCICY matrix.

    Explicit generalized sections are deliberately not part of this class.
    A configuration matrix fixes their line bundles, but it does not by itself
    provide the local section representatives needed for numerical work.
    """

    key: str
    name: str
    ambient_dimensions: tuple[int, ...]
    positive_columns: tuple[Degree, ...]
    generalized_columns: tuple[Degree, ...]
    kahler_line_bundle: Degree
    complex_dimension: int = 3
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.name:
            raise ValueError("configuration key and name must be non-empty")
        if not self.ambient_dimensions or min(self.ambient_dimensions) < 1:
            raise ValueError("ambient projective dimensions must be positive")
        if self.complex_dimension <= 0:
            raise ValueError("complex dimension must be positive")

        factor_count = len(self.ambient_dimensions)
        columns = self.positive_columns + self.generalized_columns
        if not columns:
            raise ValueError("a gCICY configuration must contain defining columns")
        if any(len(column) != factor_count for column in columns):
            raise ValueError("every degree column must match the number of ambient factors")
        if len(self.kahler_line_bundle) != factor_count:
            raise ValueError("the Kahler line bundle must match the ambient factors")
        if any(entry < 0 for column in self.positive_columns for entry in column):
            raise ValueError("positive-stage columns cannot contain negative degrees")
        if any(not any(entry < 0 for entry in column) for column in self.generalized_columns):
            raise ValueError("every generalized column must contain a negative degree")
        if min(self.kahler_line_bundle) <= 0:
            raise ValueError("the pipeline requires a positive ambient Kahler line bundle")

        expected_dimension = sum(self.ambient_dimensions) - len(columns)
        if expected_dimension != self.complex_dimension:
            raise ValueError(
                f"configuration has dimension {expected_dimension}, expected {self.complex_dimension}"
            )
        row_sums = tuple(sum(column[row] for column in columns) for row in range(factor_count))
        calabi_yau_sums = tuple(dimension + 1 for dimension in self.ambient_dimensions)
        if row_sums != calabi_yau_sums:
            raise ValueError(
                f"Calabi-Yau row sums {row_sums} do not match {calabi_yau_sums}"
            )

    @property
    def type_pair(self) -> tuple[int, int]:
        return len(self.positive_columns), len(self.generalized_columns)

    @property
    def type_label(self) -> str:
        return f"({self.type_pair[0]},{self.type_pair[1]})"

    @property
    def codimension(self) -> int:
        return sum(self.type_pair)

    @property
    def ambient_affine_dimension(self) -> int:
        return sum(self.ambient_dimensions)

    @property
    def projective_chart_count(self) -> int:
        return prod(dimension + 1 for dimension in self.ambient_dimensions)

    @property
    def implicit_coordinate_choice_count(self) -> int:
        return comb(self.ambient_affine_dimension, self.complex_dimension)

    @property
    def columns(self) -> tuple[Degree, ...]:
        return self.positive_columns + self.generalized_columns

    def kahler_power(self, degree: Degree) -> int:
        """Return k when ``degree`` represents the configured line bundle L^k."""

        if len(degree) != len(self.kahler_line_bundle):
            raise ValueError("section degree does not match the ambient factors")
        powers = []
        for value, base in zip(degree, self.kahler_line_bundle, strict=True):
            if value <= 0 or value % base != 0:
                raise ValueError("section degree must be a positive power of the Kahler line bundle")
            powers.append(value // base)
        if len(set(powers)) != 1:
            raise ValueError("all section-degree components must use the same Kahler power")
        return powers[0]

    def to_dict(self) -> dict:
        rows = []
        for row, dimension in enumerate(self.ambient_dimensions):
            rows.append(
                {
                    "ambient": f"P^{dimension}",
                    "degrees": [int(column[row]) for column in self.columns],
                }
            )
        return {
            "key": self.key,
            "name": self.name,
            "type": self.type_label,
            "ambient_dimensions": list(self.ambient_dimensions),
            "positive_columns": [list(column) for column in self.positive_columns],
            "generalized_columns": [list(column) for column in self.generalized_columns],
            "kahler_line_bundle": list(self.kahler_line_bundle),
            "complex_dimension": self.complex_dimension,
            "codimension": self.codimension,
            "projective_chart_count": self.projective_chart_count,
            "implicit_coordinate_choice_count": self.implicit_coordinate_choice_count,
            "matrix_rows": rows,
            "source": self.source,
        }
