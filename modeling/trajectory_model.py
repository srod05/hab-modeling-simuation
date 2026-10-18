"""Fit and plot polynomial paths through high-altitude balloon observations.

Points are positions in two or three dimensions, in observation order. Supply
actual timestamps to :func:`fit_trajectory` when they are available; otherwise
the model uses the observation index as its parameter. A quadratic model fits
each coordinate independently against that shared parameter, so it can describe
a curved path without requiring one position coordinate to be a function of
another.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

Coordinate = tuple[float, ...]
Point = tuple[float, float]


@dataclass(frozen=True)
class PolynomialTrajectory:
    """A polynomial parametric curve in two or more dimensions.

    ``coefficients[dimension][power]`` stores coefficients in ascending
    powers of the normalized parameter. Use :meth:`predict` to evaluate the
    curve in the original parameter units.
    """

    coefficients: tuple[tuple[float, ...], ...]
    parameter_center: float
    parameter_scale: float
    parameter_min: float
    parameter_max: float
    rms_residuals: tuple[float, ...]

    @property
    def dimensions(self) -> int:
        """Number of coordinates modeled by this curve."""
        return len(self.coefficients)

    @property
    def degree(self) -> int:
        """Polynomial degree of this curve."""
        return len(self.coefficients[0]) - 1

    def predict(self, parameter: float) -> Coordinate:
        """Return the predicted position at a time or sample parameter."""
        normalized = (float(parameter) - self.parameter_center) / self.parameter_scale
        return tuple(
            sum(coefficient * normalized**power for power, coefficient in enumerate(axis_coefficients))
            for axis_coefficients in self.coefficients
        )

    def predict_many(self, parameters: Iterable[float]) -> list[Coordinate]:
        """Evaluate the trajectory at several parameter values."""
        return [self.predict(parameter) for parameter in parameters]


def fit_line(points: Iterable[Point]) -> tuple[float, float]:
    """Return ``(slope, intercept)`` for the least-squares line ``y = mx + b``.

    At least two points are required, and all x-values must not be identical.
    This is retained for callers that specifically need a straight 2D fit.
    """
    data = [(float(x), float(y)) for x, y in points]
    if len(data) < 2:
        raise ValueError("at least two coordinate points are required")

    x_values = [x for x, _ in data]
    y_values = [y for _, y in data]
    x_mean = sum(x_values) / len(data)
    y_mean = sum(y_values) / len(data)
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0:
        raise ValueError("a line cannot be fit when all x-values are identical")

    slope = sum((x - x_mean) * (y - y_mean) for x, y in data) / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept



def fit_quadratic(points: Iterable[Point]) -> tuple[float, float, float]:
    """Fit ``y = a*x² + b*x + c`` and return ``(a, b, c)``.

    This is useful when the observed path is known to be single-valued in x.
    For general paths, including paths that turn back in x, use
    :func:`fit_trajectory` instead.
    """
    data = np.asarray([(float(x), float(y)) for x, y in points], dtype=float)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError("at least three coordinate points are required for a quadratic fit")
    if not np.isfinite(data).all():
        raise ValueError("coordinate points must be finite numbers")

    x_values, y_values = data[:, 0], data[:, 1]
    if np.unique(x_values).size < 3:
        raise ValueError("a quadratic fit requires at least three distinct x-values")

    center = (float(x_values.min()) + float(x_values.max())) / 2
    scale = (float(x_values.max()) - float(x_values.min())) / 2
    normalized_x = (x_values - center) / scale
    design = np.column_stack((normalized_x**2, normalized_x, np.ones_like(normalized_x)))
    normalized_a, normalized_b, normalized_c = np.linalg.lstsq(design, y_values, rcond=None)[0]

    a = normalized_a / scale**2
    b = normalized_b / scale - (2 * normalized_a * center) / scale**2
    c = normalized_a * center**2 / scale**2 - normalized_b * center / scale + normalized_c
    return float(a), float(b), float(c)


def fit_trajectory(
    points: Iterable[Sequence[float]],
    *,
    times: Iterable[float] | None = None,
    degree: int = 2,
) -> PolynomialTrajectory:
    """Fit a polynomial trajectory through ordered 2D/3D (or higher) points.

    Each coordinate is modeled as a polynomial of the same time parameter.
    When ``times`` is omitted, point order is used (0, 1, 2, ...); those values
    are sample indices, not seconds. For flight measurements, pass elapsed
    seconds alongside positions, for example ``fit_trajectory(xyz, times=t)``.

    The default degree is two, which gives a quadratic parametric path. The
    least-squares fit requires at least ``degree + 1`` observations and enough
    distinct parameter values to determine the coefficients.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")

    positions = np.asarray([tuple(float(value) for value in point) for point in points], dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("trajectory points must have at least two coordinates each")
    if positions.shape[0] < degree + 1:
        raise ValueError(f"at least {degree + 1} points are required for degree {degree}")
    if not np.isfinite(positions).all():
        raise ValueError("trajectory coordinates must be finite numbers")

    if times is None:
        parameter = np.arange(positions.shape[0], dtype=float)
    else:
        parameter = np.asarray([float(value) for value in times], dtype=float)
        if parameter.ndim != 1 or parameter.size != positions.shape[0]:
            raise ValueError("times must contain one value for each trajectory point")
        if not np.isfinite(parameter).all():
            raise ValueError("times must be finite numbers")

    parameter_min, parameter_max = float(parameter.min()), float(parameter.max())
    if parameter_max == parameter_min:
        raise ValueError("trajectory times must not all be identical")
    parameter_center = (parameter_min + parameter_max) / 2
    parameter_scale = (parameter_max - parameter_min) / 2
    normalized_parameter = (parameter - parameter_center) / parameter_scale
    design = np.vander(normalized_parameter, N=degree + 1, increasing=True)
    fitted, _, rank, _ = np.linalg.lstsq(design, positions, rcond=None)
    if rank < degree + 1:
        raise ValueError("not enough distinct time values to fit this polynomial degree")

    coefficients = tuple(
        tuple(float(value) for value in fitted[:, dimension])
        for dimension in range(positions.shape[1])
    )
    residuals = positions - design @ fitted
    rms_residuals = tuple(float(value) for value in np.sqrt(np.mean(residuals**2, axis=0)))
    return PolynomialTrajectory(
        coefficients=coefficients,
        parameter_center=parameter_center,
        parameter_scale=parameter_scale,
        parameter_min=parameter_min,
        parameter_max=parameter_max,
        rms_residuals=rms_residuals,
    )


def plot_trajectory(
    points: Sequence[Sequence[float]],
    output_path: str | Path | None = None,
    *,
    show: bool = True,
    degree: int = 2,
    times: Iterable[float] | None = None,
) -> PolynomialTrajectory:
    """Plot observed positions and their fitted polynomial curve.

    Two-coordinate data is shown in the x-y plane; three-coordinate data is
    shown in 3D. The fitted model is returned so callers can make predictions.
    """
    model = fit_trajectory(points, times=times, degree=degree)
    if model.dimensions not in (2, 3):
        raise ValueError("plotting supports two- or three-dimensional positions")

    import matplotlib.pyplot as plt

    parameters = np.linspace(model.parameter_min, model.parameter_max, 200)
    curve = np.asarray(model.predict_many(parameters), dtype=float)
    observed = np.asarray(points, dtype=float)

    if model.dimensions == 2:
        figure, axis = plt.subplots()
        axis.scatter(observed[:, 0], observed[:, 1], label="Observed positions")
        axis.plot(curve[:, 0], curve[:, 1], color="tab:red", label=f"Degree {degree} fit")
        axis.set_xlabel("X coordinate")
        axis.set_ylabel("Y coordinate")
        axis.set_title("High-Altitude Balloon Trajectory")
    else:
        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(observed[:, 0], observed[:, 1], observed[:, 2], label="Observed positions")
        axis.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="tab:red", label=f"Degree {degree} fit")
        axis.set_xlabel("X coordinate")
        axis.set_ylabel("Y coordinate")
        axis.set_zlabel("Altitude / Z coordinate")
        axis.set_title("High-Altitude Balloon Trajectory")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return model


def load_points(path: str | Path) -> list[Coordinate]:
    """Load comma-separated 2D or higher-dimensional positions from a file.

    Blank lines and lines beginning with ``#`` are ignored. Each data row must
    contain at least two numbers. Coordinates should be in a consistent order,
    such as ``x,y`` or ``x,y,z``; row order supplies the default parameter.
    """
    points: list[Coordinate] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            values = tuple(float(value.strip()) for value in stripped.split(","))
            if len(values) < 2:
                raise ValueError("at least two coordinates are required")
            points.append(values)
        except ValueError as error:
            raise ValueError(f"invalid point on line {line_number}: {line!r}") from error
    return points


def load_trajectory_data(
    path: str | Path,
) -> tuple[list[Coordinate], list[float] | None]:
    """Load positions and optional numeric times from a CSV file.

    Files without a header contain one position per row (``x,y`` or ``x,y,z``).
    A header containing ``time`` or ``t`` may include the time column in any
    position, for example ``time,x,y,z``. Other columns are treated as the
    coordinates, in their listed order.
    """
    lines = Path(path).read_text().splitlines()
    header: list[str] | None = None
    records: list[list[float]] = []
    recognized_coordinate_headers = {
        "x", "y", "z", "east", "north", "up", "latitude", "longitude", "altitude",
        "lat", "lon", "alt",
    }
    time_headers = {"t", "time"}

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = [field.strip() for field in stripped.split(",")]
        try:
            records.append([float(field) for field in fields])
        except ValueError as error:
            normalized_fields = [field.lower() for field in fields]
            is_header = (
                header is None
                and not records
                and len(fields) >= 2
                and (
                    (time_headers & set(normalized_fields))
                    or set(normalized_fields) <= recognized_coordinate_headers
                )
            )
            if is_header:
                header = normalized_fields
                continue
            raise ValueError(f"invalid point on line {line_number}: {line!r}") from error

    if not records:
        return [], None

    if header is None:
        points = [tuple(record) for record in records]
        if any(len(point) < 2 for point in points):
            raise ValueError("trajectory points must contain at least two coordinates")
        if len({len(record) for record in records}) != 1:
            raise ValueError("all trajectory rows must contain the same number of values")
        return points, None

    if len(header) != len(records[0]) or any(len(record) != len(header) for record in records):
        raise ValueError("CSV header and trajectory rows must have the same number of columns")
    time_index = next((index for index, name in enumerate(header) if name in time_headers), None)
    if time_index is None:
        points = [tuple(record) for record in records]
        if len(points[0]) < 2:
            raise ValueError("trajectory points must contain at least two coordinates")
        return points, None

    coordinate_indices = [index for index in range(len(header)) if index != time_index]
    if len(coordinate_indices) < 2:
        raise ValueError("a timed trajectory needs at least two coordinate columns")
    times = [record[time_index] for record in records]
    points = [tuple(record[index] for index in coordinate_indices) for record in records]
    return points, times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="CSV with ordered x,y[,z] positions; a time/t header column is optional",
    )
    parser.add_argument("-o", "--output", type=Path, help="save the plot to this image path")
    parser.add_argument("--degree", type=int, default=2, help="polynomial degree (default: quadratic)")
    parser.add_argument("--no-show", action="store_true", help="save without opening a plot window")
    args = parser.parse_args()

    points, times = load_trajectory_data(args.input)
    model = plot_trajectory(points, args.output, show=not args.no_show, degree=args.degree, times=times)
    parameter_name = "time" if times is not None else "sample index"
    print(
        f"u = ({parameter_name} - {model.parameter_center:.6g}) "
        f"/ {model.parameter_scale:.6g}"
    )
    for coordinate, coefficients in enumerate(model.coefficients):
        terms = " + ".join(
            f"{value:.6g}·u^{power}" if power else f"{value:.6g}"
            for power, value in enumerate(coefficients)
        )
        print(
            f"coordinate {coordinate + 1} vs normalized {parameter_name}: {terms} "
            f"(RMS residual {model.rms_residuals[coordinate]:.6g})"
        )
    parameter_source = "CSV time column" if times is not None else "row order"
    print(f"fit: degree {model.degree}, {model.dimensions} dimensions, parameterized by {parameter_source}")


if __name__ == "__main__":
    main()
