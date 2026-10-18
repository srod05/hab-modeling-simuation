"""Simple 2D trajectory line fitting for high-altitude balloon data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

Point = tuple[float, float]


def fit_line(points: Iterable[Point]) -> tuple[float, float]:
    """Return ``(slope, intercept)`` for the least-squares line ``y = mx + b``.

    At least two points are required, and all x-values must not be identical.
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


def plot_trajectory(
    points: Sequence[Point],
    output_path: str | Path | None = None,
    *,
    show: bool = True,
) -> tuple[float, float]:
    """Plot coordinate points and their fitted line.

    Returns the fitted ``(slope, intercept)`` so callers can inspect the model.
    """
    if len(points) < 2:
        raise ValueError("at least two coordinate points are required")

    import matplotlib.pyplot as plt

    slope, intercept = fit_line(points)
    x_values = [float(x) for x, _ in points]
    y_values = [float(y) for _, y in points]
    x_min, x_max = min(x_values), max(x_values)
    line_x = [x_min, x_max] if x_min != x_max else [x_min - 1, x_max + 1]
    line_y = [slope * x + intercept for x in line_x]

    figure, axis = plt.subplots()
    axis.scatter(x_values, y_values, label="Trajectory points")
    axis.plot(line_x, line_y, color="tab:red", label=f"Fit: y = {slope:.3f}x + {intercept:.3f}")
    axis.set_title("High-Altitude Balloon Trajectory")
    axis.set_xlabel("X coordinate")
    axis.set_ylabel("Y coordinate")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return slope, intercept


def load_points(path: str | Path) -> list[Point]:
    """Load comma-separated ``x,y`` points from a text file."""
    points: list[Point] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            x_text, y_text = stripped.split(",")
            points.append((float(x_text), float(y_text)))
        except ValueError as error:
            raise ValueError(f"invalid point on line {line_number}: {line!r}") from error
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="text file containing one x,y point per line")
    parser.add_argument("-o", "--output", type=Path, help="save the plot to this image path")
    parser.add_argument("--no-show", action="store_true", help="save without opening a plot window")
    args = parser.parse_args()

    points = load_points(args.input)
    slope, intercept = plot_trajectory(points, args.output, show=not args.no_show)
    print(f"fit: y = {slope:.6f}x + {intercept:.6f}")


if __name__ == "__main__":
    main()
