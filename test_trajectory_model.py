import math

import pytest

from trajectory_model import fit_line, load_points


def test_fit_line_returns_expected_slope_and_intercept():
    slope, intercept = fit_line([(0, 1), (1, 3), (2, 5), (3, 7)])

    assert math.isclose(slope, 2.0)
    assert math.isclose(intercept, 1.0)


def test_fit_line_rejects_insufficient_points():
    with pytest.raises(ValueError, match="at least two"):
        fit_line([(1, 2)])


def test_fit_line_rejects_vertical_input():
    with pytest.raises(ValueError, match="identical"):
        fit_line([(1, 0), (1, 2)])


def test_load_points_ignores_blank_lines_and_comments(tmp_path):
    points_file = tmp_path / "points.csv"
    points_file.write_text("# x,y\n\n1,2\n3.5,4\n")

    assert load_points(points_file) == [(1.0, 2.0), (3.5, 4.0)]
