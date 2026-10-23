import copy
import json
import math
import unittest
from pathlib import Path

from wind_interpolation import (
    bilinear_interpolate,
    interpolate_wind,
    select_pressure_level,
    wind_to_speed_direction,
)


EXAMPLE = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "philadelphia.json").read_text()
)


class WindInterpolationTests(unittest.TestCase):
    def test_corner_matches_its_wind_and_altitude(self):
        points = EXAMPLE["grid_points"]
        for corner, point in points.items():
            with self.subTest(corner=corner):
                result = interpolate_wind(points, point["latitude"], point["longitude"], 10275)
                expected = point["levels"]["250"]
                self.assertEqual(result["wind"]["u_ms"], expected["u_ms"])
                self.assertEqual(result["wind"]["v_ms"], expected["v_ms"])
                self.assertEqual(result["interpolation"]["matched_altitude_m"], expected["altitude_m"])

    def test_center_is_average_of_corners(self):
        result = interpolate_wind(EXAMPLE["grid_points"], 39.5, -75.5, 10275)
        self.assertEqual(result["interpolation"]["pressure_level_hpa"], 250)
        self.assertAlmostEqual(result["interpolation"]["matched_altitude_m"], 10275)
        self.assertAlmostEqual(result["wind"]["u_ms"], 13)
        self.assertAlmostEqual(result["wind"]["v_ms"], -5.5)
        self.assertAlmostEqual(result["wind"]["speed_ms"], math.hypot(13, -5.5))

    def test_pressure_selection_uses_location_specific_altitude(self):
        points = copy.deepcopy(EXAMPLE["grid_points"])
        for corner in points:
            points[corner]["levels"]["250"]["altitude_m"] = 9000
            points[corner]["levels"]["500"]["altitude_m"] = 8000
        points["NE"]["levels"]["250"]["altitude_m"] = 11000
        result = select_pressure_level(points, 40, -75, 10500)
        self.assertEqual(result["pressure_level_hpa"], 250)
        self.assertEqual(result["matched_altitude_m"], 11000)

    def test_missing_all_altitudes_uses_marked_approximation(self):
        points = copy.deepcopy(EXAMPLE["grid_points"])
        for point in points.values():
            for level in point["levels"].values():
                del level["altitude_m"]
        result = interpolate_wind(points, 39.5, -75.5, 10000)
        self.assertEqual(result["interpolation"]["pressure_level_hpa"], 250)
        self.assertEqual(result["interpolation"]["altitude_source"], "standard_atmosphere_approximation")

    def test_direction_wraparound_and_calm(self):
        _, west_of_north = wind_to_speed_direction(0.1, -10)
        _, east_of_north = wind_to_speed_direction(-0.1, -10)
        self.assertGreater(west_of_north, 359)
        self.assertLess(east_of_north, 1)
        self.assertEqual(wind_to_speed_direction(0, 0), (0, None))
        self.assertEqual(wind_to_speed_direction(-1, 0)[1], 90)

    def test_invalid_rectangles_and_outside_request(self):
        points = copy.deepcopy(EXAMPLE["grid_points"])
        points["NE"]["latitude"] = 39.9
        with self.assertRaisesRegex(ValueError, "axis-aligned"):
            interpolate_wind(points, 39.5, -75.5, 10000)
        with self.assertRaisesRegex(ValueError, "outside the grid"):
            interpolate_wind(EXAMPLE["grid_points"], 40.1, -75.5, 10000)

    def test_missing_pressure_level_is_reported(self):
        points = copy.deepcopy(EXAMPLE["grid_points"])
        del points["NE"]["levels"]["250"]
        with self.assertRaisesRegex(ValueError, "missing .*250"):
            interpolate_wind(points, 39.5, -75.5, 10000)

    def test_partial_altitude_metadata_is_reported(self):
        points = copy.deepcopy(EXAMPLE["grid_points"])
        del points["NE"]["levels"]["250"]["altitude_m"]
        with self.assertRaisesRegex(ValueError, "Altitude metadata"):
            interpolate_wind(points, 39.5, -75.5, 10000)

    def test_invalid_values_are_reported(self):
        points = copy.deepcopy(EXAMPLE["grid_points"])
        points["SW"]["levels"]["250"]["u_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "SW.levels\\[250\\].u_ms"):
            interpolate_wind(points, 39.5, -75.5, 10000)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            bilinear_interpolate(1, 2, 3, 4, 1.1, 0.5)


if __name__ == "__main__":
    unittest.main()
