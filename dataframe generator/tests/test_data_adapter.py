import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

from wind_interpolation import NetCDFWindField, get_wind_dataframe


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "data" / "philadelphia.json").read_text(encoding="utf-8"))
NC_FILE = ROOT / "data" / "era5_pressure_levels.nc"
EXPECTED_COLUMNS = ["valid_time", "pressure_level", "latitude", "longitude", "u", "v"]


class DataAdapterTests(unittest.TestCase):
    def test_four_corner_dataframe_schema_and_values(self):
        points = EXAMPLE["grid_points"]
        frame = get_wind_dataframe(39.5, -75.5, 10275, points, valid_time="2026-10-14T12:00:00Z")
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(frame.columns.tolist(), EXPECTED_COLUMNS)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["pressure_level"], 250)
        self.assertAlmostEqual(frame.iloc[0]["u"], 13)
        self.assertAlmostEqual(frame.iloc[0]["v"], -5.5)
        self.assertEqual(frame.iloc[0]["valid_time"], pd.Timestamp("2026-10-14T12:00:00Z"))

    def test_multiple_requests_and_time_index(self):
        frame = get_wind_dataframe(
            latitude=[39.0, 39.5], longitude=[-76.0, -75.5], altitude_m=10275,
            grid_data=EXAMPLE["grid_points"], valid_time="2026-10-14", valid_time_as_index=True,
        )
        self.assertEqual(frame.index.name, "valid_time")
        self.assertEqual(frame.columns.tolist(), EXPECTED_COLUMNS[1:])
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["u"], 12)
        self.assertAlmostEqual(frame.iloc[1]["v"], -5.5)

    def test_era5_dataframe_with_time_index(self):
        rows = []
        for corner in ("SW", "SE", "NW", "NE"):
            point = EXAMPLE["grid_points"][corner]
            wind = point["levels"]["250"]
            rows.append({
                "valid_time": "2025-10-01", "pressure_level": 250,
                "latitude": point["latitude"], "longitude": point["longitude"],
                "u": wind["u_ms"], "v": wind["v_ms"],
            })
        era5 = pd.DataFrame(rows).set_index("valid_time")
        result = get_wind_dataframe(39.5, -75.5, 10000, era5)
        self.assertEqual(result.columns.tolist(), EXPECTED_COLUMNS)
        self.assertEqual(result.iloc[0]["pressure_level"], 250)
        self.assertAlmostEqual(result.iloc[0]["u"], 13)
        self.assertAlmostEqual(result.iloc[0]["v"], -5.5)
        with self.assertRaisesRegex(ValueError, "Expected one ERA5 row"):
            get_wind_dataframe(39.5, -75.5, 10000, era5.iloc[:-1])

    @unittest.skipUnless(NC_FILE.exists(), "Bundled ERA5 NetCDF file is unavailable")
    def test_netcdf_dataframe_matches_interpolation_engine(self):
        field = NetCDFWindField(NC_FILE)
        frame = get_wind_dataframe(40.5, -75.5, 10000, field)
        engine = field.estimate(40.5, -75.5, 10000)
        self.assertEqual(frame.columns.tolist(), EXPECTED_COLUMNS)
        self.assertEqual(frame.iloc[0]["pressure_level"], engine["interpolation"]["pressure_level_hpa"])
        self.assertTrue(math.isclose(frame.iloc[0]["u"], engine["wind"]["u_ms"]))
        self.assertTrue(math.isclose(frame.iloc[0]["v"], engine["wind"]["v_ms"]))
        self.assertEqual(frame.iloc[0]["valid_time"], pd.Timestamp(field.valid_time))

    def test_core_import_does_not_load_plotting_or_examples(self):
        code = (
            "import sys; import wind_interpolation; "
            "assert 'matplotlib' not in sys.modules; "
            "assert 'visualization' not in sys.modules; "
            "assert 'read_wind' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
