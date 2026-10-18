import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from data_access import WeatherData
from file_reader import ingest_noaa_csv, save_processed_weather


class WeatherDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.metadata_path = self.directory / "metadata.json"
        self.observations_path = self.directory / "observations.json"
        self.metadata_path.write_text(json.dumps({
            "station_id": "TEST",
            "station_name": "Test Station",
            "latitude": 40.0,
            "longitude": -76.0,
            "elevation": 100.0,
            "report_type": "FM-15",
            "source": "test",
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T02:00:00",
            "observation_count": 3,
            "nominal_sample_interval_seconds": 3600,
            "timestamps_regular": True,
        }))
        self.observations_path.write_text(json.dumps({
            "observations": {
                "temperature": [10.0, None, 12.0],
                "wind_speed": [1.0, 2.0, 3.0],
                "visibility": [1.0, float("inf"), 3.0],
                "sky_conditions": [None, "CLR", ""],
            }
        }))
        self.weather = WeatherData(self.metadata_path, self.observations_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_json_missing_values_use_type_appropriate_sentinels(self):
        self.assertTrue(math.isnan(self.weather.get_variable("temperature")[1]))
        self.assertIsNone(self.weather.get_variable("sky_conditions")[0])
        self.assertIsNone(self.weather.get_variable("sky_conditions")[2])

    def test_invalid_requests_are_descriptive(self):
        with self.assertRaisesRegex(KeyError, "Unknown variable"):
            self.weather.get_variable("not_a_variable")
        with self.assertRaisesRegex(ValueError, "Time must be"):
            self.weather.at_time("25:00")
        with self.assertRaisesRegex(KeyError, "No observation exists"):
            self.weather.at_time("2026-01-01T00:30:00")

    def test_empty_range_returns_aligned_empty_arrays(self):
        result = self.weather.range(
            "2026-01-02T00:00:00",
            "2026-01-02T01:00:00",
            variables=["temperature"],
        )
        self.assertEqual(result, {"time": [], "temperature": []})

    def test_numerical_validation_rejects_missing_and_non_finite_values(self):
        self.weather.validate_numerical("wind_speed")
        with self.assertRaisesRegex(ValueError, "temperature.*1 missing"):
            self.weather.validate_numerical("temperature")
        with self.assertRaisesRegex(ValueError, "visibility.*1 missing"):
            self.weather.validate_for_modeling(["visibility"])

    def test_reader_reports_missing_malformed_and_invalid_timestamps(self):
        csv_path = self.directory / "weather.csv"
        fields = [
            "STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION",
            "REPORT_TYPE", "SOURCE", "DATE", "HourlyDryBulbTemperature",
            "HourlySkyConditions",
        ]
        base = {
            "STATION": "TEST", "NAME": "Test Station", "LATITUDE": "40",
            "LONGITUDE": "-76", "ELEVATION": "100", "REPORT_TYPE": "FM-15",
            "SOURCE": "test", "HourlySkyConditions": "",
        }
        rows = [
            {**base, "DATE": "2026-01-01T00:00:00", "HourlyDryBulbTemperature": ""},
            {**base, "DATE": "2026-01-01T01:00:00", "HourlyDryBulbTemperature": "bad"},
            {**base, "DATE": "2026-01-01T02:00:00", "HourlyDryBulbTemperature": "12"},
            {**base, "DATE": "", "HourlyDryBulbTemperature": "14"},
            {**base, "DATE": "not-a-time", "HourlyDryBulbTemperature": "15"},
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        dataset = ingest_noaa_csv(csv_path)

        self.assertEqual(dataset.invalid_timestamp_count, 2)
        self.assertEqual(dataset.missing_value_counts["temperature"], 1)
        self.assertEqual(dataset.malformed_value_counts["temperature"], 1)
        report = dataset.validation_report()
        self.assertEqual(report["invalid_timestamp_count"], 2)
        self.assertEqual(report["missing_value_counts"]["temperature"], 1)
        self.assertEqual(report["malformed_value_counts"]["temperature"], 1)
        self.assertTrue(math.isnan(dataset.observations["temperature"][0]))
        self.assertTrue(math.isnan(dataset.observations["temperature"][1]))
        self.assertIsNone(dataset.observations["sky_conditions"][0])

        saved_metadata = self.directory / "saved_metadata.json"
        saved_observations = self.directory / "saved_observations.json"
        save_processed_weather(dataset, saved_metadata, saved_observations)
        metadata = json.loads(saved_metadata.read_text())
        self.assertEqual(metadata["invalid_timestamp_count"], 2)
        self.assertEqual(metadata["malformed_value_counts"]["temperature"], 1)
        payload = json.loads(saved_observations.read_text())
        self.assertEqual(payload["observations"]["temperature"][:2], [None, None])
        reloaded = WeatherData(saved_metadata, saved_observations)
        self.assertTrue(math.isnan(reloaded.get_variable("temperature")[0]))
        self.assertTrue(math.isnan(reloaded.get_variable("temperature")[1]))


if __name__ == "__main__":
    unittest.main()
