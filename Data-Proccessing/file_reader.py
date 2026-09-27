"Reads a NOAA data file and seperates meta-data and actual physical data"
"Stores the meta-data in a JSON file and physical data in a npz compressed file"
"To access the data from this compression and proccessing see data_access.py"
"""Compact ingestion of NOAA Local Climatological Data CSV files.

No unit conversion is performed. Values retain NOAA's source units; callers can
consult ``NOAA_LCD_UNITS`` before using them in a model.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


NOAA_LCD_UNITS = {
    "temperature": "degrees Fahrenheit", "dew_point": "degrees Fahrenheit",
    "relative_humidity": "percent", "sea_level_pressure": "millibars",
    "station_pressure": "millibars", "wind_direction": "degrees true",
    "wind_speed": "miles per hour", "wind_gust_speed": "miles per hour",
    "precipitation": "inches", "visibility": "miles",
    "sky_conditions": "NOAA sky-condition text",
}
_VARIABLES = {
    "temperature": "HourlyDryBulbTemperature", "dew_point": "HourlyDewPointTemperature",
    "relative_humidity": "HourlyRelativeHumidity", "sea_level_pressure": "HourlySeaLevelPressure",
    "station_pressure": "HourlyStationPressure", "wind_direction": "HourlyWindDirection",
    "wind_speed": "HourlyWindSpeed", "wind_gust_speed": "HourlyWindGustSpeed",
    "precipitation": "HourlyPrecipitation", "visibility": "HourlyVisibility",
    "sky_conditions": "HourlySkyConditions",
}
_TEXT_VARIABLES = {"sky_conditions"}
_METADATA_COLUMNS = {
    "station_id": "STATION", "station_name": "NAME", "latitude": "LATITUDE",
    "longitude": "LONGITUDE", "elevation": "ELEVATION", "report_type": "REPORT_TYPE",
    "source": "SOURCE",
}
_NUMERIC_METADATA = {"latitude", "longitude", "elevation"}


@dataclass(frozen=True)
class WeatherMetadata:
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    elevation: float
    report_type: str
    source: str
    start_time: datetime
    end_time: datetime
    observation_count: int
    nominal_sample_interval_seconds: Optional[float]
    timestamps_regular: bool


@dataclass
class WeatherDataset:
    """Sorted internal weather data; numeric missing values are ``math.nan``."""
    metadata: WeatherMetadata
    observations: dict[str, list[float | str | None]]
    time_offsets: Optional[list[float]]
    missing_value_counts: dict[str, int]
    raw_row_count: int
    duplicate_timestamp_count: int
    missing_timestamp_count: int

    def time_at(self, index: int) -> datetime:
        """Reconstruct a compact timestamp or retrieve an explicit one."""
        seconds = (index * self.metadata.nominal_sample_interval_seconds
                   if self.time_offsets is None else self.time_offsets[index])
        return self.metadata.start_time + timedelta(seconds=seconds or 0)

    def validation_report(self) -> dict[str, object]:
        return {
            "raw_row_count": self.raw_row_count,
            "usable_observation_count": self.metadata.observation_count,
            "start_time": self.metadata.start_time, "end_time": self.metadata.end_time,
            "nominal_sample_interval_seconds": self.metadata.nominal_sample_interval_seconds,
            "timestamps_regular": self.metadata.timestamps_regular,
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "missing_timestamp_count": self.missing_timestamp_count,
            "missing_value_counts": self.missing_value_counts.copy(),
        }


def _parse_time(value: str) -> datetime:
    value = value.strip()
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _number(value: str) -> float:
    try:
        return float(value.strip()) if value and value.strip() else math.nan
    except ValueError:
        return math.nan


def _constant_metadata(rows: list[dict[str, str]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for name, column in _METADATA_COLUMNS.items():
        distinct = {row.get(column, "").strip() for row in rows}
        if len(distinct) != 1:
            raise ValueError(f"Inconsistent metadata in {column}: {sorted(distinct)!r}")
        value = distinct.pop()
        if name in _NUMERIC_METADATA:
            numeric = _number(value)
            if math.isnan(numeric):
                raise ValueError(f"Missing or invalid required metadata {column!r}")
            values[name] = numeric
        elif not value:
            raise ValueError(f"Missing required metadata {column!r}")
        else:
            values[name] = value
    return values


def _timing_summary(times: list[datetime]) -> tuple[Optional[float], bool, int, int]:
    """Return nominal interval, regularity, duplicates, and inferred gap count."""
    if len(times) < 2:
        return None, True, 0, 0
    deltas = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    duplicates = sum(delta == 0 for delta in deltas)
    positive = [delta for delta in deltas if delta > 0]
    if not positive:
        return None, False, duplicates, 0
    nominal = Counter(positive).most_common(1)[0][0]
    regular = duplicates == 0 and all(delta == nominal for delta in deltas)
    missing = sum(
        round(delta / nominal) - 1 for delta in positive
        if round(delta / nominal) > 1 and math.isclose(delta / nominal, round(delta / nominal), abs_tol=1e-9)
    )
    return nominal, regular, duplicates, missing


def ingest_noaa_csv(path: str | Path) -> WeatherDataset:
    """Read, validate, sort, and compact a NOAA LCD CSV.

    Invalid or absent DATE rows are excluded from usable observations. Metadata
    is checked across every source row. A missing source column becomes an
    all-missing retained variable rather than failing the ingestion.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "DATE" not in reader.fieldnames:
            raise ValueError("NOAA CSV must include a DATE column")
        rows = list(reader)
        columns = set(reader.fieldnames)
    if not rows:
        raise ValueError("NOAA CSV contains no data rows")
    metadata_values = _constant_metadata(rows)
    dated_rows = []
    for row in rows:
        try:
            dated_rows.append((_parse_time(row.get("DATE", "")), row))
        except (TypeError, ValueError):
            continue
    if not dated_rows:
        raise ValueError("NOAA CSV contains no usable DATE values")
    dated_rows.sort(key=lambda item: item[0])
    times = [item[0] for item in dated_rows]
    nominal, regular, duplicates, missing_timestamps = _timing_summary(times)

    observations: dict[str, list[float | str | None]] = {name: [] for name in _VARIABLES}
    missing_counts = {name: 0 for name in _VARIABLES}
    for _, row in dated_rows:
        for name, column in _VARIABLES.items():
            raw = row.get(column, "") if column in columns else ""
            if name in _TEXT_VARIABLES:
                value: float | str | None = raw.strip() or None
                absent = value is None
            else:
                value = _number(raw)
                absent = math.isnan(value)
            observations[name].append(value)
            missing_counts[name] += absent
    start, end = times[0], times[-1]
    metadata = WeatherMetadata(
        **metadata_values, start_time=start, end_time=end, observation_count=len(times),
        nominal_sample_interval_seconds=nominal, timestamps_regular=regular,
    )
    return WeatherDataset(
        metadata=metadata, observations=observations,
        time_offsets=None if regular else [(time - start).total_seconds() for time in times],
        missing_value_counts=missing_counts, raw_row_count=len(rows),
        duplicate_timestamp_count=duplicates, missing_timestamp_count=missing_timestamps,
    )
