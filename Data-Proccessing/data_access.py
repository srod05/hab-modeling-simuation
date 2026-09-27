# This file pulls the information created in the file_access.py file
"""
Available data points include

Station/dataset metadata:
    station_id, station_name, latitude, longitude, elevation, report_type,
    source, start_time, end_time, observation_count,
    nominal_sample_interval_seconds, timestamps_regular

Time-varying observations (kept in the original NOAA LCD units):
    temperature, dew_point, relative_humidity, sea_level_pressure,
    station_pressure, wind_direction, wind_speed, wind_gust_speed,
    precipitation, visibility, sky_conditions

Timing data:
    time_offsets is present only when timestamps are irregular. Otherwise each
    timestamp is reconstructed from start_time and sample_interval.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date as Date
from datetime import datetime, time, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence


_REQUIRED_METADATA = {
    "station_id", "station_name", "latitude", "longitude", "elevation",
    "report_type", "source", "start_time", "end_time", "observation_count",
    "nominal_sample_interval_seconds", "timestamps_regular",
}
_TIME_ONLY = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class WeatherData:
    """Load cleaned weather files once and expose time-aware observation access.

    Observation sequences returned by this class are the loaded arrays, not
    copies. Treat them as read-only. Range results contain newly-created slices
    so that their indexes remain aligned with the returned ``time`` array.
    """

    def __init__(self, metadata_path: str | Path, observations_path: str | Path):
        self._metadata_path = Path(metadata_path)
        self._observations_path = Path(observations_path)
        self._metadata = self._load_metadata(self._metadata_path)
        self._observations, self._time_offsets = self._load_observations(self._observations_path)
        self._validate()

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any]:
        try:
            with path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load metadata file {path}: {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError("Metadata file must contain a JSON object")
        missing = _REQUIRED_METADATA - metadata.keys()
        if missing:
            raise ValueError(f"Metadata missing required fields: {sorted(missing)}")
        for field in ("start_time", "end_time"):
            try:
                metadata[field] = datetime.fromisoformat(metadata[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Metadata {field!r} must be an ISO-8601 datetime") from error
        return metadata

    @staticmethod
    def _load_observations(path: Path) -> tuple[dict[str, Sequence[Any]], Optional[Sequence[float]]]:
        if path.suffix.lower() == ".npz":
            try:
                import numpy as np
                archive = np.load(path, mmap_mode="r", allow_pickle=False)
            except (ImportError, OSError, ValueError) as error:
                raise ValueError(f"Cannot load NPZ observations file {path}: {error}") from error
            offsets = archive["time_offsets"] if "time_offsets" in archive.files else None
            observations = {key: archive[key] for key in archive.files if key != "time_offsets"}
            # file_reader encodes NPZ sky-condition missing values as empty
            # unicode strings because object arrays are unsafe to load.
            if "sky_conditions" in observations:
                observations["sky_conditions"] = [
                    str(value) if str(value) else None for value in observations["sky_conditions"]
                ]
            return observations, offsets
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load observations file {path}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("Observations file must contain a JSON object")
        observations = payload.get("observations", payload)
        offsets = payload.get("time_offsets")
        # In direct JSON form, time_offsets is control data rather than a variable.
        if observations is payload:
            observations = {key: value for key, value in payload.items() if key != "time_offsets"}
        if not isinstance(observations, dict) or not observations:
            raise ValueError("Observations must be a non-empty mapping of variable arrays")
        if not all(isinstance(values, list) for values in observations.values()):
            raise ValueError("Each JSON observation variable must be an array")
        return observations, offsets

    def _validate(self) -> None:
        count = self._metadata["observation_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("metadata observation_count must be a positive integer")
        if self.start_time > self.end_time:
            raise ValueError("metadata start_time must not be later than end_time")
        lengths = {name: len(values) for name, values in self._observations.items()}
        wrong = {name: length for name, length in lengths.items() if length != count}
        if wrong:
            raise ValueError(f"Observation length does not match observation_count {count}: {wrong}")
        regular = self._metadata["timestamps_regular"]
        if not isinstance(regular, bool):
            raise ValueError("metadata timestamps_regular must be boolean")
        interval = self._metadata["nominal_sample_interval_seconds"]
        if regular:
            if self._time_offsets is not None:
                raise ValueError("Regular timestamps must not include time_offsets")
            if count > 1 and (not isinstance(interval, (int, float)) or interval <= 0):
                raise ValueError("Regular multi-observation data needs a positive sample interval")
            if count > 1 and self.time_at(count - 1) != self.end_time:
                raise ValueError("Regular timing metadata does not reconstruct end_time")
        else:
            if self._time_offsets is None or len(self._time_offsets) != count:
                raise ValueError("Irregular timestamps require one time_offsets value per observation")
            if any(not isinstance(offset, (int, float)) or not math.isfinite(offset) for offset in self._time_offsets):
                raise ValueError("time_offsets must contain finite numeric seconds")
            if list(self._time_offsets) != sorted(self._time_offsets):
                raise ValueError("time_offsets must be sorted chronologically")
            if self.time_at(count - 1) != self.end_time:
                raise ValueError("Irregular timing metadata does not reconstruct end_time")

    # Common metadata is intentionally exposed as attributes, not raw JSON.
    station_id = property(lambda self: self._metadata["station_id"])
    station_name = property(lambda self: self._metadata["station_name"])
    latitude = property(lambda self: self._metadata["latitude"])
    longitude = property(lambda self: self._metadata["longitude"])
    elevation = property(lambda self: self._metadata["elevation"])
    report_type = property(lambda self: self._metadata["report_type"])
    source = property(lambda self: self._metadata["source"])
    start_time = property(lambda self: self._metadata["start_time"])
    end_time = property(lambda self: self._metadata["end_time"])
    sample_interval = property(lambda self: self._metadata["nominal_sample_interval_seconds"])
    observation_count = property(lambda self: self._metadata["observation_count"])
    timestamps_regular = property(lambda self: self._metadata["timestamps_regular"])

    def variables(self) -> tuple[str, ...]:
        return tuple(self._observations)

    def get_variable(self, name: str) -> Sequence[Any]:
        try:
            return self._observations[name]
        except KeyError as error:
            raise KeyError(f"Unknown variable {name!r}. Available: {', '.join(self.variables())}") from error

    def get_all(self) -> Mapping[str, Sequence[Any]]:
        """Return a read-only mapping to loaded arrays without copying them."""
        return MappingProxyType(self._observations)

    def time_at(self, index: int) -> datetime:
        if not isinstance(index, int):
            raise TypeError("Observation index must be an integer")
        if index < 0:
            index += self.observation_count
        if not 0 <= index < self.observation_count:
            raise IndexError("Observation index out of range")
        if self._metadata["timestamps_regular"]:
            interval = self.sample_interval or 0
            return self.start_time + timedelta(seconds=index * interval)
        return self.start_time + timedelta(seconds=float(self._time_offsets[index]))

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self.observation_count
        if not 0 <= index < self.observation_count:
            raise IndexError("Observation index out of range")
        return {"time": self.time_at(index), **{name: values[index] for name, values in self._observations.items()}}

    def _datetime(self, value: str | datetime, query_date: str | Date | None) -> datetime:
        if isinstance(value, datetime):
            if query_date is not None:
                raise ValueError("Do not supply date= when using a full datetime")
            return value
        if not isinstance(value, str):
            raise TypeError("Time must be a datetime or a 'HH:MM' string")
        match = _TIME_ONLY.fullmatch(value)
        if not match:
            try:
                return datetime.fromisoformat(value)
            except ValueError as error:
                raise ValueError("Time must be datetime, ISO datetime, or 24-hour 'HH:MM'") from error
        if query_date is None:
            if self.start_time.date() != self.end_time.date():
                raise ValueError("Military time is ambiguous for multi-day data; supply date='YYYY-MM-DD'")
            day = self.start_time.date()
        elif isinstance(query_date, Date):
            day = query_date
        else:
            try:
                day = Date.fromisoformat(query_date)
            except (TypeError, ValueError) as error:
                raise ValueError("date must be an ISO date such as '2026-01-01'") from error
        return datetime.combine(day, time(int(match[1]), int(match[2])), tzinfo=self.start_time.tzinfo)

    def index_at_time(self, value: str | datetime, date: str | Date | None = None, *, nearest: bool = False) -> int:
        target = self._datetime(value, date)
        for index in range(self.observation_count):
            if self.time_at(index) == target:
                return index
        if nearest:
            return min(range(self.observation_count), key=lambda index: abs(self.time_at(index) - target))
        raise KeyError(f"No observation exists at {target.isoformat()}; use nearest=True to opt in to nearest match")

    def at_time(self, value: str | datetime, date: str | Date | None = None, *, nearest: bool = False) -> dict[str, Any]:
        return self[self.index_at_time(value, date, nearest=nearest)]

    def range(self, start: str | datetime, end: str | datetime, date: str | Date | None = None,
              *, variables: Optional[Sequence[str]] = None) -> dict[str, list[Any]]:
        start_time = self._datetime(start, date)
        end_time = self._datetime(end, date)
        if start_time > end_time:
            raise ValueError("Range start must not be after range end")
        selected = list(variables) if variables is not None else list(self.variables())
        for name in selected:
            self.get_variable(name)
        indexes = [i for i in range(self.observation_count) if start_time <= self.time_at(i) <= end_time]
        return {"time": [self.time_at(i) for i in indexes],
                **{name: [self._observations[name][i] for i in indexes] for name in selected}}

    def day(self, value: str | Date | None = None, *, variables: Optional[Sequence[str]] = None) -> dict[str, list[Any]]:
        if value is None:
            if self.start_time.date() != self.end_time.date():
                raise ValueError("day() needs a date for multi-day data")
            day = self.start_time.date()
        elif isinstance(value, Date):
            day = value
        else:
            try:
                day = Date.fromisoformat(value)
            except ValueError as error:
                raise ValueError("day must be an ISO date such as '2026-01-01'") from error
        start = datetime.combine(day, time.min, tzinfo=self.start_time.tzinfo)
        return self.range(start, start + timedelta(days=1) - timedelta(microseconds=1), variables=variables)
