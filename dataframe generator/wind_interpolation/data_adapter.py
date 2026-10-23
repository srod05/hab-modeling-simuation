"""Adapters for ERA5 NetCDF/CSV data and PAC's six-column wind DataFrame."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from .interpolation import interpolate_wind, select_pressure_level

STANDARD_GRAVITY = 9.80665  # m/s^2; geopotential / g is geopotential height
CORNERS = ("SW", "SE", "NW", "NE")
SCHEMA = ("valid_time", "pressure_level", "latitude", "longitude", "u", "v")


def _valid(value: float) -> bool:
    return math.isfinite(value) and abs(value) < 1e20


class NetCDFWindField:
    """Adapter from a regular NetCDF grid to the existing four-corner API."""

    def __init__(self, path: str | Path, time_index: int = 0):
        self.path = Path(path)
        with h5py.File(self.path, "r") as file:
            required = ("latitude", "longitude", "pressure_level", "u", "v")
            missing = [name for name in required if name not in file]
            if missing:
                raise ValueError(f"NetCDF file lacks required variables: {', '.join(missing)}")
            latitudes = np.asarray(file["latitude"][:], dtype=float)
            longitudes = np.asarray(file["longitude"][:], dtype=float)
            pressures = np.asarray(file["pressure_level"][:], dtype=float)
            u_dataset, v_dataset = file["u"], file["v"]
            expected_spatial_shape = (len(pressures), len(latitudes), len(longitudes))
            if u_dataset.ndim != 4 or v_dataset.shape != u_dataset.shape:
                raise ValueError("u and v must have matching (time, pressure, latitude, longitude) dimensions")
            if u_dataset.shape[1:] != expected_spatial_shape:
                raise ValueError("u/v dimensions do not match the pressure, latitude, and longitude coordinates")
            if isinstance(time_index, bool) or not isinstance(time_index, int) or not 0 <= time_index < u_dataset.shape[0]:
                raise ValueError(f"time_index must be between 0 and {u_dataset.shape[0] - 1}")
            u = np.asarray(u_dataset[time_index], dtype=float)
            v = np.asarray(v_dataset[time_index], dtype=float)
            geopotential = None
            if "z" in file:
                if file["z"].shape != u_dataset.shape:
                    raise ValueError("z dimensions must match u and v")
                units = file["z"].attrs.get("units", b"")
                if isinstance(units, bytes):
                    units = units.decode("utf-8")
                if units not in ("m**2 s**-2", "m2 s-2", "m^2 s^-2"):
                    raise ValueError(f"Unsupported geopotential units for z: {units!r}")
                geopotential = np.asarray(file["z"][time_index], dtype=float)
            if "valid_time" in file:
                timestamp = int(file["valid_time"][time_index])
                self.valid_time = datetime.fromtimestamp(timestamp, timezone.utc)
            else:
                self.valid_time = None

        if len(latitudes) < 2 or len(longitudes) < 2 or len(pressures) < 1:
            raise ValueError("NetCDF grid needs at least two latitudes and longitudes, and one pressure level")
        if not all(np.isfinite(array).all() for array in (latitudes, longitudes, pressures)):
            raise ValueError("Coordinate and pressure arrays must be finite")
        if (np.diff(np.sort(latitudes)) <= 0).any() or (np.diff(np.sort(longitudes)) <= 0).any():
            raise ValueError("Latitude and longitude coordinates must be unique")
        if (pressures <= 0).any() or (np.diff(np.sort(pressures)) <= 0).any():
            raise ValueError("Pressure levels must be positive and unique")
        if latitudes.min() < -90 or latitudes.max() > 90 or longitudes.min() < -180 or longitudes.max() > 180:
            raise ValueError("This adapter requires latitude in [-90, 90] and longitude in [-180, 180]")

        lat_order = np.argsort(latitudes)
        lon_order = np.argsort(longitudes)
        pressure_order = np.argsort(pressures)[::-1]
        self.latitudes = latitudes[lat_order]
        self.longitudes = longitudes[lon_order]
        self.pressures = pressures[pressure_order]
        self.u = u[pressure_order][:, lat_order][:, :, lon_order]
        self.v = v[pressure_order][:, lat_order][:, :, lon_order]
        self.heights = (
            None if geopotential is None else
            (geopotential[pressure_order][:, lat_order][:, :, lon_order] / STANDARD_GRAVITY)
        )
        self._cells: dict[tuple[int, int, tuple[float, ...] | None], dict] = {}

    def _indices(self, latitude: float, longitude: float) -> tuple[int, int]:
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError("Requested latitude and longitude must be finite")
        south, north = self.latitudes[0], self.latitudes[-1]
        west, east = self.longitudes[0], self.longitudes[-1]
        if not south <= latitude <= north or not west <= longitude <= east:
            raise ValueError(
                f"Coordinate ({latitude:g}, {longitude:g}) is outside NetCDF bounds: "
                f"latitude [{south:g}, {north:g}], longitude [{west:g}, {east:g}]"
            )
        row = min(int(np.searchsorted(self.latitudes, latitude, side="right") - 1), len(self.latitudes) - 2)
        column = min(int(np.searchsorted(self.longitudes, longitude, side="right") - 1), len(self.longitudes) - 2)
        return row, column

    def _cell(self, row: int, column: int, allowed: tuple[float, ...] | None = None) -> dict:
        cache_key = (row, column, allowed)
        if cache_key in self._cells:
            return self._cells[cache_key]
        positions = {
            "SW": (row, column), "SE": (row, column + 1),
            "NW": (row + 1, column), "NE": (row + 1, column + 1),
        }
        points = {
            corner: {
                "latitude": float(self.latitudes[latitude_index]),
                "longitude": float(self.longitudes[longitude_index]),
                "levels": {},
            }
            for corner, (latitude_index, longitude_index) in positions.items()
        }
        for level_index, pressure in enumerate(self.pressures):
            pressure = float(pressure)
            if allowed is not None and pressure not in allowed:
                continue
            samples = [
                (float(self.u[level_index, i, j]), float(self.v[level_index, i, j]))
                for i, j in positions.values()
            ]
            if not all(_valid(u) and _valid(v) for u, v in samples):
                continue
            heights = None if self.heights is None else [
                float(self.heights[level_index, i, j]) for i, j in positions.values()
            ]
            if heights is not None and not all(_valid(height) for height in heights):
                continue
            for index, corner in enumerate(CORNERS):
                u, v = samples[index]
                record = {"u_ms": u, "v_ms": v}
                if heights is not None:
                    record["altitude_m"] = heights[index]
                points[corner]["levels"][f"{pressure:g}"] = record
        self._cells[cache_key] = points
        return points

    def _cell_at(self, latitude: float, longitude: float, allowed: tuple[float, ...] | None = None) -> dict:
        row, column = self._indices(latitude, longitude)
        points = self._cell(row, column, allowed)
        if not points["SW"]["levels"]:
            raise ValueError(f"No complete wind levels around ({latitude:g}, {longitude:g})")
        return points

    def available_pressures(self, latitude: float, longitude: float) -> set[float]:
        points = self._cell_at(latitude, longitude)
        return {float(key) for key in points["SW"]["levels"]}

    def estimate(
        self, latitude: float, longitude: float, altitude_m: float,
        allowed_pressures: tuple[float, ...] | None = None,
    ) -> dict:
        """Interpolate from the four NetCDF nodes surrounding this coordinate."""
        return interpolate_wind(
            self._cell_at(latitude, longitude, allowed_pressures),
            latitude, longitude, altitude_m,
        )

    def level_altitude(self, pressure: float, latitude: float, longitude: float) -> tuple[float, bool]:
        points = self._cell_at(latitude, longitude, (pressure,))
        result = select_pressure_level(points, latitude, longitude, 0)
        return result["matched_altitude_m"], result["altitude_source"] != "grid_metadata"


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _samples(latitude: Any, longitude: Any, altitude_m: Any) -> list[tuple[float, float, float]]:
    values = []
    for name, value in (("latitude", latitude), ("longitude", longitude), ("altitude_m", altitude_m)):
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if np.ndim(value) != 1:
                raise ValueError(f"{name} must be a scalar or one-dimensional sequence")
            items = list(value)
        else:
            items = [value]
        if not items:
            raise ValueError(f"{name} cannot be empty")
        values.append([_number(item, name) for item in items])
    count = max(len(items) for items in values)
    if any(len(items) not in (1, count) for items in values):
        raise ValueError("latitude, longitude, and altitude_m sequences must have equal lengths or be scalars")
    expanded = [items * count if len(items) == 1 else items for items in values]
    return list(zip(*expanded))


def _timestamp(value: Any) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid valid_time: {value!r}") from exc
    if pd.isna(result):
        raise ValueError("valid_time must be a real timestamp")
    return result


def _prepare_era5_frame(frame: pd.DataFrame, valid_time: Any) -> tuple[pd.DataFrame, pd.Timestamp]:
    if "valid_time" not in frame.columns and frame.index.name == "valid_time":
        frame = frame.reset_index()
    required = set(SCHEMA)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ERA5 DataFrame lacks required columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], errors="raise")
    for name in ("pressure_level", "latitude", "longitude", "u", "v"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if valid_time is None:
        times = frame["valid_time"].dropna().unique()
        if len(times) != 1:
            raise ValueError("valid_time is required when the ERA5 DataFrame has zero or multiple timestamps")
        selected_time = _timestamp(times[0])
    else:
        selected_time = _timestamp(valid_time)
    frame = frame.loc[frame["valid_time"] == selected_time]
    if frame.empty:
        raise ValueError(f"No ERA5 rows match valid_time {selected_time}")
    return frame, selected_time


def _bracket(values: np.ndarray, requested: float, name: str) -> tuple[float, float]:
    coordinates = np.sort(np.unique(values.astype(float)))
    if len(coordinates) < 2:
        raise ValueError(f"ERA5 data needs at least two distinct {name} values")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"ERA5 {name} values must be finite")
    if not coordinates[0] <= requested <= coordinates[-1]:
        raise ValueError(f"Requested {name} {requested:g} is outside ERA5 bounds [{coordinates[0]:g}, {coordinates[-1]:g}]")
    index = min(int(np.searchsorted(coordinates, requested, side="right") - 1), len(coordinates) - 2)
    return float(coordinates[index]), float(coordinates[index + 1])


def _row_height(row: pd.Series) -> float | None:
    for field in ("altitude_m", "geopotential_height_m", "z"):
        if field in row and pd.notna(row[field]):
            value = _number(row[field], field)
            return value / STANDARD_GRAVITY if field == "z" else value
    return None


def _grid_from_era5_frame(frame: pd.DataFrame, latitude: float, longitude: float) -> dict:
    south, north = _bracket(frame["latitude"].to_numpy(), latitude, "latitude")
    west, east = _bracket(frame["longitude"].to_numpy(), longitude, "longitude")
    coordinates = {"SW": (south, west), "SE": (south, east), "NW": (north, west), "NE": (north, east)}
    points = {
        corner: {"latitude": lat, "longitude": lon, "levels": {}}
        for corner, (lat, lon) in coordinates.items()
    }
    for pressure in np.sort(frame["pressure_level"].unique())[::-1]:
        if not math.isfinite(pressure) or pressure <= 0:
            raise ValueError("ERA5 pressure_level must be positive finite hPa values")
        corner_rows = {}
        for corner, (lat, lon) in coordinates.items():
            matches = frame.loc[
                (frame["pressure_level"] == pressure)
                & (frame["latitude"] == lat)
                & (frame["longitude"] == lon)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one ERA5 row for {pressure:g} hPa at {corner} ({lat:g}, {lon:g}); found {len(matches)}"
                )
            corner_rows[corner] = matches.iloc[0]
        heights = {corner: _row_height(row) for corner, row in corner_rows.items()}
        if any(height is not None for height in heights.values()) and not all(
            height is not None for height in heights.values()
        ):
            raise ValueError(f"Partial altitude metadata at {pressure:g} hPa")
        for corner, row in corner_rows.items():
            record = {"u_ms": _number(row["u"], "u"), "v_ms": _number(row["v"], "v")}
            if heights[corner] is not None:
                record["altitude_m"] = heights[corner]
            points[corner]["levels"][f"{pressure:g}"] = record
    return points


def get_wind_dataframe(
    latitude: Any,
    longitude: Any,
    altitude_m: Any,
    grid_data: Mapping[str, Any] | pd.DataFrame | NetCDFWindField | str | Path,
    valid_time: Any = None,
    valid_time_as_index: bool = False,
) -> pd.DataFrame:
    """Return PAC's six ERA5 fields for one or more interpolated wind samples.

    ``grid_data`` may be four-corner data, an ERA5 DataFrame/CSV, or a loaded
    ``NetCDFWindField``/NetCDF path. Pass sequences for batch locations; scalars
    are broadcast. A four-corner mapping requires an explicit ``valid_time``.
    """
    requests = _samples(latitude, longitude, altitude_m)
    if isinstance(grid_data, (str, Path)):
        path = Path(grid_data)
        if path.suffix.lower() == ".nc":
            grid_data = NetCDFWindField(path)
        elif path.suffix.lower() == ".csv":
            grid_data = pd.read_csv(path)
        else:
            raise ValueError("grid_data path must refer to a .nc or .csv file")

    frame = None
    if isinstance(grid_data, NetCDFWindField):
        source_time = grid_data.valid_time
        if source_time is None and valid_time is None:
            raise ValueError("NetCDF data has no valid_time; pass valid_time explicitly")
        if valid_time is not None and source_time is not None and _timestamp(valid_time) != _timestamp(source_time):
            raise ValueError("valid_time does not match the loaded NetCDF time")
        timestamp = _timestamp(source_time if source_time is not None else valid_time)
    elif isinstance(grid_data, pd.DataFrame):
        frame, timestamp = _prepare_era5_frame(grid_data, valid_time)
    elif isinstance(grid_data, Mapping):
        source_time = valid_time if valid_time is not None else grid_data.get("valid_time")
        if source_time is None:
            raise ValueError("valid_time is required for four-corner grid data")
        timestamp = _timestamp(source_time)
        grid_data = grid_data.get("grid_points", grid_data)
    else:
        raise TypeError("grid_data must be four-corner data, an ERA5 DataFrame, or a NetCDFWindField")

    rows = []
    for lat, lon, alt in requests:
        if frame is not None:
            result = interpolate_wind(_grid_from_era5_frame(frame, lat, lon), lat, lon, alt)
        elif isinstance(grid_data, NetCDFWindField):
            result = grid_data.estimate(lat, lon, alt)
        else:
            result = interpolate_wind(grid_data, lat, lon, alt)
        rows.append({
            "valid_time": timestamp,
            "pressure_level": result["interpolation"]["pressure_level_hpa"],
            "latitude": lat,
            "longitude": lon,
            "u": result["wind"]["u_ms"],
            "v": result["wind"]["v_ms"],
        })
    output = pd.DataFrame(rows, columns=SCHEMA)
    return output.set_index("valid_time") if valid_time_as_index else output
