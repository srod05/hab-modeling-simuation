"""Bilinear interpolation of atmospheric wind on a four-corner grid."""

from __future__ import annotations

import math
import numbers
from typing import Any, Mapping


CORNERS = ("SW", "SE", "NW", "NE")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _prepare_grid(grid_points: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    points = _mapping(grid_points, "grid_points")
    if set(points) != set(CORNERS):
        raise ValueError("grid_points must contain exactly SW, SE, NW, and NE")

    prepared: dict[str, dict[str, Any]] = {}
    for corner in CORNERS:
        point = _mapping(points[corner], f"{corner}")
        lat = _number(point.get("latitude"), f"{corner}.latitude")
        lon = _number(point.get("longitude"), f"{corner}.longitude")
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"{corner} latitude/longitude is outside geographic bounds")
        raw_levels = _mapping(point.get("levels"), f"{corner}.levels")
        if not raw_levels:
            raise ValueError(f"{corner}.levels must contain at least one pressure level")

        levels: dict[float, dict[str, float | None]] = {}
        for raw_pressure, raw_wind in raw_levels.items():
            if isinstance(raw_pressure, bool):
                raise ValueError(f"{corner} has invalid pressure level {raw_pressure!r}")
            try:
                pressure = float(raw_pressure)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{corner} has invalid pressure level {raw_pressure!r}") from exc
            if not math.isfinite(pressure) or pressure <= 0:
                raise ValueError(f"{corner} pressure levels must be positive finite hPa values")
            if pressure in levels:
                raise ValueError(f"{corner} has duplicate pressure level {pressure:g} hPa")
            wind = _mapping(raw_wind, f"{corner}.levels[{pressure:g}]")
            altitude = wind.get("altitude_m")
            levels[pressure] = {
                "u_ms": _number(wind.get("u_ms"), f"{corner}.levels[{pressure:g}].u_ms"),
                "v_ms": _number(wind.get("v_ms"), f"{corner}.levels[{pressure:g}].v_ms"),
                "altitude_m": None if altitude is None else _number(
                    altitude, f"{corner}.levels[{pressure:g}].altitude_m"
                ),
            }
        prepared[corner] = {"latitude": lat, "longitude": lon, "levels": levels}

    sw, se, nw, ne = (prepared[c] for c in CORNERS)
    if not (
        sw["latitude"] == se["latitude"]
        and nw["latitude"] == ne["latitude"]
        and sw["longitude"] == nw["longitude"]
        and se["longitude"] == ne["longitude"]
        and sw["latitude"] < nw["latitude"]
        and sw["longitude"] < se["longitude"]
    ):
        raise ValueError("Grid coordinates must form a nondegenerate, axis-aligned SW/SE/NW/NE rectangle (without crossing the antimeridian)")

    expected_levels = set(sw["levels"])
    for corner in CORNERS[1:]:
        missing = expected_levels - set(prepared[corner]["levels"])
        extra = set(prepared[corner]["levels"]) - expected_levels
        if missing or extra:
            raise ValueError(
                f"{corner} pressure levels differ from SW; missing {sorted(missing)}, extra {sorted(extra)} hPa"
            )
    for pressure in expected_levels:
        available = [prepared[corner]["levels"][pressure]["altitude_m"] is not None for corner in CORNERS]
        if any(available) and not all(available):
            raise ValueError(
                f"Altitude metadata for {pressure:g} hPa must be present at all four corners or none"
            )
    return prepared


def _position(points: Mapping[str, Any], latitude: float, longitude: float) -> tuple[float, float]:
    south = points["SW"]["latitude"]
    north = points["NW"]["latitude"]
    west = points["SW"]["longitude"]
    east = points["SE"]["longitude"]
    if not south <= latitude <= north or not west <= longitude <= east:
        raise ValueError(
            f"Requested coordinate ({latitude}, {longitude}) is outside the grid: "
            f"latitude [{south}, {north}], longitude [{west}, {east}]"
        )
    return (longitude - west) / (east - west), (latitude - south) / (north - south)


def bilinear_interpolate(sw: float, se: float, nw: float, ne: float, x: float, y: float) -> float:
    """Interpolate four scalar corner values at normalized coordinates x, y."""
    values = [
        _number(value, name)
        for value, name in zip((sw, se, nw, ne, x, y), ("sw", "se", "nw", "ne", "x", "y"))
    ]
    sw, se, nw, ne, x, y = values
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        raise ValueError("x and y must both be between 0 and 1")
    return (1 - x) * (1 - y) * sw + x * (1 - y) * se + (1 - x) * y * nw + x * y * ne


def _standard_atmosphere_altitude(pressure_hpa: float) -> float:
    """Approximate pressure altitude in meters using the barometric formula."""
    return 44330.0 * (1.0 - (pressure_hpa / 1013.25) ** 0.1903)


def _select_level(points: Mapping[str, Any], x: float, y: float, altitude_m: float) -> tuple[float, float, str]:
    candidates = []
    for pressure in points["SW"]["levels"]:
        altitudes = [points[corner]["levels"][pressure]["altitude_m"] for corner in CORNERS]
        if altitudes[0] is None:
            matched_altitude = _standard_atmosphere_altitude(pressure)
            source = "standard_atmosphere_approximation"
        else:
            matched_altitude = bilinear_interpolate(*altitudes, x, y)
            source = "grid_metadata"
        candidates.append((abs(matched_altitude - altitude_m), -pressure, pressure, matched_altitude, source))
    _, _, pressure, matched_altitude, source = min(candidates)
    return pressure, matched_altitude, source


def select_pressure_level(
    grid_points: Mapping[str, Any], latitude: float, longitude: float, altitude_m: float
) -> dict[str, float | str]:
    """Find the pressure level closest in altitude at the requested location."""
    points = _prepare_grid(grid_points)
    latitude = _number(latitude, "latitude")
    longitude = _number(longitude, "longitude")
    altitude_m = _number(altitude_m, "altitude_m")
    x, y = _position(points, latitude, longitude)
    pressure, matched, source = _select_level(points, x, y, altitude_m)
    return {
        "pressure_level_hpa": pressure,
        "matched_altitude_m": matched,
        "altitude_difference_m": abs(matched - altitude_m),
        "altitude_source": source,
    }


def wind_to_speed_direction(u_ms: float, v_ms: float) -> tuple[float, float | None]:
    """Return speed and meteorological direction (wind FROM); calm has no direction."""
    u_ms = _number(u_ms, "u_ms")
    v_ms = _number(v_ms, "v_ms")
    speed = math.hypot(u_ms, v_ms)
    if speed == 0:
        return 0.0, None
    return speed, math.degrees(math.atan2(-u_ms, -v_ms)) % 360


def interpolate_wind(
    grid_points: Mapping[str, Any], latitude: float, longitude: float, altitude_m: float
) -> dict[str, Any]:
    """Estimate wind at a location using the nearest pressure level and four corners."""
    points = _prepare_grid(grid_points)
    latitude = _number(latitude, "latitude")
    longitude = _number(longitude, "longitude")
    altitude_m = _number(altitude_m, "altitude_m")
    x, y = _position(points, latitude, longitude)
    pressure, matched, source = _select_level(points, x, y, altitude_m)

    def interpolate_component(name: str) -> float:
        return bilinear_interpolate(
            *(points[corner]["levels"][pressure][name] for corner in CORNERS), x, y
        )

    u_ms = interpolate_component("u_ms")
    v_ms = interpolate_component("v_ms")
    speed_ms, direction_deg = wind_to_speed_direction(u_ms, v_ms)
    return {
        "requested_location": {
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": altitude_m,
        },
        "interpolation": {
            "method": "bilinear",
            "pressure_level_hpa": pressure,
            "matched_altitude_m": matched,
            "altitude_difference_m": abs(matched - altitude_m),
            "altitude_source": source,
        },
        "wind": {
            "u_ms": u_ms,
            "v_ms": v_ms,
            "speed_ms": speed_ms,
            "direction_deg": direction_deg,
        },
    }


