"""Read an atmospheric data file and return interpolated wind as a DataFrame."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wind_interpolation import get_wind_dataframe


DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "data" / "era5_pressure_levels.nc"


def read_wind_dataframe(
    latitude: float,
    longitude: float,
    altitude_m: float,
    data_file: str | Path = DEFAULT_DATA_FILE,
    valid_time: str | None = None,
) -> pd.DataFrame:
    """Return valid_time, pressure_level, latitude, longitude, u, and v."""
    return get_wind_dataframe(
        latitude=latitude,
        longitude=longitude,
        altitude_m=altitude_m,
        grid_data=data_file,
        valid_time=valid_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read ERA5 wind data into a Pandas DataFrame")
    parser.add_argument("--file", type=Path, default=DEFAULT_DATA_FILE, help="ERA5 .nc or .csv data file")
    parser.add_argument("--latitude", type=float, default=40.5)
    parser.add_argument("--longitude", type=float, default=-75.5)
    parser.add_argument("--altitude-m", type=float, default=10000)
    parser.add_argument("--valid-time", help="Required when a CSV contains multiple timestamps")
    args = parser.parse_args()
    try:
        frame = read_wind_dataframe(
            args.latitude, args.longitude, args.altitude_m, args.file, args.valid_time
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
