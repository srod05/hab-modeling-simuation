# Wind interpolation core

Import `get_wind_dataframe` for PAC's six-field DataFrame, or `interpolate_wind` for the detailed wind result. This package has no plotting side effects.

Input may be four SW/SE/NW/NE corners, an ERA5 DataFrame/CSV with `valid_time`, `pressure_level`, `latitude`, `longitude`, `u`, `v`, or a `NetCDFWindField` loaded from an ERA5 NetCDF file. Coordinates are degrees, altitude is meters, pressure is hPa, and wind components are m/s.

See the [root README](../README.md) for installation, examples, and drift integration notes.
