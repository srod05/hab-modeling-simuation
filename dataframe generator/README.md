# PAC wind DataFrame reader

This folder contains the wind interpolation engine, a data-file adapter, and the NetCDF file used to create the maps. It has no visualization code or Matplotlib dependency.

## Run

From this folder, with Python 3.10 or newer:

```sh
python3 -m pip install -e .
python3 read_wind.py --latitude 40.5 --longitude -75.5 --altitude-m 10000
python3 -m unittest discover -s tests -v
```

`read_wind.py` defaults to `data/era5_pressure_levels.nc`. Pass `--file path/to/data.nc` or `--file path/to/data.csv` to use another ERA5 file.

## Use from Python

```python
from read_wind import read_wind_dataframe

df = read_wind_dataframe(40.5, -75.5, 10000)
print(type(df))       # pandas.DataFrame
print(df.columns)     # valid_time, pressure_level, latitude, longitude, u, v
```

For another file, pass `data_file="path/to/file.nc"`. The underlying `wind_interpolation.get_wind_dataframe` function also accepts a loaded `NetCDFWindField`, a CSV/DataFrame, or four-corner data, and supports multiple requested locations. Reuse a loaded field for repeated balloon steps so the file is read once.

Latitude and longitude are decimal degrees, altitude is meters, pressure is hPa, and `u`/`v` are eastward/northward m/s. The returned DataFrame has exactly the six columns shown above. Set `valid_time_as_index=True` when calling `get_wind_dataframe` directly to put the timestamp in the index.

Requests outside the file's geographic bounds raise an error. The engine selects the nearest available pressure level; it does not interpolate vertically. A single timestamp is a snapshot, not a forecast.
