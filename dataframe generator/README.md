# PAC Wind Data Reader

- **Input:** ERA5 `.nc` or `.csv` weather file.
- **Install:** `pip install -e .`
- **Import and call:**

```python
from read_wind import read_wind_dataframe

df = read_wind_dataframe(
    latitude=40.5,
    longitude=-75.5,
    altitude_m=10000,
    data_file="data/era5_pressure_levels.nc"
)
```

- **For repeated calls (load file once):**

```python
from wind_interpolation import NetCDFWindField, get_wind_dataframe

weather = NetCDFWindField("data/era5_pressure_levels.nc")
df = get_wind_dataframe(40.5, -75.5, 10000, weather)
```

- **Output:** `pandas.DataFrame` with six columns:

`valid_time`, `pressure_level`, `latitude`, `longitude`, `u`, `v`

- **Units:** Altitude in meters, pressure in hPa, `u` (eastward) and `v` (northward) in m/s.
- **Note:** Requested coordinates must be within the weather file's geographic bounds.
