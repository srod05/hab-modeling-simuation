# hab-modeling-simuation

## Plotting the sample trajectory

You need Python 3.10 or newer and the `numpy` and `matplotlib` packages. Keep
`trajectory_model.py` and `sample_points.csv` in the same directory. The CSV is
a synthetic, noisy 3D example with elapsed time and position columns
(`time,x,y,z`); it is for demonstrating the fit, not a set of real flight
measurements.

### 1. Set up Python

On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy matplotlib
```

### 2. Show the graph

With the virtual environment activated, run this from the directory containing
the two files:

```sh
python trajectory_model.py sample_points.csv
```

This opens an interactive 3D graph with the observed points and quadratic fit.
Close the graph window to exit the program. The script also prints the fitted
equations and each coordinate's RMS residual.

To save the graph as a PNG without opening a window:

```sh
python trajectory_model.py sample_points.csv --no-show --output trajectory_fit.png
```

Then open `trajectory_fit.png` in an image viewer. Use `open trajectory_fit.png`
on macOS, `xdg-open trajectory_fit.png` on Linux, or `start trajectory_fit.png`
from Windows Command Prompt.

### Using other trajectory data

The CSV can contain one position per row (`x,y` or `x,y,z`). If it has no time
column, row order is used as the fit parameter. A numeric `time` or `t` header
column is also supported, for example `time,x,y,z`. For real flight data, use
elapsed seconds and local east/north/up coordinates when available. The model
fits a polynomial independently for each coordinate; it does not simulate
wind, lift, or balloon dynamics. Treat the curve as an approximation over the
observed interval, especially when predicting beyond it.

You can also fit and query a trajectory from Python:

```python
from trajectory_model import fit_trajectory

positions = [(east, north, altitude), ...]
elapsed_seconds = [0, 10, 20, ...]
model = fit_trajectory(positions, times=elapsed_seconds)
predicted_position = model.predict(30)
```
