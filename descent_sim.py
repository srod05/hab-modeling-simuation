"""
High-Altitude Balloon (HAB) descent simulator — Penn Aerospace Club, modeling team.

"""
from __future__ import annotations
 
import argparse
import math
from dataclasses import dataclass
 
import numpy as np
 
G0 = 9.80665          # m/s^2
R_AIR = 287.053       # J/(kg K)
R_EARTH = 6_371_000.0 # m
 
# -atmosphere
# US Standard Atmosphere 1976, layers up to 51 km: (base geopotential height m, base T K, lapse K/m)
_LAYERS = [
    (0.0,     288.15, -0.0065),
    (11000.0, 216.65,  0.0),
    (20000.0, 216.65,  0.001),
    (32000.0, 228.65,  0.0028),
    (47000.0, 270.65,  0.0),
]
 
 
def _layer_base_pressures():
    p = [101325.0]
    for i in range(1, len(_LAYERS)):
        h0, t0, L = _LAYERS[i - 1]
        h1 = _LAYERS[i][0]
        if L == 0:
            p.append(p[-1] * math.exp(-G0 * (h1 - h0) / (R_AIR * t0)))
        else:
            t1 = t0 + L * (h1 - h0)
            p.append(p[-1] * (t1 / t0) ** (-G0 / (L * R_AIR)))
    return p
 
 
_P_BASE = _layer_base_pressures()
 
 
def atmosphere(h: float) -> tuple[float, float, float]:
    """Return (temperature K, pressure Pa, density kg/m^3) at altitude h (m)."""
    h = max(0.0, min(h, 51000.0))
    i = max(k for k, layer in enumerate(_LAYERS) if layer[0] <= h)
    h0, t0, L = _LAYERS[i]
    p0 = _P_BASE[i]
    if L == 0:
        T = t0
        p = p0 * math.exp(-G0 * (h - h0) / (R_AIR * t0))
    else:
        T = t0 + L * (h - h0)
        p = p0 * (T / t0) ** (-G0 / (L * R_AIR))
    return T, p, p / (R_AIR * T)
 
 
# ----------------------------------------------------------------------------- winds
@dataclass
class WindProfile:
    height_m: np.ndarray   # ascending
    u: np.ndarray          # east component, m/s
    v: np.ndarray          # north component, m/s
 
    def at(self, h: float) -> tuple[float, float]:
        return (float(np.interp(h, self.height_m, self.u)),
                float(np.interp(h, self.height_m, self.v)))
 
    @staticmethod
    def from_dir_speed(height_m, drct_deg, speed_ms):
        d = np.radians(np.asarray(drct_deg, float))
        s = np.asarray(speed_ms, float)
        # meteorological convention: direction the wind blows FROM
        u, v = -s * np.sin(d), -s * np.cos(d)
        order = np.argsort(height_m)
        return WindProfile(np.asarray(height_m, float)[order], u[order], v[order])
 
 
def example_wind_profile() -> WindProfile:
    """ILLUSTRATIVE ONLY: westerly/northwesterly winds peaking near the jet stream (~11 km).
    Replace with a real sounding for any actual prediction."""
    h = np.array([0, 1000, 3000, 6000, 9000, 11000, 13000, 16000, 20000, 25000, 30000, 35000])
    spd = np.array([4, 8, 14, 22, 32, 38, 30, 18, 8, 5, 6, 8])          # m/s
    drct = np.array([300, 295, 290, 285, 280, 280, 280, 275, 270, 90, 90, 90])  # deg FROM
    return WindProfile.from_dir_speed(h, drct, spd)
 
 
def load_sounding_csv(path: str) -> WindProfile:
    """Load a sounding CSV. Needs columns: height_m, drct, speed_kts (Iowa State RAOB format)
    or height_m, drct, speed_ms."""
    import csv
    hs, ds, ss = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                h = float(row["height_m"])
                d = float(row["drct"])
                if "speed_ms" in row and row["speed_ms"] not in ("", "M"):
                    s = float(row["speed_ms"])
                else:
                    s = float(row["speed_kts"]) * 0.514444
            except (KeyError, ValueError, TypeError):
                continue  # skip missing ('M') levels
            hs.append(h); ds.append(d); ss.append(s)
    if len(hs) < 3:
        raise ValueError(f"Too few valid wind levels in {path}")
    return WindProfile.from_dir_speed(hs, ds, ss)
 
 
def fetch_sounding_iem(station: str, when_utc: str, out_csv: str) -> str:
    """Download a sounding from Iowa Environmental Mesonet (run on your own computer).
    station e.g. 'KIAD', when_utc e.g. '2023-11-18T12:00Z'."""
    import urllib.request
    from datetime import datetime, timedelta
    # the server requires end time > start time, so ask for a 1-hour window
    t0 = datetime.strptime(when_utc, "%Y-%m-%dT%H:%MZ")
    ets = (t0 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ")
    url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"
           f"?station={station}&sts={when_utc}&ets={ets}")
    urllib.request.urlretrieve(url, out_csv)
    return out_csv
 
 
# ----------------------------------------------------------------------------- model
@dataclass
class Payload:
    mass_kg: float = 1.5          # payload + parachute + balloon remnants after burst
    chute_diameter_m: float = 0.91  # 36" parachute
    chute_cd: float = 1.5         # typical round/elliptical chute: 0.8–1.5; check manufacturer
 
    @property
    def area(self) -> float:
        return math.pi * (self.chute_diameter_m / 2) ** 2
 
 
def simulate_descent(burst_alt_m: float, burst_lat: float, burst_lon: float,
                     payload: Payload, wind: WindProfile, ground_alt_m: float = 100.0,
                     dt: float = 0.5, wind_scale: float = 1.0):
    """Integrate from burst to ground. Returns dict of numpy arrays."""
    h, v = burst_alt_m, 0.0          # v = downward speed (m/s), starts ~0 at burst
    lat, lon, t = burst_lat, burst_lon, 0.0
    out = {k: [] for k in ("t", "h", "v", "lat", "lon", "rho")}
    m, CdA = payload.mass_kg, payload.chute_cd * payload.area
 
    def accel(h_, v_):
        rho_ = atmosphere(h_)[2]
        return G0 - 0.5 * rho_ * v_ * abs(v_) * CdA / m
 
    dt_max = dt
    while h > ground_alt_m:
        rho = atmosphere(h)[2]
        for k, val in zip(out, (t, h, v, lat, lon, rho)):
            out[k].append(val)
        # Drag relaxes speed toward terminal velocity on a time scale ~ v_term / (2g).
        # In the dense lower atmosphere this is only ~0.25 s, so shrink the step to stay stable.
        v_term = math.sqrt(2 * m * G0 / (rho * CdA))
        dt = min(dt_max, 0.2 * v_term / G0)
        # RK2 (midpoint) for vertical motion — stable enough for dt <= 1 s
        a1 = accel(h, v)
        v_mid = v + 0.5 * dt * a1
        a2 = accel(h - 0.5 * dt * v, v_mid)
        v_new = v + dt * a2
        h -= dt * v_mid
        v = v_new
        # horizontal drift with local wind
        u_w, v_w = wind.at(h)
        lat += math.degrees(v_w * wind_scale * dt / R_EARTH)
        lon += math.degrees(u_w * wind_scale * dt / (R_EARTH * math.cos(math.radians(lat))))
        t += dt
 
    for k, val in zip(out, (t, ground_alt_m, v, lat, lon, atmosphere(ground_alt_m)[2])):
        out[k].append(val)
    return {k: np.array(val) for k, val in out.items()}
 
 
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH / 1000 * np.arcsin(np.sqrt(a))
 
 
def monte_carlo(n, burst_alt_m, lat, lon, payload, wind, ground_alt_m, seed=0):
    """Vary burst altitude, Cd, mass, and wind strength to get a landing scatter."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        p = Payload(mass_kg=payload.mass_kg * rng.normal(1, 0.05),
                    chute_diameter_m=payload.chute_diameter_m,
                    chute_cd=payload.chute_cd * rng.normal(1, 0.10))
        r = simulate_descent(burst_alt_m + rng.normal(0, 1500), lat, lon, p, wind,
                             ground_alt_m, dt=1.0, wind_scale=rng.normal(1, 0.15))
        pts.append((r["lat"][-1], r["lon"][-1], r["t"][-1]))
    return np.array(pts)
 
 
# ----------------------------------------------------------------------------- plotting
def plot_results(res, mc=None, out_png="descent_result.png", title_note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    t_min = res["t"] / 60
 
    ax[0].plot(t_min, res["h"] / 1000, lw=2)
    ax[0].set(xlabel="Time since burst (min)", ylabel="Altitude (km)", title="Altitude vs. time")
    ax[0].grid(alpha=.3)
 
    ax[1].plot(res["v"], res["h"] / 1000, lw=2, color="C1")
    ax[1].set(xlabel="Descent rate (m/s)", ylabel="Altitude (km)", title="Descent rate vs. altitude")
    ax[1].set_xscale("log"); ax[1].grid(alpha=.3, which="both")
 
    ax[2].plot(res["lon"], res["lat"], lw=2, color="C2", label="Descent track")
    ax[2].plot(res["lon"][0], res["lat"][0], "k^", ms=9, label="Burst")
    ax[2].plot(res["lon"][-1], res["lat"][-1], "rX", ms=11, label="Predicted landing")
    if mc is not None:
        ax[2].scatter(mc[:, 1], mc[:, 0], s=6, alpha=.35, color="C3", label=f"Monte Carlo (n={len(mc)})")
    ax[2].set(xlabel="Longitude", ylabel="Latitude", title="Ground track")
    ax[2].set_aspect(1 / math.cos(math.radians(res["lat"][0])))
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
 
    fig.suptitle(f"HAB descent simulation {title_note}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    return out_png
 
 
# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="HAB descent simulator")
    ap.add_argument("--burst-alt", type=float, default=30000, help="burst altitude, m")
    ap.add_argument("--lat", type=float, default=40.00, help="burst latitude")
    ap.add_argument("--lon", type=float, default=-76.20, help="burst longitude")
    ap.add_argument("--ground", type=float, default=100, help="landing ground elevation, m")
    ap.add_argument("--mass", type=float, default=1.5, help="descending mass, kg")
    ap.add_argument("--chute-d", type=float, default=0.91, help="parachute diameter, m")
    ap.add_argument("--cd", type=float, default=1.5, help="parachute drag coefficient")
    ap.add_argument("--sounding", help="CSV sounding file (height_m, drct, speed_kts)")
    ap.add_argument("--monte-carlo", type=int, default=0, help="number of MC runs")
    ap.add_argument("--out", default="descent_result.png")
    ap.add_argument("--csv", default="descent_track.csv", help="save track as CSV")
    a = ap.parse_args()
 
    wind = load_sounding_csv(a.sounding) if a.sounding else example_wind_profile()
    note = f"(winds: {a.sounding})" if a.sounding else "(EXAMPLE winds — not real data)"
    payload = Payload(a.mass, a.chute_d, a.cd)
 
    res = simulate_descent(a.burst_alt, a.lat, a.lon, payload, wind, a.ground)
    dist = haversine_km(res["lat"][0], res["lon"][0], res["lat"][-1], res["lon"][-1])
    print(f"Descent time      : {res['t'][-1]/60:.1f} min")
    print(f"Peak fall speed   : {res['v'].max():.1f} m/s at {res['h'][res['v'].argmax()]/1000:.1f} km")
    print(f"Landing speed     : {res['v'][-1]:.2f} m/s")
    print(f"Landing point     : {res['lat'][-1]:.4f}, {res['lon'][-1]:.4f}  ({dist:.1f} km from burst)")
 
    mc = None
    if a.monte_carlo:
        mc = monte_carlo(a.monte_carlo, a.burst_alt, a.lat, a.lon, payload, wind, a.ground)
        d = haversine_km(res["lat"][-1], res["lon"][-1], mc[:, 0], mc[:, 1])
        print(f"Monte Carlo       : 68% of landings within {np.percentile(d, 68):.1f} km, "
              f"95% within {np.percentile(d, 95):.1f} km of nominal")
 
    np.savetxt(a.csv, np.column_stack([res[k] for k in ("t", "h", "v", "lat", "lon")]),
               delimiter=",", header="t_s,alt_m,descent_rate_ms,lat,lon", comments="")
    print("Saved:", plot_results(res, mc, a.out, note), "and", a.csv)
 
 
if __name__ == "__main__":
    main()
 