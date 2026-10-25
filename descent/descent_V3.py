import math
import pandas as pd


# CONSTANTS
# -----------------------------

GRAVITY = 9.81            # m/s^2
TIME_STEP = 5.0           # s  (1 s gives the same track to <50 m, 5x slower)
METERS_PER_DEG_LAT = 111320.0
R_AIR = 287.053           # J/(kg K)

INCH_TO_M = 0.0254

# Parachute (real size from team lead)
PARACHUTE_DIAMETER_IN = 30.0
PARACHUTE_DIAMETER = PARACHUTE_DIAMETER_IN * INCH_TO_M      # 0.762 m
PARACHUTE_CD = 1.5        # typical value, not measured

# Payload box (real size from team lead): 10 x 10 x 6.5 in
BOX_LENGTH = 10.0 * INCH_TO_M
BOX_WIDTH = 10.0 * INCH_TO_M
BOX_CD = 1.05

# Calibrated on the 2025-09-27 flight (validation/calibrate.py, final_compare.py).
# The flight lasted ~3.5 h, so the ascent was much slower than the textbook 5 m/s.
# Free fits ranged 3.1-4.4 m/s and pushed burst to the upper bound; 3.5 m/s and
# 33 km (near the max for a 1200 g balloon) are the physically reasonable choice.
# Descent drag needed no change: fits gave x0.7-1.1, observed descent ~40 min.
ASCENT_RATE = 3.5          # m/s, constant ascent rate (textbook 5 m/s was too fast)
BURST_ALTITUDE = 33000.0   # m
DESCENT_DRAG_SCALE = 1.0   # multiplies Cd*A; >1 = falls slower than nominal


# ATMOSPHERE (US Standard Atmosphere 1976, 0-51 km)
# -----------------------------

_LAYERS = [  # base altitude m, base temperature K, lapse rate K/m, base pressure Pa
    (0.0,     288.15, -0.0065, 101325.0),
    (11000.0, 216.65,  0.0,     22632.06),
    (20000.0, 216.65,  0.001,    5474.889),
    (32000.0, 228.65,  0.0028,    868.0187),
    (47000.0, 270.65,  0.0,       110.9063),
]


def air_density(altitude):
    """Air density (kg/m^3) from the US Standard Atmosphere 1976."""
    h = max(0.0, min(altitude, 51000.0))
    for base_h, T0, L, P0 in reversed(_LAYERS):
        if h >= base_h:
            break
    dh = h - base_h
    if L == 0.0:
        T = T0
        P = P0 * math.exp(-GRAVITY * dh / (R_AIR * T0))
    else:
        T = T0 + L * dh
        P = P0 * (T / T0) ** (-GRAVITY / (R_AIR * L))
    return P / (R_AIR * T)


def air_density_exponential(altitude):
    """V2's simple model, kept for comparison."""
    return 1.225 * math.exp(-altitude / 8500.0)


# BALLOON OBJECT (same as V2)
# -----------------------------

class Balloon:
    """Stores physical properties and current state of the balloon system."""

    def __init__(self, payload_mass, balloon_mass, helium_mass,
                 launch_lat, launch_lon, launch_altitude, burst_altitude):
        self.payload_mass = payload_mass
        self.balloon_mass = balloon_mass
        self.helium_mass = helium_mass

        self.latitude = launch_lat
        self.longitude = launch_lon
        self.altitude = launch_altitude

        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.burst_altitude = burst_altitude
        self.phase = "ascent"
        self.time = 0.0            # seconds since launch
        self.history = []

    def save_state(self):
        self.history.append({
            "time": self.time, "latitude": self.latitude, "longitude": self.longitude,
            "x": self.x, "y": self.y, "altitude": self.altitude,
            "vx": self.vx, "vy": self.vy, "vz": self.vz, "phase": self.phase,
        })


# HELPERS
# -----------------------------

def placeholder_wind(lat, lon, altitude, t_utc):
    """V2's made-up wind (ESE, stronger higher up). Only for comparison."""
    speed = 5.0 + 15.0 * min(altitude / 12000.0, 1.0)
    return speed * 0.9, -speed * 0.3


def parachute_area():
    return math.pi * (PARACHUTE_DIAMETER / 2) ** 2


def box_area():
    return BOX_LENGTH * BOX_WIDTH


def total_drag_area(drag_scale=DESCENT_DRAG_SCALE):
    """Cd*A of parachute + box, m^2 (times the calibration factor)."""
    return drag_scale * (PARACHUTE_CD * parachute_area() + BOX_CD * box_area())


def terminal_velocity(balloon, drag_scale=DESCENT_DRAG_SCALE, density=air_density):
    rho = density(balloon.altitude)
    return math.sqrt(2 * balloon.payload_mass * GRAVITY / (rho * total_drag_area(drag_scale)))


def move_horizontally(balloon, u, v, dt):
    balloon.vx, balloon.vy = u, v
    balloon.x += u * dt
    balloon.y += v * dt
    balloon.latitude += v * dt / METERS_PER_DEG_LAT                     # north -> latitude
    balloon.longitude += u * dt / (METERS_PER_DEG_LAT * math.cos(math.radians(balloon.latitude)))  # east -> longitude


# FLIGHT
# -----------------------------

def run_flight(balloon, wind, launch_time_utc, ascent_rate=ASCENT_RATE,
               drag_scale=DESCENT_DRAG_SCALE, ground_altitude=0.0, dt=TIME_STEP,
               density=air_density, start_phase="ascent"):
    """
    Simulate ascent -> burst -> descent until the payload reaches the ground.

    wind: function (lat, lon, altitude_m, t_utc_hours) -> (u_east, v_north) m/s
    launch_time_utc: hours, e.g. 14.5 for 14:30 UTC (10:30 EDT)
    start_phase: "ascent" (from launch) or "descent" (start at burst, like V2)
    """
    balloon.phase = start_phase
    balloon.save_state()

    while True:
        t_utc = launch_time_utc + balloon.time / 3600.0
        u, v = wind(balloon.latitude, balloon.longitude, balloon.altitude, t_utc)

        if balloon.phase == "ascent":
            balloon.vz = ascent_rate
        else:
            balloon.vz = -terminal_velocity(balloon, drag_scale, density)

        move_horizontally(balloon, u, v, dt)
        balloon.altitude += balloon.vz * dt
        balloon.time += dt

        if balloon.phase == "ascent" and balloon.altitude >= balloon.burst_altitude:
            balloon.altitude = balloon.burst_altitude
            balloon.phase = "descent"
        elif balloon.phase == "descent" and balloon.altitude <= ground_altitude:
            balloon.altitude = ground_altitude
            balloon.phase = "landed"
            balloon.save_state()
            break
        balloon.save_state()

    return pd.DataFrame(balloon.history)


def plot_flight(df, filename="flight_V3.png", observed=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(df["time"] / 60, df["altitude"] / 1000)
    ax[0].set_xlabel("Time since launch (min)")
    ax[0].set_ylabel("Altitude (km)")
    ax[0].set_title("Altitude vs. time")
    ax[0].grid(alpha=0.3)

    ax[1].plot(df["longitude"], df["latitude"], label="Simulated")
    if observed is not None:
        ax[1].plot(observed["longitude"], observed["latitude"], "o", ms=4, label="Observed (10 min)")
    ax[1].plot(df["longitude"].iloc[0], df["latitude"].iloc[0], "k^", label="Launch")
    ax[1].plot(df["longitude"].iloc[-1], df["latitude"].iloc[-1], "rX", ms=10, label="Landing")
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Latitude")
    ax[1].set_title("Ground track")
    ax[1].set_aspect(1 / math.cos(math.radians(40.5)))
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print("Saved plot:", filename)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation"))
    from wind_field import WindField

    LAUNCH = (40.0418, -75.8602)     # 2025-09-27 launch site (from tracker screenshot)
    LAUNCH_TIME_UTC = 14.78          # 14:47 UTC = 10:47 EDT, fitted (not recorded)

    b = Balloon(payload_mass=1.5, balloon_mass=1.2, helium_mass=0.4,
                launch_lat=LAUNCH[0], launch_lon=LAUNCH[1], launch_altitude=150,
                burst_altitude=BURST_ALTITUDE)
    df = run_flight(b, WindField(), LAUNCH_TIME_UTC, ground_altitude=150)

    burst = df[df.phase == "descent"].iloc[0]
    print(f"Burst   : {burst.time/60:.0f} min after launch, {burst.latitude:.4f}, {burst.longitude:.4f}")
    print(f"Landing : {df.time.iloc[-1]/60:.0f} min after launch, "
          f"{df.latitude.iloc[-1]:.4f}, {df.longitude.iloc[-1]:.4f}")
    df.to_csv("flight_V3.csv", index=False)
    plot_flight(df)
