"""
HAB Modeling & Simulation
DESCENT module (simple version)

Basic physics only:
- After burst, the payload falls under a parachute.
- At every moment it falls at its "terminal velocity" — the speed where
  parachute drag exactly balances gravity:
        m * g = 0.5 * rho * v^2 * Cd * A   ->   v = sqrt(2 m g / (rho Cd A))
- Air gets thinner with height (simple exponential model), so the payload
  falls fast up high and slows down near the ground.
- Wind pushes it sideways (simple made-up wind for now; the Drift module
  will replace wind_at() with real data later).

Run:  python3 descent_simple.py
"""


import math
import numpy as np
import pandas as pd



GRAVITY = 9.81          # m/s^2
TIME_STEP = 1.0         # seconds

SEA_LEVEL_DENSITY = 1.225   # kg/m^3, air density at the ground
SCALE_HEIGHT = 8500.0       # m, air density drops by ~63% every 8.5 km
METERS_PER_DEG_LAT = 111320.0

INCH_TO_M = 0.0254         # 1 inch = 0.0254 m

# Parachute (real size from team lead)
PARACHUTE_DIAMETER_IN = 30.0                         # inch
PARACHUTE_DIAMETER = PARACHUTE_DIAMETER_IN * INCH_TO_M   # m  (= 0.762 m)
PARACHUTE_CD = 1.5          # drag coefficient (typical value, not measured)

# Payload box (real size from team lead): 10 x 10 in square base, 6.5 in tall
BOX_LENGTH_IN = 10.0
BOX_WIDTH_IN = 10.0
BOX_HEIGHT_IN = 6.5
BOX_LENGTH = BOX_LENGTH_IN * INCH_TO_M   # m
BOX_WIDTH = BOX_WIDTH_IN * INCH_TO_M     # m
BOX_HEIGHT = BOX_HEIGHT_IN * INCH_TO_M   # m
BOX_CD = 1.05               # drag coefficient of a box falling flat side down


# BALLOON OBJECT
# -----------------------------

class Balloon:
    """
    Stores physical properties and current state of the balloon system.
    """

    def __init__(
        self,
        payload_mass,
        balloon_mass,
        helium_mass,
        launch_lat,
        launch_lon,
        launch_altitude,
        burst_altitude
    ):

        # Physical properties
        self.payload_mass = payload_mass
        self.balloon_mass = balloon_mass
        self.helium_mass = helium_mass

        # Position
        self.latitude = launch_lat
        self.longitude = launch_lon
        self.altitude = launch_altitude

        # Cartesian displacement from launch point
        self.x = 0.0
        self.y = 0.0

        # Velocity
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        # Flight properties
        self.burst_altitude = burst_altitude
        self.phase = "ascent"

        # Simulation time
        self.time = 0.0

        # Complete trajectory storage
        self.history = []


    def save_state(self):
        """
        Save current balloon state to trajectory history.
        """

        self.history.append({
            "time": self.time,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "x": self.x,
            "y": self.y,
            "altitude": self.altitude,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "phase": self.phase
        })


# HELPER FUNCTIONS
# -----------------------------

def air_density(altitude):
    """Simple exponential atmosphere: thinner air higher up."""
    return SEA_LEVEL_DENSITY * math.exp(-altitude / SCALE_HEIGHT)


def wind_at(altitude):
    """
    PLACEHOLDER wind (made up): blows toward the east-southeast,
    stronger higher up. Returns (wind_east, wind_north) in m/s.
    The Drift module will replace this with real wind data.
    """
    speed = 5.0 + 15.0 * min(altitude / 12000.0, 1.0)   # 5 m/s at ground -> 20 m/s at 12 km+
    wind_east = speed * 0.9
    wind_north = -speed * 0.3
    return wind_east, wind_north


def descent_mass(balloon):
    """Mass that falls under the parachute (payload; burst balloon ignored)."""
    return balloon.payload_mass


def parachute_area():
    """Area of the parachute seen from below (circle), m^2."""
    return math.pi * (PARACHUTE_DIAMETER / 2) ** 2


def box_area():
    """Area of the payload box facing the airflow (square bottom), m^2."""
    return BOX_LENGTH * BOX_WIDTH


def total_drag_area():
    """Cd * A of parachute + box added together, m^2."""
    return PARACHUTE_CD * parachute_area() + BOX_CD * box_area()


def terminal_velocity(balloon):
    """Speed where total drag (parachute + box) = gravity, at the current altitude."""
    rho = air_density(balloon.altitude)
    return math.sqrt(2 * descent_mass(balloon) * GRAVITY / (rho * total_drag_area()))


# DESCENT
# -----------------------------

def descent_step(balloon, dt=TIME_STEP):
    """Move the balloon forward by one time step during descent."""

    # 1. vertical: fall at terminal velocity (negative = downward)
    balloon.vz = -terminal_velocity(balloon)

    # 2. horizontal: drift with the wind
    balloon.vx, balloon.vy = wind_at(balloon.altitude)

    # 3. update position
    balloon.altitude += balloon.vz * dt
    balloon.x += balloon.vx * dt
    balloon.y += balloon.vy * dt


    balloon.latitude += balloon.vy * dt / METERS_PER_DEG_LAT
    balloon.longitude += balloon.vx * dt / (METERS_PER_DEG_LAT * math.cos(math.radians(balloon.latitude)))

    balloon.time += dt


def run_descent(balloon, ground_altitude=0.0, dt=TIME_STEP):
    """Simulate from burst until the payload reaches the ground."""
    balloon.phase = "descent"
    balloon.save_state()

    while balloon.altitude > ground_altitude:
        descent_step(balloon, dt)
        if balloon.altitude < ground_altitude:
            balloon.altitude = ground_altitude
            balloon.phase = "landed"
        balloon.save_state()

    return pd.DataFrame(balloon.history)



def plot_descent(df, filename="descent_simple.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(df["time"] / 60, df["altitude"] / 1000)
    ax[0].set_xlabel("Time since burst (min)")
    ax[0].set_ylabel("Altitude (km)")
    ax[0].set_title("Altitude vs. time")
    ax[0].grid(alpha=0.3)

    ax[1].plot(df["longitude"], df["latitude"])
    ax[1].plot(df["longitude"].iloc[0], df["latitude"].iloc[0], "k^", label="Burst")
    ax[1].plot(df["longitude"].iloc[-1], df["latitude"].iloc[-1], "rX", markersize=10, label="Landing")
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Latitude")
    ax[1].set_title("Ground track")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print("Saved plot:", filename)



if __name__ == "__main__":

    # Example numbers (guesses). Start the descent at the burst point.
    b = Balloon(
        payload_mass=1.5,        # kg
        balloon_mass=1.2,        # kg
        helium_mass=0.4,         # kg (not used in descent)
        launch_lat=40.04,        # Lancaster, PA area
        launch_lon=-76.30,
        launch_altitude=30000,   # start the descent at burst altitude
        burst_altitude=30000
    )

    print(f"Parachute    : {PARACHUTE_DIAMETER_IN:.0f} in diameter, area {parachute_area():.3f} m^2")
    print(f"Payload box  : {BOX_LENGTH_IN:g} x {BOX_WIDTH_IN:g} x {BOX_HEIGHT_IN:g} in, "
          f"bottom area {box_area():.3f} m^2")

    df = run_descent(b, ground_altitude=100)

    print(f"Descent time : {df['time'].iloc[-1] / 60:.1f} min")
    print(f"Max speed    : {-df['vz'].min():.1f} m/s")
    print(f"Landing speed: {-df['vz'].iloc[-1]:.1f} m/s")
    print(f"Landing point: {df['latitude'].iloc[-1]:.4f}, {df['longitude'].iloc[-1]:.4f}")

    df.to_csv("descent_simple.csv", index=False)
    print("Saved data: descent_simple.csv")

    plot_descent(df)
