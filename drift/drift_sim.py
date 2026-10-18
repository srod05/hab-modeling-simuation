"""
DRIFT SIMULATION VERSION 3
James Wan
10/18/2026
"""

# IMPORTS
# -----------------------------

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# CONSTANTS
# -----------------------------

GRAVITY = 9.81          # m/s^2
TIME_STEP = 1.0         # seconds

METERS_PER_DEG_LAT = 111320.0


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

# DRIFT SIMULATION
# -----------------------------

def load_df():
    """
    Loads the ERA5 data from a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv("era5_october_data.csv", index_col=0, parse_dates=True)
    return df

def get_wind(balloon, df):
    """
    Gets the wind's x and y components at the balloon's location
    Sets the wind as the balloon's vx and vy

    Args:
        balloon: The Balloon object
        df: The pandas DataFrame containing ERA5 wind data
    """
     
    target_lat, target_lon = balloon.latitude, balloon.longitude

    coords = np.array(df[['latitude', 'longitude']].values)
    target = np.array([[target_lat, target_lon]])

    distances = cdist(target, coords, metric='euclidean')

    nearest_index = distances.argmin(axis=1)

    nearest_row = df.iloc[nearest_index[0]]

    balloon.vx, balloon.vy = nearest_row['u'], nearest_row['v']


def drift_step(balloon, dt=TIME_STEP):
    """
    Runs one step of the drift simulation
    Calculates the change in position and updates 

    Args:
        balloon: The Balloon object
        dt: Change in time (seconds)
    """

    # calculate change in position
    dx = balloon.vx * dt
    dy = balloon.vy * dt
        
    # update cartesian position
    balloon.x += dx
    balloon.y += dy

    # update lat/long
    balloon.latitude += dx / METERS_PER_DEG_LAT
    balloon.longitude += dy / (METERS_PER_DEG_LAT * math.cos(math.radians(balloon.latitude)))

    # update time
    balloon.time += dt

# RUN SIMULATION
# -----------------------------

b = Balloon(
    payload_mass=1.5,
    balloon_mass=1.2,
    helium_mass=0.4,
    launch_lat=40.0,
    launch_lon=-75.0,
    launch_altitude=9000.0,
    burst_altitude=9000.0
)

b.phase = "drift"
df = load_df()

while (b.time <= 3600.0):  # Simulate for 1 hour
    get_wind(b, df)
    drift_step(b)
    b.save_state()

history_df = pd.DataFrame(b.history)
history_df.to_csv("balloon_trajectory.csv", index=False)
print(history_df.head())

# PLOT
# -----------------------------

plt.figure(figsize=(10, 6))
plt.plot(history_df['longitude'], history_df['latitude'], marker='o', markersize=2, linestyle='-', color='blue')
plt.title("Drift trajectory")
plt.grid()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("drift_trajectory.png")
print(plt.show())
