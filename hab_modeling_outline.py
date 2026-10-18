"""
HAB Modeling & Simulation
Main Simulation Architecture

Modules:
- Data Intake
- Ascent
- Descent
- Drift
- Model Calibration
- Visualization
"""

# IMPORTS
# -----------------------------

import math
import numpy as np
import pandas as pd


# CONSTANTS
# -----------------------------

GRAVITY = 9.81          # m/s^2
TIME_STEP = 1.0         # seconds


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

# -----------------------------
