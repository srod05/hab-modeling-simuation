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

# Atmosphere (simple exponential model)
SEA_LEVEL_DENSITY = 1.225   # kg/m^3
SCALE_HEIGHT = 8500.0       # m

# Descent hardware
INCH_TO_M = 0.0254
PARACHUTE_DIAMETER = 30.0 * INCH_TO_M   
PARACHUTE_CD = 1.5     #drag coefficient                 
BOX_SIDE = 10.0 * INCH_TO_M             # m (10 x 10 in bottom face)
BOX_CD = 1.05                        

# Total drag area Cd*A of parachute + payload box, m^2
DESCENT_DRAG_AREA = (PARACHUTE_CD * math.pi * (PARACHUTE_DIAMETER / 2) ** 2
                     + BOX_CD * BOX_SIDE ** 2)


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


    # DESCENT
    # -----------------------------

    def descent_step(self, dt=TIME_STEP, ground_altitude=0.0):
        """
        Vertical motion after burst: payload falls at terminal velocity,
        where gravity balances parachute + box drag:
            m g = 0.5 * rho * v^2 * CdA   ->   v = sqrt(2 m g / (rho * CdA))
        """
        rho = SEA_LEVEL_DENSITY * math.exp(-self.altitude / SCALE_HEIGHT)
        self.vz = -math.sqrt(2 * self.payload_mass * GRAVITY / (rho * DESCENT_DRAG_AREA))
        self.altitude += self.vz * dt

        if self.altitude <= ground_altitude:
            self.altitude = ground_altitude
            self.phase = "landed"

# -----------------------------
