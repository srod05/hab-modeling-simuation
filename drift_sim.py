# drift simulation

from datetime import datetime, timedelta
import math

class Balloon:

    def __init__(self, lat, long, altitude, time):
        '''
        Initializes the balloon

        Args:
            lat:  Initial latitude (degrees)
            long: Initial longitude (degrees)
            altitude: Initial altitude (m)
            time: Initial time (as datetime)
        '''
        self.lat = lat
        self.long = long
        self.altitude = altitude
        self.time = time
    
    def update_position(self, wind_x, wind_y, new_altitude, new_time):
        '''
        Calculates the new position of the balloon

        Args:
            wind_x: Current wind east vector (m/s)
            wind_y: Current wind north vector (m/s)
            new_altitude: New latitude
            new_time: New time
        
        Returns:
            float tuple
            The new latitude, longitude, and altitude.
            Also updates the balloon internally
        '''
        x_distance = wind_x * (new_time - self.time).total_seconds()
        y_distance = wind_y * (new_time - self.time).total_seconds()
        new_lat  = self.lat + x_distance / 111320 # 111,320 m per degree of latitude
        new_long = self.long + y_distance / (111320 * math.cos(math.radians(self.lat)))

        self.lat = new_lat
        self.long = new_long
        self.altitude = new_altitude

        return self.get_position()

    def get_position(self):
        '''
        Returns the balloon's current position in latitude and longitude degrees and its height (meters).
        '''
        return (self.lat, self.long, self.altitude)


t = datetime.now()
b = Balloon(0, 0, 0, t)

# test
print(f"Balloon's starting position: {b.get_position()}")
print(f"After one minute: {b.update_position(10, 10, 30, t + timedelta(seconds=60))}")

# balloonIsFloating = True

# while (balloonIsFloating):
    # in 60-second increments, check data for nearest wind vector neighbor
    # run update_position() based on this data
    # considerations if we only have surface wind data: power or log law
    # farther away from surface:  use other data (radiosonde or ERA5)
    # do i need to consider pressure levels?