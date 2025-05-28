import numpy as np
import pandas as pd
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from sgp4.api import Satrec
import math
from datetime import timedelta

# =============================================================================
# Constants
# =============================================================================
r_Earth = 6378137.0  # Earth radius in meters
mu = 3.986004418e14  # Gravitational parameter for Earth (m^3/s^2)
seconds_per_day = 86400

# =============================================================================
# Orbital elements -> TLE conversion
# =============================================================================
def convert_to_tle(satellite_number, classification, intl_designator, epoch_date, inclination, raan,
                   eccentricity, arg_perigee, mean_anomaly, mean_motion, rev_number):
    year = epoch_date.year
    doy = (epoch_date - pd.Timestamp(f'{year}-01-01')).days
    t_ = f'{epoch_date.hour / 24 + epoch_date.minute / 1440 + epoch_date.second / 86400:.8f}'
    t_ = t_[1:]
    epoch = f"{year % 100:02d}{doy:03d}{t_}"[:14]
    
    line1 = f"1 {satellite_number:05d}{classification} {intl_designator}   {epoch}  .00000000  00000-0  18718-3 0  9997"
    line2 = (f"2 {satellite_number:05d} {inclination:8.4f} {raan:8.4f} {eccentricity * 1e7:07.0f} "
             f"{arg_perigee:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f}{rev_number:5d}")
    
    def compute_checksum(line):
        return sum(int(char) if char.isdigit() else 1 for char in line if char.isdigit() or char == '-') % 10
    
    checksum1 = compute_checksum(line1)
    checksum2 = compute_checksum(line2)
    
    line1 = line1[:-1] + str(checksum1)
    line2 = line2[:-1] + str(checksum2)
    
    return line1, line2

def compute_mean_motion(sma_m):

    # Mean motion in radians per second
    mean_motion_rad_per_sec = np.sqrt(mu / (sma_m ** 3))

    # Convert to revolutions per day
    mean_motion_rev_per_day = (mean_motion_rad_per_sec / (2 * math.pi)) * seconds_per_day

    return mean_motion_rev_per_day


# =============================================================================
# Propagator
# =============================================================================
def prop_orbit(initial_state, CustomAtmosphere, step=600.0, horizon=3 * 86400.0):
    """
    Propagates the orbit of a satellite over a given duration using a SGP4 propagator.
    """
    tmstp = initial_state.get('Timestamp', pd.Timestamp('2013-11-30 00:00:00.00000'))

    rp0 = r_Earth + 400 * 1e3  # perigee radius (m)
    rap0 = r_Earth + 600 * 1e3  # apogee radius (m)

    a0 = initial_state.get('Semi-major Axis (km)', None)
    semi_major_axis = a0 * 1e3 if a0 is not None else (rp0 + rap0) / 2
    e0 = initial_state.get('Eccentricity', (rap0 - rp0) / (rap0 + rp0))
    i0 = initial_state.get('Inclination (deg)', 45)  # deg
    w0 = initial_state.get('Argument of Perigee (deg)', 30)  # deg
    ra0 = initial_state.get('RAAN (deg)', 0)  # deg
    M0 = initial_state.get('True Anomaly (deg)', 0)  # deg

    mean_motion = compute_mean_motion(semi_major_axis)

    satellite_number = 26405
    classification = 'U'
    intl_designator = '00039B'
    rev_number = 99
    
    j2000_epoch = Time(tmstp.strftime("%Y-%m-%dT%H:%M:%S"), scale="utc")
    
    # get the TLE
    s, t = convert_to_tle(satellite_number, classification, intl_designator,
                          tmstp, i0, ra0, e0, w0, M0, mean_motion, rev_number)

    satellite = Satrec.twoline2rv(s, t)

    # Time Vector
    time_steps = np.arange(0, horizon, step)
    epochs = [j2000_epoch + timedelta(seconds=t) for t in time_steps]
    tstamps = [e_.strftime("%Y-%m-%dT%H:%M:%S") for e_ in epochs]
    
    jdarray = np.array([e.jd1 for e in epochs])
    frarray = np.array([e.jd2 for e in epochs])

    # Pre-allocate arrays for results

    # Compute positions for each time step
    error_code, teme_array, _ = satellite.sgp4_array(jdarray, frarray)

    cartrep = coord.CartesianRepresentation(x=teme_array[:,0], y=teme_array[:,1], z=teme_array[:,2], unit=u.km)

    gcrs = coord.GCRS(cartrep, obstime=epochs)
    itrs = gcrs.transform_to(coord.ITRS(obstime=epochs))
    loc = coord.EarthLocation(*itrs.cartesian.get_xyz())
    lat_array = loc.lat.value
    lon_array = loc.lon.value
    alt_array = loc.height.value


    # Return results
    return tstamps, lat_array, lon_array, alt_array


def main():
    # Initial state
    initial_state = {
        'Timestamp': pd.Timestamp('2013-11-30 00:00:00.00000'),
        'Semi-major Axis (km)': 7000.0,
        'Eccentricity': 0.0,
        'Inclination (deg)': 98.0,
        'Argument of Perigee (deg)': 0.0,
        'RAAN (deg)': 0.0,
        'True Anomaly (deg)': 0.0
    }

    # Propagate orbit
    timestamps, lat_array, lon_array, alt_array = prop_orbit(initial_state, None)

if __name__ == '__main__':
    main()

