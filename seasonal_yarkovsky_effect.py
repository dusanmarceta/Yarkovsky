import numpy as np
from functions import seasonal_yarkovsky_effect
import matplotlib.pyplot as plt

# =============================================================================
#                       Physical characteristics
# =============================================================================
rho = 1000.  # bulk density (kg/m^3)
k = 10.  # coefficient of thermal conductivity (W/(m*K))
eps = 1.  # emissivity of the surface
cp = 1000.  # Heat capacity at constant pressure (J/kg K)
albedo = 0.  # Bond albedo
a1 = 5.  # semi-axis of the ellipsod in equatorial plane (m)
a2 = 5.  # semi-axis of the ellipsod in equatorial plane (m)
a3 = 5.  # semi-axis of the ellipsod along the rotation axis (m)

# =============================================================================
#                               Rotation
# =============================================================================
rotation_period = 12*3600.  # Rotation period (seconds)
precession_period = np.inf # Period of the precession of the spin axis (+ for direct, - for retrograde precession) (seconds)
axis_lat = np.deg2rad(0.)  #  Latitude of the north pole relative to the orbital plane (rad)
axis_long = np.deg2rad(0.) # Longitude of the north pole measured from the direction of perihelion (rad)
# =============================================================================
#                                Orbit
# =============================================================================
eccentricity = 0.0  # Eccentricity of the orbit
semi_major_axis = 1.0  # Semimajor axis of the orbit (au)


# =============================================================================
#                           Numerical grid
# =============================================================================
facet_size = 0.5  #  Average height of the triangular surface elements (m)
number_of_thermal_wave_depths = 2  #  Total depth of the grid, measured as the number of depths of a diurnal thermal wave
first_layer_depth = 0.3  # Depth of the first layer below the surface, expressed as a fraction of the diurnal thermal wave depth
number_of_layers = 10  #  Number of layers of the numerical grid
time_step_factor = 2  # Critical time step scaling factor


# =============================================================================
#                               Calculation
# =============================================================================
total_drift, drift, vreme = seasonal_yarkovsky_effect(a1, a2, a3,  # shape of the asteroid
                              rho, k, albedo, cp, eps,  # physical characteristics
                              axis_lat, axis_long, rotation_period, precession_period,  # rotation state
                              semi_major_axis, eccentricity,  # orbital elements
                              facet_size, number_of_thermal_wave_depths, first_layer_depth, number_of_layers, time_step_factor) # numerical grid parameters



time = np.linspace(0, vreme, len(drift))/vreme

drift = np.delete(drift, np.arange(27, 40))
time = np.delete(time, np.arange(27, 40))

plt.plot(time, drift, 'k', linewidth = 2)

plt.plot([0, 1], [total_drift, total_drift], '--k', linewidth = 2)

plt.xlabel('fraction of orbital period', fontsize=30)
plt.ylabel('da/dt (m/s)', fontsize=30)
plt.grid()
plt.xlim([0, 1])

# Povećaj font tick labela
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

