import numpy as np
from functions import seasonal_yarkovsky_effect

#import sys
#
#i1 = int(sys.argv[1])
#i2 = int(sys.argv[2])
#i3 = int(sys.argv[3])
#i4 = int(sys.argv[4])
#i5 = int(sys.argv[5])

i1, i2, i3, i4, i5 = 0, 0, 0, 0, 0

izlazni_fajl = 'diurnal_' + str(i1) + str(i2) + str(i3) + str(i4) + str(i5) + '.txt'


facet_size_arr = [1., 0.1]
number_of_thermal_wave_depths_arr = [2, 5]
first_layer_depth_arr = [0.2, 0.02]
number_of_layers_arr = [10, 50]
time_step_factor_arr = [3, 10]

# =============================================================================
#                       Physical characteristics
# =============================================================================
rho = 1000.  # bulk density (kg/m^3)
k = 10.  # coefficient of thermal conductivity (W/(m*K))
eps = 1.  # emissivity of the surface
cp = 1000.  # Heat capacity at constant pressure (J/kg K)
albedo = 0.  # Bond albedo
a1 = 10.  # semi-axis of the ellipsod in equatorial plane (m)
a2 = 10.  # semi-axis of the ellipsod in equatorial plane (m)
a3 = 10.  # semi-axis of the ellipsod along the rotation axis (m)

# =============================================================================
#                               Rotation
# =============================================================================
rotation_period = 10800.  # Rotation period (seconds)
axis_lat = np.deg2rad(30.)  #  Latitude of the north pole relative to the orbital plane (rad)
axis_long = np.deg2rad(0.) # Longitude of the north pole measured from the direction of perihelion (rad)

# =============================================================================
#                                Orbit
# =============================================================================
eccentricity = 0.0  # Eccentricity of the orbit
semi_major_axis = 1.0  # Semimajor axis of the orbit (au)


# =============================================================================
#                           Numerical grid
# =============================================================================
facet_size = facet_size_arr[i1]  #  Average height of the triangular surface elements (m)
number_of_thermal_wave_depths = number_of_thermal_wave_depths_arr[i2]  #  Total depth of the grid, measured as the number of depths of a diurnal thermal wave
first_layer_depth = first_layer_depth_arr[i3]  # Depth of the first layer below the surface, expressed as a fraction of the diurnal thermal wave depth
number_of_layers = number_of_layers_arr[i4]  #  Number of layers of the numerical grid
time_step_factor = time_step_factor_arr[i5]  # Critical time step scaling factor


# =============================================================================
#                               Calculation
# =============================================================================
total_drift = seasonal_yarkovsky_effect(a1, a2, a3,  # shape of the asteroid
                              rho, k, albedo, cp, eps,  # physical characteristics
                              axis_lat, axis_long, rotation_period,  # rotation state
                              semi_major_axis, eccentricity,  # orbital elements
                              facet_size, number_of_thermal_wave_depths, first_layer_depth, number_of_layers, time_step_factor) # numerical grid parameters
