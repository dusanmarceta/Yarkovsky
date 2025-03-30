import numpy as np
from functions import diurnal_yarkovsky_effect

# =============================================================================
#                       Physical characteristics
# =============================================================================
rho = 1000.  # bulk density (kg/m^3)
k = 1e-3  # coefficient of thermal conductivity (W/(m*K))
eps = 1.  # emissivity of the surface
cp = 1000.  # Heat capacity at constant pressure (J/kg K)
albedo = 0.  # Bond albedo
a1 = 35.  # semi-axis of the ellipsod in equatorial plane (m)
a2 = 35.  # semi-axis of the ellipsod in equatorial plane (m)
a3 = 35.  # semi-axis of the ellipsod along the rotation axis (m)

# =============================================================================
#                               Rotation
# =============================================================================
rotation_period = 60.  # Rotation period (seconds)
axis_lat = np.deg2rad(60.)  #  Latitude of the north pole relative to the orbital plane (rad)
axis_long = np.deg2rad(0.) # Longitude of the north pole measured from the direction of perihelion (rad)

# =============================================================================
#                                Orbit
# =============================================================================
eccentricity = 0.3  # Eccentricity of the orbit
semi_major_axis = 1.0  # Semimajor axis of the orbit (au)
number_of_locations = 10  # Number of points along the orbit where the Yarkovsky effect is computed

# =============================================================================
#                           Numerical grid
# =============================================================================
facet_size = 5.  #  Average height of the triangular surface elements (m)
number_of_thermal_wave_depths = 2  #  Total depth of the grid, measured as the number of depths of a diurnal thermal wave
first_layer_depth = 0.05  # Depth of the first layer below the surface, expressed as a fraction of the diurnal thermal wave depth
number_of_layers = 10  #  Number of layers of the numerical grid
time_step_factor = 3  # Critical time step scaling factor

# =============================================================================
#                           Convergence criteria
# =============================================================================
max_tol = 5e-2  # Required relative difference between the maxima of two successive rotations
min_tol = 5e-2  # Required relative difference between the minima of two successive rotations
mean_tol = 5e-2  # Required relative difference between the means of two successive rotations
amplitude_tol = 5e-2  # Required relative amplitude (total relative variation over one full rotation)
maximum_number_of_rotations = 100  # Maximum allowed number of full rotations.



# =============================================================================
#                               Calculation
# =============================================================================
mean_anomaly, drift, total_drift = diurnal_yarkovsky_effect(a1, a2, a3, # shape of the asteroid
                             rho, k, albedo, cp, eps, # physical characteristics
                             axis_lat, axis_long, rotation_period, # rotation state
                             semi_major_axis, eccentricity, number_of_locations, # orbital elements
                             facet_size, number_of_thermal_wave_depths, first_layer_depth, number_of_layers, time_step_factor, # numerical grid parameters
                             max_tol, min_tol, mean_tol, amplitude_tol, maximum_number_of_rotations) # convergence parameters
