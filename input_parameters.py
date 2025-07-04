import numpy as np
# =============================================================================
#                       Physical characteristics
# =============================================================================
rho = 2049.  # bulk density (kg/m^3)
k = 1e-4  # coefficient of thermal conductivity (W/(m*K))
eps = 1.  # emissivity of the surface
cp = 1000.  # Heat capacity at constant pressure (J/kg K)
albedo = 0.  # Bond albedo
a1 = 7.  # semi-axis of the ellipsod in equatorial plane (m)
a2 = 7.  # semi-axis of the ellipsod in equatorial plane (m)
a3 = 7.  # semi-axis of the ellipsod along the rotation axis (m)

# =============================================================================
#                               Rotation
# =============================================================================
rotation_period = 34.  # Rotation period (seconds)
precession_period = np.inf # Period of the precession of the spin axis (+ for direct, - for retrograde precession) (seconds)
axis_lat = np.deg2rad(90.)  #  Latitude of the north pole relative to the orbital plane (rad)
axis_long = np.deg2rad(0.) # Longitude of the north pole measured from the direction of perihelion (rad)
# =============================================================================
#                                Orbit
# =============================================================================
#eccentricity = 0.52018004318  # Eccentricity of the orbit
#semi_major_axis = 2.06282123907  # Semimajor axis of the orbit (au)
eccentricity = 0.0  # Eccentricity of the orbit
semi_major_axis = 1  # Semimajor axis of the orbit (au)
number_of_locations = 1  # Number of points along the orbit where the Yarkovsky effect is computed

# =============================================================================
#                           Numerical grid
# =============================================================================
facet_size = 0.7  #  Average height of the triangular surface elements (m)
number_of_thermal_wave_depths = 10  #  Total depth of the grid, measured as the number of depths of a diurnal thermal wave
first_layer_depth = 0.1  # Depth of the first layer below the surface, expressed as a fraction of the diurnal thermal wave depth
number_of_layers = 20  #  Number of layers of the numerical grid
time_step_factor = 3  # Critical time step scaling factor

# =============================================================================
#                           Convergence criteria
# =============================================================================
max_tol = 5e-10  # Required relative difference between the maxima of two successive rotations
min_tol = 5e-10  # Required relative difference between the minima of two successive rotations
mean_tol = 5e-10  # Required relative difference between the means of two successive rotations
amplitude_tol = 5e-10  # Required relative amplitude (total relative variation over one full rotation)
maximum_number_of_rotations = 100  # Maximum allowed number of full rotations.

# =============================================================================
#                           Heat conduction
# =============================================================================
lateral_heat_conduction = 0 # controls if code takes into account lateral heat conduction (put 1 to turn it on)