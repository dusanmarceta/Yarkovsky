import numpy as np
# =============================================================================
#                       Physical characteristics
# =============================================================================
rho = 1000.  # bulk density (kg/m^3)
k = 1e-2  # coefficient of thermal conductivity (W/(m*K))
eps = 1.  # emissivity of the surface
cp = 1000.  # Heat capacity at constant pressure (J/kg K)
albedo = 0.  # Bond albedo
a1 = 5.  # semi-axis of the ellipsod in equatorial plane (m)
a2 = 5.  # semi-axis of the ellipsod in equatorial plane (m)
a3 = 5.  # semi-axis of the ellipsod along the rotation axis (m)

# =============================================================================
#                               Rotation
# =============================================================================
rotation_period = 3600.  # Rotation period (seconds)
precession_period = np.inf # Period of the precession of the spin axis (+ for direct, - for retrograde precession) (seconds)
axis_lat = np.deg2rad(40.)  #  Latitude of the north pole relative to the orbital plane (rad)
axis_long = -1.3962634015954636 # Longitude of the north pole measured from the direction of perihelion (rad)
# =============================================================================
#                                Orbit
# =============================================================================
#eccentricity = 0.62018004318  # Eccentricity of the orbit
#semi_major_axis = 2.06282123907  # Semimajor axis of the orbit (au)
eccentricity = 0.6  # Eccentricity of the orbit
semi_major_axis = 1.  # Semimajor axis of the orbit (au)
initial_position = 0.770690726658035
number_of_orbits = 0  # Number of orbits along which the Yarkovsky effect is calculated (if there is no complex rotation (e.g. spin-axis precession) this should be 1)
number_of_locations_per_orbit = 0  # Number of points along each orbit where the Yarkovsky effect is computed

# =============================================================================
#                           Numerical grid
# =============================================================================
facet_size = 0.5  #  Average height of the triangular surface elements (m)
number_of_thermal_wave_depths = 3  #  Total depth of the grid, measured as the number of depths of a diurnal thermal wave
first_layer_depth = 0.1  # Depth of the first layer below the surface, expressed as a fraction of the diurnal thermal wave depth
number_of_layers = 20  #  Number of layers of the numerical grid
time_step_factor = 3  # Critical time step scaling factor

# =============================================================================
#                           Convergence criteria
# =============================================================================
max_tol = 5e-2  # Required relative difference between the maxima of two successive rotations
min_tol = 5e-2  # Required relative difference between the minima of two successive rotations
mean_tol = 5e-2  # Required relative difference between the means of two successive rotations
amplitude_tol = 5e-2  # Required relative amplitude (total relative variation over one full rotation)
maximum_number_of_rotations = 10  # Maximum allowed number of full rotations.

# =============================================================================
#                           Heat conduction
# =============================================================================
lateral_heat_conduction = 0 # controls if code takes into account lateral heat conduction (put 1 to turn it on)

# =============================================================================
#                           Interpolation
# =============================================================================
interpolation = 0 