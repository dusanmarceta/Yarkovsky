import numpy as np
# =============================================================================
#                       Physical characteristics
# =============================================================================
rho = 2000.  # bulk density (kg/m^3)
k = 1e-1  # coefficient of thermal conductivity (W/(m*K))
eps = 0.95  # emissivity of the surface
cp = 800.  # Heat capacity at constant pressure (J/kg K)
albedo = 0.1  # Bond albedo
a1 = 50.  # semi-axis of the ellipsod in equatorial plane (m)
a2 = 50.  # semi-axis of the ellipsod in equatorial plane (m)
a3 = 50.  # semi-axis of the ellipsod along the rotation axis (m)

# =============================================================================
#                               Rotation
# =============================================================================
rotation_period = 24 * 3600.  # Rotation period (seconds)
precession_period = np.inf # Period of the precession of the spin axis (+ for direct, - for retrograde precession) (seconds)
axis_lat = np.deg2rad(0.)  #  Latitude of the north pole relative to the orbital plane (rad)
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
facet_size = 5.  #  Average height of the triangular surface elements (m)
number_of_thermal_wave_depths = 4  #  Total depth of the grid, measured as the number of depths of a diurnal thermal wave
first_layer_depth = 0.1  # Depth of the first layer below the surface, expressed as a fraction of the diurnal thermal wave depth
number_of_layers = 32  #  Number of layers of the numerical grid
time_step_factor = 3  # Critical time step scaling factor

# =============================================================================
#                           Heat conduction
# =============================================================================
lateral_heat_conduction = 1 # controls if code takes into account lateral heat conduction (put 1 to turn it on)

# =============================================================================
#                           Interpolation
# =============================================================================
interpolation = 0 