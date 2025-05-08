import numpy as np
from functions import diurnal_yarkovsky_effect, kepler, ecc2true
import matplotlib.pyplot as plt
from constants import au

# =============================================================================
#                       Physical characteristics
# =============================================================================
rho = 2490.  # bulk density (kg/m^3)
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
eccentricity = 0.5195  # Eccentricity of the orbit
semi_major_axis = 2.06355  # Semimajor axis of the orbit (au)
number_of_locations = 6  # Number of points along the orbit where the Yarkovsky effect is computed

# =============================================================================
#                           Numerical grid
# =============================================================================
facet_size = a1 / 10  #  Average height of the triangular surface elements (m)
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
maximum_number_of_rotations = 3  # Maximum allowed number of full rotations.



# =============================================================================
#                               Calculation
# =============================================================================
total_effect, drift_evolution, drift_for_location, M_for_location, T_equator_rotation, T_noon, T_midnight, layer_depths,  grid_lon, grid_lat, T_surface = diurnal_yarkovsky_effect(a1, a2, a3, # shape of the asteroid
                             rho, k, albedo, cp, eps, # physical characteristics
                             axis_lat, axis_long, rotation_period, precession_period, # rotation state
                             semi_major_axis, eccentricity, number_of_locations, # orbital elements
                             facet_size, number_of_thermal_wave_depths, first_layer_depth, number_of_layers, time_step_factor, # numerical grid parameters
                             max_tol, min_tol, mean_tol, amplitude_tol, maximum_number_of_rotations) # convergence parameters




f = np.zeros_like(M_for_location)
for i in range(len(M_for_location)):
    E = kepler(eccentricity, M_for_location[i], 1e-6)
    f[i] = np.mod(ecc2true(E, eccentricity), 2*np.pi)
    

np.savetxt('layer_depths.txt', layer_depths)
np.savetxt('drift.txt', np.column_stack((f, drift_for_location)))

np.savetxt('longituda.txt', grid_lon)
np.savetxt('latituda.txt', grid_lat)



with open("T_equator.txt", "a") as file:
    for i in range(number_of_locations):
        row = T_equator_rotation[i]  # neka je ovo 1D np.array (npr. dužine 10)
        file.write(" ".join(str(x) for x in row) + "\n")       
file.close()  


with open("T_noon.txt", "a") as file:
    for i in range(number_of_locations):
        row = T_noon[i]  # neka je ovo 1D np.array (npr. dužine 10)
        file.write(" ".join(str(x) for x in row) + "\n")        
file.close()  

with open("T_midnight.txt", "a") as file:
    for i in range(number_of_locations):
        row = T_midnight[i]  # neka je ovo 1D np.array (npr. dužine 10)
        file.write(" ".join(str(x) for x in row) + "\n")        
file.close()  


for i in range(number_of_locations):
    fajl = 'T_surface_' + str(i) + '.txt'
    
    np.savetxt(fajl, T_surface[i])





print('-----------------')
print(total_effect * 365.25*86400*1e6/au)

location = 0
# Plot the interpolated data
plt.figure(figsize=(10, 5))
contour = plt.contourf((grid_lon + 180)/15, grid_lat, T_surface[location], levels=100, cmap='jet')
plt.xlabel("Local time (h)", fontsize = 20)
plt.ylabel("Latitude (deg)", fontsize = 20)
plt.xticks(np.arange(0, 27, 3), fontsize = 16)
plt.yticks(fontsize = 16)
cbar = plt.colorbar(contour)
cbar.set_label('Temperature (K)', fontsize=20)  # Povećaj font labela
cbar.ax.tick_params(labelsize=16)  
plt.grid()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(layer_depths * 1000, T_noon[location], linewidth = 2, color = 'r', label = 'Noon')
plt.plot(layer_depths * 1000, T_midnight[location], linewidth = 2, color = 'b', label = 'Midnight')
plt.xlabel("Depth below surface (mm)", fontsize = 20)
plt.ylabel("Temperature (K)", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize = 16)
plt.grid()
plt.show()



    
plt.figure(figsize=(10, 5))
plt.plot(np.rad2deg(f[:-1]), drift_for_location[:-1] * 365.25*86400*1e6/au)
plt.xlabel("True anomaly (deg)", fontsize = 20)
plt.ylabel("da/dt (au/My)", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.grid()
plt.show()