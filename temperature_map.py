import numpy as np
import matplotlib.pyplot as plt
'''
Plots temperature contours at the depth closest to the specified value.

Inputs:
    location            - Orbital location at which the temperature map is plotted
    depth_0             - Desired depth at which to plot the temperature map (in meters)
    number_of_contours  - Number of contour levels to display
    temperature_file    - File containing the temperature data
    grid_file           - File containing the hour angle, latitude data and depth data
    
Note: The temperature map is limited to the latitude range defined in the ha_lat_file.
      Near the poles, where data is unavailable, the map cannot be generated.
'''

#---------------------------- Input data --------------------------------------
location = 0
depth_0 = 4.5e-03
number_of_contours = 100

temperature_file = 'temperature.npy'
grid_file = 'grid.npz'

# -----------------------------------------------------------------------------

ha = np.load(grid_file)['hour_angle']
lats = np.load(grid_file)['latitude']
depths = np.load(grid_file)['layer_depths']
temperature = np.load(temperature_file)


if location >= len(temperature):
    raise SystemExit(f"Invalid location index: {location}. Maximum allowed index is {len(temperature) - 1}.")


if depth_0 > np.max(depths):
    raise SystemExit(
        "Requested depth ({:.2f} m) exceeds the maximum depth in the mesh ({:.2f} m).".format(
            depth_0, np.max(depths)
        )
    )

temperature = temperature[location]
depth_index = np.argmin(np.abs(depths - depth_0))



contour = plt.contourf(ha/15, lats, temperature[depth_index], number_of_contours, cmap='jet')
plt.xlabel('hour angle (h)', fontsize=24)
plt.ylabel('latitude (deg)', fontsize=24)
plt.title(f'Temperature map at depth of {np.round(depths[depth_index], 6)} m', fontsize=20)
plt.xticks(np.arange(-12, 15, 3), fontsize=20)
plt.yticks(fontsize=20)

# Colorbar
cbar = plt.colorbar(contour)
cbar.set_label('Temperature (K)', fontsize=20)
cbar.ax.tick_params(labelsize=18)  # font size for colorbar ticks

plt.grid(True)
plt.show()