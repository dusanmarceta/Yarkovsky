import numpy as np
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt
'''
Plots the diurnal temperature cycle at a specified latitude, depth, and orbital location.

Inputs:
    location          - Orbital location at which the temperature cycle is plotted
    lat_0             - Desired latitude at which to plot the temperature cycle (in degrees)
    depth_0           - Desired depth at which to plot the temperature cycle (in meters)
    temperature_file  - File containing the temperature data
    grid_file         - File containing the hour angle, latitude data and depth data

Note: The temperature profile can only be retrieved within the latitude range defined in the ha_lat_file.
      Near the poles, where data is not available, the profile cannot be generated.
'''

#---------------------------- Input data --------------------------------------
location = 0
lat_0 = 84
depth_0 = 1e-4

temperature_file = 'temperature.npy'
grid_file = 'grid.npz'

# -----------------------------------------------------------------------------

ha = np.load(grid_file)['hour_angle']
lats = np.load(grid_file)['latitude']
depths = np.load(grid_file)['layer_depths']
temperature = np.load(temperature_file)


if location >= len(temperature):
    raise SystemExit(f"Invalid location index: {location}. Maximum allowed index is {len(temperature) - 1}.")

if lat_0 > np.max(lats) or lat_0 < np.min(lats):
    raise SystemExit(
        "Requested latitude ({:.2f}°) is out of bounds. Valid range is from {:.2f}° to {:.2f}°.".format(
            lat_0, np.min(lats), np.max(lats)
        )
    )

if depth_0 > np.max(depths):
    raise SystemExit(
        "Requested depth ({:.2f} m) exceeds the maximum depth in the mesh ({:.2f} m).".format(
            depth_0, np.max(depths)
        )
    )

temperature = temperature[location]

if depth_0<depths[0]:
    depth_0 = depths[0]

ha_profile = np.linspace(np.min(ha), np.max(ha), 300)

# 1) Temperature interpolation at lat_0 along longitudes for each layer
temp_along_lon_per_depth = []

points_interp = np.array([[lat_0, lon] for lon in ha_profile])
points = np.column_stack((lats.ravel(), ha.ravel()))  # shape (m*n, 2)

for i in range(len(temperature)):
    print('level {} out of {}'.format(i+1, len(temperature)))
    values = temperature[i].ravel()  # uzmi samo jedan sloj
    temp_profile = griddata(points, values, points_interp, method='linear')
    temp_along_lon_per_depth.append(temp_profile)

temp_along_lon_per_depth = np.array(temp_along_lon_per_depth)  # shape (k, len(lon_profile))

# 2) Interpolation in depth (1D) for each longitude
interp_temp = []

for j in range(len(ha_profile)):
    f = interp1d(depths, temp_along_lon_per_depth[:, j], kind='linear', bounds_error=False)
    interp_temp.append(f(depth_0))

interp_temp = np.array(interp_temp)

# 3) plot
plt.plot(ha_profile/15, interp_temp)
plt.xlabel('Hour angle (h)', fontsize = 24)
plt.ylabel('Temperature (K)', fontsize = 24)
plt.title(f'Diurnal temperature at {depth_0} m and latitude {lat_0} degrees', fontsize = 24)

plt.xticks(np.arange(-12, 15, 3), fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.show()
