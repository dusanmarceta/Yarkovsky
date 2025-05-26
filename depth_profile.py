import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
'''
Generates the temperature profile as a function of depth at a specified latitude, hour angle, and orbital location.

Inputs:
    lat_0             - Target latitude for the temperature profile (in degrees)
    hour_angle_0      - Target hour angle for the temperature profile (in hours)
    location          - Orbital location at which the temperature profile is generated
    temperature_file  - File containing the temperature data
    grid_file         - File containing the hour angle, latitude data and depth data

Note: The temperature profile can only be retrieved within the latitude range defined in the lon_lat_file.
      Near the poles, where data is not available, the profile cannot be generated.
'''


#---------------------------- Input data --------------------------------------
# Target point
lat_0 = -80
hour_angle_0 = 0
location = 0

temperature_file = 'temperature.npy'
grid_file = 'grid.npz'
# -----------------------------------------------------------------------------

lons = np.load(grid_file)['hour_angle']
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

temperature = temperature[location]

# Pripremi tačku za interpolaciju
target_point = np.array([[lat_0, hour_angle_0]])

# Pripremi ulazne tačke za griddata
points = np.column_stack((lats.ravel(), lons.ravel()))  # shape (m*n, 2)

# 1D profil temperature po dubinama
temp_profile = []

for i in range(len(temperature)):
    print('level {} out of {}'.format(i+1, len(temperature)))
    values = temperature[i].ravel()  # 2D slice po dubini → vector (m*n,)
    temp = griddata(points, values, target_point, method='linear')[0]
    temp_profile.append(temp)

temp_profile = np.array(temp_profile)

# Prikaz
plt.plot(depths, temp_profile, linewidth = 2)
#plt.gca().invert_yaxis()
plt.ylabel('Temperature', fontsize = 24)
plt.xlabel('Depth (m)', fontsize = 24)
plt.title(f'Temperature profile at (lat={lat_0}°, hour angle={hour_angle_0}°)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.show()
