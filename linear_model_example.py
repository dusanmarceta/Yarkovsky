import numpy as np
from linear_model_functions import yarko_diurnal_circular, yarko_seasonal_circular,yarko_eccentric
from constants import au, y2s







rho = 1000 # bulk density (kg/m^3)
k = 1e-2 # coefficient of thermal conductivity (W/(m*K))
epsi = 1. # emissivity of the surface
cp = 1000. # Heat capacity at constant pressure (J/kg K)
albedo = 0.  # Bond albedo
semi_major_axis = 1. # au
eccentricity = 0.3
R = 35. # asteroid radius (m)
rotation_period = 3600. # seconds
gamma = 30


drift_diurnal = yarko_diurnal_circular(rho, k, cp, R, semi_major_axis, gamma, rotation_period/ 3600, 1-albedo, epsi)

drift_seasonal = yarko_seasonal_circular(rho, k, cp, R, semi_major_axis, gamma, rotation_period/ 3600, 1-albedo, epsi)

drift_eccentric = yarko_eccentric(semi_major_axis, eccentricity, rho, k, cp, R, gamma, rotation_period/ 3600, 1 - albedo, epsi, 0)

                                        


print('\n======= Yarkovsky diurnal =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_diurnal, 6), np.round(drift_diurnal * y2s /1000, 6), np.round(drift_diurnal * y2s / au * 1e6, 6)))

print('======= Yarkovsky seasonal =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_seasonal, 6), np.round(drift_seasonal * y2s /1000, 6), np.round(drift_seasonal * y2s / au * 1e6, 6)))


print('======= Yarkovsky eccentric =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_eccentric, 6), np.round(drift_eccentric * y2s /1000, 6), np.round(drift_eccentric * y2s / au * 1e6, 6)))

