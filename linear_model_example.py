import numpy as np
from linearni_model import yarko_diurnal_circular
from constants import au, y2s

# Constants

rho = 1000 # gustina (kg/m^3)
k = 1e-2 # koeficijent toplotne provodljivosti (W/(m*K))
epsi = 1. # emissivity of the surface element
cp = 1000. # Toplotni kapacitet pri konstantnom pritisku (J/kg K)
albedo = 0.
semi_major_axis = 1. # au
R = 0.5 # asteroid radius (m)
rotation_period = 18 # seconds
gam = 60. # spin axis obliquity (deg)

drift_a = yarko_diurnal_circular(rho, k, cp, R, semi_major_axis, gam, rotation_period/ 3600, 1-albedo, epsi)

print('\n======= Yarkovsky drift (analiticki) =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_a, 6), np.round(drift_a * y2s /1000, 3), np.round(drift_a * y2s / au * 1e6, 3)))
print('======= Yarkovsky drift (analiticki) =======')




