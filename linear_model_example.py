import numpy as np
from functions import yarko_diurnal_circular


# Constants
au = 149597870700. # astronomical unit
y2s = 31557600.0 # number of seconds in julian year
k=1.5 # thermal conductivity
cp = 1000 # heat capacity
epsi = 1. # emisivity
rho = 2500 # bulk density
alpha = 1. # absorption coefficient
a0 = 2. # semi-major axis (au)
gam = 180. # spin axis obliquity (deg)
rotPer = 1.# rotational period (h)
R = 0.5 # asteroid radius (m)

    

drift = yarko_diurnal_circular(rho, k, cp, R, a0, gam, rotPer, alpha, epsi)




