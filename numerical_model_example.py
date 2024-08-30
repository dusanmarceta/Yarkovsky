import numpy as np
from functions import numerical_yarko


# Constants
au = 149597870700. # astronomical unit
y2s = 31557600.0 # number of seconds in julian year
k=1.5 # thermal conductivity
cp = 1000 # heat capacity
epsi = 1. # emisivity
rho = 2500 # bulk density
albedo = 0. # bond albedo
a0 = 2. # semi-major axis (au)
rotPer = 3600.# rotational period (s)
R = 0.5 # asteroid radius (m)
axis_lat = np.deg2rad(-90) # longitude of the north pole
axis_long = 0 # latitude of the north pole
time_step = rotPer/48
tolerance = 0.05 # convergence tolerance







# numerical mesh
Nr = 9 # number of cells in radial direction
Nphi = 36 # number of cells in longitudinal direction
Ntheta = 19 # number of cells in latitudinal direction



# calculating Yarkovsky drift (m/s)
drift = numerical_yarko(R, rho, k, albedo, cp, epsi, axis_lat, axis_long, rotPer, Nr, Nphi, Ntheta, Nr_shell = Nr, time_step = time_step, a=a0, e = 0, t = 0, M0 = 0, rotation_0 = 0,  tol = tolerance, lin = 0, conductivity = 1)

    

