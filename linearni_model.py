import numpy as np
'''
LINEAR MODEL
'''
                
               # Constants
pi = np.pi
twopi = 2 * np.pi
deg2rad = pi / 180.0
rad2deg = 180.0 / pi
gmsun = 1.32712440018e20  # gravitational constant * mass of Sun [m^3/s^2]
uGc = 6.67430e-11  # universal gravitational constant [m^3 kg^-1 s^-2]
au2m = 1.495978707e11  # astronomical unit to meters
m2au = 1 / au2m
s2my = 1 / (365.25 * 24 * 3600 * 1e6)  # seconds to million years
h2s = 3600  # hours to seconds
cLight = 299792458.  # speed of light [m/s]
lumSun = 3.828e26  # solar luminosity [W]
sBoltz = 5.670374419e-8  # Stefan-Boltzmann constant [W m^-2 K^-4]

def compute_depths(R, rho, K, C, sma, P):
    mAst = 4.0 * pi * rho * R**3.0 / 3.0
    mu = gmsun + uGc * mAst
    omega_rev = np.sqrt(mu / (sma * au2m)**3.0)
    omega_rot = twopi / (P * h2s)
    ls = np.sqrt(K / (rho * C * omega_rev))
    ld = ls * np.sqrt(omega_rev / omega_rot)
    return ls, ld

def computeYarkoMaxMin_circular(rho, K, C, radius, semiaxm, rotPer, alpha, epsi):
    yarkomax = computeYarko_circular(rho, K, C, radius, semiaxm, 0.0, rotPer, alpha, epsi)
    f90 = computeYarko_circular(rho, K, C, radius, semiaxm, 90.0, rotPer, alpha, epsi)
    f180 = computeYarko_circular(rho, K, C, radius, semiaxm, 180.0, rotPer, alpha, epsi)
    if abs(yarkomax / (2.0 * f90)) < 1.0:
        gam = np.arccos(yarkomax / (2.0 * f90)) * rad2deg
        yarkoadd = computeYarko_circular(rho, K, C, radius, semiaxm, gam, rotPer, alpha, epsi)
        yarkomin = min(yarkoadd, f180)
        gammin = gam
    else:
        yarkomin = f180
        gammin = 180.0
    return yarkomax, yarkomin, gammin

def computeYarko_circular(rho, K, C, radius, semiaxm, gam, rotPer, alpha, epsi):
    das = yarko_seasonal_circular(rho, K, C, radius, semiaxm, gam, rotPer, alpha, epsi)
    dad = yarko_diurnal_circular(rho, K, C, radius, semiaxm, gam, rotPer, alpha, epsi)
    yarko = (das + dad) * m2au * s2my
    return yarko

def yarko_seasonal_circular(rho, K, C, R, a0, gam, rotPer, alpha, epsi):
    gam_rad = gam * deg2rad
    mAst = 4.0 * pi * rho * R**3.0 / 3.0
    mu = gmsun + uGc * mAst
    omega_rev = np.sqrt(mu / (a0 * au2m)**3.0)
    ls = np.sqrt(K / (rho * C * omega_rev))
    Rps = R / ls
    dist = a0 * au2m
    E_star = lumSun / (4.0 * pi * dist**2.0)
    phi = pi * R**2.0 * E_star / (mAst * cLight)
    T_star = (alpha * E_star / (epsi * sBoltz))**0.25
    Theta = np.sqrt(rho * K * C * omega_rev) / (epsi * sBoltz * T_star**3.0)
    F_omega_rev = Fnu_eval(Rps, Theta)
    yarko_s = 4.0 * alpha * phi * F_omega_rev * np.sin(gam_rad)**2.0 / (9.0 * omega_rev)
    return yarko_s

def yarko_diurnal_circular(rho, K, C, R, a0, gam, rotPer, alpha, epsi):
    gam_rad = gam * deg2rad
    mAst = 4.0 * pi * rho * R**3.0 / 3.0
    mu = gmsun + uGc * mAst
    omega_rev = np.sqrt(mu / (a0 * au2m)**3.0)
    omega_rot = twopi / (rotPer * h2s)
    ld = np.sqrt(K / (rho * C * omega_rot))
    Rpd = R / ld
    dist = a0 * au2m
    E_star = lumSun / (4.0 * pi * dist**2.0)
    phi = pi * R**2.0 * E_star / (mAst * cLight)
    T_star = (alpha * E_star / (epsi * sBoltz))**0.25 
    Theta = np.sqrt(rho * K * C * omega_rot) / (epsi * sBoltz * T_star**3.0)
    F_omega_rot = Fnu_eval(Rpd, Theta)
    yarko_d = -8.0 * alpha * phi * F_omega_rot * np.cos(gam_rad) / (9.0 * omega_rev)
    return yarko_d

def Fnu_eval(R, Theta):
    if R > 30.0:
        k1, k2, k3 = 0.5, 0.5, 0.5
    else:
        x = np.sqrt(2.0) * R
        A = -(x + 2.0) - np.exp(x) * ((x - 2.0) * np.cos(x) - x * np.sin(x))
        B = -x - np.exp(x) * (x * np.cos(x) + (x - 2.0) * np.sin(x))
        U = 3.0 * (x + 2.0) + np.exp(x) * (3.0 * (x - 2.0) * np.cos(x) + x * (x - 3.0) * np.sin(x))
        V = x * (x + 3.0) - np.exp(x) * (x * (x - 3.0) * np.cos(x) - 3.0 * (x - 2.0) * np.sin(x))
        den = x * (A**2.0 + B**2.0)
        k1 = (A * V - B * U) / den
        k2 = (A * (A + U) + B * (B + V)) / den
        k3 = ((A + U)**2.0 + (B + V)**2.0) / (den * x)
    Fnu = -k1 * Theta / (1.0 + 2.0 * k2 * Theta + k3 * Theta**2.0)
    return Fnu 