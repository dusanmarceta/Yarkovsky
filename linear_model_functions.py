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


def cross_prod(a, b):
    return np.cross(a, b)


def k1k2k3_eval(R, Theta):
    if R > 30.0:
        k1 = 0.5
        k2 = 0.5
        k3 = 0.5
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
    
    return k1, k2, k3


def keplereq(ell, ecc):
    jmax = 100
    uk = np.pi  # Initial guess
    elle = np.mod(ell, 2 * np.pi)
    
    for j in range(jmax):
        ukp1 = uk + (elle - uk + ecc * np.sin(uk)) / (1.0 - ecc * np.cos(uk))
        
        # Stop when the difference is smaller than eps = 1e-12
        if np.abs(ukp1 - uk) <= 1e-12:
            break
        
        uk = ukp1
    
    return ukp1


def kep2car(kep):
    gmSun = 1.32712440018e20  # Gravitational parameter for the Sun in m^3/s^2
    deg2rad = np.pi / 180.0
    kappa = np.sqrt(gmSun)
    kappa2 = kappa ** 2
    
    aa = kep[0]
    ecc = kep[1]
    inc = kep[2] * deg2rad
    omeg = kep[3]
    Omnod = kep[4]
    ell = kep[5]
    
    # Convert into Delaunay elements
    L = kappa * np.sqrt(aa)
    G = L * np.sqrt(1.0 - ecc ** 2)
    Z = G * np.cos(inc)
    
    # Rotation matrix of angle omega for the asteroid
    co = np.cos(omeg)
    so = np.sin(omeg)
    R_om = np.array([[co, -so, 0],
                     [so, co, 0],
                     [0, 0, 1]])
    
    # Rotation matrix of angle Inclination for the asteroid
    ci = np.cos(inc)
    si = np.sin(inc)
    R_i = np.array([[1, 0, 0],
                    [0, ci, -si],
                    [0, si, ci]])
    
    # R_i * R_om
    RiRom = np.dot(R_i, R_om)
    
    # Rotation matrix of angle Omega for the asteroid
    con = np.cos(Omnod)
    son = np.sin(Omnod)
    R_Omn = np.array([[con, -son, 0],
                      [son, con, 0],
                      [0, 0, 1]])
    
    ROmnRi = np.dot(R_Omn, R_i)
    
    # R_Omn * R_i * R_om
    Rot = np.dot(R_Omn, RiRom)
    
    # Solve the Kepler equation
    u = keplereq(ell, ecc)
    cosu = np.cos(u)
    sinu = np.sin(u)
    
    # Compute the position in the orbital plane
    pos = np.zeros(3)
    pos[0] = aa * (cosu - ecc)
    pos[1] = aa * np.sqrt(1.0 - ecc**2) * sinu
    pos[2] = 0.0
    
    # Derivative of the position w.r.t u
    dposdu = np.zeros(3)
    dposdu[0] = L**2 / kappa2 * (-sinu)
    dposdu[1] = L * G * cosu / kappa2
    dposdu[2] = 0.0
    
    # Compute the direction of the velocity
    norma = np.sqrt(np.dot(dposdu, dposdu))
    versvel = dposdu / norma
    
    # Compute the norm of the velocity
    normpos = np.sqrt(np.dot(pos, pos))
    normvel = kappa * np.sqrt(2.0 / normpos - 1.0 / aa)
    
    # Assign the velocity
    vel = normvel * versvel
    
    # Assign the cartesian elements
    tmpcar = np.zeros(6)
    tmpcar[0:3] = np.dot(Rot, pos)
    tmpcar[3:6] = np.dot(Rot, vel)
    
    return tmpcar



def trapezoid_average(time, deltaT, fun, npoints):
    integral = 0.0
    # Start adding the function evaluations
    for j in range(1, npoints-1):  # Python uses 0-based indexing, so we start at 1
        integral += fun[j]
    
    # Add the first and last points
    integral += 0.5 * (fun[0] + fun[-1])
    
    # Multiply for the step
    integral *= deltaT
    
    # Divide by the total integration time
    average = integral / time
    
    return average


def yarkovsky_vf(semiaxm, ecc, posvel, rho, K0, C, R, gam, rotPer, alpha, epsi, expo):
    pos = np.array(posvel[:3])
    vel = np.array(posvel[3:])
    
    # Računanje mase asteroida
    mAst = 4.0 * np.pi * rho * R**3 / 3.0
    mu = gmsun + uGc * mAst
    
    # Pretvaranje obliqunosti u radijane
    gam_rad = gam * deg2rad
    
    # Računanje srednje kretanja i frekvencije rotacije
    omega_rev = np.sqrt(mu / (semiaxm * au2m)**3)
    omega_rot = twopi / (rotPer * h2s)
    

    # Računanje jediničnih vektora
    dist = np.linalg.norm(pos)
    verspos = pos / dist
    
    # Računanje toplotne provodljivosti u zavisnosti od udaljenosti od Sunca
    K = K0 * (dist * au2m)**expo
    
    # Računanje dubina penetracije
    ls = np.sqrt(K / (rho * C * omega_rev))
    ld = ls * np.sqrt(omega_rev / omega_rot)
    
    # Računanje e1
    e1 = np.array([np.sin(gam_rad), 0.0, np.cos(gam_rad)])
    
    # Računanje e2
    e2 = cross_prod(verspos, e1)
    
    # Računanje e3
    e3 = cross_prod(e1, e2)
    
    # Računanje solarne energije i subsolarne temperature
    E_star = lumSun / (4.0 * np.pi * dist**2)
    T_star = (alpha * E_star / (epsi * sBoltz))**0.25
    
    # Računanje k koeficijenta
    kappa = 4.0 * alpha * np.pi * R**2 * E_star / (9.0 * mAst * cLight)
    
    # Računanje koeficijenata za e2 i e3 (dnevni efekat)
    Rpd = R / ld
    Theta = np.sqrt(rho * K * C * omega_rot) / (epsi * sBoltz * T_star**3)
    k1, k2, k3 = k1k2k3_eval(Rpd, Theta)
    
    gamma1 = -k1 * Theta / (1.0 + 2.0 * k2 * Theta + k3 * Theta**2)
    gamma2 = (1.0 + k2 * Theta) / (1.0 + 2.0 * k2 * Theta + k3 * Theta**2)
    
    # Računanje koeficijenta za e1 (sezonski efekat)
    Rps = R / ls
    Thetabar = np.sqrt(rho * K * C * omega_rev) / (epsi * sBoltz * T_star**3)
    k1, k2, k3 = k1k2k3_eval(Rps, Thetabar)
    
    gamma1bar = -k1 * Thetabar / (1.0 + 2.0 * k2 * Thetabar + k3 * Thetabar**2)
    gamma2bar = (1.0 + k2 * Thetabar) / (1.0 + 2.0 * k2 * Thetabar + k3 * Thetabar**2)
    
    # Računanje normalnog vektora za orbitalni ravninu, N
    normalvec = cross_prod(pos, vel)
    normnormalvec = np.linalg.norm(normalvec)
    normalvec = normalvec / normnormalvec
    
    # Računanje N x n = N x pos/|pos|
    aux = cross_prod(normalvec, verspos)
    
    # Računanje koeficijenta za e1
    coeff1 = gamma2bar * np.dot(verspos, e1) + gamma1bar * np.dot(aux, e1)
    
    # Računanje ukupnog vektorskog polja Yarkovskog pomaka
    yarko = kappa * (coeff1 * e1 + gamma1 * e2 + gamma2 * e3)
    
    return yarko


def yarko_eccentric(semiaxm, ecc, rho, K, C, R, gam, rotPer, alpha, epsi, expo):
    npoints = 500
    mAst = 4.0 * np.pi * rho * R**3 / 3.0
    mu = gmsun + uGc * mAst
    meanMotion = np.sqrt(mu / (semiaxm * au2m)**3)
    
    period = 2 * np.pi
    deltau = period / float(npoints - 1)
    
    kep = np.zeros(6)
    kep[0] = semiaxm * au2m
    kep[1] = ecc
    kep[2] = 0.0
    kep[3] = 0.0
    kep[4] = 0.0
    kep[5] = 0.0
    
    dadt = np.zeros(npoints)
    
    for j in range(npoints):
        u = float(j) * deltau
        
        # Compute the mean anomaly
        kep[5] = u - ecc * np.sin(u)
        
        # Convert from Keplerian elements to Cartesian elements
        car = kep2car(kep)
        
        # Take position and velocity
        vel = car[3:]
        
        # Compute the Yarkovsky vector field
        yarko = yarkovsky_vf(semiaxm, ecc, car, rho, K, C, R, gam, rotPer, alpha, epsi, expo)
        
        # Vary the force with the eccentricity of the orbit
        dldu = 1.0 - ecc * np.cos(u)
        dadt[j] = 2.0 * np.dot(yarko, vel) / (meanMotion**2 * semiaxm * au2m) * dldu
        dadt[j] *= m2au / s2my
        
    # Average the Yarkovsky force over the period
    ecc_ye = trapezoid_average(period, deltau, dadt, npoints) * au2m * s2my
    return ecc_ye
    
    