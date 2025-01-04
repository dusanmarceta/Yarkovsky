import numpy as np
from functions import prismatic_mesh,  sun_position
from constants import G, au, speed_of_light, sigma, S0, y2s
import matplotlib.pyplot as plt

tolerancija = 0.05 # kada je relativna razlika izmedju susednih ekstrema manja od ove vrednosti prekida se racun (slika u prilogu mejla)
number_of_local_extrema = 10
'''
Fizicke karakteristike asteorida
'''

rho = 1000 # gustina (kg/m^3)
k = 1e-2 # koeficijent toplotne provodljivosti (W/(m*K))
eps = 1. # emissivity of the surface element
cp = 1000. # Toplotni kapacitet pri konstantnom pritisku (J/kg K)
albedo = 0.
eccentricity=0.
semi_major_axis = 1. # au
semi_axis_a, semi_axis_b, semi_axis_c = 0.5, 0.5, 0.5 # radijusi troosnog elipsoida (m)
rotation_period = 18 # seconds
axis_lat = np.deg2rad(90.) # latituda severnog pola (ovo je 90-gama)

'''
Karakteristike mreze

ukupan broj celija je N * number_of_ls * rezolucija_po_dubini,
gde je N broj trouglica kojima je podeljena povrsina asteroida

'''

facet_size = 0.1 # visina trouglica kojima je podeljena povrsina asteroida (m). Broj celija se menja sa kvadratom ovog parametra!!!
number_of_ls = 4 # broj dubina prodiranja termalnog talasa do koje se racuna temperatura. Broj celija zavisi linearno od ovog parametra!!!
rezolucija_po_dubini = 10 # od ove vrednosti zavisi visina celije, tj. debljina sloja. Sa ovom vrednoscu se deli ls da bi se dobila visina celije. . Broj celija zavisi linearno od ovog parametra!!!

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

'''
vremenski korak

racuna se kriticni vremenski korak (Spitale, eq. 2). 
On se deli sa t_factor i uzima se manji izmedju toga i inicijalnog koraka (time_step)
'''
t_factor = 10 # faktor sa kojim se deli kriticni vremenski korak
time_step = rotation_period / 24 # inicijalni vremenski korak

'''
Simulacija
'''

ls = np.sqrt(k*rotation_period/rho/cp/2*np.pi) # dubina prodiranja termalnog talasa

dubina = number_of_ls*ls # ukupna dubina do koje se vrsi podela na celije
korak = ls / rezolucija_po_dubini # korak sa kojim se deli asteroid po dubini
layers  = np.arange(korak, dubina + korak , korak) # dubine slojeva

# pravljenje mreze
neighbor_cells, volumes, areas, distances, surface_cells, surface_areas, surface_normals = prismatic_mesh(semi_axis_a, semi_axis_b, semi_axis_c, facet_size, layers)

# odredjivanje vremenskog koraka

d_min = np.min(distances) # najmanje rastojanje izmedju susednih celija mreze
dt_critical=d_min**2*rho*cp/k # kriticni vremenski korak
time_step = np.min([time_step, dt_critical/t_factor]) # uzima se manji

mean_motion=np.sqrt(G/(semi_major_axis*au)**3) #srednje kretanje (rad/s)
mass = 4/3 * semi_axis_a * semi_axis_b * semi_axis_c * np.pi * rho # masa asteroida (kg)
T_equilibrium = (S0/semi_major_axis**2/4/sigma)**0.25 # ravnotezna temperatura


T = np.zeros(len(volumes)) + T_equilibrium # inicijalne temperature svih celija

drift = [] # da/dt (m/s)
i=0

# Konvergencija
detektovano=0 # broj detektovanih lokalnih ekstrema
loc_min=1
loc_max=1

vreme = 0 # ukupno vreme

while True:
    
    vreme += time_step
    if np.mod(i, 1000) == 0:
        print('iteracija br:{} \ntrenutna razlika izmedju ekstrema: {}\nbroj detektovanih ekstrema: {}'.format(i, np.round((loc_max - loc_min)/loc_max, 2), detektovano))
        print('-------------------------------------------')
        # ovo printa dokle je stigla konvergencija
    r_sun, r_trans, sun_distance, solar_irradiance = sun_position(axis_lat, 0., rotation_period, semi_major_axis, eccentricity, vreme, 0, 0)            
    delta_T = T[neighbor_cells] - T[:, None] # razlika temperatura svake celije u odnosu na susedne. Ovaj niz ima dimenziju Nx5, gde je N ukupan broj celija, a svaka ima 5 susednih.
    
    J = np.sum(k * delta_T / distances * areas, axis=1) # fluks usled kondukcije
        
    # external flux
    J[surface_cells] += surface_areas * ((np.maximum(solar_irradiance * (1-albedo) * 
     np.dot(surface_normals, r_sun), 0)) - sigma*eps*(T[surface_cells])**4)
    

    dTdt = J/(rho * volumes * cp) # promena temperature
    T += dTdt * time_step # nova temepratura u sledecem koraku

    # ukupna sila
    F=2/3*eps*sigma/ speed_of_light * np.sum((T[surface_cells][:, None])**4 * surface_normals * surface_areas[:, None], axis=0)
    
    # transverzalna sila koja daje Jarkovski
    B = np.dot(F, r_trans)

    # drift
    dadt = 2 * semi_major_axis / mean_motion / sun_distance * np.sqrt(1 - eccentricity**2) * B / mass
    drift.append(dadt)
    
    # konvergencija
    if i>2:
        if abs(drift[i-1])>abs(drift[i-2]) and abs(drift[i-1])>abs(drift[i]):
            loc_max = abs(dadt)
            detektovano +=1
            
            
        if abs(drift[i-1])<abs(drift[i-2]) and abs(drift[i-1])<abs(drift[i]):
            loc_min = abs(dadt)
            detektovano +=1

    # uslov konvergencije
    if  detektovano>=number_of_local_extrema and (loc_max - loc_min)/loc_max < tolerancija: # ovo treba proveriti (desava se da uzme dva maksimuma ili dva minimuma)
        rezultat = (loc_max + loc_min)/2
        
        break 

    
    i += 1
    
print('\n======= Yarkovsky drift (numericki) =======')
print('\n{} m/s \n\n{} km/god \n\n{} au/my\n'.format(np.round(rezultat, 6), np.round(rezultat * y2s /1000, 3), np.round(rezultat * y2s / au * 1e6, 3)))
print('======= Yarkovsky drift (numericki) =======')