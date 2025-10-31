import numpy as np
import matplotlib.pyplot as plt
import re

from astropy.constants import au, GM_sun


year = 2*np.pi/np.sqrt(GM_sun.value / au.value**3)

step = 86400 

P_precession_values = np.arange(10 * step, year, 5 * step)

drift = []
for i in range(1, len(P_precession_values)):
    
    fajl = f'drift_precession_60loc_{i}.txt'
    
    with open(fajl, "r") as f:
        for line in reversed(f.readlines()):
            if line.startswith("# Total Yarkovsky drift"):
                value = float(re.search(r"([-+eE0-9.]+)", line.split(":")[-1]).group(1))
                break
    drift.append(value)
    
plt.plot(P_precession_values[1:], drift)


yarko_inf = 1.69244024e-04

plt.plot(P_precession_values, np.ones(len(P_precession_values)) * yarko_inf)
#
#choice = 13
#
#fajl = f'drift_precession_45_{choice}.txt'
#
#data = np.genfromtxt(
#    fajl,
#    comments="#",       # preskoči komentare
#    skip_footer=0       # preskoči poslednja 2 reda
#)
#
#MA, drift = data[1:].T
#
#plt.plot(MA, drift)