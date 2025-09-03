import numpy as np
from astropy.constants import au, GM_sun


year = 2*np.pi/np.sqrt(GM_sun.value / au.value**3)

step = 86400 

P_precession_values = np.arange(-year, year, 20 * step)


# Pročitaj originalni template fajl
with open('template_precession.py', 'r') as f:
    template = f.read()

for i, precession in enumerate(P_precession_values):

    modified = template
    modified = modified.replace('precession_period = np.inf', f'precession_period = {precession}')
    filename = f'input/input_precession_{i}.py'
    with open(filename, 'w') as f:
        f.write(modified)