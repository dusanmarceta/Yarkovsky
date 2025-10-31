import numpy as np
import sys
sys.path.append('..')  # dodaje folder iznad u putanju

from auxiliary_functions import true2ecc



axis_step = 5
location_number = 20
axis_long = np.arange(-90, 90 + axis_step, axis_step)

ecc = 0.6

fraction = []

for i in range(location_number):
    
    
    f = 2*np.pi / location_number * i
    
    E = true2ecc(f, ecc)
    
    M = E - ecc * np.sin(E)
    
    
    fraction.append(np.mod(M / (2*np.pi), 1))
    


# Pročitaj originalni template fajl
with open('template_axis_position.py', 'r') as f:
    template = f.read()

for i, fraction in enumerate(fraction):
    for j, long in enumerate(axis_long):

        modified = template
        modified = modified.replace('eccentricity = 0.5', f'eccentricity = {ecc}')
        modified = modified.replace('axis_long = np.deg2rad(45.)', f'axis_long = {np.deg2rad(long)}')
        modified = modified.replace('initial_position = 0', f'initial_position = {fraction}')
        filename = f'input_axis_position/input_{i}_{j}.py'
        with open(filename, 'w') as f:
            f.write(modified)