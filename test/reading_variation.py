import numpy as np
import matplotlib.pyplot as plt




 
fajl = 'output/drift_4_orbite.txt'

data = np.loadtxt(fajl, comments='#', skiprows = 1)
mean_anomaly = data[:, 0]
drift = data[:, 1]

plt.plot(mean_anomaly, drift)
plt.grid()


