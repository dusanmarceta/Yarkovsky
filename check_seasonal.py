import numpy as np
import matplotlib.pyplot as plt

filename = 'yarko_seasonal.txt'

values = []
drift_total = None

with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("Mean_Anomaly"):
            continue
        elif line.startswith("# Total Yarkovsky drift"):
            drift_total = float(line.split(":")[1].strip())
        else:
            values.append(float(line))

values = np.array(values)

print("Vrednosti:", values)
print("Ukupni Yarkovsky drift:", drift_total)