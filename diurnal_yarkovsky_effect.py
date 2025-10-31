import argparse
import numpy as np
import importlib.util
from functions import diurnal_yarkovsky_effect

# -----------------------------
# Argument parser setup
# -----------------------------
parser = argparse.ArgumentParser(description="Compute Yarkovsky effect.")
parser.add_argument('-input', type=str, required=True, help='Input parameter file (Python file)')
parser.add_argument('-temp', type=str, default='temperature_field.npz', help='Output filename for surface temperature')
parser.add_argument('-yarko', type=str, default='drift_along_orbit.txt', help='Output filename for orbit drift data')
parser.add_argument('-grid', type=str, default='grid.npz', help='Output filename for hour angle/latitude grid')
parser.add_argument('-prog', type=str, default='progress.log', help='Output filename for simulation progress log')
args = parser.parse_args()

# -----------------------------
# Load input .py file as module
# -----------------------------
spec = importlib.util.spec_from_file_location("input_params", args.input)
params = importlib.util.module_from_spec(spec)
spec.loader.exec_module(params)

# -----------------------------
# Prepare input values
# -----------------------------
p = params
total_effect, drift_for_location, M_for_location, layer_depths, grid_ha, grid_lat, T_asteroid, axis_long_for_location = diurnal_yarkovsky_effect(
    p.a1, p.a2, p.a3,
    p.rho, p.k, p.albedo, p.cp, p.eps,
    p.axis_lat, p.axis_long, p.rotation_period, p.precession_period,
    p.semi_major_axis, p.eccentricity, p.initial_position, p.number_of_orbits, p.number_of_locations_per_orbit,
    p.facet_size, p.number_of_thermal_wave_depths, p.first_layer_depth, p.number_of_layers, p.time_step_factor,
    p.max_tol, p.min_tol, p.mean_tol, p.amplitude_tol, p.maximum_number_of_rotations, args.prog, p.lateral_heat_conduction, p.interpolation
)

# -----------------------------
# Save outputs
# -----------------------------
if args.yarko:
    np.savetxt(
        args.yarko,
        np.column_stack([
            M_for_location,
            drift_for_location,
            np.rad2deg(axis_long_for_location)
        ]),
        header='Mean_Anomaly(rad), Drift(m/s), spin-axis longitude (deg)',
        comments='',
        fmt=['%.6f', '%.6e', '%.2f']  # 6 decimala za prvu, eksponencijalni zapis za drugu, 2 decimale za treću
    )
    
    # Dopiši total drift kao komentar na kraju
    with open(args.yarko, 'a') as f:
        f.write(f'\n# Total Yarkovsky drift (m/s): {total_effect:.8e}\n')

if args.grid:
    np.savez(
        args.grid, 
        hour_angle=grid_ha, 
        latitude=grid_lat, 
        layer_depths=layer_depths
    )

if args.temp:
    np.save(args.temp, T_asteroid)