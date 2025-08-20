## 🌞 Numerical Thermophysical Asteroid Model

This model enables the calculation of the isolated diurnal and seasonal components of the Yarkovsky effect.
It also provides the asteroid’s temperature field, as well as the evolution of the Yarkovsky drift over both the
rotation and orbital periods.

## 🌞 Diurnal Yarkovsky Effect

### 📄 Description

The script `diurnal_yarkovsky_effect.py` computes **diurnal Yarkovsky effect** based on thermal and orbital properties.  
It supports **optional saving** of intermediate data such as the **temperature field** and **computational grid**.

Script for visualizing the results are also prvoded (see below.)

---

### 🚀 Usage

```bash
python3 diurnal_yarkovsky_effect.py -input input_parameters.py -yarko drift.txt -temp temperature.npy -grid grid.npz
```

If you do not need some of the outputs, you can simply omit the corresponding arguments.  
For example, to run the script without saving temperature or grid data, use:

```bash
python3 diurnal_yarkovsky_effect.py -input input_parameters.py -yarko drift.txt
```

---

### ⚙️ Arguments

| Argument      | Required | Description                                                                 |
|---------------|----------|-----------------------------------------------------------------------------|
| `-input`      | ✅        | Path to a Python script (`.py`) containing the simulation parameters.        |
| `-yarko`      | ❌        | Output `.txt` file where the computed Yarkovsky drift will be saved.        |
| `-temp`       | ❌        | Output `.npy` file to save the temperature data.                            |
| `-grid`       | ❌        | Output `.npz` file to save the computational grid, including:               |
|               |          | &nbsp;&nbsp;&nbsp;&nbsp;• `hour_angle`                                      |
|               |          | &nbsp;&nbsp;&nbsp;&nbsp;• `latitude`                                        |
|               |          | &nbsp;&nbsp;&nbsp;&nbsp;• `layer_depths`                                    |


An example input parameter file (input_parameters.py) is provided with the project.
This file defines the physical, thermal, and orbital properties required for the simulation.

You should edit this example file to set your own parameters as needed.
While you can freely modify the parameter values, the overall structure and format of the file must be preserved to ensure correct parsing by the script.

---

### ⚙️ Convergence Criteria

Convergence of the Yarkovsky effect calculation for a spherical asteroid is illustrated in Figure 1. The user can control the convergence through parameters specified in the input file, which are described in the table below. If some of the parameters are not relevant to the user, they can simply be set to a very large value.


![Image](https://github.com/user-attachments/assets/c635e055-e478-4d70-82f6-18d9541e0f4a)

*Figure 1: Convergence of the Yarkovsky effect calculation for a spherical asteroid rotating around a fixed spin axis.*


| Argument          |  Description                                                                       |
|-------------------|------------------------------------------------------------------------------------|
| `max_tol`         | Required relative difference between the maxima of two successive rotations (max_1 and max_2)|
| `min_tol`         | Required relative difference between the minima of two successive rotations (min_1 and min_2)|
| `mean_tol`        | Required relative difference between the means of two successive rotations (mean values of the rotations marked in blue and red)|
| `amplitude_tol`   | Required maximum relative amplitude (total relative variation over one full rotation)|


Unlike the case of a spherical asteroid rotating around a fixed spin axis, where the Yarkovsky effect remains constant throughout the rotation, this is not the case for a non-spherical asteroid or a spherical asteroid rotating around a precessing axis. In these cases, the Yarkovsky effect varies over a single rotation period, as illustrated in Figure 2. 


![Image](https://github.com/user-attachments/assets/6f8c2aa1-b5d1-4e1f-b322-ff62905ba980)

*Figure 2: Convergence of the Yarkovsky effect calculation for a non-spherical or precessing spherical asteroid.*


Consequently, it does not make sense to require convergence of the `amplitude_tol` parameter, since it depends on the asteroid's shape and rotational state. Here, the convergence process is controlled solely by the first three parameters from the previous table.

In addition to the parameters mentioned above, the user can also specify the maximum number of asteroid rotation periods after which the calculation will terminate, regardless of whether the specified convergence criteria have been met, using the `maximum_number_of_rotations` parameter.

---

### 📊 Visualization Tools

The computed temperature field and Yarkovsky drift can be **visualized** using the following helper scripts:

#### 🔹 `temperature_map.py`
➡️ Plots temperature maps (e.g., surface temperature distribution) for a given depth and orbital location.  
📌 *See the script header for detailed usage.*

#### 🔹 `diurnal_profile.py`
➡️ Plots the **diurnal temperature variation** at a specified latitude and depth over one rotation period.  
📌 *See the script header for detailed usage.*

#### 🔹 `depth_profile.py`
➡️ Plots how temperature **changes with depth** at a specific orbital location, hour angle and latitude.  
📌 *See the script header for detailed usage.*

---

✳️ *Coming soon: Seasonal Yarkovsky Effect module*
