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

### 📝 Notes

- If `-temp` or `-grid` is not provided, the script will **skip saving** the corresponding data.
- The `input_parameters.py` file should define the **physical, thermal, and orbital properties** required for the simulation.

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
