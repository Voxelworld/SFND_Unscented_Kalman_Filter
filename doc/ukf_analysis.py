# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Unscented Kalman Filter: Highway Analysis
#
# This Jupyter notebook will analyse the Unscented Kalman Filter (UKF) computations of a single test run.
#
# To capture a test run of the highway scene, start the simulation by writing the terminal output to a CVS file:
#
# `$ ukf_highway.exe >ukf_log.csv`

# %% [markdown]
# ### Read log file

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ukf_log.csv", skiprows=0);
df.describe()

# %%
df.head(4)

# %% [markdown]
# ## Trace Plot
#
# The following figure shows the trace of the lidar measurements (circles) vs. the prediction (line) for each named instance.

# %%
plt.figure(figsize=(8,10))
plt.title("Lidar Map: Sensor vs. Prediction")
for i, name in enumerate(np.unique(df.name)):
    df_i = df[df.name == name]
    c = plt.cm.tab10(i)
    plt.scatter(df_i.lidar_y, df_i.lidar_x, marker="o", s=50, color=c, alpha=0.1, label=f"{name} lidar")
    plt.plot(df_i.x_py, df_i.x_px, color=c, label=f"{name} prediction")    
plt.scatter([0],[0], marker="o", color="black", s=100, label="Ego car")
plt.gca().invert_xaxis()
plt.legend()
plt.grid()
plt.show();

# %% [markdown]
# <img src="https://video.udacity-data.com/topher/2019/April/5cb8ef7d_ukf-highway-projected/ukf-highway-projected.gif" width=500 align="left"/>

# %% [markdown]
# ## Velocity Profiles
#
# The following figure shown the corresponding speed profile of the 3 cars. Note: Speed is measured relative to the motion of the ego car.

# %%
plt.figure(figsize=(8,8))
plt.title("Predicted Velocity Profiles")
for name in np.unique(df.name):
    plt.plot(df[df.name==name].x_v, label=name)
plt.xlabel("t")
plt.ylabel("m/s")
plt.grid()
plt.legend()
plt.show()


# %% [markdown]
# ## Modelled Process Noise
#
# Next we check the consistency of the selected process noise parameters.
#
# The *Normalized Innovation Squared (NIS)* has been computed for the lidar and radar measurement, as introduced in the lesson [Parameters and Consistency](https://classroom.udacity.com/nanodegrees/nd313/parts/da5e72fc-972d-42ae-bb76-fca3d3b2db06/modules/a247c8c2-7d8c-4298-a3d9-a5eee48805cc/lessons/daf3dee8-7117-48e8-a27a-fc4769d2b954/concepts/f3ba9445-452d-4727-8209-b317d44ff1f1). It relates the difference between prediction and measurement $\Delta_{z}$ to the selected covariance matrix $S$ of the process noise.
#
# NIS value $\epsilon = \Delta_{z}^T \cdot S^{-1} \cdot \Delta_{z}$ follows a $\chi^2$ distribution. To make sure that 95% of the time the prediction doesn't exceeds the expected process variance the NIS should be below the threshold of 6 (2D lidar measurements) or 7.82 (3D radar measurements) respectively.
#
# This is checked in the following figures. Beside some spikes the NIS does not exceed the threshold, which means the deviation in the measurements doesn't exceed the modelled noise.
#
# Compare this to the results of the filter consistency checks in the [lessons video](https://classroom.udacity.com/nanodegrees/nd313/parts/da5e72fc-972d-42ae-bb76-fca3d3b2db06/modules/a247c8c2-7d8c-4298-a3d9-a5eee48805cc/lessons/daf3dee8-7117-48e8-a27a-fc4769d2b954/concepts/b9251b43-1412-4c2b-8a0b-6ef3f1eb729a).

# %%
def plot_nis(series, name="radar", threshold=7.815):
    
    plt.figure(figsize=(12,6))
    title = f"$NIS_{{{name}}}(t)$"
    plt.title(title)
    plt.plot(series, label=f"NIS {name}")
    plt.hlines(threshold, xmin=0, xmax=len(df), color="red", label="95 %")
    plt.xlabel("t")
    plt.ylabel("NIS value")
    plt.legend()
    plt.grid()
    plt.show()

    plt.title(f"Histogram of $NIS_{{{name}}} values$")
    n,_,_ = plt.hist(series, cumulative=False, bins=30)
    plt.vlines(threshold, ymin=0, ymax=max(n), color="red")
    plt.xlabel("NIS value")
    plt.show()


# %% [markdown]
# ### Radar

# %%
plot_nis(df[df.sensor_type=="radar"].nis_radar, "radar", 7.815)

# %% [markdown]
# ### Lidar

# %%
plot_nis(df[df.sensor_type=="lidar"].nis_lidar, "lidar", 6)
