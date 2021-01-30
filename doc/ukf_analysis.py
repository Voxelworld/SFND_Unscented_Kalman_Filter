# -*- coding: utf-8 -*-
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
# To capture a test run of the highway scene, start the simulation with the `--log` flag to write the console or a CVS file:
#
# `$ ukf_highway.exe --log [ukf_log.csv]`

# %% [markdown]
# ### Read log files

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_all = pd.read_csv("ukf_log.csv")
df_lidar = pd.read_csv("ukf_log_lidar.csv")
df_radar = pd.read_csv("ukf_log_radar.csv")
df_gt = pd.read_csv("highway_traffic_steps.csv")
df_all.describe()

# %%
df_all.head(3)

# %%
df_gt.head(1)


# %% [markdown]
# ## Plotting Trace Maps
#
# This section shows different maps of the following scene:<br/>
# <img src="https://video.udacity-data.com/topher/2019/April/5cb8ef7d_ukf-highway-projected/ukf-highway-projected.gif" width=500 />

# %%
def plot_map(df, sensors=[]):
    plt.title(f"Map: {sensors}")
    for i, name in enumerate(np.unique(df.name)):
        c = plt.cm.tab10(i)
        df_i = df[df.name == name] # car's ukf measurements
        df_t = df_gt[df_gt.name == name] # car's ground truth
        if "lidar" in sensors:
            plt.scatter(df_i.lidar_y, df_i.lidar_x, marker="o", s=50, color=c, alpha=0.1, label=f"{name} lidar")
        if "radar" in sensors:
            radar_x = df_i.radar_r * np.cos(df_i.radar_phi)
            radar_y = df_i.radar_r * np.sin(df_i.radar_phi)
            plt.scatter(radar_y, radar_x, marker="o", s=50, color=c, alpha=0.1, label=f"{name} radar")
        if "prediction" in sensors:
            plt.plot(df_i.x_py, df_i.x_px, linestyle=":", color=c, label=f"{name} prediction")
        if "groundtruth" in sensors:
            plt.plot(df_t.y, df_t.x, linestyle="-", color=c, alpha=0.5, label=f"{name} ground truth")
    plt.scatter([0],[0], marker="o", color="black", s=100, label="Ego car")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.xlabel('m')
    plt.ylabel('m')
    plt.grid()


# %% [markdown]
# ### Quality of Sensor Data
# First we plot maps of the lidar and radar measurements, including the cars ground truth traces.

# %%
plt.figure(figsize=(12,12))
plt.subplot(1,2,1)
plot_map(df_all, ["lidar", "groundtruth"])
plt.subplot(1,2,2)
plot_map(df_all, ["radar", "groundtruth"])
plt.show()

# %% [markdown]
# #### Notes
# * Measurements are relative to the ego car, which is the fixed black dot at position (0,0).
# * Radar positions are noisy, especially at larger distances. Radars angle measurements have a standard deviation of ~ 1.7°.

# %% [markdown]
# ## Kalman Filter Predictions
# The next figure shows the Kalman Filter prediction for three different sensor scenarios:
# * Result based on **lidar and radar** measurements
# * using only **lidar** measurements
# * using only **radar** measurements

# %%
plt.figure(figsize=(16,12))

plt.subplot(1,3,1)
plot_map(df_all, ["groundtruth", "prediction"])
plt.title("Prediction based on Lidar & Radar")

plt.subplot(1,3,2)
plot_map(df_lidar, ["lidar", "prediction"])
plt.title("Lidar-only Prediction")

plt.subplot(1,3,3)
plot_map(df_radar, ["radar", "prediction"])
plt.title("Radar-only Prediction")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Velocity Profiles
#
# The following figure shown the corresponding speed profile of the cars

# %%
def plot_speed(df):
    for i, name in enumerate(np.unique(df.name)):
        c = plt.cm.tab10(i)
        df_i = df[df.name == name] # car's ukf measurements
        plt.plot(df_i.t, df_i.x_v, color=c, label=name)

        # Raw speed measurement in radar_phi direction, not projected to car direction x_jaw_angle:
        #plt.plot(df_i.t, df_i.radar_dr, color=c, label=f"{name} radar speed") 

        df_t = df_gt[df_gt.name == name] # car's ground truth
        plt.plot(df_t.t, df_t.v, color=c, alpha=0.5, label=f"{name} ground truth")
    plt.xlabel("t")
    plt.ylabel("m/s")
    plt.grid()
    plt.legend()
    
plt.figure(figsize=(16,8))

plt.subplot(1,3,1)
plot_speed(df_all)
plt.title("Speed prediction based on Lidar & Radar")

plt.subplot(1,3,2)
plot_speed(df_lidar)
plt.title("Lidar-only Prediction")

plt.subplot(1,3,3)
plot_speed(df_radar)
plt.title("Radar-only Prediction")
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Notes
# * TODO: Fig (c)/blue: Radar-only UKF can't catch up to true initial speed of v_car1 = 5 m/s. Check state init + covariance!
#
# #### Check: Radar data car1

# %%
name = "car1"
c = plt.cm.tab10(0)

df_i = df_radar[df_radar.name == name] # car's ukf measurements
df_t = df_gt[df_gt.name == name] # car's ground truth

plt.figure(figsize=(10,10))
plt.title("Car1 speed measurements")
plt.plot(df_t.t, df_t.v, alpha=0.5, label=f"{name} ground truth")
plt.plot(df_i.t, df_i.radar_dr, label="car1 radar doppler")

# 
speed_x = df_i.radar_dr / np.cos(df_i.radar_phi)
speed_x = np.clip(speed_x, -7, 7);
plt.plot(df_i.t, speed_x, ":", label="car1 radar doppler_x")
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
    plt.hlines(threshold, xmin=0, xmax=max(series.index), color="red", label="95 %")
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
threshold = 7.815
#plot_nis(df_all[df_all.sensor_type=="radar"].nis_radar, "radar", threshold)
plot_nis(df_radar.nis_radar, "radar", threshold)

# %% [markdown]
# ### Lidar

# %%
threshold = 6
#plot_nis(df_all[df_all.sensor_type=="lidar"].nis_lidar, "lidar", threshold)
plot_nis(df_lidar.nis_lidar, "lidar", threshold)

# %%
