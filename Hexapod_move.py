#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 18:39:13 2025

@author: jamesz
"""
import time
import itertools
import numpy as np
import mujoco
from mujoco import viewer

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "/Users/jamesz/Desktop/Hexapod.xml"  # Update with your actual path
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# -----------------------------
# Simple sinusoidal gait controller
# -----------------------------
def gait_controller(t, leg_count=6):
    """
    Generates target joint angles for a simple walking gait.
    Returns a dict: {joint_name: angle}
    """
    ctrl = {}
    amplitude = 0.5  # radians
    frequency = 1.0  # Hz

    for leg in range(leg_count):
        phase = (leg % 2) * np.pi  # alternate legs
        # Coxa joint rotates left/right
        ctrl[f"leg{leg}_coxa_j"] = amplitude * np.sin(2*np.pi*frequency*t + phase)
        # Femur joint rotates up/down
        ctrl[f"leg{leg}_femur_j"] = 0.5 + 0.2 * np.sin(2*np.pi*frequency*t + phase)
        # Tibia joint rotates
        ctrl[f"leg{leg}_tibia_j"] = 0.7 + 0.2 * np.sin(2*np.pi*frequency*t + phase)
    return ctrl

# -----------------------------
# Apply control safely
# -----------------------------
def apply_control(ctrl_dict):
    joint_names = []
    for j in range(model.njnt):
        start = j * model.names_map.max
        end = start + model.names_map.max
        name = bytes(model.names[start:end]).split(b'\x00', 1)[0].decode('utf-8')
        joint_names.append(name)
    for jname, angle in ctrl_dict.items():
        if jname in joint_names:
            idx = joint_names.index(jname)
            data.ctrl[idx] = angle

# -----------------------------
# Simulation parameters
# -----------------------------
SIM_TIME = 10.0  # seconds
DT = model.opt.timestep
t = 0.0

# -----------------------------
# Passive viewer setup (macOS compatible)
# -----------------------------
try:
    with viewer.launch_passive(model, data) as sim_viewer:
        # Optional: show wireframe of the robot
        sim_viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
        sim_viewer.sync()

        while sim_viewer.is_running() and t < SIM_TIME:
            step_start = time.time()

            # Generate gait and apply control
            ctrl = gait_controller(t)
            apply_control(ctrl)

            # Step the physics
            mujoco.mj_step(model, data)

            # Sync with viewer
            sim_viewer.sync()

            # Increment time
            t += DT

            # Simple wall-clock timing
            time_until_next_step = DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except Exception as e:
    print("Passive viewer could not launch. Running headless...")
    print(str(e))

    # Headless fallback
    t = 0.0
    while t < SIM_TIME:
        ctrl = gait_controller(t)
        apply_control(ctrl)
        mujoco.mj_step(model, data)
        t += DT

    print("Simulation finished (headless mode)")
