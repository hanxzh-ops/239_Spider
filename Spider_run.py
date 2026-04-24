#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 17:17:56 2025

@author: jamesz
"""
"""
run_hexapod.py
Simple controller for the provided hexapod XML using mujoco_py.
Tripod gait with sinusoidal position targets applied to position actuators.

Requirements:
- mujoco_py
- mujoco (MuJoCo binary properly installed and MJKEY or license set up)
- numpy

Run:
python run_hexapod.py
"""

"""
run_hexapod_fk.py
Hexapod robot control with forward kinematics–style joint configuration.
Uses actuator ctrlranges to compute percentage-based target positions.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "hexapod.xml"   # path to your model


def percent_to_value(ctrlrange, percent):
    """Map percentage [0,1] → actuator control value."""
    lo, hi = ctrlrange
    return lo + percent * (hi - lo)


def set_actuators(data, model, actuator_idxs, percents):
    for idx, pct in zip(actuator_idxs, percents):
        lo, hi = model.actuator_ctrlrange[idx]
        data.ctrl[idx] = percent_to_value((lo, hi), pct)

def mj_name_from_ptr(model, adr):
    """Return a MuJoCo name string starting at byte index adr."""
    # model.names is a bytes object containing all names separated by '\x00'
    end = model.names.find(b'\x00', adr)   # find null terminator
    return model.names[adr:end].decode('utf-8')


def step_wait(model, data, viewer, duration, dt=0.002):
    steps = int(duration / dt)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
def get_actuator_names(model):
    names = []
    for i in range(model.nu):
        adr = model.name_actuatoradr[i]
        names.append(mj_name_from_ptr(model, adr))
    return names


def main():
    
    # ----------------- Load Model -------------------
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)


    with mujoco.viewer.launch_passive(model, data) as viewer:

        actuator_names = [model.names[id] for id in model.name_actuatoradr]
        
        actuator_names = get_actuator_names(model)
        print("Actuators in model:")
        for i, name in enumerate(actuator_names):
            print(i, name)
        legs = 6

        # get actuator indices
        coxa_idxs  = [actuator_names.index(f"act_leg{i}_coxa")  for i in range(legs)]
        femur_idxs = [actuator_names.index(f"act_leg{i}_femur") for i in range(legs)]
        tibia_idxs = [actuator_names.index(f"act_leg{i}_tibia") for i in range(legs)]

        print("Starting simulation...")

        # -------------- 1. Default Standing Pose ----------------
        for i in range(legs):
            data.ctrl[coxa_idxs[i]]  = percent_to_value(model.actuator_ctrlrange[coxa_idxs[i]], 0.5)
            data.ctrl[femur_idxs[i]] = percent_to_value(model.actuator_ctrlrange[femur_idxs[i]], 0.85)
            data.ctrl[tibia_idxs[i]] = percent_to_value(model.actuator_ctrlrange[tibia_idxs[i]], 0.85)

        step_wait(model, data, viewer, 2.0)

        # -------------- 2. Femur to 100% ------------------------
        for i in range(legs):
            data.ctrl[femur_idxs[i]] = percent_to_value(model.actuator_ctrlrange[femur_idxs[i]], 1.0)

        step_wait(model, data, viewer, 1.5)

        # -------------- 3. Return Femur to 85% ------------------
        for i in range(legs):
            data.ctrl[femur_idxs[i]] = percent_to_value(model.actuator_ctrlrange[femur_idxs[i]], 0.85)

        step_wait(model, data, viewer, 1.5)

        # -------------- 4. Lift legs 0, 3, 4 --------------------
        lift_legs = [0, 3, 4]
        for leg in lift_legs:
            data.ctrl[femur_idxs[leg]] = percent_to_value(model.actuator_ctrlrange[femur_idxs[leg]], 0.0)

        step_wait(model, data, viewer, 2.0)

        print("Simulation complete — viewer still active.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

    print("Actuators in model:")
    
    for i, name in enumerate(model.actuator_names):
        print(i, name)
if __name__ == "__main__":
    main()
