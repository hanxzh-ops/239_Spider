#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 16:09:11 2025

@author: jamesz
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# ===============================

import matplotlib
matplotlib.use("Agg")          # <--- FIX
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ===============================
# ===============================
# Load model
# ===============================
model = mujoco.MjModel.from_xml_path("hexapod.xml")
data = mujoco.MjData(model)

#
# BODY ID AND SENSOR IDS
torso_id = model.body("torso").id

# orientation sensor
quat_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_orientation")
# gyro sensor
gyro_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_gyro")

if quat_sid < 0:
    raise RuntimeError("Orientation sensor 'torso_orientation' not found.")

if gyro_sid < 0:
    raise RuntimeError("Gyro sensor 'torso_gyro' not found.")

#

# ==========================================================
# Default standing joint targets (from your specification)
# ==========================================================
FEMUR_DEFAULT = 1.31
TIBIA_DEFAULT = 1.52

# Coxa joint midpoints based on your FK section
COXA_DEFAULT = {
    "leg0_coxa_j": 0.5 * (0.0 + 1.57),
    "leg1_coxa_j": 0.5 * (-1.57 + 0.0),
    "leg2_coxa_j": 0.5 * (0.79 + 2.36),
    "leg3_coxa_j": 0.5 * (-0.79  -2.36),
    "leg4_coxa_j": 0.5 * (1.57 + 3.14),
    "leg5_coxa_j": 0.5 * (-3.14 + -1.57),
}

# Convenient table for indexing joints
FEMUR_JOINTS = [f"leg{i}_femur_j" for i in range(6)]
TIBIA_JOINTS = [f"leg{i}_tibia_j" for i in range(6)]
COXA_JOINTS  = [f"leg{i}_coxa_j" for i in range(6)]

# ==========================================================
# Helper: set PD target for any joint
# ==========================================================
def set_joint_angles(target_dict):
    """
    target_dict: {joint_name: angle}
    Properly maps joint_name -> actuator name in your XML.
    """
    for name, angle in target_dict.items():
        j_id = model.joint(name).id
        
        # FIX: actuator names do NOT have '_j'
        act_name = "act_" + name.replace("_j", "")
        a_id = model.actuator(act_name).id

        data.ctrl[a_id] = angle


# ==========================================================
# STANDING POSE
# ==========================================================
def stand_default():
    targets = {}
    # Coxa
    targets.update(COXA_DEFAULT)
    # Femur
    for j in FEMUR_JOINTS:
        targets[j] = FEMUR_DEFAULT
    # Tibia
    for j in TIBIA_JOINTS:
        targets[j] = TIBIA_DEFAULT
    set_joint_angles(targets)


# ==========================================================
# Simple walking trajectory generation (tripod gait)
# ==========================================================
TRIPOD_A = [0, 3, 4]  # swing group
TRIPOD_B = [1, 2, 5]  # stance group

def fk_step_joint_targets(phase):
    """
    Computes a proper tripod gait:
    - swing legs LIFT -> FORWARD -> DROP
    - stance legs push back
    phase ∈ [0,1]
    """

    # Tripod groups
    TRIPOD_A = [0, 3 ,4]
    TRIPOD_B = [1, 2, 5]

    # Swing amplitude parameters
    LIFT_FEMUR = 0.50     # rad leg lifts (bigger = higher clearance)
    LIFT_TIBIA = -0.35    # tibia tucks inward
    COXA_FORWARD = 0.55   # rad forward in swing
    COXA_BACKWARD = -0.35 # stance push

    # Determine which tripod is swinging
    if phase < 0.5:
        swing = TRIPOD_A
        stance = TRIPOD_B
        swing_phase = phase * 2.0   # normalize 0..0.5 -> 0..1
    else:
        swing = TRIPOD_B
        stance = TRIPOD_A
        swing_phase = (phase - 0.5) * 2.0

    targets = {}

    # Initialize everyone to default standing
    for i in range(6):
        targets[f"leg{i}_coxa_j"] = COXA_DEFAULT[f"leg{i}_coxa_j"]
        targets[f"leg{i}_femur_j"] = FEMUR_DEFAULT
        targets[f"leg{i}_tibia_j"] = TIBIA_DEFAULT

    # ===========================
    # SWING LEG TRAJECTORY
    # ===========================
    # Lift & lower trajectory using smooth parabola
    lift = np.sin(np.pi * swing_phase)              # 0→1→0
    forward = np.sin(swing_phase * np.pi)           # same smoothing

    for leg in swing:
        # Lift leg up
        targets[f"leg{leg}_femur_j"] = (
            FEMUR_DEFAULT - LIFT_FEMUR * lift
        )

        # Tibia folds inward slightly during lift
        targets[f"leg{leg}_tibia_j"] = (
            TIBIA_DEFAULT + LIFT_TIBIA * lift
        )

        # Coxa moves leg forward while airborne
        sign = -1 if leg in [2, 3] else 1
        sign2 = -1 if leg in [0, 3, 4] else 1
        targets[f"leg{leg}_coxa_j"] = (
            COXA_DEFAULT[f"leg{leg}_coxa_j"] + sign2 * sign * COXA_FORWARD * forward
        )

    # ===========================
    # STANCE LEG TRAJECTORY
    # ===========================
    for leg in stance:
        # Coxa pushes backward (body moves forward)
        sign = -1 if leg in [2, 3] else 1
        sign2 = -1 if leg in [0, 3, 4] else 1
        targets[f"leg{leg}_coxa_j"] = (
            COXA_DEFAULT[f"leg{leg}_coxa_j"] +sign2 * sign * COXA_BACKWARD
        )
        # femur/tibia remain locked for support

    return targets



# ==========================================================
# MAIN VIEWER LOOP
# ==========================================================
with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Standing...")
    stand_default()

    # let it settle
    for _ in range(300):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    print("Walking + IMU Logging (10 seconds)...")
    T = 1.5
    t0 = time.time()

    # storage buffers
    times = []
    roll_log = []
    pitch_log = []
    
    yaw_log = []
    gyro_log = []

    # simulate exactly 10 seconds
    while viewer.is_running():
        t = time.time() - t0
        if t > 10.0:
            break

        phase = (t % T) / T

        targets = fk_step_joint_targets(phase)
        
        set_joint_angles(targets)

        # -------- IMU READING ---------
        # orientation is stored in data.sensordata
        quat = data.sensordata[quat_sid * 4 : quat_sid * 4 + 4]
        # convert from (w,x,y,z) to scipy order (x,y,z,w)
        qw, qx, qy, qz = quat
        
        rot = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rot.as_euler("xyz", degrees=True)

        # gyro (angular velocity)
        gx = data.sensordata[gyro_sid * 3 + 0]
        
        gy = data.sensordata[gyro_sid * 3 + 1]
        gz = data.sensordata[gyro_sid * 3 + 2]

        # store logs
        times.append(t)
        roll_log.append(roll)
        pitch_log.append(pitch)
        yaw_log.append(yaw)
        gyro_log.append([gx, gy, gz])

        # --------------------------------

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

print("Logging complete.")

# ==========================================================
# PLOT IMU ORIENTATION
# ==========================================================
plt.figure(figsize=(12, 6))
plt.plot(times, roll_log,  label="Roll (deg)")
plt.plot(times, pitch_log, label="Pitch (deg)")
plt.plot(times, yaw_log,   label="Yaw (deg)")
plt.xlabel("Time (s)")
plt.ylabel("Rotation (degrees)")
plt.title("Torso Orientation (IMU: framequat) During Walking")
plt.grid(True)
plt.legend()
plt.savefig("imu_orientation_test1.png")
plt.close()

# gyro plot

# ==========================================================
# OPTIONAL: GYRO PLOT
# ==========================================================
gyro_arr = np.array(gyro_log)
plt.figure(figsize=(12, 6))
plt.plot(times, gyro_arr[:,0], label="Gyro X")
plt.plot(times, gyro_arr[:,1], label="Gyro Y")
plt.plot(times, gyro_arr[:,2], label="Gyro Z")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Torso Angular Velocity (IMU Gyro) During Walking")
plt.grid(True)
plt.legend()
plt.savefig("imu_gyro_test1.png")
plt.close()


'''
    while viewer.is_running():
        t = time.time() - t0
        phase = (t % T) / T

        targets = fk_step_joint_targets(phase)
        set_joint_angles(targets)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
'''