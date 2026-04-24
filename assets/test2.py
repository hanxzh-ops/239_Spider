#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 02:37:01 2025

@author: jamesz
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")          # <--- FIX
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ===============================
# Load model
# ===============================
model = mujoco.MjModel.from_xml_path("hexapod.xml")
data  = mujoco.MjData(model)

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
TRIPOD_A = [0,  3,4]  # swing group

TRIPOD_B = [1,  2,  5]  # stance group

def apply_body_stabilization(targets):
    """
    Add stabilizing offsets to femur, tibia, coxa joints
    to counteract roll, pitch, yaw motions during tripod gait.
    Uses only IMU-like signals from torso body in Mujoco.
    """

    # -----------------------------
    # 1. Read torso actual orientation
    # -----------------------------
    torso_id = model.body("torso").id
    quat = data.xquat[torso_id]        # w, x, y, z
    pos  = data.xpos[torso_id]         # world position
    # Convert quat to small-angle Euler
    # roll (x), pitch (y), yaw (z)
    qw, qx, qy, qz = quat
    roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = np.arcsin(2*(qw*qy - qz*qx))
    # yaw not needed for stabilization

    # -----------------------------
    # 2. Gains (tune only NUMBERS)
    # -----------------------------
    ROLL_GAIN  = 0.30     # lower = smoother, higher = stiffer
    PITCH_GAIN = 0.25
    HEIGHT_GAIN = 0.05     # compensates vertical drop during swing
    LATERAL_GAIN = 0.04    # compensates sideways roll drift

    # -----------------------------
    # 3. Compute corrections
    # -----------------------------
    femur_roll_corr  = -ROLL_GAIN  * roll
    femur_pitch_corr = -PITCH_GAIN * pitch

    # vertical drop compensation
    femur_height_corr = HEIGHT_GAIN * (pos[2] - 0.05)

    # lateral correction based on roll
    coxa_lat_corr = -LATERAL_GAIN * roll

    # -----------------------------
    # 4. Apply to joint targets
    # -----------------------------
    for leg in range(6):

        # femur (pitch control)
        fj = f"leg{leg}_femur_j"
        if fj in targets:
            targets[fj] += femur_roll_corr + femur_pitch_corr + femur_height_corr

        # coxa (lateral control)
        cj = f"leg{leg}_coxa_j"
        if cj in targets:
            # Left legs positive, right legs negative
            side = 1 if leg in [0,2,4] else -1
            targets[cj] += side * coxa_lat_corr

    return targets


def fk_step_joint_targets(phase):
    """
    Gait sequence (8-phase cycle):

    0.000–0.125  g1 lift
    0.125–0.250  g1 hold (pause)
    0.250–0.375  g2 back + g1 forward
    0.375–0.500  g1 drop
    0.500–0.625  g2 lift
    0.625–0.750  g2 hold (pause)
    0.750–0.875  g1 back + g2 forward
    0.875–1.000  reset to start
    """

    # Tripods
    g1 = [0, 3,  4]
    
    g2 = [1,2  , 5]
    

    # Parameters
    LIFT_FEMUR = 0.45
    LIFT_TIBIA  = -0.30
    COXA_FWD    = 0.55
    COXA_BACK   = -0.35

    targets = {}

    # Start with default pose
    for i in range(6):
        targets[f"leg{i}_coxa_j"]  = COXA_DEFAULT[f"leg{i}_coxa_j"]
        targets[f"leg{i}_femur_j"] = FEMUR_DEFAULT
        targets[f"leg{i}_tibia_j"] = TIBIA_DEFAULT

    # --------------------------
    # Helper actions
    # --------------------------
    def lift_leg(leg):
        targets[f"leg{leg}_femur_j"] = FEMUR_DEFAULT - LIFT_FEMUR
        targets[f"leg{leg}_tibia_j"] = TIBIA_DEFAULT + LIFT_TIBIA

    def drop_leg(leg):
        targets[f"leg{leg}_femur_j"] = FEMUR_DEFAULT
        targets[f"leg{leg}_tibia_j"] = TIBIA_DEFAULT

    def swing_forward(leg):
        # include your sign flip for leg2 & 3
        sign = -1 if leg in [2,3] else 1
        sign2 = -1 if leg in [0,3,4] else 1
        targets[f"leg{leg}_coxa_j"] = (
            COXA_DEFAULT[f"leg{leg}_coxa_j"] + sign2 * sign * COXA_FWD
        )

    def swing_back(leg):
        sign = -1 if leg in [2,3] else 1
        sign2 = -1 if leg in [0,3,4] else 1
        targets[f"leg{leg}_coxa_j"] = (
            COXA_DEFAULT[f"leg{leg}_coxa_j"] + sign2 * sign * COXA_BACK
        )

    # --------------------------
    # Phase slice
    # --------------------------
    if phase < 0.125:  # g1 lift
        for leg in g1:
            lift_leg(leg)

    elif phase < 0.250:  # g1 hold
        for leg in g1:
            lift_leg(leg)

    elif phase < 0.375:  # g2 back + g1 forward
    
        for leg in g2:
            swing_back(leg)
        for leg in g1:
            swing_forward(leg)

    elif phase < 0.500:  # g1 drop
        for leg in g1:
            drop_leg(leg)

    elif phase < 0.625:  # g2 lift
        for leg in g2:
            lift_leg(leg)

    elif phase < 0.750:  # g2 hold
        for leg in g2:
            lift_leg(leg)

    elif phase < 0.875:  # g1 back + g2 forward
        for leg in g1:
            swing_back(leg)
        for leg in g2:
            swing_forward(leg)

    else:  # final reset (slight pre-return)
        pass
    #targets = apply_body_stabilization(targets)

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
    #foot gait
    foot_x_log = [[] for _ in range(6)]
    foot_y_log = [[] for _ in range(6)]

    while viewer.is_running():
        t = time.time() - t0
        phase = (t % T) / T

        targets = fk_step_joint_targets(phase)
        set_joint_angles(targets)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
        for i in range(6):
            body_name = f"leg{i}_tibia"
        try:
            bid = model.body(body_name).id
            px, py, pz = data.xpos[bid]
            foot_x_log[i].append(px)
            foot_y_log[i].append(py)
        except:
        # If foot bodies are geoms instead of bodies, modify accordingly
            pass

        # Timestamp
        times.append(t)
        
print("Logging complete. Rendering XY gait plot...")

plt.figure(figsize=(8,8))

colors = ["red","blue","green","orange","purple","cyan"]

for i in range(6):
    plt.plot(foot_x_log[i], foot_y_log[i], color=colors[i], label=f"Leg {i}")

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Hexapod Foot Trajectories (XY View, 2D Gait Plot)")
plt.legend()
plt.axis("equal")

plt.savefig("gait_xy_plot.png", dpi=200)
print("Saved gait_xy_plot.png")



'''
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
plt.savefig("imu_orientation_nostab.png")
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
plt.savefig("imu_gyro_nostab.png")
plt.close()
'''