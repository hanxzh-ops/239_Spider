#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 17:16:16 2025

@author: jamesz
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# ===============================
# Load model
# ===============================
model = mujoco.MjModel.from_xml_path("hexapod.xml")
data  = mujoco.MjData(model)

# ==========================================================
# Default standing joint targets
# ==========================================================
FEMUR_DEFAULT = 1.31
TIBIA_DEFAULT = 1.52

COXA_DEFAULT = {
    "leg0_coxa_j": 0.5 * (0.0 + 1.57),
    "leg1_coxa_j": 0.5 * (-1.57 + 0.0),
    "leg2_coxa_j": 0.5 * (0.79  + 2.36),
    "leg3_coxa_j": 0.5 * (-0.79 - 2.36),
    "leg4_coxa_j": 0.5 * (1.57 + 3.14),
    "leg5_coxa_j": 0.5 * (-3.14 + -1.57),
}

FEMUR_JOINTS = [f"leg{i}_femur_j" for i in range(6)]
TIBIA_JOINTS = [f"leg{i}_tibia_j" for i in range(6)]
COXA_JOINTS  = [f"leg{i}_coxa_j"  for i in range(6)]

# ==========================================================
# Helper: map joint target to actuator
# ==========================================================
def set_joint_angles(target_dict):
    for name, angle in target_dict.items():
        act_name = "act_" + name.replace("_j", "")
        a_id = model.actuator(act_name).id
        data.ctrl[a_id] = angle

# ==========================================================
# Standing Pose
# ==========================================================
def stand_default():
    targets = {}
    targets.update(COXA_DEFAULT)
    for j in FEMUR_JOINTS:
        targets[j] = FEMUR_DEFAULT
    for j in TIBIA_JOINTS:
        targets[j] = TIBIA_DEFAULT
    set_joint_angles(targets)

# ==========================================================
# Tripods
# ==========================================================
LEFT_SIDE  = [0, 2, 4]   # left legs
RIGHT_SIDE = [1, 3, 5]   # right legs

# We keep your original tripod structure:
TRIPOD_A = [0, 3, 4]
TRIPOD_B = [1, 2, 5]

# ==========================================================
# Gait generator: Turning LEFT
# ==========================================================
def turning_gait(phase):
    """
    Pure turning gait (no forward push):
    Sequence (turning left):
    1. lift g2
    2. twist g1
    3. twist g2
    4. drop g2
    5. lift g1
    6. twist g2
    7. twist g1
    8. drop g1

    g1 and g2 both twist in SAME direction now.
    Mid legs 2 and 3 still twist opposite for geometry.
    """

    g1 = [0, 3, 4]    # tripod A
    g2 = [1, 2, 5]    # tripod B

    LIFT_FEMUR = 0.50
    LIFT_TIBIA = -0.30
    TWIST_AMP  = 0.50

    # -----------------------------------------------------
    # FIXED: g1 and g2 twist SAME direction (+1) for rotation
    # mid-legs 2 and 3 still flip sign
    # -----------------------------------------------------
    def twist_direction(leg):
        # turning LEFT twist baseline is +1
        base = +1

        # mid-legs flip sign
        if leg == 2:
            return base
        if leg == 3:
            return base

        return base

    # Prepare baseline
    targets = {}
    for i in range(6):
        targets[f"leg{i}_coxa_j"]  = COXA_DEFAULT[f"leg{i}_coxa_j"]
        targets[f"leg{i}_femur_j"] = FEMUR_DEFAULT
        targets[f"leg{i}_tibia_j"] = TIBIA_DEFAULT

    # 8-phase timing
    p8 = phase * 8.0

    # -----------------------------------------------------
    # PHASE 1: lift g2
    # -----------------------------------------------------
    if p8 < 1.0:
        for leg in g2:
            targets[f"leg{leg}_femur_j"] = FEMUR_DEFAULT - LIFT_FEMUR
            targets[f"leg{leg}_tibia_j"] = TIBIA_DEFAULT + LIFT_TIBIA

    # -----------------------------------------------------
    # PHASE 2: twist g1
    # -----------------------------------------------------
    elif p8 < 2.0:
        for leg in g1:
            s = twist_direction(leg)
            targets[f"leg{leg}_coxa_j"] = COXA_DEFAULT[f"leg{leg}_coxa_j"] + s * TWIST_AMP

    # -----------------------------------------------------
    # PHASE 3: twist g2
    # -----------------------------------------------------
    elif p8 < 3.0:
        for leg in g2:
            s = twist_direction(leg)
            targets[f"leg{leg}_coxa_j"] = COXA_DEFAULT[f"leg{leg}_coxa_j"] + s * TWIST_AMP

    # -----------------------------------------------------
    # PHASE 4: drop g2
    # -----------------------------------------------------
    elif p8 < 4.0:
        for leg in g2:
            targets[f"leg{leg}_femur_j"] = FEMUR_DEFAULT
            targets[f"leg{leg}_tibia_j"] = TIBIA_DEFAULT

    # -----------------------------------------------------
    # PHASE 5: lift g1
    # -----------------------------------------------------
    elif p8 < 5.0:
        for leg in g1:
            targets[f"leg{leg}_femur_j"] = FEMUR_DEFAULT - LIFT_FEMUR
            targets[f"leg{leg}_tibia_j"] = TIBIA_DEFAULT + LIFT_TIBIA

    # -----------------------------------------------------
    # PHASE 6: twist g2
    # -----------------------------------------------------
    elif p8 < 6.0:
        for leg in g2:
            s = twist_direction(leg)
            targets[f"leg{leg}_coxa_j"] = COXA_DEFAULT[f"leg{leg}_coxa_j"] + s * TWIST_AMP

    # -----------------------------------------------------
    # PHASE 7: twist g1
    # -----------------------------------------------------
    elif p8 < 7.0:
        for leg in g1:
            s = twist_direction(leg)
            targets[f"leg{leg}_coxa_j"] = COXA_DEFAULT[f"leg{leg}_coxa_j"] + s * TWIST_AMP

    # -----------------------------------------------------
    # PHASE 8: drop g1
    # -----------------------------------------------------
    else:
        for leg in g1:
            targets[f"leg{leg}_femur_j"] = FEMUR_DEFAULT
            targets[f"leg{leg}_tibia_j"] = TIBIA_DEFAULT

    return targets


# ==========================================================
# MAIN LOOP → TURNING DEMO
# ==========================================================
with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Standing...")
    stand_default()

    for _ in range(200):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    print("Turning left...")
    T = 3
    t0 = time.time()

    while viewer.is_running():
        t = time.time() - t0
        phase = (t % T) / T

        targets = turning_gait(phase)
        set_joint_angles(targets)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
