#!/usr/bin/env python3
# walk_mujoco.py
# Author: Henry

import os
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

from src.robot import SpiderRobot   

# robot parameters
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "assets", "hexapod.xml")

# Leg geometry change if needed
# Len1 = 53; Len2=80; Len3=144;  meter as unit
L1 = 0.025
L2 = 0.04
L3 = 0.05

# Gait parameters
STEP_FREQ = 1.2           # Hz (gait cycles per second)
STEP_LENGTH = 0.08        # meters (total forward travel of foot relative to body)
STEP_HEIGHT = 0.03        # meters (swing foot clearance)
DUTY_FACTOR = 0.5         # fraction of cycle spent in support (tripod often ~0.5)
CONTROL_DT = 0.016        # 60 Hz

# Leg indexing convention: legs 0..5. use tripod as even vs odd legs:
TRIPOD_A = [0, 2, 4]     # even indexes
TRIPOD_B = [1, 3, 5]     # odd indexes

# Default foot positions in body frame (stationary stand pose) - change if needed
# (x, y, z) positions of foot when robot stands (body frame).
STANCE_POSITIONS = {
    0: np.array([ 0.10,  0.07, -0.18]),
    1: np.array([ 0.10, -0.07, -0.18]),
    2: np.array([ 0.0,   0.12, -0.18]),
    3: np.array([ 0.0,  -0.12, -0.18]),
    4: np.array([-0.10,  0.07, -0.18]),
    5: np.array([-0.10, -0.07, -0.18]),
}

# Actuator names, change if needed
ACT_NAMES = {
    'coxa': lambda i: f"act_leg{i}_coxa",
    'femur': lambda i: f"act_leg{i}_femur",
    'tibia': lambda i: f"act_leg{i}_tibia",
}

# math: IK

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def inverse_kinematics_leg(x, y, z):
    """
    Closed-form IK for the 3-DOF leg.
    Returns theta1 (coxa yaw), theta2 (femur pitch), theta3 (tibia pitch).
    Assumes leg base coordinate: x forward, y lateral, z up (consistent with doc).
    """
    # theta1 = atan2(y, x)
    theta1 = math.atan2(y, x)

    # R = sqrt(x^2 + y^2)
    R = math.hypot(x, y)

    # Effective planar distance from 'coxa pivot offset L1' to foot
    # Compute Lr:
    # alphaR = atan(-z / (R - L1))
    denomR = (R - L1)
    # avoid singular
    if abs(denomR) < 1e-8:
        denomR = 1e-8 if denomR >= 0 else -1e-8
    alphaR = math.atan2(-z, denomR)
    Lr = math.hypot(z, denomR)

    # law-of-cosines for alpha1
    # clamp cosine to [-1,1] to avoid numeric errors
    cos_alpha1 = clamp((L2*L2 + Lr*Lr - L3*L3) / (2.0 * L2 * Lr), -1.0, 1.0)
    alpha1 = math.acos(cos_alpha1)

    theta2 = alpha1 - alphaR

    # alpha2 for tibia
    cos_alpha2 = clamp((Lr*Lr + L3*L3 - L2*L2) / (2.0 * Lr * L3), -1.0, 1.0)
    alpha2 = math.acos(cos_alpha2)
    theta3 = alpha1 + alpha2

    # Note: depending on joint sign conventions, may need to flip signs for theta2/theta3.
    return theta1, theta2, theta3

def foot_trajectory_straight(t_phase, step_len=STEP_LENGTH, step_h=STEP_HEIGHT, duty=DUTY_FACTOR):
    """
    Given local phase in [0,1), returns foot position along x (forward) and z (vertical).
    keep lateral (y) constant at stance baseline.
    Support phase: foot on ground translating backward from +step/2 -> -step/2 (relative to body).
    Swing phase: foot moves from -step/2 -> +step/2 with z-lift (sin shape).
    """
    if t_phase < duty:
        # SUPPORT: map 0..duty -> +step/2 .. -step/2
        s = t_phase / duty
        x = (0.5 - s) * step_len
        z = -0.12
    else:
        # SWING: map duty..1 -> -step/2 .. +step/2 with sinusoidal lift
        s = (t_phase - duty) / (1.0 - duty)  # 0..1
        x = (-0.5 + s) * step_len
        z = step_h * math.sin(math.pi * s)   # smooth lift
    return x, z


# Controller
def build_actuator_id_map(model):
    """Return dict: (leg_idx, 'coxa'/'femur'/'tibia') -> actuator_id"""
    id_map = {}
    for i in range(6):
        for part in ['coxa', 'femur', 'tibia']:
            name = ACT_NAMES[part](i)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                raise RuntimeError(f"Actuator '{name}' not found in model. Update ACT_NAMES mapping.")
            id_map[(i, part)] = aid
    return id_map

def apply_joint_targets(model, data, act_map, joint_angles):
    """
    joint_angles: dict of (leg_idx) -> (theta1, theta2, theta3) in radians
    This writes into data.ctrl by mapping desired angle into actuator ctrl range.
    Assumes each actuator directly controls the joint angle in radians.
    """
    for i, (th1, th2, th3) in joint_angles.items():
        for part, theta in [('coxa', th1), ('femur', th2), ('tibia', th3)]:
            aid = act_map[(i, part)]
            lo, hi = model.actuator_ctrlrange[aid]
            # Clip angle to actuator range (assumes ctrlrange specified in radians)
            val = clamp(theta, lo, hi)
            data.ctrl[aid] = val

# Main loop
def main():
    if not os.path.exists(XML_PATH):
        print("XML not found:", XML_PATH)
        return

    bot = SpiderRobot(XML_PATH)   
    model = bot.model
    data = bot.data

    # Build actuator id map
    act_map = build_actuator_id_map(model)

    # Precompute leg order array for convenience
    legs = [0,1,2,3,4,5]

    start_time = time.time()

    # Choose base body translation - keep body stationary relative to world; gait moves feet.
    body_pos = np.array([0.0, 0.0, 0.0])

    print("Starting walking controller: tripod gait. Press ESC/close window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - start_time
            cycle_phase = (t * STEP_FREQ) % 1.0  # 0..1 within gait cycle

            # For tripod gait: even legs in phase A shifted by 0, odd legs shifted by 0.5
            joint_targets = {}

            for i in legs:
                # phase offset for this leg
                phase_offset = 0.0 if (i in TRIPOD_A) else 0.5
                leg_phase = (cycle_phase + phase_offset) % 1.0

                # compute foot desired displacement relative to stance
                dx_rel, dz_rel = foot_trajectory_straight(leg_phase, step_len=STEP_LENGTH, step_h=STEP_HEIGHT, duty=DUTY_FACTOR)

                # stance baseline
                stance = STANCE_POSITIONS[i]
                # target foot in body frame: add dx in robot forward (x) and vertical
                foot_body = np.array([stance[0] + dx_rel, stance[1], stance[2] + dz_rel])

                # Convert to leg-root coords if necessary.
                # assume leg-root frame (used by IK) aligns with body frame at leg mount.
                # if different offsets used, apply transforms here.

                # IK: get joint angles for desired foot_body
                th1, th2, th3 = inverse_kinematics_leg(foot_body[0], foot_body[1], foot_body[2])
                joint_targets[i] = (th1, 1.31, 1.52)

            # Write joint targets to actuators
            apply_joint_targets(model, data, act_map, joint_targets)

            # Step simulation (use bot.step() wrapper to be consistent)
            bot.step()
            viewer.sync()

            # sleep to keep ~CONTROL_DT
            time.sleep(max(0.0, CONTROL_DT - 0.001))

if __name__ == "__main__":
    main()
