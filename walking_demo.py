#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 01:15:58 2025

@author: jamesz
"""

import os
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

from src.mj_interface import MujocoInterface
from assets.config import DEFAULT_JOINT_POS
from src.fk import fk_all
from src.ik import ik_leg_from_foot_torso

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "assets", "hexapod.xml")

# ---------------------------------------------------------------------
# Gait parameters
# ---------------------------------------------------------------------
STEP_FREQ   = 1.2      # Hz (gait cycles per second)
STEP_LENGTH = 0.08     # meters (peak-to-peak forward travel in body frame)
STEP_HEIGHT = 0.03     # meters (swing clearance)
DUTY_FACTOR = 0.5      # fraction of cycle spent in stance
CONTROL_DT  = 0.016    # 60 Hz

# Tripod pattern: even vs odd legs
TRIPOD_A = [0, 2, 4]
TRIPOD_B = [1, 3, 5]

# ---------------------------------------------------------------------
# Foot trajectory in torso frame
# ---------------------------------------------------------------------
def foot_trajectory_torso(phase, stance_pos, step_len=STEP_LENGTH, step_h=STEP_HEIGHT, duty=DUTY_FACTOR):
    """
    phase in [0, 1)
    stance_pos: np.array([x0, y0, z0]) baseline foot pos in torso frame (from FK standing)
    Returns desired foot position in torso frame.
    """
    x0, y0, z0 = stance_pos

    if phase < duty:
        # STANCE: foot on ground, moves backward relative to body
        s = phase / duty  # 0..1
        dx = (0.5 - s) * step_len   # +step/2 -> -step/2
        dz = 0.0
    else:
        # SWING: lifted off ground, moves forward
        s = (phase - duty) / (1.0 - duty)  # 0..1
        dx = (-0.5 + s) * step_len         # -step/2 -> +step/2

        # Determine lift direction: towards torso (opposite of stance z sign)
        sign_z = -1.0 if z0 < 0.0 else 1.0
        dz = sign_z * step_h * math.sin(math.pi * s)

    return np.array([x0 + dx, y0, z0 + dz])


def main():
    if not os.path.exists(XML_PATH):
        print("XML not found:", XML_PATH)
        return

    # --------------------------------------------------------------
    # 1) Initialize MuJoCo interface
    # --------------------------------------------------------------
    mj = MujocoInterface(XML_PATH)

    # --------------------------------------------------------------
    # 2) Bring robot to default standing pose (femur=1.31, tibia=1.52, coxa mid)
    # --------------------------------------------------------------
    print("[INIT] Moving to standing pose...")
    for _ in range(200):
        mj.set_joint_targets(DEFAULT_JOINT_POS)
        mj.step()

    # --------------------------------------------------------------
    # 3) Compute stance foot positions in torso frame via FK
    #    These are the "baseline" positions around which gait cycles.
    # --------------------------------------------------------------
    stance_feet_torso = fk_all(DEFAULT_JOINT_POS)  # {leg: np.array([x,y,z])}
    print("[INIT] Stance foot positions (torso frame):")
    for leg, p in stance_feet_torso.items():
        print(f"  Leg {leg}: {p}")

    # --------------------------------------------------------------
    # 4) Start tripod walking gait
    # --------------------------------------------------------------
    start_time = time.time()
    print("\n[WALK] Starting tripod walking. Close the viewer to stop.\n")

    with mujoco.viewer.launch_passive(mj.model, mj.data) as viewer:
        while viewer.is_running():
            t = time.time() - start_time
            cycle_phase = (t * STEP_FREQ) % 1.0  # 0..1

            # For IK in torso frame, we assume torso orientation ~ identity on average.
            # If you want to support large torso rotations, you can read pose each step:
            # torso_pos, torso_quat = mj.get_body_pose("torso")
            # and convert desired foot positions to world then IK in world.
            # For now we stay in torso frame for clarity.

            joint_targets = {}

            for leg in range(6):
                # Tripod phase
                phase_offset = 0.0 if (leg in TRIPOD_A) else 0.5
                leg_phase = (cycle_phase + phase_offset) % 1.0

                # Stance baseline from FK
                stance_pos = stance_feet_torso[leg]

                # Desired foot position in torso frame
                foot_des_torso = foot_trajectory_torso(
                    leg_phase,
                    stance_pos,
                    step_len=STEP_LENGTH,
                    step_h=STEP_HEIGHT,
                    duty=DUTY_FACTOR
                )

                # IK: torso frame target → leg joint angles
                ok, res = ik_leg_from_foot_torso(foot_des_torso, leg_index=leg, elbow_up=True, clamp=True)
                if not ok:
                    print(f"[IK ERROR] leg {leg}: {res}")
                    continue

                # res is dict {"legX_coxa_j": angle, ...}
                joint_targets.update(res)

            # Apply joint targets via actuators
            mj.set_joint_targets(joint_targets)

            # Step physics
            mj.step()
            viewer.sync()

            # basic timing to approximate CONTROL_DT
            elapsed = time.time() - (start_time + t)
            time.sleep(max(0.0, CONTROL_DT - elapsed))


if __name__ == "__main__":
    main()
