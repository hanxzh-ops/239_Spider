#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:42:58 2025

@author: jamesz
"""

import numpy as np
from numpy import sin, cos

from assets.config import (
    LEG_BASE_POS,
    LINK_LENGTHS,
    JOINTS,
)

# ============================================================
# Rotation helpers
# ============================================================

def rot_z(theta):
    c, s = cos(theta), sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def rot_y(theta):
    c, s = cos(theta), sin(theta)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c]
    ])

# ============================================================
# Per-leg forward kinematics
# ============================================================

def fk_leg(coxa, femur, tibia, leg_index):
    """
    Compute foot position in the *torso frame* for one leg.

    Kinematic chain (matching hexapod.xml):
        torso
          └─ leg_base (rigid, at LEG_BASE_POS)
               └─ coxa joint (rot_z)
                    └─ [d_c along local +X]  ← coxa segment (stub)
                         └─ femur joint (rot_y)
                              └─ [d_f along local +Z]
                                   └─ tibia joint (rot_y)
                                        └─ [d_t along local +Z]
                                             └─ foot tip

    coxa, femur, tibia : joint angles (radians)
    leg_index          : int in 0..5

    Returns:
        foot_pos_torso : np.array([x,y,z])
    """
    base = LEG_BASE_POS[leg_index]

    d_c = LINK_LENGTHS["coxa"]    # 0.025 m — coxa stub (along coxa local-X)
    d_f = LINK_LENGTHS["femur"]   # 0.100 m — femur segment (along femur local-Z)
    d_t = LINK_LENGTHS["tibia"]   # 0.100 m — tibia segment (along tibia local-Z)

    R1 = rot_z(coxa)    # coxa  : yaw about torso Z
    R2 = rot_y(femur)   # femur : pitch about femur Y
    R3 = rot_y(tibia)   # tibia : pitch about tibia Y

    # Coxa stub: femur joint is d_c along the coxa body's local +X
    coxa_offset          = np.array([d_c, 0.0, 0.0])

    # Femur-to-tibia joint offset in femur frame
    tibia_joint_in_femur = np.array([0.0, 0.0, d_f])

    # Foot tip in tibia frame
    tip_in_tibia         = np.array([0.0, 0.0, d_t])

    # Full tip vector in the coxa body frame (after coxa rotation):
    #   coxa_offset  +  R2 @ (R3 @ tip_in_tibia + tibia_joint_in_femur)
    tip_in_coxa = coxa_offset + R2 @ (R3 @ tip_in_tibia + tibia_joint_in_femur)

    # Rotate into torso frame and add leg base position
    foot_pos_torso = base + R1 @ tip_in_coxa

    return foot_pos_torso


# ============================================================
# Batch FK for all legs given a joint dict
# ============================================================

def fk_all(joint_angles):
    """
    Compute foot positions for all 6 legs.
    
    joint_angles: dict { "leg0_coxa_j": val, ... }

    Returns:
        feet: dict { leg: np.array([x,y,z]) } in torso frame.
    """
    feet = {}

    for leg in range(6):
        coxa  = joint_angles[f"leg{leg}_coxa_j"]
        femur = joint_angles[f"leg{leg}_femur_j"]
        tibia = joint_angles[f"leg{leg}_tibia_j"]

        feet[leg] = fk_leg(coxa, femur, tibia, leg)

    return feet


# ============================================================
# Transform feet from torso frame to world frame
# Supports walking with nonzero torso pose
# ============================================================

def feet_in_world(feet_torso, torso_pos, torso_quat):
    """
    Convert torso-frame foot positions into world-frame positions.

    torso_pos  : np.array([x,y,z])
    torso_quat : wxyz quaternion (MuJoCo format)

    Returns:
        feet_world: { leg: np.array([x,y,z]) }
    """

    # Convert quaternion to rotation matrix
    w, x, y, z = torso_quat
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),    2*(y*z - w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])

    feet_world = {}

    for leg, p in feet_torso.items():
        feet_world[leg] = torso_pos + R @ p

    return feet_world
