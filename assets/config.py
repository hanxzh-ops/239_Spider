#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 18:12:15 2025

@author: jamesz
"""
import numpy as np

# ============================================================
#  JOINT NAMES (exactly as in hexapod.xml)
# ============================================================

JOINTS = [
    "leg0_coxa_j", "leg0_femur_j", "leg0_tibia_j",
    "leg1_coxa_j", "leg1_femur_j", "leg1_tibia_j",
    "leg2_coxa_j", "leg2_femur_j", "leg2_tibia_j",
    "leg3_coxa_j", "leg3_femur_j", "leg3_tibia_j",
    "leg4_coxa_j", "leg4_femur_j", "leg4_tibia_j",
    "leg5_coxa_j", "leg5_femur_j", "leg5_tibia_j",
]

# ============================================================
# ACTUATOR NAMES (derived from XML)
# ============================================================

ACTUATOR_MAP = {j: "act_" + j.replace("_j", "") for j in JOINTS}

# ============================================================
#  LEG BASE LOCATIONS (from XML <body name="legX_base" pos="x y z">)
#  All are relative to torso frame ("torso" body)
# ============================================================

LEG_BASE_POS = {
    0: np.array([ 0.10,  0.05, 0.0]),
    1: np.array([ 0.10, -0.05, 0.0]),
    2: np.array([ 0.00,  0.05, 0.0]),
    3: np.array([ 0.00, -0.05, 0.0]),
    4: np.array([-0.10,  0.05, 0.0]),
    5: np.array([-0.10, -0.05, 0.0]),
}

# ============================================================
#  JOINT RANGES (from XML <joint range="a b">)
# ============================================================

JOINT_RANGES = {
    # ------- Coxa -------
    "leg0_coxa_j": (0.0,  1.57),
    "leg1_coxa_j": (-1.57, 0.0),
    "leg2_coxa_j": (0.79,  2.36),
    "leg3_coxa_j": (-2.36, -0.79),
    "leg4_coxa_j": (1.57, 3.14),
    "leg5_coxa_j": (-3.14, -1.57),

    # ------- Femur -------
    "leg0_femur_j": (0.52, 1.31),
    "leg1_femur_j": (0.52, 1.31),
    "leg2_femur_j": (0.52, 1.31),
    "leg3_femur_j": (0.52, 1.31),
    "leg4_femur_j": (0.52, 1.31),
    "leg5_femur_j": (0.52, 1.31),

    # ------- Tibia -------
    "leg0_tibia_j": (0.26, 1.83),
    "leg1_tibia_j": (0.26, 1.83),
    "leg2_tibia_j": (0.26, 1.83),
    "leg3_tibia_j": (0.26, 1.83),
    "leg4_tibia_j": (0.26, 1.83),
    "leg5_tibia_j": (0.26, 1.83),
}

# ============================================================
#  DEFAULT STANDING POSE (from your instructions)
# ============================================================

FEMUR_DEFAULT  = 1.31        # rad
TIBIA_DEFAULT  = 1.52        # rad

# Coxa default = midpoint of XML ctrlrange
COXA_DEFAULT = {
    0: 0.5 * (0.0 + 1.57),
    1: 0.5 * (-1.57 + 0.0),
    2: 0.5 * (0.79 + 2.36),
    3: 0.5 * (-2.36 + -0.79),
    4: 0.5 * (1.57 + 3.14),
    5: 0.5 * (-3.14 + -1.57),
}

# Build a single dict for easy controller usage
DEFAULT_JOINT_POS = {
    **{f"leg{i}_coxa_j":  COXA_DEFAULT[i] for i in range(6)},
    **{f"leg{i}_femur_j": FEMUR_DEFAULT  for i in range(6)},
    **{f"leg{i}_tibia_j": TIBIA_DEFAULT  for i in range(6)},
}

# ============================================================
#  LINK LENGTHS (inferred directly from XML)
#  femur: <body pos="0 0 0"> then femur geom at pos="0 0 0.05"
#  tibia: <body pos="0 0 0.1"> then tibia geom at pos="0 0 0.05"
# ============================================================

LINK_LENGTHS = {
    # All lengths in metres, matching hexapod.xml body offsets.
    #
    # Coxa:  femur body is at pos="0.025 0 0" in the coxa body frame,
    #        i.e. the femur joint is 25 mm along the coxa's local +X axis.
    #        This is the short lateral stub seen on real servo hexapods
    #        (e.g. Lynxmotion Phoenix, Trossen WidowX).
    "coxa":  0.025,  # coxa joint → femur joint,  along coxa local-X (m)
    "femur": 0.100,  # femur joint → tibia joint, along femur local-Z (m)
    "tibia": 0.100,  # tibia joint → foot tip,    along tibia local-Z (m)
}

# ============================================================
#  GAIT PARAMETERS (basic, safe defaults)
# ============================================================

GAIT = {
    "frequency": 1.0,      # Hz
    "lift_height": 0.04,   # meters (vertical lift of swing leg)
    "step_length": 0.05,   # meters forward
    "swing_percent": 0.5,  # % of cycle in swing
}

# Tripod groups (static from XML leg ordering)
TRIPOD_A = [0, 3, 4]
TRIPOD_B = [1, 2, 5]

