#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 23:20:50 2025

@author: jamesz
"""

import numpy as np
from src.config import COXA_LEN, FEMUR_LEN, TIBIA_LEN, LEG_SCALERS, COXA_LEN

# --- MATH CORE ---

def solve_leg_ik(x, y, z, leg_idx):
    """
    Calculates angles for a single leg based on the user-specified FK formula:
    Z_tip = -( cos(f)*L2 - cos(pi - f - t)*L3 )
    """
    # 1. Coxa (Yaw)
    theta_1 = np.arctan2(y, x)

    # 2. Planar Reach (R) & Vertical Depth (D)
    planar_dist = np.sqrt(x**2 + y**2)
    R = planar_dist - COXA_LEN 
    D = -z  # Target height (positive value)

    # Solve 2D Triangle in R-D plane
    H_sq = R**2 + D**2
    H = np.sqrt(H_sq)
    
    # Clamp reach
    max_reach = FEMUR_LEN + TIBIA_LEN - 0.001
    if H > max_reach: H = max_reach

    # Angle of vector H relative to R-axis
    phi = np.arctan2(D, R) 
    
    # Law of Cosines for Alpha (Femur internal)
    arg_alpha = (FEMUR_LEN**2 + H**2 - TIBIA_LEN**2) / (2 * FEMUR_LEN * H)
    alpha = np.arccos(np.clip(arg_alpha, -1.0, 1.0))
    
    # Theta 2 (Femur)
    theta_2 = phi + alpha

    # Law of Cosines for Gamma (Knee internal)
    arg_gamma = (FEMUR_LEN**2 + TIBIA_LEN**2 - H**2) / (2 * FEMUR_LEN * TIBIA_LEN)
    gamma = np.arccos(np.clip(arg_gamma, -1.0, 1.0))
    
    # Theta 3 (Tibia) - Geometric Match: theta_3 = pi - gamma
    theta_3 = np.pi - gamma 

    # --- Apply Direction Scaling ---
    angles = np.array([theta_1, theta_2, theta_3])
    angles *= LEG_SCALERS[leg_idx]

    return angles

def get_ik_posture(body_z):
    """
    Returns 6x3 angles for the entire robot to stand at height body_z.
    """
    all_angles = []
    foot_z_rel = -body_z
    
    for i in range(6):
        # Target: Reach 0.15m out, Down 'body_z'
        target_x = 0.15 
        target_y = 0.0
        target_z = foot_z_rel

        angles = solve_leg_ik(target_x, target_y, target_z, i)
        all_angles.append(angles)

    return np.array(all_angles)

def calculate_height_from_angles(theta_f, theta_t):
    """
    Reverses the User's FK Formula to find current height.
    Formula: Z_tip = -( cos(f)*L2 - cos(pi - f - t)*L3 )
    Height = -Z_tip
    """
    # term1 = cos(femur) * L2
    term1 = np.cos(theta_f) * FEMUR_LEN
    
    # term2 = cos(pi - femur - tibia) * L3
    term2 = np.cos(np.pi - theta_f - theta_t) * TIBIA_LEN
    
    z_tip = -(term1 - term2)
    return -z_tip