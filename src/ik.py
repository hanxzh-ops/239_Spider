#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:58:56 2025

@author: jamesz
"""

"""
src/ik.py

Closed-form inverse kinematics for the 3-DoF hexapod legs described in assets/config.py.

Assumptions (consistent with src/fk.py and hexapod.xml):
 - Leg base positions are in the torso frame (LEG_BASE_POS).
 - Coxa axis = z (rotation about vertical).
 - Femur axis = y (rotation affecting x-z plane).
 - Tibia axis = y (rotation affecting x-z plane).
 - Link lengths (d_f, d_t) correspond to:
     femur -> tibia joint offset = d_f
     tibia -> foot tip offset      = d_t

Primary functions:
 - ik_leg_from_foot_world(...)
 - ik_leg_from_foot_torso(...)
 - ik_all(...)
"""

from math import atan2, acos, cos, sin, sqrt, isclose
import numpy as np

from assets.config import LEG_BASE_POS, LINK_LENGTHS, JOINT_RANGES

# Small numerical tolerance
_EPS = 1e-9

# --- quaternion (w,x,y,z) -> rotation matrix (world <- torso) ---
def quat_to_rot_matrix(quat):
    # MuJoCo convention: w, x, y, z
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [  2*(x*y + w*z),     1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [  2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])
    return R

# --- small helper: clamp angle to [min,max] if desired ---
def clamp_angle_to_range(angle, rng):
    lo, hi = rng
    # If range spans > 2*pi we skip clamping
    if hi - lo >= 2*np.pi - 1e-6:
        return angle
    # Normalize angle into same principal region as range midpoint
    mid = 0.5*(lo + hi)
    # shift angle by multiples of 2pi so it's near mid
    a = angle
    while a - mid > np.pi:
        a -= 2*np.pi
    while a - mid < -np.pi:
        a += 2*np.pi
    # clamp
    if a < lo:
        a = lo
    if a > hi:
        a = hi
    return a

# --- IK for one leg given foot position in torso frame ---
def ik_leg_from_foot_torso(foot_torso, leg_index, elbow_up=True, clamp=True):
    """
    Inverse kinematics for a single leg.

    Args:
      foot_torso: np.array([x,y,z]) foot target expressed in the torso frame (meters).
      leg_index: int 0..5 selecting LEG_BASE_POS.
      elbow_up: bool choose between the two analytic solutions (True -> "elbow-up" variant).
      clamp: if True, clamp the computed angles to JOINT_RANGES when possible.

    Returns:
      success (bool), joint_angles (dict) or message (str)
      If success: joint_angles = {"leg{i}_coxa_j": val, "leg{i}_femur_j": val, "leg{i}_tibia_j": val}
    """

    # Retrieve model link lengths and base pos
    base = LEG_BASE_POS[leg_index]
    d_c = LINK_LENGTHS.get("coxa", 0.0)   # coxa stub length (may be 0 for old models)
    d_f = LINK_LENGTHS["femur"]
    d_t = LINK_LENGTHS["tibia"]

    # Foot position relative to the coxa joint (leg base) in torso frame
    p_leg = foot_torso - base
    x_leg = float(p_leg[0])
    y_leg = float(p_leg[1])
    z_leg = float(p_leg[2])

    # --- Coxa (yaw) ---
    # The coxa joint sweeps the whole leg in the horizontal plane so that the
    # coxa's local +X (and thus the femur joint) points toward the foot.
    coxa = atan2(y_leg, x_leg)  # radians

    # Rotate foot into the coxa-aligned leg plane (undo coxa yaw).
    c = cos(coxa); s = sin(coxa)
    Rz_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # Rz(-coxa)
    p_plane = Rz_inv @ np.array([x_leg, y_leg, z_leg])

    # Subtract the coxa stub offset so (x, z) is the foot relative to the
    # *femur joint* (not the coxa joint).  In the aligned frame the stub lies
    # exactly along +X, so we simply reduce x by d_c.
    x = float(p_plane[0]) - d_c
    z = float(p_plane[2])

    # --- Planar reachability check ---
    # We derived that:
    #    (c2 * x - s2 * z)^2 + (s2 * x + c2 * z - d_f)^2 = d_t^2
    # and after algebra this reduces to a solvable condition that yields:
    #    D = (x^2 + z^2 + d_f^2 - d_t^2) / (2*d_f)
    # and we require |D| <= sqrt(x^2 + z^2) for a real solution.
    r = sqrt(max(x*x + z*z, 0.0))
    if r < _EPS:
        return False, f"Target coincides with leg base (r≈0). leg={leg_index}"

    K = x*x + z*z + d_f*d_f - d_t*d_t
    D = K / (2.0 * d_f)

    # Check reachable
    if abs(D) > r + 1e-12:
        return False, f"Unreachable target for leg {leg_index}: |D|={abs(D):.6f} > r={r:.6f}"

    # Compute beta (angle of vector [z, x]) such that:
    #    [z, x] = r * [cos(beta), sin(beta)]
    # so beta = atan2(x, z)
    beta = atan2(x, z)

    # Solve for theta2 (femur) from cos(theta2 - beta) = D / r
    # two solutions: theta2 = beta ± acos(D/r)
    # choose sign by elbow_up flag
    # numerically clamp ratio
    ratio = D / r
    if ratio > 1.0:
        ratio = 1.0
    if ratio < -1.0:
        ratio = -1.0
    phi = acos(ratio)  # in [0, pi]

    if elbow_up:
        theta2 = beta + phi
    else:
        theta2 = beta - phi

    # Now compute theta3 (tibia)
    c2 = cos(theta2)
    s2 = sin(theta2)

    # We defined:
    #   A = c2*x - s2*z = d_t * sin(theta3)
    #   B = s2*x + c2*z = d_t * cos(theta3) + d_f
    A = c2 * x - s2 * z
    B = s2 * x + c2 * z

    # Now compute tibia angle from:
    #   d_t*sin3 = A
    #   d_t*cos3 = B - d_f
    # => theta3 = atan2(A, B - d_f)
    denom = (B - d_f)
    theta3 = atan2(A, denom)

    # Now we have (coxa, femur=theta2, tibia=theta3)
    # Map to joint names
    j_coxa = f"leg{leg_index}_coxa_j"
    j_femur = f"leg{leg_index}_femur_j"
    j_tibia = f"leg{leg_index}_tibia_j"

    # Optionally clamp to JOINT_RANGES
    if clamp:
        # clamp femur and tibia into their allowed ranges if provided
        if j_femur in JOINT_RANGES:
            theta2 = clamp_angle_to_range(theta2, JOINT_RANGES[j_femur])
        if j_tibia in JOINT_RANGES:
            theta3 = clamp_angle_to_range(theta3, JOINT_RANGES[j_tibia])
        if j_coxa in JOINT_RANGES:
            coxa = clamp_angle_to_range(coxa, JOINT_RANGES[j_coxa])

    joints = {
        j_coxa: float(coxa),
        j_femur: float(theta2),
        j_tibia: float(theta3),
    }

    return True, joints


# --- IK from world frame (convenience) ---
def ik_leg_from_foot_world(foot_world, torso_pos, torso_quat, leg_index, elbow_up=True, clamp=True):
    """
    Compute leg IK when foot target is given in world coordinates.

    Args:
      foot_world: np.array([x,y,z]) in world frame
      torso_pos: np.array([x,y,z]) torso position in world frame
      torso_quat: (w,x,y,z) torso orientation quaternion (MuJoCo format)
      leg_index, elbow_up, clamp: as above

    Returns:
      success(bool), joints(dict) or message(str)
    """
    # Convert world -> torso frame
    R = quat_to_rot_matrix(torso_quat)  # world <- torso
    # we need torso^-1 * (foot_world - torso_pos) so use R^T
    p = np.array(foot_world) - np.array(torso_pos)
    foot_torso = R.T @ p

    return ik_leg_from_foot_torso(foot_torso, leg_index, elbow_up=elbow_up, clamp=clamp)


# --- Batch IK for many legs ---
def ik_all(foot_world_map, torso_pos, torso_quat, elbow_up=True, clamp=True):
    """
    Batch IK: foot_world_map = {leg_index: np.array([x,y,z]), ...}
    Returns:
      results = { leg: (success, joints_or_message) }
    """
    results = {}
    for leg, foot in foot_world_map.items():
        success, out = ik_leg_from_foot_world(foot, torso_pos, torso_quat, leg, elbow_up=elbow_up, clamp=clamp)
        results[leg] = (success, out)
    return results


# -------------------------
# Quick test utility when invoked directly
# -------------------------
if __name__ == "__main__":
    # small smoke test: compute IK for default standing feet computed by FK
    from assets.config import DEFAULT_JOINT_POS
    from src.fk import fk_all
    # torso at origin with no rotation
    torso_pos = np.array([0.0, 0.0, 0.05])
    torso_quat = (1.0, 0.0, 0.0, 0.0)

    feet = fk_all(DEFAULT_JOINT_POS)
    # feet are in torso frame, convert to world
    # for smoke test, world==torso so world=torso_pos + p
    foot_world = {leg: torso_pos + p for leg, p in feet.items()}

    print("Running IK smoke test on default standing feet...")
    res = ik_all(foot_world, torso_pos, torso_quat)
    for leg, (ok, out) in res.items():
        if not ok:
            print(f"LEG {leg} IK FAIL:", out)
        else:
            print(f"LEG {leg} IK OK. joints:", out)
