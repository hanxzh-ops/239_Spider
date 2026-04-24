#!/usr/bin/env python3
"""
src/controller.py  —  Hexapod Locomotion Controller
=====================================================

Velocity-based control with smooth acceleration and deceleration.
fwd and turn are continuous values in [-1, 1] that are independently
ramped each frame so combined W+A / W+D arc-walking works naturally.

Velocity API (called every frame from main loop)
-------------------------------------------------
  set_velocity(fwd, turn)
      fwd  in [-1, 1]   +1 = full forward,  -1 = full backward
      turn in [-1, 1]   +1 = left (CCW),    -1 = right (CW)

  Both axes blend simultaneously:
    W alone          -> straight forward
    A / D alone      -> spin in place
    W + A / W + D    -> arc walk (forward + turning)
    Release any key  -> that axis ramps back to zero (gradual stop)
    Release all keys -> both axes ramp to zero, robot glides to a halt

Discrete commands (single-press)
---------------------------------
  "stand"      hold stance, zero desired velocity
  "idle"       hold joint positions, no actuation
  "body_up"    raise torso (works while walking)
  "body_down"  lower torso
"""

import math
import numpy as np

from assets.config import TRIPOD_A, TRIPOD_B, DEFAULT_JOINT_POS
from src.fk import fk_all
from src.ik import ik_leg_from_foot_torso

# ── States ────────────────────────────────────────────────────────────────────
IDLE  = "idle"
STAND = "stand"

# ── Gait parameters ───────────────────────────────────────────────────────────
STEP_FREQ = 0.9     # Hz   — gait cycles per second
STEP_LEN  = 0.05    # m    — peak-to-peak foot travel (at |fwd| = 1)
TURN_LEN  = 0.04    # m    — tangential foot arc      (at |turn| = 1)
STEP_H    = 0.030   # m    — swing foot lift above stance
DUTY      = 0.5     # fraction of cycle spent in stance

# ── Velocity ramp rates ───────────────────────────────────────────────────────
RAMP_ACCEL = 3.0    # velocity units / second — how fast to speed up
RAMP_DECEL = 6.0    # velocity units / second — how fast to slow down

# ── Below this magnitude the robot switches to stand mode ────────────────────
VEL_THRESH = 0.025

# ── Body-height parameters ────────────────────────────────────────────────────
HEIGHT_INC = 0.007   # m per key-press
HEIGHT_MIN = 0.025   # m (fully crouched)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ramp(current: float, desired: float, accel: float, decel: float) -> float:
    """
    Advance `current` one step toward `desired`.

    Uses `accel` when the magnitude is growing (speeding up) and `decel`
    when it is shrinking or changing sign (slowing down / reversing).
    """
    delta = desired - current
    if abs(delta) < 1e-7:
        return desired
    speeding_up = (desired * current >= 0.0) and (abs(desired) >= abs(current))
    step = min(abs(delta), accel if speeding_up else decel)
    return current + math.copysign(step, delta)


# ── Controller ────────────────────────────────────────────────────────────────

class LocomotionController:

    def __init__(self, mj_interface):
        self.mj    = mj_interface
        self.state = IDLE

        # Nominal foot positions in torso frame (refreshed by calibrate)
        self._stance = fk_all(DEFAULT_JOINT_POS)
        self._z0     = float(self._stance[0][2])

        # Body height target
        self._body_height = abs(self._z0)
        self._height_max  = self._body_height

        # Tripod phase offsets: group A at 0.0, group B at 0.5
        self._offset = {leg: (0.0 if leg in TRIPOD_A else 0.5)
                        for leg in range(6)}

        self._phase  = 0.0

        # ── Continuous velocity state ─────────────────────────────────────
        self._fwd      = 0.0   # current forward drive  [-1, 1]
        self._turn     = 0.0   # current turn rate       [-1, 1]
        self._des_fwd  = 0.0   # desired forward drive
        self._des_turn = 0.0   # desired turn rate

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self):
        """
        Sync the controller's reference stance with the actual settled robot.
        Call once after the physics warm-up loop in main.py.
        """
        q = self.mj.get_joint_angles()
        self._stance = fk_all(q)

        z_vals = [float(self._stance[leg][2]) for leg in range(6)]
        self._z0 = sum(z_vals) / len(z_vals)

        tp, _ = self.mj.get_body_pose("torso")
        self._body_height = float(tp[2])
        self._height_max  = self._body_height

        print(f"[Controller] calibrated — torso_z={self._body_height:.4f} m  "
              f"stance_z0={self._z0:.4f} m")

    # ── Velocity API ──────────────────────────────────────────────────────────

    def set_velocity(self, fwd: float, turn: float):
        """
        Set desired locomotion velocity for this control frame.

        Call every frame with values derived from the current key state.
        Pass (0, 0) when no movement keys are held — the controller will
        ramp smoothly to a stop on its own.

        fwd  in [-1, 1]:  +1 = forward,     -1 = backward
        turn in [-1, 1]:  +1 = left (CCW),  -1 = right (CW)
        """
        self._des_fwd  = float(np.clip(fwd,  -1.0, 1.0))
        self._des_turn = float(np.clip(turn, -1.0, 1.0))
        # A non-zero velocity command wakes the controller from idle
        if (abs(self._des_fwd) > 0.01 or abs(self._des_turn) > 0.01) \
                and self.state == IDLE:
            self.state = STAND

    # ── Discrete commands ─────────────────────────────────────────────────────

    def command(self, cmd: str):
        if cmd == STAND:
            self.state     = STAND
            self._des_fwd  = 0.0
            self._des_turn = 0.0
            print("[Controller] -> stand")

        elif cmd == IDLE:
            self.state     = IDLE
            self._des_fwd  = 0.0
            self._des_turn = 0.0
            print("[Controller] -> idle")

        elif cmd == "body_up":
            self._body_height = min(self._body_height + HEIGHT_INC,
                                    self._height_max)
            print(f"[Controller] body height -> {self._body_height:.3f} m")

        elif cmd == "body_down":
            self._body_height = max(self._body_height - HEIGHT_INC, HEIGHT_MIN)
            print(f"[Controller] body height -> {self._body_height:.3f} m")

        else:
            print(f"[Controller] unknown command: '{cmd}'")

    # ── Main step (60 Hz) ─────────────────────────────────────────────────────

    def step(self, dt: float):
        # Use the nominal control period directly for all time-based calculations.
        # This makes ramp rates and gait phase deterministic regardless of whether
        # the loop runs faster or slower than real-time (e.g. during unit tests).
        # Clamp to 100 ms to avoid huge jumps on the very first frame or after a
        # long pause (e.g. viewer minimised).
        dt = min(float(dt), 0.1)

        # Phase always advances — gait cycle keeps ticking
        self._phase = (self._phase + dt * STEP_FREQ) % 1.0

        if self.state == IDLE:
            # Decay residual velocity but don't issue joint commands
            self._fwd  = _ramp(self._fwd,  0.0,
                               RAMP_DECEL * dt, RAMP_DECEL * dt)
            self._turn = _ramp(self._turn, 0.0,
                               RAMP_DECEL * dt, RAMP_DECEL * dt)
            return

        # Ramp current velocities toward desired values
        self._fwd  = _ramp(self._fwd,  self._des_fwd,
                           RAMP_ACCEL * dt, RAMP_DECEL * dt)
        self._turn = _ramp(self._turn, self._des_turn,
                           RAMP_ACCEL * dt, RAMP_DECEL * dt)

        # Walk when either axis is significant; hold stance when stopped
        if abs(self._fwd) > VEL_THRESH or abs(self._turn) > VEL_THRESH:
            self._do_walk(self._fwd, self._turn)
        else:
            self._do_stand()

    # ── STAND ─────────────────────────────────────────────────────────────────

    def _do_stand(self):
        """
        Hold the nominal stance at the current body height.
        Uses per-leg torso-frame IK so joint limits bound the result.
        """
        z_t = float(np.clip(-self._body_height, -0.13, -0.02))
        j   = {}
        for leg in range(6):
            x0 = float(self._stance[leg][0])
            y0 = float(self._stance[leg][1])
            ok, res = ik_leg_from_foot_torso(
                np.array([x0, y0, z_t]),
                leg_index=leg, elbow_up=False, clamp=True)
            if ok:
                j.update(res)
        if j:
            self.mj.set_joint_targets(j)

    # ── WALK ──────────────────────────────────────────────────────────────────

    def _do_walk(self, fwd: float, turn: float):
        """
        Tripod gait.  fwd and turn scale the step amplitudes directly, so
        partial velocities produce proportionally shorter footfalls —
        the same effect as modulating stride length on a real robot.
        """
        torso_pos, _ = self.mj.get_body_pose("torso")
        torso_z      = float(torso_pos[2])

        j = {}
        for leg in range(6):
            phase  = (self._phase + self._offset[leg]) % 1.0
            target = self._foot_trajectory(phase, self._stance[leg],
                                           fwd, turn, torso_z)
            ok, res = ik_leg_from_foot_torso(
                target, leg_index=leg, elbow_up=False, clamp=True)
            if ok:
                j.update(res)
        if j:
            self.mj.set_joint_targets(j)

    def _foot_trajectory(self, phase: float, stance: np.ndarray,
                          fwd: float, turn: float,
                          torso_z: float) -> np.ndarray:
        """
        Desired foot position in the torso frame.

        Step amplitude scales with fwd / turn so the robot walks slowly at
        low velocity and at full stride at |vel| = 1.  No mode switching is
        needed — the same function handles the full speed range.

        z_ground = clip(-torso_z, lo=-0.13, hi=z0)
          torso too low  -> capped at z0    -> foot pushes body back up
          torso too high -> equals -torso_z -> foot targets ground exactly
          -0.13 m        -> IK safety floor
        """
        x0, y0, z0 = float(stance[0]), float(stance[1]), float(stance[2])

        z_ground = float(np.clip(-torso_z, -0.13, z0))

        # Tangential unit vector perpendicular to radial (for turning)
        r_xy = math.sqrt(x0 * x0 + y0 * y0)
        if r_xy > 1e-6:
            tx, ty = -y0 / r_xy, x0 / r_xy
        else:
            tx, ty = 0.0, 0.0

        if phase < DUTY:
            # Stance: foot on ground, sweeps backward relative to body
            s    = phase / DUTY
            coef = 0.5 - s          # +0.5 at start -> -0.5 at end
            dx   = fwd  * coef * STEP_LEN + turn * coef * TURN_LEN * tx
            dy   =                           turn * coef * TURN_LEN * ty
            return np.array([x0 + dx, y0 + dy, z_ground])
        else:
            # Swing: foot lifts and swings forward
            s    = (phase - DUTY) / (1.0 - DUTY)
            coef = s - 0.5          # -0.5 at start -> +0.5 at end
            dx   = fwd  * coef * STEP_LEN + turn * coef * TURN_LEN * tx
            dy   =                           turn * coef * TURN_LEN * ty
            dz   = STEP_H * math.sin(math.pi * s)   # smooth bell-curve lift
            return np.array([x0 + dx, y0 + dy, z_ground + dz])
