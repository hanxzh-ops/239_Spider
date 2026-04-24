#!/usr/bin/env python3
"""
main.py — Hexapod Spider Robot  ·  Keyboard-Controlled Demo
============================================================

Movement keys (hold for continuous motion, release to slow down)
----------------------------------------------------------------
  W  / Up-arrow   Walk forward
  S  / Down-arrow Walk backward
  A               Turn left  (CCW)
  D               Turn right (CW)

  W + A  /  W + D     Arc walk — forward while turning
  S + A  /  S + D     Arc walk — backward while turning

  Releasing any movement key ramps that axis back to zero.
  Releasing all movement keys lets the robot glide to a natural halt.

Single-press commands
---------------------
  1          Stand (return to default stance, stop all motion)
  Q          Body up   (raise torso — works while walking too)
  E          Body down (lower torso)
  0 / Space  Idle — hold joint positions, no actuation
  ESC        Quit

Setup
-----
  For smooth hold-to-move behaviour this script uses pynput for key tracking.
  If not yet installed:
      pip install pynput

  On macOS you may need to grant Accessibility permission to Terminal / Python:
      System Settings → Privacy & Security → Accessibility → enable Terminal
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import mujoco.viewer

from src.mj_interface import MujocoInterface
from src.controller   import LocomotionController
from assets.config    import DEFAULT_JOINT_POS

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH    = os.path.join(CURRENT_DIR, "assets", "hexapod.xml")

# ─────────────────────────────────────────────────────────────────────────────
# Key-state tracking  —  pynput (preferred) with timeout fallback
#
# WHY pynput:
#   MuJoCo's passive viewer key_callback fires only on GLFW_PRESS, never on
#   GLFW_REPEAT or GLFW_RELEASE.  Without release events, we can't tell when
#   the user lets go of a key, so long-hold detection is impossible through
#   the MuJoCo callback alone.
#
#   pynput hooks into the OS key event stream and delivers real press/release
#   events, independent of GLFW.  We use it only for the four movement keys;
#   all single-press commands still go through MuJoCo's callback.
#
# FALLBACK:
#   If pynput is missing or can't start (headless server, missing permissions),
#   we fall back to a timeout-based approach — each press extends a hold
#   window, but the window is short so only one key-press = ~1 step.
#   Install pynput to get full continuous-hold behaviour.
# ─────────────────────────────────────────────────────────────────────────────

_held: set = set()          # currently held movement keys (populated by pynput)
_PYNPUT_OK = False

try:
    from pynput import keyboard as _pynput_kb

    def _on_press(key):
        try:
            k = key.char.lower() if key.char else None
            if k in ('w', 'a', 's', 'd'):
                _held.add(k)
        except AttributeError:
            # Special keys  (arrow keys, etc.)
            if   key == _pynput_kb.Key.up:    _held.add('up')
            elif key == _pynput_kb.Key.down:  _held.add('down')

    def _on_release(key):
        try:
            k = key.char.lower() if key.char else None
            _held.discard(k)
        except AttributeError:
            if   key == _pynput_kb.Key.up:    _held.discard('up')
            elif key == _pynput_kb.Key.down:  _held.discard('down')

    _listener = _pynput_kb.Listener(on_press=_on_press, on_release=_on_release,
                                    suppress=False)
    _listener.daemon = True
    _listener.start()
    _PYNPUT_OK = True

except Exception as _pynput_err:
    print(f"[Input] pynput unavailable: {_pynput_err}")
    print("[Input] Install with:  pip install pynput")
    print("[Input] Falling back to timeout-based tracking (one-press = short burst).")

# ─────────────────────────────────────────────────────────────────────────────
# GLFW key codes (used by MuJoCo's callback for discrete commands)
# ─────────────────────────────────────────────────────────────────────────────
KEY_ESC   = 256
KEY_SPACE = 32
KEY_UP    = 265
KEY_DOWN  = 264
KEY_W     = ord('W')
KEY_S     = ord('S')
KEY_A     = ord('A')
KEY_D     = ord('D')
KEY_Q     = ord('Q')
KEY_E     = ord('E')
KEY_1     = ord('1')
KEY_0     = ord('0')

# ─────────────────────────────────────────────────────────────────────────────
# Timeout-based fallback parameters
#   KEY_HOLD: how long a single GLFW_PRESS event keeps a key "held".
#   No REPEAT events arrive from MuJoCo, so each press gives one burst.
#   With pynput active these are unused.
# ─────────────────────────────────────────────────────────────────────────────
KEY_HOLD    = 0.18   # seconds — single press hold window (fallback only)
_hold_exp: dict[int, float] = {}

CONTROL_DT = 1.0 / 60.0

HELP = """
╔══════════════════════════════════════════════════╗
║   Hexapod Spider — Keyboard Controls             ║
╠══════════════════════════════════════════════════╣
║  W / ↑    Walk forward    (hold to keep moving)  ║
║  S / ↓    Walk backward   (hold to keep moving)  ║
║  A        Turn left       (hold to keep turning) ║
║  D        Turn right      (hold to keep turning) ║
║                                                  ║
║  W+A / W+D   Arc walk (forward + turn)           ║
║  S+A / S+D   Arc walk (backward + turn)          ║
║                                                  ║
║  Release keys → robot gradually slows to a stop  ║
╠══════════════════════════════════════════════════╣
║  1          Stand (return to default pose)       ║
║  Q          Body up                              ║
║  E          Body down                            ║
║  0 / Space  Idle / stop (hold joints)            ║
║  ESC        Quit                                 ║
╚══════════════════════════════════════════════════╝
"""


def main():
    if not os.path.exists(XML_PATH):
        print(f"ERROR: model XML not found at\n  {XML_PATH}")
        return

    if _PYNPUT_OK:
        print("[Input] pynput active — hold keys for continuous motion.")
    else:
        print("[Input] pynput inactive — install it for hold-to-move support.")

    # ── 1. Load MuJoCo model ──────────────────────────────────────────────
    mj = MujocoInterface(XML_PATH)
    phys_steps = max(1, round(CONTROL_DT / mj.model.opt.timestep))
    print(f"Physics: {1/mj.model.opt.timestep:.0f} Hz  "
          f"control: {1/CONTROL_DT:.0f} Hz  "
          f"{phys_steps} sub-steps/frame")

    # ── 2. Settle into default standing pose ─────────────────────────────
    print("Settling into default stance ...")
    for _ in range(400):
        mj.set_joint_targets(DEFAULT_JOINT_POS)
        for _ in range(phys_steps):
            mj.step()

    # ── 3. Create locomotion controller ──────────────────────────────────
    controller = LocomotionController(mj)
    controller.calibrate()
    controller.command("stand")

    print(HELP)

    # ── 4. MuJoCo key callback — DISCRETE commands only ──────────────────
    #
    # We use this callback ONLY for single-press commands (1, Q, E, 0, Space).
    # Movement keys (W, A, S, D, arrows) are handled by pynput above so they
    # work continuously.  In fallback mode we also record W/A/S/D here.
    #
    def on_key(keycode: int):
        # ── Discrete single-press commands ────────────────────────────────
        if   keycode == KEY_1:              controller.command("stand")
        elif keycode == KEY_Q:              controller.command("body_up")
        elif keycode == KEY_E:              controller.command("body_down")
        elif keycode in (KEY_0, KEY_SPACE): controller.command("idle")

        # ── Fallback movement key tracking (only used when pynput is off) ─
        if not _PYNPUT_OK and keycode in (KEY_W, KEY_S, KEY_A, KEY_D,
                                          KEY_UP, KEY_DOWN):
            _hold_exp[keycode] = time.time() + KEY_HOLD

    # ── 5. Main simulation loop ───────────────────────────────────────────
    with mujoco.viewer.launch_passive(
            mj.model, mj.data, key_callback=on_key) as viewer:

        while viewer.is_running():
            t_start = time.time()

            # ── Compute desired velocity from key state ───────────────────
            if _PYNPUT_OK:
                # pynput: real press/release tracking — reliable hold-to-move
                fwd  = (1.0 if ('w'  in _held or 'up'   in _held) else 0.0) \
                     - (1.0 if ('s'  in _held or 'down' in _held) else 0.0)
                turn = (1.0 if ('a'  in _held) else 0.0) \
                     - (1.0 if ('d'  in _held) else 0.0)
            else:
                # Fallback: timeout window from last GLFW_PRESS event
                now = time.time()
                def _exp(k): return now < _hold_exp.get(k, 0.0)
                fwd  = (1.0 if (_exp(KEY_W) or _exp(KEY_UP))   else 0.0) \
                     - (1.0 if (_exp(KEY_S) or _exp(KEY_DOWN))  else 0.0)
                turn = (1.0 if _exp(KEY_A) else 0.0) \
                     - (1.0 if _exp(KEY_D) else 0.0)

            controller.set_velocity(fwd, turn)

            # ── Step controller + physics ─────────────────────────────────
            controller.step(CONTROL_DT)
            for _ in range(phys_steps):
                mj.step()
            viewer.sync()

            # ── Soft real-time sleep ──────────────────────────────────────
            spare = CONTROL_DT - (time.time() - t_start)
            if spare > 0:
                time.sleep(spare)


if __name__ == "__main__":
    main()
