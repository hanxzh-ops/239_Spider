# Hexapod Spider Robot — MuJoCo Simulation

A physics-based simulation of a six-legged (hexapod) robot built with [MuJoCo](https://mujoco.org/).  
The robot walks, turns, and arc-walks in real time using keyboard input.  
All motion is computed with closed-form forward and inverse kinematics; no pre-recorded motion clips are used.

---

## Table of Contents

1. [Project Goal](#1-project-goal)  
2. [System Overview](#2-system-overview)  
3. [Repository Structure](#3-repository-structure)  
4. [Robot Model](#4-robot-model)  
5. [Kinematics](#5-kinematics)  
6. [Gait and Controller](#6-gait-and-controller)  
7. [Requirements](#7-requirements)  
8. [Installation](#8-installation)  
9. [Running the Simulation](#9-running-the-simulation)  
10. [Keyboard Controls](#10-keyboard-controls)  
11. [Configuration and Tuning](#11-configuration-and-tuning)  
12. [File Reference](#12-file-reference)  
13. [Known Limitations and Future Work](#13-known-limitations-and-future-work)  

---

## 1. Project Goal

Build a fully physics-simulated hexapod robot that:

- Walks stably on flat ground using a **diagonal tripod gait**
- Responds to keyboard input with **smooth, continuous motion** (hold a key = keep moving, release = gradually stop)
- Supports **arc walking** — forward and turning simultaneously (like an analog joystick)
- Uses **analytically derived closed-form IK** rather than numerical solvers or motion capture data
- Models servo-like actuator physics including torque limits, rotor inertia, and friction

The project is also a learning platform for hexapod locomotion mechanics, MuJoCo physics simulation, and real-time robot control loop design.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Keyboard (pynput)                                          │
│   W / S / A / D → held-key set (real press + release)      │
└────────────────────┬────────────────────────────────────────┘
                     │  fwd ∈ [-1,1]   turn ∈ [-1,1]
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  LocomotionController  (60 Hz)                              │
│   • Velocity ramp  — smooth accel / decel                   │
│   • Tripod gait phase counter                               │
│   • _foot_trajectory() — stance + swing curves per leg      │
│   • IK per leg  →  joint angle targets dict                 │
└────────────────────┬────────────────────────────────────────┘
                     │  {joint_name: angle_rad, ...}
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  MujocoInterface                                            │
│   • set_joint_targets() → data.ctrl[]                       │
│   • 8× physics sub-steps per control frame  (500 Hz)        │
│   • get_body_pose(), get_joint_angles()                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  MuJoCo Physics Engine  +  Passive Viewer                   │
│   • hexapod.xml model                                       │
│   • Contact, friction, gravity, servo dynamics              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
239_Spider/
│
├── main.py                  ← Entry point — run this to start the simulation
│
├── assets/
│   ├── hexapod.xml          ← MuJoCo MJCF model (geometry, joints, actuators, sensors)
│   └── config.py            ← Canonical robot constants (link lengths, joint ranges, defaults)
│
├── src/
│   ├── controller.py        ← Velocity-based locomotion controller (gait + IK dispatch)
│   ├── fk.py                ← Closed-form forward kinematics
│   ├── ik.py                ← Closed-form inverse kinematics
│   ├── mj_interface.py      ← Thin MuJoCo wrapper (ctrl, qpos, body pose)
│   └── config.py            ← Shim that re-exports assets/config.py
│
├── requirements.txt         ← Python dependencies
└── README.md                ← This file
```

> Legacy experiment files (`Walking.py`, `walking_demo.py`, `Hexapod_move.py`, etc.) remain in the root for reference but are superseded by `main.py`.

---

## 4. Robot Model

### Body layout

```
              [leg4]  [leg2]  [leg0]
               rear    mid    front
          (L)   ●───────●───────●
                        │ torso │
          (R)   ●───────●───────●
               rear    mid    front
              [leg5]  [leg3]  [leg1]
```

| Leg | Position | Side  | Coxa neutral direction |
|-----|----------|-------|------------------------|
| 0   | Front    | Left  | 45° (front-left)        |
| 1   | Front    | Right | −45° (front-right)      |
| 2   | Mid      | Left  | 90° (pure left)         |
| 3   | Mid      | Right | −90° (pure right)       |
| 4   | Rear     | Left  | 135° (rear-left)        |
| 5   | Rear     | Right | −135° (rear-right)      |

### Leg kinematic chain (per leg, 3 DOF)

```
torso
  └─ leg_base  (rigid attachment, at torso side)
       └─ coxa_joint   (hinge, Z-axis, yaw — sweeps leg left/right)
            └─ [25 mm coxa stub along local +X]
                 └─ femur_joint  (hinge, Y-axis, pitch — lifts/lowers femur)
                      └─ [100 mm femur link along local +Z]
                           └─ tibia_joint  (hinge, Y-axis, pitch — knee flex)
                                └─ [100 mm tibia link along local +Z]
                                     └─ foot tip  (contact + touch sensor)
```

### Link lengths

| Segment | Length | Description |
|---------|--------|-------------|
| Coxa    | 25 mm  | Short lateral stub; places femur joint away from body |
| Femur   | 100 mm | Upper leg; controlled by femur pitch joint |
| Tibia   | 100 mm | Lower leg; ground contact via tibia pitch joint |

### Actuator model

Each joint is driven by a position-controlled servo with physically realistic parameters:

| Parameter      | Value        | Meaning |
|----------------|--------------|---------|
| `kp` (coxa)    | 30 N·m/rad   | Position stiffness |
| `kp` (femur)   | 28 N·m/rad   | Position stiffness |
| `kp` (tibia)   | 22 N·m/rad   | Position stiffness |
| `kv`           | 0.5 N·m·s/rad | Velocity (derivative) damping |
| `forcerange`   | ±2.5 N·m     | Peak servo torque (≈ Dynamixel XL430) |
| `armature`     | 0.004 kg·m²  | Rotor inertia |
| `damping`      | 0.5 N·m·s/rad | Joint viscous drag |
| `frictionloss` | 0.01 N·m     | Coulomb friction |

### Sensors

| Sensor | Type | Purpose |
|--------|------|---------|
| `torso_orientation` | `framequat` | Body orientation quaternion (IMU) |
| `torso_gyro`        | `gyro`      | Angular velocity at IMU site |
| `touch_leg{0–5}`   | `touch`     | Contact force at each foot tip |

---

## 5. Kinematics

### Forward Kinematics — `src/fk.py`

Analytically computes foot positions in the **torso frame** given joint angles.

The chain for each leg:

```
foot = leg_base + R_coxa @ (coxa_offset + R_femur @ (R_tibia @ tip + femur_vec))
```

where:
- `R_coxa  = Rot_Z(θ_coxa)`   — yaw rotation
- `R_femur = Rot_Y(θ_femur)`  — pitch rotation
- `R_tibia = Rot_Y(θ_tibia)`  — pitch rotation
- `coxa_offset = [0.025, 0, 0]`  — lateral coxa stub

**Key function:** `fk_all(joint_angles_dict) → {leg_id: np.array([x,y,z])}`

### Inverse Kinematics — `src/ik.py`

Closed-form 3-DOF IK given a target foot position in the torso frame.

**Steps:**

1. **Coxa angle** — `θ_coxa = atan2(y, x)` points the leg toward the target  
2. **Subtract coxa stub** — shift foot into the femur-joint frame  
3. **Rotate into leg plane** — project out coxa yaw (2-D problem remains)  
4. **Law of cosines** — solve for femur and tibia angles analytically  
5. **Clamp** — enforce joint limits from `JOINT_RANGES`

Two solution branches exist (elbow-up / elbow-down); the controller always uses **elbow-down** (`elbow_up=False`), which is the natural crouching configuration.

**Round-trip accuracy:** FK → IK → FK produces < 1 × 10⁻¹⁵ m position error.

**Key function:** `ik_leg_from_foot_torso(foot_torso, leg_index, elbow_up, clamp) → (success, joints_dict)`

---

## 6. Gait and Controller

### Tripod gait

The six legs are split into two diagonal groups that alternate:

| Group   | Legs        | Phase offset |
|---------|-------------|--------------|
| Tripod A | 0, 3, 4    | 0.0          |
| Tripod B | 1, 2, 5    | 0.5          |

At any moment, three legs are on the ground (stance) while the other three swing forward — providing a stable three-point base of support throughout the cycle.

### Foot trajectory

Each foot follows a two-phase trajectory within one gait cycle (`STEP_FREQ = 0.9 Hz`):

```
Stance phase (0 → 0.5):  foot on ground, sweeps backward
Swing  phase (0.5 → 1):  foot lifts (sine curve), swings forward
```

Foot displacement scales directly with the velocity inputs `fwd` and `turn`, so half-throttle produces half the stride length — no mode switching required.

### Velocity-based controller

The controller exposes a continuous velocity API instead of discrete states:

```
set_velocity(fwd, turn)
    fwd  ∈ [-1, 1]   +1 = full forward,  -1 = full backward
    turn ∈ [-1, 1]   +1 = left (CCW),    -1 = right (CW)
```

Both axes are **independently ramped** each control frame:

| Parameter     | Value       |
|---------------|-------------|
| `RAMP_ACCEL`  | 3.0 units/s |
| `RAMP_DECEL`  | 6.0 units/s |

Deceleration is twice as fast as acceleration so stops feel crisp. Holding W+A simultaneously produces arc walking with the forward and turn axes both active.

### Post-settling calibration

After the 400-frame physics warm-up, `controller.calibrate()` reads the actual settled joint angles and torso height. This sets the correct stance reference (`_stance`) and ground-contact depth (`_z0`) so that:

- Swing feet lift exactly `STEP_H = 30 mm` above the real ground (not an FK estimate)
- Standing IK targets match the physical resting configuration

---

## 7. Requirements

### System

| Requirement | Minimum |
|-------------|---------|
| OS          | macOS 12+, Ubuntu 20.04+, or Windows 10+ |
| Python      | 3.9 or later |
| OpenGL      | 3.3+ compatible GPU / integrated graphics |
| Display     | Required (GUI viewer) |

### macOS — Accessibility permission

`pynput` is used for key hold/release detection.  
On macOS you must grant your terminal application Accessibility access:

> **System Settings → Privacy & Security → Accessibility → enable Terminal** (or iTerm2, VS Code, etc.)

Without this, keys will only trigger once per press instead of continuously.

---

## 8. Installation

```bash
# 1. Clone or copy the project
cd 239_Spider

# 2. (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` contains:

```
mujoco>=3.0.0
numpy>=1.24.0
pynput>=1.7.0
```

> **Note:** If `pynput` cannot be installed or fails to start (e.g. on a headless server),  
> the simulation still runs but key hold detection falls back to a single-press-burst mode.  
> A message is printed at startup indicating which mode is active.

---

## 9. Running the Simulation

```bash
python main.py
```

On first launch the robot performs a ~7-second physics warm-up to settle into its default stance, then opens the interactive viewer window.

**Expected startup output:**

```
[Input] pynput active — hold keys for continuous motion.
Physics: 500 Hz  control: 60 Hz  8 sub-steps/frame
Settling into default stance ...
[Controller] calibrated — torso_z=0.0820 m  stance_z0=0.0692 m
[Controller] -> stand
```

---

## 10. Keyboard Controls

### Movement — hold for continuous motion

| Key | Action |
|-----|--------|
| `W` / `↑` | Walk forward |
| `S` / `↓` | Walk backward |
| `A` | Turn left (CCW, in place) |
| `D` | Turn right (CW, in place) |
| `W` + `A` | Arc walk — forward + left |
| `W` + `D` | Arc walk — forward + right |
| `S` + `A` | Arc walk — backward + left |
| `S` + `D` | Arc walk — backward + right |

Releasing a key ramps that axis smoothly back to zero.  
Releasing all keys lets the robot glide to a natural halt.

### Single-press commands

| Key | Action |
|-----|--------|
| `1` | Return to default stance and stop |
| `Q` | Raise torso (body up) |
| `E` | Lower torso (body down) |
| `0` / `Space` | Idle — hold joint angles, no actuation |
| `ESC` | Quit |

---

## 11. Configuration and Tuning

All tunable parameters live in two files:

### `assets/config.py` — Robot geometry

| Constant | Default | Description |
|----------|---------|-------------|
| `LINK_LENGTHS["coxa"]`  | 0.025 m | Coxa stub length |
| `LINK_LENGTHS["femur"]` | 0.100 m | Femur link length |
| `LINK_LENGTHS["tibia"]` | 0.100 m | Tibia link length |
| `FEMUR_DEFAULT`         | 1.31 rad | Default femur angle |
| `TIBIA_DEFAULT`         | 1.52 rad | Default tibia angle |
| `TRIPOD_A`              | [0, 3, 4] | First tripod group |
| `TRIPOD_B`              | [1, 2, 5] | Second tripod group |

### `src/controller.py` — Motion parameters

| Constant | Default | Description |
|----------|---------|-------------|
| `STEP_FREQ`   | 0.9 Hz   | Gait cycles per second |
| `STEP_LEN`    | 0.05 m   | Max foot travel per step (at full throttle) |
| `TURN_LEN`    | 0.04 m   | Max tangential foot arc per step |
| `STEP_H`      | 0.030 m  | Swing foot lift height |
| `DUTY`        | 0.5      | Fraction of cycle in stance |
| `RAMP_ACCEL`  | 3.0 u/s  | Velocity ramp-up rate |
| `RAMP_DECEL`  | 6.0 u/s  | Velocity ramp-down rate |
| `VEL_THRESH`  | 0.025    | Velocity below which robot holds stance |
| `HEIGHT_INC`  | 0.007 m  | Body height change per Q/E press |
| `HEIGHT_MIN`  | 0.025 m  | Minimum allowed body height |

### `assets/hexapod.xml` — Physics parameters

Key actuator values in the `<default>` block:

| Parameter      | Value | Effect |
|----------------|-------|--------|
| `kp` (coxa)    | 30    | Higher = stiffer yaw response |
| `kv`           | 0.5   | Higher = more damped, less oscillation |
| `forcerange`   | ±2.5  | Lower = servo stalls more easily |
| `armature`     | 0.004 | Higher = more inertia, slower response |
| `damping`      | 0.5   | Higher = more drag at all speeds |

---

## 12. File Reference

| File | Purpose |
|------|---------|
| `main.py` | Entry point. Loads model, runs warm-up, starts control loop and viewer |
| `assets/hexapod.xml` | MJCF robot model: bodies, joints, geoms, actuators, sensors |
| `assets/config.py` | All robot constants: link lengths, joint limits, default pose, tripod groups |
| `src/controller.py` | `LocomotionController` — velocity ramp, gait phase, FK→IK dispatch |
| `src/fk.py` | `fk_leg()`, `fk_all()` — forward kinematics in torso frame |
| `src/ik.py` | `ik_leg_from_foot_torso()` — closed-form IK with joint clamping |
| `src/mj_interface.py` | `MujocoInterface` — wraps MuJoCo data access (ctrl, qpos, body pose) |
| `src/config.py` | Thin shim that re-exports `assets/config.py` |
| `requirements.txt` | Python package dependencies |

---

## 13. Known Limitations and Future Work

### Current limitations

- **Flat ground only** — the gait planner assumes a horizontal contact surface; slopes and steps are not handled
- **Open-loop gait** — foot trajectories are pre-computed from the stance reference; no real-time contact feedback adjusts footfall timing
- **No body orientation control** — the torso can tilt slightly under uneven loading; an IMU-based attitude controller is not yet implemented
- **Fixed stride frequency** — `STEP_FREQ` is constant; a real system would vary frequency with commanded speed

### Possible extensions

- **Terrain adaptation** — use touch sensor data to detect early ground contact and adjust step height per leg
- **IMU stabilisation** — read `torso_orientation` sensor and add a body-levelling feedback loop
- **Speed-frequency coupling** — scale `STEP_FREQ` proportionally with the commanded velocity magnitude for more natural locomotion
- **Sinusoidal body sway** — add lateral/longitudinal body oscillation in sync with the gait phase (as seen in real insects)
- **Obstacle avoidance** — add lidar or depth sensor to the model and implement reactive stepping
- **Reinforcement learning** — use the MuJoCo environment as a training substrate for learned locomotion policies
- **Hardware transfer** — export joint angle targets at 60 Hz as servo PWM commands to a physical hexapod (e.g. Lynxmotion Phoenix, custom build)
