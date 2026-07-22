#!/usr/bin/env python3
"""
hexapod_env.py  —  Gymnasium environment for RL walking
=======================================================

Wraps assets/hexapod.xml as a standard Gymnasium env so Stable-Baselines3 (PPO) can
learn a walking gait FROM SCRATCH — no IK, no tripod schedule. The policy directly
commands all 18 joint position targets and is rewarded for moving the torso forward
(+x) while staying upright.

Contract (must stay in sync with ros2_ws policy_node.py):

  action  : Box(18,) in [-1, 1]
            mapped linearly to each joint's [lo, hi] range and written to data.ctrl
            (the position servos then track those targets).

  obs     : Box(49,) =
              jpos (18)   joint positions (rad)
              jvel (18)   joint velocities (rad/s)
              quat (4)    torso orientation quaternion (w,x,y,z)
              gyro (3)    torso angular velocity (rad/s)
              touch(6)    per-foot contact force

  reward  = forward_velocity                 (primary: move +x)
          + alive_bonus                       (stay up)
          - ctrl_cost   * ||action||^2        (smooth, cheap motion)
          - height_cost * (z - z_target)^2    (keep body height)
          - tilt_cost   * (1 - up_z)          (stay level)

  done    : torso falls (z < fall_z) or flips (up axis points down)

The physics runs `frame_skip` MuJoCo steps per env step so one agent action ≈ the
60 Hz control tick used everywhere else in the project.
"""
import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_XML = os.path.join(_REPO, "assets", "hexapod.xml")

# joint / range metadata (reuse the project's single source of truth)
import sys
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from assets.config import JOINTS, JOINT_RANGES, DEFAULT_JOINT_POS   # noqa: E402


class HexapodEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path: str = _XML, frame_skip: int = 8,
                 render_mode: str | None = None,
                 max_steps: int = 1000):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._viewer = None
        self._step_count = 0

        # --- joint / actuator addressing -----------------------------------
        self.joint_ids = [self.model.joint(j).id for j in JOINTS]
        self.qadr = [self.model.jnt_qposadr[i] for i in self.joint_ids]
        self.vadr = [self.model.jnt_dofadr[i] for i in self.joint_ids]
        self.act_ids = [self.model.actuator("act_" + j.replace("_j", "")).id
                        for j in JOINTS]
        self.lo = np.array([JOINT_RANGES[j][0] for j in JOINTS], dtype=np.float32)
        self.hi = np.array([JOINT_RANGES[j][1] for j in JOINTS], dtype=np.float32)
        self.default_ctrl = np.array([DEFAULT_JOINT_POS[j] for j in JOINTS],
                                     dtype=np.float32)

        # sensor addresses
        self._gyro_adr = self._sensor_adr("torso_gyro")
        self._touch_adr = [self._sensor_adr(f"touch_leg{i}") for i in range(6)]
        self.torso_id = self.model.body("torso").id

        # --- spaces ----------------------------------------------------------
        self.action_space = spaces.Box(-1.0, 1.0, shape=(18,), dtype=np.float32)
        obs_dim = 18 + 18 + 4 + 3 + 6   # = 49
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # --- reward weights (tune these) ------------------------------------
        self.w_forward = 1.0
        self.w_alive = 0.05
        self.w_ctrl = 0.002
        self.w_height = 0.5
        self.w_tilt = 0.3
        self.z_target = 0.09          # desired torso height (m), from calibration
        self.fall_z = 0.045           # below this -> terminated (collapsed)

        self._prev_x = 0.0

    # ------------------------------------------------------------------ utils
    def _sensor_adr(self, name):
        sid = self.model.sensor(name).id
        return self.model.sensor_adr[sid]

    def _scale_action(self, a):
        """policy [-1,1] -> joint targets in [lo, hi]."""
        a = np.clip(a, -1.0, 1.0)
        return self.lo + (a + 1.0) * 0.5 * (self.hi - self.lo)

    def _get_obs(self):
        d = self.data
        jpos = d.qpos[self.qadr].astype(np.float32)
        jvel = d.qvel[self.vadr].astype(np.float32)
        quat = d.xquat[self.torso_id].astype(np.float32)          # (w,x,y,z)
        gyro = d.sensordata[self._gyro_adr:self._gyro_adr + 3].astype(np.float32)
        touch = np.array([d.sensordata[a] for a in self._touch_adr],
                         dtype=np.float32)
        return np.concatenate([jpos, jvel, quat, gyro, touch])

    def _torso_up_z(self):
        """world-z component of the torso's local +z axis (1=level, <0=flipped)."""
        # rotation matrix column 2 = body z-axis in world; xmat is row-major 3x3
        return float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # start from the crouched default stance + tiny noise for robustness
        self.data.ctrl[self.act_ids] = self.default_ctrl
        # settle onto feet so we don't start mid-air
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        # small joint noise
        noise = self.np_random.uniform(-0.03, 0.03, size=18).astype(np.float32)
        self.data.qpos[self.qadr] += noise
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_x = float(self.data.xpos[self.torso_id][0])
        return self._get_obs(), {}

    def step(self, action):
        target = self._scale_action(np.asarray(action, dtype=np.float32))
        self.data.ctrl[self.act_ids] = target
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()

        # --- reward ----------------------------------------------------------
        x = float(self.data.xpos[self.torso_id][0])
        z = float(self.data.xpos[self.torso_id][2])
        dt = self.frame_skip * self.model.opt.timestep
        fwd_vel = (x - self._prev_x) / dt
        self._prev_x = x
        up_z = self._torso_up_z()

        r_forward = self.w_forward * fwd_vel
        r_alive = self.w_alive
        c_ctrl = self.w_ctrl * float(np.sum(np.square(action)))
        c_height = self.w_height * (z - self.z_target) ** 2
        c_tilt = self.w_tilt * (1.0 - up_z)
        reward = r_forward + r_alive - c_ctrl - c_height - c_tilt

        # --- termination -----------------------------------------------------
        fell = (z < self.fall_z) or (up_z < 0.3)
        terminated = bool(fell)
        truncated = self._step_count >= self.max_steps

        info = {
            "fwd_vel": fwd_vel, "x": x, "z": z, "up_z": up_z,
            "r_forward": r_forward, "c_tilt": c_tilt, "c_height": c_height,
        }

        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# quick manual smoke test: random policy for a few steps
if __name__ == "__main__":
    env = HexapodEnv()
    obs, _ = env.reset()
    print("obs dim:", obs.shape, "action dim:", env.action_space.shape)
    total = 0.0
    for _ in range(200):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print(f"ran {env._step_count} steps, return={total:.3f}, "
          f"last fwd_vel={info['fwd_vel']:.3f} z={info['z']:.3f}")
    env.close()
