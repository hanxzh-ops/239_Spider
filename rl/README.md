# rl — Learn to walk with PPO

Reinforcement-learning stack for the hexapod. The policy learns a walking gait
**from scratch** (no IK, no tripod schedule): it directly commands all 18 joint
position targets and is rewarded for moving the torso forward while staying upright.

## Files

| File | What it is |
|------|-----------|
| `hexapod_env.py` | Gymnasium env wrapping `assets/hexapod.xml`. 49-dim obs, 18-dim action, walking reward. |
| `train.py` | Trains PPO (Stable-Baselines3). Saves `ppo_hexapod.zip` + `vecnormalize.pkl`. |
| `eval.py` | Loads the policy, rolls it out, opens the viewer (or prints distance headless). |

## Install

```bash
pip install "mujoco>=3.0.0" "gymnasium>=0.29" "stable-baselines3>=2.0" numpy
```

## The contract (obs / action)

```
action  Box(18,) in [-1,1]  ->  linearly mapped to each joint's [lo,hi]  ->  data.ctrl
obs     Box(49,) = jpos(18) + jvel(18) + torso_quat(4) + gyro(3) + touch(6)
reward  forward_velocity + alive - ctrl_cost - height_error - tilt
done    torso z < 0.045 m  OR  torso tips past ~70° (up_z < 0.3)
```

This ordering is mirrored exactly in `ros2_ws/.../policy_node.py`, so a trained
policy transfers straight into the ROS2 graph.

## Workflow

```bash
cd rl

# 1. sanity check the env with a random policy
python hexapod_env.py

# 2. quick pipeline smoke test (seconds)
python train.py --timesteps 5000 --n-envs 2

# 3. real training run (CPU; ~1-3M steps to see a gait emerge)
python train.py --timesteps 2000000 --n-envs 8
tensorboard --logdir logs        # watch ep_rew_mean climb

# 4. watch it walk
python eval.py                    # needs a display for the viewer
python eval.py --no-render        # headless: prints distance walked
```

## Tuning notes (where to look when it won't walk)

Reward weights live at the top of `HexapodEnv.__init__` (`w_forward`, `w_alive`,
`w_ctrl`, `w_height`, `w_tilt`, `z_target`, `fall_z`). Common first moves:

- **Robot dives forward and face-plants** → raise `w_tilt` / `w_height`, or lower
  `w_forward`.
- **Robot freezes to farm the alive bonus** → lower `w_alive`, raise `w_forward`.
- **Jittery, high-energy motion** → raise `w_ctrl` (action-magnitude penalty).
- **Never gets moving at all** → train longer; 1M steps is often just the start.
  Try `net_arch=[256,256]` (already set) and more parallel envs.

## Roadmap toward the project goal (path mapping + autonomous control)

1. **This stage:** flat-ground forward walking. ✅ scaffold ready.
2. **Velocity-conditioned policy:** add commanded `(vx, wz)` to the observation and
   reward tracking it → the policy becomes a `cmd_vel` follower (drop-in for
   `gait_node`).
3. **Navigation layer:** a planner/`nav_node` turns a target waypoint into `cmd_vel`
   (a ROS2 *action*, see `docs/ROS2_CONCEPTS.md` §6). Path mapping feeds the planner.
4. **Terrain + sim-to-real:** domain randomization in the env, then export to a
   physical hexapod.
