# ros2_ws — Hexapod ROS2 workspace

A ROS2 (Humble, ament_python) wrapper around the MuJoCo hexapod. Read
`../docs/ROS2_CONCEPTS.md` first — it explains every node here by mapping it onto the
original single-process `main.py`.

## Node graph

```
teleop_node ──/cmd_vel──► gait_node ──/joint_cmd──► sim_node
                              ▲                         │
                              └──/joint_states──────────┘
                          (sim also pubs /imu, /touch)

# RL variant: policy_node replaces gait_node
sim_node ──/joint_states,/imu,/touch──► policy_node ──/joint_cmd──► sim_node
```

| Node | File | Subscribes | Publishes | Services |
|------|------|-----------|-----------|----------|
| `sim_node`    | `sim_node.py`    | `/joint_cmd` | `/joint_states` `/imu` `/touch` | — |
| `gait_node`   | `gait_node.py`   | `/cmd_vel` `/joint_states` | `/joint_cmd` | `/stand` `/idle` |
| `teleop_node` | `teleop_node.py` | (keyboard) | `/cmd_vel` | — |
| `policy_node` | `policy_node.py` | `/joint_states` `/imu` `/touch` | `/joint_cmd` | — |

## Build & run (needs Ubuntu 22.04 + ROS2 Humble)

```bash
# system deps: MuJoCo python is pulled from pip inside your ROS2 env
pip install "mujoco>=3.0.0" numpy
# to run the RL policy node you also need: pip install "stable-baselines3>=2.0"

cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# classic tripod gait, keyboard driven:
ros2 launch spider_control spider_sim.launch.py

# trained RL policy driving instead:
ros2 launch spider_control spider_policy.launch.py \
    model_zip:=$(pwd)/../rl/ppo_hexapod.zip
```

> This scaffold is written to build cleanly on a real ROS2 install. It is NOT
> buildable inside the Claude sandbox (no ROS2 there) — the RL side in `../rl/` is
> pure Python and is what we verify directly.

## Introspection cheatsheet

```bash
ros2 node list                    # who is running
ros2 topic list                   # the channels
ros2 topic echo /joint_states     # watch data stream
ros2 topic hz /joint_cmd          # confirm 60 Hz
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0}}"  # drive without teleop
ros2 service call /stand std_srvs/srv/Trigger
ros2 param set /gait_node control_hz 50.0
rqt_graph                         # draw the live node graph
```
