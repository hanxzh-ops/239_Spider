# ROS2 for the Hexapod — Concepts Mapped to *Your* Code

This is a learning guide. It teaches the core ROS2 ideas (nodes, topics, services,
actions, parameters, the graph) by mapping each one onto code you already have in
this repo. Read it top-to-bottom once; then use it as a reference when you build the
`spider_control` package in `ros2_ws/`.

> **Where you are now:** your simulation is a *single Python process*. `main.py`
> loads the model, runs a 60 Hz loop, reads the keyboard, calls the controller,
> steps physics, and draws the viewer — all in one file, one thread, one call stack.
>
> **Where ROS2 takes you:** the *same* pieces become *separate programs* ("nodes")
> that talk over named "topics." Nothing about the robot changes — you are just
> cutting the monolith along its natural seams. The seams already exist in your
> code; ROS2 just makes them into network boundaries.

---

## 1. The one mental model that matters

ROS2 is a **publish/subscribe message bus** plus a few extras. Picture it like this:

```
        cmd_vel (Twist)                joint_cmd (Float64MultiArray)
teleop ───────────────► gait ─────────────────────────────────────► sim
  ▲                       ▲                                            │
  │                       │        joint_states / imu / touch         │
  └───────────────────────┴────────────────◄───────────────────────────┘
                                (sensor topics)
```

- A **node** is a single-purpose program (one box above).
- A **topic** is a named, typed channel (the arrows). Anyone can *publish* to it;
  anyone can *subscribe*. Publishers and subscribers never reference each other by
  name — only by topic. That decoupling is the whole point: you can swap `teleop`
  for an `rl_policy` node and nothing else changes, because both just publish
  `cmd_vel`.
- The **message type** (`Twist`, `Float64MultiArray`, …) is the contract on a topic.

Everything else (services, actions, parameters, TF, launch) is convenience built on
top of that bus.

---

## 2. Your current code, already a node graph in disguise

Look at what `main.py`'s loop does each frame and notice how cleanly it splits:

| `main.py` responsibility                        | Becomes ROS2 node | Publishes / Subscribes                        |
|-------------------------------------------------|-------------------|-----------------------------------------------|
| read `_held` keys → `fwd`, `turn`               | **teleop_node**   | pub `cmd_vel`                                 |
| `controller.set_velocity` + `controller.step`   | **gait_node**     | sub `cmd_vel`; pub `joint_cmd`; sub `joint_states` |
| `mj.set_joint_targets`, `mj.step`, `viewer.sync`| **sim_node**      | sub `joint_cmd`; pub `joint_states`, `imu`, `touch` |
| (later) trained RL policy                        | **policy_node**   | sub `joint_states`+sensors; pub `joint_cmd` OR `cmd_vel` |

The dashed seams in the ASCII diagram of your `README.md` (Keyboard → Controller →
MujocoInterface → Physics) are *exactly* the topic boundaries. You already designed
the system as a pipeline; ROS2 just names the pipes.

### Why bother splitting it?

- **Swap parts without touching the rest.** Replace `teleop_node` with `policy_node`
  and the gait/sim never know. This is precisely how you'll drop in the RL policy.
- **Run parts on different machines.** `sim_node` on a workstation, `policy_node` on
  a Jetson, teleop on a laptop — same topics, different hosts.
- **Record and replay.** `ros2 bag record /joint_states /imu` captures a run you can
  replay offline. Great for debugging a gait or training data.
- **Introspect live.** `ros2 topic echo /cmd_vel` prints commands as they fly; `rqt_graph`
  draws the live graph. No print statements needed.

---

## 3. Nodes

A node is just a class that inherits `rclpy.node.Node`. Minimal shape:

```python
import rclpy
from rclpy.node import Node

class GaitNode(Node):
    def __init__(self):
        super().__init__("gait_node")          # the node's name in the graph
        # ... create publishers, subscribers, timers, parameters here ...

def main():
    rclpy.init()
    node = GaitNode()
    rclpy.spin(node)                            # process callbacks until Ctrl-C
    node.destroy_node()
    rclpy.shutdown()
```

`rclpy.spin(node)` is the ROS2 equivalent of your `while viewer.is_running():` loop —
it hands control to ROS2, which then calls *your* callbacks (timer ticks, incoming
messages) as events arrive. You stop writing the loop; you write the reactions.

---

## 4. Topics (the workhorse — 90% of what you'll use)

A topic is a named bus with a fixed message type. **Publish** = send; **subscribe** =
register a callback that fires on every message.

### 4a. Publisher (how `sim_node` emits sensor data)

```python
from sensor_msgs.msg import JointState

self.pub_js = self.create_publisher(JointState, "joint_states", 10)  # 10 = queue depth

# inside a 60 Hz timer callback, after stepping physics:
msg = JointState()
msg.header.stamp = self.get_clock().now().to_msg()
msg.name     = list(self.mj.jnt_qposadr.keys())        # your 18 joint names
msg.position = [float(self.mj.data.qpos[a]) for a in self.mj.jnt_qposadr.values()]
self.pub_js.publish(msg)
```

That `msg.position` list is literally what `mj_interface.get_joint_angles()` already
returns — you're just wrapping it in a message and putting it on the bus.

### 4b. Subscriber (how `gait_node` receives velocity commands)

```python
from geometry_msgs.msg import Twist

self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)

def on_cmd_vel(self, msg: Twist):
    self.fwd  = msg.linear.x     # forward drive  [-1, 1]
    self.turn = msg.angular.z    # turn rate      [-1, 1]
    # exactly the fwd/turn your controller.set_velocity(fwd, turn) wants
```

`Twist` is the standard "velocity command" message: `linear.x/y/z` + `angular.x/y/z`.
For a ground robot you use `linear.x` (forward) and `angular.z` (yaw) and ignore the
rest. This is the *de-facto* interface every mobile robot in ROS2 speaks — using it
means `teleop_twist_keyboard`, nav stacks, and joysticks all work with your robot for
free.

### 4c. Timers = your control-rate clock

Your loop runs at 60 Hz with a `time.sleep`. In ROS2 you don't sleep; you ask for a
timer:

```python
self.dt = 1.0 / 60.0
self.create_timer(self.dt, self.control_step)   # calls control_step() at 60 Hz
```

`sim_node` gets a timer that steps physics + publishes sensors; `gait_node` gets a
timer that runs `controller.step(dt)` and publishes `joint_cmd`.

### Message types you'll use

| Purpose                     | Message type                    | Package         |
|-----------------------------|---------------------------------|-----------------|
| velocity command            | `geometry_msgs/Twist`           | `geometry_msgs` |
| 18 joint position targets   | `std_msgs/Float64MultiArray`    | `std_msgs`      |
| joint positions/velocities  | `sensor_msgs/JointState`        | `sensor_msgs`   |
| IMU (orientation + gyro)    | `sensor_msgs/Imu`               | `sensor_msgs`   |
| per-foot contact (6×)       | `std_msgs/Float64MultiArray` or `sensor_msgs/JointState`-style | `std_msgs` |

Your `hexapod.xml` already has the matching sensors: `torso_orientation` (framequat) →
`Imu.orientation`, `torso_gyro` → `Imu.angular_velocity`, `touch_leg0..5` → the contact
array.

---

## 5. Services (request → single reply)

A **service** is a synchronous call-and-response, for occasional one-off requests —
*not* streaming data. In your code, the discrete commands are the natural services:

| `controller.command(...)` call | Service |
|--------------------------------|---------|
| `"stand"`                      | `/stand`   (Trigger) |
| `"idle"`                       | `/idle`    (Trigger) |
| `"body_up"` / `"body_down"`    | `/set_body_height` (custom, takes a float) |
| calibrate after warm-up        | `/calibrate` (Trigger) |

`std_srvs/Trigger` is the "just do it, no args" service. Rule of thumb: **continuous
stream → topic; occasional command with a confirmation → service.** You would *not*
put `cmd_vel` on a service (it streams at 60 Hz); you *would* put "stand up now" on one.

```python
from std_srvs.srv import Trigger
self.create_service(Trigger, "stand", self.on_stand)

def on_stand(self, request, response):
    self.controller.command("stand")
    response.success = True
    response.message = "standing"
    return response
```

Call it from the shell: `ros2 service call /stand std_srvs/srv/Trigger`.

---

## 6. Actions (long goals with feedback — your future navigation layer)

An **action** is for tasks that take *time* and stream *progress*: goal → periodic
feedback → final result, and it's *cancelable*. This is exactly the shape of the
"path mapping and autonomous control" goal in your project brief.

> "Walk to waypoint (2.0, 1.0)" is an action:
> **goal** = target pose; **feedback** = distance remaining, current pose, streamed
> each second; **result** = "arrived" / "aborted"; and you can **cancel** mid-walk.

You won't build actions in the first pass, but keep the slot in mind: your
`nav_node` (later) will host a `NavigateToPose`-style action server that decomposes a
waypoint into a stream of `cmd_vel` commands — the same `cmd_vel` your `gait_node`
already consumes. That is how the RL walking policy and a classical planner compose:
planner → `cmd_vel` → policy/gait → `joint_cmd` → sim.

Quick decision guide:

| You need…                                   | Use a… |
|---------------------------------------------|--------|
| a value many times per second               | topic |
| to trigger something once, get a yes/no     | service |
| to start a long job, watch progress, cancel | action |
| a tunable constant (kp, step height, rate)  | parameter |

---

## 7. Parameters (your config files, but live)

Right now your tuning constants live in `controller.py` (`STEP_FREQ`, `STEP_LEN`,
`STEP_H`, `RAMP_ACCEL`, …) and `assets/config.py`. In ROS2 these become **node
parameters**: declared at startup, settable from the command line or a YAML file, and
changeable *at runtime*.

```python
self.declare_parameter("step_freq", 0.9)
self.declare_parameter("step_len",  0.05)
freq = self.get_parameter("step_freq").value
```

Then `ros2 param set /gait_node step_freq 1.2` retunes the gait *without restarting*.
This is the ROS2 answer to "edit config, re-run." Every constant in your
`## 11. Configuration and Tuning` README table is a parameter candidate.

---

## 8. The transform tree (TF2) — where things are

TF2 is a system for tracking coordinate frames over time: `world → torso → leg0_base →
… → tip_leg0`. Your `fk.py` already computes `foot-in-torso`, and `feet_in_world()`
already does `world ← torso` using the torso quaternion. TF2 is the standardized,
timestamped, queryable version of that math, shared across all nodes. When you add
mapping/navigation, TF lets the planner ask "where is the torso in the map frame *at
the time that scan was taken*" and get a correct answer. For now, just know: **the
frame relationships you hand-code in FK are what TF formalizes.**

---

## 9. Launch files — starting the whole graph at once

Instead of opening four terminals, one launch file starts every node:

```python
# spider_sim.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package="spider_control", executable="sim_node",    name="sim_node"),
        Node(package="spider_control", executable="gait_node",   name="gait_node"),
        Node(package="spider_control", executable="teleop_node", name="teleop_node"),
    ])
```

`ros2 launch spider_control spider_sim.launch.py` brings the robot up. Swap
`teleop_node` for `policy_node` and you're running the RL policy instead — same sim,
same gait-less path, one line changed.

---

## 10. How a ROS2 package is laid out (what you'll build next)

A Python ROS2 package (ament_python) has a fixed skeleton. The `ros2_ws/` in this repo
scaffolds exactly this:

```
ros2_ws/
└── src/
    └── spider_control/
        ├── package.xml              # metadata + dependencies (rclpy, std_msgs, …)
        ├── setup.py                 # entry_points map node names → main() functions
        ├── setup.cfg
        ├── resource/spider_control  # ament marker file
        ├── spider_control/
        │   ├── __init__.py
        │   ├── sim_node.py          # MuJoCo + sensor pub + joint_cmd sub
        │   ├── gait_node.py         # your controller.py, wrapped
        │   ├── teleop_node.py       # keyboard → cmd_vel
        │   └── policy_node.py       # trained RL policy → joint_cmd
        └── launch/
            └── spider_sim.launch.py
```

The magic line is in `setup.py`:

```python
entry_points={"console_scripts": [
    "sim_node    = spider_control.sim_node:main",
    "gait_node   = spider_control.gait_node:main",
    "teleop_node = spider_control.teleop_node:main",
    "policy_node = spider_control.policy_node:main",
]}
```

That is what lets `ros2 run spider_control gait_node` find and start your code.

### Build & run (on a machine with ROS2 Humble installed)

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch spider_control spider_sim.launch.py
```

> **Note:** ROS2 needs a real ROS2 install (Ubuntu 22.04 + Humble, or a container).
> The scaffold is fully written and correct, but it won't `colcon build` inside this
> chat sandbox — it's meant to run on your ROS2 machine. The pure-Python RL side
> (`rl/`) has *no* ROS2 dependency and runs anywhere, which is why we verify that part
> here directly.

---

## 11. Putting the RL policy into the graph

Once you've trained a policy (next stage of the project), it slots in two ways —
you pick based on what the policy outputs:

1. **Policy outputs joint targets (your chosen setup).** `policy_node` subscribes to
   `joint_states` + sensors, runs the network, publishes `joint_cmd`. It *replaces*
   `gait_node` entirely. Launch line swaps `gait_node` → `policy_node`.

   ```
   sensors ──► policy_node ──► joint_cmd ──► sim_node
   ```

2. **Policy outputs a velocity (a future higher-level controller).** It publishes
   `cmd_vel` and keeps `gait_node` underneath. This is the "residual/hierarchical"
   arrangement.

Because everything is decoupled by topics, the trained policy from `rl/` drops in
without touching `sim_node` or the model. The `env` you train against mirrors
`sim_node`'s observation/action contract, so the sim-to-node transfer is a copy of the
same numbers.

---

## 12. A 20-minute hands-on path (once ROS2 is installed)

1. `ros2 run spider_control sim_node` — start just the sim node.
2. In another terminal: `ros2 topic list` — see `/joint_states`, `/imu`, `/touch`.
3. `ros2 topic echo /joint_states` — watch live joint data stream.
4. `ros2 topic pub /joint_cmd std_msgs/msg/Float64MultiArray "{data: [...18 values...]}"`
   — hand-drive the robot with a single command.
5. Start `gait_node` and `teleop_node`; drive with the keyboard again — but now every
   piece is a separate, inspectable program.
6. `rqt_graph` — see the picture from §1 drawn from your *actual running system*.

That progression — echo a topic, publish to a topic, then watch the graph — is the
fastest way to make the abstract concepts click.

---

## Glossary (quick reference)

| Term | One-liner | Your analog |
|------|-----------|-------------|
| **node** | one single-purpose program | one of: teleop / controller / sim |
| **topic** | named, typed, many-to-many channel | the arrows between your modules |
| **message** | the data schema on a topic | the `fwd,turn` tuple / joint dict |
| **publisher** | sends on a topic | `mj.set_joint_targets` caller |
| **subscriber** | callback on each message | controller reading key state |
| **service** | one request → one reply | `controller.command("stand")` |
| **action** | long goal + feedback + cancel | "walk to waypoint" (future) |
| **parameter** | live-tunable constant | `STEP_FREQ`, `kp`, `STEP_H` |
| **timer** | periodic callback | your 60 Hz loop body |
| **TF2** | timestamped frame tree | `fk.py` / `feet_in_world()` |
| **launch file** | starts many nodes at once | running `main.py` |
| **rclpy** | the Python ROS2 client library | (new) |
| **colcon** | the build tool | (new) |

Next: open `ros2_ws/src/spider_control/` to see every node above written out against
your real controller, then `rl/` for the Gymnasium environment and PPO training.
