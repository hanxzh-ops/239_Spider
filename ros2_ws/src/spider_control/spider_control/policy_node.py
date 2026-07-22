#!/usr/bin/env python3
"""
policy_node.py  —  trained RL policy → /joint_cmd
=================================================

Drop-in REPLACEMENT for gait_node once you have a trained PPO policy from `rl/`.
It subscribes to the same sensor topics sim_node publishes, assembles the SAME
observation vector the Gymnasium env uses at training time, runs the network, and
publishes 18 joint targets on /joint_cmd.

    /joint_states + /imu + /touch  ─►  policy_node  ─►  /joint_cmd  ─►  sim_node

Because the observation/action contract here mirrors rl/hexapod_env.py exactly, a
policy that walks in the Gym env walks here too — that is your sim-to-node transfer.

Params:
    model_zip : path to the SB3 .zip saved by rl/train.py
Run:
    ros2 run spider_control policy_node --ros-args -p model_zip:=/path/to/ppo_hexapod.zip
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Imu

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from assets.config import JOINTS, JOINT_RANGES, DEFAULT_JOINT_POS   # noqa: E402


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.declare_parameter("model_zip", os.path.join(_REPO, "rl", "ppo_hexapod.zip"))
        self.declare_parameter("control_hz", 60.0)
        model_zip = self.get_parameter("model_zip").value
        self.dt = 1.0 / float(self.get_parameter("control_hz").value)

        # load the trained policy (import here so the node fails loudly if SB3 missing)
        from stable_baselines3 import PPO
        if not os.path.exists(model_zip):
            self.get_logger().error(f"no policy at {model_zip} — train with rl/train.py first")
        self.model = PPO.load(model_zip, device="cpu")
        self.get_logger().info(f"loaded policy {model_zip}")

        # observation buffers, updated by subscriptions
        self.jpos = np.array([DEFAULT_JOINT_POS[j] for j in JOINTS], dtype=np.float32)
        self.jvel = np.zeros(18, dtype=np.float32)
        self.quat = np.array([1, 0, 0, 0], dtype=np.float32)
        self.gyro = np.zeros(3, dtype=np.float32)
        self.touch = np.zeros(6, dtype=np.float32)

        # joint limit vectors for scaling policy output [-1,1] -> radians
        self.lo = np.array([JOINT_RANGES[j][0] for j in JOINTS], dtype=np.float32)
        self.hi = np.array([JOINT_RANGES[j][1] for j in JOINTS], dtype=np.float32)

        self.pub_cmd = self.create_publisher(Float64MultiArray, "joint_cmd", 10)
        self.create_subscription(JointState, "joint_states", self.on_js, 10)
        self.create_subscription(Imu, "imu", self.on_imu, 10)
        self.create_subscription(Float64MultiArray, "touch", self.on_touch, 10)
        self.create_timer(self.dt, self.tick)

    def on_js(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        for i, j in enumerate(JOINTS):
            if j in idx:
                self.jpos[i] = msg.position[idx[j]]
                if idx[j] < len(msg.velocity):
                    self.jvel[i] = msg.velocity[idx[j]]

    def on_imu(self, msg: Imu):
        self.quat[:] = [msg.orientation.w, msg.orientation.x,
                        msg.orientation.y, msg.orientation.z]
        self.gyro[:] = [msg.angular_velocity.x, msg.angular_velocity.y,
                        msg.angular_velocity.z]

    def on_touch(self, msg: Float64MultiArray):
        if len(msg.data) == 6:
            self.touch[:] = msg.data

    def tick(self):
        # MUST match rl/hexapod_env.py _get_obs() ordering
        obs = np.concatenate([self.jpos, self.jvel, self.quat, self.gyro, self.touch])
        action, _ = self.model.predict(obs, deterministic=True)
        # policy acts in [-1,1]; map to joint ranges
        targets = self.lo + (np.clip(action, -1, 1) + 1.0) * 0.5 * (self.hi - self.lo)
        out = Float64MultiArray()
        out.data = [float(v) for v in targets]
        self.pub_cmd.publish(out)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
