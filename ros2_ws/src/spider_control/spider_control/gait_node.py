#!/usr/bin/env python3
"""
gait_node.py  —  your tripod-gait controller as a ROS2 node
===========================================================

This wraps `src/controller.LocomotionController` WITHOUT changing it. The trick:
the controller wants a `mj_interface` object with `set_joint_targets`, `get_joint_angles`
and `get_body_pose`. Instead of giving it the real MuJoCo sim, we give it a tiny
SHIM that:

  * caches the joint_states / torso pose arriving on topics from sim_node, and
  * captures the joint targets the controller computes, so we can publish them.

So the identical gait math runs, but its inputs come from `/joint_states` and its
outputs go to `/joint_cmd` — pure message passing.

Flow:   /cmd_vel ─► gait_node ─► /joint_cmd ─► sim_node ─► /joint_states ─► gait_node
Run:    ros2 run spider_control gait_node
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.controller import LocomotionController        # noqa: E402
from assets.config import JOINTS, DEFAULT_JOINT_POS    # noqa: E402


class _TopicMjShim:
    """Looks like MujocoInterface to the controller, but is fed by ROS topics."""

    def __init__(self):
        self.angles = dict(DEFAULT_JOINT_POS)          # last /joint_states
        self.torso_pos = np.array([0.0, 0.0, 0.09])    # last torso pos (est.)
        self.torso_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_targets = dict(DEFAULT_JOINT_POS)     # controller output sink
        # controller reads model.opt.timestep only in main.py, not here.

    # -- controller INPUTS ----------------------------------------------------
    def get_joint_angles(self):
        return dict(self.angles)

    def get_body_pose(self, body_name="torso"):
        return self.torso_pos, self.torso_quat

    # -- controller OUTPUT (captured, not applied) ----------------------------
    def set_joint_targets(self, joint_map):
        self.last_targets.update(joint_map)


class GaitNode(Node):
    def __init__(self):
        super().__init__("gait_node")
        self.declare_parameter("control_hz", 60.0)
        self.dt = 1.0 / float(self.get_parameter("control_hz").value)

        self.shim = _TopicMjShim()
        self.controller = LocomotionController(self.shim)
        self.controller.calibrate()      # uses the default-pose shim values
        self.controller.command("stand")

        self.fwd = 0.0
        self.turn = 0.0

        self.pub_cmd = self.create_publisher(Float64MultiArray, "joint_cmd", 10)
        self.create_subscription(JointState, "joint_states", self.on_js, 10)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)

        # discrete commands as services (see ROS2_CONCEPTS.md §5)
        self.create_service(Trigger, "stand", self._srv("stand"))
        self.create_service(Trigger, "idle",  self._srv("idle"))

        self.create_timer(self.dt, self.tick)
        self.get_logger().info("gait_node up — subscribing /cmd_vel, publishing /joint_cmd")

    # -- feedback in ----------------------------------------------------------
    def on_js(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.shim.angles[name] = float(pos)

    def on_cmd_vel(self, msg: Twist):
        # Twist convention: linear.x = forward drive, angular.z = turn rate
        self.fwd = float(np.clip(msg.linear.x, -1.0, 1.0))
        self.turn = float(np.clip(msg.angular.z, -1.0, 1.0))

    # -- 60 Hz control tick ---------------------------------------------------
    def tick(self):
        self.controller.set_velocity(self.fwd, self.turn)
        self.controller.step(self.dt)              # fills shim.last_targets
        out = Float64MultiArray()
        out.data = [float(self.shim.last_targets[j]) for j in JOINTS]
        self.pub_cmd.publish(out)

    # -- service factory ------------------------------------------------------
    def _srv(self, cmd):
        def handler(request, response):
            self.controller.command(cmd)
            self.fwd = self.turn = 0.0
            response.success = True
            response.message = cmd
            return response
        return handler


def main():
    rclpy.init()
    node = GaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
