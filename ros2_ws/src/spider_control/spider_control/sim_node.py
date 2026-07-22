#!/usr/bin/env python3
"""
sim_node.py  —  MuJoCo physics as a ROS2 node
=============================================

This node OWNS the simulation. It is the ROS2 wrapper around your existing
`src/mj_interface.MujocoInterface`. It does three jobs on a fixed timer:

  1. apply the latest joint targets received on   /joint_cmd   (Float64MultiArray, 18)
  2. step MuJoCo physics `substeps` times (500 Hz physics under a 60 Hz control tick)
  3. publish sensor state:
         /joint_states  (sensor_msgs/JointState)   — qpos + qvel for the 18 joints
         /imu           (sensor_msgs/Imu)           — torso orientation + gyro
         /touch         (std_msgs/Float64MultiArray)— 6 foot contact forces

It is the ONLY node that imports mujoco. Every other node talks to it over topics,
so you can run the gait controller or the RL policy against the exact same sim
without either of them knowing MuJoCo exists.

Run:  ros2 run spider_control sim_node
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Imu

# --- locate the repo so we can reuse src/ and assets/ ------------------------
# ros2_ws/src/spider_control/spider_control/sim_node.py  ->  repo root is 4 up.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.mj_interface import MujocoInterface           # noqa: E402
from assets.config import JOINTS, DEFAULT_JOINT_POS    # noqa: E402


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node")

        # ---- parameters (tune live with `ros2 param set`) -------------------
        default_xml = os.path.join(_REPO, "assets", "hexapod.xml")
        self.declare_parameter("model_path", default_xml)
        self.declare_parameter("control_hz", 60.0)
        self.declare_parameter("warmup_frames", 400)

        xml_path = self.get_parameter("model_path").value
        control_hz = float(self.get_parameter("control_hz").value)
        self.dt = 1.0 / control_hz

        # ---- load sim -------------------------------------------------------
        self.mj = MujocoInterface(xml_path)
        self.substeps = max(1, round(self.dt / self.mj.model.opt.timestep))
        self.get_logger().info(
            f"loaded {xml_path}  physics={1/self.mj.model.opt.timestep:.0f}Hz "
            f"control={control_hz:.0f}Hz  substeps={self.substeps}")

        # settle into the default stance so the robot starts on its feet
        warm = int(self.get_parameter("warmup_frames").value)
        for _ in range(warm):
            self.mj.set_joint_targets(DEFAULT_JOINT_POS)
            for _ in range(self.substeps):
                self.mj.step()
        self.get_logger().info("warm-up complete — standing")

        # cache sensor addresses once
        self.joint_names = JOINTS
        self.qadr = [self.mj.jnt_qposadr[j] for j in JOINTS]
        self._latest_cmd = None   # last Float64MultiArray of 18 joint targets

        # ---- ROS2 wiring ----------------------------------------------------
        self.pub_js    = self.create_publisher(JointState, "joint_states", 10)
        self.pub_imu   = self.create_publisher(Imu, "imu", 10)
        self.pub_touch = self.create_publisher(Float64MultiArray, "touch", 10)

        self.create_subscription(Float64MultiArray, "joint_cmd", self.on_cmd, 10)
        self.create_timer(self.dt, self.tick)

    # -- receive joint targets from gait_node or policy_node ------------------
    def on_cmd(self, msg: Float64MultiArray):
        if len(msg.data) == len(self.joint_names):
            self._latest_cmd = np.asarray(msg.data, dtype=float)
        else:
            self.get_logger().warn(
                f"joint_cmd length {len(msg.data)} != {len(self.joint_names)}")

    # -- fixed-rate physics + publish -----------------------------------------
    def tick(self):
        # 1. apply latest command (hold last if none yet)
        if self._latest_cmd is not None:
            cmd = {j: float(self._latest_cmd[i])
                   for i, j in enumerate(self.joint_names)}
            self.mj.set_joint_targets(cmd)

        # 2. step physics
        for _ in range(self.substeps):
            self.mj.step()

        # 3. publish sensors
        now = self.get_clock().now().to_msg()
        d = self.mj.data

        js = JointState()
        js.header.stamp = now
        js.name = list(self.joint_names)
        js.position = [float(d.qpos[a]) for a in self.qadr]
        # qvel address == qposadr for hinge joints living after the freejoint;
        # use the joint dof address for correctness:
        js.velocity = [float(d.qvel[self.mj.model.jnt_dofadr[
            self.mj.model.joint(j).id]]) for j in self.joint_names]
        self.pub_js.publish(js)

        # IMU: torso orientation quat (w,x,y,z in MuJoCo) + gyro
        _, quat = self.mj.get_body_pose("torso")
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "torso"
        imu.orientation.w = float(quat[0])
        imu.orientation.x = float(quat[1])
        imu.orientation.y = float(quat[2])
        imu.orientation.z = float(quat[3])
        gyro = self._sensor("torso_gyro", 3)
        if gyro is not None:
            imu.angular_velocity.x = float(gyro[0])
            imu.angular_velocity.y = float(gyro[1])
            imu.angular_velocity.z = float(gyro[2])
        self.pub_imu.publish(imu)

        # touch: 6 contact forces
        touch = Float64MultiArray()
        touch.data = [float(self._sensor(f"touch_leg{i}", 1)[0]) for i in range(6)]
        self.pub_touch.publish(touch)

    # -- read a named MuJoCo sensor by name -----------------------------------
    def _sensor(self, name, dim):
        try:
            sid = self.mj.model.sensor(name).id
        except Exception:
            return None
        adr = self.mj.model.sensor_adr[sid]
        return self.mj.data.sensordata[adr:adr + dim]


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
