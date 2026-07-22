#!/usr/bin/env python3
"""
teleop_node.py  —  keyboard → /cmd_vel
======================================

The ROS2 version of the pynput block in main.py. Reads W/A/S/D from the terminal
and publishes a geometry_msgs/Twist on /cmd_vel at 60 Hz:

    W / S  ->  linear.x  = +1 / -1   (forward / backward)
    A / D  ->  angular.z = +1 / -1   (turn left / right)

Uses a raw-terminal reader (no pynput dependency) so it works over SSH. Because it
only ever publishes /cmd_vel, you can delete this node and drive the robot from
`ros2 topic pub /cmd_vel ...`, a joystick, or the RL policy — the gait node cannot
tell the difference.

Run:  ros2 run spider_control teleop_node    (must be a real terminal for key input)
"""
import sys
import termios
import tty
import select
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

HELP = """
teleop_node — hold keys to drive (release ~0.3s to stop that axis)
  W / S : forward / backward      A / D : turn left / right
  space : stop        x : quit
"""


class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_node")
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.fwd = 0.0
        self.turn = 0.0
        self._decay_frames = 0
        self.create_timer(1.0 / 60.0, self.tick)
        self.get_logger().info(HELP)
        threading.Thread(target=self._key_loop, daemon=True).start()

    def tick(self):
        # simple decay so a released key eases back to zero (like the ramp in main.py)
        if self._decay_frames > 0:
            self._decay_frames -= 1
        else:
            self.fwd *= 0.85
            self.turn *= 0.85
            if abs(self.fwd) < 1e-2:
                self.fwd = 0.0
            if abs(self.turn) < 1e-2:
                self.turn = 0.0
        msg = Twist()
        msg.linear.x = float(self.fwd)
        msg.angular.z = float(self.turn)
        self.pub.publish(msg)

    def _key_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    c = sys.stdin.read(1).lower()
                    if c == "x":
                        rclpy.shutdown()
                        break
                    elif c == "w":
                        self.fwd, self._decay_frames = 1.0, 12
                    elif c == "s":
                        self.fwd, self._decay_frames = -1.0, 12
                    elif c == "a":
                        self.turn, self._decay_frames = 1.0, 12
                    elif c == "d":
                        self.turn, self._decay_frames = -1.0, 12
                    elif c == " ":
                        self.fwd = self.turn = 0.0
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rclpy.init()
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
