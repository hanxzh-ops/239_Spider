"""
spider_sim.launch.py — bring up the classic (tripod-gait) hexapod stack.

    ros2 launch spider_control spider_sim.launch.py

Starts sim_node + gait_node + teleop_node. Drive with W/A/S/D in the terminal that
launched teleop. Swap to the RL policy with spider_policy.launch.py.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="spider_control", executable="sim_node",
             name="sim_node", output="screen"),
        Node(package="spider_control", executable="gait_node",
             name="gait_node", output="screen"),
        Node(package="spider_control", executable="teleop_node",
             name="teleop_node", output="screen"),
    ])
