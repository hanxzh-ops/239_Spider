"""
spider_policy.launch.py — run the TRAINED RL policy instead of the tripod gait.

    ros2 launch spider_control spider_policy.launch.py model_zip:=/abs/path/ppo_hexapod.zip

Same sim_node; gait_node is replaced by policy_node. No teleop — the policy drives.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_zip = LaunchConfiguration("model_zip")
    return LaunchDescription([
        DeclareLaunchArgument("model_zip", default_value="ppo_hexapod.zip",
                              description="path to the SB3 PPO .zip"),
        Node(package="spider_control", executable="sim_node",
             name="sim_node", output="screen"),
        Node(package="spider_control", executable="policy_node",
             name="policy_node", output="screen",
             parameters=[{"model_zip": model_zip}]),
    ])
