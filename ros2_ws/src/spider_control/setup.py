import os
from glob import glob
from setuptools import find_packages, setup

package_name = "spider_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # install launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jaeger",
    maintainer_email="hanxzh@berkeley.edu",
    description="Hexapod MuJoCo sim + tripod gait + RL policy as ROS2 nodes.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # ros2 run spider_control <name>
            "sim_node    = spider_control.sim_node:main",
            "gait_node   = spider_control.gait_node:main",
            "teleop_node = spider_control.teleop_node:main",
            "policy_node = spider_control.policy_node:main",
        ],
    },
)
