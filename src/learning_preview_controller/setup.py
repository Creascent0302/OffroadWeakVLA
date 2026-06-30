from glob import glob
import os
from setuptools import setup

package_name = "learning_preview_controller"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "cvxpy", "osqp"],
    zip_safe=True,
    maintainer="yuheng",
    maintainer_email="yuheng@example.com",
    description="ROS2 wrapper for Learning_preview_controller",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "controller_node = learning_preview_controller.controller_node:main",
        ],
    },
)
