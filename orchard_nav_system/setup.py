from setuptools import setup, find_packages
from glob import glob
import os

package_name = "orchard_nav_system"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="student",
    maintainer_email="student@example.com",
    description="Orchard navigation prototype using RGB-D data and YOLO segmentation.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "realsense_bag_publisher = orchard_nav_system.realsense_bag_publisher:main",
            "yolo_seg_node = orchard_nav_system.yolo_seg_node:main",
            "tree_distance_node = orchard_nav_system.tree_distance_node:main",
            "robot_control_node = orchard_nav_system.robot_control_node:main",
            "visualization_node = orchard_nav_system.visualization_node:main",
            "video_recorder_node = orchard_nav_system.video_recorder_node:main",
        ],
    },
)
