from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")

    default_config = PathJoinSubstitution([
        FindPackageShare("orchard_nav_system"),
        "config",
        "params.yaml"
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="Path to config YAML file."
        ),

        Node(
            package="orchard_nav_system",
            executable="realsense_bag_publisher",
            name="realsense_bag_publisher",
            output="screen",
            parameters=[config_file],
        ),

        Node(
            package="orchard_nav_system",
            executable="yolo_seg_node",
            name="yolo_seg_node",
            output="screen",
            parameters=[config_file],
        ),

        Node(
            package="orchard_nav_system",
            executable="tree_distance_node",
            name="tree_distance_node",
            output="screen",
            parameters=[config_file],
        ),

        Node(
            package="orchard_nav_system",
            executable="robot_control_node",
            name="robot_control_node",
            output="screen",
            parameters=[config_file],
        ),
        
        Node(
            package="orchard_nav_system",
            executable="visualization_node",
            name="visualization_node",
            output="screen",
            parameters=[config_file],
        ),
    ])
