from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value="",
            description="Controller config file; empty means preview_params.yaml",
        ),
        Node(
            package="learning_preview_controller",
            executable="controller_node",
            name="learning_preview_controller",
            output="screen",
            parameters=[{
                "config_file": config_file,
                "calculate_control_on_start": False,
                "enable_start_panel": True,
                "publish_plot_samples": True,
            }],
        ),
    ])
