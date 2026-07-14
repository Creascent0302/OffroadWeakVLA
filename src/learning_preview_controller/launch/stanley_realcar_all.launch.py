from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    can_interface = LaunchConfiguration("can_interface")
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "can_interface",
            default_value="can1",
            description="SocketCAN interface name, for example can0 or can1",
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value="",
            description="Stanley config file; empty means stanley_params.yaml",
        ),
        Node(
            package="ros2_socketcan",
            executable="to_can",
            name="to_can_node",
            output="screen",
            parameters=[{"interface": can_interface}],
        ),
        Node(
            package="ros2_socketcan",
            executable="from_can",
            name="from_can_node",
            output="screen",
            parameters=[{"interface": can_interface}],
        ),
        Node(
            package="cav_bywire_lvtan",
            executable="cav_bywire_lvtan",
            name="cav_bywire_lvtan",
            output="screen",
            parameters=[{
                "can_control_mode": "equivalent_speed_steer",
            }],
        ),
        Node(
            package="learning_preview_controller",
            executable="stanley_controller",
            name="stanley_controller",
            output="screen",
            parameters=[{
                "config_file": config_file,
                "calculate_control_on_start": False,
                "enable_start_panel": True,
                "publish_plot_samples": True,
            }],
        ),
    ])
