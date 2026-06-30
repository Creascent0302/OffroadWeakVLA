from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cav_bywire_lvtan",
            executable="cav_bywire_lvtan",
            name="cav_bywire_lvtan",
            output="screen",
        ),
        Node(
            package="learning_preview_controller",
            executable="controller_node",
            name="learning_preview_controller",
            output="screen",
        ),
    ])
