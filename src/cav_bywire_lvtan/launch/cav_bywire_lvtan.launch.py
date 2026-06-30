from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cav_bywire_lvtan',
            executable='cav_bywire_lvtan',
            name='cav_bywire_lvtan',
            output='screen',
            parameters=[
                {'veh_id': 11710004},
                {'veh_name': 'lvtan_4'},
                {'veh_type': 71},
                {'move_x': 0.4},
                {'move_y': 0.3},
            ]
        ),
        Node(
            package='ros2_socketcan',
            executable='to_can',
            name='to_can_node',
            parameters=[{'interface': 'can1'}]
        ),
        Node(
            package='ros2_socketcan',
            executable='from_can',
            name='from_can_node',
            parameters=[{'interface': 'can1'}]
        ),
    ])


# ros2 launch ros2_socketcan socketcan_bridge_launch.py interface:=can1