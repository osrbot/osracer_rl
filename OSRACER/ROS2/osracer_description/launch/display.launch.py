from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from xacro import process_file


def generate_launch_description():
    urdf = Path(get_package_share_directory('osracer_description')) / 'urdf' / 'osracer_arc_e01_prtnew_asm.urdf'
    description = process_file(str(urdf)).toxml()
    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher', parameters=[{'robot_description': description}]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui'),
        Node(package='rviz2', executable='rviz2'),
    ])
