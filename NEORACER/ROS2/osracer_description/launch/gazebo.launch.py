from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from xacro import process_file


def generate_launch_description():
    package_share = Path(get_package_share_directory('osracer_description'))
    urdf = package_share / 'urdf' / 'osracer_arc_e01_prtnew_asm.urdf'
    description = process_file(str(urdf)).toxml()
    gz_launch = Path(get_package_share_directory('ros_gz_sim')) / 'launch' / 'gz_sim.launch.py'
    actions = [
        DeclareLaunchArgument('gz_args', default_value='-r empty.sdf'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gz_launch)), launch_arguments={'gz_args': LaunchConfiguration('gz_args')}.items()),
        Node(package='robot_state_publisher', executable='robot_state_publisher', parameters=[{'robot_description': description, 'use_sim_time': True}]),
        Node(package='ros_gz_sim', executable='create', arguments=['-name', 'osracer_arc_e01_prtnew_asm', '-topic', 'robot_description'], output='screen'),
    ]
    return LaunchDescription(actions)
