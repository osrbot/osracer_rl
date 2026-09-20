"""Launch the racing policy node in shadow mode by default."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('checkpoint_path', description='Frozen policy checkpoint JSON'),
        DeclareLaunchArgument('racing_root', default_value='',
                              description='Path to the racing project that provides the racing package'),
        DeclareLaunchArgument('shadow_mode', default_value='true',
                              description='true computes commands without publishing'),
        DeclareLaunchArgument('max_speed_mps', default_value='1.0'),
        DeclareLaunchArgument('max_steering_rad', default_value='0.3'),
        DeclareLaunchArgument('ackermann_topic', default_value='/race/raw_ackermann_cmd'),
    ]
    node = Node(
        package='osracer_policy',
        executable='policy_node',
        name='osracer_policy_node',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
            'racing_root': LaunchConfiguration('racing_root'),
            'shadow_mode': LaunchConfiguration('shadow_mode'),
            'max_speed_mps': LaunchConfiguration('max_speed_mps'),
            'max_steering_rad': LaunchConfiguration('max_steering_rad'),
            'ackermann_topic': LaunchConfiguration('ackermann_topic'),
        }],
    )
    return LaunchDescription(arguments + [node])
