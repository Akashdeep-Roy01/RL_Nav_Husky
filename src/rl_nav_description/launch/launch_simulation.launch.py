import os
import xacro
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    # Process the URDF file
    pkg_path = os.path.join(get_package_share_directory('rl_nav_description'))
    xacro_file = os.path.join(pkg_path,'description','husky.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    default_rviz_config_path = os.path.join(pkg_path, 'config','config.rviz')
    world_path=os.path.join(pkg_path,'worlds/maze7.sdf')

    # Create a robot_state_publisher node
    params = {'robot_description': robot_description_config.toxml(),"use_sim_time":True}
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )

    # Include the Gazebo launch file, provided by the gazebo_ros package
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
                    launch_arguments={'use_sim_time': 'true'}.items()
             )

    # Run the spawner node from the gazebo_ros package. 
    spawn_robot = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'my_bot'],output='screen')
    # Launch!
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',default_value='True',description='Use sim time if true'),
        DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path,
                                            description='Absolute path to rviz config file'),
        DeclareLaunchArgument(name='world', default_value=world_path,description='world_path'),
        DeclareLaunchArgument(name='gui', default_value="False",description='gz_server'),
        gazebo,
        spawn_robot,
        node_robot_state_publisher,
        rviz_node
    ])