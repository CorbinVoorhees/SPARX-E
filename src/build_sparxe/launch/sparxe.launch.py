from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # Instantiate phidgets_spatial directly (instead of including the stock
    # spatial-launch.py) so we can disable the onboard AHRS gyro-bias
    # auto-compensation. With the stock threshold=1.0, motor vibration during a
    # drive fools the device's at-rest heuristic and it re-zeros the gyro onto a
    # moving reading, latching a false angular rate that persists after the
    # motors stop (yaw ramps/keeps rotating). threshold=0.0 disables it — the
    # nav filter owns bias estimation.
    phidgets_params = {
        'use_orientation': False,
        'spatial_algorithm': 'ahrs',
        'ahrs_angular_velocity_threshold': 0.0,
        'ahrs_angular_velocity_delta_threshold': 0.1,
        'ahrs_acceleration_threshold': 0.1,
        'ahrs_mag_time': 10.0,
        'ahrs_accel_time': 10.0,
        'ahrs_bias_time': 1.25,
        'heating_enabled': False,
        'hub_port': 0,
    }
    return LaunchDescription([
        ComposableNodeContainer(
            name='phidget_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='phidgets_spatial',
                    plugin='phidgets::SpatialRosI',
                    name='phidgets_spatial',
                    parameters=[phidgets_params]),
            ],
            output='both',
        ),
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter',
            output='screen',
            parameters=[{
                'use_mag': True,
                'publish_tf': True,
                'world_frame': 'enu',
            }],
            remappings=[
                ('imu/data_raw', '/imu/data_raw'),
                ('imu/mag', '/imu/mag'),
                ('imu/data', '/imu/data')
            ]
        ),
        # Node(
        #     package='camera',
        #     executable='camera_stream_node',
        #     name='camera',
        # ),
        Node(
            package='status_updater',
            executable='status_updater_node',
            name='status_updater',
        ),
        Node(
            package='motor_commander',
            executable='motor_commander_node',
            name='motor_commander',
        ),
        Node(
            package='bridges',
            executable='arduino_bridge_node',
            name='arduino_bridge',
        ),
    ])
