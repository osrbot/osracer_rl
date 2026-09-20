"""Run the frozen sensor-only racing policy on an OSRacer vehicle.

Subscribes: /scan (LaserScan), /odometry/filtered (Odometry), /race/safety_stop (Bool)
Publishes:  /race/raw_ackermann_cmd (AckermannDrive)

The node publishes through the vehicle's existing controller topic, so the
race bringup keeps its own safety_node and speed_profile_node in the loop.
shadow_mode computes and logs commands without publishing, which is the first
commissioning step on recorded bags or a stationary car.
"""
from __future__ import annotations

import math

import rclpy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from osracer_policy.checkpoint import CheckpointMismatch, load_actor
from osracer_policy.mapping import action_to_ackermann, build_observation


class PolicyNode(Node):
    def __init__(self):
        super().__init__('osracer_policy_node')
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('racing_root', '')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('ackermann_topic', '/race/raw_ackermann_cmd')
        self.declare_parameter('safety_stop_topic', '/race/safety_stop')
        self.declare_parameter('shadow_mode', True)
        self.declare_parameter('max_speed_mps', 1.0)
        self.declare_parameter('max_steering_rad', 0.3)
        self.declare_parameter('wheel_radius_m', 0.045)
        self.declare_parameter('scan_timeout_s', 0.2)
        self.declare_parameter('odom_timeout_s', 0.3)
        self.declare_parameter('shadow_log_period_s', 1.0)

        self.shadow = bool(self.get_parameter('shadow_mode').value)
        self.max_speed = float(self.get_parameter('max_speed_mps').value)
        self.max_steering = float(self.get_parameter('max_steering_rad').value)
        self.wheel_radius = float(self.get_parameter('wheel_radius_m').value)
        self.scan_timeout = float(self.get_parameter('scan_timeout_s').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout_s').value)
        self.log_period = float(self.get_parameter('shadow_log_period_s').value)

        try:
            self.actor, self.spec = load_actor(
                self.get_parameter('checkpoint_path').value,
                racing_root=self.get_parameter('racing_root').value or None)
        except (CheckpointMismatch, OSError, ValueError) as exc:
            # Fail closed: without a hash-matched actor the node must not drive.
            self.get_logger().fatal(f'Refusing to run policy: {exc}')
            raise

        self.scan = None
        self.odom = None
        self.scan_time = None
        self.odom_time = None
        self.safety_stop = True
        self.tick = 0
        self.last_published = (0., 0.)
        self.last_log = 0.

        if not self.shadow:
            self.cmd_pub = self.create_publisher(
                AckermannDrive, self.get_parameter('ackermann_topic').value, 10)
        else:
            self.cmd_pub = None
        self.create_subscription(Bool, self.get_parameter('safety_stop_topic').value,
                                 self.on_safety, 10)
        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value,
                                 self.on_scan, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value,
                                 self.on_odom, 10)
        self.get_logger().info(
            f'policy ready: {self.spec.get("policy_version")} '
            f'({"shadow" if self.shadow else "live"}, limit {self.max_speed:.1f} m/s)')

    def on_safety(self, msg):
        self.safety_stop = bool(msg.data)

    def on_scan(self, msg):
        self.scan, self.scan_time = msg, self.now()
        self.plan()

    def on_odom(self, msg):
        self.odom, self.odom_time = msg, self.now()

    def now(self):
        return self.get_clock().now().nanoseconds*1e-9

    def stale(self):
        if self.scan is None or self.odom is None:
            return 'missing input'
        now = self.now()
        if now-self.scan_time > self.scan_timeout:
            return 'scan timeout'
        if now-self.odom_time > self.odom_timeout:
            return 'odometry timeout'
        return None

    def plan(self):
        reason = self.stale()
        if reason or self.safety_stop:
            return self.publish(0., 0., reason or 'safety_stop')
        age = self.now()-self.scan_time
        observation = build_observation(
            self.scan, self.odom, self.last_published[1:2]*2, tick=self.tick,
            wheel_radius=self.wheel_radius, age=age)
        self.tick += 1
        wheels, steering = self.actor.action(observation)
        speed, angle = action_to_ackermann(
            wheels, steering, wheel_radius=self.wheel_radius,
            max_speed=self.max_speed, max_steering=self.max_steering)
        self.publish(speed, angle)

    def publish(self, speed, angle, reason='tracking'):
        self.last_published = (speed, angle)
        if self.cmd_pub is not None:
            msg = AckermannDrive()
            msg.speed = float(speed)
            msg.steering_angle = float(angle)
            self.cmd_pub.publish(msg)
        now = self.now()
        if self.shadow and now-self.last_log >= self.log_period:
            self.last_log = now
            self.get_logger().info(
                f'shadow {reason}: speed {speed:+.3f} m/s steering {math.degrees(angle):+.1f} deg')


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
