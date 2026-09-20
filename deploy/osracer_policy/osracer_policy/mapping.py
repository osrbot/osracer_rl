"""Pure conversions between OSRacer ROS 2 messages and the racing policy.

No ROS imports live here so the mapping can be replayed and unit-tested on a
workstation with recorded data. Every function takes plain dicts or duck-typed
messages, which keeps the vehicle node thin.
"""
from __future__ import annotations

import math

import numpy as np

WHEEL_RADIUS_DEFAULT = .045
RAY_COUNT_DEFAULT = 361
FOV_DEG_DEFAULT = 270.
MAX_RANGE_DEFAULT = 15.


def _scan_field(scan, name, default=None):
    value = getattr(scan, name, None)
    if value is None:
        if default is None:
            raise ValueError(f'LaserScan is missing {name}')
        return default
    return value


def scan_to_packet(scan, ray_count=RAY_COUNT_DEFAULT, fov_deg=FOV_DEG_DEFAULT,
                   max_range=MAX_RANGE_DEFAULT, timestamp=None, age=0.):
    """Resample a LaserScan onto the policy's uniform 270 degree grid.

    Missing, infinite and out-of-range beams become invalid at maximum range,
    which is exactly what the trained observation contract expects.
    """
    angle_min = float(_scan_field(scan, 'angle_min'))
    angle_increment = float(_scan_field(scan, 'angle_increment'))
    ranges = np.asarray(_scan_field(scan, 'ranges', []), float)
    if ranges.size == 0 or angle_increment == 0.:
        raise ValueError('LaserScan has no beams')
    limit = float(_scan_field(scan, 'range_max', max_range)) or max_range
    limit = min(limit, max_range)
    usable = np.isfinite(ranges) & (ranges > .04) & (ranges < limit)
    source_angles = angle_min+angle_increment*np.arange(ranges.size)
    targets = np.linspace(-math.radians(fov_deg)/2, math.radians(fov_deg)/2, ray_count)
    half_bin = math.radians(fov_deg)/max(ray_count-1, 1)/2
    out_ranges = np.full(ray_count, max_range)
    out_valid = np.zeros(ray_count, bool)
    for index, target in enumerate(targets):
        near = np.flatnonzero(usable & (np.abs(_wrap(source_angles-target)) <= half_bin))
        if near.size:
            out_ranges[index] = float(np.min(ranges[near]))
            out_valid[index] = True
    return {'ranges': out_ranges, 'valid': out_valid, 'validmask': out_valid.copy(),
            'angles': targets, 'timestamp': float(timestamp if timestamp is not None else 0.),
            'age': float(age), 'frame': 'laser', 'frame_id': 'laser'}


def _wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


def odom_to_wheel_velocity(odom, wheel_radius=WHEEL_RADIUS_DEFAULT):
    """Four identical wheel rates from the single body speed the chassis reports.

    The vehicle has no per-wheel encoders, so replicating the EKF forward speed
    is the only sensor-only option; it is documented as an approximation.
    """
    twist = getattr(odom, 'twist', None)
    speed = float(getattr(twist, 'twist', twist).linear.x) if twist is not None else 0.
    rate = speed/wheel_radius if wheel_radius else 0.
    return np.full(4, rate)


def build_observation(scan, odom, steer_pos, tick=0, ray_count=RAY_COUNT_DEFAULT,
                      fov_deg=FOV_DEG_DEFAULT, max_range=MAX_RANGE_DEFAULT,
                      wheel_radius=WHEEL_RADIUS_DEFAULT, age=0.):
    return {'wheel_vel': odom_to_wheel_velocity(odom, wheel_radius),
            'steer_pos': np.asarray(steer_pos, float).reshape(2),
            'lidar': scan_to_packet(scan, ray_count=ray_count, fov_deg=fov_deg,
                                    max_range=max_range, timestamp=tick, age=age)}


def action_to_ackermann(wheels, steering, wheel_radius=WHEEL_RADIUS_DEFAULT,
                        max_speed=None, max_steering=None):
    """Collapse the policy's four wheel rates and two steering angles.

    The chassis takes one speed and one Ackermann angle. The rear overdrive the
    simulation uses for drifting has no actuator here, so the front pair defines
    the commanded body speed.
    """
    wheels = np.asarray(wheels, float).reshape(4)
    steering = np.asarray(steering, float).reshape(2)
    speed = float(np.mean(wheels[:2]))*wheel_radius
    angle = float(np.mean(steering))
    if max_speed is not None:
        speed = max(-max_speed, min(max_speed, speed))
    if max_steering is not None:
        angle = max(-max_steering, min(max_steering, angle))
    return speed, angle
