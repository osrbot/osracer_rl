"""Pure, local-frame motion geometry for the laser safety supervisor.

These helpers never keep a world pose. A twist is (forward m/s, left m/s,
counter-clockwise rad/s) in the current vehicle frame.
"""
from __future__ import annotations

import math
import numpy as np

WHEELBASE = .28764


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def twist_displacement(twist, duration):
    """Exact SE(2) displacement for a constant body-frame twist."""
    vx, vy, yaw_rate = np.asarray(twist, float)
    if duration < 0 or not np.isfinite([vx, vy, yaw_rate, duration]).all():
        raise ValueError('Finite twist and nonnegative finite duration required')
    angle = float(yaw_rate*duration)
    if abs(angle) < 1e-7:
        a = duration*(1-angle*angle/6)
        b = duration*(angle/2-angle**3/24)
    else:
        a, b = math.sin(angle)/yaw_rate, (1-math.cos(angle))/yaw_rate
    return np.array([a*vx-b*vy, b*vx+a*vy]), angle


def compensate_points(points, twist, age):
    """Map points from their scan-time body frame into the current body frame."""
    translation, yaw = twist_displacement(twist, age)
    return (np.asarray(points, float)-translation)@rotation(yaw)


def predict_poses(twist, actual_steer, target_steer, *, horizon=.4,
                  response_rate=3., braking=False, braking_deceleration=4.,
                  mode='slip', yaw_bias_decay=None, command_speed=None,
                  max_acceleration=12.):
    """Predict local poses while retaining observed lateral motion and yaw.

    ``slip`` preserves the observed body velocity direction. ``coast`` keeps
    its inertial direction while the body rotates, representing lost lateral
    traction during wheel lock. Neither mode resets the observed yaw rate to
    the steering-angle bicycle prediction. The yaw discrepancy may persist or
    decay gradually; these are explicit model hypotheses, not certified bounds.

    A positive command_speed is a target with finite acceleration, not a claim
    about initial speed. Existing faster motion is retained for that hypothesis.
    """
    vx, vy, observed_yaw_rate = map(float, twist)
    values = [vx, vy, observed_yaw_rate, actual_steer, target_steer, horizon,
              response_rate, braking_deceleration, max_acceleration]
    if (not np.isfinite(values).all() or horizon < 0 or response_rate <= 0
            or braking_deceleration < 0 or max_acceleration <= 0):
        raise ValueError('Invalid local motion prediction inputs')
    if mode not in {'slip', 'coast'}:
        raise ValueError('Unknown local motion hypothesis')
    if yaw_bias_decay is not None and (not math.isfinite(yaw_bias_decay) or yaw_bias_decay <= 0):
        raise ValueError('Yaw decay time must be positive')
    if command_speed is not None and not math.isfinite(command_speed):
        raise ValueError('Command speed must be finite')
    reversing = command_speed is not None and command_speed < 0.
    initial_speed = math.hypot(vx, vy)
    speed, steer, yaw = initial_speed, float(actual_steer), 0.
    position = np.zeros(2)
    direction = np.array([vx, vy])/initial_speed if initial_speed > 1e-8 else np.array([1., 0.])

    def bicycle_yaw(forward, steering, magnitude):
        limit = 5.5/max(magnitude, .1)
        return float(np.clip(forward*math.tan(steering)/WHEELBASE, -limit, limit))

    yaw_bias = observed_yaw_rate-bicycle_yaw(vx, steer, speed)
    previous_yaw_rate = observed_yaw_rate
    poses = [[0., 0., 0.]]
    count = max(1, int(math.ceil(horizon/.02)))
    dt = horizon/count
    for step in range(count):
        steer += float(np.clip(target_steer-steer, -response_rate*dt, response_rate*dt))
        next_speed = speed
        next_direction = direction
        accelerating = command_speed is not None and speed < command_speed
        commanding = accelerating or reversing
        if reversing:
            # A bounded reverse command accelerates backwards along the body
            # axis. The supervisor uses it to verify a back-off manoeuvre.
            initial_body = direction*speed
            if mode == 'coast':
                initial_body = initial_body@rotation(yaw)
            next_body = initial_body.copy()
            next_body[0] = max(float(command_speed), initial_body[0]-max_acceleration*dt)
            velocity = .5*(initial_body+next_body)
            next_speed = float(np.linalg.norm(next_body))
            next_direction = next_body/max(next_speed, 1e-12)
            if mode == 'coast':
                next_direction = next_direction@rotation(yaw).T
            middle_speed = float(np.linalg.norm(velocity))
        elif accelerating:
            # Positive wheel commands accelerate along the vehicle forward axis,
            # including when a nearly stopped estimate is slightly negative or
            # sideways. Preserve the initial velocity; never normalize its noise
            # into a commanded reverse/sideways acceleration.
            initial_body = direction*speed
            if mode == 'coast':
                initial_body = initial_body@rotation(yaw)
            next_body = initial_body.copy()
            next_body[0] = min(float(command_speed), initial_body[0]+max_acceleration*dt)
            velocity = .5*(initial_body+next_body)
            next_speed = float(np.linalg.norm(next_body))
            next_direction = next_body/max(next_speed, 1e-12)
            if mode == 'coast':
                next_direction = next_direction@rotation(yaw).T
            middle_speed = float(np.linalg.norm(velocity))
        else:
            if braking and command_speed is None:
                next_speed = max(0., speed-braking_deceleration*dt)
            middle_speed = .5*(speed+next_speed)
            velocity = direction*middle_speed
            if mode == 'coast':
                velocity = velocity@rotation(yaw)
        bias = yaw_bias if yaw_bias_decay is None else yaw_bias*math.exp(-(step+1)*dt/yaw_bias_decay)
        next_yaw_rate = bicycle_yaw(float(velocity[0]), steer, middle_speed)+bias
        average_yaw_rate = .5*(previous_yaw_rate+next_yaw_rate)
        if mode == 'coast':
            position += (velocity@rotation(yaw).T if commanding else direction*middle_speed)*dt
            yaw += average_yaw_rate*dt
        else:
            displacement, angle = twist_displacement([*velocity, average_yaw_rate], dt)
            position += displacement@rotation(yaw).T
            yaw += angle
        poses.append([float(position[0]), float(position[1]), yaw])
        speed, previous_yaw_rate, direction = next_speed, next_yaw_rate, next_direction
    return np.asarray(poses)


def swept_footprints(footprint, poses):
    return np.asarray([np.asarray(footprint)@rotation(float(pose[2])).T+pose[:2]
                       for pose in np.asarray(poses)])
