"""Observation adapter for the real OSRacer vehicle.

The car has one motor driving all four wheels, one encoder that produces the
odometry speed, and a bus servo that steers through a single tie rod and an
inverted trapezoid with no angle feedback. Two things the training contract
assumes therefore do not exist on the vehicle:

* per-wheel encoders -- only one drivetrain speed is measurable;
* a measured steering joint angle -- only the command is known.

This module degrades a simulated observation to exactly that information so the
loss can be measured in the simulator before anyone drives the car. It never
looks at world pose, track state or opponent truth.
"""
from __future__ import annotations

import numpy as np

from .policy import DT


class SteeringLag:
    """First-order servo model with a slew limit: command -> expected position.

    Identified on the bench by commanding steps and recording the road-wheel
    angle. Until a real sensor is fitted this is the best available estimate of
    where the wheels actually are.
    """

    def __init__(self, time_constant=.08, rate_limit=3.5):
        if time_constant <= 0 or rate_limit <= 0:
            raise ValueError('Expected positive servo time constant and rate limit')
        self.time_constant, self.rate_limit = float(time_constant), float(rate_limit)
        self.reset()

    def reset(self):
        self.position = 0.

    def step(self, command, dt=DT):
        command = float(command)
        target = self.position+(command-self.position)*min(1., dt/self.time_constant)
        travel = float(np.clip(target-self.position, -self.rate_limit*dt, self.rate_limit*dt))
        self.position += travel
        return self.position


class RealVehicleSensor:
    """Wrap a laser sensor and expose only what the car can measure."""

    def __init__(self, sensor, steering_lag=None, wheel_radius=.045):
        self.sensor = sensor
        self.steering_lag = steering_lag or SteeringLag()
        self.wheel_radius = float(wheel_radius)
        self.last_command = 0.

    def reset(self):
        self.sensor.reset()
        self.steering_lag.reset()
        self.last_command = 0.

    @property
    def specs(self):
        specs = dict(self.sensor.specs)
        specs['observation_profile'] = 'real_vehicle'
        specs['wheel_speed_source'] = 'single drivetrain encoder replicated to four wheels'
        specs['steering_source'] = (f'open-loop estimate, first-order lag '
                                    f'tau={self.steering_lag.time_constant}s '
                                    f'rate={self.steering_lag.rate_limit}rad/s')
        return specs

    def record_action(self, action):
        """Remember the commanded steering; the servo model advances on it."""
        steering = np.asarray(action[1], float).reshape(-1)
        self.last_command = float(np.mean(steering))
        self.steering_lag.step(self.last_command)

    def observe(self, state, lidar_pose, opponents=(), tick=None):
        observation = dict(self.sensor.observe(state, lidar_pose, opponents=opponents, tick=tick))
        # One encoder on the drivetrain: all four wheels report the same rate.
        wheels = np.asarray(state['wheel_vel'], float).reshape(4)
        rate = float(np.mean(wheels[:2]))
        observation['wheel_vel'] = np.full(4, rate)
        # No joint feedback: report where the servo is expected to be.
        position = float(self.steering_lag.position)
        observation['steer_pos'] = np.full(2, position)
        return observation


def make_sensor(sensor, profile):
    if profile in (None, 'sim', 'simulation'):
        return sensor
    if profile == 'real':
        return RealVehicleSensor(sensor)
    raise ValueError(f'Unknown observation profile: {profile}')
