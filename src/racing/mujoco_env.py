"""Compatibility imports for the former MuJoCo backend module."""

from .simulators.mujoco.environment import (
    CONTROL_DT,
    PHYSICS_DT,
    ROOT,
    STEERING_LIMIT,
    STEER_NAMES,
    WHEEL_NAMES,
    WHEEL_RADIUS,
    RaceMujocoEnv,
    build_model,
    mujoco,
)

__all__ = [
    "CONTROL_DT", "PHYSICS_DT", "ROOT", "STEERING_LIMIT", "STEER_NAMES",
    "WHEEL_NAMES", "WHEEL_RADIUS", "RaceMujocoEnv", "build_model", "mujoco",
]
