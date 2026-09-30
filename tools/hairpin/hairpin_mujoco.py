#!/usr/bin/env python3
"""Native MuJoCo free-body OSRACER for the Isaac-to-MuJoCo hairpin task.

All motion after reset comes from actuator forces and contact integration.
Targets are wheel joint velocities (LF, RF, LR, RR; RR negative forward),
and steering joint positions (left, right), in radians and seconds.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'src'), str(Path(__file__).resolve().parents[2])]
from racing.simulators.mujoco.model import (
    HAIRPIN_CONTROL_DT as CONTROL_DT,
    PHYSICS_DT,
    ROOT,
    STEER_NAMES,
    WHEEL_NAMES,
    WHEEL_RADIUS,
    build_base_model as build_model,
    mujoco,
)


class MujocoEnv:
    def __init__(self, model_path=None):
        self.version = mujoco.__version__
        self.model_path = build_model(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.m, self.d = self.model, self.data
        joint_ids = [self.model.joint(n).id for n in WHEEL_NAMES + STEER_NAMES]
        self._qpos = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        self._qvel = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        self._actuators = [self.model.actuator(n + "_drive").id for n in WHEEL_NAMES + STEER_NAMES]
        self._body_id = self.model.body("base_link").id
        self._renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [1.0, .8, 0]
        self.camera.distance = 3.5
        self.camera.azimuth = 90
        self.camera.elevation = -67
        self._option = mujoco.MjvOption()
        self._option.geomgroup[3] = 0
        self.reset()

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        y = 0. if seed == 0 else rng.uniform(-.025, .025)
        yaw = 0. if seed == 0 else rng.uniform(-.03, .03)
        self.data.qpos[:3] = [0, y, .055]
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        # Physical settling is part of reset; no kinematic placement afterward.
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0
        return self.state()

    def step(self, wheel_targets, steering_targets):
        targets = np.concatenate((np.asarray(wheel_targets, dtype=float), np.asarray(steering_targets, dtype=float)))
        if targets.shape != (6,) or not np.isfinite(targets).all():
            raise ValueError("Expected four finite wheel velocities and two finite steering angles")
        self.data.ctrl[self._actuators] = targets
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def state(self):
        # mj_step integrates qpos after its forward pass; refresh derived body
        # transforms/velocities so all observed fields refer to the same time.
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.xpos[self._body_id]
        mat = self.data.xmat[self._body_id].reshape(3, 3)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self._body_id, velocity, 0)
        return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                "yaw": math.atan2(mat[1, 0], mat[0, 0]),
                "pitch": math.asin(float(np.clip(-mat[2, 0], -1, 1))),
                "roll": math.atan2(mat[2, 1], mat[2, 2]),
                "vx": float(velocity[3]), "vy": float(velocity[4]),
                "yaw_rate": float(velocity[2]),
                "wheel_vel": self.data.qvel[self._qvel[:4]].copy() * [1, 1, 1, -1],
                "steer_pos": self.data.qpos[self._qpos[4:]].copy()}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self._renderer.update_scene(self.data, camera=self.camera, scene_option=self._option)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    import json
    env = MujocoEnv()
    results = {}
    for label, angle in (("straight", 0.), ("left_turn", .35)):
        initial = env.reset(seed=0)
        for _ in range(90):
            final = env.step([10., 10., 10., -10.], [angle, angle])
        results[label] = {"initial": initial, "final": final,
                          "warnings": env.data.warning.number.tolist(),
                          "contacts": env.data.ncon}
    from PIL import Image
    Image.fromarray(env.render()).save(ROOT / "output/hairpin/mujoco_probe.png")
    env.close()
    print(json.dumps(results, indent=2, default=lambda x: x.tolist()))
