#!/usr/bin/env python3
"""Native MuJoCo free-body OSRACER for the Isaac-to-MuJoCo hairpin task.

All motion after reset comes from actuator forces and contact integration.
Targets are wheel joint velocities (LF, RF, LR, RR; RR negative forward),
and steering joint positions (left, right), in radians and seconds.
"""
from __future__ import annotations

import glob
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
for package in ("mujoco", "OpenGL"):
    try:
        __import__(package)
    except ImportError:
        candidates = glob.glob(str(Path.home() / ".cache/uv/archive-v0" / "*" / package))
        if candidates:
            sys.path.insert(0, str(Path(candidates[0]).parent))

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "OSRACER/MuJoCo/osracer_description"
WHEEL_NAMES = tuple(f"{side}_{axle}_wheel_joint" for axle in ("front", "rear") for side in ("left", "right"))
STEER_NAMES = ("left_steering_hinge_joint", "right_steering_hinge_joint")
WHEEL_RADIUS = 0.045
PHYSICS_DT = 1.0 / 480
CONTROL_DT = 1.0 / 30


def _road(world, asset):
    """Noncontact road markings: the physical surface is a single level plane."""
    points = [(float(x), 0.0) for x in np.linspace(-0.55, 1.2, 24)]
    points += [(1.2 + .8 * math.cos(t), .8 + .8 * math.sin(t))
               for t in np.linspace(-math.pi / 2, math.pi / 2, 81)[1:]]
    points += [(float(x), 1.6) for x in np.linspace(1.2, -.55, 24)[1:]]
    # One closed ribbon avoids coplanar overlaps and z fighting in the road.
    vertices, faces = [], []
    for i, p in enumerate(points):
        previous, following = points[max(0, i - 1)], points[min(len(points) - 1, i + 1)]
        tangent = np.array(following) - previous
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]]) * .29
        for z in (.0006, .0002):
            for side in (1, -1):
                vertices.append([p[0] + side * normal[0], p[1] + side * normal[1], z])
    for i in range(len(points) - 1):
        a, b = 4 * i, 4 * (i + 1)
        faces.extend([[a, a + 1, b + 1], [a, b + 1, b],
                      [a + 2, b + 3, a + 3], [a + 2, b + 2, b + 3],
                      [a, b, b + 2], [a, b + 2, a + 2],
                      [a + 1, a + 3, b + 3], [a + 1, b + 3, b + 1]])
    last = 4 * (len(points) - 1)
    faces.extend([[0, 2, 3], [0, 3, 1], [last, last + 1, last + 3], [last, last + 3, last + 2]])
    ET.SubElement(asset, "mesh", name="road_ribbon", vertex=" ".join(str(v) for p in vertices for v in p),
                  face=" ".join(str(v) for p in faces for v in p))
    ET.SubElement(world, "geom", name="road_surface", type="mesh", mesh="road_ribbon",
                  rgba=".13 .16 .20 1", contype="0", conaffinity="0")
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length, angle = math.hypot(dx, dy), math.atan2(dy, dx)
        x, y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        for side in (-1, 1):
            px, py = x - side * .285 * math.sin(angle), y + side * .285 * math.cos(angle)
            ET.SubElement(world, "geom", name=f"edge_{i}_{side}", type="box",
                          pos=f"{px} {py} .001", size=f"{length / 2 + .005} .009 .0003",
                          euler=f"0 0 {angle}", rgba=".90 .93 .94 1", contype="0", conaffinity="0")
        if i % 4 < 2:
            ET.SubElement(world, "geom", name=f"center_{i}", type="box",
                          pos=f"{x} {y} .0012", size=f"{length / 2} .006 .0003",
                          euler=f"0 0 {angle}", rgba=".88 .71 .29 1", contype="0", conaffinity="0")
    for name, x, y in (("start", 0., 0.), ("finish", 0., 1.6)):
        for j in range(10):
            ET.SubElement(world, "geom", name=f"{name}_{j}", type="box",
                          pos=f"{x} {y - .26 + j * .057} .0016", size=".025 .0285 .0004",
                          rgba=".95 .95 .95 1" if j % 2 else ".12 .15 .18 1", contype="0", conaffinity="0")


def build_model(path: Path | None = None) -> Path:
    """Derive a drivable model; never edit the source XML or CAD meshes."""
    path = Path(path or ROOT / "output/hairpin/mujoco_model.xml")
    tree = ET.parse(ASSET / "robot.xml")
    root = tree.getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str((ASSET / mesh.attrib["file"]).resolve()))
    ET.SubElement(root, "option", timestep=str(PHYSICS_DT), integrator="implicitfast",
                  gravity="0 0 -9.81", iterations="80", cone="elliptic")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient=".45 .45 .45", diffuse=".8 .8 .8", specular=".1 .1 .1")
    ET.SubElement(visual, "rgba", haze=".12 .17 .22 1")
    world = root.find("worldbody")
    base = world.find("body")
    base.set("pos", "0 0 .05")
    ET.SubElement(base, "freejoint", name="floating_base")
    # Distinct masks disable internal CAD mesh collisions while retaining all
    # original collision shapes against the plane, except the wheel proxies.
    for geom in base.iter("geom"):
        if geom.get("group") == "3":
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
            geom.set("friction", "1 .005 .0001")
    for joint in base.iter("joint"):
        joint.set("damping", ".0001")
        joint.set("frictionloss", ".001")
    for name in WHEEL_NAMES:
        body = base.find(f".//body[@name='{name.replace('_joint', '_link')}']")
        collision = body.find("geom[@group='3']")
        collision.attrib.pop("mesh", None)
        collision.set("type", "cylinder")
        collision.set("size", f"{WHEEL_RADIUS} .02")
        collision.set("pos", body.find("inertial").get("pos"))
        collision.set("quat", ".7071067811865476 .7071067811865475 0 0")
        collision.set("condim", "3")
        collision.set("solref", ".008 1")
        collision.set("solimp", ".95 .99 .001")
    for actuator in root.find("actuator"):
        actuator.set("forcerange", "-.3 .3" if actuator.tag == "velocity" else "-1 1")
        if actuator.tag == "velocity":
            actuator.set("kv", ".03")
        else:
            actuator.set("kp", "3")
            actuator.set("kv", ".08")
    ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 .1",
                  contype="1", conaffinity="2", friction="1 .005 .0001", rgba=".29 .37 .36 1")
    ET.SubElement(world, "light", name="key", pos="1 -1 4", dir="0 0 -1", diffuse=".8 .8 .8")
    _road(world, root.find("asset"))
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


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
