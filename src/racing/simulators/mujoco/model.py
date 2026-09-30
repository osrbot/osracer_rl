"""Shared MuJoCo vehicle constants and derived-model builder."""
from __future__ import annotations

import math
import os
from pathlib import Path
from ...paths import REPO_ROOT
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

ROOT = REPO_ROOT
ASSET = ROOT / "assets/vehicles/osracer/MuJoCo/osracer_description"
WHEEL_NAMES = tuple(
    f"{side}_{axle}_wheel_joint"
    for axle in ("front", "rear")
    for side in ("left", "right")
)
STEER_NAMES = ("left_steering_hinge_joint", "right_steering_hinge_joint")
WHEEL_RADIUS = 0.045
PHYSICS_DT = 1.0 / 480
HAIRPIN_CONTROL_DT = 1.0 / 30


def _hairpin_road(world, asset):
    """Add the noncontact road markings used by the bounded hairpin fixture."""
    points = [(float(x), 0.0) for x in np.linspace(-0.55, 1.2, 24)]
    points += [
        (1.2 + .8 * math.cos(t), .8 + .8 * math.sin(t))
        for t in np.linspace(-math.pi / 2, math.pi / 2, 81)[1:]
    ]
    points += [(float(x), 1.6) for x in np.linspace(1.2, -.55, 24)[1:]]
    vertices, faces = [], []
    for i, point in enumerate(points):
        previous, following = points[max(0, i - 1)], points[min(len(points) - 1, i + 1)]
        tangent = np.array(following) - previous
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]]) * .29
        for z in (.0006, .0002):
            for side in (1, -1):
                vertices.append([point[0] + side * normal[0], point[1] + side * normal[1], z])
    for i in range(len(points) - 1):
        a, b = 4 * i, 4 * (i + 1)
        faces.extend([[a, a + 1, b + 1], [a, b + 1, b],
                      [a + 2, b + 3, a + 3], [a + 2, b + 2, b + 3],
                      [a, b, b + 2], [a, b + 2, a + 2],
                      [a + 1, a + 3, b + 3], [a + 1, b + 3, b + 1]])
    last = 4 * (len(points) - 1)
    faces.extend([[0, 2, 3], [0, 3, 1], [last, last + 1, last + 3],
                  [last, last + 3, last + 2]])
    ET.SubElement(asset, "mesh", name="road_ribbon",
                  vertex=" ".join(str(v) for point in vertices for v in point),
                  face=" ".join(str(v) for face in faces for v in face))
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
                          euler=f"0 0 {angle}", rgba=".90 .93 .94 1",
                          contype="0", conaffinity="0")
        if i % 4 < 2:
            ET.SubElement(world, "geom", name=f"center_{i}", type="box",
                          pos=f"{x} {y} .0012", size=f"{length / 2} .006 .0003",
                          euler=f"0 0 {angle}", rgba=".88 .71 .29 1",
                          contype="0", conaffinity="0")
    for name, x, y in (("start", 0., 0.), ("finish", 0., 1.6)):
        for j in range(10):
            ET.SubElement(world, "geom", name=f"{name}_{j}", type="box",
                          pos=f"{x} {y - .26 + j * .057} .0016",
                          size=".025 .0285 .0004",
                          rgba=".95 .95 .95 1" if j % 2 else ".12 .15 .18 1",
                          contype="0", conaffinity="0")


def build_base_model(path: Path | None = None) -> Path:
    """Derive a free-body drivable model without editing source assets."""
    path = Path(path or ROOT / "output/hairpin/mujoco_model.xml")
    tree = ET.parse(ASSET / "robot.xml")
    root = tree.getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str((ASSET / mesh.attrib["file"]).resolve()))
    ET.SubElement(root, "option", timestep=str(PHYSICS_DT), integrator="implicitfast",
                  gravity="0 0 -9.81", iterations="80", cone="elliptic")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient=".45 .45 .45", diffuse=".8 .8 .8",
                  specular=".1 .1 .1")
    ET.SubElement(visual, "rgba", haze=".12 .17 .22 1")
    world = root.find("worldbody")
    base = world.find("body")
    base.set("pos", "0 0 .05")
    ET.SubElement(base, "freejoint", name="floating_base")
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
                  contype="1", conaffinity="2", friction="1 .005 .0001",
                  rgba=".29 .37 .36 1")
    ET.SubElement(world, "light", name="key", pos="1 -1 4", dir="0 0 -1",
                  diffuse=".8 .8 .8")
    _hairpin_road(world, root.find("asset"))
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


__all__ = [
    "ASSET", "HAIRPIN_CONTROL_DT", "PHYSICS_DT", "ROOT", "STEER_NAMES",
    "WHEEL_NAMES", "WHEEL_RADIUS", "build_base_model", "mujoco",
]
