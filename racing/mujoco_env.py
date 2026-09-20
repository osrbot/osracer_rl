"""Native multi-car NEORACER dynamics with independent six-actuator controls.

Wheel commands are LF, RF, LR, RR radians/second (RR negative forward).
Steering commands are left/right radians. The validated hairpin adapter's
contact model, actuator settings and source CAD inertias are retained.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from scripts.hairpin_mujoco import (
    PHYSICS_DT, STEER_NAMES, WHEEL_NAMES, build_model as build_base_model, mujoco,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DT = 1 / 60
STEERING_LIMIT = .45
WHEEL_RADIUS = .045


def _triangle_deck(world, asset, triangle, name, thickness):
    """One convex triangular slab; a whole bridge mesh would fill its underpass."""
    top = np.asarray(triangle, dtype=float)
    if np.cross(top[1]-top[0], top[2]-top[0])[2] < 0:
        top = top[[0, 2, 1]]
    bottom = top - [0., 0., thickness]
    origin = top.mean(axis=0)
    vertices = np.vstack((top, bottom)) - origin
    faces = [[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1],
             [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]]
    ET.SubElement(asset, 'mesh', name=name+'_mesh',
                  vertex=' '.join(str(v) for p in vertices for v in p),
                  face=' '.join(str(v) for f in faces for v in f))
    ET.SubElement(world, 'geom', name=name, type='mesh', mesh=name+'_mesh',
                  pos=' '.join(str(v) for v in origin),
                  rgba='.11 .14 .18 1', contype='1', conaffinity='2147483646',
                  friction='1 .005 .0001', group='0')


def _ribbon_mesh(world, asset, track):
    points = np.asarray(track.points, dtype=float)
    elevations = np.asarray(getattr(track, 'elevations', np.zeros(len(points))), dtype=float)
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    vertices, faces = [], []
    for i, point in enumerate(points):
        tangent = points[(i + 1) % len(points)] - points[i - 1]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]]) * track.width / 2
        for z in (elevations[i]+.001, elevations[i]+.0003):
            for side in (1, -1):
                vertices.append([*(point + side * normal), z])
    for i in range(len(points)):
        a, b = 4 * i, 4 * ((i + 1) % len(points))
        faces.extend([[a, a+1, b+1], [a, b+1, b],
                      [a+2, b+3, a+3], [a+2, b+2, b+3],
                      [a, b, b+2], [a, b+2, a+2],
                      [a+1, a+3, b+3], [a+1, b+3, b+1]])
    ET.SubElement(asset, 'mesh', name='race_road_mesh',
                  vertex=' '.join(str(v) for p in vertices for v in p),
                  face=' '.join(str(v) for f in faces for v in f))
    ET.SubElement(world, 'geom', name='race_road', type='mesh', mesh='race_road_mesh',
                  rgba='.11 .14 .18 1', contype='0', conaffinity='0', group='1')


def _road(world, asset, track):
    source_path = getattr(track, 'path', None)
    road_obj = Path(source_path).parent / 'track.obj' if source_path is not None else None
    if road_obj is not None and road_obj.is_file():
        # The triangulated union follows the generated boundaries even where
        # nearby road sections merge. Shell inertia permits the flat OBJ mesh;
        # this static visual has no contact or vehicle mass contribution.
        ET.SubElement(asset, 'mesh', name='race_road_mesh',
                      file=str(road_obj.resolve()), inertia='shell')
        ET.SubElement(world, 'geom', name='race_road', type='mesh', mesh='race_road_mesh',
                      pos='0 0 .001', rgba='.11 .14 .18 1',
                      contype='0', conaffinity='0', group='1')
    else:
        _ribbon_mesh(world, asset, track)
    if hasattr(track, 'road_triangles_3d') and np.max(getattr(track, 'elevations', [0.])) > 0:
        thickness = getattr(track, 'metadata', {}).get('bridge', {}).get('deck_thickness', .08)
        for i, triangle in enumerate(track.road_triangles_3d()):
            if np.max(np.asarray(triangle)[:, 2]) > 1e-8:
                _triangle_deck(world, asset, triangle, f'bridge_deck_{i}', thickness)
    boundaries = getattr(track, 'boundary_segments_3d', track.boundary_segments)
    boundaries = boundaries() if callable(boundaries) else boundaries
    boundaries = np.asarray(boundaries)
    if boundaries.shape[-1] == 2:
        boundaries = np.pad(boundaries, ((0, 0), (0, 0), (0, 1)))
    for i, (a, b) in enumerate(boundaries):
        delta = b-a
        length = float(np.linalg.norm(delta))
        if length < 1e-8:
            continue
        midpoint = (a+b)/2
        angle = math.atan2(delta[1], delta[0])
        pitch = -math.atan2(delta[2], np.linalg.norm(delta[:2]))
        cy, sy = math.cos(angle/2), math.sin(angle/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        wall_quat = f'{cy*cp} {-sy*sp} {cy*sp} {sy*cp}'
        # A narrow, tall barrier is both physical and visible to the pitched
        # source laser frame. Bridge barriers follow their own layer's slope.
        # Analytic scans intersect the boundary centre plane, 2 cm from this
        # box's near face in the normal direction (more along oblique rays).
        ET.SubElement(world, 'geom', name=f'barrier_{i}', type='box',
                      pos=f'{midpoint[0]} {midpoint[1]} {midpoint[2]+.3}',
                      size=f'{length/2+.002} .02 .3', quat=wall_quat,
                      rgba='.82 .87 .92 1' if (i//5) % 2 else '.8 .17 .13 1',
                      contype='1', conaffinity='2147483646',
                      friction='1 .005 .0001', group='0')
    xy, yaw = track.at(0.)
    start_z = track.elevation_at(0.) if hasattr(track, 'elevation_at') else 0.
    for i in range(20):
        offset = (i + .5) * track.width/20 - track.width/2
        p = np.asarray(xy) + offset*np.array([-math.sin(yaw), math.cos(yaw)])
        ET.SubElement(world, 'geom', name=f'race_start_{i}', type='box',
                      pos=f'{p[0]} {p[1]} {start_z+.0017}', euler=f'0 0 {yaw}',
                      size=f'.04 {track.width/40} .0003',
                      rgba='.95 .95 .95 1' if i % 2 else '.02 .02 .02 1',
                      contype='0', conaffinity='0', group='1')


def build_model(track, num_cars=2, path=None):
    """Build an independent scene without changing original CAD/model assets."""
    if not 2 <= num_cars <= 16:
        raise ValueError('num_cars must be between 2 and 16')
    track_id = str(track.id)
    if Path(track_id).name != track_id or track_id in ('', '.', '..'):
        raise ValueError('track.id must be a filename-safe identifier')
    path = Path(path or ROOT / 'output/racing/mujoco' / f'{track_id}.xml')
    with tempfile.TemporaryDirectory(prefix='neoracer-mj-') as temporary:
        tree = ET.parse(build_base_model(Path(temporary)/'base.xml'))
    root = tree.getroot()
    root.set('model', f'neoracer_race_{track_id}')
    world, asset, actuators = root.find('worldbody'), root.find('asset'), root.find('actuator')
    template = copy.deepcopy(world.find('body'))
    world.remove(world.find('body'))
    # Explicit simulation reflector, shared by every car. It makes opponents
    # observable above their low chassis in the source laser's pitched plane.
    # inertiafromgeom=false retains the original CAD inertias and total mass.
    for name, group, rgba in (('lidar_target', '3', '.95 .95 .95 1'),
                              ('lidar_target_visual', '2', '.95 .95 .95 1')):
        ET.SubElement(template, 'geom', name=name, type='box',
                      pos='.10 0 .325', size='.05 .08 .275', mass='0', group=group,
                      rgba=rgba, contype='0', conaffinity='0')
    drives = list(actuators)
    actuators.clear()
    for geom in list(world.findall('geom')):
        if geom.get('name') != 'ground':
            world.remove(geom)
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name') == 'road_ribbon':
            asset.remove(mesh)
    all_car_bits = sum(1 << (i+1) for i in range(num_cars))
    ground = world.find("geom[@name='ground']")
    ground.set('conaffinity', str(all_car_bits))
    extent = np.max(np.abs(np.asarray(track.points)), axis=0) + 20.
    ground.set('size', f'{extent[0]} {extent[1]} .1')
    for i in range(num_cars):
        prefix = f'car{i}_'
        body = copy.deepcopy(template)
        car_bit = 1 << (i+1)
        for element in body.iter():
            if 'name' in element.attrib:
                element.set('name', prefix + element.get('name'))
            if element.tag == 'geom' and element.get('group') == '3':
                element.set('contype', str(car_bit))
                element.set('conaffinity', str(1 | (all_car_bits ^ car_bit)))
        # Tint only the body visual; all CAD meshes and masses are shared intact.
        body.find(f"geom[@name='{prefix}base_link_visual_0']").set(
            'rgba', ('1 .75 .05 1', '.05 .6 1 1', '.8 .15 .75 1', '.15 .85 .45 1')[i % 4])
        for name in STEER_NAMES:
            joint = body.find(f".//joint[@name='{prefix}{name}']")
            joint.set('limited', 'true')
            joint.set('range', f'{-STEERING_LIMIT} {STEERING_LIMIT}')
        world.append(body)
        for source_drive in drives:
            drive = copy.deepcopy(source_drive)
            drive.set('name', prefix + drive.get('name'))
            drive.set('joint', prefix + drive.get('joint'))
            if drive.tag == 'position':
                drive.set('ctrllimited', 'true')
                drive.set('ctrlrange', f'{-STEERING_LIMIT} {STEERING_LIMIT}')
            actuators.append(drive)
    _road(world, asset, track)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space='  ')
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return path


class RaceMujocoEnv:
    """60 Hz controls, 480 Hz physics, and native inter-car/track contacts."""

    def __init__(self, track, num_cars=2, render=False):
        self.track, self.num_cars = track, int(num_cars)
        self.version = mujoco.__version__
        self.model_path = build_model(track, self.num_cars)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.m, self.d = self.model, self.data
        self._cars = []
        self._geom_car = np.full(self.model.ngeom, -1, dtype=int)
        self._barriers = set()
        for i in range(self.num_cars):
            prefix = f'car{i}_'
            joints = [self.model.joint(prefix+n).id for n in WHEEL_NAMES+STEER_NAMES]
            free = self.model.joint(prefix+'floating_base').id
            self._cars.append({
                'body': self.model.body(prefix+'base_link').id,
                'laser': self.model.body(prefix+'laser').id,
                'target': self.model.geom(prefix+'lidar_target').id,
                'free_qpos': self.model.jnt_qposadr[free],
                'qpos': self.model.jnt_qposadr[joints],
                'qvel': self.model.jnt_dofadr[joints],
                'actuators': [self.model.actuator(prefix+n+'_drive').id for n in WHEEL_NAMES+STEER_NAMES],
            })
        for g in range(self.model.ngeom):
            name = self.model.geom(g).name or ''
            if name.startswith('car'):
                self._geom_car[g] = int(name.split('_', 1)[0][3:])
            elif name.startswith('barrier_'):
                self._barriers.add(g)
        self._renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 5., 110., -58.
        self._option = mujoco.MjvOption()
        self._option.geomgroup[3] = 0
        self.reset()
        if render:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)

    def reset(self, seed=0, starts=None):
        starts = [(2.*i, 0.) for i in range(self.num_cars)] if starts is None else list(starts)
        if len(starts) != self.num_cars:
            raise ValueError('starts must contain one (s, offset) pair per car')
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        for car, (s, offset) in zip(self._cars, starts):
            if hasattr(self.track, 'at3d'):
                xyz, yaw, grade = self.track.at3d(float(s), float(offset))
                xy, height = np.asarray(xyz)[:2], float(xyz[2])
            else:
                xy, yaw = self.track.at(float(s), float(offset))
                height, grade = 0., 0.
            if seed != 0:
                xy = np.asarray(xy) + rng.uniform(-.01, .01)*np.array([-math.sin(yaw), math.cos(yaw)])
                yaw += rng.uniform(-.01, .01)
            q = int(car['free_qpos'])
            self.data.qpos[q:q+3] = [*xy, height+.055]
            pitch = -math.atan(float(grade))
            cy, sy, cp, sp = math.cos(yaw/2), math.sin(yaw/2), math.cos(pitch/2), math.sin(pitch/2)
            self.data.qpos[q+3:q+7] = [cy*cp, -sy*sp, cy*sp, sy*cp]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.
        self._contacts = [set() for _ in self._cars]
        self._collision_totals = np.zeros(self.num_cars, dtype=int)
        self._collect_contacts()
        return self.state()

    def _collect_contacts(self):
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            ca, cb = self._geom_car[a], self._geom_car[b]
            if ca >= 0 and cb >= 0 and ca != cb:
                pair = ('car', min(a, b), max(a, b))
                self._contacts[ca].add(pair)
                self._contacts[cb].add(pair)
            elif ca >= 0 and b in self._barriers:
                self._contacts[ca].add(('barrier', a, b))
            elif cb >= 0 and a in self._barriers:
                self._contacts[cb].add(('barrier', b, a))

    def step(self, actions):
        actions = list(actions)
        if len(actions) != self.num_cars:
            raise ValueError('Expected one (four wheel velocities, two steering angles) action per car')
        targets = []
        for wheels, steer in actions:
            wheels, steer = np.asarray(wheels, dtype=float), np.asarray(steer, dtype=float)
            if wheels.shape != (4,) or steer.shape != (2,) or not (np.isfinite(wheels).all() and np.isfinite(steer).all()):
                raise ValueError('Expected four finite wheel velocities and two finite steering angles')
            targets.append(np.r_[wheels, np.clip(steer, -STEERING_LIMIT, STEERING_LIMIT)])
        for car, target in zip(self._cars, targets):
            self.data.ctrl[car['actuators']] = target
        self._contacts = [set() for _ in self._cars]
        for _ in range(round(CONTROL_DT/PHYSICS_DT)):
            mujoco.mj_step(self.model, self.data)
            self._collect_contacts()
        self._collision_totals += np.array([bool(c) for c in self._contacts], dtype=int)
        return self.state()

    def state(self):
        mujoco.mj_forward(self.model, self.data)
        states = []
        for i, car in enumerate(self._cars):
            pos = self.data.xpos[car['body']]
            mat = self.data.xmat[car['body']].reshape(3, 3)
            velocity, cg_velocity = np.empty(6), np.empty(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, car['body'], velocity, 0)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, car['body'], cg_velocity, 0)
            local = mat.T @ velocity[3:]
            contacts = self._contacts[i]
            wheels = self.data.qvel[car['qvel'][:4]]
            states.append({
                'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                'yaw': math.atan2(mat[1, 0], mat[0, 0]),
                'pitch': math.asin(float(np.clip(-mat[2, 0], -1, 1))),
                'roll': math.atan2(mat[2, 1], mat[2, 2]),
                'vx': float(velocity[3]), 'vy': float(velocity[4]), 'vz': float(velocity[5]),
                'yaw_rate': float(velocity[2]),
                'cg_vx': float(cg_velocity[3]), 'cg_vy': float(cg_velocity[4]),
                'rear_slip_beta': math.atan2(local[1], local[0]) if np.linalg.norm(local[:2]) > .1 else 0.,
                'wheel_vel': (wheels * [1, 1, 1, -1]).tolist(),
                'wheel_vel_raw': wheels.tolist(),
                'steer_pos': self.data.qpos[car['qpos'][4:]].tolist(),
                'lidar_pose': {'position': self.data.xpos[car['laser']].tolist(),
                               'rotation': self.data.xmat[car['laser']].reshape(3, 3).tolist()},
                'lidar_target': {'center': self.data.geom_xpos[car['target']].tolist(),
                                 'rotation': self.data.geom_xmat[car['target']].reshape(3, 3).tolist(),
                                 'half_size': self.model.geom_size[car['target']].tolist()},
                'collision': bool(contacts), 'collision_count': len(contacts),
                'barrier_collision': any(c[0] == 'barrier' for c in contacts),
                'car_collision': any(c[0] == 'car' for c in contacts),
                'collision_steps': int(self._collision_totals[i]),
                'time': float(self.data.time),
            })
        return states

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self.camera.lookat[:] = self.data.xpos[self._cars[0]['body']]
        self._renderer.update_scene(self.data, camera=self.camera, scene_option=self._option)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
