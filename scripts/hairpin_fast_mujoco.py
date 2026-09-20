#!/usr/bin/env python3
"""Native MuJoCo adapter for the high-speed hairpin, without engine-specific tuning.

The original CAD, source model, and slower experiment remain unchanged. The
new road is visual only; ground contact, actuator gains and torque limits come
from hairpin_mujoco. Steering limits are explicit experimental assumptions.
"""
from __future__ import annotations
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from hairpin_mujoco import MujocoEnv, mujoco, build_model as build_base_model
from hairpin_mujoco import WHEEL_NAMES, STEER_NAMES, PHYSICS_DT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/hairpin_fast'
CONTROL_DT = 1 / 60
APPROACH, RADIUS, ROAD_WIDTH, STEERING_LIMIT = 6., .8, 1.2, .45


def _road(world, asset):
    points = [(float(x), 0.) for x in np.linspace(0., APPROACH, 61)]
    points += [(APPROACH + RADIUS*math.cos(a), RADIUS + RADIUS*math.sin(a))
               for a in np.linspace(-math.pi/2, math.pi/2, 101)[1:]]
    points += [(float(x), 2*RADIUS) for x in np.linspace(APPROACH, 0., 61)[1:]]
    vertices, faces = [], []
    for i, p in enumerate(points):
        tangent = np.array(points[min(len(points)-1, i+1)]) - points[max(0, i-1)]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]]) * ROAD_WIDTH/2
        for z in (.0006, .0002):
            for side in (1, -1):
                vertices.append([p[0]+side*normal[0], p[1]+side*normal[1], z])
    for i in range(len(points)-1):
        a, b = 4*i, 4*(i+1)
        faces.extend([[a,a+1,b+1],[a,b+1,b], [a+2,b+3,a+3],[a+2,b+2,b+3],
                      [a,b,b+2],[a,b+2,a+2], [a+1,a+3,b+3],[a+1,b+3,b+1]])
    last=4*(len(points)-1)
    faces.extend([[0,2,3],[0,3,1],[last,last+1,last+3],[last,last+3,last+2]])
    ET.SubElement(asset, 'mesh', name='fast_road_ribbon',
                  vertex=' '.join(str(v) for p in vertices for v in p),
                  face=' '.join(str(v) for p in faces for v in p))
    ET.SubElement(world, 'geom', name='fast_road_surface', type='mesh', mesh='fast_road_ribbon',
                  rgba='.13 .16 .20 1', contype='0', conaffinity='0')
    for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
        dx,dy=b[0]-a[0],b[1]-a[1]
        length,angle=math.hypot(dx,dy),math.atan2(dy,dx)
        x,y=(a[0]+b[0])/2,(a[1]+b[1])/2
        for side in (-1,1):
            offset=side*(ROAD_WIDTH/2-.009)
            ET.SubElement(world,'geom',name=f'fast_edge_{i}_{side}',type='box',
                          pos=f'{x-offset*math.sin(angle)} {y+offset*math.cos(angle)} .001',
                          size=f'{length/2+.003} .009 .0003',euler=f'0 0 {angle}',
                          rgba='.90 .93 .94 1',contype='0',conaffinity='0')
        if i%8<4:
            ET.SubElement(world,'geom',name=f'fast_center_{i}',type='box',
                          pos=f'{x} {y} .0012',size=f'{length/2} .008 .0003',
                          euler=f'0 0 {angle}',rgba='.88 .71 .29 1',contype='0',conaffinity='0')
    for name,y in (('start',0.),('finish',2*RADIUS)):
        for j in range(20):
            ET.SubElement(world,'geom',name=f'fast_{name}_{j}',type='box',
                          pos=f'0 {y-ROAD_WIDTH/2+(j+.5)*ROAD_WIDTH/20} .0016',
                          size=f'.03 {ROAD_WIDTH/40} .0004',
                          rgba='.95 .95 .95 1' if j%2 else '.12 .15 .18 1',
                          contype='0',conaffinity='0')


def build_model(path: Path | None = None) -> Path:
    path=Path(path or OUT/'mujoco_model.xml')
    build_base_model(path)
    tree=ET.parse(path)
    root=tree.getroot()
    world,asset=root.find('worldbody'),root.find('asset')
    for geom in list(world.findall('geom')):
        name=geom.get('name','')
        if name=='road_surface' or name.startswith(('edge_','center_','start_','finish_')):
            world.remove(geom)
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name')=='road_ribbon':
            asset.remove(mesh)
    _road(world,asset)
    for name in STEER_NAMES:
        joint=root.find(f".//joint[@name='{name}']")
        joint.set('limited','true')
        joint.set('range',f'{-STEERING_LIMIT} {STEERING_LIMIT}')
        actuator=root.find(f"./actuator/position[@joint='{name}']")
        actuator.set('ctrllimited','true')
        actuator.set('ctrlrange',f'{-STEERING_LIMIT} {STEERING_LIMIT}')
    ET.indent(tree,space='  ')
    tree.write(path,encoding='utf-8',xml_declaration=True)
    return path


class FastMujocoEnv(MujocoEnv):
    def __init__(self, model_path=None, render=False):
        self.version=mujoco.__version__
        self.model_path=build_model(model_path)
        self.model=mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data=mujoco.MjData(self.model)
        self.m,self.d=self.model,self.data
        joint_ids=[self.model.joint(n).id for n in WHEEL_NAMES+STEER_NAMES]
        self._qpos=np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        self._qvel=np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        self._actuators=[self.model.actuator(n+'_drive').id for n in WHEEL_NAMES+STEER_NAMES]
        self._body_id=self.model.body('base_link').id
        self._renderer=None
        self.camera=mujoco.MjvCamera()
        self.camera.distance=4.
        self.camera.azimuth=90
        self.camera.elevation=-65
        self.camera.lookat[:]=[1.5,RADIUS,0.]
        self._option=mujoco.MjvOption()
        self._option.geomgroup[3]=0
        self.reset()
        if render:
            self._renderer=mujoco.Renderer(self.model,height=720,width=1280)

    def step(self,wheel_targets,steering_targets):
        targets=np.concatenate((np.asarray(wheel_targets,dtype=float),np.asarray(steering_targets,dtype=float)))
        if targets.shape!=(6,) or not np.isfinite(targets).all():
            raise ValueError('Expected four finite wheel velocities and two finite steering angles')
        targets[4:]=np.clip(targets[4:],-STEERING_LIMIT,STEERING_LIMIT)
        self.data.ctrl[self._actuators]=targets
        for _ in range(round(CONTROL_DT/PHYSICS_DT)):
            mujoco.mj_step(self.model,self.data)
        return self.state()

    def state(self):
        state=super().state()
        # mjOBJ_BODY uses the inertial COM; mjOBJ_XBODY uses the base frame,
        # which is at the rear axle and matches the reported x/y path position.
        state['cg_vx'],state['cg_vy']=state['vx'],state['vy']
        velocity=np.empty(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_XBODY,
                                self._body_id,velocity,0)
        state['vx'],state['vy']=float(velocity[3]),float(velocity[4])
        local=self.data.xmat[self._body_id].reshape(3,3).T@velocity[3:]
        forward,lateral=float(local[0]),float(local[1])
        state['rear_slip_beta']=math.atan2(lateral,forward) if math.hypot(forward,lateral)>.1 else 0.
        return state

    def render(self):
        self.camera.lookat[:]=[float(np.clip(self.data.xpos[self._body_id,0],1.5,5.5)),RADIUS,0.]
        return super().render()
