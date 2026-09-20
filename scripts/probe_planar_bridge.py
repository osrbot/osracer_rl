#!/usr/bin/env python3
"""Privileged straight bridge contact fixture, never a sensor-only race result."""
import json,sys,hashlib,math,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from racing.tracks import Track
from scripts.race_bridge_probe import serial
from racing.isaac_env import RaceIsaacEnv
ISAAC_ENV_SOURCE=(ROOT/"racing/isaac_env.py").read_bytes()
PROBE_SOURCE=Path(__file__).read_bytes()

def main():
 p=argparse.ArgumentParser();p.add_argument('--representation',default='native_cylinder',choices=['native_cylinder','rounded_cylinder','sdf120']);p.add_argument('--tag',default='bridge_planar_v4');p.add_argument('--contact-slop',type=float,default=0.);p.add_argument('--physics-hz',type=int,default=480);a=p.parse_args()
 env=RaceIsaacEnv(Track('bahrain'),num_cars=1,tire_representation=a.representation,physics_hz=a.physics_hz)
 from pxr import UsdGeom,UsdPhysics,PhysxSchema,UsdShade,Gf
 material=UsdShade.Material.Get(env.stage,'/World/TireMaterial')
 for name,x1,x2,z1,z2 in [('Ascent',170.,190.,0.,.9),('Platform',190.,198.,.9,.9),('Descent',198.,218.,.9,0.)]:
  dx=x2-x1;dz=z2-z1;theta=math.atan2(dz,dx);thickness=.08
  p=UsdGeom.Cube.Define(env.stage,'/World/PlanarBridge/'+name);p.CreateSizeAttr(1.)
  xf=UsdGeom.Xformable(p);xf.AddTranslateOp().Set(Gf.Vec3d((x1+x2)/2+math.sin(theta)*thickness/2,150.,(z1+z2)/2-math.cos(theta)*thickness/2))
  xf.AddRotateXYZOp().Set(Gf.Vec3f(0.,-math.degrees(theta),0.));xf.AddScaleOp().Set(Gf.Vec3f(math.hypot(dx,dz),3.,thickness))
  UsdPhysics.CollisionAPI.Apply(p.GetPrim());api=PhysxSchema.PhysxCollisionAPI.Apply(p.GetPrim());api.CreateContactOffsetAttr(.001);api.CreateRestOffsetAttr(0.);UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(material,materialPurpose='physics')
 wheel_settings={}
 for q in env.stage.Traverse():
  if str(q.GetPath()).startswith('/World/Car') and 'wheel_link' in str(q.GetPath()) and q.HasAPI(UsdPhysics.RigidBodyAPI):
   api=PhysxSchema.PhysxRigidBodyAPI.Apply(q)
   if a.contact_slop:api.CreateContactSlopCoefficientAttr(a.contact_slop)
   wheel_settings[str(q.GetPath())]=api.GetContactSlopCoefficientAttr().Get()
 from carb.settings import get_settings
 from omni.physx.bindings._physx import SETTING_COLLISION_APPROXIMATE_CYLINDERS
 cylinder_approximation=get_settings().get(SETTING_COLLISION_APPROXIMATE_CYLINDERS)
 from scripts.racing_contact_recorder import attach
 contacts,subscription=attach(env)
 env.world.reset();env.reset(starts=[(0.,0.)]);r=env.robots[0];r.set_world_pose(np.array([150.,150.,.055]),np.array([1.,0.,0.,0.]));r.set_world_velocity(np.zeros(6));r.set_joint_positions(np.zeros(r.num_dof));r.set_joint_velocities(np.zeros(r.num_dof));env._target(0,np.zeros(4),np.zeros(2))
 for _ in range(env.settle_steps):env.world.step(render=False)
 contacts['records'].clear()
 rows=[];state=env.states()[0]
 for i in range(1500):
  contacts['control_tick'][0]=i
  angle=math.atan2(150.-state['y'],.8)-state['yaw'];steer=float(np.clip(math.atan2(2*.28855*math.sin(angle),.8),-.45,.45))
  state=env.step([([155.,155.,155.,-155.],[steer,steer])])[0]
  x=state['x'];z=(x-170)*.045 if 170<x<190 else(.9 if 190<=x<=198 else((218-x)*.045 if 198<x<218 else 0.))
  rows.append(serial({'t':(i+1)/60,'action':{'wheel_velocity':[155.,155.,155.,-155.],'steering_position':[steer,steer]},'height_error':state['z']-z-.045,'state':state}))
  if x>225 or abs(state['roll'])>.5 or abs(state['pitch'])>.5:break
 report={'fixture_only':True,'physics_hz':env.physics_hz,'control_hz':60,'native_physics_dt':env.world.get_physics_dt(),'settle_seconds':env.settle_steps/env.physics_hz,'representation':a.representation,'ramp_length_m':20.,'height_m':.9,'grade':.045,'isaac_env_sha256':hashlib.sha256(ISAAC_ENV_SOURCE).hexdigest(),'rows':rows,'reached_goal':rows[-1]['state']['x']>225}
 report['cylinder_approximation_enabled']=cylinder_approximation;report['wheel_contact_slop_coefficients']=wheel_settings
 report['support_contacts']=contacts['records'];report['contact_targets']=contacts['targets']
 report['summary']={k:max(abs(row['state'][k]) for row in rows) for k in ['roll','pitch']};report['summary'].update(max_height_error=max(abs(row['height_error']) for row in rows),peak_speed=max(math.hypot(row['state']['vx'],row['state']['vy']) for row in rows),max_y_error=max(abs(row['state']['y']-150) for row in rows),entry_speed=next((math.hypot(row['state']['vx'],row['state']['vy']) for row in rows if row['state']['x']>=170),None))
 report['passed']=bool(report['reached_goal'] and report['summary']['max_y_error']<.15 and report['summary']['max_height_error']<.03 and report['summary']['entry_speed']>6.)
 out=ROOT/'output/racing'/f'{a.tag}_{a.representation}.json';out.write_text(json.dumps(report,indent=2)+'\n');out.with_suffix('.isaac_env.py').write_bytes(ISAAC_ENV_SOURCE);out.with_suffix('.probe.py').write_bytes(PROBE_SOURCE);print('PLANAR_RESULT',json.dumps({k:v for k,v in report.items() if k not in ('rows','support_contacts')}),flush=True);env.close()
if __name__=='__main__':main()
