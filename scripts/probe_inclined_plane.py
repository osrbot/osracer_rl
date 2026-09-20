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
 p=argparse.ArgumentParser();p.add_argument('--representation',default='native_cylinder',choices=['native_cylinder','rounded_cylinder','sdf120']);p.add_argument('--tag',default='bridge_inclined_plane_v1');p.add_argument('--support',choices=['plane','box'],default='plane');p.add_argument('--contact-slop',type=float,default=0.);p.add_argument('--box-thickness',type=float,default=.08);p.add_argument('--physics-hz',type=int,default=480);p.add_argument('--gpu-dynamics',action='store_true');p.add_argument('--custom-adapter',action='store_true');a=p.parse_args()
 if a.custom_adapter:
  import os
  schema_path=str(ROOT/'experimental/bridge_contact_adapter/schema')
  os.environ['PXR_PLUGINPATH_NAME']=schema_path+os.pathsep+os.environ.get('PXR_PLUGINPATH_NAME','')
 app=None;adapter=None;adapter_registry=None
 if a.gpu_dynamics or a.custom_adapter:app=RaceIsaacEnv.create_app()
 if a.custom_adapter:
  if a.representation!='native_cylinder':raise ValueError('custom adapter fixture requires unchanged native cylinder geometry')
  import ctypes
  from pxr import Plug
  Plug.Registry().RegisterPlugins(str(ROOT/'experimental/bridge_contact_adapter/schema'))
  adapter=ctypes.CDLL(str(ROOT/'output/racing/bridge_adapter_sdk/libbridge_contact_adapter.so'))
  adapter.bridge_register.restype=ctypes.c_size_t
  adapter_registry=adapter.bridge_register()
  if not adapter_registry:raise RuntimeError('IPhysxCustomGeometry adapter registration failed')
 if a.gpu_dynamics:
  import isaacsim.core.api as core_api
  OriginalWorld=core_api.World
  def diagnostic_world(*args,**kwargs):
   world=OriginalWorld(*args,**kwargs)
   from isaacsim.core.simulation_manager import SimulationManager
   SimulationManager.enable_gpu_dynamics(True)
   SimulationManager.set_broadphase_type('GPU')
   return world
  core_api.World=diagnostic_world
 env=RaceIsaacEnv(Track('bahrain'),num_cars=1,tire_representation=a.representation,physics_hz=a.physics_hz,simulation_app=app)
 if a.gpu_dynamics:core_api.World=OriginalWorld
 from pxr import UsdGeom,UsdPhysics,PhysxSchema,UsdShade,Gf
 adapter_prims=[];baseline_properties=None
 if adapter is not None:
  view=env.robots[0]._articulation_view
  baseline_properties={'body_names':view.body_names,'masses':serial(view._physics_view.get_masses()),'inertias':serial(view._physics_view.get_inertias())}
  env.world.stop()
  for prim in env.stage.Traverse():
   if prim.GetName()=='DrivingCollider':
    prim.AddAppliedSchema('BridgeCylinderAPI')
    if 'BridgeCylinderAPI' not in prim.GetAppliedSchemas():raise RuntimeError('custom API not recognized by USD schema registry')
    adapter_prims.append(str(prim.GetPath()))
 material=UsdShade.Material.Get(env.stage,'/World/TireMaterial')
 theta=math.atan(.045)
 if a.support=='box':
  for q in env.stage.Traverse():
   if str(q.GetPath()).startswith('/World/defaultGroundPlane/') and q.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
  x1,x2=145.,200.;thickness=a.box_thickness
  q=UsdGeom.Cube.Define(env.stage,'/World/PlanarBridge/ConstantIncline');q.CreateSizeAttr(1.)
  xf=UsdGeom.Xformable(q);xf.AddTranslateOp().Set(Gf.Vec3d((x1+x2)/2+math.sin(theta)*thickness/2,150.,((x1+x2)/2-150)*.045-math.cos(theta)*thickness/2));xf.AddRotateXYZOp().Set(Gf.Vec3f(0.,-math.degrees(theta),0.));xf.AddScaleOp().Set(Gf.Vec3f((x2-x1)/math.cos(theta),3.,thickness))
  UsdPhysics.CollisionAPI.Apply(q.GetPrim());api=PhysxSchema.PhysxCollisionAPI.Apply(q.GetPrim());api.CreateContactOffsetAttr(.001);api.CreateRestOffsetAttr(0.);UsdShade.MaterialBindingAPI.Apply(q.GetPrim()).Bind(material,materialPurpose='physics')
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
 env.world.reset()
 if a.support=='plane':
  ground=env.stage.GetPrimAtPath('/World/defaultGroundPlane')
  ground.GetAttribute('xformOp:orient').Set(Gf.Quatd(math.cos(theta/2),0.,-math.sin(theta/2),0.))
  ground.GetAttribute('xformOp:translate').Set(Gf.Vec3d(0.,0.,-150*.045))
 env.reset(starts=[(0.,0.)]);r=env.robots[0];r.set_world_pose(np.array([150.,150.,.055]),np.array([math.cos(theta/2),0.,-math.sin(theta/2),0.]));r.set_world_velocity(np.zeros(6));r.set_joint_positions(np.zeros(r.num_dof));r.set_joint_velocities(np.zeros(r.num_dof));env._target(0,np.zeros(4),np.zeros(2))
 for _ in range(env.settle_steps):env.world.step(render=False)
 runtime={'gpu_dynamics_requested':a.gpu_dynamics,'scene_attributes':{},'sdf_query':None}
 for prim in env.stage.Traverse():
  if prim.IsA(UsdPhysics.Scene):
   runtime['scene_attributes'][str(prim.GetPath())]={attr.GetName():serial(attr.Get()) for attr in prim.GetAttributes() if 'physxScene:' in attr.GetName()}
 try:
  view=env.robots[0]._articulation_view
  runtime['articulation_properties']={'body_names':view.body_names,'masses':serial(view._physics_view.get_masses()),'inertias':serial(view._physics_view.get_inertias())}
 except Exception as exc:runtime['articulation_properties']={'error':repr(exc)}
 try:
  import ctypes
  inspector=ctypes.CDLL(str(ROOT/'output/racing/bridge_adapter_sdk/libbridge_contact_adapter.so'))
  runtime['native_inspector_sha256']=hashlib.sha256((ROOT/'output/racing/bridge_adapter_sdk/libbridge_contact_adapter.so').read_bytes()).hexdigest()
  runtime['native_shapes']={}
  for prim in env.stage.Traverse():
   if prim.GetName()=='DrivingCollider':
    fields=(ctypes.c_ulong*9)();code=inspector.bridge_inspect_shape(str(prim.GetPath()).encode(),fields)
    runtime['native_shapes'][str(prim.GetPath())]={'status':code,'geometry_type':fields[0],'shape_flags':fields[1],'sdf_dimensions':list(fields[2:5]),'has_sdf_data':bool(fields[5]),'scene_flags':fields[6],'scene_gpu_dynamics':bool(fields[7]),'scene_pcm':bool(fields[8])}
 except Exception as exc:runtime['native_shapes']={'error':repr(exc)}
 if a.representation=='sdf120':
  try:
   import torch
   import omni.physics.tensors as tensors
   from pxr import UsdUtils
   stage_id=UsdUtils.StageCache.Get().GetId(env.stage).ToLongInt()
   sdf_sim=tensors.create_simulation_view('torch',stage_id=stage_id)
   sdf_view=sdf_sim.create_sdf_shape_view('/World/Car*/Links/*wheel_link/DrivingCollider',3)
   count=sdf_view.count
   samples=torch.tensor([[[0.,0.,0.],[.046,0.,0.],[0.,.021,0.]]]*count,device='cuda',dtype=torch.float32)
   values=sdf_view.get_sdf_and_gradients(samples).detach().cpu().numpy().tolist() if count else []
   runtime['sdf_query']={'count':count,'paths':list(sdf_view.object_paths),'valid':sdf_view.check(),'samples':samples.cpu().tolist(),'values':values,'error':None}
  except Exception as exc:runtime['sdf_query']={'error':repr(exc)}
 if adapter is not None:
  counters=(ctypes.c_ulong*5)();adapter.bridge_counters(counters)
  runtime['custom_adapter']={'registration_id':adapter_registry,'prims':adapter_prims,'native_baseline_articulation_properties':baseline_properties,'counters_before':list(counters),'counter_names':['created','contact_calls','analytic_contacts','edge_fallback_contacts','unsupported_geometry_pairs'],'geometry_scope':'box and plane only; no scene query callbacks; fixture only'}
 runtime_path=ROOT/'output/racing'/f'{a.tag}_{a.support}_{a.representation}.runtime.json'
 runtime_path.write_text(json.dumps(runtime,indent=2,default=str)+'\n')
 print('RUNTIME_QUALIFICATION',json.dumps(runtime,default=str),flush=True)
 contacts['records'].clear()
 rows=[];state=env.states()[0]
 for i in range(600):
  contacts['control_tick'][0]=i
  angle=math.atan2(150.-state['y'],.8)-state['yaw'];steer=float(np.clip(math.atan2(2*.28855*math.sin(angle),.8),-.45,.45))
  state=env.step([([155.,155.,155.,-155.],[steer,steer])])[0]
  x=state['x'];z=(x-150)*.045
  rows.append(serial({'t':(i+1)/60,'action':{'wheel_velocity':[155.,155.,155.,-155.],'steering_position':[steer,steer]},'height_error':state['z']-z-.045,'state':state}))
  if x>180 or abs(state['roll'])>.5 or abs(state['pitch'])>.5:break
 report={'fixture_only':True,'physics_hz':env.physics_hz,'control_hz':60,'physics_substeps':env.physics_substeps,'native_physics_dt':env.world.get_physics_dt(),'settle_seconds':env.settle_steps/env.physics_hz,'representation':a.representation,'ramp_length_m':None,'surface':a.support,'box_thickness_m':a.box_thickness if a.support=='box' else None,'start_inside_support':True,'height_m':None,'grade':.045,'isaac_env_sha256':hashlib.sha256(ISAAC_ENV_SOURCE).hexdigest(),'rows':rows,'reached_goal':rows[-1]['state']['x']>180}
 if adapter is not None:
  adapter.bridge_counters(counters);runtime['custom_adapter']['counters_after']=list(counters)
  runtime['custom_adapter']['qualified']=bool(counters[0]>=4 and counters[1]>0 and counters[4]==0)
 report['runtime_qualification']=runtime
 report['cylinder_approximation_enabled']=cylinder_approximation;report['wheel_contact_slop_coefficients']=wheel_settings
 report['support_contacts']=contacts['records'];report['contact_targets']=contacts['targets']
 report['summary']={k:max(abs(row['state'][k]) for row in rows) for k in ['roll','pitch']};report['summary'].update(max_height_error=max(abs(row['height_error']) for row in rows),peak_speed=max(math.hypot(row['state']['vx'],row['state']['vy']) for row in rows),max_y_error=max(abs(row['state']['y']-150) for row in rows),entry_speed=next((math.hypot(row['state']['vx'],row['state']['vy']) for row in rows if row['state']['x']>=170),None))
 runtime_ok=True
 if a.representation=='sdf120':
  shapes=list(runtime.get('native_shapes',{}).values())
  runtime_ok=len(shapes)==4 and all(isinstance(x,dict) and x.get('status')==0 and x.get('geometry_type')==8 and x.get('has_sdf_data') and x.get('scene_gpu_dynamics') and x.get('scene_pcm') for x in shapes)
  runtime['actual_sdf_runtime_qualified']=runtime_ok
 if adapter is not None:
  shapes=list(runtime.get('native_shapes',{}).values())
  runtime_ok=runtime['custom_adapter']['qualified'] and len(shapes)==4 and all(isinstance(x,dict) and x.get('geometry_type')==10 for x in shapes)
  runtime_ok=runtime_ok and baseline_properties==runtime['articulation_properties']
  runtime['custom_adapter']['qualified']=runtime_ok
 report['runtime_qualified']=bool(runtime_ok)
 report['passed']=bool(runtime_ok and report['reached_goal'] and report['summary']['max_y_error']<.15 and report['summary']['max_height_error']<.03 and report['summary']['entry_speed']>6.)
 out=ROOT/'output/racing'/f'{a.tag}_{a.support}_{a.representation}.json';out.write_text(json.dumps(report,indent=2,default=str)+'\n');out.with_suffix('.isaac_env.py').write_bytes(ISAAC_ENV_SOURCE);out.with_suffix('.probe.py').write_bytes(PROBE_SOURCE);print('PLANAR_RESULT',json.dumps({k:v for k,v in report.items() if k not in ('rows','support_contacts')},default=str),flush=True);env.close()
 if adapter is not None:adapter.bridge_unregister()
 if app is not None:app.close()
if __name__=='__main__':main()
