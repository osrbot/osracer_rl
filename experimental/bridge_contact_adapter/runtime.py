"""Explicit experimental adapter environment; production default stays unchanged."""
import ctypes,hashlib,json,os
from pathlib import Path
import numpy as np
from racing.isaac_env import RaceIsaacEnv
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
LOADER_SOURCE_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SUPPORTED=['sphere','plane','capsule','box','convex_core','convex_mesh','triangle_mesh','BridgeCylinderAPI wheel']
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 return x

def properties(env):
 out=[]
 for robot in env.robots:
  view=robot._articulation_view
  out.append({'body_names':view.body_names,'masses':serial(view._physics_view.get_masses()),'inertias':serial(view._physics_view.get_inertias())})
 return out

class AdapterRaceIsaacEnv(RaceIsaacEnv):
 def __init__(self,*args,simulation_app=None,**kwargs):
  schema=str(HERE/'schema');os.environ['PXR_PLUGINPATH_NAME']=schema+os.pathsep+os.environ.get('PXR_PLUGINPATH_NAME','')
  self._adapter_owns_app=simulation_app is None
  app=simulation_app if simulation_app is not None else self.create_app()
  from pxr import Plug,UsdPhysics
  Plug.Registry().RegisterPlugins(schema)
  self.adapter=ctypes.CDLL(str(ROOT/'output/racing/bridge_adapter_sdk/libbridge_contact_adapter.so'))
  binary=ROOT/'output/racing/bridge_adapter_sdk/libbridge_contact_adapter.so'
  self._adapter_identity=json.loads(binary.with_suffix('.build.json').read_text())
  if self._adapter_identity['binary_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest():raise RuntimeError('Adapter binary/build manifest mismatch')
  self._adapter_identity['loader_source_sha256']=LOADER_SOURCE_HASH
  self.adapter.bridge_register.restype=ctypes.c_size_t
  self.adapter.bridge_rejected_internal_edges.restype=ctypes.c_ulong
  self.registration_id=self.adapter.bridge_register()
  if not self.registration_id:raise RuntimeError('Custom geometry registration failed')
  if self.adapter.bridge_query_selftest():raise RuntimeError('Custom geometry query self-test failed')
  kwargs['tire_representation']='native_cylinder'
  super().__init__(*args,simulation_app=app,**kwargs)
  self.baseline_properties=properties(self)
  self.world.stop();self.adapter_paths=[]
  for prim in self.stage.Traverse():
   if prim.GetName()=='DrivingCollider':
    prim.AddAppliedSchema('BridgeCylinderAPI')
    if 'BridgeCylinderAPI' not in prim.GetAppliedSchemas():raise RuntimeError('Schema must be registered before application startup')
    self.adapter_paths.append(str(prim.GetPath()))
  self.world.reset();self.reset()
  self.physics_metadata=self.adapter_qualification()
  if not self.physics_metadata['qualified']:raise RuntimeError('Adapter startup qualification failed: '+json.dumps(self.physics_metadata))

 def adapter_qualification(self):
  from pxr import UsdPhysics
  binary=ROOT/'output/racing/bridge_adapter_sdk/libbridge_contact_adapter.so'
  sources=self._adapter_identity['compiled_source_sha256'].copy();sources['loader:runtime.py']=self._adapter_identity['loader_source_sha256']
  base=(ctypes.c_ulong*5)();extra=(ctypes.c_ulong*5)();self.adapter.bridge_counters(base);self.adapter.bridge_extended_counters(extra)
  shapes={};inventory={}
  for prim in self.stage.Traverse():
   if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
   fields=(ctypes.c_ulong*9)();status=self.adapter.bridge_inspect_shape(str(prim.GetPath()).encode(),fields)
   if status==0:inventory[str(prim.GetPath())]=int(fields[0])
   if str(prim.GetPath()) in self.adapter_paths:shapes[str(prim.GetPath())]={'status':status,'geometry_type':fields[0],'scene_flags':fields[6],'gpu_dynamics':bool(fields[7]),'pcm':bool(fields[8])}
  current=properties(self)
  qualified=len(shapes)==4*self.num_cars and all(s['status']==0 and s['geometry_type']==10 for s in shapes.values()) and current==self.baseline_properties and base[4]==0 and extra[0]==0 and extra[4]==0
  result={'experimental_adapter':'BridgeCylinderAPI','qualified':bool(qualified),'binary_sha256':self._adapter_identity['binary_sha256'],'source_sha256':sources,'sdk':'PhysX 5.9.0 / Omni 110.1 public headers','sdk_sources':self._adapter_identity['sdk_sources'],'sdk_tag':'110.1-omni-and-physx-5.9.0','usd_abi':'0.25.11','binary_patch_commit_verified':False,'physics_hz':self.physics_hz,'actual_physics_dt_s':float(self.world.get_physics_dt()),'settling_steps':self.settle_steps,'actual_settling_seconds':self.settle_steps*float(self.world.get_physics_dt()),'solver_position_iterations':[serial(r.get_solver_position_iteration_count()) for r in self.robots],'solver_velocity_iterations':[serial(r.get_solver_velocity_iteration_count()) for r in self.robots],'control_hz':60,'physics_substeps':self.physics_substeps,'supported_collision_types':SUPPORTED,'unsupported_scope':['heightfields','particles','deformables','unregistered custom geometry'],'rejected_internal_edge_candidates':int(self.adapter.bridge_rejected_internal_edges()),'queries':['raycast','overlap','translation-only conservative sweep'],'counters':dict(zip(['created','contact_calls','analytic_contacts','edge_contacts','unsupported_pairs'],map(int,base))),'query_counters':dict(zip(['numerical_failures','raycasts','overlaps','sweeps','query_failures'],map(int,extra))),'shapes':shapes,'scene_geometry_inventory':inventory,'native_baseline_mass_inertia':self.baseline_properties,'actual_mass_inertia':current,'mass_inertia_unchanged':current==self.baseline_properties}
  self.physics_metadata=result
  return result

 def close(self):
  self.physics_metadata=self.adapter_qualification()
  super().close()
  self.adapter.bridge_unregister()
  if self._adapter_owns_app:self.app.close()
