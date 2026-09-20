#!/usr/bin/env python3
"""Native contact representation A/B. Privileged fixtures, not race scores."""
import argparse,json,sys,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from racing.tracks import Track
from race_bridge_probe import controls,serial

def main():
 p=argparse.ArgumentParser();p.add_argument('--representation',choices=['native_cylinder','convex120','convex120_scaled','convex120_compound','convex120_precise','rounded_cylinder','sdf120'],required=True);p.add_argument('--tag',default='contact_ab');a=p.parse_args()
 from racing.isaac_env import RaceIsaacEnv
 env=RaceIsaacEnv(Track('suzuka'),num_cars=1,tire_representation=a.representation)
 from pxr import UsdUtils,PhysicsSchemaTools,UsdPhysics
 import omni.physx
 result={'representation':a.representation,'fixture_only':True,'isaac_env_sha256':hashlib.sha256((ROOT/'racing/isaac_env.py').read_bytes()).hexdigest(),'cases':{}}
 result['physics_scene']={str(q.GetPath()):{str(v.GetName()):str(v.Get()) for v in q.GetAttributes() if 'gpu' in v.GetName().lower()} for q in env.stage.Traverse() if q.IsA(UsdPhysics.Scene)}
 if a.representation not in ('native_cylinder','rounded_cylinder','sdf120'):
  def hull_result(status,hulls):
   result['cooked_hulls']=[{'vertices':[[v.x,v.y,v.z] for v in h.vertices],'polygons':len(h.polygons)} for h in hulls]
  omni.physx.get_physx_cooking_interface().request_convex_collision_representation(stage_id=UsdUtils.StageCache.Get().GetId(env.stage).ToLongInt(),collision_prim_id=PhysicsSchemaTools.sdfPathToInt('/World/Car0/Links/left_front_wheel_link/DrivingCollider'+('/Part0' if a.representation=='convex120_compound' else '')),run_asynchronously=False,on_result=hull_result)
 result['mass_properties']={str(q.GetPath()):{'mass':UsdPhysics.MassAPI(q).GetMassAttr().Get(),'inertia':serial(np.asarray(UsdPhysics.MassAPI(q).GetDiagonalInertiaAttr().Get()))} for q in env.stage.Traverse() if q.HasAPI(UsdPhysics.RigidBodyAPI) and str(q.GetPath()).startswith('/World/Car')}
 for case,start,goal in [('high_speed_ascent',195.,228.),('high_speed_descent',245.,257.)]:
  states=env.reset(starts=[(start,0.)]);progress=[start];rows=[]
  for i in range(900):
   actions,progress,_=controls(env.track,states,progress,155.)
   states=env.step(actions);_,progress,ctes=controls(env.track,states,progress,0.)
   state=states[0];rows.append(serial({'t':(i+1)/60,'s':progress[0],'cte':ctes[0],'height_error':state['z']-.045-env.track.elevation_at(progress[0]),'state':state}))
   if progress[0]>=goal or state['collision'] or abs(state['roll'])>.5 or abs(state['pitch'])>.5:break
  result['cases'][case]={'reached_goal':progress[0]>=goal,'rows':rows}
 def flat_reset():
  env.reset(starts=[(195.,0.)]);r=env.robots[0];r.set_world_pose(np.array([150.,150.,.055]),np.array([1.,0.,0.,0.]));r.set_world_velocity(np.zeros(6));r.set_joint_positions(np.zeros(r.num_dof));r.set_joint_velocities(np.zeros(r.num_dof));env._target(0,np.zeros(4),np.zeros(2))
  for _ in range(200):env.world.step(render=False)
 for case,steps in [('flat_brake',240),('flat_turn',480)]:
  flat_reset();rows=[]
  for i in range(steps):
   speed=(155. if i<120 else 0.) if case=='flat_brake' else 66.6666667
   steer=0. if case=='flat_brake' else math.atan(.28855/4.)
   state=env.step([([speed,speed,speed,-speed],[steer,steer])])[0]
   rows.append(serial({'t':(i+1)/60,'state':state}))
  result['cases'][case]={'rows':rows}
 for case,data in result['cases'].items():
  rows=data['rows'];states=[r['state'] for r in rows];speeds=[math.hypot(r['vx'],r['vy']) for r in states]
  data.update(peak_speed=max(speeds),end_speed=speeds[-1],max_roll=max(abs(r['roll']) for r in states),max_pitch=max(abs(r['pitch']) for r in states),collision_steps=sum(r['collision'] for r in states))
  if 'height_error' in rows[0]:data.update(max_height_error=max(abs(r['height_error']) for r in rows),max_cte=max(abs(r['cte']) for r in rows),final_s=rows[-1]['s'],ramp_entry_speed=next((speeds[i] for i,r in enumerate(rows) if r['s']>=209.125941),None))
  else:data['max_flat_height_error']=max(abs(r['z']-.045) for r in states)
  if case=='flat_brake':
   b=states[119];stop=next((i for i in range(120,len(states)) if speeds[i]<.1),None);data.update(brake_entry_speed=speeds[119],stop_time=None if stop is None else(stop-119)/60,stop_distance=None if stop is None else math.hypot(states[stop]['x']-b['x'],states[stop]['y']-b['y']))
  if case=='flat_turn':
   xy=np.array([[r['x'],r['y']] for r in states[120:]]);xy-=xy.mean(axis=0);fit=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],(xy*xy).sum(axis=1),rcond=None)[0];data.update(fitted_radius=math.sqrt(fit[2]+fit[0]**2+fit[1]**2),mean_speed=float(np.mean(speeds[120:])))
 out=ROOT/'output/racing'/f'{a.tag}_{a.representation}.json';out.write_text(json.dumps(serial(result),indent=2)+'\n');print('CONTACT_AB',json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in result['cases'].items()}),flush=True);env.close()
if __name__=='__main__':main()
