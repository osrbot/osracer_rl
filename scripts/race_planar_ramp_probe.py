#!/usr/bin/env python3
"""Privileged native 4.5% planar ramp fixture; no benchmark policy score."""
import json,sys,math,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from racing.tracks import Track
from race_bridge_probe import serial

def main():
 p=argparse.ArgumentParser();p.add_argument('--representation',default='native_cylinder');p.add_argument('--tag',default='planar_ramp');a=p.parse_args()
 from racing.isaac_env import RaceIsaacEnv
 env=RaceIsaacEnv(Track('suzuka'),num_cars=1,tire_representation=a.representation)
 from pxr import UsdGeom,UsdPhysics,PhysxSchema,UsdShade,Gf
 env.world.stop()
 pitch=-math.atan(.045);normal=np.array([math.sin(pitch),0,math.cos(pitch)])
 center=np.array([159.,150.,.405])-normal*.04
 cube=UsdGeom.Cube.Define(env.stage,'/World/PlanarRamp');cube.CreateSizeAttr(1.)
 xf=UsdGeom.Xformable(cube);xf.AddTranslateOp().Set(Gf.Vec3d(*center));xf.AddRotateYOp().Set(math.degrees(pitch));xf.AddScaleOp().Set(Gf.Vec3f(math.hypot(22.,.99),1.5,.08))
 UsdPhysics.CollisionAPI.Apply(cube.GetPrim());contact=PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim());contact.CreateContactOffsetAttr(.001);contact.CreateRestOffsetAttr(0.)
 material=UsdShade.Material(env.stage.GetPrimAtPath('/World/TireMaterial'));UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material,materialPurpose='physics')
 env.world.reset();env.reset(starts=[(195.,0.)]);r=env.robots[0];r.set_world_pose(np.array([140.,150.,.055]),np.array([1.,0.,0.,0.]));r.set_world_velocity(np.zeros(6));r.set_joint_positions(np.zeros(r.num_dof));r.set_joint_velocities(np.zeros(r.num_dof));env._target(0,np.zeros(4),np.zeros(2))
 for _ in range(200):env.world.step(render=False)
 rows=[]
 for tick in range(900):
  st=env.step([([155.,155.,155.,-155.],[0.,0.])])[0]
  rows.append(serial({'t':(tick+1)/60,'state':st,'height_error':st['z']-.045-max(0.,.045*(st['x']-150.))}))
  if st['x']>=169. or abs(st['roll'])>.5 or abs(st['pitch'])>.5 or abs(st['y']-150)>.55:break
 result={'fixture_only':True,'representation':a.representation,'grade':.045,'ramp_from_x':150.,'goal_x':169.,'reached_goal':rows[-1]['state']['x']>=169.,'max_height_error':max(abs(r['height_error']) for r in rows),'entry_speed':next((math.hypot(r['state']['vx'],r['state']['vy']) for r in rows if r['state']['x']>=150.),None),'max_roll':max(abs(r['state']['roll']) for r in rows),'max_pitch':max(abs(r['state']['pitch']) for r in rows),'rows':rows}
 (ROOT/'output/racing'/f'{a.tag}_{a.representation}.json').write_text(json.dumps(result,indent=2)+'\n');print('PLANAR_RAMP',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True);env.close()
if __name__=='__main__':main()
