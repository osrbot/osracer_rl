#!/usr/bin/env python3
"""Inspect a frozen observed-motion ensemble at a native failed recovery frame.

Sensor reconstruction uses trace environment states only offline. The filter
receives the same allowed observation and previously logged local hypotheses.
"""
from pathlib import Path
import sys,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from racing.safety import LocalSafetySupervisor,CENTER_LIMIT
from racing.sensors import LidarSensor
from racing.tracks import Track
result=Path(sys.argv[1]);seed=int(sys.argv[2]);out=Path(sys.argv[3])
r=next(r for r in json.loads(result.read_text()) if r['seed']==seed)
rows=json.loads(result.with_name(result.stem+f'_seed{seed}_trace.json').read_text())
sensor=LidarSensor(Track(r['track']),seed=seed);states=r['initial_states']
for tick,row in enumerate(rows):
 obs=sensor.observe(states[0],states[0]['lidar_pose'],opponents=[states[1]],tick=tick)
 if tick==len(rows)-1:break
 states=row['states']
safety=row['safety'];hypotheses=safety['motion']['hypotheses']
class LoggedMotion(LocalSafetySupervisor):
 def _motion_hypotheses(self,*args,**kwargs):return hypotheses
s=LoggedMotion();s.speed_bound=safety['speed_bound_m_s']
action=s.filter(obs,*safety['proposed_actions'])
error=max(float(np.max(abs(np.asarray(a)-b))) for a,b in zip(action,row['actions'][0]))
actual=float(np.mean(obs['steer_pos']));target=safety['requested_center_steer_rad']
scores=[]
for candidate in np.unique(np.r_[np.linspace(-CENTER_LIMIT,CENTER_LIMIT,25),actual,target,0.]):
 per_scene=[]
 for scene in s._scenes:
  for mode,decay in scene['models']:
   kwargs=dict(motion=scene['motion'],mode=mode,yaw_bias_decay=decay,obstacles=scene['obstacles'],oriented=scene['oriented'],braking_deceleration=0. if scene['retained_speed'] else 4.)
   present=s._predict(scene['walls'],0.,actual,actual,horizon=0.,**kwargs)
   gap=s._predict(scene['walls'],.5,actual,candidate,response_rate=1.5,command_speed=.5,**kwargs)
   end=s._predict(scene['walls'],.5,actual,candidate,response_rate=1.5,command_speed=.5,end_only=True,**kwargs)
   per_scene.append({'motion':scene['motion'],'retained':scene['retained_speed'],'mode':mode,'decay':decay,'present':present,'gap':gap,'end':end})
 scores.append({'candidate':float(candidate),'scenes':per_scene})
out.write_text(json.dumps({'source':str(result),'safety_sha256':hashlib.sha256((ROOT/'racing/safety.py').read_bytes()).hexdigest(),'scope':'Offline inspection of logged observed-motion ensemble, not a new closed-loop result','frame_t':row['t'],'same_action_error':error,'actual':actual,'target':target,'diagnostics':s.diagnostics,'candidates':scores},indent=2)+'\n')
print(json.dumps({'same_action_error':error,'best_by_min_end':max(scores,key=lambda p:min(x['end'] for x in p['scenes']))},indent=2))
