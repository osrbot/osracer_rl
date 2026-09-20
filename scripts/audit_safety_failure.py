#!/usr/bin/env python3
"""Replay native safety observations to inspect an exact failed controller trace."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from racing.sensors import LidarSensor
from racing.safety import LocalSafetySupervisor
from racing.tracks import Track
p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
result_path=Path(a.result); result=next(x for x in json.loads(result_path.read_text()) if x['seed']==a.seed)
trace_path=result_path.with_name(result_path.stem+f'_seed{a.seed}_trace.json');rows=json.loads(trace_path.read_text());track=Track(result['track']);sensor=LidarSensor(track,seed=a.seed);supervisor=LocalSafetySupervisor();states=result['initial_states'];data=[];max_error=0.
for i,row in enumerate(rows):
 obs=sensor.observe(states[0],states[0]['lidar_pose'],opponents=[states[1]],tick=i)
 wheels,steers=supervisor.filter(obs,*row['safety']['proposed_actions'])
 error=max(float(np.max(abs(wheels-np.array(row['actions'][0][0])))),float(np.max(abs(steers-np.array(row['actions'][0][1])))));max_error=max(max_error,error)
 target=float(supervisor.diagnostics['requested_center_steer_rad']);actual=float(np.mean(obs['steer_pos']));speed=supervisor.speed_bound
 current=supervisor._predict([],0.,actual,target);full=supervisor._predict([],speed,actual,target)
 old=supervisor.horizon;supervisor.horizon=.12;short=supervisor._predict([],speed,actual,target);supervisor.horizon=old
 if np.isfinite(current):data.append({'t':row['t'],'native_action_error':error,'present_compact_clearance_m':current,'requested_compact_clearance_m':full,'short_requested_compact_clearance_m':short,'speed_bound':speed,'intervened':supervisor.diagnostics['intervened'],'oriented_points':len(supervisor._oriented_obstacles),'compact_points':len(supervisor._obstacles),'requested_steer':target})
 states=row['states']
Path(a.output).write_text(json.dumps({'source':str(result_path),'seed':a.seed,'max_native_action_error':max_error,'rows':data},indent=2,allow_nan=False))
print(json.dumps({'max_native_action_error':max_error,'short_risk_frames':sum(r['short_requested_compact_clearance_m']<0 for r in data),'last_rows':data[-18:]}))
