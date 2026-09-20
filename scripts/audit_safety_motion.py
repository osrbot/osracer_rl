#!/usr/bin/env python3
"""Compare sensor-derived relative motion with native trace truth offline only."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from racing.sensors import LidarSensor
from racing.odometry import LidarOdometry
from racing.tracks import Track
p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--seed',required=True,type=int);p.add_argument('--output',required=True);a=p.parse_args()
result_path=Path(a.result);res=next(r for r in json.loads(result_path.read_text()) if r['seed']==a.seed);rows=json.loads(result_path.with_name(result_path.stem+f'_seed{a.seed}_trace.json').read_text());sensor=LidarSensor(Track(res['track']),seed=a.seed);motion=LidarOdometry();states=res['initial_states'];out=[]
for i,row in enumerate(rows):
 obs=sensor.observe(states[0],states[0]['lidar_pose'],opponents=[states[1]],tick=i);estimate=motion.update(obs);s=states[0];c,z=math.cos(s['yaw']),math.sin(s['yaw']);vx=c*s['vx']+z*s['vy'];vy=-z*s['vx']+c*s['vy'];speed=float(np.mean(np.array(obs['wheel_vel'][:2])*np.cos(obs['steer_pos']))*.045);out.append({'t':row['t'],'estimate':estimate,'truth_for_audit_only':{'forward_speed':vx,'lateral_speed':vy,'yaw_rate':s['yaw_rate']},'bicycle_yaw_rate':speed*math.tan(float(np.mean(obs['steer_pos'])))/.28764});states=row['states']
Path(a.output).write_text(json.dumps({'scope':'Truth used for offline estimator audit only, not actor input','rows':out},indent=2));print(json.dumps([r for r in out if 5.7<r['t']<6.1 and r['estimate']['age']<.001]))
