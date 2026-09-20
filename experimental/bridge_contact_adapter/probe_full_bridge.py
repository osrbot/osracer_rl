#!/usr/bin/env python3
"""Privileged high-speed real Suzuka bridge/underpass fixture, not policy eval."""
import sys,json,math,argparse,traceback,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from racing.tracks import Track
from scripts.race_bridge_probe import controls,serial
from scripts.racing_contact_recorder import attach
from experimental.bridge_contact_adapter.runtime import AdapterRaceIsaacEnv

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',default='bridge_adapter_full_v4');p.add_argument('--wheel-speed',type=float,default=155.);p.add_argument('--render',action='store_true');p.add_argument('--max-seconds',type=float,default=22.);a=p.parse_args()
 out=ROOT/'output/racing'/a.tag;track=Track('suzuka');bridge=track.metadata['bridge']
 report={'fixture_only':True,'sensor_only_policy_evaluation':False,'native_wheel_command_rad_s':a.wheel_speed,'track_sha256':hashlib.sha256(track.path.read_bytes()).hexdigest(),'bridge':bridge,'cases':[],'passed':False};env=None
 try:
  env=AdapterRaceIsaacEnv(track,num_cars=2,render=a.render)
  raw,subscription=attach(env,maximum_control_tick=2000)
  env.world.reset()
  for case in ['full_bridge','crossing']:
   crossing=case=='crossing';starts=[bridge['upper_s']-.7,bridge['lower_s']-.7] if crossing else [bridge['elevated_start_s']-20.,bridge['lower_s']-1.]
   states=env.reset(starts=[(s,0.) for s in starts]);progress=starts.copy();rows=[];raw['records'].clear();failures=[];entry_speed=None;min_xy=1e9;gap=None;max_height_error=0;max_roll=0;screenshot=None
   goal=bridge['upper_s']+1.5 if crossing else bridge['elevated_end_s']+2.
   for tick in range(round(a.max_seconds*60)):
    raw['control_tick'][0]=tick
    actions,before,_=controls(track,states,progress,30. if crossing else a.wheel_speed,None if crossing else bridge['lower_s']+4.)
    if not crossing and before[1]<bridge['lower_s']+4.:actions[1]=([35.,35.,35.,-35.],actions[1][1])
    states=env.step(actions);_,progress,cte=controls(track,states,before,0.)
    errors=[abs(s['z']-track.elevation_at(progress[j])-.045) for j,s in enumerate(states)]
    max_height_error=max(max_height_error,*errors);max_roll=max(max_roll,*(abs(s['roll']) for s in states))
    xy=math.hypot(states[0]['x']-states[1]['x'],states[0]['y']-states[1]['y']);min_xy=min(min_xy,xy)
    if xy<.1:gap=states[0]['z']-states[1]['z']
    speed=math.hypot(states[0]['vx'],states[0]['vy'])
    if entry_speed is None and progress[0]>=bridge['elevated_start_s']:entry_speed=speed
    rows.append(serial({'t':(tick+1)/60,'states':states,'actions':actions,'progress':progress,'cte':cte,'height_errors':errors,'xy_separation':xy}))
    for j,s in enumerate(states):
     if s['collision']:failures.append(f'car{j}: physical collision')
     if abs(cte[j])>track.width/2-.125:failures.append(f'car{j}: off road')
     if abs(s['roll'])>.5 or abs(s['pitch'])>.5 or s['z']<.015:failures.append(f'car{j}: unstable')
    if a.render and screenshot is None and ((xy<.1) if crossing else progress[0]>=bridge['upper_s']):
     from PIL import Image,ImageDraw
     im=Image.fromarray(env.render());ImageDraw.Draw(im).text((20,20),'PRIVILEGED GEOMETRY FIXTURE / real bridge / native contact adapter',fill='white');screenshot=str(out.with_name(out.name+'_'+case+'.png'));im.save(screenshot)
    if failures or progress[0]>=goal:break
   if progress[0]<goal:failures.append('goal not reached')
   if max_height_error>.03:failures.append('height tracking error exceeds 3 cm')
   if not crossing and (entry_speed or 0)<6:failures.append('bridge entry below 6 m/s')
   if crossing and (min_xy>=.1 or gap is None or gap<bridge['surface_height']-.03):failures.append('physical underpass clearance not demonstrated')
   physics=env.adapter_qualification()
   if not physics['qualified']:failures.append('native adapter runtime qualification failed')
   path=out.with_name(out.name+'_'+case+'.json')
   data={'case':case,'passed':not failures,'failures':sorted(set(failures)),'fixture_only':True,'start_s':starts,'goal_s':goal,'entry_speed':entry_speed,'peak_speed':max(math.hypot(r['states'][0]['vx'],r['states'][0]['vy']) for r in rows),'max_height_error':max_height_error,'max_roll':max_roll,'min_xy_separation':min_xy,'crossing_height_gap':gap,'screenshot':screenshot,'rows':rows,'support_contacts':raw['records'],'physics_metadata':physics}
   path.write_text(json.dumps(data,indent=2)+'\n');report['cases'].append({k:v for k,v in data.items() if k not in ['rows','support_contacts','physics_metadata']});report['cases'][-1]['artifact']=str(path);report['physics_metadata']=physics
   out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print('FULL_BRIDGE_CASE',json.dumps(report['cases'][-1]),flush=True)
  report['passed']=all(c['passed'] for c in report['cases'])
 except Exception:report['error']=traceback.format_exc();traceback.print_exc()
 finally:
  out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print('FULL_BRIDGE_RESULT',report['passed'],str(out.with_suffix('.json')),flush=True)
  if env is not None:env.close()
if __name__=='__main__':main()
