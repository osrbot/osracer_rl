#!/usr/bin/env python3
"""Short native contact diagnostic; privileged probe, never race qualification."""
import json,sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from racing.tracks import Track
from racing.isaac_env import RaceIsaacEnv
from race_bridge_probe import controls,serial
track=Track('suzuka');env=RaceIsaacEnv(track,num_cars=1)
from pxr import PhysxSchema,UsdPhysics,PhysicsSchemaTools
env.world.stop()
bridges=[p.GetPath() for p in env.stage.Traverse() if str(p.GetPath()).startswith("/World/BridgeContacts/")]
for p in env.stage.Traverse():
 if str(p.GetPath()).startswith('/World/Car') and p.HasAPI(UsdPhysics.RigidBodyAPI):
  PhysxSchema.PhysxContactReportAPI(p).CreateReportPairsRel().SetTargets(bridges)
env.world.reset()
records=[];tick=[0]
def path(n):return str(PhysicsSchemaTools.intToSdfPath(n))
def on_contacts(headers,data):
 for h in headers:
  paths={k:path(getattr(h,k)) for k in ('actor0','actor1','collider0','collider1')}
  if not any('BridgeContacts' in p for p in paths.values()):continue
  rec={'tick':tick[0],**paths,'contacts':[]}
  for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
   rec['contacts'].append({k:serial(np.asarray(getattr(d,k))) for k in ('position','normal','impulse','separation')})
  records.append(rec)
import omni.physx
sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
states=env.reset(starts=[(195.,0.)]);progress=[195.];rows=[]
for i in range(240):
 tick[0]=i
 actions,progress,_=controls(track,states,progress,155.)
 states=env.step(actions)
 rows.append(serial({'t':(i+1)/60,'states':states,'s':progress}))
 if abs(states[0]['roll'])>.5:break
Path('output/racing/bridge_contact_diagnostic.json').write_text(json.dumps({'contacts':records,'trace':rows},indent=2))
print('DIAG_DONE',len(records),flush=True)
sub=None;env.close()
