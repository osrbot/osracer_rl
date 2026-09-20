#!/usr/bin/env python3
"""Frozen-checkpoint sensor-only rollout of the explicit experimental backend."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from racing.tracks import Track
from racing.policy_bundle import actor_source_hashes,configuration
from racing.run import rollout,save,RUNTIME_SOURCE_SHA256
from experimental.bridge_contact_adapter.runtime import AdapterRaceIsaacEnv

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=ROOT/'candidate.json');p.add_argument('--tag',default='suzuka_c03_adapter_development');p.add_argument('--seconds',type=float,default=200.);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
 spec=json.loads(a.checkpoint.read_text());hashes=actor_source_hashes(spec)
 assert hashes==spec['actor_source_sha256'],(hashes,spec['actor_source_sha256'])
 out=ROOT/'output/racing/isaac';out.mkdir(parents=True,exist_ok=True);env=None
 identity={'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'actor_source_sha256':hashes,'runtime_source_sha256':RUNTIME_SOURCE_SHA256,'actor_configuration':configuration(spec),'experimental':True,'sensor_only':True,'workspace':str(ROOT),'tag':a.tag,'seed':a.seed}
 try:
  env=AdapterRaceIsaacEnv(Track('suzuka'),num_cars=2,render=False)
  ticks=[0];step=env.step
  def observed_step(actions):
   result=step(actions);ticks[0]+=1
   if ticks[0]%600==0:print('ADAPTER_ACTOR_PROGRESS',ticks[0]/60,flush=True)
   return result
  env.step=observed_step
  result,rows=rollout(env,spec['parameters'],seed=a.seed,seconds=a.seconds,actor_spec=spec)
  result.update(identity);result['physics_metadata']=env.adapter_qualification()
  result['qualification_valid_lap']=bool(result['valid_lap'] and result['physics_metadata']['qualified'])
  save(out/f'{a.tag}_seed{a.seed}_trace.json',rows)
  save(out/f'{a.tag}_seed{a.seed}.json',result)
  print('ADAPTER_ACTOR_RESULT',json.dumps({k:v for k,v in result.items() if k not in ['initial_states','physics_metadata']}),flush=True)
 except Exception:
  save(out/f'{a.tag}_seed{a.seed}_error.json',dict(identity,error=traceback.format_exc()));raise
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
