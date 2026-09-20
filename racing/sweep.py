"""Evaluate a frozen checkpoint on all supported circuit assets in MuJoCo.

Isaac is evaluated with separate native processes via scripts/run_racing.py.
Every attempted circuit receives a row; unsupported topology is explicit.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .run import ROOT,OUT,rollout,save,Video,POLICY_SOURCE_SHA,RUNTIME_SOURCE_SHA256
from .policy_bundle import actor_version,actor_source_hashes,configuration
from .tracks import Track,catalog
from .policy import RacingPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--seconds',type=float,default=150)
    p.add_argument('--tag',default='sweep');p.add_argument('--tracks',nargs='*');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--record-tracks',nargs='*',default=[],
                   help='Track ids to record as native video; failures are kept as control evidence')
    args=p.parse_args()
    from .mujoco_env import RaceMujocoEnv
    checkpoint=Path(args.checkpoint).resolve();data=json.loads(checkpoint.read_text());params=np.array(data['parameters'])
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();summary=[]
    version=actor_version(data);actor_hashes=actor_source_hashes(data)
    track_ids=args.tracks or sorted(p.parent.name for p in (ROOT/'tracks').glob('*/track.json'))
    for tid in track_ids:
        track=Track(tid)
        if track.metadata.get('scored_racing_supported') is False:
            summary.append({'track':tid,'engine':'mujoco','status':'unsupported_grade_separation','valid_lap':False,'lap_time_s':None})
            save(OUT/f'{args.tag}_summary.json',summary);continue
        env=None;started=time.monotonic()
        try:
            env=RaceMujocoEnv(track)
            video=Video(OUT/'mujoco'/f'{tid}_{args.tag}_seed{args.seed}.mp4','mujoco',track) if tid in args.record_tracks else None
            result,rows=rollout(env,params,args.seed,args.seconds,video=video,actor_spec=data if 'actor_type' in data else None)
            if video:result['video']=video.close()
            result.update(engine='mujoco',parameters=params.tolist(),checkpoint_path=str(checkpoint),checkpoint_sha256=sha,
                parameters_sha256=hashlib.sha256(json.dumps(params.tolist(),separators=(',',':')).encode()).hexdigest(),
                policy_version=version,policy_source_sha256=POLICY_SOURCE_SHA,
                actor_source_sha256=actor_hashes,runtime_source_sha256=RUNTIME_SOURCE_SHA256,
                checkpoint_policy_version=data.get('policy_version'),parameter_only_migration=data.get('policy_version')!=version,
                track_sha256=hashlib.sha256(track.path.read_bytes()).hexdigest())
            result.update(configuration(data))
            save(OUT/'mujoco'/f'{tid}_{args.tag}_seed{args.seed}_trace.json',rows)
            save(OUT/'mujoco'/f'{tid}_{args.tag}.json',[result])
            summary.append({k:result.get(k) for k in ['track','engine','valid_lap','lap_time_s','failure','progress_m','lap_fraction',
                'peak_speed_m_s','mean_speed_m_s','effective_overtakes','drift_duration_s','max_rear_slip_deg','checkpoint_sha256']})
        except Exception as exc:
            summary.append({'track':tid,'engine':'mujoco','status':'error','error':str(exc),'valid_lap':False,'lap_time_s':None})
        finally:
            if env:env.close()
        print('SWEEP',json.dumps(summary[-1]),'wall',round(time.monotonic()-started,1),flush=True)
        save(OUT/f'{args.tag}_summary.json',summary)


if __name__=='__main__':main()
