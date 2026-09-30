"""Evaluate a frozen checkpoint on all supported circuit assets in MuJoCo.

Isaac is evaluated with separate native processes via tools/runtime/run_racing.py.
Every attempted circuit receives a row; unsupported topology is explicit.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .run import ROOT,rollout,save,Video,POLICY_SOURCE_SHA,RUNTIME_SOURCE_SHA256
from .artifacts import RunLayout
from ..control.policy_bundle import actor_version,actor_source_hashes,configuration
from ..tracks import Track,catalog
from ..control.policy import RacingPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--seconds',type=float,default=150)
    p.add_argument('--tag',default='sweep');p.add_argument('--tracks',nargs='*');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--record-tracks',nargs='*',default=[],
                   help='Track ids to record as native video; failures are kept as control evidence')
    p.add_argument('--run-id',help='Run directory name; defaults to --tag')
    p.add_argument('--runs-root',help='Override the default runs/ artifact root')
    args=p.parse_args()
    try:layout=RunLayout.open(args.run_id or args.tag,args.runs_root)
    except ValueError as exc:p.error(str(exc))
    from ..simulators import create_simulator
    checkpoint=Path(args.checkpoint).resolve();data=json.loads(checkpoint.read_text());params=np.array(data['parameters'])
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();summary=[]
    version=actor_version(data);actor_hashes=actor_source_hashes(data)
    track_ids=args.tracks or sorted(p.parent.name for p in (ROOT/'assets/tracks').glob('*/track.json'))
    for tid in track_ids:
        track=Track(tid)
        if track.metadata.get('scored_racing_supported') is False:
            summary.append({'track':tid,'engine':'mujoco','status':'unsupported_grade_separation','valid_lap':False,'lap_time_s':None})
            save(layout.metrics/'summary.json',summary);continue
        env=None;started=time.monotonic()
        try:
            env=create_simulator('mujoco',track)
            video=Video(layout.video_dir('mujoco',tid)/f'seed{args.seed}.mp4','mujoco',track) if tid in args.record_tracks else None
            result,rows=rollout(env,params,args.seed,args.seconds,video=video,actor_spec=data if 'actor_type' in data else None)
            if video:result['video']=video.close()
            result.update(engine='mujoco',parameters=params.tolist(),checkpoint_path=str(checkpoint),checkpoint_sha256=sha,
                parameters_sha256=hashlib.sha256(json.dumps(params.tolist(),separators=(',',':')).encode()).hexdigest(),
                policy_version=version,policy_source_sha256=POLICY_SOURCE_SHA,
                actor_source_sha256=actor_hashes,runtime_source_sha256=RUNTIME_SOURCE_SHA256,
                checkpoint_policy_version=data.get('policy_version'),parameter_only_migration=data.get('policy_version')!=version,
                track_sha256=hashlib.sha256(track.path.read_bytes()).hexdigest())
            result.update(configuration(data))
            save(layout.trajectory_dir('mujoco',tid)/f'seed{args.seed}.json',rows)
            save(layout.evaluation_path('mujoco',tid),[result])
            summary.append({k:result.get(k) for k in ['track','engine','valid_lap','lap_time_s','failure','progress_m','lap_fraction',
                'peak_speed_m_s','mean_speed_m_s','effective_overtakes','drift_duration_s','max_rear_slip_deg','checkpoint_sha256']})
        except Exception as exc:
            summary.append({'track':tid,'engine':'mujoco','status':'error','error':str(exc),'valid_lap':False,'lap_time_s':None})
        finally:
            if env:env.close()
        print('SWEEP',json.dumps(summary[-1]),'wall',round(time.monotonic()-started,1),flush=True)
        save(layout.metrics/'summary.json',summary)


if __name__=='__main__':main()
