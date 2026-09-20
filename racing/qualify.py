"""Resumable native two-engine qualification of one immutable checkpoint.

Every child saves its full trace. A successful process exit is not evidence of
a successful lap; independent verification determines evidence and task status.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def results_match(rows,seeds,identities,conditions,record=False):
    """Never reuse a stale result merely because its child exited successfully."""
    return isinstance(rows,list) and [r.get('seed') for r in rows]==seeds and all(
        all(r.get(key)==value for key,value in identities.items()) and
        r.get('evaluation_conditions')==dict(conditions,recorded=record and i==0)
        for i,r in enumerate(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--tag',default='qualification')
    parser.add_argument('--engines',nargs='+',choices=['isaac','mujoco'],default=['isaac','mujoco'])
    parser.add_argument('--tracks',nargs='+')
    parser.add_argument('--seconds',type=float,default=200)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--seeds',type=int,nargs='+',help='Explicit evaluation seed list, including held-out seeds')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--record-tracks',nargs='*',default=['bahrain'])
    parser.add_argument('--lidar-noise',type=float,default=0.)
    parser.add_argument('--lidar-dropout',type=float,default=0.)
    parser.add_argument('--lidar-latency',type=float,default=0.)
    parser.add_argument('--opponent-speed',type=float,default=2.8)
    parser.add_argument('--opponent-gap',type=float,default=3.)
    parser.add_argument('--start-s',type=float,default=0.)
    args=parser.parse_args()
    if args.seeds:args.episodes=len(args.seeds)
    if args.episodes<1:parser.error('--episodes must be positive')
    evaluation_seeds=args.seeds or [0]+[73+i for i in range(1,args.episodes)]
    checkpoint=Path(args.checkpoint).resolve()
    from .policy_bundle import actor_version,actor_source_hashes
    ck=json.loads(checkpoint.read_text())
    source=ROOT/'racing/policy.py'
    if ck.get('policy_version')!=actor_version(ck) or ck.get('policy_source_sha256')!=digest(source):
        parser.error('Checkpoint must match the exact current actor source and version. Retrain or explicitly create a new evaluated baseline.')
    if ck.get('actor_type','reactive')!='reactive' or ck.get('safety_supervisor'):
        if ck.get('actor_source_sha256')!=actor_source_hashes(ck):
            parser.error('Checkpoint bundle module hashes do not match the current controller.')
    checkpoint_sha,source_sha=digest(checkpoint),digest(source)
    runtime_sha={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'racing').glob('*.py') if p.stem in
                 {'run','policy','policy_bundle','policy_closed_loop','safety','safety_motion','sensors','odometry','metrics','tracks','isaac_env','mujoco_env'}}
    def unchanged():
        return all(digest(ROOT/p)==sha for p,sha in runtime_sha.items()) and digest(checkpoint)==checkpoint_sha
    track_ids=args.tracks or sorted(p.parent.name for p in (ROOT/'tracks').glob('*/track.json'))
    output=ROOT/'output/racing';output.mkdir(parents=True,exist_ok=True)
    logdir=output/'logs'/args.tag;logdir.mkdir(parents=True,exist_ok=True)
    manifest={'schema_version':1,'checkpoint_path':str(checkpoint),'checkpoint_sha256':checkpoint_sha,
              'policy_source_sha256':source_sha,'runtime_source_sha256':runtime_sha,
              'tag':args.tag,'episodes':args.episodes,'seeds':evaluation_seeds,'results':[]}

    def evaluate(engine,tid):
        track=ROOT/'tracks'/tid/'track.json';track_sha=digest(track)
        result_path=output/engine/f'{tid}_{args.tag}.json'
        base={'engine':engine,'track':tid,'checkpoint_sha256':checkpoint_sha,'track_sha256':track_sha}
        data=json.loads(track.read_text())
        if data.get('scored_racing_supported') is False:
            return dict(base,status='unsupported_geometry',valid_lap=False)
        if not unchanged():
            return dict(base,status='frozen_inputs_changed',valid_lap=False)
        reusable=False
        conditions={'seconds':args.seconds,'start_s':args.start_s,'opponent_speed':args.opponent_speed,'opponent_gap':args.opponent_gap,
                    'lidar_noise':args.lidar_noise,'lidar_dropout':args.lidar_dropout,'lidar_latency':args.lidar_latency}
        def matches(rows):
            return results_match(rows,evaluation_seeds,{
                'checkpoint_sha256':checkpoint_sha,'track_sha256':track_sha,
                'policy_source_sha256':source_sha,'runtime_source_sha256':runtime_sha},
                conditions,record=tid in args.record_tracks)
        if result_path.exists():
            rows=json.loads(result_path.read_text())
            reusable=matches(rows)
        started=time.monotonic()
        if not reusable:
            command=([str(ROOT/'scripts/run_isaac.sh')] if engine=='isaac' else [sys.executable])
            if engine=='isaac':command.insert(0,'bash')
            command += [str(ROOT/'scripts/run_racing.py'),'--engine',engine,'--track',tid,'--checkpoint',str(checkpoint),
                        '--tag',args.tag,'--seconds',str(args.seconds),'--episodes',str(args.episodes)]
            command += ['--lidar-noise',str(args.lidar_noise),'--lidar-dropout',str(args.lidar_dropout),
                        '--lidar-latency',str(args.lidar_latency)]
            command += ['--opponent-speed',str(args.opponent_speed),'--opponent-gap',str(args.opponent_gap),
                        '--start-s',str(args.start_s)]
            if tid in args.record_tracks:command+=['--record']
            if args.seeds:command+=['--seeds',*[str(seed) for seed in args.seeds]]
            try:
                with (logdir/f'{engine}_{tid}.log').open('w') as log:
                    proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=max(600,args.seconds*args.episodes*20))
                if proc.returncode or not result_path.exists():
                    return dict(base,status='native_execution_failed',valid_lap=False,exit_code=proc.returncode)
            except subprocess.TimeoutExpired:
                return dict(base,status='native_timeout',valid_lap=False)
        if not unchanged() or digest(track)!=track_sha:
            return dict(base,status='frozen_inputs_changed',valid_lap=False)
        if not matches(json.loads(result_path.read_text())):
            return dict(base,status='result_inputs_mismatch',valid_lap=False)
        with (logdir/f'{engine}_{tid}_verify.log').open('w') as log:
            audit=subprocess.run([sys.executable,'-m','racing.verify',str(result_path),'--checkpoint',str(checkpoint)],
                                 cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        rows=json.loads(result_path.read_text())
        return dict(base,status='evaluated',reused=reusable,wall_seconds=time.monotonic()-started,
                    audit_exit_code=audit.returncode,valid_lap=all(r['valid_lap'] for r in rows),
                    episodes=[{k:r.get(k) for k in ['seed','valid_lap','lap_time_s','failure','peak_speed_m_s',
                        'mean_speed_m_s','effective_overtakes','drift_duration_s','max_continuous_drift_duration_s',
                        'drift_event_count','continuous_drift_qualified']} for r in rows])

    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        jobs=[pool.submit(evaluate,e,t) for e in args.engines for t in track_ids]
        for job in as_completed(jobs):
            row=job.result();manifest['results'].append(row)
            (output/f'{args.tag}_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
            print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
