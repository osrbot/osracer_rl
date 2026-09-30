"""CEM search evaluated on both native engines and multiple development seeds.

Launch with Isaac's Python. Only the evaluator sees privileged metrics. Search
uses a bounded curriculum segment; final full-lap qualification is separate.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np

from . import run
from ..control.policy_bundle import actor_version,actor_source_hashes,configuration
from ..paths import source_files
from ..tracks import Track
from .artifacts import RunLayout
from .configuration import apply_joint_overrides, task_track
from .progress import TrainingProgress
from .tracking import MetricLogger

NAMES=['cruise_speed','rear_overdrive','drift_trigger','max_boost_seconds',
       'boost_countersteer_gain','yaw_rate_guard']
LOWER=np.array([6.5,1.5,.27,.4,.10,3.8])
UPPER=np.array([9.,3.5,.36,1.,.25,5.5])
SPREAD=np.array([.25,.3,.015,.18,.035,.35])


def vector(checkpoint):
    p=checkpoint['parameters'];c=checkpoint.get('controls',{})
    return np.array([p[0],p[10],p[11],c.get('max_boost_seconds',.6),
                     c.get('boost_countersteer_gain',.2),c.get('yaw_rate_guard',3.8)])


def candidate(prior,values):
    result=json.loads(json.dumps(prior))
    result['parameters'][0],result['parameters'][10],result['parameters'][11]=map(float,values[:3])
    result.setdefault('controls',{}).update(dict(zip(NAMES[3:],map(float,values[3:]))))
    return result


def aggregate(episodes,expected):
    """Safety dominates; worst-seed drift cannot hide behind a good mean."""
    fields={'failure','collision_steps','offroad_steps','duration_s','progress_m',
            'max_continuous_drift_duration_s','effective_overtakes','lap_fraction'}
    complete=[fields.issubset(r) and all(isinstance(r[k],(float,int)) and np.isfinite(r[k])
              for k in fields-{'failure'}) for r in episodes]
    clean=[ok and r['failure'] in (None,'timeout') and not r['collision_steps']
           and not r['offroad_steps'] for ok,r in zip(complete,episodes)]
    drift=[float(r.get('max_continuous_drift_duration_s',0.)) for r in episodes]
    passed=[r.get('effective_overtakes',0)>=1 for r in episodes]
    all_clean=len(episodes)==expected and all(clean)
    forward=[ok and r['duration_s']>0 and r['progress_m']/r['duration_s']>=2.5
             for ok,r in zip(complete,episodes)]
    all_forward=len(episodes)==expected and all(forward)
    all_passed=len(episodes)==expected and all(passed)
    if not all_clean:
        score=-10000.-1000*sum(not value for value in clean)
    elif not all_forward:
        score=-1000.+100.*np.mean(forward)
    elif not all_passed:
        score=100.*np.mean(passed)
    else:
        fractions=[float(r.get('lap_fraction',0.)) for r in episodes]
        qualified=[duration+1e-10>=.1 for duration in drift]
        score=(1000.+200.*all(qualified)+100.*min(min(drift)/.25,1.)
               +30.*np.mean(qualified)+20.*np.mean(passed)
               +10.*min(fractions)+np.mean(fractions))
    return {'score':float(score),'all_segments_clean':bool(all_clean),
            'episodes_evaluated':len(episodes),'episodes_expected':expected,
            'all_forward_progress':bool(all_forward),'minimum_progress_speed_m_s':2.5,
            'all_overtakes':bool(all_passed),
            'all_continuous_drift':bool(len(episodes)==expected and all(d+1e-10>=.1 for d in drift)),
            'minimum_continuous_drift_s':min(drift,default=0.)}


def load_mujoco():
    # Preload before SimulationApp: its Newton extension bundles MuJoCo 3.8,
    # which must not silently replace the project's target-engine version.
    try:module=importlib.import_module('mujoco')
    except ModuleNotFoundError:
        packages=run.ROOT/'.venv'/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        if not (packages/'mujoco').is_dir():
            raise RuntimeError('Joint training needs a project venv matching Isaac Python; run setup_racing.sh with Python 3.12.')
        sys.path.append(str(packages))
        module=importlib.import_module('mujoco')
    if module.__version__!='3.10.0':
        raise RuntimeError(f'Joint training requires MuJoCo 3.10.0, found {module.__version__}; preload the project version before SimulationApp')
    return module


def audit_trial(item,values,seeds,track,contract_sha):
    from ..evaluation.verify import audit_trace
    if item.get('contract_sha256')!=contract_sha or not np.array_equal(item.get('variables'),values):
        raise RuntimeError('Saved trial does not match its immutable proposal and contract')
    episodes=item.get('episodes',[])
    planned=[(engine,seed) for engine in ['isaac','mujoco'] for seed in seeds]
    actual=[(row.get('engine'),row.get('seed')) for row in episodes]
    if not episodes or actual!=planned[:len(actual)] or len(actual)>len(planned):
        raise RuntimeError('Saved trial has missing, duplicate or reordered engine/seed identities')
    if len(actual)<len(planned) and episodes[-1].get('failure') in (None,'timeout'):
        raise RuntimeError('An incomplete successful trial cannot be reused')
    for row in episodes:
        trace=Path(row['trace_path'])
        if not trace.is_file() or hashlib.sha256(trace.read_bytes()).hexdigest()!=row['trace_sha256']:
            raise RuntimeError('Saved trial trace is missing or changed')
        audit=audit_trace(row,json.loads(trace.read_text()),track)
        if audit['errors']:raise RuntimeError(f'Trial trace audit failed: {audit["errors"]}')
    # Scores are derived again, never trusted from a mutable cached JSON field.
    return dict(item,**aggregate(episodes,len(planned)))


def main():
    parser=argparse.ArgumentParser(description=__doc__,epilog=(
        'ASAP-style example: osracer-train-joint checkpoint=policy.json '
        'experiment_name=joint-cem'))
    parser.add_argument('overrides',nargs='*',metavar='KEY=VALUE')
    parser.add_argument('--checkpoint')
    parser.add_argument('--experiment-name','--run-id',dest='run_id')
    parser.add_argument('--task','--track',dest='track',type=task_track,default='bahrain')
    parser.add_argument('--tag',default='joint_cem',help=argparse.SUPPRESS)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,4,5],help=argparse.SUPPRESS)
    parser.add_argument('--seconds',type=float,default=30.,help=argparse.SUPPRESS)
    parser.add_argument('--population',type=int,default=6,help=argparse.SUPPRESS)
    parser.add_argument('--generations',type=int,default=2,help=argparse.SUPPRESS)
    parser.add_argument('--seed',type=int,default=194,help=argparse.SUPPRESS)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--runs-root',help=argparse.SUPPRESS)
    parser.add_argument('--tensorboard-dir',help='Override this run\'s TensorBoard event directory')
    parser.add_argument('--no-tensorboard',action='store_true',help='Write JSONL metrics only')
    parser.add_argument('--json-console',action='store_true',
                        help='Print compact machine-readable candidate JSON instead of formatted progress')
    args=parser.parse_args()
    try:apply_joint_overrides(args,args.overrides)
    except (TypeError,ValueError) as exc:parser.error(str(exc))
    if not args.checkpoint:parser.error('provide checkpoint=<path> or --checkpoint <path>')
    if args.population<3 or args.generations<1 or args.seconds<=0:parser.error('Positive duration/generations and population >=3 are required')
    if len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:parser.error('Development seeds must be distinct and nonnegative')
    prior_path=Path(args.checkpoint).resolve();prior=json.loads(prior_path.read_text())
    if prior.get('actor_type')!='closed_loop':parser.error('A closed_loop checkpoint is required')
    if prior.get('actor_source_sha256')!=actor_source_hashes(prior):parser.error('Freeze the current actor source in the input checkpoint first')
    initial=vector(prior)
    if not np.isfinite(initial).all() or np.any(initial<LOWER) or np.any(initial>UPPER):
        parser.error('Initial checkpoint search variables must already be inside the declared effective bounds')
    try:layout=RunLayout.open(args.run_id or args.tag,args.runs_root)
    except ValueError as exc:parser.error(str(exc))
    target=layout.metrics/'training';target.mkdir(parents=True,exist_ok=True)
    tensorboard_path=Path(args.tensorboard_dir).expanduser().resolve() if args.tensorboard_dir else layout.tensorboard
    if not args.resume and not args.no_tensorboard and any(tensorboard_path.glob('events.out.tfevents.*')):
        parser.error(f'TensorBoard directory already contains events: {tensorboard_path}')
    run.save(layout.config/'joint_training.json',vars(args) | {'layout':str(layout.root)})
    sources=source_files()
    runtime={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()}
    for name,path in sources.items():
        destination=layout.source/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()!=runtime[name]:
            parser.error('Training tag belongs to different code; select a new tag')
        destination.write_bytes(path.read_bytes())
    track=Track(args.track)
    contract={'prior_sha256':hashlib.sha256(prior_path.read_bytes()).hexdigest(),'track':track.id,
        'track_sha256':hashlib.sha256(track.path.read_bytes()).hexdigest(),'seeds':args.seeds,
        'seconds':args.seconds,'population':args.population,'generations':args.generations,'seed':args.seed,
        'variables':NAMES,'lower':LOWER.tolist(),'upper':UPPER.tolist(),
        'minimum_progress_speed_m_s':2.5,'runtime_source_sha256':runtime}
    contract_path=target/'contract.json'
    if args.resume and not contract_path.exists() and list(target.glob('g*_c*.json')):
        parser.error('Cannot resume existing trials without their original contract')
    if args.resume and contract_path.exists() and json.loads(contract_path.read_text())!=contract:
        parser.error('Cannot resume changed training conditions')
    if contract_path.exists() and not args.resume:parser.error('Training tag exists; use --resume or a new tag')
    run.save(contract_path,contract)
    contract_sha=hashlib.sha256(contract_path.read_bytes()).hexdigest()
    mujoco=load_mujoco()
    from ..simulators import create_simulator
    source=create_simulator('isaac',track)
    destination=None
    tracker=MetricLogger(layout.metrics,args.tensorboard_dir,not args.no_tensorboard,
                         purge_step=0 if args.resume else None)
    try:
        destination=create_simulator('mujoco',track)
        run.save(target/'environment.json',{'python':sys.version,'numpy':np.__version__,
            'mujoco':mujoco.__version__,'mujoco_module':mujoco.__file__,'isaac':'6.0.1'})
        rng=np.random.default_rng(args.seed);mean=initial.copy();std=SPREAD.copy()
        best=None;history=[];expected=2*len(args.seeds)
        progress=TrainingProgress('isaac + mujoco',track.id,args.generations,args.population,
                                  run_id=layout.run_id)
        for generation in range(args.generations):
            proposals=np.clip(rng.normal(mean,std,(args.population,len(mean))),LOWER,UPPER)
            proposals[0]=mean
            if best is not None:proposals[1]=best['variables']
            trials=[]
            for index,values in enumerate(proposals):
                iteration_started=time.monotonic();reused=False
                trial_path=target/f'g{generation:02d}_c{index:02d}.json'
                if args.resume and trial_path.exists():
                    audit_started=time.monotonic();reused=True
                    item=audit_trial(json.loads(trial_path.read_text()),values,args.seeds,track,contract_sha)
                    collection_s=time.monotonic()-audit_started
                else:
                    spec=candidate(prior,values);episodes=[];started=time.monotonic()
                    for engine,env in [('isaac',source),('mujoco',destination)]:
                        for seed in args.seeds:
                            result,rows=run.rollout(env,np.array(spec['parameters']),seed=seed,seconds=args.seconds,actor_spec=spec)
                            trace=layout.trajectory_dir(engine,track.id)/f'g{generation:02d}_c{index:02d}_seed{seed}.json'
                            run.save(trace,rows)
                            result.update(engine=engine,trace_path=str(trace),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
                            episodes.append(result)
                            run.save(trial_path.with_suffix('.partial.json'),{'variables':values.tolist(),'episodes':episodes})
                            if result['failure'] not in (None,'timeout'):break
                        if episodes[-1]['failure'] not in (None,'timeout'):break
                    item={'generation':generation,'candidate':index,'variables':values.tolist(),'contract_sha256':contract_sha,
                          'episodes':episodes,'wall_seconds':time.monotonic()-started,**aggregate(episodes,expected)}
                    item=audit_trial(item,values,args.seeds,track,contract_sha)
                    run.save(trial_path,item)
                    collection_s=item['wall_seconds']
                trials.append(item);history.append(item)
                tracker.log(generation*args.population+index,'candidate',{k:item[k] for k in
                    ['score','all_segments_clean','all_forward_progress','all_overtakes','all_continuous_drift','minimum_continuous_drift_s']})
                if best is None or item['score']>best['score']:
                    best=item
                    spec=candidate(prior,item['variables'])
                    spec.update(schema_version=1,algorithm='CEM structured sensor-feedback policy, joint native-engine curriculum',
                        trained_in=list(dict.fromkeys(row['engine'] for trial in history for row in trial['episodes'])),
                        planned_training_engines=['isaac','mujoco'],training_tracks=[track.id],training_seeds=args.seeds,
                        training_seconds=args.seconds,qualification_status='Training candidate; requires independent full-lap qualification',
                        policy_version=actor_version(spec),actor_source_sha256=actor_source_hashes(spec),
                        runtime_source_sha256=run.RUNTIME_SOURCE_SHA256,training_contract=str(contract_path),
                        training_best={k:item[k] for k in ['generation','candidate','score','all_segments_clean','all_overtakes','all_continuous_drift','minimum_continuous_drift_s']})
                    run.save(layout.checkpoints/'policy.json',spec)
                run.save(target/'history.json',history)
                if args.json_console:
                    print('JOINT_CEM',generation,index,json.dumps({k:item[k] for k in ['score','all_segments_clean','all_overtakes','all_continuous_drift','minimum_continuous_drift_s']}),flush=True)
                else:
                    progress.joint_candidate(generation,index,item,collection_s,
                        time.monotonic()-iteration_started,best['score'],reused)
            elites=sorted(trials,key=lambda item:item['score'],reverse=True)[:max(2,args.population//3)]
            values=np.array([item['variables'] for item in elites])
            scores=np.array([item['score'] for item in trials],float)
            tracker.log((generation+1)*args.population-1,'generation',{'best_score':float(scores.max()),'mean_score':float(scores.mean()),
                'score_std':float(scores.std()),'clean_rate':float(np.mean([item['all_segments_clean'] for item in trials])),
                'overtake_rate':float(np.mean([item['all_overtakes'] for item in trials]))})
            mean=.2*mean+.8*values.mean(axis=0)
            std=np.maximum(.025*(UPPER-LOWER),.2*std+.8*values.std(axis=0))
            if not args.json_console:
                progress.generation(generation,scores,
                    float(np.mean([item['all_segments_clean'] for item in trials])),
                    'Clean segment rate')
    finally:
        tracker.close()
        if destination is not None:destination.close()
        source.close()
    print('OSRACER_ARTIFACTS',json.dumps({'run_id':layout.run_id,'run_dir':str(layout.root),
        'checkpoint':str(layout.checkpoints/'policy.json'),
        'metrics_jsonl':str(layout.metrics/'metrics.jsonl'),
        'tensorboard':str(tracker.tensorboard_dir) if tracker.writer else None},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
