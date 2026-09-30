"""Native sensor-only racing, parameter training, evaluation and recording.

Privileged states stay in the environment, ray generator and evaluator. The
actor receives copies of encoders and the held 15 Hz scan only.
"""
import argparse
from datetime import datetime
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from ..tracks import Track
from ..control.policy import RacingPolicy
from ..perception.sensors import LidarSensor
from ..control.policy_bundle import make_actor,core_policy_class,configuration,actor_version,actor_source_hashes
from ..simulators import available_simulators, create_simulator
from ..paths import REPO_ROOT, RUNS_ROOT, source_files
from .artifacts import RunLayout
from .configuration import (apply_run_overrides, load_training_profile,
                            selected_training_profile, task_track)
from .progress import TrainingProgress, print_evaluation
from .tracking import MetricLogger

ROOT=REPO_ROOT
OUT=RUNS_ROOT
DT=1/60
POLICY_SOURCE_SHA=hashlib.sha256((ROOT/'src/racing/control/policy.py').read_bytes()).hexdigest()
RUNTIME_SOURCE_SHA256={name:hashlib.sha256(path.read_bytes()).hexdigest()
                      for name,path in source_files().items()}


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(serial(value),ensure_ascii=False,indent=2,allow_nan=False))


def diagnostic_values(value):
    """Represent unavailable diagnostic bounds as null, never mask state errors."""
    value=serial(value)
    if isinstance(value,dict):return {k:diagnostic_values(v) for k,v in value.items()}
    if isinstance(value,list):return [diagnostic_values(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


class Video:
    def __init__(self,path,engine,track):
        from PIL import ImageFont
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.engine,self.track=engine,track
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
        self.frames=0
        self.proc=subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo',
            '-pix_fmt','rgb24','-s','1280x720','-r','30','-i','pipe:0','-an','-c:v','libx264',
            '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)

    def write(self,env,row,idx):
        if idx%2:return
        from PIL import Image,ImageDraw
        im=Image.fromarray(env.render());d=ImageDraw.Draw(im)
        d.rectangle((0,0,1280,76),fill=(15,22,34))
        d.text((18,10),f'OSRACER | {self.engine} | {self.track.name}',font=self.font,fill='white')
        d.text((18,42),'1x playback | wheel/steering encoders + 15 Hz laser | no global actor pose',font=self.font,fill='#79dce8')
        d.rectangle((0,630,1280,720),fill=(15,22,34))
        s=row['states'][0];v=np.hypot(s['vx'],s['vy'])
        d.text((18,640),f"t {row['t']:.2f}s   Body {v:.2f} m/s   Rear slip {np.degrees(s['rear_slip_beta']):+.1f} deg   {row['phase']}",font=self.font,fill='white')
        d.text((18,678),f"Progress {row['progress']:.1f}/{self.track.length:.1f}m   Overtakes {row['overtakes']}   Contacts {row['collision_steps']}",font=self.font,fill='#79dce8')
        self.proc.stdin.write(np.asarray(im).tobytes());self.frames+=1
        if self.frames==60:im.save(self.path.with_suffix('.png'))

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait()!=0:raise RuntimeError('Video encoding failed')
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(self.path),'-f','null','-'],check=True)
        try:path=str(self.path.relative_to(ROOT))
        except ValueError:path=str(self.path)
        return {'path':path,'frames':self.frames,'fps':30,
                'duration_s':self.frames/30,'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),'decoded':True}


def rollout(env,parameters=None,seed=0,seconds=100,video=None,start_s=0.,opponent_speed=2.8,opponent_gap=3.,sensor_options=None,actor_spec=None,observation_profile='sim',actor=None):
    from ..evaluation.metrics import RaceMetrics
    track=env.track
    states=env.reset(seed,starts=[(start_s,0.),(start_s+opponent_gap,0.)])
    initial_states=serial(states)
    from ..vehicle.real import make_sensor
    sensors=[make_sensor(LidarSensor(track,seed=seed+j,**(sensor_options or {})),observation_profile) for j in range(len(states))]
    policy = actor if actor is not None else (
        make_actor(parameters,actor_spec) if actor_spec is not None else RacingPolicy(parameters))
    if actor is not None and hasattr(policy,'reset'):policy.reset()
    opponent=RacingPolicy()  # Fixed opponent: search must not change task difficulty.
    opponent.p[0]=opponent_speed
    policies=[policy,opponent]
    metrics=RaceMetrics(track,dt=DT)
    metrics.update(states[0],opponents=[states[1]])
    metrics.time=0.  # Seed position at reset; no simulated time elapsed yet.
    def location(s):
        return np.array([s['x'],s['y'],s['z']]) if getattr(track,'has_elevation',False) else np.array([s['x'],s['y']])
    progress=0.;last_s=track.project(location(states[0]))[0]
    speeds=[];slips=[];ctes=[];rows=[];collision_steps=0;offroad_steps=0;failure=None
    overtake=0;behind=True;ahead_time=0.;last_motion=0.;opponent_progress=opponent_gap;opp_s=track.project(location(states[1]))[0]
    for i in range(round(seconds/DT)):
        actions=[];observations=[]
        for j,(sensor,actor) in enumerate(zip(sensors,policies)):
            obs=sensor.observe(states[j],states[j]['lidar_pose'],opponents=[states[1-j]],tick=i)
            observations.append(obs);actions.append(actor.action(obs))
            if hasattr(sensor,'record_action'):
                # The vehicle node knows only what it last commanded; the servo
                # model advances on that, exactly like the real open loop.
                sensor.record_action(actions[-1])
        states=env.step(actions)
        state=states[0];speed=float(np.hypot(state['vx'],state['vy']))
        s,cte,_=track.project(location(state))
        ds=(s-last_s+track.length/2)%track.length-track.length/2;last_s=s
        jump=abs(ds)>max(.6,2*speed*DT)
        if jump:ds=0.
        progress+=ds
        os,_,_=track.project(location(states[1]))
        opponent_progress+=(os-opp_s+track.length/2)%track.length-track.length/2;opp_s=os
        collision=bool(state['collision']);collision_steps+=int(collision)
        offroad=abs(cte)>track.width/2-.125;offroad_steps+=int(offroad)
        metrics.update(state,opponents=[states[1]],collision=collision,offroad=offroad)
        # Episode-level measured passing statistic is independently recomputed
        # by RaceMetrics below; this overlay uses the same persistence concept.
        relative=progress-opponent_progress
        if relative<-.5:behind=True;ahead_time=0.
        elif relative>.5 and behind and not(collision or offroad):
            ahead_time+=DT
            if ahead_time>=.3:overtake+=1;behind=False;ahead_time=0.
        elif collision or offroad:ahead_time=0.
        speeds.append(speed);slips.append(float(state['rear_slip_beta']));ctes.append(float(cte))
        row={'t':(i+1)*DT,'states':states,'progress':progress,'cte':cte,'phase':policy.phase,
             'overtakes':overtake,'collision_steps':collision_steps,'actions':actions,
             'scan_timestamp':observations[0]['lidar']['timestamp'],
             'scan_age':observations[0]['lidar']['age'] if np.isfinite(observations[0]['lidar']['age']) else None}
        if hasattr(policy,'feedback'):row['actor_feedback']=diagnostic_values(policy.feedback)
        rows.append(serial(row))
        if video:video.write(env,row,i)
        if jump:failure='projection_jump';break
        if speed>.25:last_motion=(i+1)*DT
        if abs(state['roll'])>.5 or abs(state['pitch'])>.5 or state['z']<.015:failure='unstable';break
        if collision:failure='collision';break
        if offroad:failure='offroad';break
        if (i+1)*DT-last_motion>4:failure='stalled';break
        if progress>=track.length:break
    duration=(i+1)*DT
    complete=progress>=track.length and failure is None
    if not complete and failure is None:failure='timeout'
    summary=metrics.summary()
    valid=complete and collision_steps==0 and offroad_steps==0
    speedarr=np.array(speeds);sliparr=np.array(slips)
    from ..evaluation.metrics import drift_statistics
    drift=drift_statistics(speedarr,sliparr,dt=DT)
    drift_time=drift['drift_duration_s']
    result={'seed':seed,'track':track.id,'completed':complete,'valid_lap':valid,
        'initial_states':initial_states,
        'lap_time_s':duration if valid else None,'duration_s':duration,'progress_m':progress,
        'lap_fraction':progress/track.length,'peak_speed_m_s':max(speeds,default=0.),
        'mean_speed_m_s':float(np.mean(speeds)) if speeds else 0.,
        'max_rear_slip_deg':float(np.degrees(np.max(np.abs(sliparr[speedarr>1.2])))) if np.any(speedarr>1.2) else 0.,
        'slip_speed_gate_m_s':1.2,'drift_angle_threshold_deg':20.0,
        'drift_duration_s':drift_time,'overtakes_overlay':overtake,'metric_audit':summary,
        'collision_steps':collision_steps,'offroad_steps':offroad_steps,'failure':failure,
        'max_abs_cte_m':max(np.abs(ctes),default=0.),'start_s':start_s,
        'opponent_speed_cap_m_s':opponent_speed,'sensor':sensors[0].specs}
    result.update(drift)
    result['opponent_parameters']=opponent.p.tolist();result['opponent_gap_m']=opponent_gap
    result['effective_overtakes']=summary['effective_overtakes']
    result['score']=progress+100*valid+5*summary['effective_overtakes']-.6*duration-60*bool(failure and failure!='timeout')
    result['score']+=valid*drift['continuous_drift_qualified']*(30*min(drift['max_continuous_drift_duration_s']/.2,1)+5*min(result['max_rear_slip_deg']/30,1))
    return result,rows


def train(env,args):
    training_dir=args.layout.metrics/'training'
    training_dir.mkdir(parents=True,exist_ok=True)
    args.trained_checkpoint=args.layout.checkpoints/'policy.json'
    rng=np.random.default_rng(args.seed)
    Core=core_policy_class(args.actor_spec)
    mean=Core.initial.copy();std=Core.spread.copy();best=None;history=[]
    if args.checkpoint and Path(args.checkpoint).exists():mean=np.array(json.loads(Path(args.checkpoint).read_text())['parameters'])
    if args.drift_curriculum:
        std[:10]=0.;std[10]=.4;std[11]=.025
        mean[-2]=.7;mean[-1]=.34
    progress=TrainingProgress(args.engine,env.track.id,args.generations,args.population,
                              run_id=args.layout.run_id)
    with MetricLogger(args.layout.metrics,args.tensorboard_dir,not args.no_tensorboard) as tracker:
        args.tensorboard_active=tracker.writer is not None
        args.tensorboard_path=tracker.tensorboard_dir
        for generation in range(args.generations):
            candidates=np.clip(rng.normal(mean,std,(args.population,len(mean))),Core.lower,Core.upper)
            candidates[0]=mean
            if args.drift_curriculum and generation==0:
                candidates[:,10]=np.linspace(0,1.75,args.population)
            if best:candidates[1]=best['parameters']
            trials=[]
            for j,p in enumerate(candidates):
                iteration_started=time.monotonic()
                collection_started=time.monotonic()
                result,rows=rollout(env,p,seed=0,seconds=args.seconds,opponent_speed=args.opponent_speed,
                                    start_s=args.start_s,opponent_gap=args.opponent_gap,actor_spec=args.actor_spec)
                collection_s=time.monotonic()-collection_started
                if args.drift_curriculum and args.seconds<=20 and result['failure']=='timeout' and result['progress_m']>=10:
                    longest=result['max_continuous_drift_duration_s']
                    result['score']+=50*min(longest/.2,1)*result['continuous_drift_qualified']
                    result['curriculum_segment_valid']=True
                item={'generation':generation,'candidate':j,'parameters':p.tolist(),'metrics':result,'score':result['score']}
                history.append(item);trials.append(item)
                tracker.log(generation*args.population+j,'candidate',{'score':result['score'],**result})
                if best is None or item['score']>best['score']:
                    best=item
                    save(args.trained_checkpoint,{'schema_version':1,'algorithm':'CEM sensor-only structured feedback',
                        'trained_in':'Isaac Sim 6.0.1' if args.engine=='isaac' else 'MuJoCo',
                        'parameters':p.tolist(),'parameter_names':Core.names,'training_tracks':[env.track.id],
                        **configuration(args.actor_spec),
                        'policy_version':actor_version(args.actor_spec),
                        'actor_source_sha256':args.frozen_actor_hashes,
                        'policy_source_sha256':POLICY_SOURCE_SHA,
                        'runtime_source_sha256':RUNTIME_SOURCE_SHA256,
                        'actor_observation':['wheel_vel','steer_pos','lidar'],'frame_id':'laser','scan_hz':15,'control_hz':60,
                        'metrics':result,'source_prior':'Ackermann wheel control and bounded rear overdrive adapted from hairpin; no privileged teacher observations enter actor'})
                    save(args.layout.trajectory_dir('training',env.track.id)/'best_trace.json',rows)
                save(training_dir/'history.json',history)
                if args.json_console:
                    print('RACE_TRAIN',generation,j,json.dumps(result),flush=True)
                else:
                    progress.candidate(generation,j,result,len(rows),collection_s,
                                       time.monotonic()-iteration_started,best['score'])
            scores=np.array([item['score'] for item in trials],float)
            tracker.log((generation+1)*args.population-1,'generation',{'best_score':float(scores.max()),'mean_score':float(scores.mean()),
                'score_std':float(scores.std()),'success_rate':float(np.mean([item['metrics']['valid_lap'] for item in trials]))})
            elites=sorted(trials,key=lambda x:x['score'],reverse=True)[:max(2,args.population//4)]
            ps=np.array([x['parameters'] for x in elites]);mean=.2*mean+.8*ps.mean(axis=0)
            std=np.maximum(.025*(Core.upper-Core.lower),.2*std+.8*ps.std(axis=0))
            if not args.json_console:
                progress.generation(generation,scores,
                    float(np.mean([item['metrics']['valid_lap'] for item in trials])))
    return np.array(best['parameters'])


def main(argv=None):
    raw_argv=list(sys.argv[1:] if argv is None else argv)
    override_tokens=[token for token in raw_argv if '=' in token and not token.startswith('--')]
    try:profile=load_training_profile(selected_training_profile(override_tokens))
    except (KeyError,TypeError,ValueError) as exc:
        raise SystemExit(f'configuration error: {exc}') from exc
    p=argparse.ArgumentParser(description=__doc__,epilog=(
        'ASAP-style example: osracer-train +simulator=mujoco '
        '+task=racing/bahrain experiment_name=demo'))
    p.add_argument('overrides',nargs='*',metavar='KEY=VALUE',
                   help='Composable settings such as +train=quick or train.population=16')
    common=p.add_argument_group('common experiment settings')
    common.add_argument('--simulator','--engine',dest='engine',choices=available_simulators())
    common.add_argument('--task','--track',dest='track',type=task_track,default=profile.track,
                        help='Task or track (for example racing/bahrain)')
    common.add_argument('--experiment-name','--run-id',dest='run_id',
                        help='Run name; training generates one when omitted')
    common.add_argument('--checkpoint')
    common.add_argument('--tensorboard-dir',help='Override this run\'s TensorBoard event directory')
    common.add_argument('--no-tensorboard',action='store_true',default=not profile.tensorboard,
                        help='Write JSONL metrics only')
    common.add_argument('--json-console',action='store_true',
                        help='Print machine-readable episode JSON instead of compact progress')
    p.add_argument('--train',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--record',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--generations',type=int,default=profile.generations,help=argparse.SUPPRESS)
    p.add_argument('--population',type=int,default=profile.population,help=argparse.SUPPRESS)
    p.add_argument('--seconds',type=float,default=profile.episode_length_s,help=argparse.SUPPRESS)
    p.add_argument('--episodes',type=int,default=profile.evaluation_episodes,help=argparse.SUPPRESS)
    p.add_argument('--seed',type=int,default=profile.seed,help=argparse.SUPPRESS)
    p.add_argument('--opponent-speed',type=float,default=profile.opponent_speed_m_s,help=argparse.SUPPRESS)
    p.add_argument('--seeds',type=int,nargs='+',help=argparse.SUPPRESS)
    p.add_argument('--start-s',type=float,default=0.,help=argparse.SUPPRESS)
    p.add_argument('--tag',default='evaluation',help=argparse.SUPPRESS)
    p.add_argument('--drift-curriculum',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--opponent-gap',type=float,default=profile.opponent_gap_m,help=argparse.SUPPRESS)
    p.add_argument('--lidar-noise',type=float,default=0.,help=argparse.SUPPRESS)
    p.add_argument('--lidar-dropout',type=float,default=0.,help=argparse.SUPPRESS)
    p.add_argument('--lidar-latency',type=float,default=0.,help=argparse.SUPPRESS)
    p.add_argument('--observation-profile',choices=['sim','real'],default='sim',
                   help=argparse.SUPPRESS)
    p.add_argument('--runs-root',help=argparse.SUPPRESS)
    args=p.parse_args(raw_argv)
    try:apply_run_overrides(args,args.overrides,available_simulators())
    except (TypeError,ValueError) as exc:p.error(str(exc))
    args.training_profile=profile.name
    if not args.engine:p.error('select a simulator with +simulator=mujoco or --simulator mujoco')
    if args.episodes<1 or args.generations<1 or args.population<1 or args.seconds<=0:
        p.error('training counts and episode length must be positive')
    run_id=args.run_id or (f'{datetime.now():%Y%m%d_%H%M%S}-{args.engine}-{args.track}' if args.train else args.tag)
    try:args.layout=RunLayout.open(run_id,args.runs_root)
    except ValueError as exc:p.error(str(exc))
    if args.train and (args.layout.metrics/'training/history.json').exists():
        p.error(f'run {args.layout.run_id!r} already contains training history; choose a new --run-id')
    tensorboard_path=Path(args.tensorboard_dir).expanduser().resolve() if args.tensorboard_dir else args.layout.tensorboard
    if args.train and not args.no_tensorboard and any(tensorboard_path.glob('events.out.tfevents.*')):
        p.error(f'TensorBoard directory already contains events: {tensorboard_path}')
    phase='training' if args.train else 'evaluation'
    save(args.layout.config/f'{phase}_{args.engine}_{args.track}.json',vars(args) | {'layout':str(args.layout.root)})
    checkpoint_data=None;checkpoint_sha=None;neural_checkpoint=False
    if args.checkpoint:
        checkpoint_bytes=Path(args.checkpoint).read_bytes()
        checkpoint_sha=hashlib.sha256(checkpoint_bytes).hexdigest()
        neural_checkpoint=Path(args.checkpoint).suffix=='.pt'
        if neural_checkpoint:
            from ..control.neural_policy import NeuralPolicy, load_checkpoint
            _,checkpoint_data=load_checkpoint(args.checkpoint)
            args.actor_factory=lambda:NeuralPolicy(args.checkpoint)
        else:checkpoint_data=json.loads(checkpoint_bytes)
    args.actor_spec=checkpoint_data if checkpoint_data and 'actor_type' in checkpoint_data else None
    frozen_actor_hashes=(checkpoint_data.get('metadata',{}).get('source_sha256',{})
                         if neural_checkpoint else actor_source_hashes(args.actor_spec))
    args.frozen_actor_hashes=frozen_actor_hashes
    track=Track(args.track)
    env=create_simulator(args.engine,track,render=args.record)
    try:
        params=RacingPolicy.initial.copy()
        if checkpoint_data and not neural_checkpoint:params=np.array(checkpoint_data['parameters'])
        if args.train:
            params=train(env,args)
            # Evaluation describes the newly trained checkpoint, not its prior.
            args.checkpoint=str(args.trained_checkpoint)
            checkpoint_bytes=Path(args.checkpoint).read_bytes()
            checkpoint_data=json.loads(checkpoint_bytes)
            checkpoint_sha=hashlib.sha256(checkpoint_bytes).hexdigest()
        results=[];recorded_video=None
        evaluation_seeds=args.seeds or [0]+[args.seed+i for i in range(1,args.episodes)]
        for i,seed in enumerate(evaluation_seeds):
            evaluation_started=time.monotonic()
            video=Video(args.layout.video_dir(args.engine,track.id)/f'seed{seed}.mp4',args.engine,track) if args.record and i==0 else None
            result,rows=rollout(env,params,seed=seed,seconds=args.seconds,video=video,start_s=args.start_s,
                                opponent_speed=args.opponent_speed,opponent_gap=args.opponent_gap,
                                sensor_options={'noise_std_m':args.lidar_noise,'dropout_prob':args.lidar_dropout,'latency_s':args.lidar_latency},
                                actor_spec=args.actor_spec,observation_profile=args.observation_profile,
                                actor=args.actor_factory() if neural_checkpoint else None)
            if video:
                result['video']=video.close();recorded_video=video.path
            result['engine']=args.engine
            if neural_checkpoint:
                result['model_format']=checkpoint_data['format']
                result['algorithm']='ppo'
                result['policy_version']=checkpoint_data['policy_version']
                result['observation_contract']=checkpoint_data['observation']
                result['action_contract']=checkpoint_data['action']
            else:
                result['parameters']=params.tolist()
                result['parameters_sha256']=hashlib.sha256(json.dumps(params.tolist(),separators=(',',':')).encode()).hexdigest()
                result['policy_version']=actor_version(args.actor_spec)
                result.update(configuration(args.actor_spec))
            result['actor_source_sha256']=frozen_actor_hashes
            result['policy_source_sha256']=POLICY_SOURCE_SHA
            result['runtime_source_sha256']=RUNTIME_SOURCE_SHA256
            result['observation_profile']=args.observation_profile
            result['evaluation_conditions']={'seconds':args.seconds,'start_s':args.start_s,
                'observation_profile':args.observation_profile,
                'opponent_speed':args.opponent_speed,'opponent_gap':args.opponent_gap,
                'lidar_noise':args.lidar_noise,'lidar_dropout':args.lidar_dropout,
                'lidar_latency':args.lidar_latency,'recorded':bool(video)}
            result['track_sha256']=hashlib.sha256(track.path.read_bytes()).hexdigest()
            if args.checkpoint:
                result['checkpoint_sha256']=checkpoint_sha
                result['checkpoint_path']=str(Path(args.checkpoint).resolve())
                checkpoint_version=checkpoint_data.get('policy_version')
                result['checkpoint_policy_version']=checkpoint_version
                result['parameter_only_migration']=checkpoint_version!=result['policy_version']
            save(args.layout.trajectory_dir(args.engine,track.id)/f'seed{seed}.json',rows)
            results.append(result)
            # Preserve completed episodes if a later native episode is interrupted.
            save(args.layout.evaluation_path(args.engine,track.id),results)
            if args.json_console:
                print('RACE_EVAL',json.dumps(result),flush=True)
            else:
                print_evaluation(result,i+1,len(evaluation_seeds),time.monotonic()-evaluation_started)
        save(args.layout.evaluation_path(args.engine,track.id),results)
    except Exception:
        traceback.print_exc()
        save(args.layout.logs/f'{args.engine}_{track.id}_error.json',{'error':traceback.format_exc()})
        raise
    finally:env.close()
    artifacts={'run_id':args.layout.run_id,'run_dir':str(args.layout.root),
        'checkpoint':str(getattr(args,'trained_checkpoint',args.checkpoint)) if (args.train or args.checkpoint) else None,
        'metrics_jsonl':str(args.layout.metrics/'metrics.jsonl') if args.train else None,
        'tensorboard':str(args.tensorboard_path) if args.train and getattr(args,'tensorboard_active',False) else None,
        'video':str(recorded_video) if recorded_video else None}
    print('OSRACER_ARTIFACTS',json.dumps(artifacts,ensure_ascii=False),flush=True)
    return artifacts


if __name__ == "__main__":
    main()
