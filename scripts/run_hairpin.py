#!/usr/bin/env python3
"""Train in Isaac Sim, evaluate identical checkpoint in either engine, record."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from hairpin_common import OUT, ROOT, DT, Policy, rollout

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))

def train(env, args):
    rng = np.random.default_rng(args.seed)
    mean = Policy.initial.copy()
    std = np.array([.09,.16,.22,.18,.15,.04,.025])
    best = None
    history = []
    start = time.monotonic()
    for generation in range(args.generations):
        population = np.clip(rng.normal(mean, std, (args.population, len(mean))),Policy.lower,Policy.upper)
        population[0] = mean
        if best: population[1] = best['parameters']
        results = []
        # Two initial conditions per candidate; same seeds within generation.
        for candidate, parameters in enumerate(population):
            metrics = [rollout(env, parameters, seed)[0] for seed in (0, generation+1)]
            reward = float(np.mean([m['reward'] for m in metrics]))
            item = dict(generation=generation, candidate=candidate, reward=reward,
                        parameters=parameters.tolist(), episodes=metrics)
            results.append(item); history.append(item)
            if best is None or reward > best['reward']:
                best = item
                save_json(args.checkpoint, dict(algorithm='CEM structured feedback policy',
                    trained_in='Isaac Sim 6.0.1 / PhysX', parameters=best['parameters'],
                    parameter_names=['lookahead_m','cruise_m_s','curvature_slowdown','steering_gain',
                                     'lateral_error_gain','steering_feedback','wheel_speed_feedback'],
                    reward=reward, generation=generation, candidate=candidate, seed=args.seed,
                    wheel_targets_order=['LF','RF','LR','RR'], wheel_signs=[1,1,1,-1],
                    observation='Path-relative pose, preview target, measured wheel velocities and steering positions',
                    control_hz=30, radius_m=.8, wheel_radius_m=.045))
            print('TRAIN',generation,candidate,'reward',round(reward,3),
                  'success',[m['success'] for m in metrics],
                  'time_s',round(time.monotonic()-start,1),flush=True)
            save_json(OUT/'training_history.json', history)
        elite = sorted(results,key=lambda x:x['reward'],reverse=True)[:max(2,args.population//4)]
        values = np.array([e['parameters'] for e in elite])
        mean = .25 * mean + .75 * values.mean(axis=0)
        std = np.maximum(.02*(Policy.upper-Policy.lower), .25*std+.75*values.std(axis=0))
    print('TRAIN_DONE',args.checkpoint,flush=True)

def select_checkpoint(env, args):
    """Select on fixed Isaac-only seeds; transfer evaluation is never optimized."""
    history=json.loads((OUT/'training_history.json').read_text())
    finalists=[]
    seen=set()
    for item in sorted(history,key=lambda x:x['reward'],reverse=True):
        key=tuple(item['parameters'])
        if key in seen:continue
        seen.add(key)
        scores=[rollout(env,item['parameters'],seed)[0] for seed in (10,11,12)]
        finalist=dict(parameters=item['parameters'],episodes=scores,
                      source_generation=item['generation'],source_candidate=item['candidate'],
                      reward=float(np.mean([s['reward'] for s in scores])))
        finalists.append(finalist)
        print('SELECT',len(finalists),finalist['reward'],flush=True)
        if len(finalists)==5:break
    winner=max(finalists,key=lambda x:x['reward'])
    checkpoint=json.loads(args.checkpoint.read_text())
    checkpoint.update(parameters=winner['parameters'],reward=winner['reward'],
                      generation=winner['source_generation'],candidate=winner['source_candidate'],
                      selection_seeds=[10,11,12],selection_engine='Isaac Sim / PhysX')
    save_json(args.checkpoint,checkpoint)
    save_json(OUT/'checkpoint_selection.json',finalists)

class Video:
    def __init__(self, path, engine, baseline=False):
        from PIL import ImageFont
        self.path, self.engine, self.frames = path, engine, 0
        self.baseline=baseline
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
        self.small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
        self.encoder = subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','error',
            '-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','pipe:0',
            '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(path)],stdin=subprocess.PIPE)

    def __call__(self, env, row, frame):
        from PIL import Image, ImageDraw
        image = Image.fromarray(env.render())
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,1280,79),fill=(15,22,34))
        draw.text((25,12),f'OSRACER | 180-degree hairpin | {self.engine}',font=self.font,fill='white')
        label='Untrained initial feedback policy' if self.baseline else 'Isaac-trained CEM feedback policy'
        draw.text((25,48),label+' | Wheel velocity + steering position targets',font=self.small,fill=(130,215,229))
        draw.rectangle((0,658,1280,720),fill=(15,22,34))
        speed=float(np.hypot(row['vx'],row['vy']))
        draw.text((25,670),f"t {row['t']:5.2f}s     speed {speed:.2f} m/s     cross-track {row['cte']:+.3f}m     yaw {np.degrees(row['yaw']):+.1f}deg",font=self.small,fill='white')
        self.encoder.stdin.write(image.tobytes()); self.frames += 1
        if frame in (0,100,200,300): image.save(self.path.with_name(self.path.stem+f'_{frame:04d}.png'))
        if frame%90==0:print('VIDEO',self.engine,frame,flush=True)

    def close(self):
        self.encoder.stdin.close()
        if self.encoder.wait(timeout=60): raise RuntimeError('Video encoding failed')
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(self.path),'-f','null','-'],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(self.path)]))
        assert int(probe['streams'][0]['nb_read_frames']) == self.frames
        return dict(path=str(self.path),frames=self.frames,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),probe=probe)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--engine',choices=['isaac','mujoco'],required=True)
    p.add_argument('--train',action='store_true')
    p.add_argument('--record',action='store_true')
    p.add_argument('--checkpoint',type=Path,default=OUT/'policy.json')
    p.add_argument('--generations',type=int,default=5)
    p.add_argument('--population',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--select',action='store_true',help='Select finalists using fixed Isaac-only seeds')
    args=p.parse_args()
    if args.train and args.engine!='isaac':p.error('Training is restricted to Isaac; MuJoCo is zero-shot evaluation')
    if args.select and args.engine!='isaac':p.error('Checkpoint selection is restricted to Isaac')
    if args.baseline and (args.train or args.select):p.error('Baseline cannot train or select')
    OUT.mkdir(parents=True,exist_ok=True)
    if args.engine=='isaac':
        from hairpin_isaac import IsaacEnv
        env=IsaacEnv(render=args.record)
    else:
        from hairpin_mujoco import MujocoEnv
        env=MujocoEnv()
    try:
        if args.train:train(env,args)
        if args.train or args.select:select_checkpoint(env,args)
        params=Policy.initial if args.baseline else json.loads(args.checkpoint.read_text())['parameters']
        evaluations=[]
        for seed in [0]+list(range(100,100+max(0,args.episodes-1))):
            metrics,trace=rollout(env,params,seed)
            evaluations.append(metrics)
            print('EVAL',args.engine,json.dumps(metrics),flush=True)
            suffix='_baseline' if args.baseline else ''
            if seed==0:save_json(OUT/f'{args.engine}{suffix}_trajectory.json',trace)
        report=dict(engine=args.engine,baseline=args.baseline,episodes=evaluations,
            runtime_version=env.version, control_hz=30,
            physics_hz=240 if args.engine=='isaac' else 480,
            success_rate=float(np.mean([m['success'] for m in evaluations])),
            checkpoint=None if args.baseline else str(args.checkpoint),
            checkpoint_sha256=None if args.baseline else hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            zero_shot_transfer=args.engine=='mujoco' and not args.baseline,
            eval_randomization='Nominal + held-out initial lateral offset +/- .025m, yaw +/- .03rad; fixed friction and mass')
        if args.record:
            label='Isaac Sim 6.0.1 / PhysX' if args.engine=='isaac' else ('MuJoCo baseline' if args.baseline else 'MuJoCo / zero-shot transfer')
            video=Video(OUT/f'{args.engine}{suffix}_hairpin.mp4',label,args.baseline)
            try: metrics,trace=rollout(env,params,0,capture=video)
            finally: report['video']=video.close()
            report['recorded_episode']=metrics
            save_json(OUT/f'{args.engine}{suffix}_recorded_trajectory.json',trace)
        save_json(OUT/(f'{args.engine}_'+('baseline.json' if args.baseline else 'evaluation.json')),report)
    finally:env.close()

if __name__=='__main__':main()
