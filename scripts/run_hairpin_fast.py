#!/usr/bin/env python3
"""Train 5+ m/s entry and controlled rear slip in native Isaac; transfer unchanged."""
import argparse,hashlib,json,time,subprocess
import numpy as np
from hairpin_fast_common import OUT,DT,FastPolicy,rollout,RMIN,reward

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2))

def train(env,args):
    rng=np.random.default_rng(73)
    mean=FastPolicy.initial.copy();std=FastPolicy.spread.copy();history=[];best=None
    if args.resume and (OUT/'policy.json').exists():
        mean=np.array(json.loads((OUT/'policy.json').read_text())['parameters'])
        if (OUT/'training_history.json').exists():
            old=json.loads((OUT/'training_history.json').read_text())
            source=max(old,key=lambda x:reward(x['metrics']))
            mean=np.array(source['parameters'])
            print('RESUME_ISAAC_MARGIN',source['generation'],source['candidate'],reward(source['metrics']),flush=True)
    start=time.monotonic()
    for generation in range(args.generations):
        population=np.clip(rng.normal(mean,std,(args.population,len(mean))),FastPolicy.lower,FastPolicy.upper)
        population[0]=mean
        if best:population[1]=best['parameters']
        trials=[]
        for j,p in enumerate(population):
            metrics,rows=rollout(env,p,seed=0)
            item=dict(generation=generation,candidate=j,parameters=p.tolist(),metrics=metrics,reward=metrics['reward'])
            trials.append(item);history.append(item)
            if best is None or item['reward']>best['reward']:
                best=item
                save(OUT/'policy.json',dict(parameters=p.tolist(),parameter_names=FastPolicy.names,
                    algorithm='CEM structured high-speed drift feedback',trained_in='Isaac Sim 6.0.1 / PhysX',
                    policy_version=2,slip_feedback='Latched initiate/hold/recover; beta target -.65rad; qualify -.4rad for consecutive .15s; yaw/time recovery guards',
                    metrics=metrics,control_hz=60,steering_limit_rad=.45,ideal_min_radius_m=RMIN,
                    objective='Entry body speed >=5m/s, rear oversteer slip >20deg for >=.1s, completed 180-degree hairpin',
                    training_margin='Reward targets 45deg peak oversteer and .4s above20deg; acceptance remains20deg/.1s',
                    scene=dict(straight_m=6,radius_m=.8,road_width_m=1.2),generation=generation,candidate=j))
                save(OUT/'training_best_trace.json',rows)
            print('TRAIN_FAST',generation,j,json.dumps(metrics),'wall',round(time.monotonic()-start,1),flush=True)
            save(OUT/'training_history.json',history)
        elite=sorted(trials,key=lambda v:v['reward'],reverse=True)[:max(3,args.population//4)]
        values=np.array([e['parameters'] for e in elite])
        mean=.2*mean+.8*values.mean(axis=0)
        std=np.maximum(.025*(FastPolicy.upper-FastPolicy.lower),.2*std+.8*values.std(axis=0))

class Video:
    def __init__(self,engine,trained=True):
        from PIL import ImageFont
        self.engine=engine;self.frames=0;self.path=OUT/f'{engine}_fast_hairpin.mp4'
        self.trained=trained
        self.peak_slip=0.
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
        self.small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
        self.proc=subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo',
            '-pix_fmt','rgb24','-s','1280x720','-r','30','-i','pipe:0','-an','-c:v','libx264',
            '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE)

    def __call__(self,env,row,index):
        if index%2:return
        from PIL import Image,ImageDraw
        image=Image.fromarray(env.render());draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,1280,78),fill=(15,22,34))
        draw.text((24,12),f'OSRACER | High-speed entry + 180-degree hairpin | {self.engine}',font=self.font,fill='white')
        label='Same Isaac-trained policy' if self.trained else 'Untrained initial policy'
        draw.text((24,46),'Real-time playback | Native contact dynamics | '+label,font=self.small,fill=(130,215,229))
        draw.rectangle((0,626,1280,720),fill=(15,22,34))
        wheel=np.array(row['wheel_vel'])*.045
        draw.text((24,640),f"Body {row['body_speed']:.2f} m/s    Wheel mean {wheel.mean():.2f} m/s    Rear slip {np.degrees(row['rear_slip_beta']):+.1f} deg",font=self.small,fill='white')
        draw.text((24,674),f"t {row['t']:.2f}s    {row['phase']}    yaw {np.degrees(row['yaw']):+.0f} deg    cross-track {row['cte']:+.2f}m",font=self.small,fill=(130,215,229))
        self.proc.stdin.write(image.tobytes());self.frames+=1
        if index%60==0:image.save(OUT/f'{self.engine}_fast_{index:04d}.png')
        if row['body_speed']>1.2 and -row['rear_slip_beta']>self.peak_slip:
            self.peak_slip=-row['rear_slip_beta']
            image.save(OUT/f'{self.engine}_fast_peak_slip.png')
    def close(self):
        self.proc.stdin.close()
        if self.proc.wait(timeout=60):raise RuntimeError('ffmpeg encoding failed')
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(self.path),'-f','null','-'],check=True)
        return dict(path=str(self.path),frames=self.frames,fps=30,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--engine',choices=['isaac','mujoco'],required=True)
    p.add_argument('--train',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--generations',type=int,default=8);p.add_argument('--population',type=int,default=16)
    p.add_argument('--episodes',type=int,default=10);p.add_argument('--record',action='store_true')
    p.add_argument('--initial',action='store_true')
    p.add_argument('--select',action='store_true')
    p.add_argument('--strict-margin',action='store_true')
    p.add_argument('--eval-seed-start',type=int,default=100)
    args=p.parse_args()
    if args.train and args.engine!='isaac':p.error('Policy training is restricted to Isaac')
    if args.select and args.engine!='isaac':p.error('Policy selection is restricted to Isaac')
    OUT.mkdir(parents=True,exist_ok=True)
    if args.engine=='isaac':
        from hairpin_fast_isaac import FastIsaacEnv
        env=FastIsaacEnv(render=args.record)
    else:
        from hairpin_fast_mujoco import FastMujocoEnv
        env=FastMujocoEnv(render=args.record)
    try:
        if args.train:train(env,args)
        if args.train or args.select:
            history=json.loads((OUT/'training_history.json').read_text())
            candidates=[];seen=set()
            def robust_source(item):
                m=item['metrics']
                return m['success'] and m['max_oversteer_slip_deg']>=55 and m['drift_duration_s']>=.35
            ordered=sorted(history,key=lambda x:(robust_source(x) if args.strict_margin else True,x['reward']),reverse=True)
            for item in ordered:
                key=tuple(item['parameters'])
                if key in seen:continue
                seen.add(key)
                metrics=[rollout(env,item['parameters'],seed)[0] for seed in (10,11,12)]
                candidate=dict(parameters=item['parameters'],episodes=metrics,
                    source_generation=item['generation'],source_candidate=item['candidate'],training_metrics=item['metrics'],
                    margin_pass=all(m['success'] and m['max_oversteer_slip_deg']>=55 and m['drift_duration_s']>=.35 for m in metrics),
                    success_count=sum(m['success'] for m in metrics),reward=float(np.mean([m['reward'] for m in metrics])))
                candidates.append(candidate)
                print('SELECT_FAST',len(candidates),candidate['success_count'],candidate['reward'],'margin',candidate['margin_pass'],flush=True)
                if len(candidates)>=5:break
            eligible=[c for c in candidates if c['margin_pass']] if args.strict_margin else candidates
            if not eligible:
                save(OUT/'checkpoint_selection_failed.json',candidates)
                raise RuntimeError('No candidate met 55deg/.35s source validation margin')
            winner=max(eligible,key=lambda c:(c['success_count'],c['reward']))
            checkpoint=json.loads((OUT/'policy.json').read_text())
            checkpoint.update(parameters=winner['parameters'],metrics=winner['training_metrics'],
                generation=winner['source_generation'],candidate=winner['source_candidate'],selection=winner,selection_seeds=[10,11,12])
            checkpoint['strict_source_validation_margin']=args.strict_margin
            save(OUT/'policy.json',checkpoint);save(OUT/'checkpoint_selection.json',candidates)
        params=FastPolicy.initial if args.initial else json.loads((OUT/'policy.json').read_text())['parameters']
        evaluations=[]
        for seed in [0]+list(range(args.eval_seed_start,args.eval_seed_start+args.episodes-1)):
            metrics,trace=rollout(env,params,seed)
            print('EVAL_FAST',args.engine,json.dumps(metrics),flush=True)
            evaluations.append(metrics)
            save(OUT/f'{args.engine}_trace_seed{seed}.json',trace)
        report=dict(engine=args.engine,version=env.version,episodes=evaluations,
                    checkpoint_sha256=None if args.initial else hashlib.sha256((OUT/'policy.json').read_bytes()).hexdigest())
        if args.record:
            video=Video(args.engine,trained=not args.initial)
            try:metrics,trace=rollout(env,params,0,capture=video)
            finally:report['video']=video.close()
            report['recorded_episode']=metrics;save(OUT/f'{args.engine}_recorded_trace.json',trace)
        save(OUT/f'{args.engine}_evaluation.json',report)
    finally:env.close()

if __name__=='__main__':main()
