"""High-speed entry / controlled-slip hairpin task, shared across engines."""
from pathlib import Path
import numpy as np
from hairpin_common import ROOT, WHEELS, STEERS, SIGNS, wrap

OUT=ROOT/'output/hairpin_fast'
DT=1/60
STRAIGHT=6.
RADIUS=.8
WIDTH=1.2
LENGTH=2*STRAIGHT+np.pi*RADIUS
RW=.045
WB=.28764
TRACK=.212
STEER_LIMIT=.45  # Explicit assumed per-wheel mechanical limit, not calibrated.
RMIN=WB/np.tan(STEER_LIMIT)+TRACK/2
CENTER_LIMIT=np.arctan(WB/RMIN)

def path(s):
    s=np.asarray(s)
    a=(s-STRAIGHT)/RADIUS-np.pi/2
    return np.stack([np.where(s<STRAIGHT,s,np.where(s<STRAIGHT+np.pi*RADIUS,STRAIGHT+RADIUS*np.cos(a),LENGTH-s)),
                     np.where(s<STRAIGHT,0.,np.where(s<STRAIGHT+np.pi*RADIUS,RADIUS+RADIUS*np.sin(a),2*RADIUS))],axis=-1)

SAMPLES=np.linspace(0,LENGTH,1800)
POINTS=path(SAMPLES)

def project(state):
    xy=np.array([state['x'],state['y']])
    i=int(np.argmin(np.sum((POINTS-xy)**2,axis=1)))
    s=float(SAMPLES[i]);heading=float(np.clip((s-STRAIGHT)/RADIUS,0,np.pi))
    e=xy-POINTS[i]
    return s,float(-np.sin(heading)*e[0]+np.cos(heading)*e[1]),float(wrap(state['yaw']-heading))

class FastPolicy:
    names=['cruise_m_s','brake_distance_m','corner_m_s','straight_lookahead_m','turn_lookahead_m',
           'steering_gain','yaw_damping','drift_start_offset_m','drift_end_yaw_rad',
           'rear_overdrive_m_s','drift_steering_bias_rad','slip_feedback']
    initial=np.array([5.8,1.6,2.0,.85,.4,1.,.02,-.05,1.7,6.,.1,.25])
    lower=np.array([5.5,.6,1.3,.5,.2,.7,0.,-.5,.5,2.5,-.2,0.])
    upper=np.array([7.,2.4,3.,1.5,.9,1.6,.14,.5,2.5,12.,.3,.8])
    spread=np.array([.25,.35,.35,.15,.15,.2,.025,.2,.45,2.,.12,.15])

    def __init__(self,parameters):
        self.p=np.asarray(parameters,dtype=float)
        self.last_speed=0.
        self.last_steer=0.
        self.phase='accelerate'
        self.past_turn=False
        self.slip_mode='idle'
        self.slip_time=0.
        self.slip_hold=0.

    def action(self,state):
        cruise,brake,corner,look_straight,look_turn,gain,damping,start,end,rear,bias,beta_gain=self.p
        s,cte,heading_error=project(state)
        if s>STRAIGHT+np.pi*RADIUS-.3:self.past_turn=True
        approach=s<STRAIGHT-brake
        exit_phase=self.past_turn
        target_speed=cruise if approach or exit_phase else corner
        look=look_straight if approach or exit_phase else look_turn
        preview=path(min(LENGTH+.5,s+look))
        dx,dy=preview-[state['x'],state['y']]
        local_y=-np.sin(state['yaw'])*dx+np.cos(state['yaw'])*dy
        curvature=2*local_y/max(dx*dx+dy*dy,.01)
        steer=gain*np.arctan(WB*curvature)
        yaw=float(state['yaw']%(2*np.pi))
        measured=np.hypot(state['vx'],state['vy'])
        if self.slip_mode=='idle' and s>STRAIGHT+start and .03<yaw<end and not self.past_turn:
            self.slip_mode='active'
        if self.slip_mode=='active':
            self.slip_time+=DT
            self.slip_hold=self.slip_hold+DT if state['rear_slip_beta']<-.4 and measured>1.2 else 0.
            if ((yaw>end and self.slip_hold>=.15) or yaw>2.75 or self.slip_time>.85 or self.past_turn
                    or (measured<.7 and self.slip_time>.2)):
                self.slip_mode='recover'
        drifting=self.slip_mode=='active'
        if drifting:
            steer+=bias+beta_gain*(state['rear_slip_beta']+.35)
        steer-=damping*state['yaw_rate']
        steer=float(np.clip(steer,-CENTER_LIMIT,CENTER_LIMIT))
        steer=float(np.clip(steer,self.last_steer-3.*DT,self.last_steer+3.*DT))
        target_speed+=.4*(target_speed-measured)
        speed=float(np.clip(target_speed,self.last_speed-14*DT,self.last_speed+8*DT))
        self.last_steer,self.last_speed=steer,speed
        k=np.tan(steer)/WB
        left,right=1-TRACK*k/2,1+TRACK*k/2
        steering=np.clip(np.arctan2(WB*k,[left,right]),-STEER_LIMIT,STEER_LIMIT)
        # Adapt rear slip using measured feedback in both engines. In particular,
        # body heading alone cannot determine whether lateral saturation formed.
        if drifting:
            slip_error=.65+state['rear_slip_beta']
            scale=float(np.clip(1+2*slip_error,.3,2.))
            rear_speed=speed+max(0.,rear-speed)*scale
        else:rear_speed=speed
        wheels=np.array([speed*np.hypot(left,WB*k),speed*np.hypot(right,WB*k),rear_speed*left,rear_speed*right])/RW
        self.phase='drift' if drifting else ('exit' if exit_phase else ('accelerate' if approach else 'brake/turn'))
        return wheels*SIGNS,steering

def reward(metrics):
    m=metrics
    score=100*m['completed']+8*m['final_progress_m']-2*m['duration_s']-40*m['rms_cte_m']-20*m['max_abs_cte_m']
    # Train with margin beyond the unchanged 20-degree/0.1-second acceptance.
    score+=50*min(1.,m['drift_duration_s']/.4)+15*min(1.,m['max_oversteer_slip_deg']/45.)
    score+=8*min(6.,m['entry_peak_m_s'])-30*max(0.,5.-m['entry_peak_m_s'])
    return float(score)

def rollout(env,parameters,seed=0,capture=None,max_seconds=10):
    state=env.reset(seed);policy=FastPolicy(parameters)
    rows=[];failure='timeout';completed=False
    for i in range(round(max_seconds/DT)):
        wheels,steering=policy.action(state)
        state=env.step(wheels,steering)
        s,cte,heading=project(state)
        speed=float(np.hypot(state['vx'],state['vy']))
        row={k:np.asarray(v).tolist() for k,v in state.items()}
        row.update(t=(i+1)*DT,progress=s,cte=cte,heading_error=heading,body_speed=speed,
                   phase=policy.phase,wheel_targets=wheels.tolist(),steering_targets=steering.tolist())
        rows.append(row)
        if capture is not None:capture(env,row,i)
        if not all(np.isfinite(np.asarray(v)).all() for v in state.values()):failure='nonfinite';break
        if abs(state['roll'])>.5 or abs(state['pitch'])>.5 or state['z']<.015:failure='unstable';break
        if abs(cte)>.55:failure='lane_departure';break
        if s>LENGTH-.2 and np.hypot(state['x'],state['y']-2*RADIUS)<.3 and abs(wrap(state['yaw']-np.pi))<.2:
            completed=True;failure=None;break
    speeds=np.array([r['body_speed'] for r in rows]);betas=np.array([r['rear_slip_beta'] for r in rows])
    ss=np.array([r['progress'] for r in rows]);errors=np.array([r['cte'] for r in rows])
    entry=speeds[ss<STRAIGHT];turn=speeds[(ss>=STRAIGHT)&(ss<=STRAIGHT+np.pi*RADIUS)]
    drifting=(betas<-.35)&(speeds>1.2)&(ss>STRAIGHT-.5)&(ss<STRAIGHT+np.pi*RADIUS)
    drift_time=float(np.sum(drifting)*DT)
    entry_max=float(np.max(entry)) if len(entry) else 0.
    metrics=dict(seed=seed,completed=completed,success=bool(completed and entry_max>=5 and drift_time>=.1),
                 failure=failure,duration_s=len(rows)*DT,entry_peak_m_s=entry_max,
                 peak_speed_m_s=float(speeds.max()),turn_min_m_s=float(turn.min()) if len(turn) else None,
                 turn_mean_m_s=float(turn.mean()) if len(turn) else None,drift_duration_s=drift_time,
                 max_rear_slip_deg=float(np.degrees(np.max(np.abs(betas[speeds>1.2])))) if np.any(speeds>1.2) else 0.,
                 max_oversteer_slip_deg=float(np.degrees(max(0.,-np.min(betas[speeds>1.2])))) if np.any(speeds>1.2) else 0.,
                 max_abs_cte_m=float(np.abs(errors).max()),rms_cte_m=float(np.sqrt(np.mean(errors**2))),
                 final_progress_m=float(ss[-1]),final_yaw_error_rad=float(wrap(state['yaw']-np.pi)),
                 max_roll_deg=float(max(abs(r['roll']) for r in rows)*180/np.pi))
    metrics['reward']=reward(metrics)
    if completed and not metrics['success']:
        metrics['failure']='insufficient_entry_speed' if entry_max<5 else 'insufficient_sustained_rear_slip'
    return metrics,rows
