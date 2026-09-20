"""Native PhysX acceleration, braking and steering diagnostic; no pose replay."""
import json
import numpy as np
from hairpin_common import ROOT, SIGNS
from hairpin_isaac import IsaacEnv
OUT=ROOT/'output/hairpin_fast'
env=IsaacEnv(output=OUT, physics_dt=1/480, control_dt=1/60, steering_limit=.45,max_joint_velocity_deg=30000.)
results={}
try:
    for target in (5.,10.):
        state=env.reset();trace=[]
        for i in range(240):
            command=min(target,(i+1)/60*8)
            state=env.step(np.full(4,command/.045)*SIGNS,np.zeros(2))
            trace.append({k:np.asarray(v).tolist() for k,v in state.items()})
        results[str(target)]=dict(trace=trace,max_body_speed=max(np.hypot(s['vx'],s['vy']) for s in trace))
        print('SPEED',target,'actual',results[str(target)]['max_body_speed'],'final',state,flush=True)
    env.reset();trace=[]
    for i in range(300):
        t=i/60
        speed=min(5.5,t*8)
        steer=0.
        rear=speed
        if t>=1.2: speed=1.5;rear=0.;steer=.45
        if t>=1.5: rear=8.
        if t>=1.9: rear=1.5;steer=-.3
        state=env.step(np.array([speed,speed,rear,-rear])/.045,np.array([steer,steer]))
        row={k:np.asarray(v).tolist() for k,v in state.items()};row['t']=t
        trace.append(row)
        if i%30==0:print('SLIDE',row,flush=True)
    results['slide']=trace
    (OUT/'probe_isaac.json').write_text(json.dumps(results,indent=2))
finally:env.close()
