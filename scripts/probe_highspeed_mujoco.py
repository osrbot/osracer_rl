#!/usr/bin/env python3
"""Bounded native-contact feasibility diagnostics; no policy training or state forcing.

Run python scripts/probe_highspeed_mujoco.py. Source assets and existing model
outputs are untouched: the derived model is built in a temporary directory.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import tempfile
import time
import numpy as np
from hairpin_mujoco import MujocoEnv, mujoco, WHEEL_RADIUS, CONTROL_DT

ROOT = Path(__file__).resolve().parents[1]
L, TRACK, LIMIT = .28764, .212, .45  # individual steering cap is an assumption
SIGNS = np.array([1., 1., 1., -1.])
RMIN = L / math.tan(LIMIT) + TRACK / 2


def ackermann(speed, radius=None):
    k = 0 if radius is None else 1 / radius
    left, right = 1 - TRACK*k/2, 1 + TRACK*k/2
    steer = np.arctan2(L*k, [left, right])
    wheels = speed/WHEEL_RADIUS * np.array([math.hypot(left,L*k), math.hypot(right,L*k),left,right])
    return wheels*SIGNS, steer


def sample(env, s, target, steer):
    origin_velocity = np.empty(6)
    mujoco.mj_objectVelocity(env.m, env.d, mujoco.mjtObj.mjOBJ_XBODY,
                            env._body_id, origin_velocity, 0)
    ovx, ovy = origin_velocity[3:5]
    origin_forward = ovx*math.cos(s['yaw']) + ovy*math.sin(s['yaw'])
    origin_lateral = -ovx*math.sin(s['yaw']) + ovy*math.cos(s['yaw'])
    origin_slip = math.atan2(origin_lateral, origin_forward) if math.hypot(ovx,ovy)>.1 else 0.
    v = math.hypot(s['vx'],s['vy'])
    f = s['vx']*math.cos(s['yaw']) + s['vy']*math.sin(s['yaw'])
    lateral = -s['vx']*math.sin(s['yaw']) + s['vy']*math.cos(s['yaw'])
    slip = math.atan2(lateral, f) if v>.1 else 0
    return dict(t=float(env.d.time),x=s['x'],y=s['y'],z=s['z'],yaw=s['yaw'],
                speed_m_s=v,forward_m_s=f,lateral_m_s=lateral,slip_rad=slip,
                base_origin_speed_m_s=float(math.hypot(ovx,ovy)),
                base_origin_slip_rad=float(origin_slip),
                base_origin_velocity_world_m_s=[float(ovx),float(ovy)],
                roll=s['roll'],pitch=s['pitch'],yaw_rate=s['yaw_rate'],
                wheel_surface_m_s=(s['wheel_vel']*WHEEL_RADIUS).tolist(),
                wheel_target_surface_m_s=(target*SIGNS*WHEEL_RADIUS).tolist(),
                steer=s['steer_pos'].tolist(),steer_target=steer.tolist(),
                contacts=int(env.d.ncon),
                torque=env.d.actuator_force[env._actuators[:4]].tolist())


def run(env, name, duration, control, tail_after=0):
    s = env.reset(0)
    rows=[]
    for step in range(round(duration/CONTROL_DT)):
        t = step*CONTROL_DT
        wheels, steer = control(t, s)
        s = env.step(wheels,steer)
        rows.append(sample(env,s,wheels,steer))
        if not np.isfinite(env.d.qpos).all() or abs(s['roll'])>1.0 or abs(s['pitch'])>1.0:
            break
    tail = [r for r in rows if r['t']>=tail_after] or rows
    values=lambda key:np.array([r[key] for r in rows])
    tail_values=lambda key:np.array([r[key] for r in tail])
    summary=dict(name=name,duration_s=rows[-1]['t'],final=rows[-1],
        max_body_speed_m_s=float(values('speed_m_s').max()),
        max_abs_roll_rad=float(abs(values('roll')).max()),
        max_abs_pitch_rad=float(abs(values('pitch')).max()),
        z_range_m=[float(values('z').min()),float(values('z').max())],
        min_contact_count=int(values('contacts').min()),
        fraction_control_samples_no_contacts=float(np.mean(values('contacts')==0)),
        tail_mean_body_speed_m_s=float(tail_values('speed_m_s').mean()),
        tail_mean_forward_speed_m_s=float(tail_values('forward_m_s').mean()),
        tail_mean_wheel_surface_m_s=np.mean(tail_values('wheel_surface_m_s'),axis=0).tolist(),
        tail_mean_abs_slip_rad=float(abs(tail_values('slip_rad')).mean()),
        tail_mean_abs_base_origin_slip_rad=float(abs(tail_values('base_origin_slip_rad')).mean()),
        tail_mean_steering_rad=np.mean(tail_values('steer'),axis=0).tolist(),
        tail_mean_yaw_rate=float(tail_values('yaw_rate').mean()),
        heading_change_rad=float(np.unwrap(values('yaw'))[-1]-values('yaw')[0]),
        warnings=env.d.warning.number.tolist())
    pts=np.array([[r['x'],r['y']] for r in tail])
    if len(pts)>3:
        centered=pts-pts.mean(axis=0)
        fitted=np.linalg.lstsq(np.column_stack((2*centered,np.ones(len(pts)))),np.sum(centered**2,axis=1),rcond=None)[0]
        radius=math.sqrt(max(0,fitted[2]+np.sum(fitted[:2]**2)))
        residual=np.linalg.norm(centered-fitted[:2],axis=1)-radius
        summary.update(tail_fit_base_origin_circle_radius_m=float(radius),tail_circle_fit_rms_m=float(np.sqrt(np.mean(residual**2))))
    # 10 Hz trace supports inspection of braking/slide without an excessive file.
    summary['trace_10hz']=rows[::3]
    return summary


def main():
    start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='neoracer-speed-probe-') as tmp:
        env=MujocoEnv(Path(tmp)/'mujoco_model.xml')
        results=[]
        for goal in (5.,10.):
            results.append(run(env,f'straight_ramp_{goal:g}',7.,lambda t,s,g=goal:ackermann(min(g,3*t)),tail_after=5.))
        results.append(run(env,'low_speed_max_individual_steer_ackermann',12.,lambda t,s:ackermann(min(.5,t),RMIN),tail_after=3.))
        results.append(run(env,'low_speed_parallel_steer_cap',12.,lambda t,s:(np.ones(4)*min(.5,t)/WHEEL_RADIUS*SIGNS,np.ones(2)*LIMIT),tail_after=3.))
        for radius in (3.,RMIN):
            def turn(t,s,r=radius):
                return ackermann(min(5.,3*t),None if t<2.5 else r)
            results.append(run(env,f'entry_5_constant_speed_radius_{radius:.3f}',5.,turn,tail_after=2.5))
        def brake_turn(t,s):
            v=min(5.,3*t) if t<2.5 else max(1.4,5.-7*(t-2.5))
            return ackermann(v,None if t<2.5 else RMIN)
        results.append(run(env,'entry_5_brake_to_1p4_at_max_steer',5.,brake_turn,tail_after=3.2))
        def rear_brake(t,s):
            v=min(5.,3*t) if t<2.5 else max(1.4,5.-7*(t-2.5))
            w,a=ackermann(v,None if t<2.5 else RMIN)
            if 2.5<=t<2.75:
                w[2:]=0.
            return w,a
        results.append(run(env,'entry_5_rear_brake_pulse_release',5.,rear_brake,tail_after=3.2))
        def differential(t,s):
            w,a=ackermann(min(5.,3*t),None if t<2.5 else RMIN)
            if t>=2.5:
                w *= [1,1,.8,1.2]
            return w,a
        results.append(run(env,'entry_5_rear_differential_20pct',5.,differential,tail_after=2.5))
        metadata=dict(engine='MuJoCo',version=env.version,physics_dt_s=float(env.m.opt.timestep),
            mass_kg=float(env.m.body_mass.sum()),wheel_radius_m=WHEEL_RADIUS,
            wheelbase_m=L,track_m=TRACK,individual_steering_limit_rad_assumed=LIMIT,
            steering_source='Source robot.xml has limited=false and no range: .45 rad is an explicit diagnostic assumption, not a verified mechanical stop.',
            nominal_ackermann_rear_axle_min_radius_m=RMIN,
            wheel_collision_types=[int(env.m.geom(n.replace('_joint','_link')+'_collision_0').type[0]) for n in ('left_front_wheel_joint','right_front_wheel_joint','left_rear_wheel_joint','right_rear_wheel_joint')],
            cylinder_type_enum=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            wheel_torque_cap_Nm=.3,velocity_servo_kv_Nms=.03,
            friction_mu=1.,speed5_friction_radius_bound_m=25/9.81,
            speed10_friction_radius_bound_m=100/9.81,
            friction_speed_at_nominal_min_radius_m_s=math.sqrt(9.81*RMIN),
            speed5_to1p4_ideal_braking_distance_m=(25-1.4**2)/(2*9.81),
            notes=[
                'All runs use native contact integration from rest. No body pose or velocity assignment after reset.',
                'Drive gains, torque cap and friction are simulation assumptions inherited from hairpin_mujoco.build_model, not measured hardware limits.',
                'Exact MuJoCo cylinder primitives replace original collision meshes; wheel contacts are not rolling on a faceted collision polygon.',
                'Simple isotropic Coulomb contact has no calibrated pneumatic tire slip curve, suspension, or motor torque-speed envelope.',
                'Contact diagnostics are sampled at control rate and do not establish absence of between-sample contact loss.',
                'Circle fitting high-speed transient paths is only descriptive; radius is valid as steering-circle evidence only for the settled low-speed run.',
                'Existing state vx/vy from mjOBJ_BODY are at the base inertial COM, whereas x/y are the rear-axle base origin. Normal turning induces COM sideslip without rear tire drift. Explicit mjOBJ_XBODY rear-origin velocities and slips are additionally reported.',
                '5-10 m/s names are wheel-surface commands; measured body speeds are reported separately.',
                'No engine transfer or tuned hairpin-success claim follows from this diagnostic.'
            ])
        env.close()
    report=dict(metadata=metadata,runs=results,compute_seconds=time.monotonic()-start)
    out=ROOT/'output/hairpin_fast/probe_mujoco.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(metadata=metadata,runs=[{k:v for k,v in r.items() if k not in ('trace_10hz','final')} for r in results],compute_seconds=report['compute_seconds']),indent=2))

if __name__=='__main__':
    main()
