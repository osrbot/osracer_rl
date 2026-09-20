"""Construct an actor from a checkpoint without changing the fixed opponent.

The bundle contains controller configuration and exact module digests. It
never receives an environment, a track, or a privileged state.
"""
import hashlib
from pathlib import Path

import numpy as np

# A supervisor that keeps refusing every commanded action is itself evidence
# that the car is blocked; after this long the actor is asked to back off.
BLOCKED_ESCAPE_SECONDS = .6
# The supervisor's verdict flickers between scans, so a blocked car must not
# have its timer reset by a single optimistic packet: MuJoCo Spa sat behind an
# opponent envelope with the wheel encoders reading 0.005 m/s and the timer
# never passed 0.37 s.
INTERVENTION_HOLD_SECONDS = .5


def configuration(spec=None):
    spec=spec or {}
    kind=spec.get('actor_type','reactive')
    if kind not in ('reactive','closed_loop'):
        raise ValueError(f'Unknown actor_type: {kind}')
    controls=dict(spec.get('controls',{}))
    supported={'target_slip_deg','max_boost_seconds','steering_bias','confidence_threshold',
               'boost_countersteer_gain','yaw_rate_guard'}
    if set(controls)-supported:
        raise ValueError(f'Unknown controller options: {set(controls)-supported}')
    if controls and kind!='closed_loop':
        raise ValueError('Closed-loop controls require actor_type=closed_loop')
    return {'actor_type':kind,'controls':controls,'safety_supervisor':bool(spec.get('safety_supervisor',False))}


def core_policy_class(spec=None):
    if configuration(spec)['actor_type']=='closed_loop':
        from .policy_closed_loop import ClosedLoopRacingPolicy
        return ClosedLoopRacingPolicy
    from .policy import RacingPolicy
    return RacingPolicy


def actor_version(spec=None):
    cfg=configuration(spec)
    suffix=''
    if cfg['safety_supervisor']:
        from .safety import LocalSafetySupervisor
        suffix='+'+getattr(LocalSafetySupervisor,'version','footprint-safety')
    return core_policy_class(cfg).version+suffix


def actor_source_hashes(spec=None):
    cfg=configuration(spec)
    names=['policy_bundle.py','policy.py']
    if cfg['actor_type']=='closed_loop':names+=['policy_closed_loop.py','odometry.py']
    if cfg['safety_supervisor']:names+=['safety.py','safety_motion.py','odometry.py']
    root=Path(__file__).parent
    return {f'racing/{name}':hashlib.sha256((root/name).read_bytes()).hexdigest() for name in dict.fromkeys(names)}


class SupervisedActor:
    def __init__(self,actor):
        from .policy import DT, RW
        from .safety import LocalSafetySupervisor
        self.actor=actor
        self.supervisor=LocalSafetySupervisor()
        self.version=actor.version+'+'+getattr(LocalSafetySupervisor,'version','footprint-safety')
        self.dt, self.wheel_radius, self.blocked_seconds = DT, RW, 0.
        self.intervention_hold = 0.

    @property
    def p(self):return self.actor.p

    @property
    def feedback(self):
        return dict(getattr(self.actor,'feedback',{}),safety=dict(self.supervisor.diagnostics))

    @property
    def phase(self):
        if self.supervisor.diagnostics.get('reversing'):
            return 'reverse_escape'
        if self.supervisor.diagnostics.get('intervened'):
            return 'safety_recovery' if self.supervisor.diagnostics.get('recovery_end_clearance_m') is not None else 'safety_braking'
        return self.actor.phase

    def reset(self):
        self.actor.reset();self.supervisor.reset()
        self.blocked_seconds=self.intervention_hold=0.

    def action(self,observation):
        wheels,steers=self.actor.action(observation)
        result=self.supervisor.filter(observation,wheels,steers)
        measured=float(np.mean(np.abs(np.asarray(observation['wheel_vel'],float)[:2])))*self.wheel_radius
        estimate=self.supervisor.motion_diagnostics.get('estimate') or {}
        if estimate.get('valid'):
            body_speed=float(np.hypot(estimate.get('lidar_forward_speed',0.),
                                      estimate.get('lateral_speed',0.)))
        else:
            body_speed=measured
        if self.supervisor.diagnostics.get('intervened'):
            self.intervention_hold=INTERVENTION_HOLD_SECONDS
        else:
            self.intervention_hold=max(0.,self.intervention_hold-self.dt)
        # Wheels turning while the body does not move is the third blocked
        # signature: the car is pressed against something (or has no traction)
        # with the planner still asking to drive, and the supervisor has no
        # reason to intervene because nothing is closing on the scan.
        slipping=measured>.5 and body_speed<.15
        refused=(self.intervention_hold>0. or slipping) and body_speed<.15
        # Leaky accumulation: one noisy longitudinal estimate must not erase
        # half a second of continuous evidence that the car is not moving.
        self.blocked_seconds=(self.blocked_seconds+self.dt if refused
                              else max(0.,self.blocked_seconds-self.dt))
        if self.blocked_seconds>=BLOCKED_ESCAPE_SECONDS and hasattr(self.actor,'request_reverse_escape'):
            if self.actor.request_reverse_escape():
                self.blocked_seconds=0.
                wheels,steers=self.actor.action(observation)
                result=self.supervisor.filter(observation,wheels,steers)
        if (self.supervisor.diagnostics.get('commanded_reverse')
                and not self.supervisor.diagnostics.get('reversing')
                and hasattr(self.actor,'refund_reverse_escape')):
            # A refused back-off must not cost the episode its escape budget.
            self.actor.refund_reverse_escape()
        if self.supervisor.diagnostics.get('intervened') and hasattr(self.actor,'abort_drift'):
            self.actor.abort_drift()
        self.supervisor.diagnostics['blocked_seconds']=self.blocked_seconds
        self.supervisor.diagnostics['blocked_body_speed_mps']=body_speed
        self.supervisor.diagnostics['blocked_refused']=refused
        self.supervisor.diagnostics['intervention_hold_s']=self.intervention_hold
        return result


def make_actor(parameters=None,spec=None):
    cfg=configuration(spec)
    actor=core_policy_class(cfg)(parameters,**cfg['controls'])
    return SupervisedActor(actor) if cfg['safety_supervisor'] else actor
