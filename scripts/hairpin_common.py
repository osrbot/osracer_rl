"""Engine-independent feedback policy and native-dynamics evaluation contract."""
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/hairpin'
DT = 1 / 30
RADIUS, WHEEL_RADIUS, WHEELBASE, TRACK = .8, .045, .28764, .212
WHEELS = ['left_front_wheel_joint', 'right_front_wheel_joint',
          'left_rear_wheel_joint', 'right_rear_wheel_joint']
STEERS = ['left_steering_hinge_joint', 'right_steering_hinge_joint']
SIGNS = np.array([1., 1., 1., -1.])
LENGTH = 2.4 + np.pi * RADIUS

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def path(s):
    s = np.clip(np.asarray(s), 0, LENGTH)
    theta = (s - 1.2) / RADIUS - np.pi / 2
    x = np.where(s < 1.2, s, np.where(s <= 1.2 + np.pi * RADIUS,
                 1.2 + RADIUS * np.cos(theta), LENGTH - s))
    y = np.where(s < 1.2, 0, np.where(s <= 1.2 + np.pi * RADIUS,
                 RADIUS + RADIUS * np.sin(theta), 2 * RADIUS))
    return np.stack([x, y], axis=-1)

SAMPLES = np.linspace(0, LENGTH, 1000)
POINTS = path(SAMPLES)

def project(state):
    xy = np.array([state['x'], state['y']])
    i = int(np.argmin(np.sum((POINTS - xy) ** 2, axis=1)))
    s = float(SAMPLES[i])
    yaw = float(np.clip((s - 1.2) / RADIUS, 0, np.pi))
    delta = xy - POINTS[i]
    cte = float(-np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1])
    return s, cte, float(wrap(state['yaw'] - yaw))

class Policy:
    """CEM-trained structured policy; no engine-specific parameters or state.

    Parameters: lookahead, cruise speed, curvature slowdown, steering gain,
    lateral-error gain, measured steering feedback, wheel-speed feedback.
    Path-relative pose + wheel encoders + steering encoders form observations.
    """
    initial = np.array([.34, .65, .45, 1., .25, .03, .025])
    lower = np.array([.16, .35, .05, .65, 0., 0., 0.])
    upper = np.array([.6, 1.0, 1.2, 1.5, .8, .2, .1])

    def __init__(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)
        self.last_steer = 0.
        self.last_speed = 0.

    def action(self, state):
        look, cruise, slow, gain, lateral, feedback, vfeedback = self.parameters
        s, cte, _ = project(state)
        target = path(s + look)
        dx, dy = target - [state['x'], state['y']]
        local_y = -np.sin(state['yaw']) * dx + np.cos(state['yaw']) * dy
        curvature = 2 * local_y / max(dx * dx + dy * dy, .01)
        delta = gain * np.arctan(WHEELBASE * curvature) - lateral * cte
        delta += feedback * (delta - float(np.mean(state['steer_pos'])))
        delta = float(np.clip(delta, -.55, .55))
        delta = float(np.clip(delta, self.last_steer - .05, self.last_steer + .05))
        velocity = cruise / (1 + slow * abs(curvature))
        velocity *= min(1., max(.25, (LENGTH - s) / .35))
        measured = float(np.mean(state['wheel_vel'])) * WHEEL_RADIUS
        velocity += vfeedback * (velocity - measured)
        velocity = float(np.clip(velocity, self.last_speed - .06, self.last_speed + .035))
        self.last_speed, self.last_steer = velocity, delta
        k = np.tan(delta) / WHEELBASE
        left, right = 1 - TRACK * k / 2, 1 + TRACK * k / 2
        steer = np.arctan2(WHEELBASE * k, np.array([left, right]))
        wheels = velocity / WHEEL_RADIUS * np.array([
            np.hypot(left, WHEELBASE * k), np.hypot(right, WHEELBASE * k), left, right])
        return wheels * SIGNS, steer

def load_policy(filename):
    return Policy(json.loads(Path(filename).read_text())['parameters'])

def rollout(env, parameters, seed=0, capture=None, max_seconds=18):
    policy = Policy(parameters)
    state = env.reset(seed)
    rows = []
    success = False
    failure = 'timeout'
    max_progress = 0.
    for i in range(int(max_seconds / DT)):
        wheel, steer = policy.action(state)
        state = env.step(wheel, steer)
        s, cte, heading = project(state)
        max_progress = max(max_progress, s)
        dist = float(np.linalg.norm(np.array([state['x'], state['y']]) - path(LENGTH)))
        finite = all(np.isfinite(np.asarray(v)).all() for v in state.values())
        row = {k: np.asarray(v).tolist() for k, v in state.items()}
        row.update(t=(i + 1) * DT, progress=s, cte=cte, heading_error=heading,
                   wheel_targets=wheel.tolist(), steering_targets=steer.tolist())
        rows.append(row)
        if capture is not None:
            capture(env, row, i)
        if not finite or abs(state['roll']) > .5 or abs(state['pitch']) > .5 or state['z'] < .015:
            failure = 'unstable'; break
        if abs(cte) > .28:
            failure = 'lane_departure'; break
        if s > LENGTH - .12 and dist < .16 and abs(wrap(state['yaw'] - np.pi)) < .18:
            success = True; failure = None; break
    errors = np.array([r['cte'] for r in rows])
    metrics = dict(seed=int(seed), success=success, failure=failure,
                   duration_s=len(rows) * DT, progress_m=max_progress,
                   max_abs_cte_m=float(np.max(np.abs(errors))),
                   rms_cte_m=float(np.sqrt(np.mean(errors ** 2))),
                   final_heading_error_rad=float(wrap(state['yaw'] - np.pi)),
                   endpoint_error_m=dist)
    score = (100 if success else 0) + 10 * max_progress - .9 * metrics['duration_s'] \
            - 60 * metrics['rms_cte_m'] - 20 * metrics['max_abs_cte_m']
    metrics['reward'] = score
    return metrics, rows
