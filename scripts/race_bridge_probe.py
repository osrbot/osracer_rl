#!/usr/bin/env python3
"""Native bridge geometry fixture; uses privileged steering, never a race policy.

python3 scripts/race_bridge_probe.py --engine mujoco --render
bash scripts/run_isaac.sh scripts/race_bridge_probe.py --engine isaac --render
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from racing.tracks import Track

DT = 1/60


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')


def controls(track, states, progress, wheel_speed, park_lower_after=None):
    """Privileged pure-pursuit fixture: .8 m lookahead, native actuator outputs."""
    actions, projected, errors = [], [], []
    for j, state in enumerate(states):
        s, cte, _ = track.project([state['x'], state['y']], s_hint=progress[j], z=state['z'])
        delta = (s-progress[j]+track.length/2) % track.length-track.length/2
        projected.append(progress[j]+delta)
        errors.append(float(cte))
        target, _ = track.at(s+.8)
        angle = math.atan2(target[1]-state['y'], target[0]-state['x'])-state['yaw']
        steering = float(np.clip(math.atan2(2*.28855*math.sin(angle), .8), -.45, .45))
        speed = wheel_speed
        if j == 1 and park_lower_after is not None and projected[j] >= park_lower_after:
            speed = 0.
        actions.append(([speed, speed, speed, -speed], [steering, steering]))
    return actions, projected, errors


def run_case(env, case, output, max_seconds=60., render=False):
    track, bridge = env.track, env.track.metadata['bridge']
    crossing = case == 'crossing'
    starts = ([bridge['upper_s']-.7, bridge['lower_s']-.7] if crossing else
              [bridge['elevated_start_s']-1., bridge['lower_s']-1.])
    states = env.reset(seed=0, starts=[(s, 0.) for s in starts])
    progress = starts.copy()
    initial = serial(states)
    rows, failures = [], []
    max_height = max_cte = max_height_error = max_progress_step = 0.
    min_xy = float('inf')
    min_height_gap = float('inf')
    contact_steps = [0, 0]
    goal = bridge['upper_s']+1.5 if crossing else bridge['elevated_end_s']+1.
    screenshot = None
    for tick in range(round(max_seconds/DT)):
        actions, before, _ = controls(track, states, progress, 30. if crossing else 35.,
                                      None if crossing else bridge['lower_s']+4.)
        states = env.step(actions)
        _, progress, ctes = controls(track, states, before, 0.)
        max_progress_step = max(max_progress_step, *(abs(a-b) for a, b in zip(before, progress)))
        for j, state in enumerate(states):
            contact_steps[j] += int(state['collision'])
            if state['collision']:
                failures.append(f'car{j}: native collision')
            if abs(ctes[j]) > track.width/2-.125:
                failures.append(f'car{j}: off road')
            if abs(state['roll']) > .5 or abs(state['pitch']) > .5 or state['z'] < .015:
                failures.append(f'car{j}: unstable body pose')
        xy_distance = math.hypot(states[0]['x']-states[1]['x'], states[0]['y']-states[1]['y'])
        min_xy = min(min_xy, xy_distance)
        if crossing and xy_distance < .4:
            min_height_gap = min(min_height_gap, states[0]['z']-states[1]['z'])
        max_height = max(max_height, states[0]['z'])
        max_cte = max(max_cte, *(abs(c) for c in ctes))
        max_height_error = max(max_height_error, abs(states[0]['z']-track.elevation_at(progress[0])-.045))
        rows.append(serial({'t': (tick+1)*DT, 'states': states, 'actions': actions,
                            'route_s': progress, 'cte': ctes, 'xy_separation_m': xy_distance}))
        if render and screenshot is None and (xy_distance < .1 if crossing else progress[0] >= bridge['upper_s']):
            from PIL import Image
            screenshot = output.with_name(output.name+f'_{case}.png')
            Image.fromarray(env.render()).save(screenshot)
        if max_progress_step > .6:
            failures.append('Discontinuous route progress')
        if failures or progress[0] >= goal:
            break
    if progress[0] < goal:
        failures.append('Fixture did not finish within time limit')
    if max_height_error > .03:
        failures.append('Bridge surface tracking error exceeds 3 cm')
    if crossing:
        if min_xy >= .1:
            failures.append('Vehicles did not meet within 10 cm in XY')
        if min_height_gap < bridge['surface_height']-.03:
            failures.append('Vertical separation at the crossing is too small')
        if progress[1] <= bridge['lower_s']:
            failures.append('Lower vehicle did not pass the crossing')
    else:
        if max_height < bridge['surface_height']+.025:
            failures.append('Upper vehicle did not ascend onto the bridge')
        if abs(states[0]['z']-.045) > .01:
            failures.append('Upper vehicle did not descend to ground height')
    trace_path = output.with_name(output.name+f'_{case}_trace.json')
    save(trace_path, {'fixture_type': 'privileged_native_bridge_geometry_probe',
                      'policy_evaluation': False, 'initial_states': initial, 'samples': rows})
    result = {'case': case, 'passed': not failures, 'failures': sorted(set(failures)),
              'duration_s': len(rows)*DT, 'initial_s': starts, 'final_s': progress,
              'goal_s': goal, 'progress_m': progress[0]-starts[0],
              'max_height_m': max_height, 'final_heights_m': [s['z'] for s in states],
              'max_height_tracking_error_m': max_height_error, 'max_abs_cte_m': max_cte,
              'max_progress_step_m': max_progress_step, 'collision_steps': contact_steps,
              'minimum_xy_separation_m': min_xy,
              'minimum_crossing_height_gap_m': min_height_gap if math.isfinite(min_height_gap) else None,
              'trace_path': str(trace_path), 'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
              'screenshot': str(screenshot) if screenshot else None}
    print('BRIDGE_FIXTURE_CASE '+json.dumps(serial(result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['mujoco', 'isaac'], required=True)
    parser.add_argument('--track', default='suzuka')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=60.)
    parser.add_argument('--tag', default='bridge_fixture')
    args = parser.parse_args()
    track = Track(args.track)
    if not track.metadata.get('bridge'):
        parser.error('Track has no bridge metadata')
    output = ROOT/'output/racing'/args.engine/f'{track.id}_{args.tag}'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'fixture_type': 'privileged_native_bridge_geometry_probe',
              'policy_evaluation': False, 'sensor_only_policy_qualification': False,
              'scope': 'Contact geometry, continuous ramps and physical over/under clearance only',
              'controller': 'Privileged pose and ordered route projection; pure pursuit; native wheel velocity/front steering position actuators',
              'control_hz': 60, 'physics_hz': 480, 'engine': args.engine, 'track': track.id,
              'track_sha256': hashlib.sha256(track.path.read_bytes()).hexdigest(),
              'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'bridge': track.metadata['bridge'], 'cases': [], 'passed': False}
    env = None
    try:
        if args.engine == 'isaac':
            from racing.isaac_env import RaceIsaacEnv as Env
        else:
            from racing.mujoco_env import RaceMujocoEnv as Env
        env = Env(track, num_cars=2, render=args.render)
        for case in ('full_bridge', 'crossing'):
            report['cases'].append(run_case(env, case, output, max_seconds=args.max_seconds, render=args.render))
            save(output.with_suffix('.json'), report)
        if args.engine == 'mujoco':
            report['native_warning_counts'] = env.data.warning.number.tolist()
        else:
            report['native_warning_counts'] = None
            report['warning_scope'] = 'PhysX diagnostics are in process stdout/stderr; no warning-count API is assumed.'
        report['passed'] = all(c['passed'] for c in report['cases']) and not any(report['native_warning_counts'] or [])
    except Exception:
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        save(output.with_suffix('.json'), report)
        print('BRIDGE_FIXTURE_RESULT '+json.dumps({'path': str(output.with_suffix('.json')), 'passed': report['passed'],
                                                   'policy_evaluation': False}), flush=True)
        if env is not None:
            env.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
