"""Independently audit saved native racing evidence without running its policy.

Usage: python3 -m racing.verify output/racing/mujoco/bahrain_evaluation.json
Failed driving episodes remain evidence; integrity and task success are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .tracks import Track

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DT = 1 / 60


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parameter_sha256(parameters):
    return hashlib.sha256(json.dumps(parameters, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def resolve_path(path, artifact):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [ROOT / path, Path(artifact).parent / path, Path.cwd() / path]
    return next((p.resolve() for p in candidates if p.exists()), candidates[0].resolve())


def _delta(current, previous, length):
    return (current - previous + length / 2) % length - length / 2


def _project_state(track, xy, state=None, s_hint=None):
    if np.max(getattr(track, 'elevations', [0.])) > 0:
        z = float(state['z']) if state is not None and 'z' in state else None
        return track.project(xy, z=z, s_hint=s_hint)
    return track.project(xy)


def audit_trace(result, rows, track):
    """Recompute metrics directly from body states; do not trust overlay values."""
    errors, limitations = [], []
    if not rows:
        return {'evidence_integrity_passed': False, 'task_success': False,
                'errors': ['Trace has no states'], 'limitations': [], 'recomputed': {}}
    times = np.asarray([r['t'] for r in rows], dtype=float)
    sample_dt = np.diff(np.r_[0., times])
    if not np.isfinite(times).all() or np.any(sample_dt <= 0):
        return {'evidence_integrity_passed': False, 'task_success': False,
                'errors': ['Trace timestamps must be finite and strictly increasing'],
                'limitations': [], 'recomputed': {}}
    full = bool(np.allclose(sample_dt, CONTROL_DT, atol=1e-8, rtol=0))
    if not full:
        limitations.append('Sparse trace: speed extrema and contacts between samples are not observable; durations use right-end sample integration and passes are sampled estimates.')
    if abs(times[-1] - float(result['duration_s'])) > CONTROL_DT / 2:
        errors.append('Terminal trace timestamp does not match episode duration')
    if np.any(np.abs(times / CONTROL_DT - np.round(times / CONTROL_DT)) > 1e-6):
        errors.append('Trace timestamps are not aligned to 60 Hz controls')
    starts = result.get('initial_states')
    if starts:
        previous_xy = [np.array([s['x'], s['y']], float) for s in starts]
        previous_s = [_project_state(track, xy, state)[0] for xy, state in zip(previous_xy, starts)]
    else:
        previous_xy = [np.asarray(track.at(float(result.get('start_s', 0.)) + 3*j)[0], float)
                       for j in range(len(rows[0]['states']))]
        previous_s = [_project_state(track, xy, s_hint=float(result.get('start_s', 0.))+3*j)[0]
                      for j, xy in enumerate(previous_xy)]
        limitations.append('Reset states were not saved; initial progress is anchored to requested starts, with 1 cm comparison tolerance.')
    if len(previous_xy) < 2:
        errors.append('Race requires at least two native vehicle states')
    progress = np.zeros(len(previous_xy))
    relative = np.array([_delta(previous_s[0], other, track.length) for other in previous_s[1:]])
    armed, persistence = relative < -.5, np.zeros(len(relative))
    speeds, slips, ctes, collisions, offroads = [], [], [], [], []
    displacements, progress_steps, pass_records = [], [], []
    trajectory_errors, jump_records = [], []
    unsafe_seen = False
    for index, (row, dt) in enumerate(zip(rows, sample_dt)):
        states = row['states']
        if len(states) != len(previous_xy):
            raise ValueError('Vehicle count changes inside trace')
        actions = row.get('actions')
        if actions is None or len(actions) != len(states):
            errors.append(f'Missing per-car actions at sample {index}')
        else:
            for action in actions:
                if len(action) != 2 or np.shape(action[0]) != (4,) or np.shape(action[1]) != (2,):
                    errors.append(f'Invalid wheel/steering action shape at sample {index}')
                elif not np.isfinite(np.r_[action[0], action[1]]).all():
                    errors.append(f'Nonfinite action at sample {index}')
        deltas, unsafe = [], []
        for j, state in enumerate(states):
            values = np.asarray([state[k] for k in ('x', 'y', 'vx', 'vy', 'yaw')], float)
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite body state at sample {index}, car {j}')
            xy = values[:2]
            s, cte, _ = _project_state(track, xy, state, s_hint=previous_s[j])
            distance = float(np.linalg.norm(xy - previous_xy[j]))
            ds = float(_delta(s, previous_s[j], track.length))
            speed = float(np.hypot(state['vx'], state['vy']))
            jump = distance > 30*dt + .15 or abs(ds) > 30*dt + .15
            # The runner also applies a more restrictive projection guard.
            if j == 0 and full and abs(ds) > max(.6, 2*speed*CONTROL_DT):
                jump = True
            if jump:
                jump_records.append({'sample': index, 'car': j, 't': float(row['t']),
                                     'displacement_m': distance, 'progress_delta_m': ds})
            collision = bool(state.get('collision', False))
            offroad = abs(cte) > track.width/2 - .125
            unsafe.append(collision or offroad or jump)
            deltas.append(ds)
            progress[j] += ds
            previous_xy[j], previous_s[j] = xy, s
            if 'time' in state and abs(float(state['time']) - row['t']) > 1e-6:
                errors.append(f'Native state time differs from trace time at sample {index}, car {j}')
            if j == 0:
                beta = math.atan2(math.sin(math.atan2(state['vy'], state['vx'])-state['yaw']),
                                  math.cos(math.atan2(state['vy'], state['vx'])-state['yaw'])) if speed > .1 else 0.
                if all(k in state for k in ('vz', 'roll', 'pitch')):
                    cy, sy = math.cos(state['yaw']), math.sin(state['yaw'])
                    cp, sp = math.cos(state['pitch']), math.sin(state['pitch'])
                    cr, sr = math.cos(state['roll']), math.sin(state['roll'])
                    forward = cy*cp*state['vx'] + sy*cp*state['vy'] - sp*state['vz']
                    lateral = (cy*sp*sr-sy*cr)*state['vx'] + (sy*sp*sr+cy*cr)*state['vy'] + cp*sr*state['vz']
                    beta = math.atan2(lateral, forward) if math.hypot(forward, lateral) > .1 else 0.
                speeds.append(speed); slips.append(beta); ctes.append(float(cte))
                collisions.append(collision); offroads.append(offroad)
                displacements.append(distance); progress_steps.append(ds)
                unsafe_seen |= unsafe[0]
                if 'progress' in row and abs(float(row['progress']) - progress[0]) > .01:
                    trajectory_errors.append(index)
        for j in range(len(relative)):
            relative[j] += deltas[0] - deltas[j+1]
            if unsafe[0] or unsafe[j+1]:
                armed[j], persistence[j] = False, 0.
            elif relative[j] < -.5:
                armed[j], persistence[j] = True, 0.
            elif armed[j] and relative[j] > .5:
                persistence[j] += dt
                if persistence[j] + 1e-10 >= .3:
                    pass_records.append({'opponent': j+1, 't': float(row['t'])})
                    armed[j], persistence[j] = False, 0.
            else:
                persistence[j] = 0.
    if trajectory_errors:
        errors.append(f'Overlay progress disagrees with independent projection at {len(trajectory_errors)} samples')
    if jump_records:
        errors.append(f'{len(jump_records)} implausible native trajectory/projection jumps')
    speeds, slips = np.asarray(speeds), np.asarray(slips)
    moving = speeds > 1.2
    drift = moving & (np.abs(slips) > np.deg2rad(20.))
    drift_duration = float(np.dot(drift, sample_dt))
    drift_longest = drift_run = 0.
    drift_events = 0
    for active, dt in zip(drift, sample_dt):
        previous_run = drift_run
        drift_run = drift_run + dt if active else 0.
        drift_longest = max(drift_longest, drift_run)
        if previous_run+1e-10 < .1 <= drift_run+1e-10:
            drift_events += 1
    collision_count, offroad_count = int(sum(collisions)), int(sum(offroads))
    independent = {
        'trace_samples': len(rows), 'full_control_trace': full,
        'sample_dt_min_s': float(min(sample_dt)), 'sample_dt_max_s': float(max(sample_dt)),
        'sample_dt_median_s': float(np.median(sample_dt)),
        'duration_s': float(times[-1]), 'progress_m': float(progress[0]),
        'lap_fraction': float(progress[0]/track.length),
        'peak_speed_m_s': float(max(speeds)),
        'mean_speed_m_s': float(np.average(speeds, weights=sample_dt)),
        'peak_position_difference_speed_m_s': float(np.max(np.asarray(displacements)/sample_dt)),
        'mean_position_difference_speed_m_s': float(sum(displacements)/times[-1]),
        'max_sample_displacement_m': float(max(displacements)),
        'max_abs_progress_step_m': float(max(np.abs(progress_steps))),
        'jump_records': jump_records,
        'max_abs_cte_m': float(max(np.abs(ctes))),
        'collision_samples': collision_count, 'offroad_samples': offroad_count,
        'max_rear_slip_deg_above_1_2_m_s': float(np.rad2deg(np.max(np.abs(slips[moving]), initial=0.))),
        'max_rear_slip_deg_all_speeds': float(np.rad2deg(max(np.abs(slips)))),
        'drift_duration_s': drift_duration, 'longest_contiguous_drift_s': float(drift_longest),
        'max_continuous_drift_duration_s': float(drift_longest),
        'drift_event_count': drift_events,
        'continuous_drift_qualified': bool(full and drift_longest+1e-10 >= .1),
        'effective_overtakes': len(pass_records), 'pass_records': pass_records,
    }
    if not all(all(k in r['states'][0] for k in ('vz', 'roll', 'pitch')) for r in rows):
        limitations.append('Slip uses planar world velocity and body yaw where vertical velocity is absent; small roll/pitch differences from the engine local-frame angle are possible.')
    comparisons = [('progress_m', independent['progress_m'], .01),
                   ('max_abs_cte_m', independent['max_abs_cte_m'], 1e-5)]
    if full:
        comparisons += [('peak_speed_m_s', independent['peak_speed_m_s'], 1e-6),
                        ('mean_speed_m_s', independent['mean_speed_m_s'], 1e-6),
                        ('collision_steps', collision_count, 0), ('offroad_steps', offroad_count, 0),
                        ('effective_overtakes', len(pass_records), 0),
                        ('drift_duration_s', drift_duration, CONTROL_DT + 1e-8),
                        ('max_continuous_drift_duration_s', drift_longest, CONTROL_DT + 1e-8),
                        ('drift_event_count', drift_events, 0),
                        ('continuous_drift_qualified', drift_longest+1e-10 >= .1, 0)]
    else:
        # Extrema in a sparse trace are lower bounds, not summary replacements.
        comparisons = comparisons[:1]
        for name, observed in [('peak_speed_m_s', max(speeds)), ('max_abs_cte_m', max(np.abs(ctes))),
                               ('collision_steps', collision_count), ('offroad_steps', offroad_count)]:
            if float(result.get(name, observed)) + 1e-6 < observed:
                errors.append(f'Reported {name} is below observed trace lower bound')
    for name, measured, tolerance in comparisons:
        if name in result and abs(float(result[name])-measured) > tolerance:
            errors.append(f'{name}: reported {result[name]}, recomputed {measured}')
    max_moving_key = 'max_rear_slip_deg_above_1_2_m_s'
    if full and max_moving_key in result and abs(float(result[max_moving_key])-independent[max_moving_key]) > .5:
        errors.append('Reported moving slip maximum differs by more than 0.5 degrees')
    if full and 'max_rear_slip_deg' in result:
        # Old artifacts explicitly reported an ungated maximum. New artifacts
        # declare the gate so stationary collision rotation is not called drift.
        expected_slip = independent[max_moving_key] if result.get('slip_speed_gate_m_s') == 1.2 else independent['max_rear_slip_deg_all_speeds']
        if abs(float(result['max_rear_slip_deg']) - expected_slip) > .5:
            errors.append('Reported rear slip maximum differs from independently reconstructed angle by more than 0.5 degrees')
    physically_complete = progress[0] >= track.length - .01
    if result.get('completed') and not physically_complete:
        errors.append('Completed lap claim lacks a full circuit of continuous progress')
    if result.get('valid_lap') and (unsafe_seen or not physically_complete):
        errors.append('Valid lap claim conflicts with trajectory/contact evidence')
    task_success = bool(full and physically_complete and not unsafe_seen and
                        result.get('completed') and result.get('valid_lap') and not result.get('failure'))
    return {'evidence_integrity_passed': not errors, 'task_success': task_success and not errors,
            'recorded_failure': result.get('failure'), 'errors': errors,
            'limitations': limitations, 'recomputed': independent}


def audit_video(metadata, artifact, episode_duration):
    errors = []
    path = resolve_path(metadata['path'], artifact)
    result = {'path': str(path), 'errors': errors}
    if not path.is_file():
        errors.append('Video file is missing')
        return result
    result['sha256'] = sha256(path)
    if metadata.get('sha256') and result['sha256'] != metadata['sha256']:
        errors.append('Video SHA-256 mismatch')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-count_frames', '-show_entries',
                            'stream=width,height,avg_frame_rate,nb_read_frames,duration:format=duration',
                            '-of', 'json', str(path)], capture_output=True, text=True, timeout=300)
    if probe.returncode:
        errors.append('ffprobe failed: ' + probe.stderr[-1000:])
        return result
    document = json.loads(probe.stdout)
    stream = document['streams'][0]
    numerator, denominator = stream['avg_frame_rate'].split('/')
    fps = float(numerator) / float(denominator)
    frames = int(stream['nb_read_frames'])
    duration = float(stream.get('duration', document['format']['duration']))
    result.update(width=stream['width'], height=stream['height'], fps=fps, frames=frames, duration_s=duration)
    if (stream['width'], stream['height']) != (1280, 720):
        errors.append('Video resolution is not 1280x720')
    if abs(fps-30.) > 1e-6:
        errors.append('Video is not 30 fps')
    if frames != int(metadata['frames']):
        errors.append('Video frame count differs from recorded metadata')
    if abs(duration-float(metadata['duration_s'])) > 1/30 + 1e-6:
        errors.append('Video duration differs from recorded metadata')
    if abs(duration-float(episode_duration)) > 1/30 + 1e-6:
        errors.append('Video playback duration does not match simulation duration at 1x')
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
                              '-f', 'null', '-'], capture_output=True, text=True, timeout=300)
    result['full_decode_passed'] = decoded.returncode == 0 and not decoded.stderr.strip()
    if not result['full_decode_passed']:
        errors.append('Full video decode failed: ' + decoded.stderr[-1000:])
    return result


def audit_artifact(path, checkpoint=None):
    path = Path(path).resolve()
    saved = json.loads(path.read_text())
    results = saved if isinstance(saved, list) else [saved]
    episodes = []
    for result in results:
        trace_path = path.with_name(f"{path.stem}_seed{result.get('seed', 0)}_trace.json")
        report = {'seed': result.get('seed'), 'evidence_integrity_passed': False,
                  'task_success': False, 'recorded_failure': result.get('failure'),
                  'errors': [], 'limitations': []}
        try:
            rows = json.loads(trace_path.read_text())
            track = Track(result['track'])
            report = audit_trace(result, rows, track)
            report.update(seed=result['seed'], trace_path=str(trace_path), trace_sha256=sha256(trace_path))
            if result.get('track_sha256'):
                report['track_sha256'] = sha256(track.path)
                if result['track_sha256'] != report['track_sha256']:
                    report['errors'].append('Track SHA-256 mismatch: saved geometry is required for this audit')
            if result.get('policy_source_sha256'):
                report['policy_source_sha256'] = result['policy_source_sha256']
                report['current_policy_source_matches'] = sha256(ROOT/'racing/policy.py') == result['policy_source_sha256']
                if not report['current_policy_source_matches']:
                    report['limitations'].append('Policy source has changed since recording; the saved source digest is retained, but policy replay was not performed.')
            if result.get('trace_sha256') and result['trace_sha256'] != report['trace_sha256']:
                report['errors'].append('Trace SHA-256 mismatch')
            params = result.get('parameters')
            if params is None:
                report['errors'].append('Episode parameters are missing')
            else:
                report['parameters_sha256'] = parameter_sha256(params)
                if result.get('parameters_sha256') and result['parameters_sha256'] != report['parameters_sha256']:
                    report['errors'].append('Parameter SHA-256 mismatch')
            checkpoint_path = checkpoint or result.get('checkpoint_path')
            if result.get('checkpoint_sha256') or checkpoint_path:
                cp = resolve_path(checkpoint_path, path) if checkpoint_path else ROOT/'output/racing/policy.json'
                if not cp.is_file():
                    report['errors'].append('Checkpoint provenance cannot be verified: file missing')
                else:
                    report['checkpoint_path'], report['checkpoint_sha256'] = str(cp), sha256(cp)
                    if result.get('checkpoint_sha256') and result['checkpoint_sha256'] != report['checkpoint_sha256']:
                        report['errors'].append('Checkpoint SHA-256 mismatch')
                    checkpoint_data=json.loads(cp.read_text())
                    cp_params = checkpoint_data.get('parameters')
                    if cp_params != params:
                        report['errors'].append('Saved episode parameters differ from checkpoint parameters')
                    if checkpoint_data.get('actor_source_sha256'):
                        if result.get('actor_source_sha256')!=checkpoint_data['actor_source_sha256']:
                            report['errors'].append('Actor bundle module hashes differ from checkpoint')
                        from .policy_bundle import configuration
                        if configuration(result)!=configuration(checkpoint_data):
                            report['errors'].append('Actor configuration differs from checkpoint')
            else:
                report['limitations'].append('No checkpoint was declared; parameter values are hashed, but no external checkpoint identity can be established.')
            if result.get('video'):
                report['video'] = audit_video(result['video'], path, result['duration_s'])
                report['errors'].extend(report['video']['errors'])
            else:
                report['video'] = {'present': False}
                report['limitations'].append('No video was recorded for this episode.')
            report['evidence_integrity_passed'] = not report['errors']
            report['task_success'] &= report['evidence_integrity_passed']
        except Exception as exc:
            report['evidence_integrity_passed'] = report['task_success'] = False
            report['errors'].append(f'{type(exc).__name__}: {exc}')
        episodes.append(report)
    return {'schema_version': 1, 'artifact': str(path), 'artifact_sha256': sha256(path),
            'evidence_integrity_passed': bool(episodes) and all(r['evidence_integrity_passed'] for r in episodes),
            'task_success': bool(episodes) and all(r['task_success'] for r in episodes),
            'episodes': episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+')
    parser.add_argument('--checkpoint', help='Exact checkpoint file, if its path was not saved')
    args = parser.parse_args()
    integrity = True
    for artifact in args.artifacts:
        path = Path(artifact)
        try:
            report = audit_artifact(path, checkpoint=args.checkpoint)
        except Exception as exc:
            report = {'artifact': str(path), 'evidence_integrity_passed': False,
                      'task_success': False, 'errors': [f'{type(exc).__name__}: {exc}']}
        output = path.with_name(path.stem + '_verification.json')
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        integrity &= report['evidence_integrity_passed']
        print(json.dumps({'verification': str(output), 'evidence_integrity_passed': report['evidence_integrity_passed'],
                          'task_success': report['task_success']}))
    raise SystemExit(0 if integrity else 1)


if __name__ == '__main__':
    main()
