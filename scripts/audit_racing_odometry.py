#!/usr/bin/env python3
"""Freeze native traces, reconstruct measured scans, audit local odometry.

Truth enters only LidarSensor's physical scene and the error evaluator. The
odometry call receives exactly the actor observation returned by observe().
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racing.tracks import Track
from racing.sensors import LidarSensor, ACTOR_KEYS
from racing.odometry import LidarOdometry

DT = 1/60


def digest(data):
    return hashlib.sha256(data).hexdigest()


def freeze_json(source, target):
    # A training writer may temporarily expose a partial JSON document. Never
    # audit that document or read it again after a successful frozen snapshot.
    for attempt in range(10):
        payload = Path(source).read_bytes()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(.1)
            continue
        target.write_bytes(payload)
        return parsed, {'source': str(Path(source).resolve()), 'snapshot': str(target.resolve()),
                        'sha256': digest(payload), 'bytes': len(payload)}


def truth(state):
    yaw, pitch, roll = [float(state.get(k, 0.)) for k in ('yaw', 'pitch', 'roll')]
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
    velocity = np.array([state['vx'], state['vy'], state.get('vz', 0.)])
    local = rotation.T@velocity
    return {'forward_speed': float(local[0]), 'lateral_speed': float(local[1]),
            'yaw_rate': float(state.get('yaw_rate', 0.)),
            'slip_angle': float(np.arctan2(local[1], local[0])),
            'speed': float(np.linalg.norm(velocity[:2])), 'pitch': pitch, 'z': float(state.get('z', 0.))}


def error(predicted, measured, angular=False):
    delta = predicted-measured
    return float(np.arctan2(np.sin(delta), np.cos(delta))) if angular else float(delta)


def summarize(rows, predicate):
    selected = [r for r in rows if predicate(r['truth'])]
    accepted = [r for r in selected if r['estimate']['valid']]
    summary = {'samples': len(selected), 'valid_samples': len(accepted),
               'valid_coverage': len(accepted)/len(selected) if selected else None}
    for label in ('instantaneous_errors', 'interval_errors'):
        stats = {}
        for field in ('lateral_speed', 'yaw_rate', 'slip_angle'):
            values = np.array([r[label][field] for r in accepted if r[label] is not None])
            stats[field] = {'rmse': float(np.sqrt(np.mean(values**2))) if len(values) else None,
                            'absolute_p95': float(np.percentile(abs(values), 95)) if len(values) else None,
                            'bias': float(np.mean(values)) if len(values) else None}
        summary[label] = stats
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--result', required=True, help='Episode result JSON or training policy JSON containing metrics')
    parser.add_argument('--track', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, trace_artifact = freeze_json(args.trace, output/'frozen_trace.json')
    source_result, result_artifact = freeze_json(args.result, output/'frozen_result.json')
    fixture = isinstance(rows, dict) and 'samples' in rows
    if fixture:
        native_initial, rows = rows['initial_states'], rows['samples']
        candidates = [r for r in source_result['cases'] if abs(r['duration_s']-rows[-1]['t']) < DT/2]
        if len(candidates) != 1:
            raise ValueError('Cannot uniquely match native fixture case')
        result = dict(candidates[0], initial_states=native_initial,
                      track_sha256=source_result['track_sha256'])
    elif isinstance(source_result, list):
        candidates = [r for r in source_result if abs(r['duration_s']-rows[-1]['t']) < DT/2]
        if len(candidates) != 1:
            raise ValueError('Cannot uniquely match result to trace duration')
        result = candidates[0]
    else:
        result = source_result.get('metrics', source_result)
    if abs(result['duration_s']-rows[-1]['t']) > DT/2:
        raise ValueError('Frozen training result and trace are from different candidates')
    intervals = np.diff(np.r_[0., [r['t'] for r in rows]])
    if not np.allclose(intervals, DT, atol=1e-8, rtol=0):
        raise ValueError('Exact 15 Hz sample hold reconstruction requires a complete 60 Hz trace')
    if not result.get('initial_states'):
        raise ValueError('Exact scan timing reconstruction requires native initial_states')
    track = Track(args.track)
    track_data = track.path.read_bytes()
    (output/'frozen_track.json').write_bytes(track_data)
    if result.get('track_sha256') and result['track_sha256'] != digest(track_data):
        raise ValueError('Track geometry changed since trace capture; refusing inconsistent reconstruction')
    spec = result.get('sensor', LidarSensor(track).specs)
    sensor = LidarSensor(track, dt=DT, scan_hz=spec['scan_hz'], ray_count=spec['rays'],
                         fov_deg=spec['fov_deg'], max_range=spec['max_range_m'],
                         min_range=spec['min_range_m'], wall_height=spec['wall_height_m'],
                         noise_std_m=spec.get('noise_std_m', 0.),
                         dropout_prob=spec.get('beam_dropout_probability', 0.),
                         latency_s=spec.get('requested_latency_s', 0.), seed=spec.get('noise_seed', 0))
    runtime_bytes = {name: (ROOT/'racing'/name).read_bytes() for name in ['sensors.py', 'odometry.py', 'tracks.py']}
    runtime = {name: digest(payload) for name, payload in runtime_bytes.items()}
    (output/'frozen_runtime').mkdir(exist_ok=True)
    for name, payload in runtime_bytes.items():
        (output/'frozen_runtime'/name).write_bytes(payload)
    odometry = LidarOdometry()
    samples = [(0., result['initial_states'])]+[(r['t'], r['states']) for r in rows]
    truths = [truth(states[0]) for _, states in samples]
    records, last_packet_time, last_interval_truth = [], None, None
    for tick, (timestamp, states) in enumerate(samples):
        obs = sensor.observe(states[0], states[0]['lidar_pose'], opponents=states[1:], tick=tick)
        assert set(obs) == ACTOR_KEYS
        estimate = odometry.update(obs)  # Only permitted measured actor inputs.
        packet_time = obs['lidar']['timestamp']
        scan_updated = packet_time != last_packet_time
        if scan_updated and estimate['dt'] > 0:
            end = round(packet_time/DT)
            start = round((packet_time-estimate['dt'])/DT)
            segment = truths[max(0, start+1):end+1]
            last_interval_truth = {k: float(np.mean([r[k] for r in segment]))
                                   for k in ('forward_speed', 'lateral_speed', 'yaw_rate')}
            last_interval_truth['slip_angle'] = float(np.arctan2(last_interval_truth['lateral_speed'], last_interval_truth['forward_speed']))
        if scan_updated:
            last_packet_time = packet_time
        gt = truths[tick]
        fields = ('lateral_speed', 'yaw_rate', 'slip_angle')
        records.append({'t': timestamp, 'scan_updated': scan_updated, 'truth': gt, 'estimate': estimate,
                        'instantaneous_errors': {k: error(estimate[k], gt[k], k == 'slip_angle') for k in fields},
                        'interval_errors': {k: error(estimate[k], last_interval_truth[k], k == 'slip_angle') for k in fields}
                        if last_interval_truth is not None else None})
    categories = {'all': lambda r: True, 'high_speed_ge_5_m_s': lambda r: r['speed'] >= 5.,
                  'moving_1_2_to_5_m_s': lambda r: 1.2 < r['speed'] < 5.,
                  'low_speed_le_1_2_m_s': lambda r: r['speed'] <= 1.2,
                  'drift_gt_20_deg_moving': lambda r: r['speed'] > 1.2 and abs(r['slip_angle']) > np.deg2rad(20),
                  'ramp_pitch_gt_2_deg': lambda r: abs(r['pitch']) > np.deg2rad(2),
                  'elevated_z_gt_0_15_m': lambda r: r['z'] > .15}
    report = {'schema_version': 1, 'engine': args.engine, 'track': track.id,
              'trace': trace_artifact, 'result': result_artifact, 'track_sha256': digest(track_data),
              'runtime_sha256': runtime, 'sensor': sensor.specs,
              'actor_input_keys': sorted(ACTOR_KEYS), 'source_episode_completed': result.get('completed'),
              'source_is_geometry_fixture': fixture,
              'source_fixture_scope': source_result.get('scope') if fixture else None,
              'reconstructed_sensor_specs_assumed': 'sensor' not in result,
              'units': {'lateral_speed': 'm/s', 'yaw_rate': 'rad/s', 'slip_angle': 'rad'},
              'buckets_60_hz_control': {name: summarize(records, predicate) for name, predicate in categories.items()},
              'buckets_15_hz_new_scans': {name: summarize([r for r in records if r['scan_updated']], predicate)
                                         for name, predicate in categories.items()},
              'limitations': ['Reconstructed scan geometry uses the recorded actual laser poses and opponent reflector boxes.',
                              'Truth is used only by the physical sensor and error evaluator, never by odometry.',
                              'Interval errors compare each estimate with its acquisition interval; instantaneous errors include 15 Hz hold latency.',
                              'Empty drift/ramp buckets are missing validation evidence, not successful validation.',
                              'Without recorded vz, reference local velocity assumes zero vertical velocity.']}
    if runtime != {name: digest((ROOT/'racing'/name).read_bytes()) for name in runtime}:
        raise RuntimeError('Odometry/sensor/track runtime changed during replay; rerun on a frozen implementation')
    (output/'samples.json').write_text(json.dumps(records, allow_nan=False))
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': str(output/'report.json'), 'all': report['buckets_15_hz_new_scans']['all']}))


if __name__ == '__main__':
    main()
