#!/usr/bin/env python3
"""Audit saved hairpin evidence without rerunning either simulator."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/hairpin'
SIGNS = np.array([1., 1., 1., -1.])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def lane_margin(rows):
    # Exact nearest distance to the two extended straight segments and right
    # semicircle; footprint boundary is sampled, not a rigorous swept CAD hull.
    u = np.linspace(0., 1., 41)
    xmin, xmax, halfwidth = -.0801, .3974, .127
    edges = np.concatenate([
        np.c_[xmin + (xmax-xmin)*u, np.full_like(u, side*halfwidth)]
        for side in (-1, 1)] + [
        np.c_[np.full_like(u, x), -halfwidth+2*halfwidth*u]
        for x in (xmin, xmax)])
    worst = 0.
    for row in rows:
        c, s = np.cos(row['yaw']), np.sin(row['yaw'])
        points = edges @ np.array([[c, s], [-s, c]]) + [row['x'], row['y']]
        x, y = points.T
        clipped = np.clip(x, -.55, 1.2)
        straight = np.minimum(np.hypot(x-clipped, y), np.hypot(x-clipped, y-1.6))
        dx, dy = x-1.2, y-.8
        angle = np.clip(np.arctan2(dy, dx), -np.pi/2, np.pi/2)
        curve = np.hypot(dx-.8*np.cos(angle), dy-.8*np.sin(angle))
        worst = max(worst, float(np.max(np.minimum(straight, curve))))
    return {'minimum_approximate_margin_m': .29-worst,
            'road_halfwidth_m': .29, 'rectangle_m': [xmin, xmax, -halfwidth, halfwidth],
            'samples_per_edge': 41,
            'limitation': 'Sampled neutral full-body rectangle at saved poses; steering changes wheel footprint. Approximation only, not rigorous CAD swept-volume or between-frame clearance.'}


def tracking(rows):
    wheel = np.asarray([r['wheel_vel'] for r in rows])
    wheel_target = np.asarray([r['wheel_targets'] for r in rows]) * SIGNS
    steer = np.asarray([r['steer_pos'] for r in rows])
    steer_target = np.asarray([r['steering_targets'] for r in rows])
    return {'max_abs_wheel_velocity_error_rad_s': float(np.max(np.abs(wheel-wheel_target))),
            'max_abs_steering_position_error_rad': float(np.max(np.abs(steer-steer_target))),
            'wheel_error_by_joint_rad_s': np.max(np.abs(wheel-wheel_target), axis=0).tolist(),
            'steering_error_by_joint_rad': np.max(np.abs(steer-steer_target), axis=0).tolist(),
            'wheel_order': ['LF', 'RF', 'LR', 'RR'],
            'note': 'Measured end-of-step encoders versus applied targets; wheel signs normalized forward. No tracking acceptance threshold imposed.'}


def main():
    report = {'passed': False, 'checks': {}, 'engines': {}, 'errors': [], 'warnings': []}
    def check(name, condition):
        report['checks'][name] = bool(condition)
    def read(path):
        return json.loads(path.read_text())
    try:
        policy_path = OUT/'policy.json'
        policy = read(policy_path)
        history = read(OUT/'training_history.json')
        sha = digest(policy_path)
        report['checkpoint_sha256'] = sha
        check('policy_and_training_history_finite', finite(policy) and finite(history))
        check('training_history_nonempty', isinstance(history, list) and len(history) > 0)
        evaluations = {}
        for engine in ('isaac', 'mujoco'):
            try:
                evaluation = read(OUT/f'{engine}_evaluation.json')
                evaluations[engine] = evaluation
                nominal = read(OUT/f'{engine}_trajectory.json')
                recorded = read(OUT/f'{engine}_recorded_trajectory.json')
                check(f'{engine}_checkpoint_matches', evaluation.get('checkpoint_sha256') == sha)
                check(f'{engine}_all_finite', finite(evaluation) and finite(nominal) and finite(recorded))
                if not report['checks'][f'{engine}_all_finite']:
                    raise ValueError('Nonfinite evaluation or trace values')
                episodes = evaluation['episodes']
                nominal_results = [e for e in episodes if e['seed'] == 0]
                heldout = [e for e in episodes if e['seed'] >= 100]
                check(f'{engine}_nominal_success', bool(nominal_results) and all(e['success'] for e in nominal_results))
                check(f'{engine}_heldout_success', bool(heldout) and all(e['success'] for e in heldout))
                check(f'{engine}_all_episode_success', bool(episodes) and all(e['success'] for e in episodes))
                check(f'{engine}_recorded_success', evaluation['recorded_episode']['success'])
                check(f'{engine}_traces_nonempty', bool(nominal) and bool(recorded))
                if not nominal or not recorded:
                    raise ValueError('Empty nominal or recorded trace')
                n = min(len(nominal), len(recorded))
                a = np.asarray([[r['x'], r['y']] for r in nominal[:n]])
                b = np.asarray([[r['x'], r['y']] for r in recorded[:n]])
                deviation = float(np.max(np.linalg.norm(a-b, axis=1)))
                duration_delta = abs(float(nominal[-1]['t'])-float(recorded[-1]['t']))
                check(f'{engine}_recording_xy_consistent', deviation <= .02)
                check(f'{engine}_recording_duration_consistent', duration_delta <= .1)
                data = {'nominal_frames': len(nominal), 'recorded_frames': len(recorded),
                        'trace_lengths_equal': len(nominal) == len(recorded),
                        'maximum_paired_rear_axle_xy_difference_m': deviation,
                        'duration_difference_s': duration_delta,
                        'comparison': 'Common frame indices; unmatched tail covered by duration check.',
                        'nominal_tracking': tracking(nominal), 'recorded_tracking': tracking(recorded),
                        'nominal_lane': lane_margin(nominal), 'recorded_lane': lane_margin(recorded)}
                report['engines'][engine] = data
                for kind in ('nominal_lane', 'recorded_lane'):
                    if data[kind]['minimum_approximate_margin_m'] < 0:
                        report['warnings'].append(f'{engine} {kind}: sampled rectangle extends beyond approximate road boundary')
                video_path = Path(evaluation['video']['path'])
                if not video_path.is_absolute():
                    video_path = ROOT/video_path
                probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration',
                    '-of', 'json', str(video_path)], capture_output=True, text=True)
                check(f'{engine}_ffprobe', probe.returncode == 0)
                if probe.returncode:
                    raise RuntimeError(probe.stderr)
                data['video_probe'] = json.loads(probe.stdout)
                stream = data['video_probe']['streams'][0]
                check(f'{engine}_video_frame_count', int(stream['nb_read_frames']) == len(recorded))
                check(f'{engine}_video_hash', digest(video_path) == evaluation['video']['sha256'])
                decode = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video_path),
                                         '-f', 'null', '-'], capture_output=True, text=True)
                data['decode_exit_code'] = decode.returncode
                data['decode_stderr'] = decode.stderr
                check(f'{engine}_video_decode', decode.returncode == 0)
            except Exception as error:
                report['errors'].append(f'{engine}: {type(error).__name__}: {error}')
        check('both_engine_checkpoint_hashes_identical', len(evaluations) == 2 and
              evaluations['isaac'].get('checkpoint_sha256') == evaluations['mujoco'].get('checkpoint_sha256') == sha)
    except Exception as error:
        report['errors'].append(f'checkpoint/training: {type(error).__name__}: {error}')
    for engine in ('openusd', 'mujoco'):
        try:
            if engine == 'openusd':
                original = read(ROOT/'output/usability/openusd/validation.json')
                check('original_usd_unchanged', digest(ROOT/'NEORACER/USD/osracer_description/robot.usd') == original['sha256'])
            else:
                original = read(ROOT/'output/usability/mujoco/report.json')
                for name in ('robot.xml', 'scene.xml'):
                    check(f'original_{name}_unchanged', digest(ROOT/'NEORACER/MuJoCo/osracer_description'/name) == original['source_files_sha256'][name])
        except Exception as error:
            report['errors'].append(f'original {engine}: {type(error).__name__}: {error}')
    report['passed'] = not report['errors'] and bool(report['checks']) and all(report['checks'].values())
    report['scope'] = 'Saved simulated hairpin completion, recording consistency, source integrity and media validation. Does not establish real-world transfer, full-vehicle lane containment, or stopped-at-finish behavior.'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'passed': report['passed'], 'errors': report['errors'], 'warnings': report['warnings']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
