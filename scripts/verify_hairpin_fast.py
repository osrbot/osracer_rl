#!/usr/bin/env python3
"""Independently audit saved high-speed runs; no simulator or policy imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/hairpin_fast'
DT = 1 / 60
LENGTH = 12 + np.pi * .8


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or bool(np.isfinite(value))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def project(xy):
    """Exact nearest projection on two finite straights and right semicircle."""
    x, y = xy.T
    angle = np.clip(np.arctan2(y - .8, x - 6), -np.pi / 2, np.pi / 2)
    candidates = np.stack([np.c_[np.clip(x, 0, 6), np.zeros(len(x))],
                           np.c_[6 + .8*np.cos(angle), .8 + .8*np.sin(angle)],
                           np.c_[np.clip(x, 0, 6), np.full(len(x), 1.6)]], axis=1)
    distance = np.linalg.norm(candidates - xy[:, None, :], axis=2)
    progress = np.stack([np.clip(x, 0, 6), 6 + .8*(angle + np.pi/2),
                         LENGTH - np.clip(x, 0, 6)], axis=1)
    index = np.argmin(distance, axis=1)
    return progress[np.arange(len(x)), index], distance.min(axis=1)


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def circle_fit(points):
    centered = points - points.mean(axis=0)
    matrix = np.c_[2*centered, np.ones(len(points))]
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(centered**2, axis=1), rcond=None)
    radius_squared = solution[2] + np.sum(solution[:2]**2)
    if rank < 3 or radius_squared <= 0:
        return {'reliable': False, 'reason': 'Degenerate circle fit'}
    center = solution[:2]
    radius = float(np.sqrt(radius_squared))
    residual = float(np.sqrt(np.mean((np.linalg.norm(centered-center, axis=1)-radius)**2)))
    angles = np.unwrap(np.arctan2(centered[:, 1]-center[1], centered[:, 0]-center[0]))
    coverage = float(np.ptp(angles))
    reliable = coverage >= .35 and residual <= .03 and residual/radius <= .1
    return dict(radius_m=radius, rms_radial_residual_m=residual,
                angle_coverage_rad=coverage, samples=len(points), reliable=bool(reliable),
                reliability_rule='>=8 samples, >=0.35 rad coverage, RMS<=0.03 m and <=10% radius',
                limitation='Least-squares rear-origin trajectory circle during transient drift; not a constant-radius maneuver or a proven minimum turning radius.')


def audit_trace(rows):
    if not rows or not finite(rows):
        raise ValueError('Missing, empty, or nonfinite trace')
    xy = np.array([[r['x'], r['y']] for r in rows])
    velocity = np.array([[r['vx'], r['vy']] for r in rows])
    speed = np.linalg.norm(velocity, axis=1)
    yaw = np.array([r['yaw'] for r in rows])
    # Independent planar rear-origin velocity direction; stored beta additionally
    # includes roll/pitch and unavailable vertical velocity.
    beta = wrap(np.arctan2(velocity[:, 1], velocity[:, 0]) - yaw)
    progress, cte = project(xy)
    time = np.array([r['t'] for r in rows])
    steering = np.asarray([r['steering_targets'] for r in rows])
    turn = (progress >= 6) & (progress <= 6 + np.pi*.8)
    apex = np.abs(progress - (6 + np.pi*.4)) <= .25
    entry = progress < 6
    qualifying = (beta < -.35) & (speed > 1.2) & (progress > 5.5) & (progress < 8.51)
    segments = runs(qualifying)
    longest = max((b-a for a, b in segments), default=0)*DT
    completed = np.linalg.norm(xy[-1] - [0, 1.6]) < .3 and abs(wrap(yaw[-1]-np.pi)) < .2 and progress[-1] > LENGTH-.2
    noflip = all(abs(r['roll']) <= .5 and abs(r['pitch']) <= .5 and r['z'] >= .015 for r in rows)
    checks = dict(completed=bool(completed), entry_at_least_5_m_s=bool(np.any(entry) and speed[entry].max() >= 5),
                  qualifying_drift_at_least_point1_s=bool(qualifying.sum()*DT >= .1-1e-9),
                  noflip=noflip, finite=True,
                  steering_targets_within_assumed_stops=bool(steering.shape == (len(rows), 2) and np.max(np.abs(steering)) <= .450001),
                  timestamps_60_hz=bool(np.allclose(time, np.arange(1, len(rows)+1)*DT, atol=1e-6)),
                  no_lane_departure=bool(cte.max() <= .56))
    crossing = np.flatnonzero(progress >= 6)
    crossing_speed = None
    if len(crossing):
        j = int(crossing[0])
        crossing_speed = float(speed[j])
        if j and progress[j] > progress[j-1]:
            crossing_speed = float(np.interp(6, progress[j-1:j+1], speed[j-1:j+1]))
    fits = []
    for a, b in segments:
        if b-a >= 8:
            fit = circle_fit(xy[a:b]); fit.update(start_s=float(time[a]), end_s=float(time[b-1]))
            fits.append(fit)
    reliable = [f['radius_m'] for f in fits if f['reliable']]
    result = dict(passed=all(checks.values()), checks=checks, rows=len(rows), duration_s=len(rows)*DT,
                  endpoint_distance_m=float(np.linalg.norm(xy[-1]-[0, 1.6])), final_yaw_error_rad=float(wrap(yaw[-1]-np.pi)),
                  entry_peak_m_s=float(speed[entry].max()) if np.any(entry) else None,
                  speed_at_turn_entry_x6_progress6_m_s=crossing_speed,
                  turn_speed_range_m_s=[float(speed[turn].min()), float(speed[turn].max())] if np.any(turn) else None,
                  apex_speed_range_m_s=[float(speed[apex].min()), float(speed[apex].max())] if np.any(apex) else None,
                  apex_definition='Rear-origin progress within 0.25 m of semicircle midpoint',
                  minimum_rear_slip_rad=float(beta[turn & (speed > 1.2)].min()) if np.any(turn & (speed > 1.2)) else None,
                  qualifying_drift_duration_s=float(qualifying.sum()*DT), longest_continuous_drift_s=float(longest),
                  drift_threshold_rad=-.35, minimum_drift_speed_m_s=1.2,
                  beta_method='atan2(world rear-origin vy,vx)-body yaw, wrapped; planar independent calculation',
                  circle_fits=fits, minimum_reliable_fitted_trajectory_radius_m=min(reliable) if reliable else None,
                  radius_required_for_pass=False)
    return result


def main():
    report = dict(passed=False, checks={}, engines={}, errors=[])
    def check(name, value):
        report['checks'][name] = bool(value)
    evaluations = {}
    try:
        policy = read(OUT/'policy.json')
        sha = digest(OUT/'policy.json')
        report['checkpoint_sha256'] = sha
        check('policy_finite', finite(policy))
        check('policy_control_60_hz', policy.get('control_hz') == 60)
        for engine in ('isaac', 'mujoco'):
            data = report['engines'][engine] = {'traces': {}}
            try:
                evaluation = read(OUT/f'{engine}_evaluation.json')
                evaluations[engine] = evaluation
                check(f'{engine}_checkpoint_matches', evaluation.get('checkpoint_sha256') == sha)
                check(f'{engine}_evaluation_finite', finite(evaluation))
                seeds = [episode['seed'] for episode in evaluation['episodes']]
                check(f'{engine}_nominal_present', 0 in seeds)
                check(f'{engine}_heldout_present', any(seed >= 100 for seed in seeds))
                check(f'{engine}_unique_seeds', len(seeds) == len(set(seeds)))
                for seed in seeds:
                    rows = read(OUT/f'{engine}_trace_seed{seed}.json')
                    result = audit_trace(rows)
                    data['traces'][str(seed)] = result
                    check(f'{engine}_seed{seed}_independent_success', result['passed'])
                nominal = read(OUT/f'{engine}_trace_seed0.json')
                recorded = read(OUT/f'{engine}_recorded_trace.json')
                data['recorded'] = audit_trace(recorded)
                check(f'{engine}_recorded_independent_success', data['recorded']['passed'])
                n = min(len(nominal), len(recorded))
                deviation = float(np.max(np.linalg.norm(np.array([[r['x'], r['y']] for r in nominal[:n]])-
                                                        np.array([[r['x'], r['y']] for r in recorded[:n]]), axis=1)))
                data['recording_xy_max_difference_m'] = deviation
                check(f'{engine}_recording_xy_consistent', deviation <= .02)
                check(f'{engine}_recording_duration_consistent', abs(len(nominal)-len(recorded))*DT <= .1)
                video_path = Path(evaluation['video']['path'])
                if not video_path.is_absolute():
                    video_path = ROOT/video_path
                probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration',
                    '-of', 'json', str(video_path)], capture_output=True, text=True, timeout=120)
                if probe.returncode:
                    raise ValueError(f'ffprobe failed: {probe.stderr}')
                stream = json.loads(probe.stdout)['streams'][0]
                data['video_probe'] = stream
                fps_parts = stream['r_frame_rate'].split('/')
                fps = float(fps_parts[0])/float(fps_parts[1])
                expected_frames = (len(recorded)+1)//2
                check(f'{engine}_video_30_hz', abs(fps-30) < 1e-6)
                check(f'{engine}_video_frame_count', int(stream['nb_read_frames']) == expected_frames)
                check(f'{engine}_video_duration', abs(float(stream['duration'])-expected_frames/30) <= .001 and
                      abs(float(stream['duration'])-len(recorded)*DT) <= DT+.001)
                check(f'{engine}_video_hash', digest(video_path) == evaluation['video']['sha256'])
                decode = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video_path), '-f', 'null', '-'],
                                        capture_output=True, text=True, timeout=120)
                check(f'{engine}_video_decodes', decode.returncode == 0)
                data['decode_stderr'] = decode.stderr
            except Exception as error:
                check(f'{engine}_evidence_complete', False)
                report['errors'].append(f'{engine}: {type(error).__name__}: {error}')
        check('same_checkpoint_both_engines', len(evaluations) == 2 and all(v.get('checkpoint_sha256') == sha for v in evaluations.values()))
    except Exception as error:
        report['errors'].append(f'policy: {type(error).__name__}: {error}')
    try:
        original = read(ROOT/'output/usability/openusd/validation.json')
        check('original_usd_unchanged', digest(ROOT/'OSRACER/USD/osracer_description/robot.usd') == original['sha256'])
        original = read(ROOT/'output/usability/mujoco/report.json')
        for name in ('robot.xml', 'scene.xml'):
            check(f'original_{name}_unchanged', digest(ROOT/'OSRACER/MuJoCo/osracer_description'/name) == original['source_files_sha256'][name])
    except Exception as error:
        report['errors'].append(f'original assets: {type(error).__name__}: {error}')
    report['low_speed_theory'] = dict(wheelbase_m=.28764, track_m=.212, assumed_per_wheel_stop_rad=.45,
        rear_axle_center_minimum_radius_m=float(.28764/np.tan(.45)+.212/2),
        formula='wheelbase/tan(inner wheel stop)+track/2',
        limitation='Low-speed no-slip Ackermann theory, approximately 0.70147 m; assumed steering stops, no measured hardware limit. Not an achieved drift radius.')
    report['scope'] = 'Saved simulated high-speed entry and hairpin completion. Entry peak is not apex speed. Radius is optional and derived from rear-position trajectory, never speed/body-yaw-rate during drift. No hardware or tighter-radius target is established.'
    report['passed'] = not report['errors'] and bool(report['checks']) and all(report['checks'].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'passed': report['passed'], 'errors': report['errors'],
                      'failed_checks': [k for k, v in report['checks'].items() if not v]}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
