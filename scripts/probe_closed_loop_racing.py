#!/usr/bin/env python3
"""Isolated experimental native probe; never updates training or leaderboards."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', default='bahrain')
    parser.add_argument('--seconds', type=float, default=90.)
    parser.add_argument('--ratio', type=float, default=1.)
    parser.add_argument('--trigger', type=float, default=.30)
    parser.add_argument('--target-slip-deg', type=float, default=25.)
    parser.add_argument('--pulse', type=float, default=.4)
    parser.add_argument('--steer-bias', type=float, default=.1)
    parser.add_argument('--start-s', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default='v6_probe')
    parser.add_argument('--safety', action='store_true')
    args = parser.parse_args()
    run_probe(args)


def run_probe(args):
    from racing import run
    from racing.tracks import Track
    from racing.mujoco_env import RaceMujocoEnv
    from racing.policy_closed_loop import ClosedLoopRacingPolicy
    class ProbePolicy(ClosedLoopRacingPolicy):
        instances = []
        def __init__(self, parameters=None):
            super().__init__(parameters, target_slip_deg=args.target_slip_deg,
                             max_boost_seconds=args.pulse, steering_bias=args.steer_bias)
            self.feedback_rows = []
            self.supervisor = None
            if getattr(args, 'safety', False) and not self.instances:
                from racing.safety import LocalSafetySupervisor
                self.supervisor = LocalSafetySupervisor()
            self.instances.append(self)

        def action(self, observation):
            action = super().action(observation)
            if self.supervisor is not None:
                action = self.supervisor.filter(observation, *action)
                if self.supervisor.diagnostics.get('must_brake'):
                    self.abort_drift('wall_supervisor')
                self.feedback['safety'] = dict(self.supervisor.diagnostics)
            self.feedback_rows.append(dict(self.feedback, scan_timestamp=observation['lidar']['timestamp'],
                                           scan_age=observation['lidar']['age']))
            return action
    run.RacingPolicy = ProbePolicy  # In-memory binding, never an on-disk edit.
    params = ProbePolicy.initial.copy()
    params[-2:] = [args.ratio, args.trigger]
    out = ROOT/'output/racing/experimental'/args.tag
    out.mkdir(parents=True, exist_ok=True)
    runtime_names = ['policy.py', 'policy_closed_loop.py', 'odometry.py', 'run.py', 'metrics.py',
                     'sensors.py', 'tracks.py', 'mujoco_env.py', 'safety.py']
    runtime_bytes = {name: (ROOT/'racing'/name).read_bytes() for name in runtime_names}
    for name, payload in runtime_bytes.items():
        (out/name).write_bytes(payload)
    track = Track(args.track)
    (out/'track.json').write_bytes(track.path.read_bytes())
    (out/'probe_closed_loop_racing.py').write_bytes(Path(__file__).read_bytes())
    env = RaceMujocoEnv(track, render=False)
    try:
        result, rows = run.rollout(env, params, seconds=args.seconds, start_s=args.start_s, seed=getattr(args, 'seed', 0))
    finally:
        env.close()
    result.update(experimental=True, qualified_for_leaderboard=False,
                  policy_version=ProbePolicy.version, parameters=params.tolist(), engine='mujoco',
                  target_slip_deg=args.target_slip_deg, safety_supervisor=getattr(args, 'safety', False),
                  max_boost_seconds=args.pulse, steering_bias=args.steer_bias,
                  max_entry_body_speed_m_s=2.55, recovery_slip_threshold_deg=35.,
                  track_sha256=hashlib.sha256(track.path.read_bytes()).hexdigest(),
                  runtime_sha256={name: hashlib.sha256(payload).hexdigest()
                                  for name, payload in runtime_bytes.items()})
    (out/'trace.json').write_text(json.dumps(run.serial(rows)))
    (out/'feedback.json').write_text(json.dumps(run.serial(ProbePolicy.instances[0].feedback_rows)))
    (out/'result.json').write_text(json.dumps(run.serial(result), indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['experimental', 'completed', 'duration_s', 'failure',
                     'progress_m', 'effective_overtakes', 'drift_duration_s',
                     'max_continuous_drift_duration_s', 'max_rear_slip_deg']}), flush=True)
    return result


if __name__ == '__main__':
    main()
