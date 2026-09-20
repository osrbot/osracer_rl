#!/usr/bin/env python3
"""Experimental native v5+wall-supervisor rollout; the opponent stays plain v5.

Uses racing.run.rollout unchanged. A process-local role factory wraps only its
first (ego) policy construction and restores the original factory afterward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from racing import run as runner
from racing.policy import RacingPolicy as BasePolicy
from racing.safety import LocalSafetySupervisor
from racing.tracks import Track


class SupervisedEgo(BasePolicy):
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.supervisor = LocalSafetySupervisor()
        self.safety_trace = []

    def action(self, observation):
        proposed = super().action(observation)
        action = self.supervisor.filter(observation, *proposed)
        self.safety_trace.append(dict(self.supervisor.diagnostics, proposed_actions=runner.serial(proposed)))
        if self.supervisor.diagnostics['intervened']:
            self.phase = 'wall_safety_braking'
        return action


class RoleFactory:
    """Exactly ego then opponent, matching the production rollout constructor order."""
    def __init__(self):
        self.instances = []

    def __call__(self, parameters=None):
        if len(self.instances) >= 2:
            raise RuntimeError('Unexpected additional policy construction in rollout')
        policy = SupervisedEgo(parameters) if not self.instances else BasePolicy(parameters)
        self.instances.append(policy)
        return policy


class ProgressLogEnv:
    def __init__(self, native):
        self.native, self.track, self.tick = native, native.track, 0

    def reset(self, *args, **kwargs):
        self.tick = 0
        return self.native.reset(*args, **kwargs)

    def step(self, actions):
        states = self.native.step(actions)
        self.tick += 1
        if self.tick % 600 == 0:
            s = self.track.project([states[0]['x'], states[0]['y'], states[0]['z']])[0]
            print('SAFETY_PROBE_PROGRESS '+json.dumps({'time_s': self.tick/60, 'route_s': s,
                                                       'speed_m_s': float(np.hypot(states[0]['vx'], states[0]['vy']))}), flush=True)
        return states

    def __getattr__(self, name):
        return getattr(self.native, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['isaac', 'mujoco'], default='isaac')
    parser.add_argument('--track', default='monaco')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'output/racing/baseline_v5_policy.json')
    parser.add_argument('--seconds', type=float, default=200.)
    parser.add_argument('--heldout-seed', type=int, default=74)
    parser.add_argument('--primary-seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+', help='Explicit seeds; overrides the primary/heldout pair')
    parser.add_argument('--split', choices=['development', 'heldout'], default='development')
    parser.add_argument('--tag', default='safety_v1')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    checkpoint_bytes = checkpoint.read_bytes()
    parameters = np.asarray(json.loads(checkpoint_bytes)['parameters'], float)
    track = Track(args.track)
    directory = ROOT/'output/racing'/args.engine
    artifact = directory/f'{track.id}_{args.tag}.json'
    if artifact.exists():
        parser.error(f'{artifact} already exists; choose a new --tag to preserve prior evidence')
    snapshots = ROOT/'output/racing/experiments'/f'{args.engine}_{track.id}_{args.tag}'/'source'
    sources = ['racing/run.py', 'racing/policy.py', 'racing/safety.py', 'racing/sensors.py',
               'racing/metrics.py', 'racing/tracks.py', 'racing/safety_motion.py', 'racing/odometry.py', f'racing/{args.engine}_env.py',
               'scripts/probe_racing_safety.py']
    hashes = {}
    for name in sources:
        data = (ROOT/name).read_bytes()
        destination = snapshots/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    results, native = [], None
    try:
        if args.engine == 'isaac':
            from racing.isaac_env import RaceIsaacEnv as Env
        else:
            from racing.mujoco_env import RaceMujocoEnv as Env
        native = Env(track, num_cars=2, render=False)
        env = ProgressLogEnv(native)
        seeds = args.seeds if args.seeds else [args.primary_seed, args.heldout_seed]
        for seed in seeds:
            factory, original = RoleFactory(), runner.RacingPolicy
            runner.RacingPolicy = factory
            try:
                result, rows = runner.rollout(env, parameters, seed=seed, seconds=args.seconds,
                                               opponent_speed=2.8, opponent_gap=3.)
            finally:
                runner.RacingPolicy = original
            ego, opponent = factory.instances
            if isinstance(opponent, SupervisedEgo):
                raise AssertionError('Opponent must remain original v5')
            for row, diagnostics in zip(rows, ego.safety_trace):
                row['safety'] = diagnostics
            trace = directory/f'{track.id}_{args.tag}_seed{seed}_trace.json'
            runner.save(trace, rows)
            result.update(engine=args.engine, parameters=parameters.tolist(),
                          parameters_sha256=hashlib.sha256(json.dumps(parameters.tolist(), separators=(',', ':')).encode()).hexdigest(),
                          checkpoint_path=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                          track_sha256=hashlib.sha256(track.path.read_bytes()).hexdigest(),
                          trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                          policy_version=f'gap-center-v5 + experimental local wall safety ({args.tag})',
                          policy_source_sha256=hashes['racing/policy.py'], runtime_source_sha256=hashes,
                          source_snapshot_directory=str(snapshots),
                          actor_observation=['wheel_vel', 'steer_pos', 'lidar'],
                          role_factory='first ego: v5+LocalSafetySupervisor; second opponent: original v5, capped 2.8 m/s',
                          safety_intervention_frames=sum(d['intervened'] for d in ego.safety_trace),
                          evaluation_split=args.split,
                          experimental=True)
            results.append(result)
            runner.save(artifact, results)
            print('SAFETY_PROBE_RESULT '+json.dumps({k: result[k] for k in
                    ['seed', 'completed', 'valid_lap', 'failure', 'duration_s', 'progress_m',
                     'effective_overtakes', 'collision_steps', 'offroad_steps', 'safety_intervention_frames']}), flush=True)
            if not result['valid_lap'] and not args.seeds:
                print('SAFETY_PROBE_STOP: failed seed; remaining heldout runs skipped', flush=True)
                break
    except Exception:
        runner.save(artifact.with_name(artifact.stem+'_error.json'), {'error': traceback.format_exc(), 'completed_results': results})
        raise
    finally:
        if native is not None:
            native.close()
    raise SystemExit(0 if len(results) == len(seeds) and all(r['valid_lap'] for r in results) else 1)


if __name__ == '__main__':
    main()
