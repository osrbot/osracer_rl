#!/usr/bin/env python3
"""Compare explicit parameter variants on native tracks without retraining.

Each variant patches named RacingPolicy parameters of one immutable checkpoint.
The checkpoint identity is recorded for every episode, so a variant result can
never be mistaken for the frozen candidate it was derived from.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from racing.policy import RacingPolicy
from racing.policy_bundle import actor_source_hashes, actor_version, configuration
from racing.run import RUNTIME_SOURCE_SHA256, rollout, save
from racing.tracks import Track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['mujoco', 'isaac'], default='mujoco')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--variants', required=True, help='JSON list of {name, set:{param: value}}')
    parser.add_argument('--tracks', nargs='+', required=True)
    parser.add_argument('--seconds', type=float, default=200.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()

    data = json.loads(args.checkpoint.read_text())
    if 'actor_type' not in data:
        parser.error('Checkpoint must declare actor_type')
    base = np.array(data['parameters'], float)
    names = list(RacingPolicy.names)
    variants = json.loads(Path(args.variants).read_text())
    for variant in variants:
        unknown = set(variant['set']) - set(names)
        if unknown:
            parser.error(f'Unknown parameters in variant {variant["name"]}: {sorted(unknown)}')
    from racing.mujoco_env import RaceMujocoEnv
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    rows = []
    output = ROOT/'output/racing'/args.tag
    output.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        parameters = base.copy()
        for name, value in variant['set'].items():
            parameters[names.index(name)] = float(value)
        for track_id in args.tracks:
            track = Track(track_id)
            env = RaceMujocoEnv(track)
            try:
                result, trajectory = rollout(env, parameters, args.seed, args.seconds, actor_spec=data)
            finally:
                env.close()
            result.update(engine=args.engine, variant=variant['name'], variant_set=variant['set'],
                          track=track.id, parameters=parameters.tolist(),
                          checkpoint_path=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                          policy_version=actor_version(data), actor_source_sha256=actor_source_hashes(data),
                          runtime_source_sha256=RUNTIME_SOURCE_SHA256, **configuration(data))
            save(output/f'{track.id}_{variant["name"]}_seed{args.seed}_trace.json', trajectory)
            rows.append({k: result.get(k) for k in
                         ['track', 'variant', 'variant_set', 'valid_lap', 'lap_time_s', 'failure',
                          'progress_m', 'lap_fraction', 'peak_speed_m_s', 'mean_speed_m_s',
                          'effective_overtakes', 'max_continuous_drift_duration_s', 'collision_steps',
                          'offroad_steps', 'checkpoint_sha256', 'policy_version']})
            print('VARIANT', json.dumps(rows[-1]), flush=True)
            save(output/'summary.json', rows)
if __name__ == '__main__':
    main()
