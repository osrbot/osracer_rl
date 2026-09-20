#!/usr/bin/env python3
"""Create an explicitly untrained, source-bound prior for joint CEM training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from racing.policy import RacingPolicy
from racing.policy_bundle import actor_source_hashes, actor_version, make_actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    parameters = RacingPolicy.initial.copy()
    parameters[-2] = 3.5
    spec = {
        'schema_version': 1,
        'algorithm': 'untrained structured sensor-feedback prior',
        'trained_in': [],
        'qualification_status': 'Untrained prior; no driving qualification claimed',
        'actor_type': 'closed_loop',
        'safety_supervisor': True,
        'parameters': parameters.tolist(),
        'controls': {'target_slip_deg': 30., 'max_boost_seconds': .6,
                     'steering_bias': .1, 'confidence_threshold': .6,
                     'boost_countersteer_gain': .2, 'yaw_rate_guard': 3.8},
    }
    actor = make_actor(parameters, spec)
    if not (actor.p == parameters).all():
        raise RuntimeError('Prior parameters were clipped; update the explicit prior')
    spec['policy_version'] = actor_version(spec)
    spec['actor_source_sha256'] = actor_source_hashes(spec)
    spec['policy_source_sha256'] = hashlib.sha256((ROOT/'racing/policy.py').read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Created untrained prior: {args.output.resolve()}')


if __name__ == '__main__':
    main()
