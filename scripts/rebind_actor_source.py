#!/usr/bin/env python3
"""Rebind an existing checkpoint's parameters to the current controller source.

Parameters, controls and engine provenance are copied verbatim; only the actor
version and module digests are recomputed. Behaviour can change because the
controller source changed, so the result is explicitly recorded as a
parameter-only migration and every lap must be qualified again.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from racing.policy_bundle import actor_source_hashes, actor_version
from racing.policy import RacingPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--note', default='')
    parser.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                        help='Override one named RacingPolicy parameter; repeatable')
    args = parser.parse_args()
    source = json.loads(args.checkpoint.read_text())
    identity = ('policy_version', 'actor_source_sha256', 'policy_source_sha256', 'runtime_source_sha256')
    spec = {key: value for key, value in source.items() if key not in identity}
    previous = {key: source.get(key) for key in identity}
    overrides = {}
    for item in args.set:
        name, _, raw = item.partition('=')
        if not name or not raw:
            parser.error(f'--set expects NAME=VALUE, got {item!r}')
        if name not in RacingPolicy.names:
            parser.error(f'Unknown parameter {name!r}; expected one of {", ".join(RacingPolicy.names)}')
        try:
            value = float(raw)
        except ValueError:
            parser.error(f'--set {item!r} is not a number')
        spec['parameters'][RacingPolicy.names.index(name)] = value
        overrides[name] = value
    spec['policy_version'] = actor_version(spec)
    spec['actor_source_sha256'] = actor_source_hashes(spec)
    spec['policy_source_sha256'] = hashlib.sha256((ROOT / 'racing/policy.py').read_bytes()).hexdigest()
    spec['rebound_from'] = {
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'policy_version': previous['policy_version'],
        'actor_source_sha256': previous['actor_source_sha256'],
        'policy_source_sha256': previous['policy_source_sha256'],
        'parameters_unchanged': True,
    }
    if args.note:
        spec['rebound_from']['note'] = args.note
    if overrides:
        spec['rebound_from']['parameter_overrides'] = overrides
    spec['qualification_status'] = ('Parameters migrated from the recorded checkpoint; the controller '
                                    'source changed, so no previous lap result transfers.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Rebound checkpoint: {args.output.resolve()}')


if __name__ == '__main__':
    main()
