"""Load a frozen checkpoint and refuse to run if the source does not match."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


class CheckpointMismatch(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_actor(checkpoint_path, racing_root=None):
    """Return (actor, spec). Raises CheckpointMismatch when hashes disagree.

    The controller modules come from the ``racing`` package (pure numpy). The
    recorded ``actor_source_sha256`` must match the modules that are actually
    imported, otherwise the vehicle would run different code than the one that
    produced the qualification evidence.
    """
    checkpoint_path = Path(checkpoint_path)
    if racing_root is not None:
        root = Path(racing_root).resolve()
        source_root = root / 'src'
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    if checkpoint_path.suffix == '.pt':
        try:
            from racing.control.neural_policy import NeuralPolicy, load_checkpoint
            from racing.paths import source_files
            _, spec = load_checkpoint(checkpoint_path)
        except (ImportError, RuntimeError, ValueError) as exc:
            raise CheckpointMismatch(f'cannot load PPO checkpoint: {exc}') from exc
        expected = spec.get('metadata', {}).get('source_sha256') or {}
        if not expected:
            raise CheckpointMismatch('PPO checkpoint does not contain frozen source hashes')
        actual = {name: sha256(path) for name, path in source_files().items()
                  if name in expected}
        mismatched = {name: (expected[name], actual.get(name)) for name in expected
                      if actual.get(name) != expected[name]}
        if mismatched:
            details = ', '.join(
                f'{name} {was} != {now}' for name, (was, now) in mismatched.items())
            raise CheckpointMismatch(
                'Controller source does not match the PPO checkpoint: ' + details)
        return NeuralPolicy(checkpoint_path), spec

    spec = json.loads(checkpoint_path.read_text())
    if 'parameters' not in spec or 'actor_type' not in spec:
        raise CheckpointMismatch(f'{checkpoint_path} is not an actor checkpoint')
    from racing.control.policy_bundle import actor_source_hashes, make_actor
    expected = spec.get('actor_source_sha256') or {}
    actual = actor_source_hashes(spec)
    mismatched = {name: (expected.get(name), digest) for name, digest in actual.items()
                  if expected.get(name) != digest}
    if mismatched:
        raise CheckpointMismatch(
            'Controller source does not match the checkpoint: ' +
            ', '.join(f'{name} {was} != {now}' for name, (was, now) in mismatched.items()))
    return make_actor(spec['parameters'], spec), spec
