"""Load a frozen checkpoint and refuse to run if the source does not match."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
    spec = json.loads(Path(checkpoint_path).read_text())
    if 'parameters' not in spec or 'actor_type' not in spec:
        raise CheckpointMismatch(f'{checkpoint_path} is not an actor checkpoint')
    if racing_root is not None:
        import sys
        sys.path.insert(0, str(Path(racing_root).resolve()))
    from racing.policy_bundle import actor_source_hashes, make_actor
    expected = spec.get('actor_source_sha256') or {}
    actual = actor_source_hashes(spec)
    mismatched = {name: (expected.get(name), digest) for name, digest in actual.items()
                  if expected.get(name) != digest}
    if mismatched:
        raise CheckpointMismatch(
            'Controller source does not match the checkpoint: ' +
            ', '.join(f'{name} {was} != {now}' for name, (was, now) in mismatched.items()))
    return make_actor(spec['parameters'], spec), spec
