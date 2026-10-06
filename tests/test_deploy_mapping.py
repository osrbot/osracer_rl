import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deployment/ros2/osracer_policy'))

from osracer_policy.mapping import action_to_ackermann, build_observation, scan_to_packet  # noqa: E402
from osracer_policy.checkpoint import CheckpointMismatch, load_actor  # noqa: E402


class Scan:
    """Minimal LaserScan stand-in: 270 degrees at 0.25 degree spacing."""

    def __init__(self, fill=4.):
        self.angle_min = math.radians(-135.)
        self.angle_increment = math.radians(.25)
        self.range_min = .04
        self.range_max = 20.
        self.ranges = [fill]*1081


class Odom:
    class _Twist:
        class _Linear:
            x = 2.5
        linear = _Linear()
    twist = _Twist()


def test_scan_is_resampled_onto_the_training_grid():
    scan = Scan(fill=3.)
    scan.ranges[540] = float('inf')          # a dropped beam
    scan.ranges[541] = 30.                    # beyond the policy's range limit
    packet = scan_to_packet(scan, timestamp=1.25, age=.02)
    assert packet['angles'].shape == (361,)
    assert packet['ranges'].shape == (361,)
    assert abs(packet['angles'][0]+math.radians(135.)) < 1e-9
    assert abs(packet['angles'][-1]-math.radians(135.)) < 1e-9
    assert packet['frame_id'] == 'laser'
    assert packet['valid'].all()
    assert packet['validmask'].sum() == 361
    assert packet['timestamp'] == 1.25


def test_missing_beams_become_invalid_at_max_range():
    scan = Scan(fill=float('inf'))
    packet = scan_to_packet(scan, max_range=15.)
    assert not packet['valid'].any()
    assert np.allclose(packet['ranges'], 15.)


def test_observation_and_action_round_trip():
    observation = build_observation(Scan(), Odom(), [.1, -.1], tick=7, wheel_radius=.05)
    np.testing.assert_allclose(observation['wheel_vel'], 2.5/.05)
    np.testing.assert_allclose(observation['steer_pos'], [.1, -.1])
    speed, angle = action_to_ackermann([10., 10., 20., -20.], [.2, .4],
                                       wheel_radius=.05, max_speed=1.5, max_steering=.3)
    assert speed == pytest.approx(.5)        # front pair defines body speed
    assert angle == pytest.approx(.3)        # clamped to the configured limit


@pytest.fixture
def structured_checkpoint(tmp_path):
    """Synthetic checkpoint; a clean checkout needs no private training archive."""
    from racing.control.policy_bundle import actor_source_hashes, actor_version, make_actor
    spec = {'actor_type': 'reactive'}
    spec.update(parameters=make_actor(spec=spec).p.tolist(),
                policy_version=actor_version(spec),
                actor_source_sha256=actor_source_hashes(spec))
    checkpoint = tmp_path/'policy.json'
    checkpoint.write_text(json.dumps(spec))
    return checkpoint


def test_legacy_frozen_checkpoint_requires_explicit_source_migration(structured_checkpoint):
    checkpoint = structured_checkpoint
    from racing.control.policy_bundle import actor_source_hashes, actor_version
    spec = json.loads(checkpoint.read_text())
    # Emulate the pre-src-layout source names without redistributing trained weights.
    spec['actor_source_sha256'] = {
        name.replace('racing/control/', 'racing/'): digest
        for name, digest in spec['actor_source_sha256'].items()
    }
    checkpoint.write_text(json.dumps(spec))
    with pytest.raises(CheckpointMismatch):
        load_actor(checkpoint, racing_root=ROOT)

    spec['actor_source_sha256'] = actor_source_hashes(spec)
    migrated = checkpoint.with_name('migrated.json')
    migrated.write_text(json.dumps(spec))
    actor, spec = load_actor(migrated, racing_root=ROOT)
    assert spec['policy_version'] == actor_version(spec)
    weights = actor.p
    assert weights.shape == (12,)


def test_tampered_checkpoint_is_refused(structured_checkpoint):
    checkpoint = structured_checkpoint
    # Establish that the same generated bundle is accepted before tampering.
    load_actor(checkpoint, racing_root=ROOT)
    spec = json.loads(checkpoint.read_text())
    spec['actor_source_sha256'] = dict(spec['actor_source_sha256'], **{'racing/control/policy.py': '0'*64})
    tampered = checkpoint.with_name('tampered_checkpoint.json')
    tampered.write_text(json.dumps(spec))
    with pytest.raises(CheckpointMismatch):
        load_actor(tampered, racing_root=ROOT)


def test_ppo_checkpoint_loads_for_deployment(tmp_path):
    import hashlib
    from racing.control.neural_policy import ActorCritic, POLICY_VERSION, save_checkpoint
    from racing.paths import source_files
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in source_files().items()}
    checkpoint = tmp_path/'policy.pt'
    save_checkpoint(checkpoint, ActorCritic(), {
        'max_speed_m_s': 6., 'source_sha256': hashes,
    })
    actor, spec = load_actor(checkpoint, racing_root=ROOT)
    assert spec['algorithm'] == 'ppo'
    assert actor.version == POLICY_VERSION
