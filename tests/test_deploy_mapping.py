import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deploy/osracer_policy'))

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


def test_frozen_checkpoint_matches_the_running_source():
    checkpoint = ROOT/'output/racing/policies/frozen-candidates-g00c_v10c_lateral.json'
    actor, spec = load_actor(checkpoint, racing_root=ROOT)
    assert 'verified-escape' in spec['policy_version']
    weights = actor.p
    assert weights.shape == (12,)


def test_tampered_checkpoint_is_refused():
    checkpoint = ROOT/'output/racing/policies/frozen-candidates-g00c_v10c_lateral.json'
    import json
    spec = json.loads(checkpoint.read_text())
    spec['actor_source_sha256'] = dict(spec['actor_source_sha256'], **{'racing/policy.py': '0'*64})
    tampered = ROOT/'output/racing/policies/.tampered_checkpoint.json'
    tampered.write_text(json.dumps(spec))
    try:
        with pytest.raises(CheckpointMismatch):
            load_actor(tampered, racing_root=ROOT)
    finally:
        tampered.unlink()
