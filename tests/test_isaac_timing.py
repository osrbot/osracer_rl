"""Control and reset durations must survive physics substep refinement."""
from unittest.mock import Mock

import numpy as np
import pytest

from racing.isaac_env import RaceIsaacEnv


class StopNativeStartup(Exception):
    pass


def configured_env(monkeypatch, physics_hz=None):
    """Use the real timing setup, then stop before importing the native runtime."""
    def stop():
        raise StopNativeStartup
    monkeypatch.setattr(RaceIsaacEnv, 'create_app', staticmethod(stop))
    env = RaceIsaacEnv.__new__(RaceIsaacEnv)
    with pytest.raises(StopNativeStartup):
        if physics_hz is None:
            env.__init__(None)
        else:
            env.__init__(None, physics_hz=physics_hz)
    env.world = Mock()
    env._target = Mock()
    env.states = Mock(return_value=['state'])
    env.contacts = np.zeros(1, dtype=int)
    env.car_contacts = env.contacts.copy()
    env.wall_contacts = env.contacts.copy()
    return env


@pytest.mark.parametrize('physics_hz', [None, 480, 1920])
def test_one_action_keeps_sixtieth_second_duration(monkeypatch, physics_hz):
    env = configured_env(monkeypatch, physics_hz)
    actions = [([1, 2, 3, -4], [.1, .2])]
    assert env.step(actions) == ['state']
    env._target.assert_called_once_with(0, *actions[0])
    assert env.world.step.call_count / env.physics_hz == pytest.approx(1 / 60)
    if physics_hz is None:
        assert env.world.step.call_count == 8  # Historical production default.


@pytest.mark.parametrize('physics_hz', [480, 1920])
def test_reset_preserves_real_settling_duration(monkeypatch, physics_hz):
    env = configured_env(monkeypatch, physics_hz)
    env.track = Mock(has_elevation=False)
    env.track.at.return_value = (np.array([1., 2.]), .3)
    robot = Mock(num_dof=6)
    env.robots = [robot]
    env.num_cars = 1
    env.contacts = np.ones(1, dtype=int)
    env.car_contacts = env.contacts.copy()
    env.wall_contacts = env.contacts.copy()
    env.reset()
    assert env.world.step.call_count / physics_hz == pytest.approx(200 / 480)
    assert not env.contacts.any()
    robot.set_world_velocity.assert_called_once()


@pytest.mark.parametrize('physics_hz', [0, -480, 481, 479, True, 480.5, '480'])
def test_invalid_rate_fails_before_starting_native_app(monkeypatch, physics_hz):
    app = Mock()
    monkeypatch.setattr(RaceIsaacEnv, 'create_app', app)
    with pytest.raises(ValueError, match='60 Hz'):
        RaceIsaacEnv(None, physics_hz=physics_hz)
    app.assert_not_called()
