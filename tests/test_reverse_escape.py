import unittest

import numpy as np

from racing.policy import (DT, ESCAPE_PROGRESS_REQUIRED, ESCAPE_SECONDS,
                           ESCAPE_STUCK_SECONDS, RacingPolicy)
from racing.policy_bundle import BLOCKED_ESCAPE_SECONDS, make_actor
from racing.sensors import LidarSensor


class OpenCorridor:
    def boundary_segments(self):
        return np.array([[[-10., -.75], [15., -.75]], [[-10., .75], [15., .75]]])


class DeadEnd:
    """Straight corridor with a wall 0.2 m ahead of the laser origin."""

    def boundary_segments(self):
        return np.array([[[-10., -.75], [15., -.75]], [[-10., .75], [15., .75]],
                         [[.2, -.75], [.2, .75]]])


def run(track, ticks=90):
    policy = RacingPolicy()
    sensor = LidarSensor(track)
    state = {'wheel_vel': np.zeros(4), 'steer_pos': np.zeros(2)}
    commands = []
    for tick in range(ticks):
        observation = sensor.observe(state, ([0., 0., .205], np.eye(3)), tick=tick)
        wheels, _ = policy.action(observation)
        commands.append(wheels.copy())
    return policy, commands


class TestReverseEscape(unittest.TestCase):
    def test_stationary_car_facing_a_wall_reverses_for_a_bounded_time(self):
        policy, commands = run(DeadEnd())
        reverse = [index for index, command in enumerate(commands) if command[0] < 0.]
        self.assertTrue(reverse, 'a car parked inside the emergency stop distance must reverse')
        self.assertGreaterEqual(reverse[0], round(ESCAPE_STUCK_SECONDS/DT)-1)
        self.assertLessEqual(len(reverse), round((ESCAPE_SECONDS+.1)/DT))
        for index in reverse:
            command = commands[index]
            # Unified sign convention: forward is [v, v, v, -v] with v > 0.
            self.assertTrue(np.all(command[:3] < 0.) and command[3] > 0.)
        self.assertEqual(policy.phase, 'reverse_escape' if reverse[-1] == len(commands)-1 else policy.phase)

    def test_open_corridor_never_commands_reverse(self):
        policy, commands = run(OpenCorridor())
        self.assertTrue(all(command[0] >= 0. for command in commands))
        self.assertNotEqual(policy.phase, 'reverse_escape')

    def test_external_request_enters_one_bounded_escape(self):
        policy = RacingPolicy()
        self.assertTrue(policy.request_reverse_escape())
        self.assertFalse(policy.request_reverse_escape())
        sensor = LidarSensor(OpenCorridor())
        wheels, _ = policy.action(sensor.observe({'wheel_vel': np.zeros(4), 'steer_pos': np.zeros(2)},
                                                 ([0., 0., .205], np.eye(3)), tick=0))
        self.assertTrue(np.all(wheels[:3] < 0.) and wheels[3] > 0.)

    def test_escape_requires_renewed_forward_progress(self):
        policy = RacingPolicy()
        self.assertTrue(policy.request_reverse_escape())
        policy.escape_remaining, policy.escape_cooldown = 0., 0.
        # Backing off again immediately is what drove a car backwards around
        # the circuit, so the manoeuvre stays locked until it drives forward.
        self.assertFalse(policy.request_reverse_escape())
        policy.escape_progress = ESCAPE_PROGRESS_REQUIRED
        self.assertTrue(policy.request_reverse_escape())

    def test_refused_escape_returns_its_allowance(self):
        policy = RacingPolicy()
        self.assertTrue(policy.request_reverse_escape())
        # The supervisor vetoes before the car moves: refunding keeps the
        # allowance for a later attempt instead of stranding the car.
        self.assertTrue(policy.refund_reverse_escape())
        self.assertEqual(policy.escapes, 0)
        self.assertEqual(policy.escape_progress, ESCAPE_PROGRESS_REQUIRED)
        self.assertTrue(policy.request_reverse_escape())
        self.assertTrue(policy.refund_reverse_escape())
        # Nothing is left to refund once the manoeuvre is no longer pending.
        self.assertFalse(policy.refund_reverse_escape())

    def test_refused_supervisor_action_asks_the_actor_to_back_off(self):
        actor = make_actor(spec={'actor_type': 'closed_loop', 'safety_supervisor': True})
        sensor = LidarSensor(DeadEnd())
        state = {'wheel_vel': np.zeros(4), 'steer_pos': np.zeros(2)}
        seen = []
        for tick in range(2*round(BLOCKED_ESCAPE_SECONDS/DT)+6):
            observation = sensor.observe(state, ([0., 0., .205], np.eye(3)), tick=tick)
            wheels, _ = actor.action(observation)
            if actor.supervisor.diagnostics.get('reversing'):
                seen.append(wheels.copy())
        self.assertTrue(seen, 'a supervisor that never releases the car must request a back-off')
        self.assertTrue(np.all(seen[0][:3] < 0.) and seen[0][3] > 0.)
