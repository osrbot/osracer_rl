import unittest
import numpy as np
from racing.policy import RacingPolicy
from racing.policy_closed_loop import ClosedLoopRacingPolicy
from racing.sensors import LidarSensor


class Track:
    boundary_segments = np.array([[[-20., -.75], [20., -.75]], [[-20., .75], [20., .75]], [[5., -.75], [5., .75]]])


def observation():
    state = {'wheel_vel': np.full(4, 2./.045), 'steer_pos': np.array([.35, .35])}
    return LidarSensor(Track()).observe(state, ([0, 0, .2], np.eye(3)))


class MeasuredEstimate:
    def __init__(self):
        self.data = {'valid': True, 'confidence': .9, 'age': 0., 'lateral_speed': -.3,
                     'lidar_forward_speed': 2., 'forward_speed': 3.,
                     'observable_axes': {'longitudinal': True, 'lateral': True, 'yaw': True}}

    def reset(self):
        pass

    def update(self, _):
        return self.data


class TestClosedLoopPolicy(unittest.TestCase):
    def controller(self):
        parameters = RacingPolicy.initial.copy(); parameters[-2] = 1.5
        measured = MeasuredEstimate()
        return ClosedLoopRacingPolicy(parameters, odometry=measured, max_boost_seconds=.4), measured

    def test_delayed_initial_scan_keeps_feedback_json_finite(self):
        import json
        params = RacingPolicy.initial.copy(); params[-2] = 1.5
        actor = ClosedLoopRacingPolicy(params)
        sensor = LidarSensor(Track(), latency_s=.2)
        state = {'wheel_vel': np.full(4, 2./.045), 'steer_pos': np.zeros(2)}
        for _ in range(4):
            obs = sensor.observe(state, ([0, 0, .2], np.eye(3)))
            wheels, steering = actor.action(obs)
            self.assertFalse(actor.feedback['valid'])
            self.assertTrue(np.isfinite(wheels).all())
            self.assertTrue(np.isfinite(steering).all())
            json.dumps(actor.feedback, allow_nan=False)
            self.assertEqual(actor.feedback['beta_predicted'], 0.)

    def test_zero_ratio_exactly_preserves_baseline(self):
        base, closed = RacingPolicy(), ClosedLoopRacingPolicy()
        obs = observation()
        for i in range(60):
            obs['lidar']['timestamp'] = (i//4)*4/60
            obs['lidar']['age'] = i%4/60
            a, b = base.action(obs), closed.action(obs)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])

    def test_observable_lidar_forward_velocity_replaces_spinning_front_encoder(self):
        actor, measured = self.controller()
        measured.data.update(lidar_forward_speed=2.5, forward_speed=4., lateral_speed=-1.1)
        actor.action(observation())
        self.assertEqual(actor.feedback['forward_source'], 'lidar')
        self.assertAlmostEqual(actor.feedback['beta_estimate'], np.arctan2(-1.1, 2.5))
        measured.data['observable_axes']['longitudinal'] = False
        actor.action(observation())
        self.assertEqual(actor.feedback['forward_source'], 'front_encoders')
        self.assertAlmostEqual(actor.feedback['beta_estimate'], np.arctan2(-1.1, 4.))

    def test_lost_confidence_and_excessive_slip_immediately_remove_boost(self):
        for loss in ['confidence', 'slip']:
            actor, measured = self.controller()
            obs = observation(); actor.action(obs)
            self.assertGreater(actor.feedback['rear_ratio'], 0.)
            if loss == 'confidence':
                measured.data['confidence'] = .4
            else:
                measured.data['lateral_speed'] = -2.
            actor.action(obs)
            self.assertEqual(actor.feedback['rear_ratio'], 0.)
            self.assertEqual(actor.phase, 'drift_recover')

    def test_pulse_has_hard_time_limit_and_feedback_reduces_ratio(self):
        actor, measured = self.controller()
        obs = observation(); actor.action(obs)
        starting_ratio = actor.feedback['rear_ratio']
        measured.data['lateral_speed'] = -2*np.tan(np.deg2rad(28))
        actor.action(obs)
        self.assertLess(actor.feedback['rear_ratio'], starting_ratio)
        for _ in range(25):
            actor.action(obs)
        self.assertLessEqual(actor.boost_elapsed, .4+1e-9)
        self.assertEqual(actor.feedback['rear_ratio'], 0.)
        self.assertNotEqual(actor.phase, 'drift_boost')

    def test_boost_waits_for_body_braking_and_front_wheels_roll(self):
        actor, measured = self.controller()
        measured.data.update(lidar_forward_speed=3.3, lateral_speed=0., yaw_rate=1.)
        obs = observation(); actor.action(obs)
        self.assertEqual(actor.feedback['rear_ratio'], 0.)
        measured.data.update(lidar_forward_speed=2., lateral_speed=-.2)
        wheels, steering = actor.action(obs)
        self.assertGreater(actor.feedback['rear_ratio'], 0.)
        expected_x = 2.-np.array([.212/2, -.212/2])
        expected_y = -.2+.28764
        expected = expected_x*np.cos(steering)+expected_y*np.sin(steering)
        np.testing.assert_allclose(wheels[:2]*.045, expected)
        self.assertGreater(min(wheels[:2]*.045), actor.last_speed)

    def test_extended_experimental_bounds_do_not_mutate_v5(self):
        params = RacingPolicy.initial.copy(); params[-2] = 3.5
        actor = ClosedLoopRacingPolicy(params, odometry=MeasuredEstimate(),
                                      max_boost_seconds=.6, steering_bias=.14)
        obs = observation()
        for _ in range(30):
            actor.action(obs)
        self.assertEqual(actor.phase, 'drift_boost')
        self.assertGreater(actor.feedback['rear_ratio'], 2.5)
        self.assertEqual(RacingPolicy.upper[-2], 2.5)
        for _ in range(8):
            actor.action(obs)
        self.assertLessEqual(actor.boost_elapsed, .6+1e-9)
        self.assertEqual(actor.feedback['rear_ratio'], 0.)

    def test_countersteer_begins_below_target_without_disabling_induction(self):
        low, mlow = self.controller()
        high, mhigh = self.controller()
        mlow.data.update(lateral_speed=0.)
        mhigh.data.update(lateral_speed=-2*np.tan(np.deg2rad(12)))
        a = low.action(observation())
        b = high.action(observation())
        self.assertGreater(high.feedback['rear_ratio'], 0.)
        self.assertLess(float(np.mean(b[1])), float(np.mean(a[1])))
        self.assertLess(abs(high.feedback['beta_estimate']), high.target_slip)

    def test_predictive_guard_cuts_boost_before_delayed_angle_hits_limit(self):
        actor, measured = self.controller()
        obs = observation()
        measured.data['lateral_speed'] = -2*np.tan(np.deg2rad(5))
        actor.action(obs)
        self.assertGreater(actor.feedback['rear_ratio'], 0.)
        obs['lidar']['timestamp'] = 4/60
        measured.data['lateral_speed'] = -2*np.tan(np.deg2rad(20))
        actor.action(obs)
        self.assertAlmostEqual(abs(np.degrees(actor.feedback['beta_estimate'])), 20.)
        self.assertGreater(abs(np.degrees(actor.feedback['beta_predicted'])), 35.)
        self.assertEqual(actor.feedback['rear_ratio'], 0.)
        self.assertEqual(actor.closed_state, 'recover')

    def test_reversed_forward_velocity_cannot_fake_zero_slip_recovery(self):
        actor, measured = self.controller()
        obs = observation(); actor.action(obs)
        measured.data.update(lidar_forward_speed=-.2, lateral_speed=2.)
        for _ in range(40):
            actor.action(obs)
        self.assertGreater(np.degrees(actor.feedback['beta_estimate']), 90.)
        self.assertEqual(actor.closed_state, 'recover')
        self.assertEqual(actor.feedback['rear_ratio'], 0.)

    def test_yaw_rate_growth_cuts_pulse_before_angle_guard(self):
        actor, measured = self.controller()
        obs = observation()
        measured.data.update(lateral_speed=-2*np.tan(np.deg2rad(8)), yaw_rate=3.)
        actor.action(obs)
        obs['lidar']['timestamp'] = 4/60
        measured.data.update(lateral_speed=-2*np.tan(np.deg2rad(12)), yaw_rate=4.)
        actor.action(obs)
        self.assertLess(abs(actor.feedback['beta_predicted']), actor.target_slip)
        self.assertEqual(actor.closed_state, 'recover')
        self.assertEqual(actor.feedback['rear_ratio'], 0.)

    def test_target_attainment_latches_pulse_off_when_slip_falls(self):
        actor, measured = self.controller()
        obs = observation(); actor.action(obs)
        measured.data['lateral_speed'] = -2*np.tan(np.deg2rad(26))
        actor.action(obs)
        self.assertEqual(actor.closed_state, 'recover')
        self.assertEqual(actor.feedback['rear_ratio'], 0.)
        measured.data['lateral_speed'] = -2*np.tan(np.deg2rad(15))
        actor.action(obs)
        self.assertEqual(actor.closed_state, 'recover')
        self.assertEqual(actor.feedback['rear_ratio'], 0.)

    def test_public_safety_abort_requires_recovery_and_cooldown(self):
        actor, measured = self.controller()
        obs = observation(); actor.action(obs)
        actor.abort_drift('wall_supervisor')
        self.assertEqual(actor.closed_state, 'recover')
        self.assertEqual(actor.feedback['rear_ratio'], 0.)
        measured.data['lateral_speed'] = 0.
        for _ in range(10):
            actor.action(obs)
        self.assertGreater(actor.closed_cooldown, 0.)
        self.assertEqual(actor.feedback['rear_ratio'], 0.)

    def test_idle_safety_abort_does_not_inject_recovery_countersteering(self):
        actor, measured = self.controller()
        base = RacingPolicy(actor.p.copy()); base.p[-2] = 0.
        measured.data.update(lidar_forward_speed=3., lateral_speed=-1.)
        obs = observation()
        actor.abort_drift('wall_supervisor')
        self.assertEqual(actor.closed_state, 'idle')
        self.assertGreater(actor.closed_cooldown, 0.)
        actual, expected = actor.action(obs), base.action(obs)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actor.closed_state, 'idle')

    def test_repeated_abort_does_not_restart_recovery_timer(self):
        actor, measured = self.controller()
        actor.closed_state = 'recover'
        actor.recovery_elapsed = .25
        actor.abort_drift('wall_supervisor')
        self.assertEqual(actor.recovery_elapsed, .25)

    def test_search_controls_keep_finite_limits_and_v6k_defaults(self):
        actor = ClosedLoopRacingPolicy()
        self.assertEqual(actor.boost_countersteer_gain, .2)
        self.assertEqual(actor.max_boost_seconds, .6)
        self.assertEqual(actor.yaw_rate_guard, 3.8)
        for name, (lower, upper) in actor.control_bounds.items():
            self.assertEqual(getattr(ClosedLoopRacingPolicy(**{name: -100}), name), lower)
            self.assertEqual(getattr(ClosedLoopRacingPolicy(**{name: 100}), name), upper)
            for bad in [float('nan'), float('inf'), -float('inf')]:
                with self.assertRaisesRegex(ValueError, 'must be finite'):
                    ClosedLoopRacingPolicy(**{name: bad})

    def test_yaw_guard_can_be_searched_without_disabling_slip_abort(self):
        actor, measured = self.controller()
        actor.yaw_rate_guard = 5.5
        obs = observation()
        measured.data.update(lateral_speed=-2*np.tan(np.deg2rad(8)), yaw_rate=3.)
        actor.action(obs)
        obs['lidar']['timestamp'] = 4/60
        measured.data.update(lateral_speed=-2*np.tan(np.deg2rad(12)), yaw_rate=4.)
        actor.action(obs)
        self.assertEqual(actor.closed_state, 'boost')
        measured.data.update(lateral_speed=-2*np.tan(np.deg2rad(36)), yaw_rate=4.)
        actor.action(obs)
        self.assertEqual(actor.closed_state, 'recover')
        self.assertEqual(actor.feedback['rear_ratio'], 0.)

    def test_one_second_pulse_is_still_hard_bounded(self):
        parameters = RacingPolicy.initial.copy(); parameters[-2] = 1.5
        measured = MeasuredEstimate()
        actor = ClosedLoopRacingPolicy(parameters, odometry=measured, max_boost_seconds=1.)
        obs = observation()
        for _ in range(45):
            actor.action(obs)
        self.assertEqual(actor.closed_state, 'boost')
        for _ in range(17):
            actor.action(obs)
        self.assertEqual(actor.closed_state, 'recover')
        self.assertLessEqual(actor.boost_elapsed, 1.+1e-9)
        self.assertEqual(actor.feedback['rear_ratio'], 0.)


if __name__ == '__main__':
    unittest.main()
