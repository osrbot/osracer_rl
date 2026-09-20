import copy
import unittest

import numpy as np

from racing.safety import LocalSafetySupervisor, _wall_fits, _reflector_forward, _reflector_geometry
from racing.sensors import LidarSensor


def observation(right=-.75, left=.75, speed=3., slope=0.):
    class Corridor:
        def boundary_segments(self):
            return np.array([[[-10., right-10*slope], [15., right+15*slope]],
                             [[-10., left-10*slope], [15., left+15*slope]]])
    # Static source laser calibration. The actor receives only the resulting scan.
    w, x, y, z = .99969349188272216, -7.0832408137250425e-05, -.024589108572215627, .002879758622449709
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    sensor = LidarSensor(Corridor())
    return sensor.observe({'wheel_vel': [speed/.045]*4, 'steer_pos': [0., 0.]},
                           {'position': [.096555277688949018, 0., .205], 'rotation': rotation})


class ReadySafetySupervisor(LocalSafetySupervisor):
    """Geometry fixtures provide a prior scan; initialization is tested below."""
    def filter(self, obs, wheels, steers):
        if self.odometry.previous is None and np.isfinite(obs['lidar'].get('age', 0.)):
            previous = copy.deepcopy(obs)
            previous['lidar']['timestamp'] -= 1/15
            self.odometry.update(previous)
        return super().filter(obs, wheels, steers)


def assert_rolling_brake(test, wheels, measured_speed):
    """Danger braking must roll the wheels down, never lock them."""
    wheels = np.asarray(wheels, float)
    test.assertAlmostEqual(wheels[0], wheels[1])
    test.assertAlmostEqual(wheels[0], wheels[2])
    test.assertAlmostEqual(wheels[3], -wheels[0])
    test.assertGreater(wheels[0], 0.)
    test.assertLessEqual(abs(wheels[0])*.045, measured_speed+1e-9)


class TestLocalSafetySupervisor(unittest.TestCase):
    def test_open_corridor_keeps_native_actions(self):
        supervisor = ReadySafetySupervisor()
        wheels, steer = supervisor.filter(observation(), [60., 60., 60., -60.], [0., 0.])
        np.testing.assert_array_equal(wheels, [60., 60., 60., -60.])
        np.testing.assert_array_equal(steer, [0., 0.])
        self.assertFalse(supervisor.diagnostics['must_brake'])
        self.assertGreaterEqual(supervisor.diagnostics['wall_fit_count'], 2)

    def test_reverse_escape_requires_visible_clear_rear_quadrants(self):
        supervisor = ReadySafetySupervisor()
        wheels, _ = supervisor.filter(observation(speed=0.), [-15., -15., -15., 15.], [0., 0.])
        self.assertTrue(supervisor.diagnostics['reversing'])
        self.assertEqual(supervisor.diagnostics['reason'], 'verified_reverse_escape')
        np.testing.assert_array_equal(wheels, np.array([-1., -1., -1., 1.])*(.6/.045))
        self.assertTrue(supervisor.diagnostics['predicted_feasible'])

    def test_unverified_reverse_request_is_zeroed(self):
        supervisor = ReadySafetySupervisor()

        class DeadEnd:
            def boundary_segments(self):
                return np.array([[[-10., -.75], [15., -.75]], [[-10., .75], [15., .75]],
                                 [[-.2, -.75], [-.2, .75]]])

        w, x, y, z = .99969349188272216, -7.0832408137250425e-05, -.024589108572215627, .002879758622449709
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                             [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                             [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        obs = LidarSensor(DeadEnd()).observe({'wheel_vel': [0.]*4, 'steer_pos': [0., 0.]},
                                             {'position': [.096555277688949018, 0., .205], 'rotation': rotation})
        wheels, _ = supervisor.filter(obs, [-15., -15., -15., 15.], [0., 0.])
        self.assertFalse(supervisor.diagnostics['reversing'])
        self.assertEqual(supervisor.diagnostics['reason'], 'reverse_not_verified')
        np.testing.assert_array_equal(wheels, np.zeros(4))

    def test_near_wall_turn_brakes_and_selects_a_feasible_heading(self):
        supervisor = ReadySafetySupervisor()
        wheels, steer = supervisor.filter(observation(right=-.28, left=1.22, speed=3.5),
                                          [80., 80., 80., -80.], [-.35, -.35])
        self.assertTrue(supervisor.diagnostics['must_brake'])
        assert_rolling_brake(self, wheels, 3.5)
        self.assertGreater(supervisor.diagnostics['selected_clearance_m'], supervisor.diagnostics['requested_clearance_m'])
        self.assertGreater(np.mean(steer), -.1)
        self.assertTrue(supervisor.diagnostics['predicted_feasible'])

    def test_encoder_lock_does_not_immediately_erase_speed(self):
        supervisor = ReadySafetySupervisor()
        obs = observation(speed=4.)
        supervisor.filter(obs, [0.]*4, [0.]*2)
        obs['wheel_vel'] = np.zeros(4)
        supervisor.filter(obs, [0.]*4, [0.]*2)
        self.assertGreater(supervisor.diagnostics['speed_bound_m_s'], 3.9)

    def test_compact_reflector_is_not_a_wall(self):
        points = np.c_[np.linspace(.9, 1.1, 30), np.full(30, .35)]
        self.assertEqual(_wall_fits(points, np.ones(30, bool)), [])

    def test_stopped_inside_margin_can_creep_away_without_reducing_clearance(self):
        supervisor = ReadySafetySupervisor()
        obs = observation(right=-1.30, left=.21, speed=0., slope=-.12)
        obs['steer_pos'] = np.array([-.05, -.08])
        wheels, steer = supervisor.filter(obs,
                                          [40., 40., 40., -40.], [.35, .35])
        self.assertTrue(supervisor.diagnostics['recovering'])
        self.assertLess(np.mean(steer), 0.)
        self.assertAlmostEqual(wheels[0]*.045, .5)
        self.assertGreaterEqual(supervisor.diagnostics['selected_clearance_m'], 0.)

    def test_escape_does_not_drive_through_a_predicted_wall_overlap(self):
        supervisor = ReadySafetySupervisor()
        wheels, _ = supervisor.filter(observation(right=-1.38, left=.12, speed=0.),
                                      [40., 40., 40., -40.], [-.35, -.35])
        self.assertFalse(supervisor.diagnostics['recovering'])
        np.testing.assert_array_equal(wheels, np.zeros(4))

    def test_nearly_parallel_wall_accepts_small_positive_escape_progress(self):
        obs = observation(right=-1.309, left=.191, speed=0., slope=-.0123)
        obs['steer_pos'] = np.array([-.05, -.118])
        supervisor = ReadySafetySupervisor()
        supervisor.filter(obs, [40., 40., 40., -40.], [-.342, -.45])
        self.assertTrue(supervisor.diagnostics['recovering'])
        self.assertGreaterEqual(supervisor.diagnostics['selected_clearance_m'], .025)

    def test_initial_latency_packet_brakes_before_nonfinite_age_math(self):
        for age in [float('inf'), float('nan'), .2]:
            obs = observation(speed=3.)
            obs['lidar']['age'] = age
            obs['lidar']['valid'][:] = False
            supervisor = ReadySafetySupervisor()
            wheels, steer = supervisor.filter(obs, [40., 40., 40., -40.], [.3, .3])
            np.testing.assert_array_equal(wheels, np.zeros(4))
            self.assertTrue(np.isfinite(steer).all())
            self.assertEqual(supervisor.diagnostics['reason'], 'stale_or_empty_scan')

    def test_shallow_conservative_overlap_can_only_shrink_and_fully_clear(self):
        obs = observation(right=-1.30, left=.215, speed=0., slope=-.146)
        obs['steer_pos'] = np.array([-.05, -.118])
        supervisor = ReadySafetySupervisor()
        wheels, steering = supervisor.filter(obs, [40., 40., 40., -40.], [-.342, -.45])
        self.assertTrue(supervisor.diagnostics['preparing_steering'])
        np.testing.assert_array_equal(wheels, np.zeros(4))
        obs['steer_pos'] = steering
        supervisor.filter(obs, [40., 40., 40., -40.], [-.342, -.45])
        self.assertTrue(supervisor.diagnostics['recovering'])
        self.assertGreaterEqual(supervisor.diagnostics['selected_clearance_m'],
                                supervisor.diagnostics['requested_clearance_m']-1e-6)
        self.assertGreater(supervisor.diagnostics['recovery_end_clearance_m'], .03)

    def test_escape_rejects_initially_deeper_overlap_even_if_turning_away(self):
        supervisor = ReadySafetySupervisor()
        # Parallel close wall: turning away swings the rear corner into it.
        wheels, _ = supervisor.filter(observation(right=-1.345, left=.155, speed=0.),
                                      [40., 40., 40., -40.], [-.35, -.35])
        self.assertFalse(supervisor.diagnostics['recovering'])
        np.testing.assert_array_equal(wheels, np.zeros(4))

    def test_recovery_can_spend_positive_reserve_without_wall_intersection(self):
        obs = observation(right=-1.30, left=.19, speed=0., slope=.028)
        obs['steer_pos'] = np.array([-.05, -.24])
        supervisor = ReadySafetySupervisor()
        supervisor.filter(obs, [40., 40., 40., -40.], [-.342, -.45])
        self.assertTrue(supervisor.diagnostics['recovering'])
        self.assertGreater(supervisor.diagnostics['selected_clearance_m'], 0.)
        self.assertGreaterEqual(supervisor.diagnostics['selected_clearance_m'],
                                .5*supervisor.diagnostics['recovery_present_clearance_m']-1e-6)
        self.assertGreater(supervisor.diagnostics['recovery_end_clearance_m'],
                           supervisor.diagnostics['recovery_present_clearance_m'])

    def test_shallow_overlap_need_not_recover_full_reserve_in_one_horizon(self):
        obs = observation(right=-1.305, left=.195, speed=0., slope=-.105)
        obs['steer_pos'] = np.array([-.05, -.145])
        supervisor = ReadySafetySupervisor()
        wheels, steering = supervisor.filter(obs, [40., 40., 40., -40.], [-.342, -.45])
        self.assertTrue(supervisor.diagnostics['preparing_steering'])
        np.testing.assert_array_equal(wheels, np.zeros(4))
        obs['steer_pos'] = steering
        supervisor.filter(obs, [40., 40., 40., -40.], [-.342, -.45])
        self.assertTrue(supervisor.diagnostics['recovering'])
        self.assertGreater(supervisor.diagnostics['recovery_end_clearance_m'], .001)

    def test_parallel_wall_escape_accounts_for_rear_overhang(self):
        obs = observation(right=-1.315, left=.185, speed=0., slope=.0013)
        obs['steer_pos'] = np.array([-.025, -.025])
        supervisor = ReadySafetySupervisor()
        wheels, _ = supervisor.filter(obs, [40., 40., 40., -40.], [-.342, -.45])
        self.assertTrue(supervisor.diagnostics['recovering'])
        self.assertGreater(supervisor.diagnostics['recovery_end_clearance_m'],
                           supervisor.diagnostics['recovery_present_clearance_m'])
        self.assertAlmostEqual(wheels[0]*.045, .5)

    def test_compact_obstacle_inside_swept_body_is_not_missed_between_corners(self):
        supervisor = ReadySafetySupervisor()
        supervisor._obstacles = np.array([[.7, 0.]])
        self.assertLess(supervisor._predict([], 1., 0., 0.), 0.)
        supervisor._obstacles = np.array([[.7, 1.]])
        self.assertGreater(supervisor._predict([], 1., 0., 0.), .3)

    def test_compact_scan_cluster_is_available_to_overrides_without_replacing_nominal_vehicle_policy(self):
        obs = observation(speed=3.)
        angles = obs['lidar']['angles']
        hit = abs(angles) < np.arctan(.08/2.)
        obs['lidar']['ranges'][hit] = 2./np.cos(angles[hit])
        obs['lidar']['valid'][hit] = True
        supervisor = ReadySafetySupervisor()
        wheels, _ = supervisor.filter(obs, [60., 60., 60., -60.], [0., 0.])
        self.assertGreaterEqual(supervisor.diagnostics['compact_return_points'], 2)
        self.assertFalse(supervisor.diagnostics['must_brake'])
        np.testing.assert_array_equal(wheels, [60., 60., 60., -60.])
        self.assertGreater(supervisor._predict([], 3., 0., 0., horizon=.12), 0.)

    def test_full_l_shaped_reflector_preserves_measured_heading(self):
        p = np.r_[np.c_[np.linspace(.1, 0., 9), np.zeros(9)],
                  np.c_[np.zeros(12), np.linspace(.014, .16, 12)]]
        for yaw in [0., .3, -.8, 1.8]:
            c, s = np.cos(yaw), np.sin(yaw)
            direction = _reflector_forward(p@np.array([[c, s], [-s, c]])+[.3, .4])
            self.assertIsNotNone(direction)
            self.assertGreater(abs(direction@np.array([c, s])), .999)
            center, _ = _reflector_geometry(p@np.array([[c, s], [-s, c]])+[.3, .4])
            np.testing.assert_allclose(center, np.array([.05, .08])@np.array([[c, s], [-s, c]])+[.3, .4], atol=1e-10)

    def test_ordinary_short_wall_and_ambiguous_partial_face_are_not_oriented_cars(self):
        for length in [.08, .10, .44]:
            p = np.c_[np.linspace(0., length, 30), np.zeros(30)]
            self.assertIsNone(_reflector_forward(p))

    def test_close_confirmed_car_triggers_short_horizon_before_wall_risk(self):
        obs = observation(speed=3.)
        angles = obs['lidar']['angles']
        hit = abs(angles) < np.arctan(.08/.85)
        obs['lidar']['ranges'][hit] = .85/np.cos(angles[hit])
        obs['lidar']['valid'][hit] = True
        supervisor = ReadySafetySupervisor()
        wheels, _ = supervisor.filter(obs, [60., 60., 60., -60.], [0., 0.])
        self.assertTrue(supervisor.diagnostics['must_brake'])
        self.assertEqual(supervisor.diagnostics['reason'], 'near_term_vehicle_footprint')
        assert_rolling_brake(self, wheels, 3.)

    def test_measured_compact_heading_separates_longitudinal_and_lateral_padding(self):
        supervisor = ReadySafetySupervisor()
        supervisor._oriented_obstacles = np.array([[.16, .5, 1., 0.]])
        lateral_gap = supervisor._predict([], 0., 0., 0.)
        supervisor._oriented_obstacles = np.array([[.16, .5, 0., 1.]])
        longitudinal_gap = supervisor._predict([], 0., 0., 0.)
        self.assertAlmostEqual(lateral_gap, .20)
        self.assertAlmostEqual(longitudinal_gap, .04)

    def test_inputs_are_not_mutated_and_privileged_extras_are_ignored(self):
        obs = observation()
        original = copy.deepcopy(obs)
        supervisor = ReadySafetySupervisor()
        first = supervisor.filter(obs, [70., 70., 70., -70.], [.05, .05])
        np.testing.assert_array_equal(obs['lidar']['valid'], original['lidar']['valid'])
        np.testing.assert_array_equal(obs['lidar']['ranges'], original['lidar']['ranges'])
        injected = copy.deepcopy(original)
        injected.update(x=1e9, y=-1e9, yaw=2., body_slip=100., opponent_positions=[[0., 0.]])
        second = ReadySafetySupervisor().filter(injected, [70., 70., 70., -70.], [.05, .05])
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)



class TestScanMotionSafety(unittest.TestCase):
    def test_production_needs_two_scans_and_reset_discards_history(self):
        s = LocalSafetySupervisor()
        obs = observation()
        wheels = [60., 60., 60., -60.]
        first, _ = s.filter(obs, wheels, [0., 0.])
        np.testing.assert_array_equal(first, np.zeros(4))
        self.assertEqual(s.diagnostics['reason'], 'motion_estimate_unavailable')
        obs['lidar']['timestamp'] += 1/15
        second, _ = s.filter(obs, wheels, [0., 0.])
        np.testing.assert_array_equal(second, wheels)
        s.reset()
        self.assertIsNone(s._motion_scan_timestamp)
        self.assertEqual(s._scan_motion, [])
        reset, _ = s.filter(obs, wheels, [0., 0.])
        np.testing.assert_array_equal(reset, np.zeros(4))

    def test_bad_age_is_guarded_before_odometry(self):
        class Forbidden:
            def update(self, _):
                raise AssertionError('Invalid packet reached odometry')
        for age in [-.01, float('nan'), float('inf'), .15]:
            s = LocalSafetySupervisor()
            s.odometry = Forbidden()
            obs = observation()
            obs['lidar']['age'] = age
            wheels, _ = s.filter(obs, [50., 50., 50., -50.], [0., 0.])
            np.testing.assert_array_equal(wheels, np.zeros(4))

    @staticmethod
    def estimator(s):
        class Stub:
            def update(self, _):
                return self.current
        stub = Stub()
        s.odometry = stub
        def call(t, yaw, yaw_valid=True, longitudinal=True, age=.03):
            stub.current = {'dt': 1/15, 'yaw_rate': yaw, 'valid': False,
                'observable_axes': {'yaw': yaw_valid, 'longitudinal': longitudinal, 'lateral': True},
                'confidence': .65, 'matches': 30, 'residual_m': .01, 'inlier_fraction': .9,
                'reason': 'local_estimate_fixture', 'lidar_forward_speed': 4., 'lateral_speed': -.5}
            return s._motion_hypotheses({'lidar': {'timestamp': t, 'age': age}},
                np.empty((0, 2)), np.empty(0, bool), .5, [])
        return call

    def test_partial_axes_rejected_gap_and_held_packet_history(self):
        s = LocalSafetySupervisor()
        call = self.estimator(s)
        self.assertTrue(call(0., 1.))
        self.assertFalse(call(1/15, 2., False))
        hypotheses = call(2/15, 3., longitudinal=False)
        self.assertTrue(hypotheses)  # Overall valid=False does not erase observed yaw.
        self.assertTrue(all(h['motion'][2] > 0 for h in hypotheses))
        self.assertAlmostEqual(s._yaw_acceleration, 15.)
        self.assertAlmostEqual(s._accepted_motion_gap, 2/15)
        self.assertEqual(len(s._scan_motion), 3)  # Retains uncertain yaw transport after gap.
        upper = s._motion_upper.copy()
        same = call(2/15, 3., longitudinal=False)
        self.assertEqual(hypotheses, same)
        np.testing.assert_array_equal(s._motion_upper, upper)
        self.assertFalse(call(.1, 3.))
        self.assertEqual(s.motion_diagnostics['reason'], 'invalid_or_out_of_order_scan_timestamp')
        self.assertFalse(call(float('nan'), 3.))

    def test_directed_geometry_does_not_flip_after_crossing_old_wall(self):
        from racing.safety import _scan_geometry, _transform_geometry, FOOTPRINT
        from racing.safety_motion import compensate_points
        p = np.c_[np.full(40, .55), np.linspace(-1., 1., 40)]
        g = _scan_geometry(p, np.ones(40, bool))
        wall = _transform_geometry(g, [7., 0., 0.], .1)['walls'][0]
        np.testing.assert_allclose(wall['normal'], [1., 0.], atol=1e-12)
        self.assertAlmostEqual(wall['offset'], -.15)
        self.assertAlmostEqual(wall['offset']-max(FOOTPRINT@wall['normal'])-.02, -.57)
        for twist, age in [([4., -1., 3.], .05), ([7., 2., -4.], .14)]:
            wall = _transform_geometry(g, twist, age)['walls'][0]
            points = compensate_points(p, twist, age)
            np.testing.assert_allclose(points@wall['normal'], wall['offset'], atol=1e-12)
            np.testing.assert_allclose([min(points@wall['tangent']), max(points@wall['tangent'])],
                                      [wall['minimum'], wall['maximum']], atol=1e-12)

    def test_old_reflector_face_and_roi_padding(self):
        from racing.safety import _scan_geometry, _transform_geometry
        p = np.c_[np.full(20, .45), np.linspace(.4, .56, 20)]
        valid = np.ones(20, bool)
        g = _transform_geometry(_scan_geometry(p, valid), [8., 0., 0.], .05)
        np.testing.assert_allclose(g['oriented'][0, :2], [.10, .48], atol=1e-12)
        far = p + [3.1, 0.]
        self.assertEqual(len(_scan_geometry(far, valid)['oriented']), 0)
        padded = _transform_geometry(_scan_geometry(far, valid, padding=.7), [7., 0., 0.], .1)
        self.assertEqual(len(padded['oriented']), 1)
        np.testing.assert_allclose(padded['oriented'][0, :2], [2.9, .48], atol=1e-12)

    def test_recovery_command_checks_fast_and_stationary_motion_together(self):
        class Prescribed(LocalSafetySupervisor):
            def _motion_hypotheses(self, *args, **kwargs):
                return [{'scan_motion': twist, 'motion': twist, 'models': [('slip', None)],
                         'retained_speed': True} for twist in self.motions]
        safe = Prescribed()
        safe.motions = [[0., 0., 0.], [4., 0., 0.]]
        obs = observation(right=-2., left=.21, speed=0., slope=.12)
        wheels, _ = safe.filter(obs, [50., 50., 50., -50.], [.35, .35])
        self.assertTrue(safe.diagnostics['recovering'])
        self.assertAlmostEqual(wheels[0]*.045, .5)
        unsafe = Prescribed()
        unsafe.motions = [[0., 0., 0.], [4., -1., 3.]]
        obs = observation(right=-1.2, left=.3, speed=0.)
        wheels, _ = unsafe.filter(obs, [50., 50., 50., -50.], [-.35, -.35])
        self.assertFalse(unsafe.diagnostics['recovering'])
        np.testing.assert_array_equal(wheels, np.zeros(4))


class TestLocalMotionGeometry(unittest.TestCase):
    def test_exact_se2_and_mirror(self):
        from racing.safety_motion import twist_displacement, compensate_points, predict_poses, swept_footprints
        translation, yaw = twist_displacement([4., -1., 3.], .05)
        np.testing.assert_allclose(translation, [.2029938173, -.0348408147], atol=1e-10)
        self.assertAlmostEqual(yaw, .15)
        np.testing.assert_allclose(compensate_points([[2., .75]], [4., -1., 3.], .05),
                                  [[1.894112886, .507486650]], atol=1e-9)
        np.testing.assert_array_equal(compensate_points([[2., .75]], [4., -1., 3.], 0.), [[2., .75]])
        for twist in [[0., 1., 0.], [0., 0., 3.], [4., -1., 3.], [4., .1, 1e-12]]:
            poses = predict_poses(twist, 0., 0.)
            translation, yaw = twist_displacement(twist, .4)
            np.testing.assert_allclose(poses[-1], [*translation, yaw], atol=1e-10)
            mirror = predict_poses([twist[0], -twist[1], -twist[2]], 0., 0.)
            np.testing.assert_allclose(mirror, poses*[1., -1., -1.], atol=1e-10)
        poses = predict_poses([4., -1., 3.], 0., 0., mode='coast')
        np.testing.assert_allclose(poses[-1, :2], [1.6, -.4], atol=1e-10)
        footprint = swept_footprints([[.4, .14]], [[.2029938173, -.0348408147, .15]])
        np.testing.assert_allclose(footprint[0, 0], [.577580910, .163362389], atol=1e-9)

    def test_recovery_accelerates_from_initial_velocity_and_validates_inputs(self):
        from racing.safety_motion import predict_poses
        slow = predict_poses([0., 0., 0.], 0., 0., command_speed=.5)
        fast = predict_poses([4., 0., 0.], 0., 0., command_speed=.5)
        self.assertAlmostEqual(slow[1, 0], .0024)
        self.assertAlmostEqual(fast[-1, 0], 1.6)
        # A negative command is the bounded reverse manoeuvre the supervisor
        # verifies before permitting a back-off; it is no longer invalid input.
        reverse = predict_poses([0., 0., 0.], 0., 0., command_speed=-.6, horizon=.5)
        self.assertLess(reverse[-1, 0], -.2)
        self.assertAlmostEqual(reverse[-1, 1], 0.)
        self.assertTrue(np.all(np.diff(reverse[:, 0]) <= 0.))
        for kwargs in [{'command_speed': float('nan')},
                       {'max_acceleration': -1.}, {'braking_deceleration': -1.},
                       {'yaw_bias_decay': float('nan')}, {'yaw_bias_decay': 0.}]:
            with self.assertRaises(ValueError):
                predict_poses([0., 0., 0.], 0., 0., **kwargs)


class TestForwardRecoveryAcceleration(unittest.TestCase):
    def test_tiny_reverse_or_sideways_motion_does_not_command_reverse_acceleration(self):
        from racing.safety_motion import predict_poses
        for mode in ['slip', 'coast']:
            negative = predict_poses([-.00122, .000077, 0.], 0., 0., command_speed=.5, mode=mode)
            self.assertAlmostEqual(negative[1, 0], .0023756)
            self.assertAlmostEqual(negative[1, 1], .00000154)
            self.assertGreater(negative[-1, 0], .18)
            side = predict_poses([0., .2, 0.], 0., 0., command_speed=.5, mode=mode)
            self.assertAlmostEqual(side[1, 0], .0024)
            self.assertAlmostEqual(side[1, 1], .004)
            reverse = predict_poses([-.4, 0., 0.], 0., 0., command_speed=.5, mode=mode)
            self.assertAlmostEqual(reverse[1, 0], -.0056)
            self.assertGreater(reverse[-1, 0], 0.)
            mirror = predict_poses([0., -.2, 0.], 0., 0., command_speed=.5, mode=mode)
            np.testing.assert_allclose(mirror, side*[1., -1., -1.], atol=1e-12)


class TestBrakingSteeringSelection(unittest.TestCase):
    def test_feasible_countersteer_is_not_replaced_by_opposite_larger_gap(self):
        class Scored(LocalSafetySupervisor):
            def _motion_hypotheses(self, *args, **kwargs):
                return [{'scan_motion': [3., .5, -3.], 'motion': [3., .5, -3.],
                         'models': [('slip', None)], 'retained_speed': False}]
            def _clearance(self, walls, speed, actual, target, **options):
                if options.get('oriented_only'):
                    return float('inf')
                if options.get('braking'):
                    return .4 if target < 0. else .08
                return -.2
        supervisor = Scored()
        proposal = np.array([.25, .22])
        wheels, steering = supervisor.filter(observation(speed=3.), [60., 60., 60., -60.], proposal)
        self.assertTrue(supervisor.diagnostics['must_brake'])
        assert_rolling_brake(self, wheels, 3.)
        np.testing.assert_array_equal(steering, proposal)
        self.assertAlmostEqual(supervisor.diagnostics['selected_clearance_m'], .08)


if __name__ == '__main__':
    unittest.main()
