import unittest
import numpy as np
from racing.sensors import LidarSensor, ACTOR_KEYS
from racing.policy import RacingPolicy
from racing.metrics import RaceMetrics, drift_statistics


class StraightTrack:
    width, length = 1.5, 100.
    boundary_segments = np.array([[[-20, -.75], [20, -.75]], [[-20, .75], [20, .75]],
                                  [[5, -.75], [5, .75]]], float)

    def project(self, xy):
        return xy[0] % self.length, xy[1], 0.


def state(x=0, y=0):
    return {'x': x, 'y': y, 'z': .05, 'yaw': 0., 'wheel_vel': np.ones(4), 'steer_pos': np.zeros(2)}


class RacingContracts(unittest.TestCase):
    def test_continuous_drift_rejects_accumulated_disjoint_spikes(self):
        speed = np.full(18, 2.)
        slip = np.deg2rad([25]*3+[0]*3+[25]*3+[0]*3+[25]*3+[0]*3)
        result = drift_statistics(speed, slip)
        self.assertAlmostEqual(result['drift_duration_s'], .15)
        self.assertAlmostEqual(result['max_continuous_drift_duration_s'], .05)
        self.assertEqual(result['drift_event_count'], 0)
        self.assertFalse(result['continuous_drift_qualified'])
        slip = np.deg2rad([25]*6+[0]*2+[-25]*8+[0]*2)
        result = drift_statistics(speed, slip)
        self.assertEqual(result['drift_event_count'], 2)
        self.assertAlmostEqual(result['max_continuous_drift_duration_s'], 8/60)
        self.assertTrue(result['continuous_drift_qualified'])
        speed[:] = 1.2
        result = drift_statistics(speed, slip)
        self.assertFalse(result['continuous_drift_qualified'])
        self.assertEqual(result['drift_duration_s'], 0.)

    def test_raised_wall_and_bridge_scan_use_physical_height(self):
        class Bridge(StraightTrack):
            has_elevation=True
            def boundary_segments_3d(self):
                return np.array([[[5,-2,.9],[5,2,.9]]])
            def road_triangles_3d(self):
                return np.array([[[1,-2,.9],[4,-2,.9],[4,2,.9]],
                                 [[1,-2,.9],[4,2,.9],[1,2,.9]]])
        sensor=LidarSensor(Bridge(),ray_count=3,fov_deg=90)
        # A lower-level horizontal beam passes under the raised wall.
        ranges,valid=sensor.raycast(([0,0,.2],np.eye(3)))
        self.assertFalse(valid[1])
        ranges,valid=sensor.raycast(([0,0,1.1],np.eye(3)))
        self.assertAlmostEqual(ranges[1],5.)
        angle=-np.pi/6;c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        ranges,valid=sensor.raycast(([0,0,.2],rotation))
        self.assertTrue(valid[1])
        self.assertAlmostEqual(ranges[1],1.24)  # Hits physical underside, z=.82.

    def test_sample_hold_and_actor_boundary(self):
        sensor = LidarSensor(StraightTrack())
        pose = (np.array([0, 0, .2]), np.eye(3))
        first = sensor.observe(state(), pose)
        self.assertEqual(set(first), ACTOR_KEYS)
        self.assertEqual(first['lidar']['frame_id'], 'laser')
        for tick in range(1, 4):
            obs = sensor.observe(state(tick), (np.array([tick, 0, .2]), np.eye(3)))
            np.testing.assert_array_equal(obs['lidar']['ranges'], first['lidar']['ranges'])
            self.assertAlmostEqual(obs['lidar']['age'], tick/60)
        fresh = sensor.observe(state(1), (np.array([1, 0, .2]), np.eye(3)))
        self.assertEqual(fresh['lidar']['age'], 0.)
        self.assertAlmostEqual(fresh['lidar']['timestamp'], 4/60)
        self.assertNotEqual(fresh['lidar']['ranges'][180], first['lidar']['ranges'][180])

    def test_actual_beam_plane_and_finite_height(self):
        sensor = LidarSensor(StraightTrack(), ray_count=3, fov_deg=90, wall_height=.6)
        ranges, valid = sensor.raycast(([0, 0, .2], np.eye(3)))
        self.assertAlmostEqual(ranges[1], 5.)
        pitch = -.2
        c, s = np.cos(pitch), np.sin(pitch)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        tilted, tilted_valid = sensor.raycast(([0, 0, .2], rotation))
        self.assertFalse(tilted_valid[1])
        self.assertEqual(tilted[1], 15.)
        # Translation is from the laser origin, not the car origin.
        shifted, _ = sensor.raycast(([1, 0, .2], np.eye(3)))
        self.assertAlmostEqual(shifted[1], 4.)

    def test_seeded_noise_dropout_and_delivery_latency(self):
        pose = ([0, 0, .2], np.eye(3))
        clean = LidarSensor(StraightTrack()).observe(state(), pose)['lidar']
        a = LidarSensor(StraightTrack(), noise_std_m=.02, dropout_prob=.25, seed=17)
        b = LidarSensor(StraightTrack(), noise_std_m=.02, dropout_prob=.25, seed=17)
        noisy = a.observe(state(), pose)['lidar']
        twin = b.observe(state(), pose)['lidar']
        np.testing.assert_array_equal(noisy['ranges'], twin['ranges'])
        self.assertGreater(np.count_nonzero(clean['valid'] & ~noisy['valid']), 0)
        hits = clean['valid'] & noisy['valid']
        self.assertGreater(np.std(noisy['ranges'][hits]-clean['ranges'][hits]), .01)
        np.testing.assert_array_equal(noisy['ranges'][~noisy['valid']], 15.)
        a.reset()
        np.testing.assert_array_equal(a.observe(state(), pose)['lidar']['ranges'], noisy['ranges'])
        delayed = LidarSensor(StraightTrack(), latency_s=.05)
        for tick in range(3):
            obs = delayed.observe(state(), pose, tick=tick)
            self.assertFalse(np.any(obs['lidar']['valid']))
        first = delayed.observe(state(), ([1, 0, .2], np.eye(3)), tick=3)['lidar']
        self.assertEqual(first['timestamp'], 0.)
        self.assertAlmostEqual(first['age'], .05)
        self.assertAlmostEqual(first['ranges'][180], 5.)
        for tick in range(4, 7):
            held = delayed.observe(state(), ([1, 0, .2], np.eye(3)), tick=tick)['lidar']
            self.assertEqual(held['timestamp'], 0.)
        fresh = delayed.observe(state(), pose, tick=7)['lidar']
        self.assertAlmostEqual(fresh['timestamp'], 4/60)
        self.assertAlmostEqual(fresh['ranges'][180], 4.)
        self.assertAlmostEqual(fresh['age'], .05)
        dropped = LidarSensor(StraightTrack(), dropout_prob=1).observe(state(), pose)
        self.assertFalse(np.any(dropped['lidar']['valid']))
        wheel, _ = RacingPolicy().action(dropped)
        np.testing.assert_array_equal(wheel, 0.)

    def test_reflector_uses_exact_box(self):
        sensor = LidarSensor(StraightTrack(), ray_count=3, fov_deg=90)
        car = {'lidar_target': {'position': [2, 0, .2], 'rotation': np.eye(3), 'half_size': [.05, .08, .10]}}
        ranges, _ = sensor.raycast(([0, 0, .2], np.eye(3)), [car])
        self.assertAlmostEqual(ranges[1], 1.95)
        car['lidar_target']['position'][2] = 1.
        ranges, _ = sensor.raycast(([0, 0, .2], np.eye(3)), [car])
        self.assertAlmostEqual(ranges[1], 5.)

    def test_actor_ignores_privileged_extras_and_respects_limits(self):
        obs = LidarSensor(StraightTrack()).observe(state(), ([0, 0, .2], np.eye(3)))
        dirty = dict(obs, x=9999, yaw=-2., track_id='cheat', progress=100, opponents=[state(2)])
        a, b = RacingPolicy(), RacingPolicy()
        for _ in range(10):
            wa, sa = a.action(obs)
            wb, sb = b.action(dirty)
            np.testing.assert_array_equal(wa, wb)
            np.testing.assert_array_equal(sa, sb)
            self.assertLessEqual(max(abs(sa)), .45)
            self.assertGreaterEqual(min(wa[:3]), 0)
            self.assertLessEqual(wa[3], 0)

    def test_gap_center_corrects_offset_and_heading_before_wall(self):
        sensor = LidarSensor(StraightTrack())
        centered = sensor.observe(state(), ([0, 0, .2], np.eye(3)))
        center_actor = RacingPolicy()
        _, steer = center_actor._plan(centered['lidar'], 5.)
        self.assertLess(abs(steer), .015)
        sensor.reset()
        offset = sensor.observe(state(), ([0, .25, .2], np.eye(3)))
        _, steer = RacingPolicy()._plan(offset['lidar'], 5.)
        self.assertLess(steer, -.015)
        sensor.reset()
        angle = .15
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        angled = sensor.observe(state(), ([0, 0, .2], rot))
        _, steer = RacingPolicy()._plan(angled['lidar'], 5.)
        self.assertLess(steer, -.02)

    def test_compact_reflector_envelope_retains_rear_side_clearance(self):
        pose = ([0, .3, .2], np.eye(3))
        empty = LidarSensor(StraightTrack()).observe(state(), pose)
        _, center_steer = RacingPolicy()._plan(empty['lidar'], 4.)
        reflector = {'lidar_target': {'center': [-.25, 0, .4], 'rotation': np.eye(3),
                                     'half_size': [.05, .08, .275]}}
        beside = LidarSensor(StraightTrack()).observe(state(), pose, [reflector])
        _, passing_steer = RacingPolicy()._plan(beside['lidar'], 4.)
        # The narrow target is already behind the laser, but inferred body
        # extent still prohibits returning right into the other vehicle.
        self.assertLess(center_steer, 0.)
        self.assertGreater(passing_steer, 0.)
        reflector['lidar_target']['center'][0] = -.5
        cleared = LidarSensor(StraightTrack()).observe(state(), pose, [reflector])
        _, cleared_steer = RacingPolicy()._plan(cleared['lidar'], 4.)
        self.assertAlmostEqual(cleared_steer, center_steer)

    def test_rear_spin_cannot_change_front_encoder_speed_planning(self):
        obs = LidarSensor(StraightTrack()).observe(state(), ([0, .15, .2], np.eye(3)))
        obs['wheel_vel'] = np.full(4, 2/.045)
        obs['steer_pos'] = np.array([.33, .36])
        spinning = dict(obs, wheel_vel=np.array([2/.045, 2/.045, 200., 250.]))
        params = RacingPolicy.initial.copy()
        params[-2] = 1.
        normal_actor, spin_actor = RacingPolicy(params), RacingPolicy(params)
        normal_action = normal_actor.action(obs)
        spin_action = spin_actor.action(spinning)
        np.testing.assert_array_equal(normal_action[0], spin_action[0])
        np.testing.assert_array_equal(normal_action[1], spin_action[1])
        self.assertEqual(normal_actor.target_speed, spin_actor.target_speed)
        self.assertEqual(normal_actor.target_steer, spin_actor.target_steer)
        self.assertEqual(normal_actor.phase, 'rear_overdrive')

    def test_lidar_clearance_gates_and_cancels_drift_pulses(self):
        pose = ([0, 0, .2], np.eye(3))
        measured = dict(state(), wheel_vel=np.full(4, 2/.045), steer_pos=np.array([.35, .35]))
        clear = LidarSensor(StraightTrack()).observe(measured, pose)
        params = RacingPolicy.initial.copy()
        params[-2] = 1.
        actor = RacingPolicy(params)
        actor.action(clear)
        self.assertEqual(actor.phase, 'rear_overdrive')
        reflector = {'lidar_target': {'center': [2., 0, .4], 'rotation': np.eye(3),
                                     'half_size': [.05, .08, .275]}}
        near = LidarSensor(StraightTrack()).observe(measured, pose, [reflector], tick=4)
        actor.action(near)
        self.assertTrue(actor.near_compact_obstacle)
        self.assertEqual(actor.drift_seconds, 0.)
        self.assertNotEqual(actor.phase, 'rear_overdrive')
        fresh_actor = RacingPolicy(params)
        fresh_actor.action(near)
        self.assertEqual(fresh_actor.drift_seconds, 0.)
        actor = RacingPolicy(params)
        actor.action(clear)
        stale = dict(clear, lidar=dict(clear['lidar'], age=.2))
        actor.action(stale)
        self.assertEqual(actor.drift_seconds, 0.)
        self.assertNotEqual(actor.phase, 'rear_overdrive')

    def test_overtake_persistence_and_collision_rejection(self):
        for collision, expected in [(False, 1), (True, 0)]:
            metrics = RaceMetrics(StraightTrack(), dt=.1)
            metrics.update(state(0), [state(1)])
            metrics.update(state(.7), [state(1)], collision=collision)
            metrics.update(state(1.6), [state(1)])
            self.assertEqual(metrics.effective_overtakes, 0)
            metrics.update(state(1.7), [state(1)])
            metrics.update(state(1.8), [state(1)])
            self.assertEqual(metrics.effective_overtakes, expected)

    def test_contact_clears_overtake_until_new_behind_state(self):
        metrics = RaceMetrics(StraightTrack(), dt=.1)
        metrics.update(state(0), [state(1)])
        metrics.update(state(1.6), [state(1)])
        metrics.update(state(1.7), [state(1)], collision=True)
        for x in [1.8, 1.9, 2., 2.1]:
            metrics.update(state(x), [state(1)])
        self.assertEqual(metrics.effective_overtakes, 0)
        metrics.update(state(.3), [state(1)])
        for x in [1.6, 1.7, 1.8]:
            metrics.update(state(x), [state(1)])
        self.assertEqual(metrics.effective_overtakes, 1)

    def test_teleport_and_invalid_lap(self):
        track = StraightTrack()
        metrics = RaceMetrics(track, dt=.1)
        metrics.update(state(0))
        metrics.update(state(80))
        self.assertEqual(metrics.progress, 0.)
        self.assertEqual(metrics.teleports, 1)
        self.assertFalse(metrics.summary()['current_lap_valid'])
        # Use a continuous circle to test wrap guards without Euclidean jumps.
        class Circle:
            length, width = 2*np.pi, 1.5
            def project(self, xy):
                return np.arctan2(xy[1], xy[0]) % (2*np.pi), 0., 0.
        metrics = RaceMetrics(Circle(), dt=.1)
        for angle in np.linspace(0, 2*np.pi+.1, 100):
            metrics.update(state(np.cos(angle), np.sin(angle)), collision=.2 < angle < .3)
        self.assertEqual(metrics.laps, 1)
        self.assertEqual(len(metrics.valid_lap_times), 0)
        self.assertFalse(metrics.lap_records[0]['valid'])

    def test_evaluator_projects_bridge_height_without_actor_truth(self):
        class LayeredTrack(StraightTrack):
            has_elevation = True
            def project(self, position):
                self.last_position = np.asarray(position)
                return position[0]+(50 if position[2] > 1 else 0), position[1], 0.
        track = LayeredTrack()
        metrics = RaceMetrics(track)
        elevated = dict(state(2), z=2.)
        metrics.update(elevated)
        np.testing.assert_array_equal(track.last_position, [2., 0., 2.])
        self.assertEqual(metrics.last_s, 52.)
        measured = LidarSensor(StraightTrack()).observe(elevated, ([2, 0, .2], np.eye(3)))
        self.assertEqual(set(measured), ACTOR_KEYS)


if __name__ == '__main__':
    unittest.main()
