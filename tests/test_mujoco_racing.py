"""Native dynamics checks; no policy or kinematic replacement participates."""
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from racing.mujoco_env import CONTROL_DT, RaceMujocoEnv


class StraightTrack:
    """Long straight section for acceleration and contact regression tests."""
    id = 'native_regression'
    width = 1.5
    points = np.array([[0., 0.], [30., 0.], [30., 8.], [0., 8.]])

    def at(self, s, offset=0.):
        return np.array([s, offset]), 0.

    def boundary_segments(self):
        return np.array([[[-1., -.75], [30., -.75]], [[-1., .75], [30., .75]]])


class BridgeTrack(StraightTrack):
    """An elevated straight and perpendicular ground-level underpass."""
    id = 'native_bridge_regression'
    metadata = {'bridge': {'deck_thickness': .08}}
    points = np.array([[0., 0.], [10., 0.], [18., 0.], [22., 0.], [30., 0.],
                       [35., 0.], [35., 8.], [0., 8.]])
    elevations = np.array([0., 0., .9, .9, 0., 0., 0., 0.])

    def at(self, s, offset=0.):
        if s >= 100:
            return np.array([20.-offset, s-105.]), np.pi/2
        return super().at(s, offset)

    def elevation_at(self, s):
        return float(np.interp(s, [0., 10., 18., 22., 30., 1000.], [0., 0., .9, .9, 0., 0.]))

    def at3d(self, s, offset=0.):
        xy, yaw = self.at(s, offset)
        grade = .9/8 if 10 < s < 18 else (-.9/8 if 22 < s < 30 else 0.)
        return np.r_[xy, self.elevation_at(s)], yaw, grade

    def boundary_segments_3d(self):
        upper = [[[x, y, self.elevation_at(x)], [x+.25, y, self.elevation_at(x+.25)]]
                 for y in (-.75, .75) for x in np.arange(0., 35., .25)]
        lower = [[[x, -5., 0.], [x, 5., 0.]] for x in (19.25, 20.75)]
        return np.asarray(upper+lower)

    def road_triangles_3d(self):
        triangles = []
        for x in np.arange(0., 35., .25):
            a, b = self.elevation_at(x), self.elevation_at(x+.25)
            quad = np.array([[x, -.75, a], [x, .75, a], [x+.25, .75, b], [x+.25, -.75, b]])
            triangles.extend([quad[[0, 1, 2]], quad[[0, 2, 3]]])
        return np.asarray(triangles)


class TestRaceMujocoEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = RaceMujocoEnv(StraightTrack(), num_cars=2)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_independent_native_acceleration_and_laser_frame(self):
        env = self.env
        initial = env.reset()
        actions = [([100., 100., 100., -100.], [0., 0.]),
                   ([80., 80., 80., -80.], [0., 0.])]
        for _ in range(120):
            states = env.step(actions)
        for i, state in enumerate(states):
            self.assertGreater(state['x'] - initial[i]['x'], 5.)
            self.assertGreater(state['vx'], 3.)
            self.assertFalse(state['collision'])
            self.assertAlmostEqual(state['time'], 120 * CONTROL_DT, places=10)
            self.assertTrue(all(v > 60. for v in state['wheel_vel']))
            laser = env._cars[i]['laser']
            np.testing.assert_allclose(state['lidar_pose']['position'], env.data.xpos[laser])
            rotation = np.asarray(state['lidar_pose']['rotation'])
            np.testing.assert_allclose(rotation, env.data.xmat[laser].reshape(3, 3))
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertGreater(state['lidar_pose']['position'][2] - state['z'], .15)
            self.assertGreater(abs(rotation[2, 0]), .03)  # Actual pitched CAD mount.
            target = env._cars[i]['target']
            np.testing.assert_allclose(state['lidar_target']['center'], env.data.geom_xpos[target])
            np.testing.assert_allclose(state['lidar_target']['half_size'], [.05, .08, .275])
        self.assertGreater(states[0]['vx'], states[1]['vx'] + .5)
        self.assertFalse(np.any(env.data.warning.number))

    def test_native_vehicle_impact_transfers_momentum(self):
        env = self.env
        initial = env.reset(starts=[(0., 0.), (.7, 0.)])
        hit = [False, False]
        for _ in range(100):
            states = env.step([([60., 60., 60., -60.], [0., 0.]),
                               ([0., 0., 0., 0.], [0., 0.])])
            hit = [seen or state['car_collision'] for seen, state in zip(hit, states)]
        self.assertEqual(hit, [True, True])
        # Car 1's commanded wheels are stopped: displacement is contact-driven.
        self.assertGreater(states[1]['x'] - initial[1]['x'], .5)
        self.assertGreater(states[1]['vx'], .5)
        self.assertGreater(states[0]['collision_steps'], 0)
        self.assertFalse(np.any(env.data.warning.number))

    def test_barrier_is_physical(self):
        env = self.env
        env.reset()
        hit = False
        for _ in range(100):
            states = env.step([([60., 60., 60., -60.], [.35, .35]),
                               ([0., 0., 0., 0.], [0., 0.])])
            hit |= states[0]['barrier_collision']
        self.assertTrue(hit)
        self.assertLess(states[0]['y'], .75)

    def test_controls_validate_atomically_and_allow_ten_mps_targets(self):
        env = self.env
        env.reset()
        before = env.data.ctrl.copy()
        with self.assertRaises(ValueError):
            env.step([([50., 50., 50., -50.], [0., 0.]),
                      ([0., 0., float('nan'), 0.], [0., 0.])])
        np.testing.assert_array_equal(env.data.ctrl, before)
        self.assertEqual(env.data.time, 0.)
        wheel_speed = 10. / .045
        env.step([([wheel_speed]*3 + [-wheel_speed], [2., -2.])]*2)
        for car in env._cars:
            np.testing.assert_allclose(env.data.ctrl[car['actuators']],
                                       [wheel_speed]*3 + [-wheel_speed, .45, -.45])

    def test_render_is_native_rgb(self):
        self.env.reset()
        frame = self.env.render()
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(float(frame.std()), 20.)

    def test_generated_track_uses_union_obj_and_shared_barrier_width(self):
        from racing.tracks import Track

        track = Track('bahrain')
        env = RaceMujocoEnv(track)
        try:
            root = ET.parse(env.model_path).getroot()
            mesh = root.find("./asset/mesh[@name='race_road_mesh']")
            self.assertEqual(mesh.get('file'), str(track.path.parent / 'track.obj'))
            self.assertEqual(mesh.get('inertia'), 'shell')
            for geom_id in env._barriers:
                self.assertAlmostEqual(env.model.geom_size[geom_id, 1], .02)
                self.assertAlmostEqual(env.model.geom_size[geom_id, 2], .3)
            self.assertFalse(any(s['collision'] for s in env.state()))
            frame = env.render()
            self.assertEqual(frame.shape, (720, 1280, 3))
            self.assertGreater(float(frame.std()), 20.)
        finally:
            env.close()

    def test_native_bridge_ascent_descent_and_underpass_clearance(self):
        env = RaceMujocoEnv(BridgeTrack())
        try:
            states = env.reset(starts=[(8., 0.), (32., 0.)])
            max_height, collision = 0., False
            for _ in range(1000):
                first = states[0]
                # Test fixture feedback commands only wheel/steering actuators.
                steer = float(np.clip(-2*first['yaw']-.3*first['y'], -.4, .4))
                states = env.step([([40., 40., 40., -40.], [steer, steer]),
                                   ([0., 0., 0., 0.], [0., 0.])])
                max_height = max(max_height, states[0]['z'])
                collision |= states[0]['collision']
                if states[0]['x'] > 30.5:
                    break
            self.assertGreater(states[0]['x'], 30.5)
            self.assertGreater(max_height, .93)
            self.assertAlmostEqual(states[0]['z'], .045, delta=.005)
            self.assertFalse(collision)
            # Both cars cross the same XY area on separate physical layers.
            states = env.reset(starts=[(19.7, 0.), (104.7, 0.)])
            self.assertAlmostEqual(states[0]['z']-states[1]['z'], .9, delta=.003)
            minimum_xy = float('inf')
            for _ in range(60):
                states = env.step([([25., 25., 25., -25.], [0., 0.])]*2)
                minimum_xy = min(minimum_xy, np.hypot(states[0]['x']-states[1]['x'], states[0]['y']-states[1]['y']))
                self.assertFalse(any(s['collision'] for s in states))
                self.assertLess(states[1]['z'], .06)
                self.assertGreater(states[0]['z'], .93)
            self.assertLess(minimum_xy, .08)
            self.assertFalse(np.any(env.data.warning.number))
        finally:
            env.close()

    def test_suzuka_cars_cross_at_different_native_heights(self):
        from racing.tracks import Track

        track = Track('suzuka')
        bridge = track.metadata['bridge']
        progress = [bridge['upper_s']-.7, bridge['lower_s']-.7]
        env = RaceMujocoEnv(track)
        try:
            states = env.reset(starts=[(s, 0.) for s in progress])
            nearest = float('inf')
            for _ in range(120):
                actions = []
                for j, state in enumerate(states):
                    s, _, _ = track.project([state['x'], state['y']], s_hint=progress[j], z=state['z'])
                    progress[j] = s
                    target, _ = track.at(s+.8)
                    angle = np.arctan2(target[1]-state['y'], target[0]-state['x'])-state['yaw']
                    steering = float(np.arctan2(2*.28855*np.sin(angle), .8))
                    actions.append(([30., 30., 30., -30.], [steering, steering]))
                states = env.step(actions)
                nearest = min(nearest, np.hypot(states[0]['x']-states[1]['x'], states[0]['y']-states[1]['y']))
                self.assertFalse(any(state['collision'] for state in states))
                self.assertAlmostEqual(states[0]['z']-states[1]['z'], bridge['surface_height'], delta=.005)
            self.assertLess(nearest, .08)
            self.assertGreater(progress[0], bridge['upper_s'])
            self.assertGreater(progress[1], bridge['lower_s'])
            self.assertFalse(np.any(env.data.warning.number))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
