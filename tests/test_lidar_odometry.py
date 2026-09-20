"""Independent synthetic scans with known rigid motion, not simulator truth inputs."""
import unittest
import numpy as np
from racing.odometry import LidarOdometry

DT = 1/15
ROOM = np.array([[[-6., -3.], [6., -3.]], [[6., -3.], [6., 3.]],
                 [[6., 3.], [-6., 3.]], [[-6., 3.], [-6., -3.]]])
CORRIDOR = np.array([[[-50., -.75], [50., -.75]], [[-50., .75], [50., .75]]])


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def motion(vx, vy, omega):
    theta = omega*DT
    if abs(theta) < 1e-9:
        return np.array([vx, vy])*DT, theta
    a, b = np.sin(theta)/theta, (1-np.cos(theta))/theta
    return np.array([[a, -b], [b, a]])@np.array([vx, vy])*DT, theta


def box(center, half=.3):
    x, y = center
    corners = np.array([[x-half, y-half], [x+half, y-half], [x+half, y+half], [x-half, y+half]])
    return np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)


def observation(scene, xy=(0., 0.), yaw=0., vx=3., omega=0., timestamp=0.,
                mount_position=(0., 0., .2), mount_rotation=np.eye(3)):
    """Ray/segment intersection in the actual synthetic mounted laser plane."""
    angles = np.linspace(-3*np.pi/4, 3*np.pi/4, 361)
    local = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))
    directions = (local@mount_rotation.T)[:, :2]@rotation(yaw).T
    origin = np.asarray(xy)+rotation(yaw)@np.asarray(mount_position)[:2]
    ranges = np.full(len(angles), 15.)
    for start, end in scene:
        edge, delta = end-start, start-origin
        den = directions[:, 0]*edge[1]-directions[:, 1]*edge[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            ray_t = (delta[0]*edge[1]-delta[1]*edge[0])/den
            along = (delta[0]*directions[:, 1]-delta[1]*directions[:, 0])/den
        hit = (abs(den) > 1e-9) & (ray_t > .04) & (along >= 0) & (along <= 1)
        ranges = np.minimum(ranges, np.where(hit, ray_t, 15.))
    steering = np.arctan(.28764*omega/max(abs(vx), .1))*np.sign(vx)
    return {'wheel_vel': np.full(4, vx/.045/max(np.cos(steering), .1)),
            'steer_pos': np.full(2, steering),
            'lidar': {'ranges': ranges, 'angles': angles, 'valid': ranges < 15.,
                      'timestamp': timestamp, 'age': 0., 'frame_id': 'laser'}}


class TestLidarOdometry(unittest.TestCase):
    def estimator(self):
        return LidarOdometry(laser_position=(0, 0, .2), laser_rotation=np.eye(3))

    def test_controlled_lateral_slip_and_rotation(self):
        vx, vy, omega = 3., -1.3, 1.2
        delta, theta = motion(vx, vy, omega)
        odom = self.estimator()
        # During sideslip, steering kinematics are deliberately a poor yaw
        # predictor: scan registration must recover the actual 1.2 rad/s.
        self.assertFalse(odom.update(observation(ROOM, vx=vx, omega=.1))['valid'])
        estimate = odom.update(observation(ROOM, xy=delta, yaw=theta, vx=vx, omega=.1, timestamp=DT))
        self.assertTrue(estimate['valid'], estimate)
        self.assertAlmostEqual(estimate['lateral_speed'], vy, delta=.10)
        self.assertAlmostEqual(estimate['yaw_rate'], omega, delta=.06)
        self.assertAlmostEqual(estimate['slip_angle'], np.arctan2(vy, vx), delta=.04)
        self.assertTrue(all(estimate['observable_axes'].values()))
        self.assertGreater(estimate['confidence'], .8)

    def test_dynamic_occluder_does_not_become_ego_motion(self):
        vx, vy, omega = 3., -.6, .5
        delta, theta = motion(vx, vy, omega)
        odom = self.estimator()
        before = np.concatenate((ROOM, box((2., .4))))
        after = np.concatenate((ROOM, box((1.5, -.6))))
        odom.update(observation(before, vx=vx, omega=omega))
        estimate = odom.update(observation(after, xy=delta, yaw=theta, vx=vx, omega=omega, timestamp=DT))
        self.assertTrue(estimate['valid'], estimate)
        self.assertAlmostEqual(estimate['lateral_speed'], vy, delta=.15)
        self.assertAlmostEqual(estimate['yaw_rate'], omega, delta=.08)

    def test_parallel_walls_report_longitudinal_degeneracy(self):
        vx, vy, omega = 4., .45, .4
        delta, theta = motion(vx, vy, omega)
        odom = self.estimator()
        odom.update(observation(CORRIDOR, vx=vx, omega=omega))
        estimate = odom.update(observation(CORRIDOR, xy=delta, yaw=theta, vx=vx, omega=omega, timestamp=DT))
        self.assertTrue(estimate['valid'], estimate)
        self.assertFalse(estimate['observable_axes']['longitudinal'])
        self.assertTrue(estimate['observable_axes']['lateral'])
        self.assertTrue(estimate['observable_axes']['yaw'])
        self.assertLessEqual(estimate['confidence'], .65)
        self.assertAlmostEqual(estimate['forward_speed'], vx)
        self.assertAlmostEqual(estimate['lateral_speed'], vy, delta=.08)
        self.assertAlmostEqual(estimate['yaw_rate'], omega, delta=.05)

    def test_known_mount_rotation_and_lever_arm(self):
        odom = LidarOdometry()
        kwargs = {'mount_position': odom.mount_position, 'mount_rotation': odom.mount_rotation}
        vx, vy, omega = 2.5, -.5, 1.5
        delta, theta = motion(vx, vy, omega)
        odom.update(observation(ROOM, vx=vx, omega=omega, **kwargs))
        estimate = odom.update(observation(ROOM, xy=delta, yaw=theta, vx=vx, omega=omega, timestamp=DT, **kwargs))
        self.assertTrue(estimate['valid'], estimate)
        self.assertAlmostEqual(estimate['lateral_speed'], vy, delta=.08)
        self.assertAlmostEqual(estimate['yaw_rate'], omega, delta=.05)

    def test_sample_hold_staleness_and_rear_spin_invariance(self):
        delta, theta = motion(3., -.7, .3)
        a, b = self.estimator(), self.estimator()
        previous = observation(ROOM)
        a.update(previous); b.update(previous)
        current = observation(ROOM, xy=delta, yaw=theta, vx=3., omega=.3, timestamp=DT)
        spinning = dict(current, wheel_vel=current['wheel_vel'].copy())
        spinning['wheel_vel'][2:] = 500.
        one, two = a.update(current), b.update(spinning)
        self.assertEqual(one, two)
        held = dict(current, lidar=dict(current['lidar'], age=.05))
        estimate = a.update(held)
        self.assertEqual(estimate['lateral_speed'], one['lateral_speed'])
        self.assertEqual(estimate['age'], .05)
        stale = dict(current, lidar=dict(current['lidar'], age=.2))
        self.assertFalse(a.update(stale)['valid'])
        # No cumulative world coordinates or absolute heading are produced.
        self.assertTrue({'x', 'y', 'yaw', 'global_pose', 'progress'}.isdisjoint(one))

    def test_missing_geometry_has_no_confident_motion(self):
        odom = self.estimator()
        obs = observation(ROOM)
        obs['lidar']['valid'][:] = False
        result = odom.update(obs)
        self.assertFalse(result['valid'])
        self.assertEqual(result['confidence'], 0.)

    def test_centimeter_noise_and_dropouts_have_bounded_error(self):
        vx, vy, omega = 3., -.8, .9
        delta, theta = motion(vx, vy, omega)
        for seed in range(5):
            rng, odom = np.random.default_rng(seed), self.estimator()
            scans = [observation(ROOM, vx=vx, omega=.1),
                     observation(ROOM, xy=delta, yaw=theta, vx=vx, omega=.1, timestamp=DT)]
            for obs in scans:
                packet = obs['lidar']
                packet['ranges'] += rng.normal(0, .01, len(packet['ranges']))
                packet['valid'] &= rng.random(len(packet['ranges'])) > .08
            odom.update(scans[0]); estimate = odom.update(scans[1])
            self.assertTrue(estimate['valid'], estimate)
            self.assertAlmostEqual(estimate['lateral_speed'], vy, delta=.12)
            self.assertAlmostEqual(estimate['yaw_rate'], omega, delta=.05)
            self.assertLess(estimate['confidence'], .9)


if __name__ == '__main__':
    unittest.main()
