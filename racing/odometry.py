"""Local LiDAR motion estimation for a sensor-only actor.

Only consecutive scans are registered; no map, world pose, course state, or
opponent truth is accepted or accumulated. A robust point-to-line fit estimates
relative motion. Front encoders resolve longitudinal ambiguity between parallel
walls; observability and confidence are returned explicitly, not concealed.
"""
from __future__ import annotations
import numpy as np


def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _source_mount_rotation():
    w, x, y, z = [.99969349188272216, -7.0832408137250425e-5,
                   -.024589108572215627, .002879758622449709]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class LidarOdometry:
    """Call ``update(actor_observation)`` at 60 Hz; fits occur only at new scans.

    Velocities describe the vehicle base/rear-axle frame. ``forward_speed`` is
    the projected measured front-wheel speed, while ``lidar_forward_speed`` is
    scan-derived where observable and prior-constrained otherwise. ``valid``
    means lateral velocity and yaw are observable, not that all three axes are.
    Static extrinsics default to the source robot's real laser mount. Vehicle
    roll/pitch changes are not estimated by this planar method. Confidence is
    a geometric quality score, not a calibrated probability; pitch/grade bias
    can survive a good planar fit and needs native validation before control.
    """
    def __init__(self, laser_position=(.096555277688949, 0., .160136400226666),
                 laser_rotation=None, min_points=24, max_correspondence=.5,
                 iterations=15):
        self.mount_position = np.asarray(laser_position, float).reshape(3)
        self.mount_rotation = _source_mount_rotation() if laser_rotation is None else np.asarray(laser_rotation, float).reshape(3, 3)
        self.min_points, self.max_correspondence = int(min_points), float(max_correspondence)
        self.iterations = int(iterations)
        self.reset()

    def reset(self):
        self.previous = None
        self.last_timestamp = None
        self.result = self._invalid()

    @staticmethod
    def _invalid(timestamp=-1., age=float('inf'), reason='need_two_scans'):
        return {'timestamp': float(timestamp), 'age': float(age), 'dt': 0.,
                'forward_speed': 0., 'lidar_forward_speed': 0., 'lateral_speed': 0.,
                'yaw_rate': 0., 'slip_angle': 0., 'confidence': 0., 'valid': False,
                'observable_axes': {'longitudinal': False, 'lateral': False, 'yaw': False},
                'residual_m': None, 'inlier_fraction': 0., 'matches': 0,
                'reason': reason}

    def _points(self, packet):
        angles = np.asarray(packet['angles'], float)
        ranges = np.asarray(packet['ranges'], float)
        valid = np.asarray(packet.get('valid', packet.get('validmask')), bool)
        if angles.ndim != 1 or angles.shape != ranges.shape or valid.shape != ranges.shape:
            raise ValueError('Expected matching one-dimensional single-plane scan arrays')
        local = np.column_stack((ranges*np.cos(angles), ranges*np.sin(angles), np.zeros(len(ranges))))
        body = local @ self.mount_rotation.T+self.mount_position
        usable = valid & np.isfinite(ranges) & (ranges > .12) & (ranges < 12.) & (abs(angles) < 2.1)
        # Rearward rays striking the ground do not form persistent wall lines.
        usable &= body[:, 2] > -.025
        points, normals = [], []
        indices = np.flatnonzero(usable)
        if not len(indices):
            return np.empty((0, 2)), np.empty((0, 2))
        split = (np.diff(indices) > 1) | (np.linalg.norm(np.diff(body[indices, :2], axis=0), axis=1) > .35)
        groups = np.split(indices, np.flatnonzero(split)+1)
        for group in groups:
            cloud = body[group, :2]
            # Compact disconnected returns are likely dynamic occluders; the
            # observed geometry alone is sufficient to exclude these points.
            if len(group) < 5 or np.max(np.ptp(cloud, axis=0)) < .45:
                continue
            for i in range(2, len(cloud)-2):
                neighbors = cloud[i-2:i+3]
                centered = neighbors-neighbors.mean(axis=0)
                eigenvalues, eigenvectors = np.linalg.eigh(centered.T@centered)
                if eigenvalues[1] < 1e-5 or eigenvalues[0] > .04*eigenvalues[1]:
                    continue
                points.append(cloud[i]); normals.append(eigenvectors[:, 0])
        return np.asarray(points).reshape(-1, 2), np.asarray(normals).reshape(-1, 2)

    def _register(self, current, previous, normals, prediction):
        transform = prediction.copy()
        info = None
        for _ in range(self.iterations):
            rotated = current @ _rotation(transform[2]).T
            moved = rotated+transform[:2]
            distance2 = np.sum((moved[:, None]-previous[None])**2, axis=2)
            nearest = np.argmin(distance2, axis=1)
            normal = normals[nearest]
            residual = np.sum(normal*(moved-previous[nearest]), axis=1)
            good = (distance2[np.arange(len(current)), nearest] < self.max_correspondence**2) & (abs(residual) < .20)
            if np.count_nonzero(good) < self.min_points:
                return None
            # Trim unmatched surfaces and moving occluders, then robustly weight
            # remaining residuals. Equal line-normal signs are unnecessary.
            cutoff = max(.025, float(np.percentile(abs(residual[good]), 85)))
            good &= abs(residual) <= cutoff
            if np.count_nonzero(good) < self.min_points:
                return None
            n, q, error = normal[good], rotated[good], residual[good]
            jacobian = np.column_stack((n, -n[:, 0]*q[:, 1]+n[:, 1]*q[:, 0]))
            weights = np.minimum(1., .025/np.maximum(abs(error), 1e-9))
            hessian = jacobian.T@(weights[:, None]*jacobian)
            gradient = jacobian.T@(weights*error)
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            observed = eigenvalues > max(1e-5, eigenvalues[-1]*1e-4)
            inverse = (eigenvectors*np.where(observed, 1/np.maximum(eigenvalues, 1e-12), 0.))@eigenvectors.T
            nullspace = eigenvectors[:, ~observed]@eigenvectors[:, ~observed].T
            correction = -inverse@gradient + nullspace@(prediction-transform)
            correction[:2] = np.clip(correction[:2], -.2, .2)
            correction[2] = np.clip(correction[2], -.15, .15)
            transform += correction
            info = (float(np.sqrt(np.mean(error**2))), float(np.mean(good)), int(np.count_nonzero(good)), nullspace)
            if np.linalg.norm(correction[:2]) < 1e-5 and abs(correction[2]) < 1e-5:
                break
        return transform, info

    def update(self, observation):
        packet = observation['lidar']
        if packet.get('frame_id', packet.get('frame', 'laser')) != 'laser':
            raise ValueError('Odometry requires scans in the calibrated laser frame')
        timestamp, age = float(packet['timestamp']), float(packet['age'])
        if self.last_timestamp == timestamp:
            result = dict(self.result, age=age)
            if age > .14:
                result.update(valid=False, confidence=0., reason='stale_scan')
            return result
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError('Scan timestamp moved backwards; reset odometry for a new episode')
        self.last_timestamp = timestamp
        wheel = np.asarray(observation['wheel_vel'], float).reshape(4)
        steering = np.asarray(observation['steer_pos'], float).reshape(2)
        speed = float(np.mean(wheel[:2]*np.cos(steering))*.045)
        points, normals = self._points(packet)
        if not np.isfinite(speed) or len(points) < self.min_points or age > .14:
            self.result = self._invalid(timestamp, age, 'insufficient_or_stale_scan')
            return dict(self.result)
        current = (timestamp, points, normals)
        if self.previous is None:
            self.previous = current
            self.result = self._invalid(timestamp, age)
            return dict(self.result)
        old_time, old_points, old_normals = self.previous
        self.previous = current
        dt = timestamp-old_time
        if not 0 < dt <= .15:
            self.result = self._invalid(timestamp, age, 'scan_gap')
            return dict(self.result)
        theta = speed*np.tan(float(np.mean(steering)))/.28764*dt
        prediction = np.array([speed*dt*np.cos(theta/2), speed*dt*np.sin(theta/2), theta])
        fitted = self._register(points, old_points, old_normals, prediction)
        if fitted is None:
            self.result = self._invalid(timestamp, age, 'registration_failed')
            return dict(self.result)
        transform, (residual, fraction, matches, nullspace) = fitted
        theta = float(transform[2])
        # SE(2) logarithm: compensate rotation during the interval rather than
        # confusing laser lever-arm motion or curved travel with rear sideslip.
        if abs(theta) < 1e-7:
            velocity = transform[:2]/dt
        else:
            a, b = np.sin(theta)/theta, (1-np.cos(theta))/theta
            velocity = np.linalg.solve(np.array([[a, -b], [b, a]]), transform[:2])/dt
        observable = {'longitudinal': bool(nullspace[0, 0] < .25),
                      'lateral': bool(nullspace[1, 1] < .25), 'yaw': bool(nullspace[2, 2] < .25)}
        confidence = float(np.exp(-residual/.04)*fraction*(1. if observable['longitudinal'] else .65))
        valid = bool(observable['lateral'] and observable['yaw'] and confidence >= .45)
        self.result = {'timestamp': timestamp, 'age': age, 'dt': dt,
                       'forward_speed': speed, 'lidar_forward_speed': float(velocity[0]),
                       'lateral_speed': float(velocity[1]), 'yaw_rate': theta/dt,
                       'slip_angle': float(np.arctan2(velocity[1], speed)) if abs(speed) > .15 else 0.,
                       'confidence': confidence, 'valid': valid, 'observable_axes': observable,
                       'residual_m': residual, 'inlier_fraction': fraction, 'matches': matches,
                       'reason': 'ok' if valid else 'unobservable_or_low_confidence'}
        return dict(self.result)
