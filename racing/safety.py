"""Experimental local wall supervisor using laser and joint encoders only.

This is separate from the frozen v5 actor. A short planar footprint prediction
is a conservative control heuristic, not a collision guarantee. Long, straight
scan patches are walls. Compact residual clusters constrain steering overrides;
recognized reflector geometry also provides a near-term vehicle brake trigger.
"""
from __future__ import annotations

import math
import numpy as np
from .odometry import LidarOdometry
from .safety_motion import compensate_points, predict_poses, rotation, swept_footprints, twist_displacement

DT, WHEEL_RADIUS, WHEELBASE, WHEEL_TRACK = 1/60, .045, .28764, .212
RECOVERY_SPEED = .5  # 0.2 m/horizon spans more than twice the 0.08 m rear overhang.
# A bounded reverse escape is the only exit from a stop against a wall the
# forward planner cannot see past. The fixed scan covers 270 degrees, so the
# rear quadrants are the only evidence available for the blind back-up arc.
# The rear-diagonal rays are the only ones that look mostly backwards inside
# the 270 degree field of view; the direct rear 90 degrees is unobservable and
# must never be assumed clear.
REVERSE_SPEED, REVERSE_HORIZON, REVERSE_REAR_CLEARANCE, REVERSE_REAR_MIN_DEG = .6, .5, .6, 120.
# Fitted wall lines are re-segmented between scans, and a patch that disappears
# makes the supervisor report a clear corridor while the car is still closing
# on it. Raw hits are primary evidence: they cannot be dropped by a fitting
# choice. They are distance limited so only geometry the car can still reach
# inside the braking horizon participates.
# The horizon is short on purpose: raw hits cover the whole corridor, so a
# full braking-distance sweep would veto every corner on a 1.5 m wide RC
# circuit purely because the outer wall is always reachable. The raw channel
# exists to contradict a wall fit that dropped a patch, not to re-plan corners.
NEAR_POINT_RADIUS, NEAR_POINT_MAX_RANGE, NEAR_POINT_MAX, NEAR_POINT_HORIZON = .0, 3.5, 120, .2
# Vehicle contacts in the v10a sweep happened 0.08-0.5 s after the supervisor
# first resolved the other car, so the near-term vehicle horizon is widened;
# 0.12 s was short enough that a side-by-side scrape arrived inside it.
VEHICLE_HORIZON = .18
# Threat assessment keeps the 4 m/s^2 model; the command sheds speed faster so
# the tyres brake hard while still turning. Commanding zero immediately locks
# the wheels (Isaac spun 180 degrees and drove backwards); following the live
# speed turns into a positive feedback loop and does not brake at all (MuJoCo
# Austin and Bahrain drove into walls).
BRAKE_COMMAND_DECELERATION = 12.
CENTER_LIMIT = float(np.arctan(WHEELBASE/(WHEELBASE/np.tan(.45)+WHEEL_TRACK/2)))
# Fixed source CAD calibration, not an engine/world pose observation.
LASER_POSITION = np.array([.096555277688949018, 0.])
_w, _x, _y, _z = .99969349188272216, -7.0832408137250425e-05, -.024589108572215627, .002879758622449709
LASER_ROTATION_XY = np.array([[1-2*(_y*_y+_z*_z), 2*(_x*_y-_w*_z)],
                              [2*(_x*_y+_w*_z), 1-2*(_x*_x+_z*_z)]])
FOOTPRINT = np.array([[-.08, -.14], [-.08, .14], [.40, -.14], [.40, .14],
                      [-.08, 0.], [.40, 0.], [.16, -.14], [.16, .14]])


def _wall_fits(points, valid, padding=0.):
    """Split connected scan patches until their TLS line is locally reliable."""
    allowed = valid & (points[:, 0] > -1.5-padding) & (points[:, 0] < 4.5+padding) & (abs(points[:, 1]) < 3.+padding)
    indices = np.flatnonzero(allowed)
    if not len(indices):
        return []
    cuts = (np.diff(indices) != 1) | (np.linalg.norm(np.diff(points[indices], axis=0), axis=1) > .23)
    clusters = np.split(indices, np.flatnonzero(cuts)+1)
    walls = []

    def fit(chunk, depth=0):
        if len(chunk) < 8 or depth > 12:
            return
        p = points[chunk]
        center = p.mean(axis=0)
        _, _, axes = np.linalg.svd(p-center, full_matrices=False)
        tangent, normal = axes[0], axes[1]
        residual = abs((p-center)@normal)
        span = np.ptp(p@tangent)
        if residual.max() > .035:
            split = int(np.argmax(residual))
            if split < 4 or split > len(chunk)-5:
                split = len(chunk)//2
            fit(chunk[:split+1], depth+1)
            fit(chunk[split:], depth+1)
        elif span >= .65 and np.sqrt(np.mean(residual**2)) <= .02:
            if center@normal < 0:
                normal = -normal
            extent = p@tangent
            walls.append({'normal': normal, 'offset': float(center@normal),
                          'tangent': tangent, 'minimum': float(extent.min()), 'maximum': float(extent.max()),
                          'span_m': float(span), 'rms_m': float(np.sqrt(np.mean(residual**2)))})

    for cluster in clusters:
        fit(cluster)
    return walls


def _reflector_geometry(points):
    """Infer reflector centre and longitudinal axis from a 16 cm face or L.

    The native fixture has 10 x 16 cm horizontal faces. A short arbitrary wall
    patch or a partial face cannot establish that heading and returns None.
    Fore/aft sign is unobservable here and is covered by the body envelope.
    """
    def line(p):
        center = p.mean(axis=0)
        _, _, axes = np.linalg.svd(p-center, full_matrices=False)
        return axes[0], float(np.ptp(p@axes[0])), float(np.sqrt(np.mean(((p-center)@axes[1])**2))), center
    axis, span, error, center = line(points)
    if .135 <= span <= .175 and error < .003:
        forward = np.array([-axis[1], axis[0]])
        if forward@(center-LASER_POSITION) < 0:
            forward = -forward
        return center+.05*forward, forward
    if len(points) < 7:
        return None
    fits = []
    for split in range(3, len(points)-2):
        first, second = line(points[:split]), line(points[split:])
        if abs(first[0]@second[0]) > .08 or max(first[2], second[2]) > .003:
            continue
        short, long = sorted([first, second], key=lambda item: item[1])
        if .055 <= short[1] <= .11 and .13 <= long[1] <= .175:
            fits.append((first[2]+second[2], short, long))
    if not fits:
        return None
    _, short, long = min(fits, key=lambda item: item[0])
    normals = np.array([[-short[0][1], short[0][0]], [-long[0][1], long[0][0]]])
    corner = np.linalg.solve(normals, np.array([normals[0]@short[3], normals[1]@long[3]]))
    forward = short[0]*np.sign((short[3]-corner)@short[0])
    lateral = long[0]*np.sign((long[3]-corner)@long[0])
    return corner+.05*forward+.08*lateral, forward


def _reflector_forward(points):
    geometry = _reflector_geometry(points)
    return None if geometry is None else geometry[1]


def _scan_geometry(points, valid, padding=0.):
    walls = _wall_fits(points, valid, padding)
    residual = valid & (points[:, 0] > -1.-padding) & (np.linalg.norm(points, axis=1) < 3.5+padding)
    for wall in walls:
        along = points@wall['tangent']
        residual &= ~((abs(points@wall['normal']-wall['offset']) < .05)
                      & (along >= wall['minimum']-.1) & (along <= wall['maximum']+.1))
    indices = np.flatnonzero(residual)
    clusters = np.split(indices, np.flatnonzero((np.diff(indices) != 1)
                         | (np.linalg.norm(np.diff(points[indices], axis=0), axis=1) > .2))+1)
    compact, oriented, oriented_return_count = [], [], 0
    for index in clusters:
        if len(index) < 2 or np.linalg.norm(np.ptp(points[index], axis=0)) > .45:
            continue
        cloud = points[index]
        geometry = _reflector_geometry(cloud)
        if geometry is None:
            compact.append(cloud)
        else:
            center, forward = geometry
            oriented.append(np.r_[center, forward][None, :])
            oriented_return_count += len(cloud)
    # Near-field raw returns travel with the same motion compensation as walls
    # so that a stale patch is not measured against the present footprint.
    near_mask = valid & (np.linalg.norm(points, axis=1) < NEAR_POINT_MAX_RANGE)
    near_indices = np.flatnonzero(near_mask)
    if len(near_indices) > NEAR_POINT_MAX:
        near_indices = near_indices[::int(np.ceil(len(near_indices)/NEAR_POINT_MAX))]
    return {'walls': walls,
            'obstacles': np.concatenate(compact) if compact else np.empty((0, 2)),
            'oriented': np.concatenate(oriented) if oriented else np.empty((0, 4)),
            'near': points[near_indices] if len(near_indices) else np.empty((0, 2)),
            'oriented_return_count': oriented_return_count}


def _transform_geometry(geometry, twist, age):
    """Transport oriented scan geometry without reselecting its visible side."""
    translation, yaw = twist_displacement(twist, age)
    rotate = rotation(yaw)
    walls = []
    for wall in geometry['walls']:
        shift = float(wall['tangent']@translation)
        walls.append(dict(wall, normal=wall['normal']@rotate,
                          offset=wall['offset']-float(wall['normal']@translation),
                          tangent=wall['tangent']@rotate,
                          minimum=wall['minimum']-shift, maximum=wall['maximum']-shift))
    oriented = geometry['oriented'].copy()
    if len(oriented):
        oriented[:, :2] = (oriented[:, :2]-translation)@rotate
        oriented[:, 2:] = oriented[:, 2:]@rotate
    return {'walls': walls, 'oriented': oriented,
            'obstacles': (geometry['obstacles']-translation)@rotate,
            'near': (geometry['near']-translation)@rotate,
            'oriented_return_count': geometry['oriented_return_count']}


class LocalSafetySupervisor:
    """filter(observation, wheels4, steers2) -> native signed actuator targets.

diagnostics.must_brake tells an upstream drift controller to exit overdrive.
Safety has final authority when composed after that controller. No rollout or
lap-success claim follows from an offline counterfactual filter evaluation.
"""
    version = "footprint-safety-v16-rolling-brake"

    def __init__(self, horizon=.4, margin=.03, braking_deceleration=4.):
        if not .3 <= horizon <= .5 or margin < 0 or braking_deceleration <= 0:
            raise ValueError('Expected horizon .3–.5 s, nonnegative margin and positive braking deceleration')
        self.horizon, self.margin, self.braking_deceleration = float(horizon), float(margin), float(braking_deceleration)
        self.reset()

    def reset(self):
        self.speed_bound = 0.
        self.brake_profile_speed = None
        self.brake_elapsed = 0.
        self._obstacles = np.empty((0, 2))
        self._oriented_obstacles = np.empty((0, 4))
        self._near_points = np.empty((0, 2))
        self.odometry = LidarOdometry()
        self._motion_scan_timestamp = None
        self._motion_seen_timestamp = None
        self._scan_motion = []
        self._motion_upper = np.zeros(2)
        self._previous_observed_yaw = None
        self._yaw_acceleration = 0.
        self._scenes = []
        self.motion_diagnostics = {'ready': False, 'reason': 'need_two_scans'}
        self.diagnostics = {}

    def _predict(self, walls, speed, actual, target, braking=False, response_rate=3., end_only=False, include_compact=True, horizon=None, oriented_only=False,
                 motion=None, mode='slip', yaw_bias_decay=None, command_speed=None,
                 obstacles=None, oriented=None, braking_deceleration=None,
                 near=None, near_radius=NEAR_POINT_RADIUS, include_near=True):
        # Include present front/rear corners and the entire steering transient.
        x = y = yaw = 0.
        steer = actual
        footprints = [FOOTPRINT.copy()]
        poses = [(0., 0., 0.)]
        for _ in range(0 if motion is not None else round((self.horizon if horizon is None else horizon)/.02)):
            steer += float(np.clip(target-steer, -response_rate*.02, response_rate*.02))
            curvature = math.tan(steer)/WHEELBASE
            curvature = float(np.clip(curvature, -5.5/max(speed*speed, .1), 5.5/max(speed*speed, .1)))
            yaw_mid = yaw + .5*speed*curvature*.02
            x += speed*math.cos(yaw_mid)*.02
            y += speed*math.sin(yaw_mid)*.02
            yaw += speed*curvature*.02
            c, s = math.cos(yaw), math.sin(yaw)
            footprints.append(FOOTPRINT@np.array([[c, s], [-s, c]]) + [x, y])
            poses.append((x, y, yaw))
            if braking:
                speed = max(0., speed-self.braking_deceleration*.02)
        if motion is not None:
            poses = predict_poses(motion, actual, target,
                                  horizon=self.horizon if horizon is None else horizon,
                                  response_rate=response_rate, braking=braking,
                                  braking_deceleration=self.braking_deceleration if braking_deceleration is None else braking_deceleration,
                                  mode=mode, yaw_bias_decay=yaw_bias_decay,
                                  command_speed=command_speed)
            footprints = swept_footprints(FOOTPRINT, poses)
        vertices = footprints[-1] if end_only else np.concatenate(footprints)
        obstacles = self._obstacles if obstacles is None else obstacles
        oriented = self._oriented_obstacles if oriented is None else oriented
        clearance = float('inf')
        for wall in walls:
            along = vertices@wall['tangent']
            relevant = (along >= wall['minimum']-.15) & (along <= wall['maximum']+.15)
            if relevant.any():
                # Sensor walls are centre planes; physical barriers are 4 cm wide.
                clearance = min(clearance, float(np.min(wall['offset']-vertices[relevant]@wall['normal']-.02)))
        # Raw near-field returns are the primary obstacle evidence, so they are
        # evaluated even where no reliable wall line was fitted.
        near = self._near_points if near is None else near
        if include_near and near is not None and len(near):
            pose = np.asarray(poses[-1:] if end_only else poses)
            delta = near[None, :, :]-pose[:, None, :2]
            c, s = np.cos(pose[:, 2, None]), np.sin(pose[:, 2, None])
            local = np.stack([c*delta[:, :, 0]+s*delta[:, :, 1],
                              -s*delta[:, :, 0]+c*delta[:, :, 1]], axis=-1)
            q = abs(local-[.16, 0.])-[.24, .14]
            signed = np.linalg.norm(np.maximum(q, 0.), axis=-1)+np.minimum(q.max(axis=-1), 0.)
            clearance = min(clearance, float(signed.min()-near_radius))
        if include_compact and not oriented_only and len(obstacles):
            pose = np.asarray(poses[-1:] if end_only else poses)
            delta = obstacles[None, :, :]-pose[:, None, :2]
            c, s = np.cos(pose[:, 2, None]), np.sin(pose[:, 2, None])
            local = np.stack([c*delta[:, :, 0]+s*delta[:, :, 1],
                              -s*delta[:, :, 0]+c*delta[:, :, 1]], axis=-1)
            q = abs(local-[.16, 0.])-[.24, .14]
            signed = np.linalg.norm(np.maximum(q, 0.), axis=-1)+np.minimum(q.max(axis=-1), 0.)
            # A return measures the reflector surface, not the full CAD body.
            # Reserve 0.30 m around compact returns before overriding the actor.
            clearance = min(clearance, float(signed.min()-.30))
        if include_compact and len(oriented):
            # A confidently observed 0.16 m reflector face determines the
            # opponent's transverse axis (the other face is only 0.10 m).
            # Apply CAD longitudinal/lateral extents along those measured axes.
            pose = np.asarray(poses[-1:] if end_only else poses)
            forward = np.c_[np.cos(pose[:, 2]), np.sin(pose[:, 2])]
            lateral = forward[:, ::-1]*[-1., 1.]
            center = pose[:, :2]+.16*forward
            other_forward = oriented[:, 2:]
            other_lateral = other_forward[:, ::-1]*[-1., 1.]
            delta = oriented[None, :, :2]-center[:, None, :]
            axes = [forward[:, None, :], lateral[:, None, :],
                    other_forward[None, :, :], other_lateral[None, :, :]]
            gaps = []
            for axis in axes:
                own_radius = .24*abs(np.sum(forward[:, None, :]*axis, axis=-1))+.14*abs(np.sum(lateral[:, None, :]*axis, axis=-1))
                # One body envelope around the fitted reflector centre. CAD
                # half-length .24 m + unknown fore/aft offset .06 m + .02 m
                # reserve; lateral half-width .14 m + .02 m reserve.
                other_radius = .32*abs(np.sum(other_forward[None, :, :]*axis, axis=-1))+.16*abs(np.sum(other_lateral[None, :, :]*axis, axis=-1))
                gaps.append(abs(np.sum(delta*axis, axis=-1))-own_radius-other_radius)
            clearance = min(clearance, float(np.max(gaps, axis=0).min()))
        return clearance

    def _clearance(self, walls, speed, actual, target, **options):
        """Pair every past scan transform with the same future motion model."""
        if not self._scenes:
            return self._predict(walls or [], speed, actual, target, **options)
        results = []
        for scene in self._scenes:
            for mode, decay in scene['models']:
                results.append(self._predict(scene['walls'] if walls is not None else [], speed, actual, target,
                    motion=scene['motion'], mode=mode, yaw_bias_decay=decay,
                    obstacles=scene['obstacles'], oriented=scene['oriented'],
                    braking_deceleration=0. if scene['retained_speed'] else self.braking_deceleration,
                    **options))
        return min(results)

    def _near_clearance(self, speed, actual, target, **options):
        """Raw-hit clearance over its own short horizon, independent of wall fits."""
        options.setdefault('horizon', NEAR_POINT_HORIZON)
        options.setdefault('include_compact', False)
        if not self._scenes:
            return self._predict([], speed, actual, target, near=self._near_points, **options)
        results = []
        for scene in self._scenes:
            for mode, decay in scene['models']:
                results.append(self._predict([], speed, actual, target,
                    motion=scene['motion'], mode=mode, yaw_bias_decay=decay,
                    near=scene['near'],
                    braking_deceleration=0. if scene['retained_speed'] else self.braking_deceleration,
                    **options))
        return min(results)

    def _motion_hypotheses(self, observation, points, valid, encoder_speed, raw_walls=None):
        packet = observation['lidar']
        timestamp, age = float(packet.get('timestamp', -1.)), float(packet.get('age', 0.))
        if not math.isfinite(timestamp) or (self._motion_seen_timestamp is not None and timestamp < self._motion_seen_timestamp):
            self.motion_diagnostics = {'ready': False, 'reason': 'invalid_or_out_of_order_scan_timestamp'}
            return []
        self._motion_seen_timestamp = timestamp
        estimate = self.odometry.update(observation)
        axes, confidence = estimate['observable_axes'], estimate['confidence']
        yaw_usable = bool(axes['yaw'] and confidence >= .45 and math.isfinite(estimate['yaw_rate']))
        fit_usable = bool(estimate.get('matches', 0) >= 24 and estimate.get('residual_m') is not None
                          and estimate['residual_m'] < .04 and estimate.get('inlier_fraction', 0.) >= .5)
        if not yaw_usable or not fit_usable:
            self.motion_diagnostics = {'ready': False, 'reason': estimate['reason'], 'estimate': estimate}
            return []
        if timestamp != self._motion_scan_timestamp:
            dt = float(estimate['dt'])
            accepted_gap = dt if self._motion_scan_timestamp is None else timestamp-self._motion_scan_timestamp
            observed_yaw = float(estimate['yaw_rate'])
            previous_yaw = observed_yaw if self._previous_observed_yaw is None else self._previous_observed_yaw
            if self._previous_observed_yaw is not None and accepted_gap > 0:
                self._yaw_acceleration = (observed_yaw-self._previous_observed_yaw)/accepted_gap
            self._previous_observed_yaw = observed_yaw
            raw_velocity = np.array([estimate['lidar_forward_speed'], estimate['lateral_speed']], float)
            if not np.isfinite(raw_velocity).all():
                self.motion_diagnostics = {'ready': False, 'reason': 'nonfinite_scan_velocity'}
                return []
            raw_walls = _wall_fits(points, valid) if raw_walls is None else raw_walls
            velocities = [raw_velocity]
            retained = not (axes['longitudinal'] and axes['lateral'])
            if retained:
                # Transport the previous velocity hypothesis into this local
                # frame; no global pose is stored. Correct only wall-normal
                # velocity components that the current scan actually measures.
                historical = self._motion_upper.copy()
                upper = historical@rotation(.5*(observed_yaw+previous_yaw)*accepted_gap)
                upper[0] = max(float(upper[0]), encoder_speed)
                if raw_walls:
                    normal = max(raw_walls, key=lambda wall: wall['span_m'])['normal']
                    upper += normal*(raw_velocity@normal-upper@normal)
                velocities.append(upper)
                if accepted_gap > 1.5*dt:
                    # Intermediate rejected scans do not establish the full yaw
                    # integral. Retain the untransported history as a separate
                    # hypothesis, rather than treating the gap as one fit dt.
                    if raw_walls:
                        historical += normal*(raw_velocity@normal-historical@normal)
                    velocities.append(historical)
                    upper = max([upper, historical], key=np.linalg.norm)
                self._motion_upper = upper.copy()
            else:
                self._motion_upper = raw_velocity.copy()
            self._scan_motion = [(velocity.copy(), retained and index > 0)
                                 for index, velocity in enumerate(velocities)]
            self._motion_scan_timestamp = timestamp
            self._accepted_motion_gap = accepted_gap
        observed_yaw = float(estimate['yaw_rate'])
        dt = float(estimate['dt'])
        yaw_cases = [(observed_yaw, observed_yaw)]
        current_yaw = observed_yaw+self._yaw_acceleration*(.5*dt+age)
        mean_yaw = observed_yaw+self._yaw_acceleration*(.5*dt+.5*age)
        if abs(current_yaw-observed_yaw) > .05:
            yaw_cases.append((mean_yaw, current_yaw))
        hypotheses = []
        for velocity, retained in self._scan_motion:
            for scan_yaw, future_yaw in yaw_cases:
                lock = float(np.linalg.norm(velocity)) > 1.2 and encoder_speed < .7*float(np.linalg.norm(velocity))
                models = [('slip', None)]
                if abs(future_yaw) > .5 or abs(float(velocity[1])) > .15:
                    models.append(('slip', .4))
                    if not lock:
                        models.append(('slip', .05))
                if lock:
                    models.append(('coast', None))
                hypotheses.append({'scan_motion': [*velocity, scan_yaw],
                                   'motion': [*velocity, future_yaw], 'models': models,
                                   'retained_speed': retained})
        self.motion_diagnostics = {'ready': True, 'reason': 'axis_aware_local_motion', 'estimate': estimate,
            'historical_velocity_hypothesis_m_s': self._motion_upper.tolist(),
            'history_is_stationary_proof': False, 'hypothesis_count': len(hypotheses),
            'yaw_acceleration_hypothesis_rad_s2': self._yaw_acceleration,
            'accepted_scan_gap_s': self._accepted_motion_gap,
            'scan_age_integrated_s': age,
            'hypotheses': hypotheses}
        return hypotheses

    def filter(self, observation, wheel_targets, steering_targets):
        wheels, steers = np.asarray(wheel_targets, float), np.asarray(steering_targets, float)
        measured = np.asarray(observation['wheel_vel'], float)
        positions = np.asarray(observation['steer_pos'], float)
        if wheels.shape != (4,) or steers.shape != (2,) or measured.shape != (4,) or positions.shape != (2,):
            raise ValueError('Expected four wheel values and two steering values')
        if not np.isfinite(np.r_[wheels, steers, measured, positions]).all():
            raise ValueError('Actuator values must be finite')
        speed = max(0., float(np.mean(measured[:2]*np.cos(positions))*WHEEL_RADIUS))
        # Locked/spinning encoders cannot prove that the body stopped immediately.
        self.speed_bound = max(speed, self.speed_bound-self.braking_deceleration*DT)
        actual = float(np.clip(np.mean(positions), -CENTER_LIMIT, CENTER_LIMIT))
        target = float(np.clip(np.mean(steers), -CENTER_LIMIT, CENTER_LIMIT))
        packet = observation['lidar']
        angles, ranges = np.asarray(packet['angles'], float), np.asarray(packet['ranges'], float)
        valid = np.asarray(packet.get('valid', packet.get('validmask', np.ones_like(ranges))), bool).copy()
        if ranges.ndim != 1 or angles.shape != ranges.shape or valid.shape != ranges.shape:
            raise ValueError('Malformed laser packet')
        valid &= np.isfinite(ranges) & np.isfinite(angles) & (ranges > .04) & (ranges < 14.99)
        raw_age = float(packet.get('age', 0.))
        # Initial latency packets have age=inf and no valid rays. Do not apply
        # motion compensation or trigonometry until a usable scan arrives.
        if not math.isfinite(raw_age) or raw_age < 0 or raw_age > .14 or not valid.any():
            self.diagnostics = {'intervened': True, 'must_brake': True,
                                'reason': 'stale_or_empty_scan', 'recovering': False,
                                'wall_fit_count': 0, 'speed_bound_m_s': self.speed_bound,
                                'measured_encoder_speed_m_s': speed,
                                'current_steer_clearance_m': None, 'requested_clearance_m': None,
                                'selected_clearance_m': None, 'requested_center_steer_rad': target,
                                'selected_center_steer_rad': actual, 'predicted_feasible': False,
                                'horizon_s': self.horizon, 'margin_m': self.margin,
                                'scope': 'local planar scan/encoder prediction; not a native rollout result'}
            # A stale scan removes the evidence, not the momentum: roll down
            # instead of locking every wheel at speed.
            return np.zeros(4), np.clip(positions, -.45, .45)
        points = np.c_[np.nan_to_num(ranges)*np.cos(angles), np.nan_to_num(ranges)*np.sin(angles)]@LASER_ROTATION_XY.T + LASER_POSITION
        age = raw_age
        raw_walls = _wall_fits(points, valid)
        hypotheses = self._motion_hypotheses(observation, points, valid, speed, raw_walls)
        if not hypotheses:
            # Missing motion is not evidence of r=vy=0. Hold the last commanded
            # safe steering (if any) and brake; never silently resume bicycle
            # predictions while a sliding car's translation is unobservable.
            held = self.diagnostics.get('selected_center_steer_rad', actual)
            curvature = math.tan(float(held))/WHEELBASE
            result_steers = np.clip(np.arctan2(WHEELBASE*curvature,
                [1-WHEEL_TRACK*curvature/2, 1+WHEEL_TRACK*curvature/2]), -.45, .45)
            self.diagnostics = {'intervened': True, 'must_brake': True, 'reason': 'motion_estimate_unavailable',
                'recovering': False, 'wall_fit_count': 0, 'compact_return_points': 0,
                'compact_oriented_points': 0, 'speed_bound_m_s': self.speed_bound,
                'measured_encoder_speed_m_s': speed, 'current_steer_clearance_m': None,
                'requested_clearance_m': None, 'selected_clearance_m': None,
                'requested_center_steer_rad': target, 'selected_center_steer_rad': float(held),
                'predicted_feasible': False, 'horizon_s': self.horizon, 'margin_m': self.margin,
                'motion': self.motion_diagnostics}
            return np.zeros(4), result_steers
        # The current ROI must not clip old returns that move into it during
        # packet age. Its furthest corner is <5.5 m from the rear axle. Translation
        # plus the rotational chord bounds displacement of every current ROI point.
        # Fit in the original scan frame with that padding, retaining visible sides.
        padding = max(float(np.linalg.norm(twist_displacement(h['scan_motion'], age)[0]))
                      + 11.*abs(math.sin(twist_displacement(h['scan_motion'], age)[1]/2))
                      for h in hypotheses)
        geometry = _scan_geometry(points, valid, padding=padding)
        self.motion_diagnostics['raw_geometry_padding_m'] = padding
        self._scenes = []
        for hypothesis in hypotheses:
            scene = _transform_geometry(geometry, hypothesis['scan_motion'], age)
            scene.update(hypothesis)
            self._scenes.append(scene)
        representative = self._scenes[0]
        walls = representative['walls']
        self._obstacles, self._oriented_obstacles = representative['obstacles'], representative['oriented']
        self._near_points = representative['near']
        oriented_return_count = representative['oriented_return_count']
        stale = age > .14 or not valid.any()
        current_clearance = self._clearance(walls, self.speed_bound, actual, actual, include_compact=False)
        requested_clearance = self._clearance(walls, self.speed_bound, actual, target, include_compact=False)
        # Also model slower physical steering tracking (observed in native Isaac).
        requested_clearance = min(requested_clearance, self._clearance(walls, self.speed_bound, actual, target, response_rate=1.5, include_compact=False))
        # The nominal racing actor handles moving vehicles. Compact returns
        # constrain our replacement actions; treating a moving opponent as a
        # stationary obstacle for every nominal action prevents ordinary passes.
        wall_danger = bool(any(scene['walls'] for scene in self._scenes) and requested_clearance < self.margin)
        # A patch the line fit dropped must not turn into a clear corridor, so
        # the raw-hit channel can veto the wall verdict on its own.
        near_clearance = min(self._near_clearance(self.speed_bound, actual, target),
                             self._near_clearance(self.speed_bound, actual, target, response_rate=1.5))
        # Only confidently measured reflector geometry triggers this near-term
        # emergency. Treating all residual points as stationary cars for .4 s
        # produced slow following and false stops beside ordinary wall patches.
        compact_clearance = min(self._clearance(None, self.speed_bound, actual, target, horizon=VEHICLE_HORIZON, oriented_only=True),
                                self._clearance(None, self.speed_bound, actual, target, response_rate=1.5, horizon=VEHICLE_HORIZON, oriented_only=True))
        compact_danger = bool(any(len(scene['oriented']) for scene in self._scenes) and compact_clearance < self.margin)
        near_danger = bool(near_clearance < self.margin)
        danger = wall_danger or compact_danger or near_danger
        result_wheels, result_steers = wheels.copy(), steers.copy()
        selected, selected_clearance = target, requested_clearance
        must_brake = danger or stale
        if must_brake:
            # Latch a deceleration profile instead of following the live speed.
            # Following it is a positive feedback loop: the command stays at
            # the measured speed while the wheels keep rolling.
            if self.brake_profile_speed is None:
                self.brake_profile_speed, self.brake_elapsed = max(self.speed_bound, speed), 0.
        else:
            self.brake_profile_speed, self.brake_elapsed = None, 0.
        recovering = False
        vehicle_creep = False
        preparing_steering = False
        recovery_present = recovery_end = None
        if must_brake:
            candidates = np.unique(np.r_[np.linspace(-CENTER_LIMIT, CENTER_LIMIT, 25), actual, target, 0.])
            scored = []
            for candidate in candidates:
                clearance = min(self._clearance(walls, self.speed_bound, actual, candidate, braking=True),
                                self._clearance(walls, self.speed_bound, actual, candidate, braking=True, response_rate=1.5))
                end_clearance = min(self._clearance(walls, self.speed_bound, actual, candidate, braking=True, end_only=True),
                                    self._clearance(walls, self.speed_bound, actual, candidate, braking=True, response_rate=1.5, end_only=True))
                scored.append((clearance, float(candidate), end_clearance))
            feasible = [score for score in scored if score[0] >= self.margin]
            preserve_proposal = False
            if feasible:
                # A feasible braking version of the actor's countersteer must
                # not lose to a larger wall gap on the opposite steering side.
                # Preserve it, or choose the nearest feasible correction. Full
                # motion/footprint constraints remain identical for all choices.
                selected_clearance, selected, _ = min(feasible, key=lambda p: (abs(p[1]-target), -p[0]))
                preserve_proposal = abs(selected-target) < 1e-12
            else:
                selected_clearance, selected, _ = max(scored, key=lambda p: (p[0], min(p[2], .5), -abs(p[1]-actual)))
            curvature = math.tan(selected)/WHEELBASE
            result_steers = np.clip(np.arctan2(WHEELBASE*curvature,
                                              [1-WHEEL_TRACK*curvature/2, 1+WHEEL_TRACK*curvature/2]), -.45, .45)
            if preserve_proposal:
                result_steers = steers.copy()
            # Brake by rolling the wheels down at the modelled deceleration.
            # Zeroing them locks the wheels, and a locked RC car that is still
            # yawing spins: Isaac Bahrain reached 3.6 rad/s with -57 degrees of
            # sideslip, turned 180 degrees, and then drove the wrong way for
            # -86.8 m of progress.
            rolling = max(0., self.brake_profile_speed-BRAKE_COMMAND_DECELERATION*self.brake_elapsed)
            self.brake_elapsed += DT
            rolling /= WHEEL_RADIUS
            result_wheels = np.array([rolling, rolling, rolling, -rolling])
            # A present footprint already inside the reserve margin makes every
            # braking prediction infeasible forever, even after the car stops.
            # Permit slow forward escape only if the swept footprint stays
            # clear (retaining half its positive gap) and the endpoint improves.
            # Existing shallow overlap has the stricter non-worsening rule.
            # This does not relax the physical wall boundary or use world pose.
            if danger and not stale and speed < RECOVERY_SPEED+.1:
                present = self._clearance(walls, 0., actual, actual, horizon=0.)
                # Being stopped by a vehicle envelope is not the same failure
                # as being stopped by a wall: the envelope is an inference
                # around one 16 cm reflector, and in a 1.5 m corridor both cars
                # can end up overlapping it while neither can move (MuJoCo
                # Singapore and Spa). Walls still bound the manoeuvre; the
                # vehicle envelope only has to avoid getting clearly worse.
                present_walls = self._clearance(walls, 0., actual, actual, horizon=0., include_compact=False)
                vehicle_only = bool(compact_danger and not wall_danger)
                escapes = []
                # The conservative rectangle can overlap by a few millimetres
                # while the actual shaped CAD body remains clear. A shallow
                # existing overlap may only shrink, and must fully clear by
                # the horizon; larger overlap retains the stop command.
                if present >= -.01 or (vehicle_only and present_walls >= self.margin):
                    for candidate in candidates:
                        gap = self._clearance(walls, RECOVERY_SPEED, actual, candidate, response_rate=1.5, command_speed=RECOVERY_SPEED)
                        end_gap = self._clearance(walls, RECOVERY_SPEED, actual, candidate, response_rate=1.5, command_speed=RECOVERY_SPEED, end_only=True)
                        if vehicle_only:
                            wall_gap = self._clearance(walls, RECOVERY_SPEED, actual, candidate, response_rate=1.5,
                                                       command_speed=RECOVERY_SPEED, include_compact=False)
                            if wall_gap >= self.margin and gap >= present-.05:
                                escapes.append((gap, wall_gap, float(candidate)))
                            continue
                        required_end = .001 if present < 0. else present+.0005
                        required_gap = present if present < 0. else present*.5
                        if gap >= required_gap-1e-6 and end_gap > required_end:
                            escapes.append((end_gap, gap, float(candidate)))
                if escapes:
                    recovery_end, selected_clearance, selected = max(escapes)
                    recovery_present = present
                    vehicle_creep = bool(vehicle_only)
                    curvature = math.tan(selected)/WHEELBASE
                    result_steers = np.clip(np.arctan2(WHEELBASE*curvature,
                                                      [1-WHEEL_TRACK*curvature/2, 1+WHEEL_TRACK*curvature/2]), -.45, .45)
                    result_wheels = np.array([1., 1., 1., -1.])*(RECOVERY_SPEED/WHEEL_RADIUS)
                    recovering = True
                elif present >= -.01:
                    # Starting to creep before the steering joint settles may
                    # initially deepen a shallow footprint overlap. A steering
                    # preparation action keeps zero wheel targets; its actual
                    # braking sweep must be safe for every motion hypothesis.
                    # A hypothetical settled-wheel escape only ranks headings;
                    # acceleration is rechecked on a later real observation.
                    preparation = []
                    required_gap = present if present < 0. else present*.5
                    required_end = .001 if present < 0. else present+.0005
                    for candidate in candidates:
                        braking_gap = self._clearance(walls, self.speed_bound, actual, candidate, braking=True, response_rate=1.5)
                        if braking_gap < required_gap-1e-6:
                            continue
                        settled_gap = self._clearance(walls, RECOVERY_SPEED, candidate, candidate,
                            response_rate=1.5, command_speed=RECOVERY_SPEED)
                        settled_end = self._clearance(walls, RECOVERY_SPEED, candidate, candidate,
                            response_rate=1.5, command_speed=RECOVERY_SPEED, end_only=True)
                        if settled_gap >= required_gap-1e-6 and settled_end > required_end:
                            preparation.append((settled_end, braking_gap, float(candidate)))
                    if preparation:
                        _, selected_clearance, selected = max(preparation)
                        curvature = math.tan(selected)/WHEELBASE
                        result_steers = np.clip(np.arctan2(WHEELBASE*curvature,
                            [1-WHEEL_TRACK*curvature/2, 1+WHEEL_TRACK*curvature/2]), -.45, .45)
                        preparing_steering = True
        # The actor may request a bounded reverse escape when its forward view
        # is blocked inside the emergency stop distance. Nothing is trusted
        # from the request itself: permission requires visible rear-diagonal
        # returns plus a backwards swept footprint inside the reserve margin.
        commanded_reverse = bool(float(np.mean(wheels[:2])) < -1e-9)
        reversing = False
        reverse_rear_clearance = None
        reverse_present_clearance = reverse_best_gap = None
        if commanded_reverse:
            result_wheels = np.zeros(4)
            if not stale:
                rear = valid & (np.abs(angles) >= np.deg2rad(REVERSE_REAR_MIN_DEG))
                reverse_rear_clearance = float(np.min(ranges[rear])) if rear.any() else None
                if reverse_rear_clearance is not None and reverse_rear_clearance >= REVERSE_REAR_CLEARANCE:
                    candidates = np.unique(np.r_[np.linspace(-CENTER_LIMIT, CENTER_LIMIT, 25), actual, target, 0.])
                    # A car already inside the reserve margin can never satisfy
                    # an absolute gap, so permission requires the backing arc
                    # to improve the gap and never to shrink it. A car with
                    # normal clearance only has to keep the whole arc clear.
                    # Physical walls decide the manoeuvre. A vehicle envelope
                    # beside the car cannot improve while backing out of a
                    # side-by-side deadlock, so it only has to avoid getting
                    # clearly worse.
                    reverse_present_clearance = self._clearance(walls, 0., actual, actual, horizon=0., include_compact=False)
                    present_all = self._clearance(walls, 0., actual, actual, horizon=0.)
                    in_contact = reverse_present_clearance < self.margin
                    escapes = []
                    for candidate in candidates:
                        gap = self._clearance(walls, -REVERSE_SPEED, actual, float(candidate),
                                              horizon=REVERSE_HORIZON, response_rate=1.5, include_compact=False,
                                              command_speed=-REVERSE_SPEED)
                        end_gap = self._clearance(walls, -REVERSE_SPEED, actual, float(candidate),
                                                  horizon=REVERSE_HORIZON, response_rate=1.5, end_only=True,
                                                  include_compact=False, command_speed=-REVERSE_SPEED)
                        gap_all = self._clearance(walls, -REVERSE_SPEED, actual, float(candidate),
                                                  horizon=REVERSE_HORIZON, response_rate=1.5,
                                                  command_speed=-REVERSE_SPEED)
                        reverse_best_gap = gap if reverse_best_gap is None else max(reverse_best_gap, gap)
                        permitted = (gap >= self.margin if not in_contact else
                                     gap >= reverse_present_clearance-1e-3 and end_gap > max(reverse_present_clearance, 0.)+.02)
                        if permitted and gap_all >= present_all-.05:
                            escapes.append((gap, -abs(float(candidate)-actual), float(candidate)))
                    if escapes:
                        selected_clearance, _, selected = max(escapes)
                        curvature = math.tan(selected)/WHEELBASE
                        result_steers = np.clip(np.arctan2(WHEELBASE*curvature,
                            [1-WHEEL_TRACK*curvature/2, 1+WHEEL_TRACK*curvature/2]), -.45, .45)
                        result_wheels = np.array([-1., -1., -1., 1.])*(REVERSE_SPEED/WHEEL_RADIUS)
                        reversing = True
        number = lambda value: float(value) if math.isfinite(value) else None
        self.diagnostics = {'intervened': bool(must_brake or commanded_reverse), 'must_brake': must_brake,
                            'reason': 'verified_reverse_escape' if reversing else ('reverse_not_verified' if commanded_reverse else ('slow_forward_vehicle_creep' if vehicle_creep else ('slow_forward_wall_escape' if recovering else ('steering_preparation' if preparing_steering else ('stale_or_empty_scan' if stale else ('near_term_vehicle_footprint' if compact_danger else ('predicted_wall_footprint' if wall_danger else 'clear'))))))),
                            'recovering': recovering,
                            'vehicle_creep': vehicle_creep,
                            'preparing_steering': preparing_steering,
                            'commanded_reverse': commanded_reverse,
                            'reversing': reversing,
                            'reverse_rear_clearance_m': number(reverse_rear_clearance) if reverse_rear_clearance is not None else None,
                            'reverse_present_clearance_m': number(reverse_present_clearance) if reverse_present_clearance is not None else None,
                            'reverse_best_gap_m': number(reverse_best_gap) if reverse_best_gap is not None else None,
                            'recovery_present_clearance_m': recovery_present,
                            'recovery_end_clearance_m': recovery_end,
                            'wall_fit_count': len(walls), 'speed_bound_m_s': self.speed_bound,
                            'near_point_count': int(len(self._near_points)),
                            'near_clearance_m': number(near_clearance),
                            'near_danger': near_danger,
                            'compact_return_points': len(self._obstacles)+oriented_return_count,
                            'compact_oriented_points': oriented_return_count,
                            'oriented_vehicle_count': len(self._oriented_obstacles),
                            'near_term_vehicle_clearance_m': number(compact_clearance),
                            'vehicle_horizon_s': VEHICLE_HORIZON,
                            'measured_encoder_speed_m_s': speed,
                            'current_steer_clearance_m': number(current_clearance),
                            'requested_clearance_m': number(requested_clearance),
                            'requested_clearance_scope': 'walls',
                            'selected_clearance_scope': 'walls_and_compact_returns' if must_brake else 'walls',
                            'selected_clearance_m': number(selected_clearance),
                            'requested_center_steer_rad': target, 'selected_center_steer_rad': selected,
                            'predicted_feasible': bool(reversing or (walls and selected_clearance >= self.margin)),
                            'horizon_s': self.horizon, 'margin_m': self.margin,
                            'motion': self.motion_diagnostics,
                            'scope': 'local planar scan/encoder prediction; not a native rollout result'}
        return result_wheels, result_steers
