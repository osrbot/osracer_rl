"""Trainable reactive racing actor. Inputs are laser and joint encoders only."""
from __future__ import annotations
import numpy as np

DT, RW, WB, TRACK, STEER_LIMIT = 1/60, .045, .28764, .212, .45
SIGNS = np.array([1., 1., 1., -1.])
CENTER_LIMIT = np.arctan(WB/(WB/np.tan(STEER_LIMIT)+TRACK/2))
# A forward-only car that noses inside the emergency stop distance stays
# parked forever. These bounds define a short, sensor-only reverse escape that
# retraces the entry arc, then hands control back to the gap planner.
ESCAPE_SPEED, ESCAPE_SECONDS, ESCAPE_COOLDOWN = .7, .8, 1.2
ESCAPE_CLEARANCE, ESCAPE_STUCK_SECONDS, ESCAPE_SPEED_EPS = .30, .35, .15
# Backing away from a wall the planner has already hit leaves the car angled
# at it. Resuming at full pace re-creates the collision, so the first moments
# after an escape stay slow enough for the gap planner to steer away.
CAUTIOUS_SPEED, CAUTIOUS_SECONDS, ESCAPE_EXIT_CLEARANCE = 1., .8, .9
# A car that is backed off again the moment it touches a wall walks backwards
# around the circuit: Isaac Bahrain accumulated -86.8 m of progress that way.
# After one escape the car must earn real forward distance before another.
ESCAPE_PROGRESS_REQUIRED, ESCAPE_MAX_PER_EPISODE = 1.5, 6
POLICY_VERSION = 'gap-center-v6-verified-escape'


class RacingPolicy:
    version = POLICY_VERSION
    names = ['cruise_m_s', 'lookahead_base_m', 'lookahead_speed_s', 'steering_gain',
             'heading_penalty', 'safety_radius_m', 'lateral_accel_m_s2',
             'brake_accel_m_s2', 'centering_gain', 'steer_smoothing',
             'rear_overdrive_ratio', 'drift_trigger_rad']
    initial = np.array([7.2, .85, .26, 1.05, .65, .21, 5.5, 5., .10, .15, 0., .30])
    lower = np.array([5., .45, .08, .65, .1, .145, 2.5, 3., 0., 0., 0., .18])
    upper = np.array([10., 1.8, .50, 1.8, 1.8, .28, 12., 12., .6, .8, 2.5, .39])
    spread = np.array([.7, .2, .07, .15, .2, .025, 1.5, 1.5, .08, .12, .05, .04])

    def __init__(self, parameters=None):
        if isinstance(parameters, dict):
            parameters = [parameters.get(k, v) for k, v in zip(self.names, self.initial)]
        self.p = self.initial.copy() if parameters is None else np.asarray(parameters, float).copy()
        if self.p.shape != self.initial.shape or not np.all(np.isfinite(self.p)):
            raise ValueError('Expected one finite parameter per RacingPolicy.names entry')
        self.p = np.clip(self.p, self.lower, self.upper)
        self.reset()

    def reset(self):
        self.last_speed = self.last_steer = 0.
        self.last_timestamp = None
        self.target_speed = self.target_steer = 0.
        self.phase = 'starting'
        self.drift_seconds = self.drift_cooldown = 0.
        self.near_compact_obstacle = False
        self.forward_clearance = 15.
        self.stuck_seconds = self.escape_remaining = self.escape_cooldown = 0.
        self.cautious_seconds = 0.
        self.escape_exit_on_clearance = False
        self.escape_progress = ESCAPE_PROGRESS_REQUIRED
        self.escapes = 0

    def request_reverse_escape(self):
        """Enter one bounded reverse escape on an explicit external request.

        The planner only sees the laser; a supervisor that keeps refusing the
        commanded action because of a moving obstacle leaves the car standing.
        That refusal is real sensor evidence and may ask for the same bounded
        back-off the planner would have chosen for itself.
        """
        if self.escape_remaining > 0. or self.escape_cooldown > 0.:
            return False
        if self.escapes >= ESCAPE_MAX_PER_EPISODE or self.escape_progress < ESCAPE_PROGRESS_REQUIRED:
            return False
        self.escape_progress, self.escapes = 0., self.escapes+1
        self.escape_remaining, self.escape_cooldown = ESCAPE_SECONDS, ESCAPE_COOLDOWN
        self.escape_exit_on_clearance = False
        self.drift_seconds = 0.
        return True

    def refund_reverse_escape(self):
        """Return the allowance when the supervisor refused the manoeuvre.

        The actor requests the back-off before the supervisor can veto it, so
        a refused attempt would otherwise burn the episode allowance while the
        car never moved: MuJoCo Spa sat behind an opponent envelope at 13 m
        with ``escape_progress`` already spent and no way to earn it back.
        """
        if self.escape_remaining <= 0.:
            return False
        self.escape_remaining = self.escape_cooldown = 0.
        self.escape_progress = ESCAPE_PROGRESS_REQUIRED
        self.escapes = max(0, self.escapes-1)
        return True

    def _plan(self, packet, speed):
        cruise, base, speedlook, gain, penalty, radius, latacc, brake, centering, smooth, _, _ = self.p
        angles = np.asarray(packet['angles'], float)
        ranges = np.asarray(packet['ranges'], float)
        valid = np.asarray(packet.get('valid', packet.get('validmask', np.ones_like(ranges))), bool)
        if ranges.ndim != 1 or angles.shape != ranges.shape or valid.shape != ranges.shape:
            raise ValueError('Malformed single-plane laser packet')
        if not np.any(valid):
            # Max-range returns do not establish a safely observable corridor.
            self.near_compact_obstacle = True
            # Blind data must not trigger the reverse escape.
            self.forward_clearance = 15.
            return 0., self.last_steer
        ranges = np.nan_to_num(ranges, nan=0., posinf=15., neginf=0.)
        ranges = np.clip(ranges, 0., 15.)
        sector = abs(angles) < np.deg2rad(100)
        a, r = angles[sector], ranges[sector]
        # Compact disconnected returns can be a narrow reflector mounted on a
        # larger vehicle. Inflate their inferred envelope longitudinally too:
        # seeing the reflector pass the laser does not mean the bodies cleared.
        # Classification uses measured point geometry only, including rear-side
        # returns in the full scan; no opponent identity or pose is supplied.
        points = ranges[:, None]*np.column_stack((np.cos(angles), np.sin(angles)))
        jumps = np.linalg.norm(np.diff(points, axis=0), axis=1) > .25
        clusters = np.split(np.arange(len(ranges)), np.flatnonzero(jumps)+1)
        extra = []
        compact_clouds = []
        self.near_compact_obstacle = False
        for cluster in clusters:
            if (len(cluster) >= 2 and np.all(valid[cluster]) and
                    np.max(np.ptp(points[cluster], axis=0)) < .32 and
                    np.median(ranges[cluster]) < 8.):
                self.near_compact_obstacle |= bool(np.median(ranges[cluster]) < 5.)
                compact_clouds.append(points[cluster])
                for longitudinal in [-.28, 0., .28]:
                    extra.append(points[cluster]+[longitudinal, 0.])
        obstacle_angles, obstacle_ranges = angles, ranges
        obstacle_radius = np.full(len(ranges), radius)
        if extra:
            extra = np.concatenate(extra)
            obstacle_angles = np.r_[angles, np.arctan2(extra[:, 1], extra[:, 0])]
            obstacle_ranges = np.r_[ranges, np.linalg.norm(extra, axis=1)]
            obstacle_radius = np.r_[obstacle_radius, np.full(len(extra), max(radius, .30))]
        angular_radius = np.arcsin(np.clip(obstacle_radius/np.maximum(obstacle_ranges, obstacle_radius), 0, 1))
        blocked = abs(a[:, None]-obstacle_angles[None, :]) <= angular_radius[None, :]
        free = np.min(np.where(blocked, np.maximum(obstacle_ranges[None, :]-obstacle_radius, 0), np.inf), axis=1)
        look = base + speedlook*speed
        # Aim at the centre of a connected drivable gap, rather than its ray
        # nearest the current heading (which causes late wall corrections).
        feasible = free >= min(look, .85*float(np.max(free)))
        indices = np.flatnonzero(feasible)
        gaps = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
        desired_heading = np.arctan(np.tan(self.last_steer)*look/WB)
        candidates = []
        for gap in gaps:
            if not len(gap):
                continue
            midpoint = (a[gap[0]]+a[gap[-1]])/2
            midpoint_index = int(gap[len(gap)//2])
            score = min(free[midpoint_index], look)/(1+penalty*abs(midpoint))
            score += .10*(a[gap[-1]]-a[gap[0]])-.05*abs(midpoint-desired_heading)
            candidates.append((score, midpoint_index, midpoint))
        _, best, heading = max(candidates)
        distance = max(.3, min(look, free[best]))
        target_x, target_y = distance*np.cos(heading)+.0966, distance*np.sin(heading)
        curvature = 2*target_y/max(target_x*target_x+target_y*target_y, .1)
        left = r[(a > .95) & (a < 1.5)]
        right = r[(a < -.95) & (a > -1.5)]
        center = 0.
        if len(left) and len(right):
            l, rr = np.percentile(left, 35), np.percentile(right, 35)
            center = np.clip(l-rr, -.6, .6)
        # A local straight-wall fit provides measured corridor heading. Reject
        # bent/short wall patches so hairpin geometry cannot masquerade as a
        # reliable heading measurement.
        x, y = r*np.cos(a), r*np.sin(a)
        headings = []
        for side in [-1, 1]:
            mask = (x > -.3) & (x < 1.6) & (side*y > .15) & (side*y < 1.3)
            if np.count_nonzero(mask) >= 8 and np.ptp(x[mask]) > .7:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                if np.std(y[mask]-(slope*x[mask]+intercept)) < .055:
                    headings.append(np.arctan(slope))
        wall_heading = float(np.median(headings)) if headings else 0.
        steer = gain*np.arctan(WB*curvature) + centering*(wall_heading+center/max(look, 1.))
        # Compact returns beside the car sit outside the forward gap search, so
        # a side-by-side scrape is invisible to the steering choice. Add a
        # bounded repulsion away from a vehicle that resolves to the side.
        for cloud in compact_clouds:
            centre = np.median(cloud, axis=0)
            distance = float(np.hypot(centre[0], centre[1]))
            bearing = float(np.arctan2(centre[1], centre[0]))
            if distance < 1.2 and abs(bearing) > np.deg2rad(55):
                lateral_bias = -.12*(1.2-distance)/1.2*np.sign(bearing)
                steer += lateral_bias
        steer = (1-smooth)*steer + smooth*self.target_steer
        steer = float(np.clip(steer, -CENTER_LIMIT, CENTER_LIMIT))
        # Use forward visibility to brake before the next scan rather than
        # allowing the steering lookahead to hide an approaching hairpin.
        front = r[abs(a) < .10]
        forward = float(np.percentile(front, 20)) if len(front) else 0.
        effective_curve = max(abs(curvature), abs(np.tan(steer)/WB))
        curve_speed = np.sqrt(latacc/max(effective_curve, .02))
        stop_speed = np.sqrt(2*brake*max(forward-.5-speed/15, 0.))
        gap_speed = np.sqrt(2*brake*max(free[best]-.3, 0.))
        target = min(cruise, curve_speed, stop_speed, gap_speed)
        if forward < .24:
            target = 0.
        self.forward_clearance = float(forward)
        return float(max(0., target)), steer

    def action(self, observation):
        # Deliberately enumerate the permitted inputs. Unknown extras never
        # influence action and are never forwarded into planning.
        wheel_vel = np.asarray(observation['wheel_vel'], float).reshape(4)
        steer_pos = np.asarray(observation['steer_pos'], float).reshape(2)
        packet = observation['lidar']
        # Rear wheels intentionally spin faster during overdrive. Project the
        # measured front encoders into the forward axis so that commanded rear
        # slip cannot falsely increase the speed used for lookahead/braking.
        # The signed value stays available because a reverse escape rolls the
        # front encoders backwards.
        measured_forward = float(np.mean(wheel_vel[:2]*np.cos(steer_pos))*RW)
        measured_speed = float(np.clip(measured_forward, 0, 15))
        timestamp = float(packet['timestamp'])
        if timestamp != self.last_timestamp:
            self.target_speed, self.target_steer = self._plan(packet, measured_speed)
            self.last_timestamp = timestamp
        fresh = float(packet['age']) <= .14
        target = self.target_speed
        if not fresh:
            target = 0.
        self.escape_cooldown = max(0., self.escape_cooldown-DT)
        if measured_forward > 0.:
            self.escape_progress += measured_forward*DT
        # Only a stationary car whose forward view is inside the emergency
        # stop distance is trapped: driving on would re-enter the same wall.
        # A shallow-angle contact can leave the front rays longer than the
        # footprint gap, so the planner's own stop command is the primary
        # signal and the frontal range is the secondary one.
        stuck = (fresh and abs(measured_forward) < ESCAPE_SPEED_EPS
                 and (target <= 0. or self.forward_clearance < ESCAPE_CLEARANCE))
        self.stuck_seconds = self.stuck_seconds+DT if stuck else 0.
        if self.escape_remaining > 0.:
            # Stop reversing as soon as the front view opens up again.
            if not fresh or (self.escape_exit_on_clearance
                             and self.forward_clearance > ESCAPE_EXIT_CLEARANCE):
                self.escape_remaining = 0.
                self.cautious_seconds = CAUTIOUS_SECONDS
            else:
                self.escape_remaining = max(0., self.escape_remaining-DT)
        elif (stuck and self.stuck_seconds >= ESCAPE_STUCK_SECONDS and self.escape_cooldown == 0.
              and self.escapes < ESCAPE_MAX_PER_EPISODE
              and self.escape_progress >= ESCAPE_PROGRESS_REQUIRED):
            self.escape_remaining, self.escape_cooldown = ESCAPE_SECONDS, ESCAPE_COOLDOWN
            self.escape_progress, self.escapes = 0., self.escapes+1
            self.escape_exit_on_clearance = True
            # Rear overdrive and reverse propulsion must never overlap.
            self.drift_seconds = 0.
        escaping = self.escape_remaining > 0.
        if self.cautious_seconds > 0.:
            self.cautious_seconds = max(0., self.cautious_seconds-DT)
            target = min(target, CAUTIOUS_SPEED)
        if escaping:
            # Back out on straight wheels. Holding the entry steering angle
            # rotates a reversing car quickly enough to end up facing the
            # wrong way down the circuit.
            target, steer_target = -ESCAPE_SPEED, 0.
        else:
            steer_target = self.target_steer
        speed = float(np.clip(target, self.last_speed-14*DT, self.last_speed+8*DT))
        steer = float(np.clip(steer_target, self.last_steer-3*DT, self.last_steer+3*DT))
        self.last_speed, self.last_steer = speed, steer
        k = np.tan(steer)/WB
        left, right = 1-TRACK*k/2, 1+TRACK*k/2
        steering = np.clip(np.arctan2(WB*k, [left, right]), -STEER_LIMIT, STEER_LIMIT)
        wheel_speed = speed*np.array([np.hypot(left, WB*k), np.hypot(right, WB*k), left, right])
        if escaping:
            self.phase = 'reverse_escape'
            return wheel_speed/RW*SIGNS, steering
        # Optional learned, bounded rear overdrive uses measured steering and
        # scan-derived curvature only; no unmeasured body slip is inferred.
        ratio, trigger = self.p[-2:]
        self.drift_cooldown = max(0., self.drift_cooldown-DT)
        drift_safe = not self.near_compact_obstacle and float(packet['age']) <= .14
        if not drift_safe:
            self.drift_seconds = 0.
        if (drift_safe and ratio > 0 and self.drift_cooldown == 0 and abs(np.mean(steer_pos)) > trigger
                and 1.2 < measured_speed < 4. and self.target_speed > 1.):
            self.drift_seconds, self.drift_cooldown = .35, 1.0
        if self.drift_seconds > 0:
            wheel_speed[2:] *= 1+ratio
            self.drift_seconds -= DT
            self.phase = 'rear_overdrive'
        else:
            self.phase = 'braking' if target < measured_speed-.2 else 'racing'
        return wheel_speed/RW*SIGNS, steering
