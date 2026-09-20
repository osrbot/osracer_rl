"""Experimental local-LiDAR-feedback drift actor; not the frozen v5 runtime.

No global pose, course state, or opponent truth enters this controller. A zero
rear-overdrive parameter takes the exact baseline action path. Positive values
enable a bounded experiment that still requires native closed-loop qualification.
"""
from __future__ import annotations
import numpy as np
from .odometry import LidarOdometry
from .policy import RacingPolicy, DT, RW, WB, TRACK, STEER_LIMIT, CENTER_LIMIT, SIGNS


class ClosedLoopRacingPolicy(RacingPolicy):
    version = 'experimental-gap-center-v7a-verified-escape'
    upper = RacingPolicy.upper.copy()
    upper[-2] = 3.5
    control_bounds = {'boost_countersteer_gain': (.10, .25),
                      'max_boost_seconds': (.4, 1.0),
                      'yaw_rate_guard': (3.8, 5.5)}

    def __init__(self, parameters=None, odometry=None, target_slip_deg=25.,
                 confidence_threshold=.6, max_boost_seconds=.6, steering_bias=.1,
                 prediction_horizon=.09, boost_countersteer_gain=.2,
                 yaw_rate_guard=3.8):
        self.odometry = LidarOdometry() if odometry is None else odometry
        self.target_slip = np.deg2rad(float(np.clip(target_slip_deg, 20., 30.)))
        self.confidence_threshold = max(.6, float(confidence_threshold))
        for name, value in [('max_boost_seconds', max_boost_seconds),
                            ('boost_countersteer_gain', boost_countersteer_gain),
                            ('yaw_rate_guard', yaw_rate_guard)]:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
            setattr(self, name, float(np.clip(value, *self.control_bounds[name])))
        self.steering_bias = float(np.clip(steering_bias, 0., .14))
        self.prediction_horizon = float(np.clip(prediction_horizon, .07, .10))
        super().__init__(parameters)

    def reset(self):
        super().reset()
        if hasattr(self, 'odometry'):
            self.odometry.reset()
        self.closed_state = 'idle'
        self.boost_elapsed = self.recovery_elapsed = self.closed_cooldown = 0.
        self.turn_sign = 0.
        self.last_beta_timestamp = None
        self.last_beta = self.beta_rate = 0.
        self.closed_last_steer = 0.
        self.feedback = {'state': 'idle', 'rear_ratio': 0., 'valid': False,
                         'beta_estimate': 0., 'forward_source': 'unavailable'}

    def abort_drift(self, reason='external_safety'):
        """Cancel a pulse without creating a drift recovery from ordinary driving.

        A wall supervisor also intervenes when no drift has been requested.
        Entering recovery in that case would inject lateral countersteering on
        the following tick and can undo the supervisor's avoidance manoeuvre.
        Repeated interventions during recovery must not reset its timer.
        """
        if self.closed_state == 'boost':
            self.closed_state = 'recover'
            self.recovery_elapsed = 0.
        self.closed_cooldown = 1.2
        if self.closed_state == 'recover':
            self.phase = 'drift_recover'
        self.feedback.update(state=self.closed_state, rear_ratio=0., abort_reason=reason)

    def action(self, observation):
        max_ratio = float(self.p[-2])
        if max_ratio <= 0:
            # This branch deliberately does not run odometry or alter history.
            return super().action(observation)
        estimate = self.odometry.update(observation)
        # Keep baseline gap planning, obstacle envelopes, acceleration ramps and
        # steering limits. Suppress its open-loop pulse while computing actions.
        self.p[-2] = 0.
        try:
            baseline = super().action(observation)
        finally:
            self.p[-2] = max_ratio
        if self.escape_remaining > 0.:
            # A bounded reverse escape outranks drift feedback for its duration:
            # countersteering or rear overdrive while backing out would fight
            # the recovery manoeuvre the supervisor just verified.
            self.closed_state = 'idle'
            return baseline
        forward_source = 'lidar' if estimate['observable_axes']['longitudinal'] else 'front_encoders'
        forward = float(estimate['lidar_forward_speed'] if forward_source == 'lidar' else estimate['forward_speed'])
        body_speed = float(np.hypot(forward, estimate['lateral_speed']))
        # Keep the full quadrant when forward velocity reverses during a spin.
        # A negative longitudinal component is not evidence of zero sideslip.
        beta = float(np.arctan2(estimate['lateral_speed'], forward)) if body_speed > .3 else 0.
        confidence = float(estimate['confidence'])
        valid = bool(estimate['valid'] and confidence >= self.confidence_threshold
                     and float(estimate['age']) <= .14 and np.isfinite([forward, beta]).all())
        packet_time = float(observation['lidar']['timestamp'])
        if valid and packet_time != self.last_beta_timestamp:
            if self.last_beta_timestamp is not None and 0 < packet_time-self.last_beta_timestamp <= .15:
                delta = np.arctan2(np.sin(beta-self.last_beta), np.cos(beta-self.last_beta))
                self.beta_rate = float(np.clip(delta/(packet_time-self.last_beta_timestamp), -8., 8.))
            else:
                self.beta_rate = 0.
            self.last_beta, self.last_beta_timestamp = beta, packet_time
        if not valid:
            self.beta_rate = 0.
        # The scan fit describes a past interval. Predict through that latency
        # before allocating more rear drive, countersteering, or enforcing the
        # unchanged 35-degree abort threshold.
        prediction_age = float(estimate['age']) if valid else 0.
        beta_predicted = (beta+self.beta_rate*(self.prediction_horizon+prediction_age)
                          if valid else beta if np.isfinite(beta) else 0.)
        yaw_rate = float(estimate.get('yaw_rate', 0.))
        steering_measured = float(np.mean(observation['steer_pos']))
        self.closed_cooldown = max(0., self.closed_cooldown-DT)
        clear = valid and not self.near_compact_obstacle
        if (self.closed_state == 'idle' and self.closed_cooldown == 0. and clear
                and 1.2 < body_speed <= 2.55 and forward > 1. and self.target_speed > 1.2
                and abs(steering_measured) > self.p[-1]):
            self.closed_state, self.boost_elapsed = 'boost', 0.
            self.turn_sign = float(np.sign(steering_measured))
        if self.closed_state == 'boost' and (not clear or forward <= 1.2
                or self.target_speed <= 1. or max(abs(beta), abs(beta_predicted)) >= np.deg2rad(35)
                or (abs(yaw_rate) >= self.yaw_rate_guard and -self.turn_sign*beta > np.deg2rad(8)
                    and -self.turn_sign*self.beta_rate > .3)
                or -self.turn_sign*beta_predicted >= self.target_slip
                or self.boost_elapsed+1e-10 >= self.max_boost_seconds):
            self.abort_drift('slip_limit_or_unavailable_clearance')
        ratio, correction = 0., 0.
        if self.closed_state == 'boost':
            signed_slip = -self.turn_sign*beta_predicted
            # Increase rear slip only while measured lateral slip is below its
            # target; reduce the request as the 20–30 degree target is reached.
            ratio = max_ratio*float(np.clip(.25+1.25*(self.target_slip-signed_slip)/self.target_slip, 0., 1.))
            # Brief induction bias is withdrawn as lateral slip develops;
            # countersteer grows continuously with the estimated sideslip.
            bias = self.steering_bias*self.turn_sign*np.clip(1-signed_slip/np.deg2rad(15), 0., 1.)
            correction = bias + self.boost_countersteer_gain*beta_predicted
            self.boost_elapsed += DT
            self.phase = 'drift_boost'
        elif self.closed_state == 'recover':
            # No rear boost during recovery. With usable lateral feedback,
            # countersteer toward zero slip; otherwise fall back to gap steering.
            correction = .45*beta if valid else 0.
            self.recovery_elapsed += DT
            self.phase = 'drift_recover'
            if valid and forward > .5 and abs(beta) < np.deg2rad(8) and self.recovery_elapsed >= .1:
                self.closed_state = 'idle'
        self.feedback = {'state': self.closed_state, 'rear_ratio': ratio,
                         'valid': valid, 'confidence': confidence, 'beta_estimate': beta,
                         'beta_predicted': beta_predicted, 'beta_rate': self.beta_rate, 'yaw_rate': yaw_rate,
                         'forward_speed': forward, 'forward_source': forward_source,
                         'body_speed': body_speed,
                         'boost_elapsed': self.boost_elapsed}
        if correction == 0. and ratio == 0.:
            self.closed_last_steer = self.last_steer
            return baseline
        steer = float(np.clip(self.last_steer+correction, -CENTER_LIMIT, CENTER_LIMIT))
        steer = float(np.clip(steer, self.closed_last_steer-3*DT, self.closed_last_steer+3*DT))
        self.closed_last_steer = self.last_steer = steer
        k = np.tan(steer)/WB
        left, right = 1-TRACK*k/2, 1+TRACK*k/2
        angles = np.clip(np.arctan2(WB*k, [left, right]), -STEER_LIMIT, STEER_LIMIT)
        speeds = self.last_speed*np.array([np.hypot(left, WB*k), np.hypot(right, WB*k),
                                           left*(1+ratio), right*(1+ratio)])
        if self.closed_state == 'boost' and valid:
            # Roll the front wheels at their estimated actual contact speed.
            # Forcing a much lower planned speed while powering the rear would
            # spend front tire friction on braking and induce understeer.
            yaw_rate = float(estimate.get('yaw_rate', 0.))
            front_x = forward-yaw_rate*np.array([TRACK/2, -TRACK/2])
            front_y = float(estimate['lateral_speed'])+yaw_rate*WB
            speeds[:2] = np.maximum(0., front_x*np.cos(angles)+front_y*np.sin(angles))
            rear_rolling = forward-yaw_rate*np.array([TRACK/2, -TRACK/2])
            speeds[2:] = np.maximum(0., rear_rolling)*(1+ratio)
        return speeds/RW*SIGNS, angles
