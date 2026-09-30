"""Privileged race evaluator; never imported or consumed by the racing actor."""
from __future__ import annotations
import numpy as np


def drift_statistics(speeds, slip_angles, dt=1/60, min_duration=.1):
    """Truth-only validation of contiguous moving lateral slip episodes.

    Separate short excursions must never add up to a qualifying drift. Samples
    with nonfinite values or at/below either strict threshold break a segment.
    """
    speeds, slip_angles = np.asarray(speeds, float), np.asarray(slip_angles, float)
    if speeds.shape != slip_angles.shape or speeds.ndim != 1 or dt <= 0:
        raise ValueError('Expected matching one-dimensional arrays and positive dt')
    moving_slip = (np.isfinite(speeds) & np.isfinite(slip_angles) &
                   (speeds > 1.2) & (abs(slip_angles) > np.deg2rad(20)))
    changes = np.diff(np.r_[False, moving_slip, False].astype(int))
    durations = (np.flatnonzero(changes == -1)-np.flatnonzero(changes == 1))*dt
    longest = float(np.max(durations, initial=0.))
    return {'drift_duration_s': float(np.count_nonzero(moving_slip)*dt),
            'max_continuous_drift_duration_s': longest,
            'drift_event_count': int(np.count_nonzero(durations+1e-10 >= min_duration)),
            'drift_min_duration_s': float(min_duration),
            'continuous_drift_qualified': bool(longest+1e-10 >= min_duration)}


class RaceMetrics:
    def __init__(self, track, dt=1/60):
        self.track, self.dt = track, float(dt)
        self.reset()

    def reset(self):
        self.time = 0.
        self.progress = 0.
        self.last_s = self.last_xy = None
        self.laps = 0
        self.valid_lap_times = []
        self.lap_records = []
        self.lap_start = 0.
        self.lap_invalid = False
        self.collisions = self.offroad_events = self.teleports = 0
        self.was_collision = self.was_offroad = False
        self.effective_overtakes = 0
        self.overtake_records = []
        self.opponents = {}
        self.max_speed = 0.
        self.distance = 0.
        self.offroad_seconds = 0.

    def _project(self, state):
        xy = np.array([state['x'], state['y']], float)
        # Height belongs exclusively to evaluation. It disambiguates a bridge
        # from the road below without revealing course coordinates to actors.
        position = np.r_[xy, float(state['z'])] if 'z' in state and getattr(self.track, 'has_elevation', False) else xy
        s, cte, _ = self.track.project(position)
        return xy, float(s), float(cte)

    def _delta(self, current, previous):
        length = float(self.track.length)
        return (current-previous+length/2) % length-length/2

    def update(self, state, opponents=(), collision=False, offroad=False, dt=None):
        dt = self.dt if dt is None else float(dt)
        if dt <= 0:
            raise ValueError('Evaluation dt must be positive')
        xy, s, cte = self._project(state)
        collision = bool(collision or state.get('collision', False))
        offroad = bool(offroad or state.get('offroad', False) or abs(cte) > self.track.width/2)
        self.time += dt
        self.collisions += int(collision and not self.was_collision)
        self.offroad_events += int(offroad and not self.was_offroad)
        self.was_collision, self.was_offroad = collision, offroad
        self.offroad_seconds += dt*offroad
        self.lap_invalid |= collision or offroad
        teleport = False
        delta = 0.
        if self.last_s is not None:
            delta = self._delta(s, self.last_s)
            displacement = float(np.linalg.norm(xy-self.last_xy))
            # Generous physical envelope still rejects resets/teleport jumps.
            teleport = displacement > 30*dt+.15 or abs(delta) > 30*dt+.15
            if teleport:
                self.teleports += 1
                self.lap_invalid = True
                self.opponents.clear()
                delta = 0.
            else:
                self.progress += delta
                self.distance += displacement
                self.max_speed = max(self.max_speed, displacement/dt)
        self.last_s, self.last_xy = s, xy
        # Full circuits are counted from the spawn location. Reverse/repeated
        # startline crossings cannot create a lap because progress is signed.
        threshold = (self.laps+1)*self.track.length
        if self.progress >= threshold:
            overshoot = (self.progress-threshold)/max(delta, 1e-12)
            crossing_time = self.time - np.clip(overshoot, 0, 1)*dt
            lap_time = float(crossing_time-self.lap_start)
            valid = not self.lap_invalid
            self.laps += 1
            self.lap_records.append({'lap': self.laps, 'time_s': lap_time, 'valid': valid})
            if valid:
                self.valid_lap_times.append(lap_time)
            self.lap_start = crossing_time
            self.lap_invalid = collision or offroad or teleport
        unsafe = collision or offroad or teleport
        present = set()
        for index, other in enumerate(opponents):
            key = str(other.get('id', index))
            present.add(key)
            oxy, os, octe = self._project(other)
            other_unsafe = bool(other.get('collision', False) or other.get('offroad', False)
                                or abs(octe) > self.track.width/2)
            rec = self.opponents.get(key)
            if rec is None:
                relative = self._delta(s, os)
                rec = {'s': os, 'xy': oxy, 'relative': relative,
                       'armed': relative < -.5 and not unsafe and not other_unsafe, 'ahead_time': 0.}
                self.opponents[key] = rec
                continue
            odelta = self._delta(os, rec['s'])
            other_jump = np.linalg.norm(oxy-rec['xy']) > 30*dt+.15 or abs(odelta) > 30*dt+.15
            if unsafe or other_unsafe or other_jump:
                rec.update(s=os, xy=oxy, relative=self._delta(s, os), armed=False, ahead_time=0.)
                continue
            rec['relative'] += delta-odelta
            rec['s'], rec['xy'] = os, oxy
            if rec['relative'] < -.5:
                rec['armed'] = True
                rec['ahead_time'] = 0.
            elif rec['armed'] and rec['relative'] > .5:
                rec['ahead_time'] += dt
                if rec['ahead_time']+1e-10 >= .3:
                    self.effective_overtakes += 1
                    self.overtake_records.append({'opponent': key, 'time_s': self.time})
                    rec['armed'] = False
                    rec['ahead_time'] = 0.
            else:
                rec['ahead_time'] = 0.
        for key in set(self.opponents)-present:
            del self.opponents[key]
        return self.summary()

    def summary(self):
        return {'elapsed_s': self.time, 'progress_m': self.progress,
                'laps': self.laps, 'valid_laps': len(self.valid_lap_times),
                'valid_lap_times_s': list(self.valid_lap_times),
                'lap_records': list(self.lap_records), 'current_lap_valid': not self.lap_invalid,
                'collision_events': self.collisions, 'offroad_events': self.offroad_events,
                'offroad_seconds': self.offroad_seconds, 'teleport_events': self.teleports,
                'effective_overtakes': self.effective_overtakes,
                'overtake_records': list(self.overtake_records),
                'max_speed_m_s': self.max_speed, 'distance_m': self.distance,
                'completed': bool(self.valid_lap_times)}
