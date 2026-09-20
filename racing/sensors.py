"""Measured actor contract and 15 Hz single-plane laser simulation.

Specs are experiment assumptions: 270 degrees, 361 rays, 15 m maximum,
0.04 m minimum, 15 Hz; optional seeded range noise, beam dropout and delivery
latency default to zero. The laser frame pose comes from the
engine's mounted laser body (including its approximately -0.049 rad pitch).
No planar, car-centred replacement scan is synthesized.
"""
from __future__ import annotations

import numpy as np

ACTOR_KEYS = frozenset({'wheel_vel', 'steer_pos', 'lidar'})


def actor_observation(wheel_vel, steer_pos, packet):
    """Copy only measured encoders and the held laser packet across the boundary."""
    return {'wheel_vel': np.asarray(wheel_vel, float).reshape(4).copy(),
            'steer_pos': np.asarray(steer_pos, float).reshape(2).copy(),
            'lidar': {k: v.copy() if isinstance(v, np.ndarray) else v
                      for k, v in packet.items()}}


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class LidarSensor:
    def __init__(self, track, dt=1/60, scan_hz=15, ray_count=361,
                 fov_deg=270, max_range=15., wall_height=.6, min_range=.04,
                 noise_std_m=0., dropout_prob=0., latency_s=0., seed=0):
        self.track, self.dt = track, float(dt)
        self.scan_hz, self.max_range = float(scan_hz), float(max_range)
        self.wall_height, self.min_range = float(wall_height), float(min_range)
        self.noise_std_m, self.dropout_prob = float(noise_std_m), float(dropout_prob)
        self.latency_s, self.seed = float(latency_s), seed
        if (self.noise_std_m < 0 or not 0 <= self.dropout_prob <= 1 or self.latency_s < 0
                or not np.all(np.isfinite([self.noise_std_m, self.dropout_prob, self.latency_s]))):
            raise ValueError('Noise/latency must be finite and nonnegative; dropout probability must be in [0, 1]')
        self.latency_ticks = int(np.ceil(self.latency_s/self.dt-1e-12))
        period = 1 / (self.dt * self.scan_hz)
        if abs(period - round(period)) > 1e-8:
            raise ValueError('Scan period must be an integer number of control ticks')
        self.period = int(round(period))
        self.angles = np.linspace(-np.deg2rad(fov_deg)/2, np.deg2rad(fov_deg)/2, ray_count)
        self.local_directions = np.column_stack((np.cos(self.angles), np.sin(self.angles), np.zeros(ray_count)))
        seg = track.boundary_segments
        self.segments = np.asarray(seg() if callable(seg) else seg, float)
        self.segment_heights = None
        self.road_triangles = None
        if getattr(track,'has_elevation',False):
            walls=np.asarray(track.boundary_segments_3d(),float)
            self.segments=walls[:,:,:2]
            self.segment_heights=walls[:,:,2]
            self.road_triangles=np.asarray(track.road_triangles_3d(),float)
            thickness=getattr(track,'metadata',{}).get('bridge',{}).get('deck_thickness',.08)
            raised=self.road_triangles[self.road_triangles[:,:,2].max(axis=1)>1e-8]
            bottom=raised-np.array([0.,0.,thickness])
            faces=[self.road_triangles,bottom]
            for a,b in [(0,1),(1,2),(2,0)]:
                faces += [np.stack([raised[:,a],bottom[:,a],bottom[:,b]],axis=1),
                          np.stack([raised[:,a],bottom[:,b],raised[:,b]],axis=1)]
            self.road_triangles=np.concatenate(faces)
        self.reset()

    def reset(self):
        self.tick = 0
        self.packet = None
        self.last_scan_tick = None
        self.last_observe_tick = None
        self.pending = []
        self.rng = np.random.default_rng(self.seed)

    @property
    def specs(self):
        return {'scan_hz': self.scan_hz, 'control_hz': 1/self.dt,
                'rays': len(self.angles), 'fov_deg': float(np.rad2deg(np.ptp(self.angles))),
                'max_range_m': self.max_range, 'min_range_m': self.min_range,
                'wall_height_m': self.wall_height, 'frame': 'laser', 'frame_id': 'laser',
                'noise': 'independent Gaussian range noise on hits' if self.noise_std_m else 'none (assumed)',
                'noise_std_m': self.noise_std_m, 'beam_dropout_probability': self.dropout_prob,
                'requested_latency_s': self.latency_s, 'delivery_latency_s': self.latency_ticks*self.dt,
                'noise_seed': self.seed, 'mount_pose': 'actual engine laser body transform',
                'geometry': 'finite-height wall centerline segments (wall thickness omitted), ground, actual physical reflector boxes; other CAD surfaces omitted'}

    def raycast(self, lidar_pose, opponents=()):
        if isinstance(lidar_pose, dict):
            origin = np.asarray(lidar_pose['position'], float)
            rotation = np.asarray(lidar_pose['rotation'], float).reshape(3, 3)
        else:
            origin, rotation = lidar_pose
            origin, rotation = np.asarray(origin, float), np.asarray(rotation, float).reshape(3, 3)
        rays = self.local_directions @ rotation.T
        nearby = np.all((self.segments.max(axis=1) >= origin[:2]-self.max_range) &
                        (self.segments.min(axis=1) <= origin[:2]+self.max_range), axis=1)
        segments = self.segments[nearby]
        starts = segments[:, 0]
        edges = segments[:, 1] - starts
        delta = starts - origin[:2]
        den = _cross(rays[:, None, :2], edges[None, :, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            distance = _cross(delta, edges)[None, :] / den
            along = _cross(delta[None, :, :], rays[:, None, :2]) / den
        with np.errstate(invalid='ignore'):
            z = origin[2] + distance * rays[:, None, 2]
        base=0.
        if self.segment_heights is not None:
            heights=self.segment_heights[nearby]
            base=heights[None,:,0]+along*(heights[None,:,1]-heights[None,:,0])
        hit = ((abs(den) > 1e-10) & (distance >= self.min_range) &
               (along >= 0) & (along <= 1) & (z >= base) & (z <= base+self.wall_height))
        ranges = np.min(np.where(hit, distance, np.inf), axis=1) if len(segments) else np.full(len(rays), np.inf)
        # Ground is a physical surface too: pitched rear beams may intersect it.
        with np.errstate(divide='ignore', invalid='ignore'):
            ground = -origin[2] / rays[:, 2]
        ranges = np.minimum(ranges, np.where((rays[:, 2] < -1e-9) & (ground >= self.min_range), ground, np.inf))
        if self.road_triangles is not None:
            # Moller-Trumbore ray intersection, both triangle faces. This sees
            # the same physical ramps/deck as the engines, including underpass.
            triangles=self.road_triangles
            mask=np.all((triangles.max(axis=1)>=origin-self.max_range)&(triangles.min(axis=1)<=origin+self.max_range),axis=1)
            for tri in np.array_split(triangles[mask],max(1,int(mask.sum()/128)+1)):
                if not len(tri):continue
                e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
                h=np.cross(rays[:,None,:],e2[None,:,:]);det=np.sum(e1[None,:,:]*h,axis=2)
                delta=origin-tri[:,0];q=np.cross(delta,e1)
                with np.errstate(divide='ignore',invalid='ignore'):
                    u=np.sum(delta[None,:,:]*h,axis=2)/det
                    v=np.sum(rays[:,None,:]*q[None,:,:],axis=2)/det
                    t=np.sum(e2*q,axis=1)[None,:]/det
                with np.errstate(invalid='ignore'):
                    valid=(abs(det)>1e-10)&(u>=0)&(v>=0)&(u+v<=1)&(t>=self.min_range)
                ranges=np.minimum(ranges,np.min(np.where(valid,t,np.inf),axis=1))
        for car in opponents:
            # Only the explicit physical reflector is modeled, not an invented
            # whole-car volume. Other CAD surfaces are omitted by this proxy.
            target = car['lidar_target']
            center = np.asarray(target.get('position', target.get('center')), float)
            rot = np.asarray(target['rotation'], float).reshape(3, 3)
            half = np.asarray(target['half_size'], float)
            local_o, local_d = (origin-center) @ rot, rays @ rot
            parallel = abs(local_d) < 1e-10
            with np.errstate(divide='ignore', invalid='ignore'):
                t1, t2 = (-half-local_o)/local_d, (half-local_o)/local_d
            near = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(axis=1)
            far = np.where(parallel, np.inf, np.maximum(t1, t2)).min(axis=1)
            outside = np.any(parallel & (abs(local_o) > half), axis=1)
            t = np.where(near >= self.min_range, near, far)
            good = (~outside) & (far >= near) & (t >= self.min_range)
            ranges = np.minimum(ranges, np.where(good, t, np.inf))
        valid = np.isfinite(ranges) & (ranges <= self.max_range)
        return np.where(valid, ranges, self.max_range), valid

    def observe(self, state, lidar_pose, opponents=(), tick=None):
        tick = self.tick if tick is None else int(tick)
        if self.last_observe_tick is not None and tick < self.last_observe_tick:
            raise ValueError('Sensor time moved backwards; call reset for a new episode')
        if self.last_scan_tick is None or tick-self.last_scan_tick >= self.period:
            ranges, valid = self.raycast(lidar_pose, opponents)
            if self.noise_std_m:
                ranges[valid] = np.clip(ranges[valid] + self.rng.normal(0., self.noise_std_m, valid.sum()),
                                       self.min_range, self.max_range)
            if self.dropout_prob:
                valid &= self.rng.random(len(valid)) >= self.dropout_prob
                ranges[~valid] = self.max_range
            packet = {'ranges': ranges, 'valid': valid, 'validmask': valid.copy(),
                      'angles': self.angles.copy(), 'timestamp': tick*self.dt,
                      'age': 0., 'frame': 'laser', 'frame_id': 'laser'}
            self.pending.append((tick+self.latency_ticks, packet))
            self.last_scan_tick = tick
        while self.pending and self.pending[0][0] <= tick:
            _, self.packet = self.pending.pop(0)
        if self.packet is None:
            # No fictitious fresh scan during delivery startup. The negative
            # timestamp distinguishes this sentinel from the first real scan.
            self.packet = {'ranges': np.full(len(self.angles), self.max_range),
                           'valid': np.zeros(len(self.angles), bool),
                           'validmask': np.zeros(len(self.angles), bool),
                           'angles': self.angles.copy(), 'timestamp': -1.,
                           'age': float('inf'), 'frame': 'laser', 'frame_id': 'laser'}
        self.packet['age'] = tick*self.dt-self.packet['timestamp'] if self.packet['timestamp'] >= 0 else float('inf')
        self.last_observe_tick = tick
        self.tick = tick+1
        return actor_observation(state['wheel_vel'], state['steer_pos'], self.packet)
