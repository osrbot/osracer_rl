"""Closed, metric track geometry shared by the racing backends.

Positive offsets and cross-track error are to the left of travel. Source tracing
order defines travel; it is not a certified official race direction/start line.
"""
from pathlib import Path
import json
import numpy as np

TRACK_ROOT = Path(__file__).resolve().parents[1] / "tracks"


class Track:
    def __init__(self, path="bahrain"):
        path = Path(path)
        if not path.exists():
            path = TRACK_ROOT / path
        if path.is_dir():
            path /= "track.json"
        self.path = path.resolve()
        self.metadata = json.loads(path.read_text())
        self.id = self.metadata["id"]
        self.name = self.metadata.get("name", self.id)
        self.points = np.asarray(self.metadata["centerline"], dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 3:
            raise ValueError("centerline must contain at least three 2D points")
        if np.linalg.norm(self.points[0] - self.points[-1]) < 1e-9:
            self.points = self.points[:-1]
        if not np.isfinite(self.points).all():
            raise ValueError("non-finite centerline")
        self.width = float(self.metadata["width"])
        if not np.isfinite(self.width) or self.width <= 0:
            raise ValueError("width must be finite and positive")
        self._delta = np.roll(self.points, -1, axis=0) - self.points
        self._segment_length = np.linalg.norm(self._delta, axis=1)
        if np.any(self._segment_length < 1e-9):
            raise ValueError("duplicate consecutive centerline points")
        self._tangent = self._delta / self._segment_length[:, None]
        self._s = np.r_[0.0, np.cumsum(self._segment_length)]
        self.length = float(self._s[-1])
        self._yaw = np.arctan2(self._delta[:, 1], self._delta[:, 0])
        elevation = np.asarray(self.metadata.get("elevation", np.zeros(len(self.points))), dtype=float)
        if len(elevation) == len(self.points)+1 and abs(elevation[0]-elevation[-1])<1e-9:
            elevation = elevation[:-1]
        if elevation.shape != (len(self.points),) or not np.isfinite(elevation).all():
            raise ValueError("elevation must be a finite value per centerline point")
        self.elevations = elevation
        self.has_elevation = bool(np.any(np.abs(elevation)>1e-9))
        self._grade = (np.roll(elevation,-1)-elevation)/self._segment_length
        self._boundaries = None

    def at(self, s, offset=0.0):
        """Return (xy, yaw radians) at wrapped arc distance, offset left in metres."""
        s = float(s) % self.length
        i = min(int(np.searchsorted(self._s, s, side="right") - 1), len(self.points)-1)
        tangent = self._tangent[i]
        xy = self.points[i] + (s - self._s[i]) * tangent
        xy = xy + float(offset) * np.array([-tangent[1], tangent[0]])
        return xy, float(self._yaw[i])

    def elevation_at(self, s):
        """Road surface height in metres; s remains horizontal route arc length."""
        s = float(s) % self.length
        return float(np.interp(s,self._s,np.r_[self.elevations,self.elevations[0]]))

    def grade_at(self, s):
        """dz/ds (dimensionless rise/run, not pitch angle)."""
        i = min(int(np.searchsorted(self._s,float(s)%self.length,side="right")-1),len(self.points)-1)
        return float(self._grade[i])

    def at3d(self, s, offset=0.0):
        """Return (xyz,yaw,grade) on the road; atan(grade) is uphill pitch."""
        xy,yaw = self.at(s,offset)
        return np.r_[xy,self.elevation_at(s)],yaw,self.grade_at(s)

    def project(self, xy, s_hint=None, z=None, search_window=10.0):
        """Nearest segment projection: (wrapped arc distance, signed CTE, yaw).

        Supply previous s via s_hint to restrict the search to nearby route
        progress, or road-relative z to distinguish bridge levels. A 3-vector
        also supplies z. Without either, a 2D crossing remains ambiguous.
        """
        xy = np.asarray(xy, dtype=float)
        if xy.shape == (3,):
            z = float(xy[2]) if z is None else z
            xy = xy[:2]
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("xy must be a finite 2-vector")
        rel = xy - self.points
        along = np.clip(np.einsum("ij,ij->i", rel, self._tangent), 0, self._segment_length)
        residual = rel - along[:, None] * self._tangent
        candidate_s = self._s[:-1]+along
        distance2 = np.einsum("ij,ij->i", residual, residual)
        if z is not None:
            if not np.isfinite(z):
                raise ValueError("z must be finite")
            height = self.elevations+along*self._grade
            distance2 += (float(z)-height)**2
        if s_hint is not None:
            if not np.isfinite(s_hint) or not np.isfinite(search_window) or search_window <= 0:
                raise ValueError("s_hint and positive search_window must be finite")
            delta = (candidate_s-float(s_hint)+self.length/2)%self.length-self.length/2
            distance2[np.abs(delta)>float(search_window)] = np.inf
            if not np.isfinite(distance2).any():
                raise ValueError("No route segment within projection search window")
        i = int(np.argmin(distance2))
        t = self._tangent[i]
        cte = t[0] * residual[i, 1] - t[1] * residual[i, 0]
        return float((self._s[i] + along[i]) % self.length), float(cte), float(self._yaw[i])

    def boundary_segments(self):
        """Return (N,2,2) road-edge segments in metres, suitable for ray casting."""
        if self._boundaries is None:
            rings = self.metadata.get("boundary_rings")
            if rings is None:
                tangent = self._tangent + np.roll(self._tangent, 1, axis=0)
                norms = np.linalg.norm(tangent, axis=1)
                tangent = tangent / np.maximum(norms[:, None], 1e-9)
                normal = np.c_[-tangent[:, 1], tangent[:, 0]]
                rings = [self.points + normal * self.width / 2, self.points - normal * self.width / 2]
            out = []
            for ring in rings:
                ring = np.asarray(ring, dtype=float)
                if np.linalg.norm(ring[0]-ring[-1]) < 1e-9:
                    ring = ring[:-1]
                out.append(np.stack((ring, np.roll(ring, -1, axis=0)), axis=1))
            self._boundaries = np.concatenate(out)
        return self._boundaries.copy()

    def boundary_segments_3d(self):
        """Road-edge segment bottom endpoints, preserving separate bridge levels."""
        if "boundary_segments_3d" in self.metadata:
            return np.asarray(self.metadata["boundary_segments_3d"],dtype=float).copy()
        xy = self.boundary_segments()
        return np.concatenate((xy,np.zeros((*xy.shape[:-1],1))),axis=-1)

    boundary_segments3d = boundary_segments_3d

    def road_mesh(self):
        """Shared surface mesh (vertices Nx3, triangles Mx3), when supplied."""
        if "road_vertices" not in self.metadata or "road_faces" not in self.metadata:
            raise ValueError("This track does not carry an inline 3D road mesh")
        return np.asarray(self.metadata["road_vertices"],float),np.asarray(self.metadata["road_faces"],int)

    def road_triangles_3d(self):
        vertices,faces = self.road_mesh()
        return vertices[faces]


def catalog():
    return json.loads((TRACK_ROOT / "catalog.json").read_text())


def season_tracks(year):
    """Return the ordered round records for a supported calendar year."""
    return catalog()["seasons"][str(year)]
