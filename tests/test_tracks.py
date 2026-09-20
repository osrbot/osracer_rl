"""Geometric and provenance contracts, including exported asset interoperability."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from racing.tracks import Track, TRACK_ROOT, catalog, season_tracks

IDS = list(catalog()["tracks"])


def test_season_calendars_and_source_integrity():
    for year in [2024, 2025]:
        races = season_tracks(year)
        assert len(races) == 24
        assert [r["round"] for r in races] == list(range(1,25))
        assert len({r["circuit_id"] for r in races}) == 24
        assert all(r["season"] == year and not r["historical_layout_verified"] for r in races)
    assert season_tracks(2024)[0]["date"] == "2024-03-02"
    assert season_tracks(2025)[0]["circuit_id"] == "melbourne"
    assert season_tracks(2025)[8]["circuit_id"] == "barcelona"
    assert season_tracks(2025)[-1]["date"] == "2025-12-07"
    manifest = json.loads((TRACK_ROOT/"sources/manifest.json").read_text())
    for name, item in manifest["files"].items():
        assert hashlib.sha256((TRACK_ROOT/"sources"/name).read_bytes()).hexdigest() == item["sha256"]


@pytest.mark.parametrize("tid", IDS)
def test_geometry_and_frenet_contract(tid):
    track = Track(tid)
    assert track.width == 1.5
    assert 150 < track.length < 650
    segment_lengths = np.linalg.norm(np.roll(track.points,-1,axis=0)-track.points,axis=1)
    assert segment_lengths.min() > .05
    assert segment_lengths.max() <= .251
    assert track.length == pytest.approx(track.metadata["length"],abs=1e-6)
    assert not np.array_equal(track.points[0],track.points[-1])
    # Use segment interiors; corners and at-grade intersections are ambiguous by definition.
    s = track._s[10] + track._segment_length[10] / 2
    for offset in [-.05, 0, .05]:
        xy,yaw = track.at(s,offset)
        projected,cte,heading = track.project(xy)
        assert projected == pytest.approx(s,abs=1e-5)
        assert cte == pytest.approx(offset,abs=1e-5)
        assert heading == pytest.approx(yaw)
    assert np.allclose(track.at(-1)[0],track.at(track.length-1)[0])
    assert np.allclose(track.at(track.length+2)[0],track.at(2)[0])
    boundary = track.boundary_segments()
    assert boundary.ndim == 3 and boundary.shape[1:] == (2,2)
    assert np.isfinite(boundary).all()
    assert np.all(np.linalg.norm(boundary[:,1]-boundary[:,0],axis=1)>0)


@pytest.mark.parametrize("tid", IDS)
@pytest.mark.parametrize("variant",[".","full_scale"])
def test_collada_texture_and_mesh_contract(tid,variant):
    path = TRACK_ROOT/tid/variant
    data = json.loads((path/"track.json").read_text())
    dae = ET.parse(path/"track.dae")
    ns = {"c":"http://www.collada.org/2005/11/COLLADASchema"}
    assert dae.find(".//c:up_axis",ns).text == "Z_UP"
    assert dae.find(".//c:unit",ns).attrib["meter"] == "1"
    uri = dae.find(".//c:image/c:init_from",ns).text
    assert (path/uri).is_file()
    assert int(dae.find(".//c:triangles",ns).attrib["count"]) == data["mesh"]["triangles"]
    for filename in ["track.obj","collision.obj","track.mtl","map.png"]:
        assert (path/filename).stat().st_size > 20
    if variant == "full_scale":
        assert data["width"] == 12
        rc = Track(tid)
        # Different sampling intervals slightly change polygonal corner length.
        assert data["length"] == pytest.approx(rc.length/rc.metadata["scale"]["centerline"],rel=.003)


@pytest.mark.parametrize("tid", IDS)
def test_independent_mesh_parsers_and_area(tid):
    collada = pytest.importorskip("collada")
    trimesh = pytest.importorskip("trimesh")
    for variant in [".","full_scale"]:
        path = TRACK_ROOT/tid/variant
        data = json.loads((path/"track.json").read_text())
        dae = collada.Collada(str(path/"track.dae"))
        assert len(dae.geometries[0].primitives[0]) == data["mesh"]["triangles"]
        mesh = trimesh.load(str(path/"track.obj"),force="mesh")
        assert np.isfinite(mesh.vertices).all()
        assert np.all(mesh.face_normals[:,2] > (.99 if data.get("has_elevation") else .999))
        assert np.all(mesh.area_faces > 1e-12)
        assert mesh.area == pytest.approx(data["mesh"]["area_m2"],rel=1e-6)


def test_suzuka_rc_bridge_requires_a_3d_backend():
    data = Track("suzuka").metadata
    assert data["training_suitability"]["grade_separation_modeled"]
    assert data["training_suitability"]["requires_3d_backend"]
    assert catalog()["tracks"]["suzuka"]["requires_3d_backend"]
    full = Track(TRACK_ROOT/"suzuka/full_scale")
    assert not full.has_elevation
    assert not full.metadata["training_suitability"]["scored_racing_supported"]


def test_bridge_clearance_grade_and_localization():
    t = Track("suzuka")
    bridge = t.metadata["bridge"]
    low,yaw,_ = t.at3d(bridge["lower_s"])
    high,_,_ = t.at3d(bridge["upper_s"])
    assert np.allclose(low[:2],high[:2],atol=1e-6)
    assert high[2]-low[2]-bridge["deck_thickness"] >= .65
    assert np.max(np.abs(t._grade)) <= .0676
    assert t.elevation_at(bridge["elevated_start_s"]-1) == 0
    assert t.elevation_at(bridge["elevated_end_s"]+1) == 0
    for crossing_s in [bridge["lower_s"],bridge["upper_s"]]:
        hint = crossing_s-3
        for s in np.arange(crossing_s-3,crossing_s+3,.05):
            xyz,_,_ = t.at3d(s)
            hint,cte,_ = t.project(xyz[:2],s_hint=hint)
            assert hint == pytest.approx(s,abs=1e-5)
            assert abs(cte)<1e-6
            assert t.project(xyz)[0] == pytest.approx(s,abs=1e-5)
    triangles = t.road_triangles_3d()
    assert triangles.shape[1:] == (3,3)
    # Intersect the exact crossing's vertical line with all mesh triangles.
    a,b,c = triangles[:,0],triangles[:,1],triangles[:,2]
    v,w,q = b[:,:2]-a[:,:2],c[:,:2]-a[:,:2],low[:2]-a[:,:2]
    det = v[:,0]*w[:,1]-v[:,1]*w[:,0]
    u = (q[:,0]*w[:,1]-q[:,1]*w[:,0])/det
    z = (v[:,0]*q[:,1]-v[:,1]*q[:,0])/det
    inside = (u>=-1e-7)&(z>=-1e-7)&(u+z<=1+1e-7)
    heights = a[:,2]+u*(b[:,2]-a[:,2])+z*(c[:,2]-a[:,2])
    assert np.any(abs(heights[inside])<1e-6)
    assert np.any(abs(heights[inside]-.9)<1e-6)
    assert t.boundary_segments_3d().shape[1:] == (2,3)
    assert t.boundary_segments_3d()[:,:,2].max()==pytest.approx(.9)


def test_flat_geometry_api_remains_compatible():
    t = Track("bahrain")
    assert not t.has_elevation
    xy,yaw = t.at(30,.1)
    xyz,yaw3,grade = t.at3d(30,.1)
    assert np.allclose(xy,xyz[:2]) and xyz[2]==0 and yaw==yaw3 and grade==0
    assert np.allclose(t.boundary_segments(),t.boundary_segments_3d()[:,:,:2])


@pytest.mark.parametrize("tid,factor",[("baku",.1),("monaco",.125),("miami",.075),("montreal",.075)])
def test_adaptive_rc_revision_has_one_inner_boundary(tid,factor):
    track = Track(tid)
    assert track.width == 1.5
    assert track.metadata["scale"]["centerline"] == factor
    assert track.metadata["geometry_revision"] == "rc-r2-annular"
    assert len(track.metadata["boundary_rings"]) == 2
    assert not track.metadata["training_suitability"]["road_buffer_joins_route_arms"]
    assert catalog()["tracks"][tid]["centerline_scale"] == factor
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon
    rings = track.metadata["boundary_rings"]
    outer,inner = sorted((Polygon(r) for r in rings),key=lambda p:p.area,reverse=True)
    assert outer.is_valid and inner.is_valid
    assert outer.contains(inner)
