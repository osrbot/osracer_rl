"""Rebuild vendored F1-derived road assets, offline and deterministically.

Run: python -m racing.build_tracks. Build dependencies: numpy, Pillow, shapely>=2.1.
The GeoJSON is an approximate contemporary snapshot, not a year-specific survey.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from shapely import constrained_delaunay_triangles
from shapely.geometry import LineString
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1] / "tracks"
IDS = {
    "bahrain":"bh-2002", "jeddah":"sa-2021", "melbourne":"au-1953",
    "suzuka":"jp-1962", "shanghai":"cn-2004", "miami":"us-2022",
    "imola":"it-1953", "monaco":"mc-1929", "montreal":"ca-1978",
    "barcelona":"es-1991", "spielberg":"at-1969", "silverstone":"gb-1948",
    "hungaroring":"hu-1986", "spa":"be-1925", "zandvoort":"nl-1948",
    "monza":"it-1922", "baku":"az-2016", "singapore":"sg-2008",
    "austin":"us-2012", "mexico_city":"mx-1962", "interlagos":"br-1940",
    "las_vegas":"us-2023", "lusail":"qa-2004", "yas_marina":"ae-2009",
}
# Width is fixed at 1.5 m; these four paths need extra space between nearby
# arms. Explicit, reproducible training geometry revisions, not historical edits.
RC_CENTERLINE_SCALES = {"baku":0.100, "monaco":0.125, "miami":0.075, "montreal":0.075}
CALENDARS = {
    2024: list(zip(IDS, ["03-02","03-09","03-24","04-07","04-21","05-05",
        "05-19","05-26","06-09","06-23","06-30","07-07","07-21","07-28",
        "08-25","09-01","09-15","09-22","10-20","10-27","11-03","11-23","12-01","12-08"])),
    2025: list(zip(["melbourne","shanghai","suzuka","bahrain","jeddah","miami",
        "imola","monaco","barcelona","montreal","spielberg","silverstone","spa",
        "hungaroring","zandvoort","monza","baku","singapore","austin","mexico_city",
        "interlagos","las_vegas","lusail","yas_marina"],
        ["03-16","03-23","04-06","04-13","04-20","05-04","05-18","05-25","06-01",
         "06-15","06-29","07-06","07-27","08-03","08-31","09-07","09-21","10-05",
         "10-19","10-26","11-09","11-22","11-30","12-07"])),
}


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def metric_coordinates(coordinates):
    """Local tangent equirectangular approximation, east/north metres, WGS84 lon/lat."""
    ll = np.asarray(coordinates, dtype=float)[:, :2]
    origin = ll[0].copy()
    xy = np.deg2rad(ll - origin) * 6371008.8
    xy[:, 0] *= math.cos(math.radians(origin[1]))
    if np.linalg.norm(xy[-1] - xy[0]) < 0.01:
        xy = xy[:-1]
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
    return xy[keep], origin


def resample(points, spacing):
    points = np.vstack([points, points[0]])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    s = np.linspace(0, distance[-1], max(8, math.ceil(distance[-1]/spacing)), endpoint=False)
    return np.c_[np.interp(s, distance, points[:, 0]), np.interp(s, distance, points[:, 1])]


def road_mesh(points, width, closed=True):
    """Buffer the closed path then triangulate its union, retaining holes.

    Unlike a naive offset strip this never creates inverted faces at hairpins.
    Widening can connect nearby arms; those connections are recorded in metadata.
    """
    line = LineString(np.vstack([points, points[0]]) if closed else points)
    road = line.buffer(width/2, quad_segs=4, join_style="round",cap_style="round" if closed else "flat")
    polys = list(road.geoms) if road.geom_type == "MultiPolygon" else [road]
    rings, vertices, faces, lookup = [], [], [], {}
    for poly in polys:
        rings.extend([np.asarray(poly.exterior.coords)[:-1].tolist()])
        rings.extend([np.asarray(h.coords)[:-1].tolist() for h in poly.interiors])
        triangles = constrained_delaunay_triangles(poly)
        for tri in triangles.geoms:
            xy = np.asarray(tri.exterior.coords)[:3, :2]
            a,b = xy[1]-xy[0],xy[2]-xy[0]
            if a[0]*b[1]-a[1]*b[0] < 0:
                xy = xy[[0,2,1]]
            indices = []
            for x,y in xy:
                key = (float(x),float(y))
                if key not in lookup:
                    lookup[key] = len(vertices)
                    vertices.append([x,y,0.0])
                indices.append(lookup[key])
            faces.append(indices)
    return np.asarray(vertices), np.asarray(faces), rings, road


def suzuka_bridge(points, width):
    """Grade-separated RC road, with an assumed smooth, driveable bridge profile.

    The later traversal is the back straight over the earlier Degner traversal.
    This relation is documented by Honda; heights/ramps are training assumptions.
    """
    delta = np.roll(points,-1,axis=0)-points
    lengths = np.linalg.norm(delta,axis=1)
    s = np.r_[0,np.cumsum(lengths)]
    segments = [LineString([a,a+d]) for a,d in zip(points,delta)]
    tree = STRtree(segments)
    crossings = []
    for i,line in enumerate(segments):
        for j in tree.query(line,predicate="crosses"):
            if i < j:
                crossing = line.intersection(segments[j])
                crossings.append((s[i]+line.project(crossing),s[j]+segments[j].project(crossing),list(crossing.coords)[0]))
    if len(crossings) != 1:
        raise ValueError(f"Expected one Suzuka route crossing, got {len(crossings)}")
    lower_s,upper_s,crossing_xy = crossings[0]
    height,thickness,half_flat,ramp = .9,.08,4.,20.
    # Move the plateau 1.5 m toward the back straight so the descent ends
    # before a tight corner; the crossing retains 2.5 m of flat deck ahead.
    profile_center = upper_s-1.5
    distance = np.abs((s[:-1]-profile_center+s[-1]/2)%s[-1]-s[-1]/2)
    u = np.clip((distance-half_flat)/ramp,0,1)
    elevation = height*(1-(3*u*u-2*u*u*u))
    elevated_ids = np.flatnonzero(elevation>1e-12)
    first,last = int(elevated_ids[0]-1),int(elevated_ids[-1]+1)
    bridge_ids = np.arange(first,last+1)
    flat_points = np.vstack([points[last:],points[:first+1]])
    flat_vertices,flat_faces,flat_rings,_ = road_mesh(flat_points,width,closed=False)
    tangent = delta/lengths[:,None]
    normal = np.c_[-tangent[:,1],tangent[:,0]]
    bisector = normal+np.roll(normal,1,axis=0)
    bisector /= np.linalg.norm(bisector,axis=1)[:,None]
    miter = width/2/np.einsum("ij,ij->i",bisector,normal)
    offsets = bisector*miter[:,None]
    # Match the two flat-buffer butt caps exactly at the zero-height seams.
    offsets[first] = normal[first-1]*width/2
    offsets[last] = normal[last]*width/2
    left = np.c_[points[bridge_ids]+offsets[bridge_ids],elevation[bridge_ids]]
    right = np.c_[points[bridge_ids]-offsets[bridge_ids],elevation[bridge_ids]]
    vertices = np.stack((left,right),axis=1).reshape(-1,3)
    faces = []
    for i in range(len(bridge_ids)-1):
        j = i+1
        faces.extend([[2*i,2*i+1,2*j+1],[2*i,2*j+1,2*j]])
    faces = np.asarray(faces)
    triangles = vertices[faces]
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    if np.any(normals[:,2]<=0):
        raise ValueError("Bridge ribbon has inverted faces; revise offset geometry")
    boundaries = [np.stack((edge[:-1],edge[1:]),axis=1) for edge in (left,right)]
    for ring in flat_rings:
        ring = np.asarray(ring)
        pairs = np.stack((ring,np.roll(ring,-1,axis=0)),axis=1)
        mids = pairs.mean(axis=1)
        seam = np.zeros(len(pairs),dtype=bool)
        for endpoint,direction in [(points[first],tangent[first-1]),(points[last],tangent[last])]:
            seam |= (np.abs((mids-endpoint)@direction)<1e-7)&(np.linalg.norm(mids-endpoint,axis=1)<=width)
        pairs = pairs[~seam]
        boundaries.append(np.concatenate((pairs,np.zeros((*pairs.shape[:-1],1))),axis=-1))
    boundaries = np.concatenate(boundaries)
    faces = np.vstack([faces,flat_faces+len(vertices)])
    vertices = np.vstack([vertices,flat_vertices])
    bridge = {"type":"grade_separated_training_overpass","surface_height":height,"deck_thickness":thickness,
        "minimum_vertical_clearance_m":height-thickness,"vehicle_height_envelope_m":.65,
        "lower_s":float(lower_s),"upper_s":float(upper_s),"crossing_xy":list(crossing_xy),
        "profile_center_s":float(profile_center),
        "elevated_start_s":float(profile_center-half_flat-ramp),"elevated_end_s":float(profile_center+half_flat+ramp),
        "plateau_start_s":float(profile_center-half_flat),"plateau_end_s":float(profile_center+half_flat),
        "ramp_length_m":ramp,"plateau_length_m":2*half_flat,"profile":"cubic smoothstep; sampled piecewise-linear surface",
        "first_deck_point_index":first,"last_deck_point_index":last,
        "maximum_analytic_grade":1.5*height/ramp,"reference_s_units":"horizontal centerline metres",
        "physical_geometry_required":True,"requires_height_aware_localization_and_sensors":True,
        "source_relation_url":"https://global.honda/en/F1/race/2026/Japan/preview/",
        "assumption":"Crossing order follows back-straight over Degner. All bridge heights, ramp dimensions and flat remainder are RC training assumptions, not surveyed elevations."}
    return elevation,vertices,faces,boundaries,bridge


def write_obj(path, vertices, faces):
    with path.open("w") as f:
        kind = "grade-separated" if np.ptp(vertices[:,2])>1e-9 else "flat"
        f.write(f"# F1-derived approximate {kind} training road. Metres, Z up.\nmtllib track.mtl\nusemtl asphalt\n")
        for x,y,z in vertices:
            f.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
        for x,y,_ in vertices:
            f.write(f"vt {x/2:.7f} {y/2:.7f}\n")
        sloped = bool(np.ptp(vertices[:,2])>1e-9)
        if sloped:
            tri = vertices[faces]
            normals = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normals /= np.linalg.norm(normals,axis=1)[:,None]
            for n in normals:
                f.write("vn " + " ".join(f"{v:.7f}" for v in n) + "\n")
        else:
            f.write("vn 0 0 1\n")
        for fi,face in enumerate(faces+1):
            ni = fi+1 if sloped else 1
            f.write("f " + " ".join(f"{i}/{i}/{ni}" for i in face) + "\n")


def write_dae(path, vertices, faces):
    ns = "http://www.collada.org/2005/11/COLLADASchema"
    ET.register_namespace("", ns)
    def sub(parent, tag, text=None, **attr):
        e = ET.SubElement(parent, "{"+ns+"}"+tag, {k:str(v) for k,v in attr.items()})
        e.text = text
        return e
    root = ET.Element("{"+ns+"}COLLADA", version="1.4.1")
    asset = sub(root,"asset")
    sub(asset,"created","2026-09-12T00:00:00Z")
    sub(asset,"modified","2026-09-12T00:00:00Z")
    sub(asset,"unit",name="meter",meter="1")
    sub(asset,"up_axis","Z_UP")
    images = sub(root,"library_images")
    sub(sub(images,"image",id="asphalt-image"),"init_from","textures/asphalt.png")
    profile = sub(sub(sub(root,"library_effects"),"effect",id="asphalt-effect"),"profile_COMMON")
    surface = sub(sub(profile,"newparam",sid="surface"),"surface",type="2D")
    sub(surface,"init_from","asphalt-image")
    sampler = sub(sub(profile,"newparam",sid="sampler"),"sampler2D")
    sub(sampler,"source","surface")
    phong = sub(sub(profile,"technique",sid="common"),"phong")
    sub(sub(phong,"diffuse"),"texture",texture="sampler",texcoord="UVSET0")
    sub(sub(phong,"specular"),"color","0.02 0.02 0.02 1")
    material = sub(sub(root,"library_materials"),"material",id="asphalt-material")
    sub(material,"instance_effect",url="#asphalt-effect")
    mesh = sub(sub(sub(root,"library_geometries"),"geometry",id="road"),"mesh")
    def source(sid, data, names):
        e = sub(mesh,"source",id=sid)
        sub(e,"float_array"," ".join(f"{v:.7f}" for v in np.asarray(data).ravel()),id=sid+"-array",count=np.asarray(data).size)
        access = sub(sub(e,"technique_common"),"accessor",source="#"+sid+"-array",count=len(data),stride=len(names))
        for name in names:
            sub(access,"param",name=name,type="float")
    source("positions",vertices,["X","Y","Z"])
    source("uv",vertices[:,:2]/2,["S","T"])
    sloped = bool(np.ptp(vertices[:,2])>1e-9)
    if sloped:
        tri = vertices[faces]
        normals = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normals /= np.linalg.norm(normals,axis=1)[:,None]
    else:
        normals = [[0,0,1]]
    source("normals",normals,["X","Y","Z"])
    sub(sub(mesh,"vertices",id="road-vertices"),"input",semantic="POSITION",source="#positions")
    tris = sub(mesh,"triangles",count=len(faces),material="asphalt")
    sub(tris,"input",semantic="VERTEX",source="#road-vertices",offset="0")
    sub(tris,"input",semantic="TEXCOORD",source="#uv",offset="1",set="0")
    sub(tris,"input",semantic="NORMAL",source="#normals",offset="2")
    sub(tris,"p"," ".join(f"{i} {i} {fi if sloped else 0}" for fi,face in enumerate(faces) for i in face))
    scene = sub(sub(root,"library_visual_scenes"),"visual_scene",id="Scene")
    inst = sub(sub(scene,"node",id="road-node"),"instance_geometry",url="#road")
    binding = sub(sub(sub(inst,"bind_material"),"technique_common"),"instance_material",symbol="asphalt",target="#asphalt-material")
    sub(binding,"bind_vertex_input",semantic="UVSET0",input_semantic="TEXCOORD",input_set="0")
    sub(sub(root,"scene"),"instance_visual_scene",url="#Scene")
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)


def texture(path):
    rng = np.random.default_rng(71)
    gray = rng.integers(62,82,size=(128,128),dtype=np.uint8)
    Image.fromarray(np.stack([gray]*3,axis=-1)).save(path)


def map_image(path, points, road, name, elevation=None):
    image = Image.new("RGB",(1024,768),(22,34,31))
    d = ImageDraw.Draw(image)
    lo = points.min(axis=0); hi = points.max(axis=0)
    scale = min(920/max(hi[0]-lo[0],1),620/max(hi[1]-lo[1],1))
    def pixel(xy):
        return [(float((x-lo[0])*scale+52),float(705-(y-lo[1])*scale)) for x,y in xy]
    polys = list(road.geoms) if road.geom_type == "MultiPolygon" else [road]
    for poly in polys:
        d.polygon(pixel(poly.exterior.coords), fill=(96,101,110))
        for h in poly.interiors:
            d.polygon(pixel(h.coords),fill=(22,34,31))
    d.line(pixel(np.vstack([points,points[0]])),fill=(235,200,71),width=1)
    if elevation is not None:
        elevated = np.flatnonzero(np.asarray(elevation)>1e-9)
        if len(elevated):
            d.line(pixel(points[elevated]),fill=(62,218,232),width=4)
            d.text((30,56),"CYAN: physical bridge + ramps | assumed RC elevation, maximum 0.90 m",fill=(62,218,232))
    x,y = pixel(points[:1])[0]
    d.ellipse((x-4,y-4,x+4,y+4),fill=(255,255,255))
    d.text((30,18),name,fill="white")
    d.text((30,37),"Approximate source snapshot | RC training derivative | start = source first point",fill=(200,210,211))
    image.save(path)


def build(root=ROOT, selected=None):
    sources = root/"sources"
    raw = (sources/"f1-circuits.geojson").read_bytes()
    provenance = json.loads((sources/"manifest.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == provenance["files"]["f1-circuits.geojson"]["sha256"]
    features = {f["properties"]["id"]:f for f in json.loads(raw)["features"]}
    track_info = {}
    for tid, source_id in IDS.items():
        feature = features[source_id]
        name = feature["properties"]["Name"]
        track_info[tid] = {"id":tid,"name":name,"path":f"{tid}/track.json",
            "layout_id":f"{tid}-source-snapshot", "geometry_status":"approximate_snapshot",
            "historical_layout_verified":False,"scored_racing_supported":tid!="suzuka",
            "centerline_scale":RC_CENTERLINE_SCALES.get(tid,0.05),"road_width_m":1.5,
            "geometry_revision":"rc-r2-annular" if tid in RC_CENTERLINE_SCALES else "rc-r1"}
        if tid == "suzuka":
            track_info[tid].update(scored_racing_supported=True,geometry_revision="rc-r2-bridge",
                requires_3d_backend=True,has_elevation=True)
        if selected and tid not in selected:
            continue
        directory = root/tid
        directory.mkdir(parents=True,exist_ok=True)
        metric, origin = metric_coordinates(feature["geometry"]["coordinates"])
        source_length = float(np.linalg.norm(np.roll(metric,-1,axis=0)-metric,axis=1).sum())
        dump(directory/"source_metric.json",{"units":"metres","coordinates":metric.tolist(),
            "origin_lon_lat":origin.tolist(),"projection":"local equirectangular; Earth radius 6371008.8 m",
            "source_length":source_length,"source_feature_id":source_id})
        for variant, factor, width, spacing in [("rc",RC_CENTERLINE_SCALES.get(tid,0.05),1.5,0.25),("full_scale",1.0,12.0,5.0)]:
            dest = directory if variant=="rc" else directory/variant
            (dest/"textures").mkdir(parents=True,exist_ok=True)
            points = resample(metric*factor,spacing)
            vertices,faces,rings,road = road_mesh(points,width)
            topology_changed = len(rings) != 2
            warnings = ["Approximate contemporary geometry; historical season layout is unverified.",
                "Flat surface: no surveyed elevation, banking, barriers, kerbs, pit lane or start grid.",
                "Source tracing order and first coordinate define training direction and start."]
            if tid == "suzuka":
                warnings.append("Suzuka bridge is flattened into an at-grade crossing; unsuitable for scored racing until grade separation is modeled.")
            if variant == "rc":
                if tid in RC_CENTERLINE_SCALES:
                    warnings.append(f"Adaptive nonuniform RC training derivative: centerline scaled {factor:g}, width independently set to 1.5 m, to separate nearby route arms.")
                else:
                    warnings.append("Nonuniform training derivative: centerline scaled 0.05, width independently set to 1.5 m; nearby lanes may merge.")
            else:
                warnings.append("Road width 12 m is an assumption; source has no surveyed road widths.")
            if topology_changed:
                warnings.append("Road buffer joins nearby or crossing route arms; ordered route progress is required to reject shortcuts.")
            length = float(np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1).sum())
            metadata = {"id":tid,"name":name,"variant":variant,"centerline":points.round(8).tolist(),
                "width":width,"length":length,"boundary_rings":rings,"closed":True,
                "layout_id":f"{tid}-source-snapshot","historical_layout_verified":False,
                "scale":{"centerline":factor,"road_width_m":width,"uniform":False,
                    "description":"RC widened training derivative" if variant=="rc" else "full-size approximate path; assumed constant width"},
                "source":{"repository":provenance["repository"],"commit":provenance["commit"],
                    "sha256":hashlib.sha256(raw).hexdigest(),"feature_id":source_id,
                    "license":"MIT","attribution":"Tomislav Bacinger; see tracks/sources/LICENSE.md",
                    "source_reported_length_m":feature["properties"]["length"],"projected_length_m":source_length},
                "georeference":{"origin_lon_lat":origin.tolist(),"axes":"x east, y north, z up",
                    "projection":"local equirectangular; radius 6371008.8 m"},
                "training_suitability":{"scored_racing_supported":tid!="suzuka",
                    "requires_grade_separation":tid=="suzuka","width_supports_multiple_rc_cars":variant=="rc",
                    "road_buffer_joins_route_arms":topology_changed,"requires_ordered_progress":True},
                "mesh":{"vertices":len(vertices),"triangles":len(faces),"area_m2":road.area,
                    "boundary_rings":len(rings),"up_axis":"Z_UP","units":"metres",
                    "road_dae":"track.dae","road_obj":"track.obj","collision_obj":"collision.obj",
                    "texture":"textures/asphalt.png"},"warnings":warnings}
            if variant == "rc" and tid in RC_CENTERLINE_SCALES:
                metadata["geometry_revision"] = "rc-r2-annular"
                metadata["scale"]["description"] = "adaptive RC training geometry: enlarged centerline, fixed 1.5 m road"
            if variant == "rc" and tid == "suzuka":
                elevation,vertices,faces,boundaries,bridge = suzuka_bridge(points,width)
                metadata.update(elevation=elevation.tolist(),road_vertices=vertices.tolist(),road_faces=faces.tolist(),
                    boundary_segments_3d=boundaries.tolist(),bridge=bridge,geometry_revision="rc-r2-bridge",has_elevation=True)
                metadata["training_suitability"].update(scored_racing_supported=True,requires_grade_separation=False,
                    grade_separation_modeled=True,requires_3d_backend=True,road_buffer_joins_route_arms=False)
                metadata["warnings"] = [w for w in warnings if not w.startswith(("Flat surface:","Suzuka bridge is flattened","Road buffer joins"))]
                metadata["warnings"] += [bridge["assumption"],
                    "Use the shared 3D deck, elevated boundaries, height-aware laser and continuous route localization; the 2D compatibility projection alone cannot distinguish levels."]
                tri = vertices[faces]
                metadata["mesh"].update(vertices=len(vertices),triangles=len(faces),
                    area_m2=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2),
                    surface="grade-separated road surface",boundary_rings_2d_compatibility_only=True)
            dump(dest/"track.json",metadata)
            write_dae(dest/"track.dae",vertices,faces)
            write_obj(dest/"track.obj",vertices,faces)
            write_obj(dest/"collision.obj",vertices,faces)
            (dest/"track.mtl").write_text("newmtl asphalt\nKa 0.2 0.2 0.2\nKd 1 1 1\nKs 0.02 0.02 0.02\nmap_Kd textures/asphalt.png\n")
            texture(dest/"textures/asphalt.png")
            map_image(dest/"map.png",points,road,name,metadata.get("elevation"))
        print(f"Built {tid}: {source_length:.1f} source metres")
    catalog_data = {"schema_version":1,"description":"Official calendar indexing of approximate F1 circuit training assets",
        "calendar_sources":{str(y):f"https://www.formula1.com/en/racing/{y}" for y in CALENDARS},
        "geometry_source":provenance,"layout_policy":"Shared contemporary source snapshot; no season-exact geometry claim.",
        "tracks":track_info,"seasons":{str(year):[{"season":year,"round":i+1,"date":f"{year}-{date}",
            "circuit_id":tid,"track_id":tid,"layout_id":f"{tid}-source-snapshot",
            "centerline_scale":RC_CENTERLINE_SCALES.get(tid,0.05),
            "geometry_revision":"rc-r2-bridge" if tid=="suzuka" else ("rc-r2-annular" if tid in RC_CENTERLINE_SCALES else "rc-r1"),
            "path":f"{tid}/track.json","historical_layout_verified":False} for i,(tid,date) in enumerate(rounds)]
            for year,rounds in CALENDARS.items()}}
    dump(root/"catalog.json",catalog_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track",action="append",choices=IDS)
    args = parser.parse_args()
    build(selected=args.track)
