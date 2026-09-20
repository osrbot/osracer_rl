# Track geometry, calendars and limitations

The catalog contains every championship round of **2024 and 2025**: 48 dated
round records referencing 24 circuit assets. It deliberately separates a race's
season/date from its geometry revision. All layouts are labeled
`<circuit>-source-snapshot`; **the geometry is not verified as season-exact**.
This release is not a complete archive of all Formula 1 seasons.

## Calendar facts

Round order and local race dates were checked against the official
[2024 Formula 1 calendar](https://www.formula1.com/en/racing/2024) and
[2025 Formula 1 calendar](https://www.formula1.com/en/racing/2025), accessed
2026-09-12 UTC. Testing and sprint races are not separate championship rounds.
Calendar policy is also supported by the FIA's
[2024 announcement](https://www.fia.com/news/2024-fia-formula-1-calendar-announced)
and [2025 announcement](https://www.fia.com/news/fia-and-formula-1-announce-2025-calendar).
The catalog preserves those official calendar URLs. Only factual calendar data
is reproduced; no Formula 1 logos, photographs or commercial simulator assets
are included.

## Geometry provenance and licensing

The source is [Tomislav Bacinger's f1-circuits GeoJSON repository](https://github.com/bacinger/f1-circuits),
pinned to commit `394d8fbe70ef2c0b0c8d23ff7bee61fa09606055`.
The original GeoJSON, README, and complete MIT license are retained under
`tracks/sources/`. `manifest.json` records the download time, immutable source
URLs and SHA-256 of each file. Downloads were rechecked against that exact
commit. The builder verifies the GeoJSON hash before generating any assets.
Each `track.json` records the source feature ID, source hash, attribution and
projected/source-reported lengths.

Copyright and permission details are in
[`tracks/sources/LICENSE.md`](../tracks/sources/LICENSE.md). Retain that file
with redistributed geometry and its derivatives. The procedural asphalt
texture and generated meshes contain no third-party image textures.

## Geometry transformations

Original longitude/latitude points are preserved in the vendored GeoJSON.
`source_metric.json` projects each circuit into local east/north metres, using
the source first coordinate as origin and an equirectangular projection with
Earth radius 6,371,008.8 m. This is a local metric approximation, not a survey or
a national grid CRS. The reference longitude/latitude and projection are
recorded. No fit to the published lap length is applied.

Two independent-width road variants are generated:

| Variant | Centerline factor | Road width | Sampling target |
| --- | ---: | ---: | ---: |
| Default RC training | 0.05 | 1.5 m | 0.25 m |
| Baku RC revision 2 | 0.100 | 1.5 m | 0.25 m |
| Monaco RC revision 2 | 0.125 | 1.5 m | 0.25 m |
| Miami / Montreal RC revision 2 | 0.075 | 1.5 m | 0.25 m |
| Suzuka RC physical bridge revision 2 | 0.05 | 1.5 m | 0.25 m |
| `full_scale/` | 1.0 | 12 m, assumed | 5 m |

The RC road is a **nonuniformly widened training derivative**, not a uniformly
scaled replica. In particular 12 m × 0.05 would be 0.6 m, whereas this road is
1.5 m wide to allow multiple NEORACER cars to run side by side. The original
data supplies no surveyed road widths; the full-scale 12 m width is also an
assumption. Uniform arc sampling slightly shortens sharp polygonal corners;
both the input projected length and output polyline length are recorded.

The four explicit RC overrides enlarge the entire centerline while retaining
1.5 m road width, separating nearby arms into a road with one outside and one
inside boundary. This is **adaptive RC training geometry**, not a correction
to the source or a historical layout. `RC_CENTERLINE_SCALES` in the builder
defines these factors; the catalog and each asset record them. Revision 2 is
identified as `rc-r2-annular`. The original projected coordinates, source pin
and full-scale assets are unchanged. Prior RC assets and their catalog are
archived at `output/racing/track_revision1/`. Results generated with those
original track hashes do not qualify the revised geometry; rerun those four
tracks after the revision.

Flat roads are polygon unions of constant-width buffers around the closed
centerline. Constrained triangulation preserves interior holes and avoids
inverted overlapping strip faces at hairpins. Suzuka RC combines a buffered
flat road with a separate raised bridge and ramps. The DAE and OBJ road meshes
have identical geometry, metre units, upward faces and Z-up coordinates. They
are open road surfaces, not watertight solids. `collision.obj` is the same
surface for static triangle-mesh collision consumers. A separate PNG is bound
through COLLADA effects/materials and OBJ's MTL, with planar repeating UVs.

## Known limitations that affect simulation

- **Suzuka RC:** `rc-r2-bridge` provides a physical overpass, ramps, elevated
  boundaries and height/route-continuous localization. The surface is 0.90 m
  high, its assumed physical deck thickness is 0.08 m and clearance is 0.82 m.
  Two 20 m ramps have maximum analytic grade 6.75%. These are explicit RC
  training assumptions, not surveyed elevations. `requires_3d_backend` must
  be honored by physics, sensors and scoring. The unchanged **full_scale**
  asset remains a flat reference and is still unsupported for bridge racing.
  See [Suzuka bridge schema](../tracks/suzuka/BRIDGE.md). Geometric support and
  native traversal tests do not establish full-lap policy qualification.
- **Nearby arms:** revision 1 joined nearby route arms at Miami, Monaco,
  Montreal and Baku. Revision 2 separates those four by increasing centerline
  scale at fixed width, and tests require exactly two nested boundary rings.
  Suzuka's RC crossing now has separated heights. Metadata flags
  `road_buffer_joins_route_arms` wherever planar arms remain merged. Enforce ordered progress to reject shortcuts.
  A global nearest-centerline query can jump between nearby arms; controllers
  should constrain progress using vehicle history when scoring laps.
- All roads omit surveyed elevation, banking, variable width, kerbs, barriers,
  pit lanes, surroundings and an official starting grid; Suzuka RC's added
  elevation is modeled rather than surveyed. Source tracing order
  and its first point define training direction and start; these have not been
  verified against official direction or start/finish positions.
- A season index is not evidence of a historically correct layout. In
  particular geometry updates at Melbourne, Barcelona, Singapore and Yas
  Marina require independent historical comparison before an exact-year claim.
- This source covers all 24 circuits needed here; no missing circuit has been
  replaced with a synthetic oval or mislabeled as surveyed geometry.

## Rebuilding and consuming

Install build dependencies `numpy`, `Pillow`, and `shapely>=2.1`, then run:

```sh
python -m racing.build_tracks
python -m racing.build_tracks --track bahrain
```

No network is required for a rebuild. `racing.tracks.Track` only requires
NumPy. It accepts a circuit ID, circuit directory or JSON path. `.points` is
an N×2 closed polyline without a duplicate endpoint, `.width` and `.length`
are metres. `at(s, offset)` returns `(xy, yaw)` and wraps distance around the
lap. `project(xy)` returns `(s, signed_cte, yaw)`. Positive offsets/errors are
left of travel; yaw is radians from +X. `boundary_segments()` returns N×2×2
segments for lidar. For the full-size variant use
`Track("tracks/bahrain/full_scale/track.json")`.

For elevated assets, `has_elevation` selects the three-dimensional API:
`at3d(s, offset)` returns `(xyz, yaw, grade)`, where grade is dimensionless
`dz/d(horizontal s)`; `elevation_at(s)` and `grade_at(s)` expose those values.
`road_triangles_3d()` returns K×3×3 surface triangles and
`boundary_segments_3d()` returns M×2×3 boundary endpoints. `project` accepts
`s_hint` (previous route arc distance) and `z`, or a three-component position,
to distinguish crossing levels. The original two-dimensional API remains
available for compatibility; by itself it cannot localize a crossing.
The `length` field continues to mean horizontal route arc distance.

`tests/test_tracks.py` validates all round mappings, source hashes, geometry,
projection signs/wrapping, XML texture bindings, both scales and mesh area.
When `pycollada` and `trimesh` are installed it independently parses every DAE
and OBJ variant and checks triangle winding/area. Parser success validates
interchange syntax and geometry; it does not certify historical accuracy or
simulator-specific rendering behavior.
