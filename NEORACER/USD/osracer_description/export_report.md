# SW2URDF Export Report

Status: WARN
Generated: 2026-09-11 10:16:54 +08:00
Plugin version: 1.6
Commit version: a0af2c5a1ea9
Commit hash: a0af2c5a1ea9
Build version: 1.6.0.0
Build time UTC: 2026-09-11T09:18:55Z
Dirty state: false
Robot name: osracer_arc_e01_prtnew_asm
ROS package: osracer_description
Selected targets: OpenUSD
Unselected target directories: retained and not validated by this export
Export meshes: true
Mesh format: STL
Export parameters: export_meshes=true, mesh_format=STL, inertial_validation_rows=210, mesh_manifest_rows=0, requested_collision_strategies=none, effective_collision_strategies=none, collision_urdf_refs=none, stl_reduction_ratios=none, stl_quality_settings=none
Elapsed: 01:16

## Health Summary

| Check | Status | Detail |
| --- | --- | --- |
| Overall | WARN | failures=0, warnings=1 |
| ROS package completeness | SKIP | critical_missing=0, optional_missing=0 |
| ROS package parity | SKIP | critical_mismatches=0, optional_mismatches=0 |
| Inertial validation | PASS | rows=210, failures=0, warnings=0 |
| Mesh manifest paths | WARN | rows=0, missing_visual=0, missing_collision=0 |
| Collision strategy | PASS | fallbacks=0, requested=none, effective=none, urdf_refs=none |
| STL reduction | WARN | stats_rows=0, high_estimate_errors=0, reduction_warnings=0, ratios=none |

## Export Parameters

| Parameter | Value |
| --- | --- |
| output_root | C:\Users\kitso\Desktop\NEORACER\ |
| robot_name | osracer_arc_e01_prtnew_asm |
| ros1_package_name | osracer_description |
| ros2_package_name | osracer_description |
| ros1_package_directory | C:\Users\kitso\Desktop\NEORACER\ROS1\osracer_description\ |
| ros2_package_directory | C:\Users\kitso\Desktop\NEORACER\ROS2\osracer_description\ |
| ros1_urdf | C:\Users\kitso\Desktop\NEORACER\ROS1\osracer_description\urdf\osracer_arc_e01_prtnew_asm.urdf |
| ros2_urdf | C:\Users\kitso\Desktop\NEORACER\ROS2\osracer_description\urdf\osracer_arc_e01_prtnew_asm.urdf |
| export_meshes | true |
| mesh_format | STL |
| inertial_validation_rows | 210 |
| mesh_manifest_rows | 0 |
| requested_collision_strategies | none |
| effective_collision_strategies | none |
| collision_urdf_refs | none |
| stl_reduction_ratios | none |
| stl_quality_settings | none |

## Package Files

| Item | Status | Required | Path |
| --- | --- | --- | --- |

## ROS Package Parity

- Parity rows: 0
- Parity mismatches: 0

| Category | Relative path | ROS 1 | ROS 2 | Required |
| --- | --- | --- | --- | --- |

## Inertial Validation

- Inertial validation rows: 210
- Failed rows: 0
- Warning rows: 0
- Physical inertia failures: 0
- Magnitude warnings: 0
- Inertia display warnings: 0
- Display blocked by invalid physics: 0
- Display failed after valid physics: 0
- Failed links: none
- Warning links: none
- Magnitude warning links: none
- Display failure links: none
- CSV: config/inertial_validation.csv

### Inertial Link Summary

| Link | Coordinate system | Status | Rows | Numeric | Physical failures | Magnitude warnings | Display warnings | Max abs error | Max relative error | Failed quantities | Warning quantities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_link | base_link | PASS | 21 | 10 | 0 | 0 | 0 | 8.13151629364128E-20 | 0% | none | none |
| camera_link | camera_link | PASS | 21 | 10 | 0 | 0 | 0 | 3.46944695195361E-18 | 0% | none | none |
| imu_link | imu_link | PASS | 21 | 10 | 0 | 0 | 0 | 0 | 0% | none | none |
| laser | laser | PASS | 21 | 10 | 0 | 0 | 0 | 0 | 0% | none | none |
| left_front_wheel_link |  Left_front_wheel | PASS | 21 | 10 | 0 | 0 | 0 | 0 | 0% | none | none |
| left_rear_wheel_link | left_rear_wheel | PASS | 21 | 10 | 0 | 0 | 0 | 0 | 0% | none | none |
| left_steering_hinge_link | Left_steering_hinge | PASS | 21 | 10 | 0 | 0 | 0 | 0 | 0% | none | none |
| right_front_wheel_link |  right_front_wheel | PASS | 21 | 10 | 0 | 0 | 0 | 0 | 0% | none | none |
| right_rear_wheel_link | right_rear_wheel | PASS | 21 | 10 | 0 | 0 | 0 | 0 | 0% | none | none |
| right_steering_hinge_link | right_steering_hinge | PASS | 21 | 10 | 0 | 0 | 0 | 0 | 0% | none | none |

## Mesh Manifest

- Mesh manifest rows: 0
- Visual meshes present: 0/0
- Collision meshes present: 0/0
- Visual mesh bytes: 0
- Collision mesh bytes: 0
- Visual STL triangles: 0
- Collision STL triangles: 0
- Average collision mesh byte reduction vs visual: none
- Average collision mesh triangle reduction vs visual: none
- Requested STL reduction ratios: none
- STL quality settings: none
- Average actual STL reduction: none
- Requested collision strategies: none
- Effective collision strategies: none
- Collision URDF refs: none
- Collision strategy fallbacks: 0
- CSV: config/mesh_manifest.csv

## STL Reduction Details

| Link | Quality | Target removal ratio | Original bytes | Original triangles | Target triangles | Final bytes | Final triangles | Actual reduction | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | | | | | | | | none | | |

## Collision Strategies

| Link | Requested | Effective | Geometry | URDF collision ref | Notes | Collision artifact exists | Collision artifact bytes | Collision artifact triangles | Byte reduction vs visual | Triangle reduction vs visual | Collision artifact URI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | none | none | none | none | none | false |  |  | none | none |  |

## Findings

- WARN: No mesh manifest rows were produced.


Native geometry and relative asset references were validated by this target's exporter. See export_report.json for its format-specific validation results.
