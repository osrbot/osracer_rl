# SW2URDF Export Report

Status: WARN
Generated: 2026-09-11 10:16:52 +08:00
Plugin version: 1.6
Commit version: a0af2c5a1ea9
Commit hash: a0af2c5a1ea9
Build version: 1.6.0.0
Build time UTC: 2026-09-11T09:18:55Z
Dirty state: false
Robot name: osracer_arc_e01_prtnew_asm
ROS package: osracer_description
Selected targets: ROS 1
Unselected target directories: retained and not validated by this export
Export meshes: true
Mesh format: STL
Export parameters: export_meshes=true, mesh_format=STL, inertial_validation_rows=210, mesh_manifest_rows=10, requested_collision_strategies=BoxPrimitive=1, ComponentBoxes=1, CylinderPrimitive=4, SimplifiedMesh=2, VisualMesh=2, effective_collision_strategies=BoxPrimitive=1, ComponentBoxes=1, CylinderPrimitive=4, SimplifiedMesh=2, VisualMesh=2, collision_urdf_refs=mesh=10, stl_reduction_ratios=0.7=10, stl_quality_settings=coarse=10
Elapsed: 01:14

## Health Summary

| Check | Status | Detail |
| --- | --- | --- |
| Overall | WARN | failures=0, warnings=8 |
| ROS 1 URDF | PASS | links=10, joints=9, mesh_refs=20, missing_mesh_refs=0, duplicate_links=0, duplicate_joints=0 |
| ROS package completeness | PASS | critical_missing=0, optional_missing=0 |
| ROS package parity | SKIP | critical_mismatches=0, optional_mismatches=0 |
| Inertial validation | PASS | rows=210, failures=0, warnings=0 |
| Mesh manifest paths | PASS | rows=10, missing_visual=0, missing_collision=0 |
| Collision strategy | PASS | fallbacks=0, requested=BoxPrimitive=1, ComponentBoxes=1, CylinderPrimitive=4, SimplifiedMesh=2, VisualMesh=2, effective=BoxPrimitive=1, ComponentBoxes=1, CylinderPrimitive=4, SimplifiedMesh=2, VisualMesh=2, urdf_refs=mesh=10 |
| STL reduction | WARN | stats_rows=10, high_estimate_errors=0, reduction_warnings=8, ratios=0.7=10 |

## Export Parameters

| Parameter | Value |
| --- | --- |
| output_root | C:\Users\kitso\Desktop\OSRACER\ |
| robot_name | osracer_arc_e01_prtnew_asm |
| ros1_package_name | osracer_description |
| ros2_package_name | osracer_description |
| ros1_package_directory | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\ |
| ros2_package_directory | C:\Users\kitso\Desktop\OSRACER\ROS2\osracer_description\ |
| ros1_urdf | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\urdf\osracer_arc_e01_prtnew_asm.urdf |
| ros2_urdf | C:\Users\kitso\Desktop\OSRACER\ROS2\osracer_description\urdf\osracer_arc_e01_prtnew_asm.urdf |
| export_meshes | true |
| mesh_format | STL |
| inertial_validation_rows | 210 |
| mesh_manifest_rows | 10 |
| requested_collision_strategies | BoxPrimitive=1, ComponentBoxes=1, CylinderPrimitive=4, SimplifiedMesh=2, VisualMesh=2 |
| effective_collision_strategies | BoxPrimitive=1, ComponentBoxes=1, CylinderPrimitive=4, SimplifiedMesh=2, VisualMesh=2 |
| collision_urdf_refs | mesh=10 |
| stl_reduction_ratios | 0.7=10 |
| stl_quality_settings | coarse=10 |

## ROS 1 URDF

- File: C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\urdf\osracer_arc_e01_prtnew_asm.urdf
- Exists: true
- XML parse: OK
- Root is robot: true
- Robot name: osracer_arc_e01_prtnew_asm
- Links: 10
- Joints: 9
- Duplicate link names: none
- Duplicate joint names: none
- Mesh references: 20
- Missing mesh references: 0

## Package Files

| Item | Status | Required | Path |
| --- | --- | --- | --- |
| ROS 1 package directory | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\ |
| ROS 1 CMakeLists.txt | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\CMakeLists.txt |
| ROS 1 package.xml | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\package.xml |
| ROS 1 URDF | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\urdf\osracer_arc_e01_prtnew_asm.urdf |
| ROS 1 config directory | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\config\ |
| ROS 1 launch directory | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\launch\ |
| ROS 1 display.launch | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\launch\display.launch |
| ROS 1 gazebo.launch | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\launch\gazebo.launch |
| ROS 1 meshes directory | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\meshes\ |
| ROS 1 visual meshes directory | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\meshes\visual |
| ROS 1 visual mesh files | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\meshes\visual |
| ROS 1 collision meshes directory | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\meshes\collision |
| ROS 1 collision mesh files | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\meshes\collision |
| ROS 1 inertial validation CSV | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\config\inertial_validation.csv |
| ROS 1 mesh manifest CSV | OK | yes | C:\Users\kitso\Desktop\OSRACER\ROS1\osracer_description\config\mesh_manifest.csv |

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

- Mesh manifest rows: 10
- Visual meshes present: 10/10
- Collision meshes present: 10/10
- Visual mesh bytes: 16203690
- Collision mesh bytes: 332240
- Visual STL triangles: 324057
- Collision STL triangles: 6628
- Original visual STL triangles: 827069
- Original visual STL bytes: 41354290
- Target ratios describe triangles to remove; actual results are measured from exported STL files.
- Average collision mesh byte reduction vs visual: 59.13%
- Average collision mesh triangle reduction vs visual: 59.15%
- Requested STL reduction ratios: 0.7=10
- STL quality settings: coarse=10
- Average actual STL reduction: 53.07%
- Requested collision strategies: BoxPrimitive=1, ComponentBoxes=1, CylinderPrimitive=4, SimplifiedMesh=2, VisualMesh=2
- Effective collision strategies: BoxPrimitive=1, ComponentBoxes=1, CylinderPrimitive=4, SimplifiedMesh=2, VisualMesh=2
- Collision URDF refs: mesh=10
- Collision strategy fallbacks: 0
- CSV: config/mesh_manifest.csv

## STL Reduction Details

| Link | Quality | Target removal ratio | Original bytes | Original triangles | Target triangles | Final bytes | Final triangles | Actual reduction | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_link | coarse | 0.7 | 32103784 | 642074 | 192623 | 10638284 | 212764 | 66.86% | reduced | Some regions failed reduction or quality checks and were retained exactly. |
| camera_link | coarse | 0.7 | 129984 | 2598 | 780 | 39084 | 780 | 69.98% | reduced |  |
| laser | coarse | 0.7 | 216684 | 4332 | 1300 | 65084 | 1300 | 69.99% | reduced |  |
| imu_link | coarse | 0.7 | 6361734 | 127233 | 38170 | 4331934 | 86637 | 31.91% | reduced | Some regions failed reduction or quality checks and were retained exactly. |
| left_steering_hinge_link | coarse | 0.7 | 59684 | 1192 | 358 | 41784 | 834 | 30.03% | reduced |  |
| left_front_wheel_link | coarse | 0.7 | 605684 | 12112 | 3634 | 272484 | 5448 | 55.02% | reduced |  |
| right_steering_hinge_link | coarse | 0.7 | 59684 | 1192 | 358 | 38784 | 774 | 35.07% | reduced |  |
| right_front_wheel_link | coarse | 0.7 | 605684 | 12112 | 3634 | 264284 | 5284 | 56.37% | reduced |  |
| left_rear_wheel_link | coarse | 0.7 | 605684 | 12112 | 3634 | 255984 | 5118 | 57.74% | reduced |  |
| right_rear_wheel_link | coarse | 0.7 | 605684 | 12112 | 3634 | 255984 | 5118 | 57.74% | reduced |  |

## Collision Strategies

| Link | Requested | Effective | Geometry | URDF collision ref | Notes | Collision artifact exists | Collision artifact bytes | Collision artifact triangles | Byte reduction vs visual | Triangle reduction vs visual | Collision artifact URI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_link | ComponentBoxes | ComponentBoxes | component_box_set_stl | package://osracer_description/meshes/collision/base_link.STL | ok | true | 127284 | 2544 | 98.8% | 98.8% | package://osracer_description/meshes/collision/base_link.STL |
| camera_link | SimplifiedMesh | SimplifiedMesh | simplified_stl | package://osracer_description/meshes/collision/camera_link.STL | target removal=70 %; triangles=2598->780; bytes=129984->39084; status=reduced; limited=False; reason= | true | 39084 | 780 | 0% | 0% | package://osracer_description/meshes/collision/camera_link.STL |
| laser | SimplifiedMesh | SimplifiedMesh | simplified_stl | package://osracer_description/meshes/collision/laser.STL | target removal=70 %; triangles=4332->1300; bytes=216684->65084; status=reduced; limited=False; reason= | true | 65084 | 1300 | 0% | 0% | package://osracer_description/meshes/collision/laser.STL |
| imu_link | BoxPrimitive | BoxPrimitive | box_primitive_stl | package://osracer_description/meshes/collision/imu_link.STL | ok | true | 684 | 12 | 99.98% | 99.99% | package://osracer_description/meshes/collision/imu_link.STL |
| left_steering_hinge_link | VisualMesh | VisualMesh | visual_mesh_copy | package://osracer_description/meshes/collision/left_steering_hinge_link.STL | ok | true | 41784 | 834 | 0% | 0% | package://osracer_description/meshes/collision/left_steering_hinge_link.STL |
| left_front_wheel_link | CylinderPrimitive | CylinderPrimitive | cylinder_primitive_stl | package://osracer_description/meshes/collision/left_front_wheel_link.STL | ok | true | 4884 | 96 | 98.21% | 98.24% | package://osracer_description/meshes/collision/left_front_wheel_link.STL |
| right_steering_hinge_link | VisualMesh | VisualMesh | visual_mesh_copy | package://osracer_description/meshes/collision/right_steering_hinge_link.STL | ok | true | 38784 | 774 | 0% | 0% | package://osracer_description/meshes/collision/right_steering_hinge_link.STL |
| right_front_wheel_link | CylinderPrimitive | CylinderPrimitive | cylinder_primitive_stl | package://osracer_description/meshes/collision/right_front_wheel_link.STL | ok | true | 4884 | 96 | 98.15% | 98.18% | package://osracer_description/meshes/collision/right_front_wheel_link.STL |
| left_rear_wheel_link | CylinderPrimitive | CylinderPrimitive | cylinder_primitive_stl | package://osracer_description/meshes/collision/left_rear_wheel_link.STL | ok | true | 4884 | 96 | 98.09% | 98.12% | package://osracer_description/meshes/collision/left_rear_wheel_link.STL |
| right_rear_wheel_link | CylinderPrimitive | CylinderPrimitive | cylinder_primitive_stl | package://osracer_description/meshes/collision/right_rear_wheel_link.STL | ok | true | 4884 | 96 | 98.09% | 98.12% | package://osracer_description/meshes/collision/right_rear_wheel_link.STL |

## Findings

- WARN: STL reduction for link base_link status=reduced, target_triangles=192623, actual_triangles=212764, reason=Some regions failed reduction or quality checks and were retained exactly.
- WARN: STL reduction for link imu_link status=reduced, target_triangles=38170, actual_triangles=86637, reason=Some regions failed reduction or quality checks and were retained exactly.
- WARN: STL reduction for link left_steering_hinge_link status=reduced, target_triangles=358, actual_triangles=834.
- WARN: STL reduction for link left_front_wheel_link status=reduced, target_triangles=3634, actual_triangles=5448.
- WARN: STL reduction for link right_steering_hinge_link status=reduced, target_triangles=358, actual_triangles=774.
- WARN: STL reduction for link right_front_wheel_link status=reduced, target_triangles=3634, actual_triangles=5284.
- WARN: STL reduction for link left_rear_wheel_link status=reduced, target_triangles=3634, actual_triangles=5118.
- WARN: STL reduction for link right_rear_wheel_link status=reduced, target_triangles=3634, actual_triangles=5118.

