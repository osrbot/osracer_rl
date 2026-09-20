# Experimental native wheel contact adapter

This module corrects finite-road contact normals for the existing 45 mm radius / 40 mm wide ideal wheel cylinder. It registers `BridgeCylinderAPI` through the public `omni::physx::IPhysxCustomGeometry` interface. **Production `RaceIsaacEnv`, shared track geometry, friction, drives, mass and inertia remain unchanged.** Use the explicit experimental environment to run it.

## Evidence and scope

The frozen v3 native comparison passed both a finite 4.5% box incline and an infinite-plane control at approximately **6.98 m/s** over 30 m. Finite-box maximum height error was **0.135 mm**, and roll **0.00130 rad**. Native body mass/inertia readbacks were identical before and after registration. Actual PhysX custom shape IDs, contact calls and zero unsupported pairs were recorded. See `output/racing/bridge_adapter_matched_contacts.json` and `output/racing/bridge_adapter_sdk/v3_frozen/`. These are privileged contact fixtures, not sensor-only laps.

The v5 native mesh fixture traversed the original smoothstep Suzuka bridge at **6.906 m/s entry / 6.948 m/s exit**, with **3.852 mm** maximum height error and **0.01193 rad** maximum roll. The segment ends 1 m beyond the bridge, matching the older `race_bridge_probe.py` geometry fixture. The extended test then struck a barrier at `s=258.622 m` in the next corner under its fixed 155 rad/s privileged steering command; **the original extended fixture remains failed**. Its complete trace and a separate segment audit are retained as `bridge_adapter_full_v5_full_bridge.json` and `bridge_adapter_full_v5_segment_audit.json`. The independent upper/lower crossing passed with 0.900004 m height separation and 0.02498 m horizontal separation. This is contact and clearance evidence, not a successful sensor-only lap.

The v4 mesh counterexample is retained: individual triangle edges produced a 1.669 N·s front-wheel impulse and launched the car on the ramp. V5 welds actual cooked vertices, removes coplanar/internal concave edge candidates, and accepts exposed convex edges only within the adjacent face-normal cone. Real finite perimeter edges and underside contacts remain. CPU coverage includes a repeated coplanar seam, a genuine outer cube edge with quantitative normal/depth checks, and below-deck separation. The current kernel suite contains **401** cases, plus 7 native query self-checks. Non-manifold input meshes have not been qualified.

The current implementation supports:

| Collision/query | Implementation |
|---|---|
| Plane and finite box face | Analytical cylinder contacts with exact face normals and finite segment clipping |
| Triangle mesh face | Actual cooked mesh triangles; analytical face contacts and finite clipping |
| Edges, convex meshes, other adapter wheels, spheres, capsules, convex cores | Analytical support functions with double-precision GJK/EPA |
| Raycast | Analytical cylinder side/cap intersection |
| Overlap | Actual supported geometry intersection |
| Sweep | Conservative advancement for translation-only queries |

Heightfields, particles, deformables and unrelated custom geometries are outside this racing-scene scope. Unsupported pairs, numerical failures and unconverged queries are counted and invalidate qualification. New geometry coverage must pass native bridge, clearance and collision checks before promotion; CPU geometry tests alone do not demonstrate a PhysX vehicle fix.

No infinite or moving hidden support is used. A box has its real finite sides and underside. Narrow-phase queries may clip **remote** box/triangle volume outside a strict superset of the complete cylinder AABB: extra distance is contact margin plus 1 cm. New query faces/edges cannot touch the cylinder within that margin. The simulated geometry is never changed. A long/short equivalent box test checks this invariant. Historical coarse FCL/EPA edge counterexamples are retained in `output/racing/bridge_adapter_sdk/edge_solver_counterexamples.json`.

## Build from a clean project

Prerequisites: installed Isaac Sim **6.0.1**, C++17 compiler, Eigen, FCL 0.7 and libccd. Ubuntu packages: `g++ libeigen3-dev libfcl-dev libccd-dev`. Set `NEORACER_ISAAC_DIR` if Isaac is installed outside the project's default location.

```sh
python3 experimental/bridge_contact_adapter/fetch_sdk.py
bash experimental/bridge_contact_adapter/build.sh
```

The fetcher downloads official archives without cloning, checks pinned SHA-256 digests, and extracts public headers and licenses. It never replaces installed libraries. SDK/build outputs live in `output/racing/bridge_adapter_sdk/` and can be regenerated.

| Dependency | Pinned source / license |
|---|---|
| PhysX and Omni headers | NVIDIA `110.1-omni-and-physx-5.9.0`; source BSD-3-Clause notices |
| OpenUSD headers | Pixar `v25.11`; modified Apache-2.0 license in upstream LICENSE.txt |
| Kit/Carbonite | Existing Isaac installation; NVIDIA's supplied license, not copied by the fetcher |
| FCL/libccd | System packages; upstream BSD license terms |

Downloaded license files are retained under the SDK output's `licenses/`. USD headers are configured for the installed `pxrInternal_v0_25_11` namespace. The SDK source tag does not establish the exact NVIDIA binary patch commit; runtime metadata states that limitation.

## Reproduce checks

```sh
g++ -std=c++17 -O2 -I/usr/include/eigen3 experimental/bridge_contact_adapter/contact_kernel.cpp experimental/bridge_contact_adapter/mesh_surface.cpp experimental/bridge_contact_adapter/test_kernel.cpp -lfcl -lccd -loctomap -loctomath -o output/racing/bridge_adapter_sdk/test_kernel
output/racing/bridge_adapter_sdk/test_kernel
bash scripts/run_isaac.sh scripts/probe_inclined_plane.py --custom-adapter --support box --tag bridge_adapter_check
bash scripts/run_isaac.sh scripts/probe_inclined_plane.py --custom-adapter --support plane --tag bridge_adapter_check
bash scripts/run_isaac.sh experimental/bridge_contact_adapter/probe_full_bridge.py --tag bridge_adapter_full_check
```

The full bridge probe uses the actual shared Suzuka triangle bridge, a 155 rad/s wheel command and a separate simultaneous upper/lower crossing. `--render` adds explicitly labelled fixture screenshots. Every run saves trajectories, raw support contacts and native qualification. It does not grant a leaderboard entry.

`experimental.bridge_contact_adapter.runtime.AdapterRaceIsaacEnv.physics_metadata` exposes binary SHA-256, source hashes, SDK tag, physics/control frequency, support scope, actual shape/scene readback, native geometry inventory, all body mass/inertia before and after, and failure counters. Evaluation must include this metadata in its identity so different shared libraries cannot share one physics identity. The custom USD schema must be discoverable **before** application startup; the experimental environment arranges this.

## Separate GPU SDF investigation

The earlier CPU-authored `sdf120` probe did not prove SDF collision was active. A later GPU test inspected actual `PxShape` triangle meshes, non-null SDF arrays and dimensions, and actual GPU dynamics/PCM scene flags through the public `IPhysx` interface. GPU SDF remained unstable in that test. `--gpu-dynamics --representation sdf120` reproduces this distinct diagnostic. The default CPU tensor view cannot supply SDF sampling, which is recorded separately from the successful native shape inspection.

## Frozen actor experiment

Create an isolated copy of a checkpoint's `racing`, `scripts`, `tracks`, and `candidate.json`; keep actor modules exactly as recorded by `actor_source_sha256`. Copy the current backend, adapter source/schema, and **both** `libbridge_contact_adapter.so` and its `.build.json` into that workspace. The loader verifies binary/manifest agreement, and the actor probe verifies checkpoint/source agreement. Do not replace a shared library while a simulation has it loaded.

```sh
bash scripts/run_isaac.sh experimental/bridge_contact_adapter/probe_actor.py --checkpoint candidate.json --tag suzuka_c03_adapter_development --seconds 200 --seed 0
```

Results and full traces are saved beneath the executing workspace's `output/racing/isaac/`. The experiment records actor, Python runtime, compiled adapter, loader, SDK and native physics identities independently. Actual world time step, settling duration and articulation solver iterations are read back in `physics_metadata`. A valid lap requires both race validity and native adapter qualification.
