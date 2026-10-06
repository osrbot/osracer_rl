# Core package layout

The Python package is split by responsibility:

- `control/`: policies, checkpoint bundles, safety, and closed-loop control.
- `perception/`: sensor contracts and local odometry.
- `simulators/`: the shared simulator lifecycle and isolated Isaac/MuJoCo backends.
- `runtime/`: training, rollout, and sweep orchestration.
- `evaluation/`: metrics, qualification, verification, and reports.
- `tracks/`: track catalog and offline asset builder.
- `vehicle/`: real-vehicle adapters.
- `config/`: packaged simulator targets and training profiles.

Usage and engineering documentation starts at [docs/README.md](../../docs/README.md).

Import backends through `racing.simulators.create_simulator`. The obsolete top-level `isaac_env.py` and `mujoco_env.py` shims have been removed; see [migration notes](../../docs/LEGACY.md).
