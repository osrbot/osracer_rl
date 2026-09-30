# Core package layout

The Python package is split by responsibility:

- `control/`: policies, checkpoint bundles, safety, and closed-loop control.
- `perception/`: sensor contracts and local odometry.
- `simulators/`: the shared simulator lifecycle and isolated Isaac/MuJoCo backends.
- `runtime/`: training, rollout, and sweep orchestration.
- `evaluation/`: metrics, qualification, verification, and reports.
- `tracks/`: track catalog and offline asset builder.
- `vehicle/`: real-vehicle adapters.
- `config/simulators/`: packaged backend target and default configuration.

`isaac_env.py` and `mujoco_env.py` are thin compatibility imports for older experiments. New code should import from `racing.simulators`.
