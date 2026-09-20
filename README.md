# OSRACER: Evidence-First Research on Autonomous Ackermann Racing

[中文文档](README.zh-CN.md) · [Evidence site](https://osrbot.github.io/osracer_rl/) · [Validation ledger](docs/VALIDATION_STATUS.md) · [Project structure](docs/PROJECT_STRUCTURE.md)

OSRACER is an open research project on high-speed autonomous Ackermann racing in native **MuJoCo** and **Isaac Sim / PhysX**. It combines vehicle-asset validation, sensor-constrained driving, multi-track evaluation, trajectory audits, videos, and failure cases in one reproducible workflow.

Simulation output is treated as evidence to be qualified—not as a marketing claim.

## Research questions

1. Can exported vehicle assets be loaded, rendered, articulated, and stepped reliably in OpenUSD/Isaac Sim and MuJoCo?
2. With wheel speed, steering, and a 15 Hz laser history—rather than global actor pose—can an Ackermann vehicle finish a lap, overtake, and improve lap time under stated physical assumptions?
3. Which limits are simulator-specific, sensor-specific, or imposed by the real vehicle's actuation architecture?

## Reviewed snapshot

This is an active codebase; it does **not** claim that all driving objectives are solved. The [validation ledger](docs/VALIDATION_STATUS.md) retains supporting and contrary evidence together.

| Question | Evidence currently supported | Boundary that remains |
| --- | --- | --- |
| Exported assets | OpenUSD load/render/short-step evidence: 20 mesh references, 10 rigid bodies, 9 physics joints, 6 movable DoF. MuJoCo `robot.xml`/`scene.xml` pass a 500-step finite-state check. | Asset-level checks are **not** proof that the exported fixed-base model is drive-ready. |
| Laps and overtakes | v10c reports 24/24 valid MuJoCo laps with one audited overtake per lap; Isaac reports 22/24 under the matched qualification. | Isaac failures remain at Spa (initial car contact) and Suzuka (bridge instability). |
| Faster laps | At 9.0 m/s cruise, MuJoCo reports 24/24 valid laps and 8.97 m/s peak; Isaac reports 20/24 comparable valid laps at 9.00–9.04 m/s peak. | Higher Isaac speed trades speed for stability; it is not the shared robust operating point. |
| 180° drift | Simulator examples demonstrate sustained slip under their stated actuation assumptions. | The single-motor 4WD vehicle does **not** reproduce it: removing rear overdrive changes maximum slip from 27.0° to 4.36°. No real-world drift claim is made. |
| Sensor robustness | Noise-only behavior can pass selected checks. | With 0.02 m noise + 5% beam dropout + 50 ms latency, 10/10 perturbation runs fail. This is an open result, not a hidden caveat. |

## Method

```text
SolidWorks assembly
    └─ solidworks_urdf_exporter_pro ──> ROS descriptions + OpenUSD + MuJoCo MJCF
                                            └─ native asset validation
                                                └─ sensor-only policy / safety layer
                                                    └─ 24-track qualification
                                                        └─ trajectory audit + video + failure archive
```

### Asset provenance and export dependency

The vehicle descriptions in `OSRACER/` are export artifacts produced with [`osrbot/solidworks_urdf_exporter_pro`](https://github.com/osrbot/solidworks_urdf_exporter_pro), a maintained SolidWorks-to-URDF workflow with ROS, OpenUSD, and MuJoCo targets. It is an **external dependency and provenance source**, not vendored code in this repository. Geometry, inertia, joint semantics, collision choices, and target-simulator verification remain independent research responsibilities.

`OSRACER/`, `OSRACER_ISAAC_DIR`, and `osracer-*` are the canonical asset, runtime, and package identifiers. The public project name and reviewed presentation media are **OSRACER**.

## Minimal reproduction

Run from the repository root. Python 3.11+ is required; Python 3.12 is the development environment. Native Isaac Sim must be installed separately.

```bash
bash scripts/setup_racing.sh --test
. .venv/bin/activate

# Model, sensor, and actuator smoke run.
python3 scripts/run_racing.py \
  --engine mujoco --track bahrain --seconds 10 --episodes 1 --tag smoke

# Deterministic contract tests.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q
```

```bash
export OSRACER_ISAAC_DIR=/path/to/isaac-sim-6.0.1
bash scripts/run_isaac.sh scripts/run_racing.py \
  --engine isaac --track bahrain --seconds 10 --episodes 1 --tag smoke_isaac
```

Large raw experiment archives are deliberately excluded from the public repository. The static site packages a reviewed subset of screenshots, videos, and auditable summaries. Read [the public-release boundary](docs/PROJECT_STRUCTURE.md#公开仓库边界) before treating a local `output/` directory as published evidence.

## Repository guide

| Path | Research role |
| --- | --- |
| `OSRACER/` | Exported ROS, OpenUSD, and MuJoCo vehicle descriptions; provenance artifact. |
| `racing/` | Native environments, sensor contract, controller, safety layer, qualification, and auditing. |
| `tracks/` | 24 unique circuit assets, source records, and scaling assumptions. |
| `scripts/` | Environment checks, batch execution, recording, and deterministic public-media rebranding. |
| `docs/` | Engineering contract, validation ledger, deployment limits, and release notes. |
| `site/` | Dependency-free evidence-site builder and curated public assets. |
| `experimental/` | Rejected candidates and failure investigations; never mixed into accepted results. |

## Reading results responsibly

- A decoded video, a passing unit test, or a finite physics step is not automatically a driving success.
- A simulator result is not automatically a real-vehicle result.
- A higher peak speed is not automatically a more robust policy.
- A failure record is part of the result and must remain in policy or simulator comparisons.

Definitions of valid laps, overtakes, drift, contacts, and independent audits are in [Engineering](docs/ENGINEERING.md). Exact positive and negative evidence is in [Validation status](docs/VALIDATION_STATUS.md). Real-vehicle applicability and the rear-overdrive A/B are in [Deployment](docs/DEPLOYMENT.md).

## Open research directions

1. Robust perception and conservative planning under beam dropout and latency.
2. Isaac-specific bridge contact and initial vehicle-contact failures.
3. A real-world observation contract with steering feedback rather than command echo.
4. A physically realizable drift mechanism—or an explicitly non-drift racing objective.

## Citation, license, and contribution

Until an archival release/DOI is minted, cite the repository URL, commit SHA, and relevant validation document instead of an unversioned metric. Contributions should include reproduction commands, runtime versions, source revision, and both positive and negative evidence.

The repository's original source code is available under the [MIT License](LICENSE). Vehicle assets, track assets, and other third-party material retain their own notices and source terms; do not assume that the MIT license relicenses them.
