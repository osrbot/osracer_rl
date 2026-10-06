# OSRACER · Sensor-driven Ackermann racing

[中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Results](docs/VALIDATION_STATUS.md) · [Presentations](publication/README.md)

OSRACER studies autonomous Ackermann racing in MuJoCo and Isaac Sim / PhysX. Train a policy, inspect its TensorBoard metrics, then run it in the native simulator to record a lap. The repository includes vehicle assets, 24 tracks, a PPO neural policy, CEM baselines, and tools to check trajectories and videos.

The policy reads wheel speeds, steering angles, and a single-plane laser scan. It does not receive global vehicle position. PPO can complete MuJoCo Bahrain laps, but the latest 24-track transfer campaign qualified only **6/24** tracks across all selection seeds. The historical 24/24 MuJoCo result belongs to the CEM controller. Their checkpoints, protocols, and results are kept separate in the [validation ledger](docs/VALIDATION_STATUS.md).

## Prerequisites

The commands below target **Linux x86_64 with Bash**. Development and validation use Ubuntu 24.04 and Python 3.12. The package declares Python ≥3.11; other OS and Python combinations have not received equivalent project validation. You can start with MuJoCo alone. Isaac requires a separate installation.

| Workflow | Required environment |
| --- | --- |
| MuJoCo physics and CPU training | Python 3.11+, `venv`, `pip`, and Git. No NVIDIA GPU required. Set `MUJOCO_GL=disable` for runs without rendering and `device=cpu` for CPU training |
| PPO training and export | PyTorch ≥2.4, ONNX ≥1.17; TensorBoard ≥2.16 and tensorboardX ≥2.6 for event logging. The setup script installs these dependencies |
| MuJoCo video recording | A working OpenGL backend. EGL is the default for offscreen rendering; OSMesa provides an alternative software renderer |
| Video encoding and playback | `ffmpeg` and `ffprobe`; opening a video also needs `ffplay` and a desktop display. The project script does not install OS packages |
| CUDA policy training | An NVIDIA driver compatible with the chosen PyTorch wheel and `torch.cuda.is_available()` returning `True`. MuJoCo physics still runs on the CPU |
| Isaac simulation | **Isaac Sim 6.0.1 Linux standalone**, a supported NVIDIA RTX GPU/driver, and Vulkan. Use Isaac's bundled Python 3.12 |
| Storage and network | Access to GitHub and Python package sources. Isaac may download extensions and assets at first startup. Allow additional space for Python packages, models, and growing `runs/` directories |

NVIDIA's Isaac 6.0 requirements list a four-core CPU, 32 GB RAM, 50 GB SSD storage, and RTX 4080 / 16 GB VRAM as baseline specifications. The listed Linux test driver is 580.95.05. These are vendor specifications, not measured OSRACER minima; training and recording need additional headroom. GPUs without RT Cores, including A100/H100, are unsupported by Isaac. Run the vendor Compatibility Checker and check the [version-specific requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html) before installing (checked 2026-10-06).

The committed vehicle and track assets are ready to use. SolidWorks, ROS 2, and a compiled `native/` extension are not prerequisites for the simulation workflow. ROS 2 is for [vehicle deployment](docs/DEPLOYMENT.md); the SolidWorks exporter is needed when regenerating vehicle assets.

## Start with MuJoCo

### 1. Install system packages and clone the repository

Reference commands for Ubuntu 24.04, with permission to install OS packages:

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip \
  ffmpeg libegl1 libgl1 libglfw3 libosmesa6

git clone https://github.com/osrbot/osracer_rl.git
cd osracer_rl
```

Keep the checkout and its `assets/` directory. The runtime resolves vehicle and track paths relative to the source tree; installing the wheel alone does not supply these assets.

### 2. Select a renderer and install Python dependencies

For a machine with working GPU/EGL drivers:

```bash
export MUJOCO_GL=egl
RACING_PYTHON=python3.12 bash tools/environment/setup_racing.sh --mujoco-only --test
source .venv/bin/activate
```

For CPU simulation or training without rendering, use the following commands. Omit `--test` on this path because that option includes actual rendering tests:

```bash
export MUJOCO_GL=disable
RACING_PYTHON=python3.12 bash tools/environment/setup_racing.sh --mujoco-only
source .venv/bin/activate
```

For software rendering, use `export MUJOCO_GL=osmesa`, keep `libosmesa6` installed, and verify a rendered run before recording a full lap. See [MuJoCo's rendering backend documentation](https://mujoco.readthedocs.io/en/3.3.7/programming.html).

The script creates `.venv`, runs `pip install -e '.[build,training,ppo]'`, and checks dependencies and assets. `--test` runs two basic contract test groups. It does not install graphics drivers, configure CUDA, or install Isaac. Set `MUJOCO_GL` again when opening a new terminal.

[pyproject.toml](pyproject.toml) defines the dependencies: **MuJoCo 3.10.0** is pinned; NumPy, SciPy, and Pillow form the base set. The `build` extra adds Shapely, trimesh, pycollada, and pytest. The `training,ppo` extras add the training and logging packages listed above. Other packages use minimum versions; there is no lockfile freezing the entire environment.

To select a particular CPU or CUDA PyTorch build, create `.venv` first and install PyTorch there using the [official installer selector](https://pytorch.org/get-started/locally/), then run the project setup script. This workflow uses a prebuilt wheel; it does not require building PyTorch with a separately installed CUDA toolkit. Follow the selected wheel's driver compatibility requirements.

### 3. Verify dependencies and a short episode

```bash
python -m pip check
python -c "import mujoco, torch, onnx, tensorboard, tensorboardX; from onnx.reference import ReferenceEvaluator; print('MuJoCo', mujoco.__version__, 'PyTorch', torch.__version__, 'ONNX', onnx.__version__); print('CUDA available:', torch.cuda.is_available())"
python tools/environment/check_racing_environment.py --mujoco-only

# Exercise the vehicle, sensors, and physics without recording.
MUJOCO_GL=disable osracer --simulator mujoco --task bahrain \
  --seconds 10 --episodes 1 --tag env-smoke
```

`pip check` checks installed dependency relationships. The explicit imports confirm that training packages are available in the selected interpreter. The environment checker's PASS does not yet cover PyTorch, ONNX, TensorBoard, CUDA computation, or actual rendering. Keep this import check: the project's compatibility fallback to an existing Isaac Python package directory is not a substitute for installing a complete standalone environment.

A ten-second episode verifies execution, not lap completion. To check GPU rendering and video encoding separately:

```bash
MUJOCO_GL=egl osracer --simulator mujoco --task bahrain \
  --seconds 3 --episodes 1 --record --tag render-smoke
```

Use `MUJOCO_GL=osmesa` for software rendering. Setup reports go to `runs/_environment/reports/`; episode outputs go into their own `runs/<run-id>/` directories. The full suite includes rendering tests: select a working EGL or OSMesa backend before running `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`.

## Train, inspect, and play

After setup, the normal workflow is three commands:

```bash
osracer-train +simulator=mujoco +task=racing/bahrain experiment_name=demo
tensorboard --logdir runs/demo/metrics/tensorboard --port 6006
osracer-play experiment_name=demo
```

Start TensorBoard in another terminal with `.venv` activated and open `http://localhost:6006`. Append `device=cpu` when training without CUDA. Append `+train=quick` to reduce the PPO budget for a pipeline check. It still includes teacher warm-up and does not establish convergence.

Playback loads `policy.pt`, runs and records an evaluation episode, then opens the MP4 on a desktop. On a server, use `osracer-play experiment_name=demo --no-open` and download the video. **`--no-open` still requires rendering**; it only skips the video player. Do not use `MUJOCO_GL=disable` for recording.

Each experiment has a separate directory. Weights are saved as `checkpoints/policy.pt`, with `policy.onnx` exported at training completion; metrics live in `metrics/` and recordings in `videos/`. The saved model is the best evaluated checkpoint and may still be the warm-start model. TensorBoard includes training candidates, so its final plotted value need not describe the selected checkpoint. Before warm-up finishes, no events have been written and TensorBoard may show “No dashboards are active.” See [the usage guide](docs/RUN_ARTIFACTS.md) for configuration, artifacts, and troubleshooting.

## Run Isaac

Install Isaac Sim 6.0.1 separately and pass its compatibility checks, then set the installation path. Its bundled Python and the project `.venv` are separate environments; packages installed in one do not automatically appear in the other.

```bash
export OSRACER_ISAAC_DIR=/path/to/isaac-sim-6.0.1
nvidia-smi
cat "$OSRACER_ISAAC_DIR/VERSION"
test -f "$OSRACER_ISAAC_DIR/python.sh"
test -f "$OSRACER_ISAAC_DIR/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib/libnccl.so.2"
test -f /etc/vulkan/icd.d/nvidia_icd.json

bash tools/runtime/run_isaac.sh -c \
  "import sys, numpy, scipy, PIL, torch, onnx, tensorboard, tensorboardX; from onnx.reference import ReferenceEvaluator; print(sys.version); print(torch.__version__, onnx.__version__)"

bash tools/runtime/run_isaac.sh -m racing.runtime.run \
  --simulator isaac --task bahrain --seconds 10 --episodes 1 --tag isaac-smoke

bash tools/runtime/run_isaac.sh -m racing.runtime.train \
  +simulator=isaac +task=racing/bahrain experiment_name=isaac-demo
bash tools/runtime/run_isaac.sh -m racing.runtime.play experiment_name=isaac-demo --no-open
```

The NCCL and Vulkan ICD paths above are requirements of the current launcher. If either is absent, check the Isaac version and driver installation layout. Renaming an installation directory does not make its version compatible, and the regular `.venv/bin/python` cannot replace the Isaac launcher.

If an import is missing, install that package with `bash "$OSRACER_ISAAC_DIR/python.sh" -m pip install <missing-package>`. Preserve Isaac's bundled torch/numpy combination instead of installing the full project extras over it. The launcher already adds project sources to `PYTHONPATH`. Joint runs using both engines also require `mujoco==3.10.0` installed and verified inside Isaac Python. See the [Isaac Python environment documentation](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html).

## Train across 24 tracks

Once single-track training has produced `runs/demo/checkpoints/policy.pt`, use it as the shared parent for all tracks:

```bash
osracer-benchmark train benchmark_name=ppo-2025 season=2025 \
  checkpoint=runs/demo/checkpoints/policy.pt +simulator=mujoco +train=benchmark
osracer-benchmark status benchmark_name=ppo-2025
```

The command saves progress at track boundaries and writes TensorBoard events plus JSON/CSV summaries. Finishing a training job does not qualify its policy; unsuccessful policies and failure records are retained. See [the benchmark protocol](docs/BENCHMARKS.md).

| Policy / experiment | Recorded result | Scope |
| --- | --- | --- |
| PPO track transfer | 6/24 MuJoCo tracks pass seeds 0–2 | Development selection set; reliable driving across all tracks remains unsolved |
| PPO Bahrain with corner guard | 6/6 valid laps on seeds 0–5; mean lap time 62.92 s | The retained model is iteration 0; the improvement comes from an inference guard, not demonstrated reward learning |
| Historical CEM v10c | MuJoCo 24/24; Isaac 22/24 | Frozen configuration and nominal sensors; Isaac still fails at Spa and Suzuka |
| Historical combined sensor perturbations | 10/10 failures with noise, dropout, and latency combined | Robustness is not established by nominal-condition results |

The real vehicle uses one motor for four-wheel drive and cannot directly execute the independent rear-wheel speed boost used in the simulation drift examples. [Deployment notes](docs/DEPLOYMENT.md) describe the actuator comparison and its limits.

## Repository map

| Directory | Contents |
| --- | --- |
| `src/racing/` | Control, perception, simulators, training, evaluation, and configuration |
| `assets/` | Vehicle descriptions, 24 tracks, and provenance |
| `tools/`, `tests/` | Launchers, checks, diagnostics, historical reproduction, and tests |
| `docs/` | Usage, protocols, results, and [references](docs/REFERENCES.md) |
| `publication/` | Shared reviewed media, website, slides, and speaker notes |
| `deployment/`, `native/` | Optional ROS 2 deployment and native contact experiments |
| `runs/` | Per-run models, metrics, trajectories, and videos; excluded from Git |

The Python package uses a `src/` layout and a shared simulator factory. Configuration commands take inspiration from ASAP; see [the backend guide](docs/SIMULATORS.md) for the implementation. CEM remains an explicit `algorithm=cem` baseline and supplies the teacher controller for PPO. Retired entry points and archived data are covered in [migration notes](docs/LEGACY.md).

Vehicle assets were exported with [SolidWorks URDF Exporter Pro](https://github.com/osrbot/solidworks_urdf_exporter_pro). Track sources and geometric assumptions are documented [here](docs/TRACK_SOURCES.md). Original code uses the [MIT License](LICENSE); third-party assets retain their own notices. When reporting a problem, include the commit, runtime versions, command, and run logs. When citing a result, also identify its policy, engine, tracks, and seeds.
