# OSRACER · 传感器驱动的阿克曼赛车

[English](README.md) · [使用文档](docs/README.md) · [实验结果](docs/VALIDATION_STATUS.md) · [演示材料](publication/README.md)

OSRACER 用 MuJoCo 和 Isaac Sim / PhysX 研究阿克曼小车的自主竞速。你可以在同一套接口下训练策略，查看 TensorBoard 曲线，再把模型放回原生仿真器录制一圈。项目包含车辆资产、24 条赛道、PPO 神经策略、CEM 基线，以及轨迹和录像的核验工具。

策略读取轮速、转角和单线激光，不读取车辆的全局位置。当前 PPO 已能在 MuJoCo Bahrain 完圈，但最近的 24 赛道迁移只有 **6/24** 通过全部筛选种子。历史 CEM 在 MuJoCo 上的 24/24 成绩属于另一套控制器。两者的配置、检查点和结果分别记录，详见[验证状态](docs/VALIDATION_STATUS.md)。

## 安装前先确认环境

下面的命令面向 **Linux x86_64、Bash**；当前开发与验证环境为 Ubuntu 24.04、Python 3.12。包声明支持 Python ≥3.11，其他系统和 Python 版本组合尚未获得同等验证。先走 MuJoCo 路线即可开始，Isaac 需要额外安装。

| 你要运行的内容 | 必须准备的环境 |
| --- | --- |
| MuJoCo 物理仿真、CPU 训练 | Python 3.11+、`venv`、`pip`、Git；无需 NVIDIA GPU。无渲染时设 `MUJOCO_GL=disable`，训练设 `device=cpu` |
| PPO 训练与模型导出 | PyTorch ≥2.4、ONNX ≥1.17；训练日志需要 TensorBoard ≥2.16、tensorboardX ≥2.6。安装脚本会安装这些依赖 |
| MuJoCo 录制视频 | 可用的 OpenGL 渲染环境。默认使用 EGL 离屏渲染；无 GPU 时可配置 OSMesa 软件渲染 |
| 视频编码、检查和桌面播放 | `ffmpeg`、`ffprobe`；自动打开视频还需要 `ffplay` 和桌面显示环境。项目安装脚本不会安装系统软件 |
| CUDA 加速策略网络 | 支持所选 PyTorch wheel 的 NVIDIA 驱动，且 `torch.cuda.is_available()` 为 `True`；MuJoCo 的物理步进仍在 CPU 上运行 |
| Isaac 仿真 | 单独安装 **Isaac Sim 6.0.1 Linux standalone**、适配的 NVIDIA RTX GPU/驱动和 Vulkan，使用 Isaac 自带的 Python 3.12 |
| 磁盘与网络 | 能访问 GitHub、Python 包源；Isaac 初次启动还可能下载扩展与资产。为 Python 包、模型和视频另留空间，`runs/` 会随实验增长 |

Isaac 6.0 官方要求列出的参考下限包括 4 核 CPU、32 GB 内存、50 GB SSD，以及 RTX 4080 / 16 GB 显存；Linux 测试驱动列为 580.95.05。它们是官方环境要求，并非本项目测出的最低配置；训练和录制还需要额外余量。无 RT Core 的 A100/H100 不在 Isaac 支持范围。安装前运行官方 Compatibility Checker，并核对[对应版本要求](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)（核对于 2026-10-06）。

已提交的车辆和赛道可直接使用。运行仿真不要求安装 SolidWorks、ROS 2 或编译 `native/`。ROS 2 用于后续[实车部署](docs/DEPLOYMENT.md)，SolidWorks 导出工具用于重新制作资产。

## 从 MuJoCo 开始

### 1. 安装系统依赖并获取仓库

Ubuntu 24.04 参考命令，需要系统包安装权限：

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip \
  ffmpeg libegl1 libgl1 libglfw3 libosmesa6

git clone https://github.com/osrbot/osracer_rl.git
cd osracer_rl
```

保留完整仓库和 `assets/` 目录。当前代码按源码目录定位车辆与赛道，单独安装 wheel 不能代替仓库资产。

### 2. 选择渲染方式，再安装 Python 环境

有可用 GPU/EGL 驱动时，在当前终端执行：

```bash
export MUJOCO_GL=egl
RACING_PYTHON=python3.12 bash tools/environment/setup_racing.sh --mujoco-only --test
source .venv/bin/activate
```

没有 GPU、只想检查仿真或做 CPU 训练时，使用下面这组命令。无渲染路径不加 `--test`，因为该选项包含实际渲染测试：

```bash
export MUJOCO_GL=disable
RACING_PYTHON=python3.12 bash tools/environment/setup_racing.sh --mujoco-only
source .venv/bin/activate
```

需要 CPU 软件录制时选 `export MUJOCO_GL=osmesa`；它需要前面安装的 `libosmesa6`，应先完成一次渲染验证。[MuJoCo 的渲染后端说明](https://mujoco.readthedocs.io/en/3.3.7/programming.html)区分了 EGL、OSMesa 和窗口渲染。

脚本创建 `.venv`，执行 `pip install -e '.[build,training,ppo]'`，并检查依赖和资产。`--test` 运行两组基础契约测试。它不会安装显卡驱动或 Isaac，也不会替你配置 CUDA。运行前设置的 `MUJOCO_GL` 只作用于当前终端，新终端需要重新设置。

Python 依赖以 [pyproject.toml](pyproject.toml) 为准：MuJoCo 固定为 **3.10.0**；基础依赖为 NumPy、SciPy、Pillow；`build` 包含 Shapely、trimesh、pycollada 和 pytest；`training,ppo` 包含上表的训练与日志依赖。其余包使用版本下限，项目尚未提供锁定全部依赖的 lockfile。

如需指定 CPU 或 CUDA 版 PyTorch，先创建 `.venv` 并按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装到该环境，再运行项目安装脚本。系统 CUDA toolkit 不是这里额外编译 PyTorch 的前置步骤；以选定 wheel 的驱动兼容要求为准。

### 3. 验证依赖和短回合

```bash
python -m pip check
python -c "import mujoco, torch, onnx, tensorboard, tensorboardX; from onnx.reference import ReferenceEvaluator; print('MuJoCo', mujoco.__version__, 'PyTorch', torch.__version__, 'ONNX', onnx.__version__); print('CUDA available:', torch.cuda.is_available())"
python tools/environment/check_racing_environment.py --mujoco-only

# 不录制，只检查车辆、传感器与物理步进。
MUJOCO_GL=disable osracer --simulator mujoco --task bahrain \
  --seconds 10 --episodes 1 --tag env-smoke
```

`pip check` 检查已安装包的依赖关系。紧接着的直接导入检查用于确认训练包确实安装在当前解释器中；当前环境检查脚本的 PASS 尚不覆盖 PyTorch、ONNX、TensorBoard、CUDA 运算或实际渲染。不要省略这一步，也不要用本机已有 Isaac 包的兼容回退代替独立环境安装。

10 秒短回合只检查运行链路，不要求完成整圈。有 GPU 并准备录制时，再验证渲染和编码：

```bash
MUJOCO_GL=egl osracer --simulator mujoco --task bahrain \
  --seconds 3 --episodes 1 --record --tag render-smoke
```

CPU 软件渲染改用 `MUJOCO_GL=osmesa`。环境报告默认写入 `runs/_environment/reports/`，回合和视频写入各自的 `runs/<run-id>/`。完整测试包含渲染用例；先切换到可用的 EGL 或 OSMesa，再运行 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`。

## 训练、看曲线、播放

完成上面的安装后，日常使用只需要三个命令：

```bash
osracer-train +simulator=mujoco +task=racing/bahrain experiment_name=demo
tensorboard --logdir runs/demo/metrics/tensorboard --port 6006
osracer-play experiment_name=demo
```

在另一个已激活 `.venv` 的终端启动 TensorBoard，浏览器打开 `http://localhost:6006`。无 CUDA 时训练追加 `device=cpu`；只检查训练链路可追加 `+train=quick`。`quick` 仍包含教师预热，不是立即结束的空跑，也不证明策略已经收敛。

播放命令会加载 `policy.pt`、重新运行并录制一个回合，再在桌面中打开 MP4。服务器使用 `osracer-play experiment_name=demo --no-open`，随后下载视频查看。**`--no-open` 仍需要渲染环境**；`MUJOCO_GL=disable` 不能用于录制。

每次实验独占一个运行目录。模型位于 `checkpoints/policy.pt`，训练结束会导出 `policy.onnx`；日志在 `metrics/`，录像在 `videos/`。保存的是评估最好的检查点，可能仍是预热模型。TensorBoard 曲线记录训练候选，曲线末值不一定属于最终保留模型。预热结束前没有 event 文件时，页面可能显示 “No dashboards are active”。目录、配置覆盖和排查步骤见[使用说明](docs/RUN_ARTIFACTS.md)。

## 使用 Isaac

先单独完成 Isaac Sim 6.0.1 安装和官方兼容性检查，再设置安装目录。项目 `.venv` 与 Isaac 自带 Python 是两个环境，前面安装的依赖不会自动复制过去。

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

NCCL 库和 Vulkan ICD 的路径是当前启动器的具体要求；如果缺失，先核对 Isaac 版本及驱动安装布局。不能只把目录名改成 6.0.1，也不能用普通 `.venv/bin/python` 代替 Isaac 启动器。

缺包时，用 `bash "$OSRACER_ISAAC_DIR/python.sh" -m pip install <缺失包>` 补到 Isaac 解释器中。保留 Isaac 随附的 torch/numpy 组合，不要直接在其中执行整套项目 extras 安装。启动器已通过 `PYTHONPATH` 提供本项目源码；双引擎联合运行还须在 Isaac Python 中安装并验证 `mujoco==3.10.0`。安装方法见 [Isaac Python 环境文档](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)。

## 24 赛道训练与当前进展

单赛道训练生成 `runs/demo/checkpoints/policy.pt` 后，可以把它作为所有赛道共同的父模型：

```bash
osracer-benchmark train benchmark_name=ppo-2025 season=2025 \
  checkpoint=runs/demo/checkpoints/policy.pt +simulator=mujoco +train=benchmark
osracer-benchmark status benchmark_name=ppo-2025
```

命令按赛道保存进度，并生成 TensorBoard 目录和 JSON/CSV 汇总。一次训练完成不代表该赛道通过，程序会保留未通过策略和失败记录。协议与复现命令见 [Benchmark 文档](docs/BENCHMARKS.md)。

| 策略 / 实验 | 已记录的结果 | 结论适用范围 |
| --- | --- | --- |
| PPO 跨赛道迁移 | MuJoCo 6/24 赛道通过 seed 0–2 | 开发筛选集，尚未达到全赛道稳定驾驶 |
| PPO Bahrain + 转弯门控 | seed 0–5 为 6/6 有效圈，平均圈时 62.92 s | 保留 iteration 0 模型，改善来自推理门控；不能归因为奖励学习 |
| 历史 CEM v10c | MuJoCo 24/24，Isaac 22/24 | 冻结参数与名义传感器条件；Isaac 的 Spa、Suzuka 仍失败 |
| 历史组合传感器扰动 | 噪声、丢束、延迟叠加时 10/10 失败 | 未通过鲁棒性验收，不能从无噪声结果外推 |

实车为单电机四驱，不能直接执行仿真里的独立后轮增速。已有漂移样例的执行器假设与实车不同；[部署说明](docs/DEPLOYMENT.md)记录了对应 A/B 和限制。

## 代码与资料在哪里

| 目录 | 内容 |
| --- | --- |
| `src/racing/` | 策略、感知、仿真器、训练、评估与配置 |
| `assets/` | 车辆描述、24 条赛道及来源记录 |
| `tools/`、`tests/` | 启动、检查、诊断、历史复现与测试 |
| `docs/` | 使用说明、实验协议、结果与[参考材料](docs/REFERENCES.md) |
| `publication/` | 共享审核媒体、网站、PPT 与讲稿 |
| `deployment/`、`native/` | 可选 ROS 2 部署和原生接触实验 |
| `runs/` | 各批次模型、指标、轨迹、录像；不提交到 Git |

项目采用 `src/` 包布局，仿真器通过统一工厂创建。配置命令参考 ASAP，具体实现与差异见[仿真器文档](docs/SIMULATORS.md)。CEM 保留为 `algorithm=cem` 基线，也为 PPO 提供教师控制器。旧数据和已移除入口见[历史迁移说明](docs/LEGACY.md)。

车辆资产由 [SolidWorks URDF Exporter Pro](https://github.com/osrbot/solidworks_urdf_exporter_pro) 导出；赛道来源和近似假设见[赛道说明](docs/TRACK_SOURCES.md)。原创代码使用 [MIT License](LICENSE)，第三方资产保留各自声明。反馈问题时，请附提交号、环境版本、完整命令和对应 run 的日志；引用实验结果时，请同时给出策略版本、引擎、赛道和种子。
