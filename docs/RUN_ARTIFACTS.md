# 运行产物目录

新训练、评估和录像统一写入 `runs/<run-id>/`。`output/racing/` 是旧版只读证据库，不再作为程序默认输出位置；为保留历史路径与哈希，不原地搬动旧数据。历史索引与转换见 [LEGACY.md](LEGACY.md)。

## 安装与最小运行

在仓库根目录使用 Python 3.11+：

```bash
bash tools/environment/setup_racing.sh --mujoco-only --test
. .venv/bin/activate
```

安装脚本建立 `.venv`，安装项目和 PPO/ONNX/TensorBoard 依赖，并将环境报告写入 `runs/_environment/reports/`。Isaac 须单独安装；不要在普通系统 Python 中直接启动其后端。

## 产物布局

```text
runs/<run-id>/
├── manifest.json
├── config/                 # 本次命令与训练契约
├── checkpoints/            # 模型权重、策略检查点
├── metrics/
│   ├── metrics.jsonl       # 可审计的逐步标量
│   ├── tensorboard/        # TensorBoard event 文件
│   ├── training/           # 候选、代际和训练历史
│   └── evaluation/         # 按引擎保存的回合汇总
├── trajectories/<engine>/<track>/
├── videos/<engine>/<track>/
├── logs/
├── reports/
└── source/                 # 必要的冻结源码快照
```

训练入口采用与 ASAP/HumanoidVerse 类似的配置组合语法。命令行选择 simulator、task 和 experiment，其余训练规模由 `src/racing/config/training/default.toml` 管理：

```bash
osracer-train +simulator=mujoco +task=racing/bahrain \
  experiment_name=ppo-bahrain-01

# 只看这次训练；训练过程中即可刷新查看
tensorboard --logdir runs/ppo-bahrain-01/metrics/tensorboard --port 6006

# 自动读取 checkpoints/policy.pt，录制一个评估回合并打开视频
osracer-play experiment_name=ppo-bahrain-01
```

`osracer-play` 会从 run 内的训练配置恢复 simulator、task、回合时长和对手设置。未提供 `experiment_name` 时，训练使用时间、仿真器和赛道自动生成 run id。快速环境检查可追加 `+train=quick`；只在实验确实需要时才覆盖底层字段：

```bash
# 缩小 PPO 阶段：2 iterations、每次 256 steps、30 s episode
osracer-train +simulator=mujoco +task=racing/bahrain +train=quick

# 高级覆盖；命名和层级与保存的配置一致
osracer-train +simulator=mujoco +task=racing/bahrain experiment_name=large-run \
  train.iterations=1500 train.steps_per_iteration=4096 env.episode_length_s=180
```

`quick` 仍包含教师预热和多轮闭环采集；`imitation_steps=4096` 不是含全部 DAgger 轮次的总步数上限。它用于检查链路，不能作为收敛验收。只测试训练管线时可额外设 `train.imitation_steps=0 train.imitation_epochs=0`，此时不要期待未训练模型完圈。

## 策略与检查点

默认算法是 PPO。策略网络使用 370 维固定输入：4 个轮速、2 个转角、361 束激光距离、激光年龄和上一归一化动作。无效激光束直接以 `-1` 编码，因此不再额外复制 361 个有效位，同时保留窄障碍簇和连续间隙的原始角分辨率。上一动作是部署包装器维护的两维控制状态，用来消除 15 Hz 扫描、60 Hz 控制下的部分可观测歧义；确定性动作以 `1/32` 步长量化，使 CPU、CUDA 与 ONNX 的浮点尾差不再改变控制序列。`pace` 配置还会把基于轮速和转向的高速急转门控写入同一模型图，因此 `.pt`、ONNX 和播放端执行相同行为。轻量一维卷积编码器提取相邻射线几何，环境真值只用于奖励和终止判定。训练先用已验证的传感器控制器做带动作扰动的 DAgger 预热，再由 PPO 更新 actor-critic；默认用 seed 0–2 选择检查点，`pace` 使用 seed 0–5，先最大化有效圈数，再改善最差进度，全部完成后才按全程均速和回报提速。可用 `env.evaluation_episodes=N` 调整筛选种子数。每个 DAgger 轮次为这些种子分别采集闭环轨迹，模仿标签与 PPO 采样动作都使用部署时的量化网格。

默认完整配置还会执行两轮锚定闭环蒸馏（`train.refinement_cycles=2`、`train.refinement_epochs=8`）：每轮在当前最佳策略实际访问的轨迹（含失败回合）上重新标注教师动作，并用原策略动作约束更新；每个 epoch 都以多种子闭环结果决定是否保存。快速 smoke 配置默认关闭该阶段。训练结束自动导出和校验 `checkpoints/policy.onnx`。ONNX 输入为动态批量 `observation: [batch, 370]`，输出为 `action: [batch, 2]`；调用 ONNX 时由宿主把上一次输出回填到输入最后两维。ONNX 输出是归一化动作，不是直接的轮速或转向关节命令。宿主应复用 `racing.control.neural_policy.encode_observation` 与 `action_to_actuators`，并使用检查点内的 `max_speed_m_s`；新回合上一动作重置为 `[-1, 0]`。也可重新导出：

```bash
osracer-export --experiment-name ppo-bahrain-01
```

常用配置名为 `simulator`、`task`、`experiment_name`、`train.*` 和 `env.*`。`--engine`、`--track`、`--run-id` 和 `--seconds` 作为脚本兼容入口继续有效。原结构化策略参数搜索保留为显式基线：在训练命令中添加 `algorithm=cem`，其检查点仍为 `policy.json`，并使用 `train.generations` 和 `train.population`。

`checkpoints/policy.pt` 保存闭环筛选结果最好的模型，可能仍是 iteration 0 预热模型，并非最后一次迭代。`checkpoint=PATH` 是初始化新实验，优化器与训练计数重新创建，不是精确续训；请使用新的 `experiment_name`。

## 训练日志与 TensorBoard

训练每次迭代都会刷新 `metrics/metrics.jsonl`、`metrics/training/history.json` 和 TensorBoard event；训练中断前已完成的迭代仍可见。`tools/environment/setup_racing.sh` 默认安装 PPO、ONNX 和 TensorBoard 依赖。使用 `--tensorboard-dir PATH` 可把 event 写入指定的持久目录，随后直接执行 `tensorboard --logdir PATH`；使用 `--no-tensorboard` 可仅保留 JSONL。

所有仿真器使用同一套 PPO 训练终端摘要。每次 learning iteration 显示采样与学习耗时、吞吐、value-function loss、surrogate loss、entropy、动作噪声、回合奖励和长度、奖励分量、终止率、累计步数与 ETA。完整历史保存在运行目录，不再把大型 JSON 倾倒到默认终端。显式 `algorithm=cem` 时则显示与其算法相符的 generation/candidate、score、圈速和安全指标。

`osracer-play` 在有桌面显示环境时默认使用 `ffplay` 打开刚生成的 MP4；服务器或 CI 环境可加 `--no-open`。Isaac 使用同一个配置接口，但需通过 Isaac Python 启动：

```bash
bash tools/runtime/run_isaac.sh -m racing.runtime.train \
  +simulator=isaac +task=racing/bahrain experiment_name=isaac-demo
bash tools/runtime/run_isaac.sh -m racing.runtime.play experiment_name=isaac-demo
```

训练命令结束时会打印 `OSRACER_ARTIFACTS`，明确列出 run、`.pt`、ONNX、JSONL 和 TensorBoard 目录；播放命令会打印 `PLAYBACK_VIDEO`。已经存在训练历史的 run id 会被拒绝，避免把两次训练曲线意外追加到同一目录。

双引擎联合训练继续使用 `osracer-train-joint`（由 Isaac Python 启动），并采用相同命名：`osracer-train-joint checkpoint=<path> experiment_name=joint-cem`。它使用相同的 checkpoint、JSONL、TensorBoard 和 `OSRACER_ARTIFACTS` 接口；`--resume` 会在 TensorBoard 中清理并重写本次 run 的步骤，避免重复 step 叠加。

### 显示 “No dashboards are active” 时

1. 确认 `--logdir` 指向本次运行的 `metrics/tensorboard/` 或包含该目录的父目录，不能只指向 JSONL、权重或视频目录。
2. 当前 PPO 在教师预热和 warm-start 评估结束后才创建事件写入器。预热期间没有 event 文件属当前实现边界；第一轮 PPO 指标写入后刷新页面。
3. 用下列命令确认是否已有事件和 scalar 标签：

```bash
tensorboard --inspect --logdir runs/ppo-bahrain-01/metrics/tensorboard
```

4. 如果没有 scalar，检查终端是否仍在预热、运行是否提前失败，或是否使用了 `--no-tensorboard` / `logger.tensorboard=false`。缺依赖时重新运行安装命令；有事件而页面无数据时检查选中的 run 和 TensorBoard 进程的日志目录。

事件按迭代刷新；突然终止时正在进行的采集段尚未写入。图中 evaluation 包含训练候选（包括后来被拒绝的模型），不能用曲线末值代替最终 `.pt` 中的筛选结果。特别是保留 iteration 0 模型时，以 checkpoint 的 `metadata.metrics.evaluation` 为准。

旧 CEM JSON 转事件和历史归档浏览见 [LEGACY.md](LEGACY.md)。
