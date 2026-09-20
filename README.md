# NEORACER 双引擎竞速实验

以现有 NEORACER 车辆资产为基础，在 Isaac Sim / PhysX 与 MuJoCo 中运行双车竞速。工程提供参数化传感器反馈控制器，以及可选的 **CEM 参数训练**。`baseline_v5_policy.json` 是明确标记的未训练基线；CEM 产出的检查点另行标记，两者不混作训练成果。当前尚未实现神经网络策略，也尚未证明跨赛道稳定完圈、有效超车和漂移全部达标。目前已验收的最新批次是 g00_c03（v10c）：双引擎各跑 24 条赛道并**全部录像**，MuJoCo 24 条全部有效、Isaac 22 条有效，48 份轨迹与 48 段录像全部通过独立审计；另有巡航 9.0 m/s 速度榜：MuJoCo 24/24 有效、峰值 8.97 m/s、24 条单圈全部更快，Isaac 20/24 有效、峰值 9.00–9.04 m/s、20 条可比单圈全部更快；另有 5 条赛道 × 留出种子 10–14 的 24/25 评估；失败场次作为对照组保留，详见 [验证状态](docs/VALIDATION_STATUS.md)。

赛道目录覆盖 **2024、2025 两个完整赛季的 48 个分站记录、24 条赛道**。几何来自带 MIT 许可的公开 GeoJSON，使用同一来源快照，未验证逐年的历史布局。默认 RC 路径缩放为 0.05 倍；Baku、Monaco、Miami/Montreal 分别使用 0.100、0.125、0.075 倍，避免邻近路段合并，路宽均独立设为 1.5 m。另附路径原尺度、假定路宽 12 m 的版本。每条赛道包含 DAE、OBJ、独立纹理 PNG、碰撞网格、地图和来源记录。

**冻结交付版**（配置、双引擎成绩、素材与复现命令）见 [RELEASE.md](docs/RELEASE.md)；输出目录已整理成便于查看的结构，入口是 [output/racing/README.md](output/racing/README.md)（策略文件索引 + 日志分组 + 可点播素材库）；面向实车的部署分析与参考节点见 [DEPLOYMENT.md](docs/DEPLOYMENT.md) 与 deploy/osracer_policy/；逐项证据与未完成项见 [验证状态](docs/VALIDATION_STATUS.md)，可直接打开其中的原生视频和交互结果报告。

停车脱困（仅用轮速、转角与 15 Hz laser 的有界倒车）、安全许可条件、以及本轮双引擎成绩与速度变体，见 [停车脱困与安全许可](docs/RECOVERY.md)。

## 快速运行

在完整工程根目录运行，保留 `NEORACER/` 和 `tracks/`。建议使用 Python 3.12 虚拟环境；Isaac 使用自身的 Python，需单独安装原生 Isaac Sim。录像还需要 `ffmpeg`、可用的图形驱动和渲染环境。

```bash
bash scripts/setup_racing.sh --test
. .venv/bin/activate

# MuJoCo：先运行一个短回合，检查模型、传感器与执行器。
python3 scripts/run_racing.py --engine mujoco --track bahrain --seconds 10 --episodes 1 --tag smoke

# 固定未训练基线，在双引擎批量验收；重复同命令可断点续跑。
python3 -m racing.qualify --checkpoint output/racing/baseline_v5_policy.json --engines isaac mujoco --workers 1 --tag baseline_v5

# 可选：在 Isaac 执行 CEM 训练，再在目标引擎评估训练产物。
bash scripts/run_isaac.sh scripts/run_racing.py --engine isaac --track bahrain --train --generations 4 --population 12 --seconds 120 --episodes 3 --tag source

# 在 MuJoCo 中直接评估相同参数并录制首回合。
python3 scripts/run_racing.py --engine mujoco --track bahrain --checkpoint output/racing/training/source/policy.json --seconds 120 --episodes 3 --record --tag transfer
```

Isaac 启动器默认查找本机的 `isaac-sim-standalone-6.0.1-linux-x86_64`。其他位置通过 `NEORACER_ISAAC_DIR` 指定；启动器还包含本机 NVIDIA Vulkan/NCCL 路径配置，换机器时需对应检查。以上命令用于复现流程，不代表预先保证验收通过。

安装脚本只创建工程内 `.venv` 并安装可编辑 Python 依赖，不下载 Isaac 或修改全局 Python。只需 MuJoCo 时使用 `bash scripts/setup_racing.sh --test --mujoco-only`。单独检查现有环境：`.venv/bin/python -I scripts/check_racing_environment.py --json --output output/racing/environment_check.json`。

结果写入 `output/racing/`。训练检查点、历史和最佳轨迹按 `training/<tag>/` 隔离；评估 JSON、完整轨迹和视频按引擎、赛道、标签和种子保存。不同实验使用不同标签，重复相同标签会更新对应文件。

`racing.qualify` 要求检查点记录的策略版本和源码 SHA-256 与当前代码精确一致；同标签下，只有检查点、赛道、源码哈希、明确种子列表和实验条件相符的结果可复用，随后重新审计。更换策略、时间上限或实验条件时使用新标签。批量状态持续写入 `<tag>_manifest.json`，进程完成不等于驾驶通过。

## 文档与验证

- [工程设计、观测契约与复现说明](docs/ENGINEERING.md)
- [赛道来源、缩放、布局与拓扑限制](docs/TRACK_SOURCES.md)
- [原高速发卡弯实验与已有结果](scripts/README_hairpin_fast.md)
- [直播 / 答辩材料（可编辑 PPTX、讲稿、媒体清单）](docs/LIVE_BRIEFING.md)
- [项目结构与公开仓库边界](docs/PROJECT_STRUCTURE.md)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q
python3 -m racing.build_tracks
```

生成离线交互报告和 Markdown 结果汇总（仅读取已有结果，不启动仿真）：

```bash
python3 -m racing.report --tag baseline_v5 --tag baseline_v5_source --checkpoint output/racing/baseline_v5_policy.json
```

打开 `output/racing/index.html` 可排序、筛选回合并浏览两季赛道地图；`output/racing/RESULTS.md` 保存静态汇总。重复 `--tag` 或使用 `--tags baseline_v5 baseline_v5_source` 可合并不同标签的源端代表赛道与目标端全扫；不存在的结果不会补造。报告独立读取回合 JSON，manifest 仅补充队列状态，并区分未训练基线、CEM 训练策略和参数先验迁移。实验仍在写入时，重跑报告命令更新快照，无需重启仿真。

Suzuka RC 已增加真实碰撞桥面、坡道及分层定位接口：桥面 0.90 m、净空 0.82 m、最大坡度 6.75%。这些是明确的训练假设；其原 `full_scale/` 仍是未分层的平面参考。详见 [桥梁说明](tracks/suzuka/BRIDGE.md)。两引擎的通桥测试夹具检查物理几何，不计作竞速策略成绩。传感器噪声实验使用 `--lidar-noise`、`--lidar-dropout`、`--lidar-latency`，示例见工程文档。

固定新种子可使用 `--seeds 81 82 83 84 85`；该选项覆盖默认评估种子安排。闭环漂移检查点还记录控制器类型、激光相对运动估计、保护器配置和对应源码哈希，参见 [共享策略包](docs/ENGINEERING.md#共享策略包与冻结验收)。

## 双引擎联合训练

以下命令从当前源码生成明确未训练的闭环先验，随后在同一个原生进程中用 Isaac Sim 6.0.1 和 MuJoCo 3.10.0 联合评价 CEM 候选。先验生成器拒绝覆盖已有文件；每次改动控制器源码后，使用新的先验文件和实验标签。

```bash
.venv/bin/python -I scripts/create_racing_prior.py --output output/racing/joint_prior.json
bash scripts/run_isaac.sh scripts/train_racing_joint.py \
  --checkpoint output/racing/joint_prior.json --tag joint_training \
  --seeds 0 5 --seconds 30 --population 6 --generations 2

# 用未参与该次搜索的种子验证完整圈，并录制首个种子。
.venv/bin/python -I scripts/qualify_racing.py \
  --checkpoint output/racing/training/joint_training/policy.json \
  --tag joint_full_lap --tracks bahrain --seeds 10 11 12 13 14 --seconds 120
```

30 秒训练段只用于搜索，最终资格单独检查整圈、有效超车、连续漂移和碰撞。训练保存源码快照、实际运行时版本、每个候选的完整轨迹及独立审计；增加 `--resume` 可在相同源码和条件下继续已中断的搜索。运行期间保持源码不变；旧候选应在其冻结工程中重现，不能只改哈希来冒充同一策略。

`scripts/qualify_racing.py` 明确导入脚本所在工程的源码，适合多个冻结工程共用 `.venv` 的情况。直接使用 `python -I -m racing.qualify` 会遵循虚拟环境中可编辑安装指向的工程，可能并非当前目录。

默认对手速度上限为 2.8 m/s、发车领先 3 m；这一条件下的超车成绩不能推定为对高速对手的成绩。资格入口也支持 `--opponent-speed 5 --opponent-gap 3 --start-s 0`，用于独立验证更快对手或其他发车位置，并将这些条件纳入断点复用校验。
