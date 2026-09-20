# OSRACER：面向自主阿克曼赛车的证据优先研究

[English](README.md) · [证据站点](https://osrbot.github.io/osracer_rl/) · [验证台账](docs/VALIDATION_STATUS.md) · [项目结构](docs/PROJECT_STRUCTURE.md)

OSRACER 是一个研究型开源项目，研究阿克曼转向车辆在原生 **MuJoCo** 与 **Isaac Sim / PhysX** 中的高速自主竞速。项目将车辆资产验证、受传感器约束的驾驶、跨赛道测评、轨迹审计、原生视频和失败对照组整合为同一可复现流程。

仿真输出被视为需要资格审查的证据，而不是宣传结论。

## 研究问题

1. 导出的车辆资产能否在 OpenUSD/Isaac Sim 与 MuJoCo 中可靠加载、渲染、运动学驱动和物理步进？
2. 在不使用全局 actor 位姿、仅使用轮速、转角和 15 Hz 激光历史的条件下，阿克曼小车能否在明确物理假设下完成整圈、主动超车并改善圈速？
3. 极限来自仿真器、传感器契约，还是实车的执行器架构？

## 已审查的研究快照

本仓库仍在研究中，**不**宣称已解决全部驾驶目标。[验证台账](docs/VALIDATION_STATUS.md) 同时保留支持性和反例性证据。

| 问题 | 当前证据支持的结论 | 仍然存在的边界 |
| --- | --- | --- |
| OpenUSD 与 MuJoCo 资产 | OpenUSD 已有加载/渲染/短时步进证据：20 个网格引用、10 个刚体、9 个物理关节、6 个可动自由度；MuJoCo `robot.xml`/`scene.xml` 已完成 500 步有限状态检查。 | 这是**资产层**检查，不等于固定基座导出模型已被证明能够地面驾驶。 |
| 完圈与超车 | v10c 在 MuJoCo 上为 24/24 有效圈，每圈一次审计通过的超车；Isaac 在匹配资格条件下为 22/24。 | Isaac 仍在 Spa 出现起步车车接触、Suzuka 出现桥面失稳。 |
| 最短圈速 / 提速 | 9.0 m/s 巡航配置下，MuJoCo 为 24/24 有效圈、峰值 8.97 m/s；Isaac 为 20/24 可比有效圈、峰值 9.00–9.04 m/s。 | Isaac 高速配置以稳定性换取速度，不能当作共享鲁棒工作点。 |
| 180° 漂移 | 在写明执行器假设的仿真中存在持续侧滑示例。 | 当前单电机四驱实车**不能**复现：移除后轮超速后最大侧滑角由 27.0° 降至 4.36°；项目不作实车漂移声明。 |
| 传感器鲁棒性 | 仅加噪声时可通过部分检查。 | 0.02 m 噪声 + 5% 丢束 + 50 ms 延迟下，10/10 扰动回合失败；这是公开的研究缺口。 |

## 方法

```text
SolidWorks 装配体
    └─ solidworks_urdf_exporter_pro ──> ROS 描述 + OpenUSD + MuJoCo MJCF
                                            └─ 原生资产验证
                                                └─ 仅传感器策略 / 安全层
                                                    └─ 24 赛道资格评估
                                                        └─ 轨迹审计 + 视频 + 失败归档
```

### 资产来源与导出依赖

`OSRACER/` 中的车辆描述是通过 [`osrbot/solidworks_urdf_exporter_pro`](https://github.com/osrbot/solidworks_urdf_exporter_pro) 导出的资产。该工具维护从 SolidWorks 到 URDF 的工作流，并可输出 ROS、OpenUSD 与 MuJoCo 目标。它是本项目的**外部依赖和资产来源**，而不是复制进本仓库的代码；几何、惯量、关节语义、碰撞选择以及目标仿真器验证均是独立研究责任。

目录 `OSRACER/`、环境变量 `OSRACER_ISAAC_DIR` 与 `osracer-*` 是规范的资产、运行时和包标识。公开项目名称及审核后的演示媒体统一为 **OSRACER**。

## 最小复现实验

请在仓库根目录执行。需要 Python 3.11+（开发环境为 Python 3.12）；原生 Isaac Sim 需单独安装。

```bash
bash scripts/setup_racing.sh --test
. .venv/bin/activate

# 模型、传感器和执行器路径的短回合。
python3 scripts/run_racing.py \
  --engine mujoco --track bahrain --seconds 10 --episodes 1 --tag smoke

# 确定性契约测试。
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q
```

```bash
export OSRACER_ISAAC_DIR=/path/to/isaac-sim-6.0.1
bash scripts/run_isaac.sh scripts/run_racing.py \
  --engine isaac --track bahrain --seconds 10 --episodes 1 --tag smoke_isaac
```

大型原始实验档案被有意排除在公开仓库之外。静态站只发布经审核的截图、视频、讲解材料和可审计摘要；将本地 `output/` 视为公开证据前，请先阅读[公开仓库边界](docs/PROJECT_STRUCTURE.md#公开仓库边界)。

## 仓库导览

| 路径 | 在研究记录中的职责 |
| --- | --- |
| `OSRACER/` | 导出的 ROS、OpenUSD 与 MuJoCo 车辆描述；资产来源工件。 |
| `racing/` | 原生环境、传感器契约、控制器、安全层、资格评估与审计。 |
| `tracks/` | 24 条唯一赛道、来源记录和缩放假设。 |
| `scripts/` | 环境检查、批处理、录像与确定性的公开媒体品牌重制。 |
| `docs/` | 工程契约、验证台账、部署边界与发布说明。 |
| `site/` | 无外部依赖的证据站构建器与精选公开资产。 |
| `experimental/` | 被拒收候选和失败调查，绝不混入通过成绩。 |

## 负责任地阅读结果

- 视频可解码、单元测试通过或物理状态有限，不自动等于驾驶成功。
- 仿真结果不自动等于实车结果。
- 峰值速度更高，不自动等于策略更鲁棒。
- 失败记录也是结果的一部分；比较策略或仿真器时不得删除。

有效圈、超车、漂移、接触与独立审计的定义见[工程说明](docs/ENGINEERING.md)。完整正负证据见[验证状态](docs/VALIDATION_STATUS.md)。实车可迁移性及后轮超速 A/B 结论见[实车部署](docs/DEPLOYMENT.md)。

## 开放研究方向

1. 面对激光丢束与延迟的鲁棒感知和保守规划；
2. Isaac 专属的桥面接触与起步车车接触问题；
3. 使用实际转向反馈而非指令回显的实车观测契约；
4. 可物理实现的漂移机制，或明确不追求漂移的竞速目标。

## 引用、许可证与贡献

在归档版本 / DOI 发布前，请引用仓库 URL、commit SHA 与相应验证文档，避免引用未版本化的单一指标。Issue 或 Pull Request 应提供复现命令、运行时版本、源码版本，以及正负两类证据。

仓库中的原创源码以 [MIT License](LICENSE) 发布。车辆资产、赛道资产和其他第三方材料保留各自的来源说明和许可条款；MIT 许可证并不会重新许可这些材料。
