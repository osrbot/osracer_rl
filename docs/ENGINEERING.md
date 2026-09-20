# 工程说明与验收边界

本工程将现有发卡弯实验扩展为双车竞速管线：同一策略通过统一接口驱动 Isaac Sim / PhysX 和 MuJoCo 的原生车辆动力学，训练与评估保留可检查的参数、轨迹和视频。赛道资产和程序已经具备运行入口；竞速表现以本次实际输出为准，不沿用旧实验通过率。

## 目录与职责

```text
NEORACER/                原始 USD、MuJoCo 与 CAD 车辆资产
racing/
  tracks.py              赛道查询、弧长投影与边界接口
  build_tracks.py        离线重建赛道和互换格式资产
  sensors.py             激光扫描、采样保持、策略观测隔离
  policy.py              CEM 可调参数反馈策略
  odometry.py            相邻激光帧的局部速度估计，不累计全局位姿
  policy_closed_loop.py  侧滑反馈、后轮增速与恢复状态机
  safety.py              基于车体包络的局部制动和低速恢复
  safety_motion.py       扫描补偿与车体局部运动预测
  policy_bundle.py       同一检查点的控制器装配与源码身份
  isaac_env.py            Isaac 多车场景、执行器、接触和渲染
  mujoco_env.py           MuJoCo 多车场景、执行器、接触和渲染
  metrics.py              独立竞速指标统计
  run.py                 训练、评估、录像与输出
  train_joint.py         两个原生引擎共同评价的 CEM 搜索
  qualify.py             冻结检查点的双引擎断点续跑与审计
  verify.py              从完整轨迹和视频重新计算并核验结果
  report.py              已有证据与队列状态的离线报告
scripts/run_racing.py    统一命令入口
scripts/qualify_racing.py 固定导入当前工程的资格入口
scripts/create_racing_prior.py 生成明确未训练的闭环先验
scripts/run_racing_batch.py 单个 Isaac 应用中的多赛道队列
scripts/rebind_actor_source.py 把已有检查点参数重绑到当前控制器源码并记录迁移来源
scripts/probe_policy_variants.py 在不重训的前提下比较显式参数变体
scripts/run_isaac.sh     本机原生 Isaac 启动环境
scripts/setup_racing.sh  创建工程内虚拟环境并检查依赖
scripts/check_racing_environment.py  只读环境检查与 JSON 清单
tracks/                 赛季索引、24 条赛道、来源与许可证
tests/                  观测、几何、指标及引擎契约测试
experimental/bridge_contact_adapter/ 独立原生接触适配器与几何验证
output/racing/          本次竞速实验输出
output/hairpin_fast/    已有高速发卡弯实验输出
docs/RECOVERY.md        停车脱困契约、安全许可、双引擎实测与失败分类
```

## 策略与观测契约

`RacingPolicy` 是包含间隙选择、转向、制动和有限后轮增速的结构化控制器，CEM 可搜索其参数。它没有神经网络权重、反向传播或视觉编码器。`baseline_v5_policy.json` 的算法字段为 `untrained structured sensor-feedback baseline`，表示未训练基线；只有标明 CEM 及训练引擎的产物才称为训练策略。策略可复用发卡弯实验的执行器控制思路，但发卡弯参数文件不能直接当作竞速检查点。

| 输入/时钟 | 当前约定 |
| --- | --- |
| `wheel_vel` | 四轮编码器角速度，长度 4；观测端统一前进符号 |
| `steer_pos` | 左、右前轮转向关节角，长度 2，rad |
| `lidar` | 361 条射线、270° 视场、0.04–15 m 范围，含角度、距离、有效标记、时间戳与扫描年龄 |
| 扫描坐标系 | 真实挂载的 `laser` 坐标系，使用引擎计算的位姿 |
| 更新频率 | 激光 15 Hz、控制 60 Hz、物理积分 480 Hz |

扫描在两个更新时刻间保持上一帧，策略只在新扫描到来时重新规划；每个控制步仍限制轮速与转向变化。输出为四轮速度目标和两个前轮转向角目标。右后轮执行器前进方向为负，统一适配器负责符号处理。转向限位 ±0.45 rad 是实验设定，未标定为真实机械止挡。

策略观测只包含上述编码器与激光数据。世界位置、航向、真实车速、侧滑角、对手位置、赛道投影和奖励可供环境、射线生成与评估使用，不送入策略。视频上的真实速度和侧滑角属于评估叠加信息。

唯一的负轮速路径是有界倒车脱困：车辆已经停住且规划器自己要求停车（或安全监督器持续拒绝指令）时，策略可以倒车 0.8 s，速度 −0.7 m/s，前轮回中位，随后进入 0.8 s 低速复位。安全层按后方对角射线可见性与墙体扫描预测放行，拒绝时输出零轮速。触发条件、许可条件和实测结果见 [停车脱困与安全许可](RECOVERY.md)。

激光保留原资产的 `base_link` 到 `laser` 固定变换，包括原有俯仰。当前测距是共享解析射线模型，覆盖有限高度边界、地面、分层赛道的桥面/桥底和车辆新增的实体反射盒；它没有扫描全部 CAD 表面，也不是引擎原生 RTX 激光。两个引擎都加入相同尺寸的可见反射盒，使低矮车辆能被实际挂载平面的射线观测到。这是明确的仿真车辆改动，部署到真车前需要对应实体或重新验证可观测性。默认扫描无噪声；可配置测距噪声、丢束和交付延迟，规格仍属于实验假设。

## 场景与赛道

默认回合运行两辆独立受控 NEORACER，后车从前车后方 3 m 开始。对手运行同类传感器策略，并使用 `--opponent-speed` 指定的巡航速度上限；它不是可保证完美避障或完圈的外部教师。

重置后车辆由执行器与原生接触动力学运动。普通赛道使用平面地面接触；分层赛道使用同一份三维桥面和坡道三角形。Isaac 使用静态三角面碰撞，MuJoCo 将每个高架三角形向下挤出为薄凸棱柱，保持桥下空间开放。导出的 DAE/OBJ 与独立纹理是互换资产，原生运行器没有直接导入这些 DAE 纹理场景。

赛季索引只证明分站、日期与赛道名称的对应关系。所有赛道使用同一近似几何快照，未核验季节专属布局。RC 中心线默认缩放 0.05 倍；Baku 使用 0.100、Monaco 使用 0.125、Miami/Montreal 使用 0.075。路宽固定为 1.5 m，属于适应多车训练的非等比衍生物。`full_scale/` 保持路径原尺度，12 m 路宽为假设。两种版本均无测绘高程、倾角、维修区或真实路肩。

Suzuka RC 的 `rc-r2-bridge` 修订增加了 0.90 m 桥面、0.08 m 厚度、0.82 m 净空和两端各 20 m 缓坡；最大坡度 6.75%，高度为训练假设。`Track.has_elevation` 触发三维处理，`at3d(s)` 返回 `(xyz, yaw, grade)`，`road_triangles_3d()` 返回 K×3×3，`boundary_segments_3d()` 返回 M×2×3；`project` 通过高度与历史弧长保持上下层连续。其 `full_scale/` 仍是原平面参考，不支持桥梁竞速，详见 [桥梁 schema](../tracks/suzuka/BRIDGE.md)。

Miami、Monaco、Montreal、Baku 的初版存在邻近路段连通；修订版以固定宽度增大中心线比例，四条均形成一个外边界和一个内边界。旧版归档在 `output/racing/track_revision1/`，旧哈希结果不能作为新几何验收。全局最近中心线投影仍需结合进度检查。来源许可、哈希和所有限制见 [TRACK_SOURCES.md](TRACK_SOURCES.md)。

## 训练与复现

从工程根目录执行；Python 依赖与可编辑安装见 [README](../README.md)。原生 Isaac 单独安装，使用其启动器运行，普通 Python 环境不会自动安装 Isaac。无显示器的 MuJoCo 默认选择 EGL。录像需 `ffmpeg`，编码使用 H.264；视频字体依赖本机 DejaVu Sans。

`bash scripts/setup_racing.sh --test` 创建或复用工程内 `.venv`，安装 `.[build]`，检查现有双引擎环境并运行观测契约和 MuJoCo 原生测试；它不安装 Isaac、驱动或系统软件。`--mujoco-only` 可只要求 MuJoCo，`RACING_PYTHON` 可选择 Python ≥3.11。只读检查使用 `.venv/bin/python -I scripts/check_racing_environment.py --json --output output/racing/environment_check.json`；完整赛道测试仍可另行运行文末的 `pytest tests`。

```bash
# 源引擎参数搜索；每代 12 个候选，每个候选最多 120 秒仿真。
bash scripts/run_isaac.sh scripts/run_racing.py --engine isaac --track bahrain --train --generations 4 --population 12 --seconds 120 --episodes 3 --tag train_eval

# 固定检查点，重新录制源引擎。
bash scripts/run_isaac.sh scripts/run_racing.py --engine isaac --track bahrain --checkpoint output/racing/training/train_eval/policy.json --seconds 120 --episodes 3 --record --tag source_replay

# 目标引擎直接加载同一参数。
python3 scripts/run_racing.py --engine mujoco --track bahrain --checkpoint output/racing/training/train_eval/policy.json --seconds 120 --episodes 3 --record --tag transfer
```

| 参数 | 含义 |
| --- | --- |
| `--train` | 在选定引擎中执行 CEM；迁移实验应仅在源端使用 |
| `--generations`、`--population` | 代数和每代候选数；建议 population 至少 4 |
| `--seconds` | 每回合仿真时间上限，失败或完圈可提前结束 |
| `--episodes` | 训练后或直接评估时的回合数 |
| `--checkpoint` | 读取竞速 JSON 参数；与 train 同用时作为搜索初值 |
| `--record` | 录制评估的第一个回合，30 fps，正常播放速度 |
| `--tag` | 评估输出标签，同时隔离 `training/<tag>/` 训练目录 |
| `--seed` | 搜索随机种子及默认评估安排的基数；默认首回合为 0 |
| `--seeds` | 明确指定评估种子列表，覆盖默认安排；训练候选种子不受影响 |
| `--start-s` | 评估起点沿中心线的弧长，单位 m |
| `--opponent-speed` | 对手巡航速度上限，单位 m/s |
| `--opponent-gap` | 起步时对手领先的弧长距离，默认 3 m |
| `--lidar-noise` | 命中距离的高斯噪声标准差，单位 m，默认 0 |
| `--lidar-dropout` | 独立射线丢失概率，默认 0 |
| `--lidar-latency` | 扫描交付延迟，单位 s，向上取整到控制步，默认 0 |
| `--drift-curriculum` | 先搜索后轮增速与触发参数的侧滑课程，仍需独立运动指标验收 |

训练候选当前使用固定回合种子 0，属于单赛道参数搜索。使用更多评估回合不等于完成训练域随机化或多赛道泛化验证。训练结束后的评估采用当前最佳候选；没有 `--checkpoint` 且不训练时采用内置初值。

批量验收与噪声实验：

```bash
# 同一份未训练基线，双引擎、逐项运行、默认遍历全部赛道。
python3 -m racing.qualify --checkpoint output/racing/baseline_v5_policy.json --engines isaac mujoco --workers 1 --tag baseline_v5

# 单赛道测距压力实验；使用新标签隔离默认无噪声结果。
python3 scripts/run_racing.py --engine mujoco --track bahrain --checkpoint output/racing/baseline_v5_policy.json --seconds 120 --episodes 3 --lidar-noise 0.01 --lidar-dropout 0.01 --lidar-latency 0.03 --tag baseline_v5_noise

# 读取已有结果及队列状态，不启动仿真。
python3 -m racing.report --tag baseline_v5 --tag baseline_v5_source --checkpoint output/racing/baseline_v5_policy.json
```

`racing.qualify` 启动时要求检查点的 `policy_version` 和 `policy_source_sha256` 与当前策略完全一致，再冻结检查点文件 SHA。重跑时，只有种子列表、时间上限、默认对手条件、噪声/延迟和录像条件，以及检查点、赛道和运行源码哈希匹配的已有结果可以复用，之后重新执行独立审计；输入在执行中改变会记录 `frozen_inputs_changed`。它不把修改过的策略代码默认为“相同检查点直接迁移”。批量入口也支持三个激光噪声参数。改变实验条件仍建议使用新标签，以保留原结果。

队列状态写入 `output/racing/<tag>_manifest.json`，每个引擎/赛道的日志位于 `output/racing/logs/<tag>/`。`evaluated` 表示子任务已执行并进入审计，不能当作完圈通过率。报告用重复 `--tag` 或 `--tags` 列表合并源端代表赛道与目标端全扫，逐行保留标签。回合 JSON 是结果来源，某个标签没有 manifest 也能纳入；manifest 只补充队列状态。实验进行时可重新运行报告命令生成新快照，不在文档中固定最终成功数。两引擎桥梁测试夹具的通桥结果仅验证物理几何，不计入控制策略的赛道覆盖或完圈成绩。CEM 训练结束后，应把训练检查点保存到独立路径，用新的标签重复同一验收协议。

## 指标与证据

完整有效圈要求完成正向累计路程，并满足碰撞、越界和状态稳定性检查。圈速只对有效圈记录，不能用失败前的高速片段替代。超车统计要求从后方转为领先并保持一段时间，独立指标还检查双方安全状态；应以 `metric_audit` 和原始轨迹审计结果为准，视频叠加计数仅用于观察。

训练输出为 `output/racing/training/<tag>/policy.json`、`training_history.json` 和最佳轨迹。每个引擎目录保存 `<track>_<tag>.json`、带种子的轨迹与可选 MP4。显式加载检查点的评估记录文件 SHA-256；视频保存帧数、时长、哈希和解码检查。录像每两个 60 Hz 控制步写一帧，按 30 fps 编码。

同标签的同名文件会更新；不同标签的训练目录相互隔离。比较实验前先归档输出，冻结检查点，在两个引擎使用相同赛道、参数、回合条件和指标，保留失败回合。当前管线的数据记录不自动构成独立验收结论。

已有高速发卡弯实验使用真实位姿、速度和侧滑等特权信息：保存结果中持续侧滑验收为 Isaac 10/10、MuJoCo 9/10。它是动力学与控制基线，不能据此声称当前仅传感器竞速策略已经达到同等水平。竞速的最终结果应另外报告，不在这里填入尚未完成的通过率。

## 后续验收目标

先验证固定检查点在两个引擎的有效完圈、无碰撞超车和多种种子稳定性，再扩展到其他赛道。漂移需根据车体运动和后轴侧滑独立计量，不能把后轮空转指令直接记为成功漂移。之后再加入测距噪声、延迟、轮胎和执行器变化，验证真实车辆驱动能力与反射目标配置；若需要神经策略，应另行实现并沿用同一观测边界和评估协议。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q
```

测试通过说明对应契约成立；训练成功、历史地图准确和真实部署可靠分别需要各自证据。

## 共享策略包与冻结验收

检查点使用 `actor_type=reactive` 或 `closed_loop` 指定控制器；`controls` 保存漂移目标、增速脉冲时长、转向偏置与估计置信度阈值，`safety_supervisor` 决定是否启用车体包络保护。所有开关都在两个引擎共用，对手保持固定的原始反馈策略。闭环控制通过连续两帧 `laser` 扫描估计相对横向速度和偏航速率，始终不积分世界坐标、不接收评估真值。

`actor_source_sha256` 包含策略装配、基础策略以及启用的闭环、里程估计和保护模块；`runtime_source_sha256` 进一步记录传感器、指标、赛道和双引擎适配器。`racing.qualify` 在开始和每个任务结束核验冻结身份，独立审计检查检查点、轨迹与指标的一致性。只有同一份参数和同一套控制源码的结果才可称为共享策略验证。联合两引擎调参属于联合标定，不应称为从未使用目标引擎的零样本迁移。

启用安全层时，身份还包含其 `safety_motion.py` 和 `odometry.py` 依赖，即使主控制器是 reactive 类型也一样。安全层版本随检查点明确记录；更新保护器后需要生成新的适配检查点并重新验收。

```bash
# 在已冻结的闭环检查点上运行新的留出种子；实际路径替换为该次训练产物。
python3 -m racing.qualify --checkpoint output/racing/training/train_eval/policy.json --tag heldout --tracks bahrain monaco --seeds 81 82 83 84 85 --workers 1
```

漂移资格基于真实车体运动独立计算：速度大于 1.2 m/s、侧滑绝对值严格大于 20°，连续至少 0.1 s。另行保留最长连续时长、事件数、累计时长和动作阶段；制动产生的侧滑不能归因为后驱增速。验收应同时检查无碰撞完圈和有效超车，短暂峰值或录像观感不代替数值证据。

## 双引擎联合 CEM

`racing/train_joint.py` 用同一份 actor 在两个原生引擎上逐个评价明确的发展种子。搜索六项变量：巡航速度、后轮增速比、转向触发阈值、增速时长、反打增益和偏航速率保护阈值。碰撞、失稳、停滞优先淘汰；其次要求每个种子取得至少 2.5 m/s 的正向净进度和有效超车，再优化最差种子的连续漂移。该训练使用短时课程，产出的候选仍须独立通过整圈和留出赛道验收。

```bash
.venv/bin/python -I scripts/create_racing_prior.py --output output/racing/joint_cem_initial.json
bash scripts/run_isaac.sh scripts/train_racing_joint.py --checkpoint output/racing/joint_cem_initial.json --tag joint_cem --seeds 0 4 5 --seconds 30 --population 6 --generations 2
# 中断后在完全相同条件与源码下恢复；已完成候选会重验轨迹并重算评分。
bash scripts/run_isaac.sh scripts/train_racing_joint.py --checkpoint output/racing/joint_cem_initial.json --tag joint_cem --seeds 0 4 5 --seconds 30 --population 6 --generations 2 --resume
```

每个候选保留两个引擎的完整轨迹、实际评价的种子和引擎、SHA-256、评分与提前淘汰原因。训练条件或源码改变必须使用新标签；缺少契约、缺失轨迹、哈希变化或评价覆盖不符不能复用。`training/<tag>/runtime/` 保存源码快照。先从与 Isaac Python 小版本一致的工程 `.venv` 预加载固定的 MuJoCo 3.10.0，再启动 SimulationApp，避免 Newton 扩展内置的 3.8.0 被误用于目标端训练；不修改全局 Python。

## 多赛道原生批处理

`run_racing_batch.py` 复用 Isaac 应用与渲染器，每条赛道都重新创建 World、物理场景和两辆车。它已用两条赛道连续录像验证；不复用车辆或上一圈状态。

```bash
bash scripts/run_isaac.sh scripts/run_racing_batch.py --tracks bahrain austin --record-tracks bahrain --checkpoint output/racing/baseline_v5_policy.json --tag source_batch --seconds 160
```

批处理执行清单只表示命令完成，驾驶成绩仍以逐回合 JSON 和独立审计为准。评估入口在每个完整回合后写入结果，因此后续回合中断不会丢失已完成证据。共享显卡上应根据实际剩余显存调度；不同原生进程同时冷启动渲染器可能造成显存不足。

可传入 `--stop-file PAUSE_SOURCE_BATCH`。需要释放显卡时创建这个文件，程序会完成当前赛道后停止，并打印下一条尚未开始的赛道；恢复时移除停止文件，仅传入剩余赛道。正在运行的一圈无需被打断。
