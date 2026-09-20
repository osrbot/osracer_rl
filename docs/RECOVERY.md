# 停车脱困与安全许可

这一页描述 2026-09-16 加入的有界倒车脱困，以及安全层对它的许可条件。它只使用轮速、转向关节角和 15 Hz 单线 laser 扫描，不读取世界位置、航向、赛道投影或对手真值。

## 修复的两个死锁

原策略只有前进档，并且在前方扫描进入紧急停车距离时把目标速度写成 0：

```python
stop_speed = np.sqrt(2*brake*max(forward-.5-speed/15, 0.))
if forward < .24:
    target = 0.
```

由此产生两类真实反例。

第一类是**车头贴墙停死**。车辆以浅角度蹭到墙后，前向射线仍可能长于车身包络到墙的距离，于是规划器持续输出 0，车辆既不能前进也不能后退，直到 4 秒无运动被记为 `stalled`。已记录的 Isaac 反例包括 Austin、Melbourne、Monza、Singapore、Yas Marina，MuJoCo 反例是 Austin（碰撞前先在 16.9 m 处失去前进能力）。

第二类是**安全层拒绝但策略不知情**。安全监督器因为移动车辆的反射盒包络持续输出零轮速，而车辆规划器本身给出正的目标速度，因此策略的停车触发条件不成立，车停在原地等待，直到超时。已记录的反例是 MuJoCo Monaco：`near_term_vehicle_footprint` 持续生效，`blocked_seconds` 增长但车辆没有脱困动作。

## 倒车脱困契约

`racing/policy.py` 增加一个有界状态机，触发条件是**已经停住**并且**规划器自己要求停车**（或前方距离小于 0.30 m）：

```python
stuck = (fresh and abs(measured_forward) < ESCAPE_SPEED_EPS
         and (target <= 0. or self.forward_clearance < ESCAPE_CLEARANCE))
```

满足 0.35 s 后进入 0.8 s 倒车，速度 −0.7 m/s，**前轮回到中位**而不是保持入弯转角：保持转角会让倒车中的车身快速旋转，实测曾把车转到逆向行驶并累计 −109 m 进度。脱困后进入 1.2 s 冷却，避免连续倒车。

同一条脱困也可以由外部请求触发（`RacingPolicy.request_reverse_escape`）。`racing/policy_bundle.py` 中，当安全监督器连续 0.6 s 拒绝指令且实测轮速接近 0 时，会发出该请求。这是纯传感器证据：监督器的拒绝本身说明车辆被阻挡。

倒车动作是唯一会输出负轮速的路径，因此也是唯一的观测契约扩展；标称行驶仍然是前进四轮速度加两个前轮转角。

## 安全层许可条件

`racing/safety.py` 不会因为策略请求倒车就放行。许可需要三条同时成立：

1. **后方可见**：固定扫描是 270°，车身正后方 90° 是盲区。只有 |angle| ≥ 120° 的射线可用，其最小距离必须 ≥ 0.60 m，否则拒绝。
2. **墙体几何不恶化**：以物理墙拟合为判据做倒车扫描预测。若当前包络与墙已有重叠，则要求倒车结束时间距增加 ≥ 0.02 m 且中途不恶化；若当前有正常间距，则要求整段倒车轨迹都不低于 0.03 m 余量。
3. **车辆包络不恶化**：另一辆车的反射盒包络只要求不显著变差（≥ 当前值 − 0.05 m）。并排超车时倒车无法改善侧向间距，若要求改善就会永久拒绝，重新制造第二类死锁。

每次许可都写入 `actor_feedback.safety`：`commanded_reverse`、`reversing`、`reverse_rear_clearance_m`、`reverse_present_clearance_m`、`reverse_best_gap_m`、`blocked_seconds`。独立审计读取同一份轨迹，因此“脱困成功”和“安全拒绝”都可以被复核。

## 已知边界

这些是控制启发式，不是碰撞保证。后方 90° 盲区意味着倒车可能撞上看不见的物体；本次只允许 0.6 m/s、最多约 0.56 m 的倒车，并要求后方对角射线可观测。安全层使用解析墙拟合而不是引擎原生射线，拟合在相邻扫描间可能重新分段。

## 实测结果

用同一份 g00_c03 参数重绑到当前源码（v10a：原始回波近场 + 计划减速制动 + 有界倒车脱困 + 窄道车辆绕行），在 24 条赛道上双引擎全扫（每回合 200 s 上限、种子 0、对手 2.8 m/s），**每条赛道都录制原生录像**：

| 引擎 | 有效整圈 | 有效超车 | 单圈范围 | 圈均速度 | 峰值速度 | 录像 / 审计 |
| --- | --- | --- | --- | --- | --- | --- |
| MuJoCo | 24/24 | 每条 1 次 | 38.9–95.1 s | 5.18 m/s（有效圈均值） | 7.23 m/s | 24 段 / 24 通过 |
| Isaac | 22/24 | 每条 1 次 | 41.2–99.7 s | 4.4–5.6 m/s | 7.29 m/s | 24 段 / 24 通过 |
| 合计 | 46/48 | 46 次 | — | — | — | 48 段 / 48 通过 |

剩余 2 条失败全部在 Isaac：Spa 是起步车车接触（3.6 s、12.4 m），Suzuka 是桥面段 `unstable`（211.6 m，约 73% 圈长）。失败场次的录像与轨迹同样保留，可作为对照组。

### 被排除的错误假设

Spa 的停滞一开始被归因于“被拒绝的倒车仍消耗脱困额度”。修复该缺陷后（`refund_reverse_escape`，并补上以扫描推算车身速度的阻塞检测）Spa 仍然停滞，原始轨迹给出了真正的原因：两台车速度都接近 0（对手 0.02–1.6 m/s）、车辆包络重叠 0.21 m、后方对角可见距离只有 0.48 m。这是窄道双车互堵，车辆既不能前进（车辆包络不允许）也不能后退（后方太近），需要的是主动让位/超车机动。

同期被撤回的实验：把紧凑回波的膨胀半径从 0.30 m 收到 0.22 m 确实能给 1.5 m 走廊腾出侧向空间，但会破坏既有契约测试 `test_compact_reflector_envelope_retains_rear_side_clearance`（不得在超越后回切到对手身侧），因此没有采纳；把后方可见下限放宽到 0.45 m、倒车行程缩短到 0.36 m 只改变了 Spa 的轨迹，没有解决僵局，同样撤回。

脱困与近场通道确实解决了它们要解决的问题：Austin、Bahrain、Monza、Barcelona、Melbourne、Singapore、Yas Marina 等原先在紧急停车带里停死的赛道恢复完圈，MuJoCo Monaco、Silverstone 与 Isaac 的停滞类失败清零。

失败没有全部消失，而且性质变了。Isaac 的 8 条失败里有 7 条是撞墙，发生时 `reversing=false`：车辆倒车脱离后重新前进，撞进同一面墙；碰撞瞬间安全层刚把 `reason` 从 `clear` 改成 `predicted_wall_footprint`。MuJoCo Monaco 的原始轨迹最能说明问题：

```text
t=14.70  requested_clearance=-0.082  must_brake=true   predicted_wall_footprint
t=14.75  requested_clearance= 0.890  must_brake=false  clear
t=14.90  requested_clearance= 0.859  must_brake=false  clear
t=14.95  requested_clearance=-0.000  must_brake=true   predicted_wall_footprint
t=15.00  collision=true
```

相邻扫描之间墙拟合从 4 段变成 3 段，间距在 0.05 s 内从 +0.89 m 跳到 −0.00 m。这是已记录的侧滑扫描补偿缺陷，不是倒车引入的。曾试验给制动加 0.2 s 锁存来掩盖跳变，Bahrain 单圈从 53 s 涨到 79 s，因此撤回；正确做法是让近场危险判定本身不依赖可重新分段的拟合墙体。

速度方面，7.2 m/s 的上限来自 `cruise_m_s` 参数。把它改成 9.0 m/s（`--set cruise_m_s=9.0`）后，MuJoCo 24 赛道为 21/24 有效，峰值 8.97 m/s，20 条可比赛道单圈全部更快（0.7%–21%），代价是 Austin 出现一次停滞、Monaco 与 Suzuka 撞墙。Isaac 尚未在该设置下复验。

## 近场原始回波通道

上表的撞墙反例说明：只要近场危险判定完全依赖解析墙拟合，拟合一旦在相邻扫描间重新分段，监督器就会把正在逼近的墙体报成“clear”。因此 v15 增加一条独立的**原始回波通道**：

```python
near_mask = valid & (np.linalg.norm(points, axis=1) < NEAR_POINT_MAX_RANGE)
near_danger = near_clearance < self.margin
danger = wall_danger or compact_danger or near_danger
```

回波点按运动假设做同样的补偿，再与包络盒求有符号距离，因此它不会被拟合选择丢掉。只用手掌大小的一段时域（0.2 s）和 3.5 m 量程：如果把整段制动距离都压到原始点上，1.5 m 宽的 RC 赛道每个弯都会被外侧墙判成阻挡。

同一批赛道上的实测对比（MuJoCo，种子 0，200 s 上限）：

| 配置 | Bahrain | Monaco | Silverstone |
| --- | --- | --- | --- |
| 仅墙拟合（v8c） | 52.97 s 有效 | 44 m 撞墙 | 39 m 撞墙 |
| 原始点全时域 0.4 s | 97.30 s 有效 | 90.87 s 有效 | 102.62 s 有效 |
| 原始点 0.2 s 时域 | 57.58 s 有效 | 99.68 s 有效 | 55.97 s 有效 |

全时域版本把 Bahrain 拖慢 84%，因此只保留 0.2 s 时域版本：两个原本撞墙的赛道都变成有效整圈，正常赛道代价约 9%。

`racing/safety_motion.py` 的 `predict_poses` 同时扩展到支持负 `command_speed`，否则倒车许可永远无法在“当前已经贴墙”的情形下通过：预测姿态仍按观测到的（接近零）运动积分，倒车结束间距不可能改善。

## 制动动作

危险制动原来直接把四个轮速目标写成 0。锁死车轮在 MuJoCo 里制动力最强，但在 Isaac 里会让仍在偏航的车继续旋转：Bahrain 在 t=25.5 s 以 3.6 rad/s、−57° 侧滑打转 180°，之后沿赛道逆向以 7.2 m/s 行驶，累计进度 −86.8 m（终点碰撞）。

两种替代都不成立：跟随实测速度下发轮速会形成正反馈（指令始终等于当前速度，实际不减速），MuJoCo Austin 与 Bahrain 直接撞墙；按威胁模型的 4 m/s² 下发则刹不住。当前实现是“进入制动时锁存速度，按 12 m/s² 计划曲线降速”，威胁评估仍按 4 m/s²，因此轮胎在被要求越来越慢的同时保持滚动，直到曲线归零。

```python
if self.brake_profile_speed is None:
    self.brake_profile_speed, self.brake_elapsed = max(self.speed_bound, speed), 0.
rolling = max(0., self.brake_profile_speed-BRAKE_COMMAND_DECELERATION*self.brake_elapsed)/WHEEL_RADIUS
result_wheels = np.array([rolling, rolling, rolling, -rolling])
```

## 每个赛道都留录像

双引擎刷榜默认对全部赛道录像，失败场次同样入库作为对照组。MuJoCo 用 `racing.sweep --record-tracks <ids>`，Isaac 用 `scripts/run_racing_batch.py --record-tracks <ids>`。`racing.report` 会把每段视频写进 [RESULTS.md](../output/racing/RESULTS.md)，`racing.verify` 逐段做解码校验，身份（赛道、种子、检查点哈希）与成绩一起核对。

## 速度榜

7.2 m/s 的上限来自 `cruise_m_s` 参数，不是引擎限制。在同一套 v10c 控制器上把巡航提到 9.0 m/s（`scripts/rebind_actor_source.py --set cruise_m_s=9.0`），MuJoCo 24 条赛道仍然全部有效完圈并各完成 1 次有效超车：

| 引擎 | 巡航 | 有效整圈 | 峰值速度 | 圈均速度 | 单圈范围 | 与 7.25 m/s 基准比较 |
| --- | --- | --- | --- | --- | --- | --- |
| MuJoCo | 7.25 m/s | 24/24 | 7.23 m/s | 5.18 m/s | 38.9–95.1 s | — |
| MuJoCo | 9.0 m/s | 24/24 | **8.97 m/s** | **5.40 m/s** | **37.5–85.2 s** | 24 条单圈全部更快，1.5%–10.4% |
| Isaac | 7.25 m/s | 22/24 | 7.29 m/s | 4.4–5.6 m/s | 41.2–99.7 s | — |
| Isaac | 9.0 m/s | 20/24 | **9.00–9.04 m/s** | 4.4–6.6 m/s | 40.4–90.5 s | 20 条可比单圈全部更快，0.2%–9.3% |

两个引擎的两个配置各有 24 段原生录像并全部通过独立审计（96 回合、96 份轨迹、96 段录像），录像与成绩绑定同一检查点哈希。这是目前满足“5–10 m/s”要求的可复现证据。

提速不是免费的：MuJoCo 在 9.0 m/s 仍然全赛道有效，Isaac 则从 22/24 降到 20/24，新增两条停滞（Austin、Las Vegas），原有的 Spa 接触与 Suzuka 桥面失稳依旧存在。这属于“速度—稳定性”权衡的真实代价，保留在榜上而不是隐藏。

## 被排除的两个实验（Isaac 剩余失败）

Isaac Spa 与 Suzuka 是当前仅剩的两条失败。两个定向实验都没有采纳：

- **放宽侧向排斥的方位阈值**（55°→40°，`g00c_v10d_wide_lateral`）：Spa 的轨迹逐位不变（均值速度 3.2727136801717966 与改动前完全一致），说明该时刻根本没有侧向紧凑回波触发排斥；对手多半在前方而不是侧向，属于跟车距离问题。
- **提高策略制动加速度**（`brake_accel_m_s2` 5.0→8.0，`g00c_v10e_brake8`）：Suzuka 的失败从 `unstable`（车头上仰 0.516 rad）变成 `collision`（均值 4.81 m/s），仍未完圈。更早制动改变了结果但没有解决问题，说明需要处理桥面接触几何或运动估计，而不是继续加大纵向制动。

9.0 m/s 下 Isaac 新增的两条停滞也做了定向修复尝试，同样全部撤回：把倒车许可从“后方可见 ≥0.6 m”改成“≥0.3 m 并按可见空间分配 tick 预算”、把“每次倒车需重新取得前进距离”换成“每回合累计倒车 ≤3 m”、把阻塞计时的衰减减半。改完后倒车确实执行了（Austin 与 Las Vegas 各 315 个 tick），但两条赛道仍然停滞：车辆倒出来以后又回到同一个位置，说明缺的是多点掉头/重新取位，而不是更多倒车额度。

## 留出种子证据

所有排行榜使用的是种子 0，因此对 v10c 巡航 9.0 m/s 配置补做了留出种子评估：Bahrain、Monaco、Singapore、Baku、Suzuka 各跑种子 10–14，共 25 个回合。

| 赛道 | 有效/总数 | 单圈范围 | 圈均速度 | 有效超车 |
| --- | --- | --- | --- | --- |
| Bahrain | 5/5 | 47.6–70.1 s | 3.9–5.7 m/s | 每条 1 次 |
| Monaco | 5/5 | 73.3–83.9 s | 5.0–5.6 m/s | 每条 1 次 |
| Singapore | 4/5 | 46.8–53.0 s | 4.6–5.2 m/s | 每条 1 次 |
| Baku | 5/5 | 85.0–104.5 s | 5.7–7.0 m/s | 每条 1 次 |
| Suzuka | 5/5 | 55.6–58.6 s | 4.9–5.2 m/s | 每条 1 次 |

合计 24/25（96%）有效，25 份轨迹与录像全部通过独立审计。唯一失败是 Singapore 种子 10 的起步车车接触——与种子 0 排行榜里同一类失败，说明该类失败是间歇性的，而不是某个种子的特例。注意 Suzuka 在 MuJoCo 五种子全过，它的失败只出现在 Isaac。

随后把留出种子扩展到全部 24 条赛道（种子 10）：21/24 有效，失败为 Hungaroring 撞墙、Miami 停滞、Singapore 车车接触，全部通过独立审计。两份留出集合去重后共 44 个回合、41 个有效圈（93%），覆盖 24 条赛道——这是当前“高速配置在未见种子上仍然可用”的量化证据。

Isaac 侧此前完全没有留出种子，本轮补齐三条代表赛道（种子 10–12，冻结的 7.25 m/s 配置）：

| 赛道 | 有效/总数 | 单圈 | 圈均速度 | 失败原因 |
| --- | --- | --- | --- | --- |
| Bahrain | 3/3 | 53.0–53.7 s | ~5.05 m/s | — |
| Monaco | 3/3 | 79.0–79.4 s | ~5.24 m/s | — |
| Suzuka | 0/3 | — | ~4.6 m/s | 1 次失稳 + 2 次撞墙 |

合计 6/9（67%）。这说明 Isaac 的不稳定性**集中在 Suzuka 桥面**：同一配置下 Bahrain 与 Monaco 在未见种子上全部有效，而 Suzuka 三种子全败。Suzuka 的处理方式因此可以明确为“桥面接触/运动估计专项”，不必推翻整个共享策略。

## 素材库

`scripts/index_media.py` 读取已有产物生成三个文件，不移动、不重写任何录像：

- [media.html](../output/racing/media.html)：可浏览素材库，按“当前证据 / 失败对照组 / 历史批次”分组，点开即播，未播放的录像不占用带宽。
- [MEDIA.md](../output/racing/MEDIA.md)：同一批素材的表格索引，含引擎、赛道、配置、种子、是否有效圈、单圈、超车数、最长连续漂移与时长。
- `MEDIA_INDEX.json`：机器可读清单，每段录像带字节数、SHA-256、解码时长/帧数/分辨率、对应回合指标与分组标记。

当前共 **376 段录像**（证据 91、失败对照组 6、历史批次 279），其中 905 张 PNG 作为每段录像的首帧预览；全部 376 段通过 `ffprobe` 解码校验，没有不可解码文件。

## 传感器扰动（开放缺口）

观测契约支持测距噪声、丢束与交付延迟，但此前的全部成绩都是理想扫描。用共享配置（7.25 m/s）在 5 条代表赛道 × 种子 10–11 上开启 `--lidar-noise 0.02 --lidar-dropout 0.05 --lidar-latency 0.05`，结果 **10/10 全部失败**（均速 1.2–2.7 m/s，即很早就撞）。逐项消融（Monaco 种子 10，基准为有效圈）定位了主因：

| 扰动 | 结果 |
| --- | --- |
| 仅噪声 0.02 m | 有效圈（415 m 完圈，78.5 s） |
| 仅丢束 5% | 124.5 m 处撞墙 |
| 仅延迟 50 ms | 37.1 m 处停滞 |
| 三者叠加 | 6.3 m 处撞墙（2.7 s） |

两个针对性修复都没有解决问题，按纪律撤回：把紧凑回波簇的“全部射线有效”放宽到“≥80% 有效”后，Monaco 丢束轨迹逐位不变（说明失效路径不是这条）；按超出标称 15 Hz 保持间隔的陈旧度对目标速度限速后，延迟场景从 37 m 停滞变成 107 m 撞墙，仍未完圈。结论：**当前策略在带噪扫描下不具备鲁棒性**，这是冻结共享版本前必须解决的问题，而不是可以忽略的边角。

### 丢束机制已定位（单束空洞）

一次丢束在 15 Hz 扫描里不是“少一个点”，而是“多一个 15 m 的空洞”：被丢的射线返回最大量程，而它的角半径只有 `arcsin(0.145/15)=0.55°`，小于相邻射线间距 0.75°——它只挡住自己。选道逻辑于是把这一根射线当成可行驶走廊，把车对准墙上的那个洞。Monaco 种子 10 在 5% 丢束下正是这样在 124.5 m 处撞上护栏。

对应的规则很小：**可行驶走廊必须宽于单束**（例如至少 3 根射线），只在存在更宽走廊时生效。实测该规则把“仅丢束”场景从 124.5 m 撞墙变成 184.5 s 有效完圈，并且对名义工况中性（Bahrain 50.65 s、Monaco 85.87 s 与改动前一致）。

它没有解决组合扰动，因此本轮没有合入：三扰动叠加时 10/10 仍失败，而且其中多数的轨迹与修复前逐位相同——它们在起步阶段就走另一条路径。原始轨迹显示 `reason=motion_estimate_unavailable`、`wall_fit_count=0`、`compact_return_points=0`：安全层的扫描匹配连续两帧失败，进入“盲车”模式，在 2.8 m/s、距墙 0.47 m 处保持转向并最大制动，随即撞上起步段护栏。**下一步的靶点是扫描匹配的鲁棒性**（例如在噪声+丢束下放宽匹配阈值或改用降级证据），而不是继续调转向或制动。

### 扫描匹配的噪声敏感性（本轮结论）

追到里程计内部后，失败原因不是安全层的确认门槛，而是里程计自己返回 `registration_failed`：`_points` 用 5 点局部窗口做直线性检验，窗口跨度只有约 3° 扫描；2 cm 距离噪声下该检验通过率骤降，提取到的特征点少于所需的 24 个，配准直接失败。放宽安全层门槛（残差 0.04→0.06、内点率 0.5→0.35）对该场景**轨迹逐位不变**，证实瓶颈在特征提取而不在确认门槛。

加宽到 7 点窗口并把直线性比例从 0.04 放宽到 0.08 后，结果**好坏参半**：

| 场景（组合扰动，种子 10） | 改动前 | 改动后 |
| --- | --- | --- |
| Monaco | 6.3 m 撞墙 | 93.1 m 撞墙 |
| Singapore | 12.0 m 撞墙 | 55.4 m 撞墙 |
| Bahrain | 152.4 m 撞墙 | 82.3 m 停滞 |
| 名义 Monaco（无扰动） | 85.9 s 有效 | 81.5 s 有效 |

它把三条赛道里的两条推进得更远，却让一条倒退，而且没有任何一条在扰动下完圈。因此本轮同样没有合入：一个只在目标场景里"部分更好"的改动不足以作废 155 个已验证回合。下一步的鲁棒性工作应把三件事一起做——噪声鲁棒的特征提取、降级感知的限速（按无效束比例与扫描年龄）、以及丢束的最小走廊宽度规则——然后一次性重新测量。

### 三项机制合并后：扰动下可以完圈

把三件事一起做，并把前两项限定为**只在退化条件下生效**，名义工况就保持接近逐位一致：

| 机制 | 触发条件 | 名义工况影响 |
| --- | --- | --- |
| 自适应加宽特征窗口（5→7 点，直线性 0.04→0.08） | 上一次配准失败后启用，配准成功即恢复 | 无（始终走 5 点窄窗） |
| 降级感知限速 | 扫描年龄 > 1/15 s 按陈旧度限速；无效束比例**超出本回合自身基线** 1% 以上时按超额限速 | 无（年龄不超过 1/15 s，无效束比例平稳） |
| 最小走廊宽度（≥3 束） | 存在更宽走廊时优先 | 轻微（Bahrain 逐位一致，Monaco 75.52→75.67 s） |

合并后的实测（种子 10，噪声 0.02 m + 丢束 5% + 延迟 50 ms，时限放宽到 420 s 以便慢速完圈）：

| 赛道 | 合并前 | 合并后 |
| --- | --- | --- |
| Monaco | 6.3 m 处撞墙（2.7 s） | **有效完圈 253.3 s**（415 m，圈均 1.67 m/s） |
| Singapore | 12.0 m 处撞墙 | **有效完圈 153.7 s**（247 m，圈均 1.63 m/s） |
| Bahrain | 152.4 m 处撞墙 | 112.6 m 处撞墙（仍失败） |

这是扰动工作流的实质进展：从"起步即撞"变成"降速后能干净跑完"。代价也很清楚——扰动下的单圈约为名义工况的 2–3 倍（例如 Monaco 名义 75.7 s、扰动 253.3 s），因为限速机制会在扫描陈旧或丢束偏多时主动降速。Bahrain 仍失败，说明这套机制还不完整。

## Isaac 的速度工作范围

两轮共五种定向修复都没能救回 9.0 m/s 下的 Isaac 停滞（Austin、Las Vegas），而且失败证据高度一致：

| 尝试 | 观察到的结果 |
| --- | --- |
| 侧向排斥方位阈值 55°→40° | Spa 轨迹逐位不变，说明该时刻没有侧向回波 |
| 策略制动加速度 5.0→8.0 | Suzuka 由 `unstable` 变 `collision`，仍未完圈 |
| 按可见空间分配倒车预算（后方 ≥0.3 m） | 倒车获准并执行 315 个 tick，车辆仍回到同一位置 |
| 累计倒车上限 3 m | 同上 |
| 多点掉头（倒车带转角 + 前进反向转角） | Las Vegas 执行 242 个 tick，仍然停滞 |

结论：**7.25 m/s 是本车在 Isaac/PhysX 接触模型下的稳定上限，9.0 m/s 超出该工作范围**（同样的 9.0 m/s 配置在 MuJoCo 全赛道有效）。因此共享/冻结配置应保持在 7.25 m/s，9.0 m/s 速度榜作为 MuJoCo 侧的可复现结果保留并明确标注引擎差异。这不是断言“Isaac 不能再快”，而是说在当前车辆、路宽与对手设置下，继续提高巡航需要先改接触模型或加宽赛道，而不是继续调恢复逻辑。

## 窄道车辆绕行

被车辆包络挡住和被墙挡住不是同一类失败。车辆包络是从一块 16 cm 反射面推出来的保守估计，而 1.5 m 宽的赛道里两台车可以同时压在这块包络上，谁都动不了——MuJoCo Singapore 与 Spa 的原始轨迹就是双方速度都接近 0、包络重叠 0.21 m、后方可见距离只有 0.48 m。

因此 `must_brake` 分支增加了一条只针对车辆的放行路径：墙体间距必须保持在余量以上，车辆包络只要求不显著变差，并在候选转向里选侧向开口最大的那个，以 0.5 m/s 前进。

```python
vehicle_only = bool(compact_danger and not wall_danger)
if present >= -.01 or (vehicle_only and present_walls >= self.margin):
    wall_gap = self._clearance(walls, RECOVERY_SPEED, candidate, include_compact=False)
    if wall_gap >= self.margin and gap >= present-.05:
        escapes.append((gap, wall_gap, candidate))
```

这条路径不会放宽物理边界：`wall_gap` 仍然是硬条件，倒车许可与后方可见性规则不变。诊断里的 `reason` 会写成 `slow_forward_vehicle_creep`，`vehicle_creep` 标记同帧可见，因此独立审计可以区分“绕行成功”和“仍在硬制动”。

## 近场车辆预警时域

v10a 批次里三起车车接触（MuJoCo Singapore、Isaac Shanghai、Isaac Spa）都发生在监督器首次解算到对手之后 0.08–0.5 s 内，而当时的近场车辆时域只有 0.12 s。把 `VEHICLE_HORIZON` 提高到 0.18 s 后，MuJoCo Singapore 从起步接触变为 48.37 s 有效圈，Bahrain 代价 0.45 s（50.23→50.68 s）；Isaac Shanghai 的轨迹逐位不变，说明那里是“已经在制动但仍发生侧向刮擦”，不是预警时机问题。`vehicle_horizon_s` 会写进每帧诊断，便于审计区分。

## 侧向避让

规划器的选道只使用前向扇区（`|angle| < 100°`）内的自由间隙，而并排行驶的对手回波出现在车身两侧，根本进不了这个扇区——所以侧向刮擦对转向选择是不可见的。v10c 在转向合成里加入一个有界的侧向排斥项：

```python
for cloud in compact_clouds:
    centre = np.median(cloud, axis=0)
    distance = float(np.hypot(centre[0], centre[1]))
    bearing = float(np.arctan2(centre[1], centre[0]))
    if distance < 1.2 and abs(bearing) > np.deg2rad(55):
        steer += -.12*(1.2-distance)/1.2*np.sign(bearing)
```

它只对已确认的紧凑回波生效，最大偏置 0.12 rad，远离侧以 1.2 m 为界线性衰减；墙体间隙、制动与安全层逻辑都不变。Isaac Shanghai 由此从车车接触变为 59.57 s 有效圈，Mu Bahrain 50.65 s、Mu Singapore 48.40 s、Isaac Bahrain 53.23 s 均无回归。

## 复现

```bash
# 观测契约与脱困单元测试
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q

# 把已有检查点的参数绑定到当前控制器源码，并记录迁移来源
.venv/bin/python -I scripts/rebind_actor_source.py \
  --checkpoint output/racing/experimental/joint_cem_g00c03_multitrack_workspace/candidate.json \
  --output output/racing/candidates/g00c03_v8a_escape.json

# 单赛道原生复现
.venv/bin/python -I scripts/run_racing.py --engine mujoco --track bahrain \
  --checkpoint output/racing/candidates/g00c03_v8a_escape.json --seconds 200 --tag check

# 双引擎全赛道刷榜
.venv/bin/python -I -m racing.sweep --checkpoint output/racing/candidates/g00c03_v8a_escape.json \
  --tag c03v8a_escape_dev24 --seconds 200 --seed 0
bash scripts/run_isaac.sh scripts/run_racing_batch.py --tracks <全部赛道> \
  --checkpoint output/racing/candidates/g00c03_v8a_escape.json --tag c03v8a_escape_dev24 --seconds 200
```
