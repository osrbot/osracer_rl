# OSRACER 高速进弯与 180° 发卡弯

本实验在 Isaac Sim / PhysX 中训练 12 参数反馈策略，将同一份策略参数直接用于 MuJoCo。目标是先在直道达到至少 5 m/s，再制动、产生可测量的后轴侧滑并完成 180° 发卡弯。两端均使用原生接触动力学、四轮速度伺服和前轮转向伺服；重置后的运动由执行器和接触积分产生。

新实验文件位于 `scripts/*hairpin_fast*`，结果写入 `output/hairpin_fast/`。旧低速实验及 `output/hairpin/` 保留，源 USD、XML 和 CAD 网格不因本实验而改写。

## 复现

在仓库根目录执行。需要可运行的 Isaac Sim 6.0.1、MuJoCo、NumPy、Pillow，以及 `ffmpeg` / `ffprobe`。Isaac 启动脚本默认使用 `/home/osrbot/rlgpu_ws/isaac-sim-standalone-6.0.1-linux-x86_64`；其他安装位置可通过 `OSRACER_ISAAC_DIR` 指定。

```bash
cd /home/osrbot/Desktop/osracer_work

# 只在 Isaac 中训练，并评估 10 个回合。
bash scripts/run_isaac.sh scripts/run_hairpin_fast.py --engine isaac --train --generations 6 --population 16 --eval-seed-start 300 --episodes 10

# 使用最终检查点重新评估 Isaac，并录制原生渲染视频。
bash scripts/run_isaac.sh scripts/run_hairpin_fast.py --engine isaac --eval-seed-start 300 --episodes 10 --record

# 将相同检查点直接用于 MuJoCo；不在 MuJoCo 中训练或选择参数。
python3 scripts/run_hairpin_fast.py --engine mujoco --eval-seed-start 300 --episodes 10 --record

# 独立重算轨迹指标，校验检查点、源文件与视频。
python3 scripts/verify_hairpin_fast.py

# 打包正常速度、0.25 倍速并排视频及结果报告。
python3 scripts/summarize_hairpin_fast.py
```

当前保存结果中，Isaac 的持续侧滑验收为 10/10，MuJoCo 为 9/10；两端各 10/10 完成高速接近和掉头。MuJoCo 种子 301 仅达到约 19.6° 侧滑，没有满足约 20°、0.1 秒的门槛。因此独立验证器按设计返回退出码 1，`verification.json` 保持 `passed=false`。当前并排视频通过以下命令打包，报告显式标明未全部通过；该选项只允许评估回合侧滑不足，其他驾驶、录制、视频和检查点一致性检查必须全部通过：

```bash
python3 scripts/summarize_hairpin_fast.py --allow-partial-slip
```

完整执行将更新新实验目录中的训练记录、策略、评估和视频。CEM 以固定随机种子采样，每代 16 个候选、共 6 代，优化 12 个参数：巡航速度、制动距离、弯中速度、两种预瞄距离、转向增益、横摆阻尼、侧滑触发位置、侧滑结束朝向、后轮驱动速度、侧滑转向偏置和侧滑反馈。候选搜索使用种子 0；程序随后仅在 Isaac 中使用种子 10、11、12 选择检查点。默认 10 回合评估使用种子 0 和 100–108。

MuJoCo 加载已选定的 `policy.json`。两引擎评估文件记录同一检查点的 SHA-256；独立验证器检查一致性。`--train` 和 `--select` 仅允许用于 Isaac。

当前奖励按约 45° 过度转向峰值和 0.4 秒合格侧滑时长提供训练余量，正式验收仍采用约 20°、0.1 秒。本次开发先运行 96 个候选，首轮迁移可完成驾驶但 MuJoCo 侧滑验收仅 2/10；首轮证据完整保留在 `output/hairpin_fast/attempt1/`。随后仅在 Isaac 中按更大的侧滑余量继续训练：

```bash
bash scripts/run_isaac.sh scripts/run_hairpin_fast.py --engine isaac --train --resume --generations 3 --population 16 --episodes 10 --record
```

`--resume` 根据当前奖励重新评分已有 Isaac 训练记录，选择起始参数，再执行新候选搜索。MuJoCo 曾用于开发验收，因此本实验不声称它在整个开发过程中完全未被观察；没有在 MuJoCo 上执行策略优化或设置专用检查点。

第二轮 MuJoCo 侧滑验收为 7/10，保存在 `attempt2/`。第三轮使用 `--select --strict-margin` 从已有 Isaac 训练候选中筛选：三个源端验证条件都必须达到至少 55° 侧滑峰值与 0.35 秒合格侧滑。正式迁移验收仍维持约 20°、0.1 秒，测试换用种子 0、200–208：

```bash
bash scripts/run_isaac.sh scripts/run_hairpin_fast.py --engine isaac --select --strict-margin --eval-seed-start 200 --episodes 10 --record
```

第三轮在 MuJoCo 的侧滑验收为 8/10，保存在 `attempt3/`。轨迹诊断显示，仅以朝向触发后轮增速释放，无法确认侧滑已经形成或保持。因此最终版本 `policy_version=2` 引入一次触发的建立—保持—恢复状态机：后轮相对前轮的增速根据实际侧滑误差调节，目标约为 `beta=-0.65 rad`；软释放要求 `beta<-0.4 rad`、速度大于 1.2 m/s 连续 0.15 秒，且到达学习的朝向阈值。额外设有 2.75 rad 朝向、0.85 秒时间、出弯和低速恢复条件，避免持续增速失控。状态机和反馈公式在两个引擎中完全相同。

该结构变更后只在 Isaac 中重新优化参数，最终评估使用新种子 0、300–308：

```bash
bash scripts/run_isaac.sh scripts/run_hairpin_fast.py --engine isaac --train --resume --generations 3 --population 16 --eval-seed-start 300 --episodes 10 --record
```

## 场景、速度与通过条件

道路中心线由 6 m 进弯直道、半径 0.8 m 的半圆和 6 m 出弯直道组成，路宽 1.2 m；转弯入口位于 `(6, 0)`，半圆圆心为 `(6, 0.8)`，终点为 `(0, 1.6)`。控制频率为 60 Hz，物理积分频率为 480 Hz。相机跟随车辆，两端使用各自引擎的原生渲染。

成功条件包括：

- 在路径进度小于 6 m 的进弯直道上，实际后轴原点速度峰值至少为 5 m/s。
- 在转弯附近，后轴侧滑角 `beta < -0.35 rad`，同时实际速度大于 1.2 m/s，合格样本累计至少 0.1 s。
- 完成 180° 转弯并到达出弯直道终点，满足位置和朝向容差，期间不翻车、不越出规定范围，状态保持有限值。

侧滑计时是累计时长；验证器另外报告最长连续合格时长，不能把两者混为一谈。`-0.35 rad` 约为 `-20.1°`；对于本场景的左转，它表示车头朝向相对后轴速度方向产生明显过度转向。

`entry_peak_m_s` 的精确定义是 **整个进弯直道、路径进度小于 6 m 的最高车体速度**，并非刚越过转弯入口时的速度。策略会提前制动；本实验关注的制动起点约在弯前 2 m，实际位置由所选检查点的 `brake_distance_m` 决定。应同时查看验证器报告的入口瞬时速度 `speed_at_turn_entry_x6_progress6_m_s`，以及半圆中点前后 0.25 m 路程内的 `apex_speed_range_m_s`。本任务不要求以 5 m/s 通过弯心。

## 修正两个会阻碍高速运行的 PhysX 限制

原低速适配器的角关节速度上限为 **5000 度/秒**。半径 0.045 m 的轮子对应的无滑移轮缘速度只有：

```text
5000 × π / 180 × 0.045 ≈ 3.93 m/s
```

仅提高轮速指令无法突破这个限制。提高关节上限后，还会受到刚体角速度的第二个上限约束：PhysX 默认的约 100 rad/s 对应轮缘速度约 4.5 m/s。高速派生场景将 USD 的关节速度上限和刚体角速度上限同时设为 **30000 度/秒**，相当于约 523.6 rad/s。必须区分 USD 属性使用的度制和运行时状态、控制接口使用的弧度制。

这些设置解除模拟器中的速度截断，不代表真实电机能够输出相应转速或功率。双引擎零转向加速探测已观察到接近 10 m/s 的实际车体速度；这只证明直线加速可行，没有证明 10 m/s 发卡弯可行。探测记录见 `probe_isaac.json` 和 `probe_mujoco.json`。

## 测量与物理假设

路径位置取车体 `base_link` 原点，该原点位于后轴附近。高速适配器将 `vx` / `vy` 统一为这一原点的世界系速度，另外保留 `cg_vx` / `cg_vy`。MuJoCo 使用 `mjOBJ_XBODY` 取得原点速度；Isaac 根据质心偏移，用 `v_origin = v_COM - omega × offset` 换算。

该修正影响漂移判断：质心在普通无滑移转弯时也会具有车体横向速度，不能直接将其当作后轴轮胎侧滑。`rear_slip_beta` 使用后轴原点速度相对车体朝向的夹角；独立验证器另外根据保存的平面速度与 yaw 重算。视频中的车体速度取后轴原点速度模长，轮速则单独显示为轮缘平均速度，避免把空转当成车辆加速。

每个前轮的转向指令和物理关节范围均限制为 **±0.45 rad**。源模型未提供可验证的机械转向止挡，因此这是实验假设，不能称为硬件规格。两引擎共享以下边界条件：

- 前后轴可独立设置轮速伺服目标；漂移阶段可以提高后轮目标轮缘速度。这是模拟执行器能力假设，尚未确认真实车辆具有同样的独立驱动能力。
- 接触摩擦、执行器增益和力矩上限沿用原适配器，不为 MuJoCo 的迁移结果单独调参；这些参数没有通过真实轮胎或电机数据标定。
- 原生轮胎接触是简化的刚性几何与摩擦模型，未包含经过标定的轮胎侧偏/纵滑曲线、悬架和电机转矩—转速曲线。
- 策略使用仿真器提供的理想位置、朝向、速度和关节状态；本实验不验证真实定位、感知、时延或噪声鲁棒性。

以轴距 `L = 0.28764 m`、轮距 `T = 0.212 m`、内前轮止挡 `delta = 0.45 rad` 计算，低速无滑移 Ackermann 后轴中心最小转弯半径为：

```text
R_min = L / tan(delta) + T / 2 ≈ 0.70146 m
```

这只是给定转向假设下的理论值。漂移阶段瞬态轨迹的圆拟合不是稳定圆周运动，也不能证明已经达到车辆最小转弯半径。验证器可报告后轴位置轨迹拟合半径、残差与角度覆盖，但该半径不作为当前通过条件；更不能用 `车速 / 车身横摆角速度` 直接代替漂移轨迹曲率半径。

若取摩擦系数 `mu = 1`，平路持续转弯的理想侧向附着约束 `R >= v² / (mu × g)` 给出：5 m/s 需要约 2.55 m 半径，10 m/s 需要约 10.19 m 半径。在 0.8 m 发卡弯中采用高速接近、提前制动和较慢弯心速度，有明确的物理原因；漂移不免除附着约束。

## 证据与结果解读

主要输出包括 `policy.json`、`training_history.json`、`checkpoint_selection.json`、两引擎的 `*_evaluation.json`、逐回合 `*_trace_seed*.json`、`*_recorded_trace.json`、`*_fast_hairpin.mp4` 和 `verification.json`。完整验证需要两端评估与视频均已生成，并需要原资产验证记录位于 `output/usability/`。

视频以 60 Hz 仿真每隔一个控制步采样，编码为 30 fps，播放速度为 **1 倍实时**。视频显示实际车体速度、轮缘速度、后轴侧滑角、仿真时间和控制阶段；验证器检查帧数、视频时长、解码完整性和录制轨迹一致性。

最终录制回合的接近峰值为 Isaac 6.03 m/s、MuJoCo 6.02 m/s；弯道入口速度分别为 3.02 和 2.39 m/s，弯心速度分别为 2.88–3.38 和 2.69–2.80 m/s。详细通过率、速度和侧滑持续时间见评估文件与独立验证结果。成功转弯与满足侧滑阈值也不自动证明“以小于机械转向最小半径完成发卡弯”；本次尚未证明这一更强结论。
