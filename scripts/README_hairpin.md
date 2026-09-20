# OSRACER：Isaac Sim 6.0 → MuJoCo 180° 回头弯

这是使用真实关节驱动与轮地接触的低速闭环驾驶任务。策略在本机 Isaac Sim **6.0.1 / PhysX** 中用交叉熵方法（CEM）训练，再将同一份 JSON 参数和 Python 推理代码直接用于 MuJoCo。不是 PPO 神经网络，也不是轨迹动画。

## 运行

在工作区根目录执行：

```bash
# 5 代 × 10 个候选 × 2 个初始条件，共 100 个训练 episode。
# 自动在固定 Isaac 条件上选出最终检查点，再测试 10 个 episode。
bash scripts/run_isaac.sh scripts/run_hairpin.py --engine isaac --train --generations 5 --population 10 --episodes 10

# 加载检查点，原生 Isaac 渲染与录像。
bash scripts/run_isaac.sh scripts/run_hairpin.py --engine isaac --episodes 10 --record

# 不重新训练、不改参数，MuJoCo 原生动力学测试与录像。
python3 scripts/run_hairpin.py --engine mujoco --episodes 10 --record

# 检查成功率、策略 SHA256、录像解码与轨迹一致性。
python3 scripts/verify_hairpin.py

# 生成并排对比视频和结果报告。
python3 scripts/summarize_hairpin.py
```

输出目录：`output/hairpin/`。`policy.json` 是可迁移策略；`training_history.json` 保留全部候选、奖励和 episode 指标；`checkpoint_selection.json` 保留固定条件下的最终选择过程。`*_evaluation.json`、`*_trajectory.json` 与 `*_recorded_trajectory.json` 是评估证据；`isaac_hairpin.mp4` 和 `mujoco_hairpin.mp4` 为 1280×720、30 fps 原生仿真录像。

可用 `--baseline` 检查未训练的初始反馈策略。已有训练日志时，`--engine isaac --select` 可在固定种子上重新选择候选。MuJoCo 模式禁止训练或选择检查点。

`run_isaac.sh` 默认使用 `/home/osrbot/rlgpu_ws/isaac-sim-standalone-6.0.1-linux-x86_64`；可用 `OSRACER_ISAAC_DIR` 替换。启动器仅对当前 Isaac 子进程预加载配套 NCCL 库。MuJoCo 脚本优先使用已安装包，否则复用本机 uv 缓存；渲染使用 EGL。录像依赖 ffmpeg、Pillow。

## 任务与策略接口

- 后轴参考点沿 1.2 米直道进入半径 0.8 米左回头弯，航向转过 180°，再沿 1.2 米直道驶出。中心线总长约 4.913 米，画面道路宽 0.58 米。
- 30 Hz 闭环控制。Isaac 物理步长 1/240 秒；MuJoCo 使用 implicitfast，步长 1/480 秒。
- 动作是四轮**关节速度目标 rad/s**和两个转向**关节位置目标 rad**。顺序为 LF、RF、LR、RR；右后轮正向行驶时的关节速度为负，其余为正。正转角向左转。
- 观测包含仿真提供的车辆位姿、路径预瞄误差、四轮编码器速度及转向编码器位置。只有轮速和转角反馈不足以确定车相对弯道的位置，因此当前版本使用理想定位与已知中心线，没有视觉或激光感知。
- 策略由纯追踪几何、横向误差反馈、转角反馈、轮速反馈和 Ackermann 动作映射组成。CEM 优化预瞄距离、巡航速度、曲率减速系数等七个连续参数。奖励包含完成奖励、路径进度、耗时、横向误差；参数边界和目标变化率受限。
- 初始参数本身是可以行驶的控制先验，训练优化其性能；该实现没有声称从随机神经网络中学习全部驾驶行为。

## 两端物理配置

保留原车视觉资产、连杆质量和惯量。所有修改都写在派生模型或 USD 会话层，原始资产不改动：

- MuJoCo 加入浮动基座、地面；Isaac 加入地面，使用原有自由车体。
- 将轮胎碰撞形状替换为半径 0.045 米、宽度 0.04 米的圆柱；地面摩擦系数 1.0。车轮圆柱中心使用原轮体质心。
- 四轮速度伺服增益 0.03 Nm/(rad/s)，力矩限制 ±0.3 Nm；转向位置伺服 kp=3 Nm/rad、kd=0.08 Nm/(rad/s)，力矩限制 ±1 Nm。
- 禁用车内自碰撞；其余 CAD 碰撞形状仍用于地面接触。Isaac 将动态三角网格明确设为 convexHull。
- 两端均从静止落地，物理稳定 0.4167 秒。种子 0 为名义初始状态，其余种子给横向位置 ±0.025 米、航向 ±0.03 rad 的扰动。
- 两引擎的接触解算器不同；MuJoCo 还保留微小关节阻尼 0.0001 和摩擦损失 0.001，Isaac 关节摩擦设为 0。这是近似对齐的模拟执行器，不是经过实车辨识的电机/轮胎模型。

## 验收与适用范围

成功条件是驶至末段、后轴距终点小于 0.16 米且航向距 180° 小于 0.18 rad，同时没有翻覆或后轴横向误差超过 0.28 米。这里的“完成”指通过弯道，**不要求在终点停车**。录像在达成条件时结束。

默认评估为 1 个名义条件与 9 个未参与训练的扰动条件（种子 100–108）。最终候选选择使用 Isaac 种子 10–12。MuJoCo 只读取固定检查点，不利用迁移结果优化策略。

本轮验证范围是固定半径、平地、固定质量和摩擦的低速回头弯；不等于高速漂移、不同路面/轮胎的泛化或实车安全认证。道路标线没有物理墙体。`verification.json` 中车体包络的道路余量是基于矩形近似的补充检查，不是精确 CAD 扫掠碰撞证明。

录像直接采样当前物理仿真画面；运行期间只更新执行器目标，只有 reset 才设置初始位姿。Isaac 渲染使用零时间推进并检查渲染前后车体位置不变；录像轨迹与无渲染轨迹另外比较。API 依据：[Isaac Sim 6.0 Core Prims](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/deprecated/isaacsim.core.prims/docs/index.html)。
