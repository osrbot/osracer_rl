# 24 赛道 PPO benchmark

`osracer-benchmark` 按赛季目录为每条赛道训练独立 PPO 策略。所有赛道使用同一个父检查点和冻结训练协议；不会把上一条赛道的结果传给下一条，避免赛历顺序改变实验结果。每个子任务均生成独立的 `.pt`、ONNX、JSONL、TensorBoard 和训练日志。

## 一行启动

先按[使用说明](RUN_ARTIFACTS.md)完成单赛道训练；下面使用该流程生成的 `runs/demo/checkpoints/policy.pt`。仓库不附带本机历史权重。最近本地批次只有 6/24 条赛道通过全部筛选种子，批量命令不保证收敛。

```bash
.venv/bin/osracer-benchmark train \
  benchmark_name=ppo-2025 \
  season=2025 \
  checkpoint=runs/demo/checkpoints/policy.pt \
  +simulator=mujoco \
  +train=benchmark
```

`benchmark` profile 默认把预算集中在 14400 步分赛段专家迁移和 3 个评估种子，只保留 1×256 PPO steps 用于训练链路与指标验证。分赛段迁移会均匀覆盖整圈，并通过父策略锚定避免破坏已有驾驶能力。每条赛道最多尝试 3 次，训练随机种子按 attempt 递增而评估种子保持 0–2，避免无效的确定性重试。完整 24 赛道训练是长任务；命令被中断后原样重跑即可从未完成赛道继续。已完成赛道只有在 `.pt`、ONNX、checkpoint 身份和全部评估种子均通过验证后才会跳过；未通过策略标记为 `unqualified`，失败现场会保留。

指定部分赛道可用于小批次验证：

```bash
.venv/bin/osracer-benchmark train \
  benchmark_name=ppo-2025-smoke tracks=melbourne,shanghai,suzuka \
  checkpoint=runs/demo/checkpoints/policy.pt \
  +simulator=mujoco +train=quick env.episode_length_s=180
```

## 状态、TensorBoard 与播放

```bash
.venv/bin/osracer-benchmark status benchmark_name=ppo-2025

.venv/bin/tensorboard \
  --logdir runs/ppo-2025/metrics/tensorboard \
  --port 6006

.venv/bin/osracer-benchmark play \
  benchmark_name=ppo-2025 track=melbourne
```

汇总位于 `runs/ppo-2025/reports/benchmark_summary.json` 和 `benchmark_summary.csv`。表中明确列出总赛道数、完成/失败数、多种子有效圈、圈速、超车、步数以及检查点哈希。TensorBoard 的一级 run 名为 `r01-melbourne` 到 `r24-yas_marina`，可直接比较各赛道曲线。

## 全程均速微调

`pace` profile 从一个已经安全的 `.pt` 检查点继续 PPO 微调，并把全程平均物理速度、沿赛道平均进度速度和圈时写入 checkpoint、JSONL 与 TensorBoard：

```bash
.venv/bin/osracer-train \
  +simulator=mujoco +task=racing/bahrain +train=pace \
  experiment_name=pace-bahrain \
  checkpoint=runs/demo/checkpoints/policy.pt \
  device=cuda
```

配速项按 `delta_progress × speed / max_speed` 累积，并用中心线距离衰减，因此只有沿赛道快速、仍在道路内部的前进获得额外奖励。碰撞、越界或停滞会立即补缴剩余回合的时间成本，提前失败不能通过少支付逐步时间成本取得更高回报。`corner_risk` 对高速急转状态施加连续惩罚；推理图同时使用轮速与转向的联合门控，只在传感器观测到车速至少 3.5m/s 且归一化转向达到 0.5 时降低一个 `1/32` 速度档。该门控保存在 `.pt` 的模型规格中，并直接导出到 ONNX。

模型选择先提高六个冻结评估种子的有效圈覆盖率；全部有效后，才按沿赛道全程均速、物理均速和原始奖励排序。量化策略的 refinement 每轮从最后一个安全模型出发，用不超过 1% 轨迹动作变化的信任域候选做闭环验收，拒绝的更新不会继续累积。

可用 `reward.pace_weight`、`reward.time_cost`、`reward.failure_horizon_scale`、`reward.corner_risk_weight`、`policy.corner_speed_threshold_m_s`、`policy.corner_steer_threshold` 和 `policy.corner_slowdown_bins` 做冻结协议下的 A/B；基线与候选必须保持父 checkpoint、训练预算、种子、对手和回合时长相同。

## 本地门控诊断记录

代码版本 `e35ce81` 的 `pace-bahrain-corner-guard-v1` 使用下列命令。此命令用于追溯，需要本机历史父检查点，不能在全新克隆中直接运行：

```bash
.venv/bin/python -m racing.runtime.ppo \
  +simulator=mujoco +task=racing/bahrain +train=pace \
  experiment_name=pace-bahrain-corner-guard-v1 \
  checkpoint=runs/ppo-2025-mujoco-v2-r04-bahrain/checkpoints/policy.pt \
  seed=73 device=cuda train.iterations=1 train.steps_per_iteration=256 \
  train.learning_epochs=1 train.refinement_cycles=0 train.refinement_epochs=0
```

| 同一组 seed 0–5 | 父检查点 | 启用门控的保留模型 |
| --- | --- | --- |
| 有效圈 | 5/6 | 6/6 |
| 碰撞回合率 | 16.7% | 0% |
| 回合物理均速的等权均值 | 4.247 m/s | 4.315 m/s |
| 回合进度速度的等权均值 | 4.240 m/s | 4.310 m/s |
| 候选平均有效圈时 | — | 62.92 s |

相对均速均值差为 +1.61%，但父检查点包含一个提前碰撞的回合，因此不能解释为相同完整圈上的配对提速。保留 checkpoint 的 `iteration=0`、`total_steps=0`；一次 PPO 更新未超过初始筛选结果，当前改善来自手工设定的推理门控，**没有证明奖励学习带来增益**。seed 0–5 已用于选择门控阈值，是开发诊断集，不是独立留出集。

产物在本地 `runs/pace-bahrain-corner-guard-v1/`，不随公开仓库提交。身份记录如下：

| 文件 | SHA-256 |
| --- | --- |
| 父 `policy.pt` | `a7ec649879ef8bf11d92e71a88e36d8845af9f921559d1f9d0f2a48365543416` |
| 保留 `policy.pt` | `2fae09086c98fc3fe005f8d82589a9c9b038f742d2706259ce0fb030bb7cf05b` |
| 导出 `policy.onnx` | `d40c8e1fc52601a48c15bce0ffecf5a545b9c2880746126f508840a14951a0be` |

该结果属于当前近似赛道资产和本地训练协议下的 benchmark。资产并不声称是赛季历史布局的毫米级复现；LoopX experiment board 中也不会把本地结果标记成独立官方评分。
