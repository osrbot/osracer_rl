# 24 赛道 PPO benchmark

`osracer-benchmark` 按赛季目录为每条赛道训练独立 PPO 策略。所有赛道使用同一个父检查点和冻结训练协议；不会把上一条赛道的结果传给下一条，避免赛历顺序改变实验结果。每个子任务均生成独立的 `.pt`、ONNX、JSONL、TensorBoard 和训练日志。

## 一行启动

```bash
.venv/bin/osracer-benchmark train \
  benchmark_name=ppo-2025 \
  season=2025 \
  checkpoint=runs/ppo-cem-pace-v2/checkpoints/policy.pt \
  +simulator=mujoco \
  +train=benchmark
```

`benchmark` profile 默认把预算集中在 14400 步分赛段专家迁移和 3 个评估种子，只保留 1×256 PPO steps 用于训练链路与指标验证。分赛段迁移会均匀覆盖整圈，并通过父策略锚定避免破坏已有驾驶能力。每条赛道最多尝试 3 次，训练随机种子按 attempt 递增而评估种子保持 0–2，避免无效的确定性重试。完整 24 赛道训练是长任务；命令被中断后原样重跑即可从未完成赛道继续。已完成赛道只有在 `.pt`、ONNX、checkpoint 身份和全部评估种子均通过验证后才会跳过；未通过策略标记为 `unqualified`，失败现场会保留。

指定部分赛道可用于小批次验证：

```bash
.venv/bin/osracer-benchmark train \
  benchmark_name=ppo-2025-smoke tracks=melbourne,shanghai,suzuka \
  checkpoint=runs/ppo-cem-pace-v2/checkpoints/policy.pt \
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
  checkpoint=runs/ppo-2025-mujoco-v2-r04-bahrain/checkpoints/policy.pt \
  device=cuda
```

配速项按 `delta_progress × speed / max_speed` 累积，并用中心线距离衰减，因此只有沿赛道快速、仍在道路内部的前进获得额外奖励。碰撞、越界或停滞会立即补缴剩余回合的时间成本，提前失败不能通过少支付逐步时间成本取得更高回报。`corner_risk` 对高速急转状态施加连续惩罚；推理图同时使用轮速与转向的联合门控，只在传感器观测到车速至少 3.5m/s 且归一化转向达到 0.5 时降低一个 `1/32` 速度档。该门控保存在 `.pt` 的模型规格中，并直接导出到 ONNX。

模型选择先提高六个冻结评估种子的有效圈覆盖率；全部有效后，才按沿赛道全程均速、物理均速和原始奖励排序。量化策略的 refinement 每轮从最后一个安全模型出发，用不超过 1% 轨迹动作变化的信任域候选做闭环验收，拒绝的更新不会继续累积。

可用 `reward.pace_weight`、`reward.time_cost`、`reward.failure_horizon_scale`、`reward.corner_risk_weight`、`policy.corner_speed_threshold_m_s`、`policy.corner_steer_threshold` 和 `policy.corner_slowdown_bins` 做冻结协议下的 A/B；基线与候选必须保持父 checkpoint、训练预算、种子、对手和回合时长相同。

冻结 seed 0–5 的本地 Bahrain 诊断中，父 checkpoint 为 5/6 有效圈、碰撞率 16.7%、平均物理速度 4.247m/s。`pace-bahrain-corner-guard-v1` 达到 6/6、零碰撞/越界、平均物理速度 4.315m/s、平均进度速度 4.310m/s、平均圈时 62.92s；均速相对父 checkpoint 提高 1.61%。产物位于 `runs/pace-bahrain-corner-guard-v1/`。这是本地诊断结果，不外推到其他赛道或 Isaac。

该结果属于当前近似赛道资产和本地训练协议下的 benchmark。资产并不声称是赛季历史布局的毫米级复现；LoopX experiment board 中也不会把本地结果标记成独立官方评分。
