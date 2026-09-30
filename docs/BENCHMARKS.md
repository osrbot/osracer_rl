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

`benchmark` profile 默认使用 20×1024 PPO steps、14400 步分赛段专家迁移、3 个评估种子和一轮带锚定的策略精炼。分赛段迁移会均匀覆盖整圈，并通过父策略锚定避免破坏已有驾驶能力。完整 24 赛道训练是长任务；命令被中断后原样重跑即可从未完成赛道继续。已完成赛道只有在 `.pt`、ONNX、checkpoint 身份和全部评估种子均通过验证后才会跳过；未通过策略标记为 `unqualified`，再次执行时创建新的 attempt，失败现场会保留。

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

该结果属于当前近似赛道资产和本地训练协议下的 benchmark。资产并不声称是赛季历史布局的毫米级复现；LoopX experiment board 中也不会把本地结果标记成独立官方评分。
