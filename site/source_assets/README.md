# 已审核站点媒体

这里是公开静态站所需的**精选、可复现**媒体副本；它不是完整实验输出仓。每个文件均从项目的 `output/` 证据档案复制，站点构建脚本只读取本目录，避免将 35 GB 轨迹、训练中间物和历史录像推送到 GitHub。

| 文件 | 证据档案中的来源 | 用途 |
| --- | --- | --- |
| `openusd-usability.mp4` / `openusd-preview.png` | `output/usability/openusd/` | OpenUSD / Isaac Sim 资产层加载与关节展示 |
| `mujoco-usability.mp4` / `mujoco-preview.png` | `output/usability/mujoco/` | MuJoCo 资产层加载与步进展示 |
| `mujoco-bahrain.mp4` / `mujoco-race.png` | `output/racing/mujoco/bahrain_c03v10c_dev24_seed0.*` | MuJoCo 代表性双车整圈 |
| `isaac-bahrain.mp4` / `isaac-race.png` | `output/racing/isaac/bahrain_c03v10c_dev24_seed0.*` | Isaac 代表性双车整圈 |
| `drift.png` | `output/hairpin_fast/mujoco_fast_peak_slip.png` | 回头弯侧滑的仿真 A/B 说明 |
| `perturb.png` | `output/racing/mujoco/bahrain_c03v10c_sensor_perturb_seed10.png` | 组合传感器扰动失败对照 |

源文件的完整媒体索引、轨迹、哈希和失败对照保留在本地 `output/` 证据档案中。更新此目录时，应重新运行 `ffprobe`，并在 `presentation/MEDIA_MANIFEST.md` 中复核结论范围。
