# 直播媒体清单

| 顺序 | 文件 | 直播用途 | 可以声称的结论 | 不可声称的结论 |
| --- | --- | --- | --- | --- |
| 1 | `output/usability/openusd/NEORACER_OpenUSD_IsaacSim.mp4` | OpenUSD 资产导入 Isaac Sim、环绕观察和关节运动 | 20 个网格解析、10 刚体、9 物理关节、6 可动自由度；完成短程 PhysX 步进 | 车辆已在 Isaac 完成可靠驾驶 |
| 2 | `output/usability/mujoco/neoracer_mujoco_usability.mp4` | MuJoCo 资产加载、关节/执行器步进 | `robot.xml`/`scene.xml` 完成 500 步、无数值警告、视频可解码 | 固定基座模型已经具备地面行驶能力 |
| 3 | `output/racing/mujoco/bahrain_c03v10c_dev24_seed0.mp4` | MuJoCo 代表性双车整圈/超车 | v10c 在 MuJoCo 24/24 有效整圈与有效超车；该段是可追溯代表样例 | 一个样例证明所有条件下都稳定 |
| 4 | `output/racing/isaac/bahrain_c03v10c_dev24_seed0.mp4` | Isaac/PhysX 代表性整圈 | Isaac v10c 有 22/24 有效整圈；录像与轨迹独立审计通过 | Isaac 已 24/24 完成，或速度 9 m/s 无代价 |
| 5 | `output/hairpin_fast/mujoco_fast_hairpin.mp4` | 180° 回头弯漂移示例（仿真） | 在特定 RC 模型、策略和后轮增速动作下，可记录到持续侧滑 | 单电机四驱实车已经能复现该漂移 |
| 6 | `output/racing/mujoco/bahrain_c03v10c_sensor_perturb_seed10.mp4` | 失败对照：传感器扰动 | 组合扰动下策略会失败，失败被保留而非隐藏 | 现策略对噪声/丢束/延迟鲁棒 |

所有媒体均已有独立解码/哈希或验证记录。直播播放前请在本机以 `ffprobe` 或媒体库 `output/racing/media.html` 复查文件存在性；不要从该表推断未列出的赛道、种子或物理条件。
