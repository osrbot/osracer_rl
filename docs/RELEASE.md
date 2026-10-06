# 历史 CEM 冻结基线

本页保留历史 CEM 配置、当时的成绩与本地证据路径，不代表当前 PPO 的成绩。复现需要对应历史检查点及匹配源码；这些原始产物不随公开仓库分发。新训练请从[使用说明](RUN_ARTIFACTS.md)开始，公开精选素材见 [publication/assets](../publication/assets/README.md)。

## 冻结配置

| 项 | 值 |
| --- | --- |
| 检查点 | `output/racing/candidates/g00c_v10c_lateral.json` |
| 训练参数来源 | 联合 CEM 候选 `g00_c03`（12 次候选评价、Isaac 20 + MuJoCo 16 原生回合） |
| 控制器 | `RacingPolicy` + 闭环漂移 + 安全监督器（原始回波近场、计划减速制动、有界倒车脱困、窄道车辆绕行、0.18 s 车辆预警） |
| 输入 | 四轮编码器、两个前轮转角、15 Hz 单线激光（laser 坐标系、270°、361 束）；无全局位置 |
| 巡航 | 7.25 m/s（共享/冻结）；9.0 m/s 仅 MuJoCo 全赛道有效 |
| 对手 | 同类传感器策略，2.8 m/s 巡航，发车领先 3 m |

历史源码哈希（当时使用 `racing/policy.py` 等路径；当前代码位于 `src/racing/control/` 与 `src/racing/perception/`）逐项记录在检查点的 `actor_source_sha256` 中，可用下列命令复核：

```bash
.venv/bin/python -I -c "import sys,json;sys.path.insert(0,'.');\
from racing.control.policy_bundle import actor_source_hashes;\
spec=json.load(open('output/racing/candidates/g00c_v10c_lateral.json'));\
print(actor_source_hashes(spec)==spec['actor_source_sha256'])"
```

目录重构后上述校验可能返回 `False`。这意味着当前代码不等同于冻结源码，应取回匹配快照并重新验收；不能只更新哈希来沿用历史资格。

## 冻结成绩

| 榜单 | 有效整圈 | 有效超车 | 单圈 | 峰值速度 | 录像 / 审计 |
| --- | --- | --- | --- | --- | --- |
| MuJoCo，24 赛道，巡航 7.25 m/s | **24/24** | 每条 1 次 | 38.9–95.1 s | 7.23 m/s | 24 段 / 24 通过 |
| Isaac，24 赛道，巡航 7.25 m/s | 22/24 | 每条 1 次 | 41.2–99.7 s | 7.29 m/s | 24 段 / 24 通过 |
| MuJoCo，24 赛道，巡航 9.0 m/s | **24/24** | 每条 1 次 | 37.5–85.2 s | 8.97 m/s | 24 段 / 24 通过 |
| Isaac，24 赛道，巡航 9.0 m/s | 20/24 | 每条 1 次 | 40.4–90.5 s | 9.00–9.04 m/s | 24 段 / 24 通过 |
| MuJoCo 留出种子（5 赛道×10–14、全部 24 赛道×10） | 41/44 | — | — | — | 49 份轨迹通过 |
| Isaac 留出种子（3 赛道×10–12） | 6/9 | 每条 1 次 | 53.0–79.4 s（Bahrain/Monaco） | — | 9 份轨迹通过 |

报告共 174 个回合、174 份轨迹全部通过证据完整性审计，其中 141 场任务成功；可排序报告见 RESULTS.md（本地 `output/racing/RESULTS.md`），交互报告见 index.html（本地 `output/racing/index.html`）。

## 素材

目录已经把散落的文件整理成便于查看的结构，入口是 output/racing/README.md（本地 `output/racing/README.md`）：

- `output/racing/policies/`：62 个策略检查点的符号链接视图与清单（含参数、版本、训练引擎、状态、SHA-256），**冻结版是 `frozen-candidates-g00c_v10c_lateral.json`**。
- `output/racing/logs/`：运行日志按 `isaac/`、`mujoco/`、`bridge/`、`training/`、`qualification/`、`misc/` 分组（本轮归位 264 个文件）。
- 顶层从 446 项降到 184 项；结果、轨迹、录像仍在原路径，未移动、未改名，因此已记录的证据哈希与引用保持有效。
- 旧日志搬移脚本已删除。现在使用 `.venv/bin/python tools/artifacts/index_legacy_output.py`，只在 `runs/_legacy/` 写入索引，不改动历史文件；详见[历史复现](LEGACY.md)。

media.html（本地 `output/racing/media.html`） 是可浏览素材库（点开即播），MEDIA.md（本地 `output/racing/MEDIA.md`） 是同一批素材的表格索引，`MEDIA_INDEX.json` 是机器清单。当前共 **411 段原生录像**（证据 91 / 失败对照组 8 / 历史批次 312，后者含被撤回实验的录像）与 905 张首帧预览，全部通过 `ffprobe` 解码校验，逐段带 SHA-256、分辨率、帧数与对应回合指标。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -I tools/assets/index_media.py \
  --tag c03v10c_dev24 --tag c03v10c_cruise9_dev24 --tag c03v10c_cruise9_holdout \
  --tag c03v10c_sensor_perturb --tag c03v10c_isaac_holdout
```

## 已知限制

1. **Isaac 的两种失败仍未解决**：Spa 起步车车接触、Suzuka 桥面段（撞坡道后车头上仰 0.516 rad 触发失稳）。Isaac 留出种子显示 Bahrain/Monaco 在种子 10–12 上全过，残余问题集中在 Suzuka 桥面。
2. **Isaac 的稳定上限是 7.25 m/s**：9.0 m/s 下新增 Austin、Las Vegas 两条停滞；同一配置在 MuJoCo 全赛道有效。五种修复尝试（倒车额度、累计倒车上限、阻塞计时衰减、三点掉头、侧向阈值）全部失败并保留证据。
3. **传感器扰动下不具备鲁棒性**：噪声 0.02 m + 丢束 5% + 延迟 50 ms 时 10/10 失败。已定位两条机制（丢束的"单束空洞"、里程计 5 点局部拟合在 2 cm 噪声下 `registration_failed`），并在实验分支上验证：三项机制合并后 Monaco 253.3 s、Singapore 153.7 s 可在扰动下有效完圈（名义工况几乎不变），但该分支未合入冻结版，因为它在名义工况下引入了 Austin 碰撞。
4. 赛道几何是公开 GeoJSON 的近似快照，路宽 1.5 m 与 Suzuka 桥面高程为训练假设；激光测距是共享解析模型，不是引擎原生 RTX 激光。

## 复现命令

```bash
# 观测契约与恢复逻辑测试
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q

# MuJoCo 全赛道（含录制）
.venv/bin/python -I -m racing.runtime.sweep --checkpoint output/racing/candidates/g00c_v10c_lateral.json \
  --tag c03v10c_dev24 --record-tracks <全部赛道> --seconds 200 --seed 0

# Isaac 全赛道（含录制）
bash tools/runtime/run_isaac.sh tools/runtime/run_racing_batch.py --tracks <全部赛道> --record-tracks <全部赛道> \
  --checkpoint output/racing/candidates/g00c_v10c_lateral.json --tag c03v10c_dev24 --seconds 200 --seeds 0

# 独立审计与报告
.venv/bin/python -I -m racing.evaluation.verify output/racing/mujoco/*_c03v10c_dev24.json
.venv/bin/python -I -m racing.evaluation.report --tag c03v10c_dev24 --checkpoint output/racing/candidates/g00c_v10c_lateral.json
```
