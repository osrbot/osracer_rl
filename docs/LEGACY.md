# 历史代码与数据迁移

清理原则是移除无人引用的旧接口和已完成的一次性操作，保留仍承担教师、对照或复现作用的实现。删除的源码仍可从 Git 历史恢复。

## 已移除的入口

| 旧位置 | 处理与替代 |
| --- | --- |
| `src/racing/isaac_env.py` | 删除兼容转发层；使用 `racing.simulators.create_simulator("isaac", track)` |
| `src/racing/mujoco_env.py` | 删除兼容转发层；使用 `racing.simulators.create_simulator("mujoco", track)` |
| `tools/assets/rebrand_public_media.py` | 删除已完成的一次性改名脚本；审核素材统一在 `publication/assets/` 管理 |
| `tools/assets/organize_outputs.py` | 删除会搬移历史日志的旧整理脚本；改用下方只读索引工具 |

外部脚本若还导入 `racing.isaac_env` 或 `racing.mujoco_env`，必须迁移到工厂入口。Isaac 仍须通过 `tools/runtime/run_isaac.sh` 使用其原生 Python，统一导入接口不代表两种原生运行时可以互换。

历史检查点中的源码哈希和路径是当时身份的一部分。当前源码与哈希不匹配时应恢复匹配快照，或创建新的检查点并重新验收；改写哈希本身不能转移历史资格。

## 保留的历史实现

| 位置 | 保留理由 |
| --- | --- |
| `src/racing/control/policy*.py`、CEM 与联合训练入口 | 当前固定对手、PPO 教师和显式 `algorithm=cem` 对照仍依赖它们 |
| `tools/hairpin/` | 审核发布材料的早期发卡弯复现实验；不作为新 PPO 训练入口 |
| `tools/diagnostics/`、相关独立审计工具 | 原生接触、桥面和失败机制的复现证据 |
| `tools/artifacts/` | 读取历史产物，向新的运行目录导出索引和事件 |
| `publication/presentation/releases/` | 审核后的发布文件，区别于临时 `build/` 产物 |

## 历史产物浏览与 TensorBoard 转换

以下命令在仓库根目录、安装依赖后运行。仅适用于本机已有 `output/racing/` 历史归档的情况；公开仓库不包含该目录。

```bash
.venv/bin/python tools/artifacts/index_legacy_output.py

# 可选：只建立按类型浏览的符号链接
.venv/bin/python tools/artifacts/index_legacy_output.py --link-view runs/_legacy/by-category

# 历史 JSON 不是 TensorBoard event，须先转换
.venv/bin/python tools/artifacts/export_legacy_tensorboard.py
.venv/bin/tensorboard --logdir runs/_legacy/tensorboard --port 6006
```

索引写入 `runs/_legacy/index.json`，按模型、轨迹、视频、日志、图像、指标和源码快照分类。转换工具为每份 `training_history.json` 创建独立的事件目录，保留候选指标、参数和代际汇总；它们都不改动旧 JSON 或搬移原始证据。

真正迁移一个历史批次时应以批次为单位核对路径与 SHA-256，不能按扩展名混合移动模型、录像和日志。新训练统一使用 [runs 布局](RUN_ARTIFACTS.md)。
