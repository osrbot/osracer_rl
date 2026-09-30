# 项目结构

项目按“可安装源码、命令工具、原生扩展、静态资产、运行产物、发布材料”划分。目录名称表达技术职责，不再使用含义宽泛的 `research/`、`scripts/` 或顶层散装源码目录。

| 路径 | 职责 | 版本控制策略 |
| --- | --- | --- |
| `src/racing/` | 唯一可安装 Python 包；控制、感知、仿真器接口、训练和评估实现 | 提交 |
| `tools/` | 薄命令入口、诊断、审计、资产维护和历史实验工具 | 提交 |
| `native/` | 可选 C++ 扩展及其构建/探针代码 | 提交，构建产物忽略 |
| `assets/vehicles/` | ROS、OpenUSD、MuJoCo 与 CAD 车辆描述 | 提交精选资产 |
| `assets/tracks/` | 赛道几何、纹理、来源与目录 | 提交 |
| `tests/` | 源码契约和回归测试 | 提交 |
| `deployment/` | ROS2 等部署适配 | 提交 |
| `docs/` | 工程契约、验证状态、部署边界和运行说明；不放演示媒体 | 提交 |
| `publication/assets/` | 网站、PPT 和演示视频共用的唯一审核媒体库 | 提交精选媒体 |
| `publication/site/` | 静态站生成器 | 提交生成器，忽略 `_site/` |
| `publication/presentation/` | PPT、讲稿、直播说明和视频制作 | 提交源材料，忽略本地生成物 |
| `runs/<run-id>/` | 新训练/评估的检查点、指标、TensorBoard、轨迹、录像和日志 | 忽略 |
| `output/racing/` | 旧版 34GB 证据库，文件内含历史路径和哈希 | 忽略并只读保留 |

## 代码边界

业务模块只能从 `racing` 包导入。`tools/` 可以调用包 API，但 `src/racing/` 不得反向导入 `tools/`。两个仿真器都实现 `racing.simulators.base.SimulatorBackend`，通过 `racing.simulators` 注册表创建；后端配置在 `src/racing/config/simulators/`。

`native/bridge_contact_adapter/` 是可选实验扩展，不随 Python wheel 安装，也不能成为 MuJoCo/Isaac 正常运行的隐式依赖。

## 产物边界

新任务的完整布局和 TensorBoard 命令见 [运行产物目录](RUN_ARTIFACTS.md)。旧数据可用 `python tools/artifacts/index_legacy_output.py` 建立索引；索引不移动、不重写、不重新哈希历史证据。

## 公开仓库边界

公开提交包含源码、测试、配置、文档和精选发布媒体。`.venv/`、`build/`、`dist/`、`*.egg-info/`、`runs/`、`output/`、缓存、GPU 日志和未审核训练中间产物均不提交。发布前仍需检查 `git status` 和 wheel 内容。

## 从哪里开始

1. 阅读 [验证状态](VALIDATION_STATUS.md) 理解结论边界。
2. 按 [运行产物目录](RUN_ARTIFACTS.md) 启动训练和 TensorBoard。
3. 查看 [冻结发布](RELEASE.md) 复现历史已审计结果。
4. 演示和媒体入口统一在 [`publication/`](../publication/README.md)。
5. 扩展仿真器前阅读 [仿真器后端](SIMULATORS.md)。
