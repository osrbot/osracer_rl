# 参考材料与采用范围

链接核对日期：2026-10-06。以下区分外部参考与本仓库实现，不将配置外观相似视为算法复现。

| 材料 | 采用范围 | 本仓库对应位置与边界 |
| --- | --- | --- |
| [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP/) | 参考仿真器、任务、算法分离，以及组合配置的命令习惯 | `src/racing/simulators/`、`config/`、`runtime/`。本仓库使用 TOML 和有限 `KEY=VALUE` 解析，不实现完整 Hydra，也未复现 ASAP 的人形运动算法 |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | PPO 策略更新的算法来源 | `runtime/ppo.py`；当前还有教师模仿、DAgger、量化动作和推理门控，是混合训练实现。连续分布在量化动作上计算概率属于近似，不能等同于论文的标准连续动作实现 |
| [TensorBoard 入门](https://www.tensorflow.org/tensorboard/get_started) | event 文件与 `--logdir` 的使用 | `runtime/artifacts.py` 的 `MetricLogger` 持久化事件与 JSONL；具体目录与预热期日志边界见[使用说明](RUN_ARTIFACTS.md) |
| [PyTorch ONNX 文档](https://docs.pytorch.org/docs/stable/onnx.html) | 模型导出接口与执行边界 | `control/neural_policy.py` 导出确定性 actor；370 维观测编码、上一动作回填和执行器转换仍由宿主负责 |
| [solidworks_urdf_exporter_pro](https://github.com/osrbot/solidworks_urdf_exporter_pro) | 车辆资产的导出来源 | `assets/vehicles/osracer/`；外部工具不随本仓库分发，几何与动力学验证由本工程单独记录 |

赛道来源、许可证和近似假设以 [赛道来源说明](TRACK_SOURCES.md)与 [`assets/tracks/sources/LICENSE.md`](../assets/tracks/sources/LICENSE.md)为准；车辆及其他第三方材料的权利边界见仓库 [LICENSE](../LICENSE) 和随附声明。参考仓库的许可证不自动覆盖本仓库或第三方资产。

## 本地结果的引用方式

新实验至少记录代码版本、父 checkpoint SHA-256、配置、训练预算、筛选种子和独立评估种子。用训练或调参种子筛选的结果只能作为开发诊断。速度比较须同时列出有效圈率、碰撞/越界、圈时和计量方式；失败截断回合的均速不能直接解释为完整圈提速。

当前 PPO 的局部结果和局限见 [BENCHMARKS.md](BENCHMARKS.md)；历史 CEM 成绩见 [RELEASE.md](RELEASE.md)。这两类结果不混用检查点身份或验收结论。
