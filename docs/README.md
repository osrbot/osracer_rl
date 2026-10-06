# 文档导航

第一次运行请先按 [README](../README.zh-CN.md#安装前先确认环境)准备环境并完成短回合检查，再阅读[训练、播放与 TensorBoard](RUN_ARTIFACTS.md)。实验结论以[当前验证状态](VALIDATION_STATUS.md)为准。

| 用途 | 文档 |
| --- | --- |
| 安装、训练、播放、模型导出与日志排查 | [使用说明](RUN_ARTIFACTS.md) |
| 24 赛道批量训练、断点续跑与诊断记录 | [Benchmark](BENCHMARKS.md) |
| 目录职责、构建产物与公开提交范围 | [项目结构](PROJECT_STRUCTURE.md) |
| 后端注册、接口与扩展方式 | [仿真器后端](SIMULATORS.md) |
| 观测、执行器、奖励与验收契约 | [工程说明](ENGINEERING.md) |
| ASAP、PPO、TensorBoard、ONNX 与资产来源 | [参考材料](REFERENCES.md) |
| ROS2 映射与实车限制 | [部署说明](DEPLOYMENT.md) |
| 提交范围、保密检查与历史边界 | [提交前审查](PUBLICATION_REVIEW.md) |
| 已删除入口、历史数据和保留代码 | [历史迁移](LEGACY.md) |
| CEM 冻结成绩与恢复逻辑 | [历史基线](RELEASE.md)、[停车脱困](RECOVERY.md) |
| 网站、讲稿、PPT 与共享媒体 | [发布材料](../publication/README.md) |

`docs/` 保存可维护的说明与结论；审核媒体统一放在 `publication/assets/`。`runs/` 和 `output/` 是忽略提交的本地产物，文档中的这类路径用于追溯，不表示公开仓库包含对应权重或原始录像。
