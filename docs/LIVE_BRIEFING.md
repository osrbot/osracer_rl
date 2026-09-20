# 直播 / 答辩材料

本项目的公开演示不以单段“跑起来”的视频代替验证。请从下列顺序讲解，并在每个结论后保留其引擎、速度、赛道集合、种子与失败边界。

## 直接打开

- [可编辑 PPTX](../presentation/build/osracer_live_briefing.pptx)
- [逐页讲稿](../presentation/SPEAKER_NOTES.md)
- [视频与截图清单](../presentation/MEDIA_MANIFEST.md)
- [资产可用性录像](../output/usability/README.md)
- [完整媒体库](../output/racing/media.html)

## 两个问题的当前回答

| 问题 | 当前回答 | 证据与边界 |
| --- | --- | --- |
| SolidWorks URDF Exporter Pro 导出的 OpenUSD/MuJoCo 资产是否在对应平台可用？ | **是，限于资产层。** 两套资产均已在 Isaac Sim 6.0.1 / MuJoCo 3.10.0 加载、渲染、短程步进并录制可解码 MP4。 | OpenUSD 的动态三角网格碰撞会回退凸包；MuJoCo 原资产为固定基座且执行器独立跟踪未可靠。这不是车辆可靠驾驶验收。 |
| 受限主动观测与宽松物理约束能否实现 180° 漂移、主动超车和最短圈速？ | **分项成立。** 仿真中已有发卡弯持续侧滑、双车有效超车、以及 MuJoCo 9.0 m/s 的 24/24 速度榜。 | 当前实车等效单电机四驱没有后轮增速；A/B 将侧滑从 27.0° 降到 4.36°，不能声称漂移可直接上车。Isaac 9.0 m/s 仅 20/24；组合传感器扰动 10/10 失败。 |

冻结配置、验收口径和复现命令见 [RELEASE.md](RELEASE.md)；逐项成功/失败证据见 [VALIDATION_STATUS.md](VALIDATION_STATUS.md)。

## 导出工具介绍：SolidWorks URDF Exporter Pro

车辆资产来自 [osrbot/solidworks_urdf_exporter_pro](https://github.com/osrbot/solidworks_urdf_exporter_pro)。它是从 SolidWorks 装配体出发的机器人描述导出工作流，适合需要将 CAD 资产带入 ROS、OpenUSD / Isaac Sim 或 MuJoCo 的开发者与研究者。

讲解时建议先说明：导出工具减少的是链接树、坐标系、关节、质量/惯量、碰撞与外观配置在多个目标之间重复维护的工作；它**不会**从 CAD 自动推断控制器、轮胎摩擦、PID 或任务参数，目标仿真器中的验证仍不可省略。

| 能力 | 对仿真实验的价值 |
| --- | --- |
| ROS 1 / ROS 2 URDF 包导出 | 保留网格、配置和导出报告，便于机器人描述、可视化及后续控制接入。 |
| OpenUSD 导出 | 生成 `robot.usd`、几何、名称映射和报告，可进入 Isaac Sim 或其他 USD 工作流。 |
| MuJoCo MJCF 导出 | 生成 `robot.xml`、`scene.xml`、网格和报告，便于搭建动力学场景与控制实验。 |
| 关节、坐标系、质量与惯量检查 | 在交付前集中审查方向、单位、质心、惯量张量和主惯量，降低 CAD 到仿真的隐蔽错误。 |
| 碰撞与外观配置 | 支持原语、组件包围盒、凸包、简化网格与预览；可在精度、碰撞稳定性与文件体积之间作有意识的取舍。 |
| 分区/简化 STL 与文件大小反馈 | 新版提供可视与碰撞网格的简化目标、实测大小反馈及受保护几何提示，便于控制大装配体导出成本。 |
| 共享命名与元数据、迁移支持 | 新版改善输出命名、元数据持久化和旧配置迁移；失败目标可保留成功输出并给出独立诊断。 |

如果你正在导出 **OpenUSD、MJCF 或 URDF** 来进行仿真实验和模拟，欢迎试用该工具。也欢迎提交更好的使用建议和 [Issue](https://github.com/osrbot/solidworks_urdf_exporter_pro/issues)，或贡献代码、测试与文档，帮助它服务更多开发者。反馈请尽量附上 SolidWorks 版本、导出器版本、装配体复现步骤、完整错误文本与导出报告。

## 重建 PPTX

```bash
python3 -m pip install --target /tmp/osracer-pptx python-pptx
PYTHONPATH=/tmp/osracer-pptx python3 presentation/build_presentation.py
```

构建脚本会直接引用项目内已有 PNG，因此不重新编码、移动或覆盖录像。PPTX 中的图表文字可编辑；直播时建议从媒体清单打开原 MP4，而不是把大视频重复嵌入到幻灯片。
