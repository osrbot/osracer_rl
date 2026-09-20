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

## 重建 PPTX

```bash
python3 -m pip install --target /tmp/osracer-pptx python-pptx
PYTHONPATH=/tmp/osracer-pptx python3 presentation/build_presentation.py
```

构建脚本会直接引用项目内已有 PNG，因此不重新编码、移动或覆盖录像。PPTX 中的图表文字可编辑；直播时建议从媒体清单打开原 MP4，而不是把大视频重复嵌入到幻灯片。
