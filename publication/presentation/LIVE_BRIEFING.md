# OSRACER 项目介绍与演示指引

OSRACER 研究阿克曼小车怎样依靠轮速、转角和单线激光完成竞速。项目把车辆资产、MuJoCo / Isaac 两套原生仿真、策略训练和回合核验接在一起。演示的重点是让观众看清一条完整链路，也能找到对应命令自己复现。

## 演示材料

- [现有 PPTX](releases/osracer_live_briefing.pptx)：13 页，图表对应历史 CEM 与资产验证批次。
- [口述讲稿](SPEAKER_NOTES.md)：按同一页序重写，补充当前 PPO 进展和环境配置说明。
- [媒体清单](MEDIA_MANIFEST.md)：录像与截图的对应关系。
- [审核媒体目录](../assets/README.md)：网站与演示共用的文件。

PPT 中的 MuJoCo 24/24、Isaac 22/24 属于历史 CEM 控制器。当前 PPO 的 24 赛道迁移结果为 6/24 通过全部筛选种子。讲稿会在开场和结果页交代这一区别，详细口径见[验证状态](../../docs/VALIDATION_STATUS.md)。

## 讲清楚这条实验链路

车辆描述来自 [SolidWorks URDF Exporter Pro](https://github.com/osrbot/solidworks_urdf_exporter_pro)。它负责把链接、关节、坐标系和惯量等信息导出到 ROS、OpenUSD 或 MuJoCo 描述中。导出后，先在目标引擎检查加载、关节、碰撞和短时物理步进，再接控制器。

竞速策略只使用传感器观测。默认训练入口是 PPO，结构化 CEM 控制器仍用于对照和教师预热。训练结果按批次保存为 `.pt`、ONNX、TensorBoard 日志和评估记录。录像由保留模型重新运行生成，可以与对应回合的轨迹一起核对。

现有结果暴露了几条明确限制。历史 CEM 在两个引擎中的失败赛道不同；传感器组合扰动测试仍未通过；仿真里的独立后轮增速也不能直接映射到单电机四驱实车。讲解这些结果时，要同时报出策略、引擎和测试条件。对执行器的讨论见[部署说明](../../docs/DEPLOYMENT.md)，对 CEM 批次的追溯见[历史基线](../../docs/RELEASE.md)。

## 现场命令演示

提前按 [README 环境要求](../../README.zh-CN.md#安装前先确认环境)准备依赖和渲染器。项目安装脚本不安装驱动或 Isaac。现场优先展示已经完成的训练 run，训练预热和完整 24 赛道任务的耗时不适合作为固定时长演示。

```bash
source .venv/bin/activate
export MUJOCO_GL=egl

# demo 须是此前已完成训练、保留有效检查点的本地 run。
tensorboard --logdir runs/demo/metrics/tensorboard --port 6006
```

在另一个已激活环境的终端中运行 `osracer-play experiment_name=demo`。讲解时展示 run 内的模型、指标和视频各放在哪里。服务端加 `--no-open` 可以跳过播放器，但仍需渲染并编码视频。

如果现场尚无训练结果，先播放审核媒体，再展示训练命令与输出布局。说明审核片段所属的历史策略，不把它当作刚刚启动的 PPO 已训练完成。

## PPT 构建入口

```bash
python3 -m pip install --target /tmp/osracer-pptx python-pptx
PYTHONPATH=/tmp/osracer-pptx python3 publication/presentation/build_presentation.py
```

构建器使用仓库里的 PNG 和历史图表文本，不会重新录制实验。本轮更新的是项目文稿与口述稿，既有 PPTX 保留其历史批次身份。若要把新的 PPO 图表放进幻灯片，应先更新构建器中的数据与标题，再重新核对媒体清单和文件元数据。
