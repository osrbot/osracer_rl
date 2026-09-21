# OSRACER 技术讲解视频

build_demo_video.py 从项目中已审核的媒体、冻结结果和训练记录生成可直接播放的中文技术讲解片。它采用 Microsoft Edge 的 zh-CN-YunyangNeural 男声，并将中文字幕直接烧录进 MP4。为了避免播放器自动加载字幕导致重复显示，构建目录不会保留独立 SRT。

讲解结构包括实验契约与控制变量、资产可用性、SolidWorks URDF Exporter Pro 的导出能力与贡献入口、MuJoCo 策略迭代、全 24 条赛道的引擎对照、同一 v10c 配置下的速度控制变量、回头弯 A/B 与感知失败边界、复现实演步骤。

## 生成

    python3 -m venv /tmp/osracer-video-venv
    /tmp/osracer-video-venv/bin/pip install -r presentation/video/requirements.txt
    EDGE_TTS=/tmp/osracer-video-venv/bin/edge-tts python3 presentation/video/build_demo_video.py

输出位于 presentation/video/build/，该目录只存放可再生成的本地媒体，刻意不进入公共项目介绍或 Git 历史。

所有结论均按固定赛道集合、种子、引擎、巡航速度和验证器口径标注；影片不把单一视频片段或 MuJoCo 结果外推成 Isaac Sim 或实车结论。
