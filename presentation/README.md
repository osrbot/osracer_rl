# OSRACER 直播材料

此目录是面向直播/答辩的可复现材料包。所有结论均从仓库中已有的验证数据、原生录像和独立审计中提取；它不把“能加载资产”写成“已具备可靠驾驶能力”，也不把局部漂移样例外推为实车可行性。

## 交付物

- `build/osracer_live_briefing.pptx`：16:9 中文可编辑演示稿（由 `build_presentation.py` 生成）。
- [讲稿](SPEAKER_NOTES.md)：13 页逐页直播话术、演示动作和不应越界的表述。
- [媒体清单](MEDIA_MANIFEST.md)：每个视频/截图的用途、结论和来源。
- `build_presentation.py`：不依赖私有素材的 PPTX 构建脚本。运行时需要 `python-pptx`。

## 重建

```bash
python3 -m pip install --target /tmp/osracer-pptx python-pptx
PYTHONPATH=/tmp/osracer-pptx python3 presentation/build_presentation.py
```

生成物为可编辑 PPTX；视频不嵌入文件，而是以仓库相对路径在“媒体清单”中列出，适合直播时从 `output/` 原生播放。这样避免重复提交大文件，也保持录像哈希和已有索引不变。

## 叙事纪律

1. **资产可用性**的范围是 OpenUSD/Isaac Sim 与 MuJoCo 的加载、渲染、短程步进和可解码录像。
2. **竞速能力**的范围必须带上引擎、赛道集、巡航速度、对手条件和种子；结果以 `docs/RELEASE.md` 为准。
3. **漂移结论**分开说：仿真中的发卡弯可观察到持续侧滑，但当前实车等效动作空间缺少后轮增速，A/B 结果显示该机制不能直接迁移。
