#!/usr/bin/env python3
"""Build a dependency-free public evidence site for GitHub Pages."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "_site"

ASSETS = {
    "assets/openusd-preview.png": "site/source_assets/openusd-preview.png",
    "assets/mujoco-preview.png": "site/source_assets/mujoco-preview.png",
    "assets/mujoco-race.png": "site/source_assets/mujoco-race.png",
    "assets/isaac-race.png": "site/source_assets/isaac-race.png",
    "assets/drift.png": "site/source_assets/drift.png",
    "assets/perturb.png": "site/source_assets/perturb.png",
    "media/openusd-usability.mp4": "site/source_assets/openusd-usability.mp4",
    "media/mujoco-usability.mp4": "site/source_assets/mujoco-usability.mp4",
    "media/mujoco-bahrain.mp4": "site/source_assets/mujoco-bahrain.mp4",
    "media/isaac-bahrain.mp4": "site/source_assets/isaac-bahrain.mp4",
    "downloads/osracer_live_briefing.pptx": "presentation/build/osracer_live_briefing.pptx",
    "documents/LIVE_BRIEFING.md": "docs/LIVE_BRIEFING.md",
    "documents/RELEASE.md": "docs/RELEASE.md",
    "documents/VALIDATION_STATUS.md": "docs/VALIDATION_STATUS.md",
    "documents/PROJECT_STRUCTURE.md": "docs/PROJECT_STRUCTURE.md",
    "documents/SPEAKER_NOTES.md": "presentation/SPEAKER_NOTES.md",
    "documents/MEDIA_MANIFEST.md": "presentation/MEDIA_MANIFEST.md",
}

HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OSRACER · 双引擎自主竞速证据库</title><style>
:root{--ink:#172b46;--blue:#2478be;--teal:#00998e;--orange:#d95a34;--paper:#faf9f6;--muted:#5d6975}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif;line-height:1.6}.wrap{max-width:1120px;margin:auto;padding:0 24px}.hero{padding:72px 0 52px;background:linear-gradient(135deg,#112b46,#2478be);color:#fff}.tag{letter-spacing:.13em;font-size:.78rem;font-weight:700;color:#86e2d8}.hero h1{font-size:clamp(2.2rem,5vw,4.4rem);line-height:1.08;margin:.35rem 0 1rem}.hero p{max-width:780px;font-size:1.2rem;opacity:.9}.nav{display:flex;gap:20px;flex-wrap:wrap;margin-top:24px}.nav a,.button{color:#fff;border:1px solid rgba(255,255,255,.55);padding:8px 14px;border-radius:999px;text-decoration:none}.section{padding:52px 0;border-bottom:1px solid #dde4e8}.section h2{font-size:2rem;line-height:1.2;margin:0 0 18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:18px}.card{background:#fff;border:1px solid #d7e0e6;border-radius:14px;padding:22px}.card h3{margin:0 0 8px;font-size:1.25rem}.metric{font-size:2.35rem;font-weight:800;line-height:1;color:var(--blue);margin:13px 0 8px}.ok{color:var(--teal)}.warn{color:var(--orange)}.no{color:#be343a}.media{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:22px}.media figure{margin:0;background:#fff;border:1px solid #d7e0e6;border-radius:14px;overflow:hidden}.media img,.media video{width:100%;display:block;background:#19232d}.media figcaption{padding:14px 16px}.button.dark{display:inline-block;background:var(--ink);border:0;margin:8px 8px 0 0}.facts{border-left:4px solid var(--orange);padding:4px 0 4px 18px;background:#fff7f3}.foot{padding:36px 0;color:var(--muted);font-size:.9rem}@media(max-width:600px){.hero{padding-top:50px}.section{padding:38px 0}}
</style></head><body>
<header class="hero"><div class="wrap"><div class="tag">EVIDENCE-LED AUTONOMOUS RACING</div><h1>OSRACER：哪些已经被证实？</h1><p>双引擎自主竞速的公开证据库。所有数字均保留引擎、赛道、速度、种子与失败边界；不以单段视频代替资格验收。</p><nav class="nav"><a href="#answers">两个问题</a><a href="#evidence">验证媒体</a><a href="#results">竞速结果</a><a href="#next">下一步</a></nav></div></header>
<main><section id="answers" class="section"><div class="wrap"><h2>两个问题的当前回答</h2><div class="grid"><article class="card"><h3>1 · 导出资产能否使用？</h3><div class="metric ok">可以</div><p>OpenUSD / Isaac Sim 与 MuJoCo 均完成加载、原生渲染、短程物理步进和可解码录像。</p><p class="facts">边界：这仅是资产层验收。碰撞回退、固定基座与执行器跟踪限制仍存在，不能据此宣称可靠驾驶。</p></article><article class="card"><h3>2 · Ackermann 小车能否漂移、超车、刷圈？</h3><div class="metric warn">分项成立</div><p>仿真中已有双车有效超车、MuJoCo 24/24 整圈与 9.0 m/s 速度榜，以及回头弯持续侧滑样例。</p><p class="facts">边界：单电机四驱实车等效动作没有后轮增速；当前漂移机制不能直接迁移。组合传感器扰动 10/10 失败。</p></article><article class="card"><h3>推荐直播材料</h3><div class="metric">12 页</div><p>可编辑中文 PPTX、逐页讲稿和媒体清单均已随工程提供。</p><a class="button dark" href="downloads/osracer_live_briefing.pptx">下载 PPTX</a><a class="button dark" href="documents/SPEAKER_NOTES.md">查看讲稿</a></article></div></div></section>
<section id="evidence" class="section"><div class="wrap"><h2>资产可用性：有录像，也有边界</h2><div class="media"><figure><video controls preload="metadata" poster="assets/openusd-preview.png"><source src="media/openusd-usability.mp4" type="video/mp4"></video><figcaption><strong>OpenUSD / Isaac Sim 6.0.1</strong><br>20 网格、10 刚体、9 物理关节、6 可动自由度；60 步短程 PhysX 运行。</figcaption></figure><figure><video controls preload="metadata" poster="assets/mujoco-preview.png"><source src="media/mujoco-usability.mp4" type="video/mp4"></video><figcaption><strong>MuJoCo 3.10.0</strong><br>robot.xml / scene.xml 各 500 步，无数值警告；原模型仍为固定基座。</figcaption></figure></div></div></section>
<section id="results" class="section"><div class="wrap"><h2>冻结 v10c 的竞速证据</h2><div class="grid"><article class="card"><h3>7.25 m/s · MuJoCo</h3><div class="metric ok">24/24</div><p>有效整圈与有效超车；24 段录像、24 份轨迹审计通过。</p></article><article class="card"><h3>7.25 m/s · Isaac</h3><div class="metric">22/24</div><p>Spa 起步车车接触、Suzuka 桥面失稳被保留为失败证据。</p></article><article class="card"><h3>9.0 m/s · MuJoCo</h3><div class="metric ok">8.97 m/s</div><p>24/24，所有 24 条赛道单圈均快于 7.25 m/s 基线。</p></article><article class="card"><h3>9.0 m/s · Isaac</h3><div class="metric warn">20/24</div><p>峰值 9.00–9.04 m/s；新增 Austin/Las Vegas 停滞，不能作为共享稳定配置。</p></article></div><div class="media" style="margin-top:22px"><figure><img src="assets/mujoco-race.png" alt="MuJoCo Bahrain 代表性整圈"><figcaption><strong>MuJoCo Bahrain</strong><br>代表性双车样例；完整计数口径见验证状态。</figcaption></figure><figure><img src="assets/isaac-race.png" alt="Isaac Bahrain 代表性整圈"><figcaption><strong>Isaac Bahrain</strong><br>代表性双车样例；不以一个样例外推 24 条赛道。</figcaption></figure></div></div></section>
<section class="section"><div class="wrap"><h2>漂移与主动观测：有实验，不能夸大</h2><div class="media"><figure><img src="assets/drift.png" alt="发卡弯侧滑截图"><figcaption><strong>回头弯漂移 A/B</strong><br>后轮增速 3.5 时最大侧滑 27.0°；设为 0 后仅 4.36°。结论是当前漂移动作依赖后轮增速。</figcaption></figure><figure><img src="assets/perturb.png" alt="传感器扰动失败截图"><figcaption><strong>组合传感器扰动失败</strong><br>0.02 m 噪声 + 5% 丢束 + 50 ms 延迟，5 赛道 × 2 种子为 10/10 失败。此项是未关闭的鲁棒性缺口。</figcaption></figure></div></div></section>
<section id="next" class="section"><div class="wrap"><h2>下一步不是重复展示，而是关闭具体缺口</h2><div class="grid"><article class="card"><h3>感知</h3><p>修复噪声下扫描匹配，结合降级感知限速与最小走廊宽度规则后重测。</p></article><article class="card"><h3>物理</h3><p>专项处理 Isaac Spa 起步接触与 Suzuka 桥接触；不把失败实验包装为修复。</p></article><article class="card"><h3>实车</h3><p>先标定/测量转向，再 shadow 回放、台架、低速逐级提速；若目标是漂移，需要改变动作硬件或重训行为。</p></article></div><p style="margin-top:24px"><a class="button dark" href="documents/RELEASE.md">冻结发布与复现命令</a><a class="button dark" href="documents/VALIDATION_STATUS.md">完整验证状态</a><a class="button dark" href="documents/PROJECT_STRUCTURE.md">项目结构</a><a class="button dark" href="documents/MEDIA_MANIFEST.md">媒体清单</a></p></div></section></main>
<footer class="foot"><div class="wrap">OSRACER · 静态站只展示已审核的精选证据。完整大媒体与机器索引请从仓库的 <code>output/racing/</code> 查阅。</div></footer></body></html>"""

def copy(relative_out: str, relative_source: str) -> None:
    source, destination = ROOT / relative_source, OUT / relative_out
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

def build() -> None:
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for destination, source in ASSETS.items(): copy(destination, source)
    (OUT / "index.html").write_text(HTML, encoding="utf-8")
    (OUT / ".nojekyll").touch()
    print(f"built {OUT} with {len(ASSETS)} tracked artifacts")

if __name__ == "__main__": build()
