#!/usr/bin/env python3
"""Build the bilingual, dependency-free OSRACER evidence site."""
from __future__ import annotations

import json
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
}

TRANSLATIONS = {
    "zh": {
        "page_title": "OSRACER · 双引擎自主竞速证据库",
        "tag": "EVIDENCE-LED AUTONOMOUS RACING",
        "hero_title": "OSRACER：哪些已经被证实？",
        "hero_body": "双引擎自主竞速的公开证据库。所有数字均保留引擎、赛道、速度、种子与失败边界；不以单段视频代替资格验收。",
        "nav_answers": "两个问题", "nav_evidence": "验证媒体", "nav_results": "竞速结果", "nav_next": "下一步",
        "answers_title": "两个问题的当前回答",
        "asset_title": "1 · 导出资产能否使用？", "asset_metric": "可以",
        "asset_body": "OpenUSD / Isaac Sim 与 MuJoCo 均完成加载、原生渲染、短程物理步进和可解码录像。",
        "asset_limit": "边界：这仅是资产层验收。碰撞回退、固定基座与执行器跟踪限制仍存在，不能据此宣称可靠驾驶。",
        "race_title": "2 · Ackermann 小车能否漂移、超车、刷圈？", "race_metric": "分项成立",
        "race_body": "仿真中已有双车有效超车、MuJoCo 24/24 整圈与 9.0 m/s 速度榜，以及回头弯持续侧滑样例。",
        "race_limit": "边界：单电机四驱实车等效动作没有后轮增速；当前漂移机制不能直接迁移。组合传感器扰动 10/10 失败。",
        "audit_title": "可复现证据", "audit_metric": "审计优先",
        "audit_body": "每项结论均回链到版本化配置、轨迹审计、原生媒体和公开的失败边界。",
        "evidence_title": "资产可用性：有录像，也有边界",
        "openusd_caption": "<strong>OpenUSD / Isaac Sim 6.0.1</strong><br>20 网格、10 刚体、9 物理关节、6 可动自由度；60 步短程 PhysX 运行。",
        "mujoco_caption": "<strong>MuJoCo 3.10.0</strong><br>robot.xml / scene.xml 各 500 步，无数值警告；原模型仍为固定基座。",
        "results_title": "冻结 v10c 的竞速证据",
        "mu725_title": "7.25 m/s · MuJoCo", "mu725_body": "有效整圈与有效超车；24 段录像、24 份轨迹审计通过。",
        "isaac725_title": "7.25 m/s · Isaac", "isaac725_body": "Spa 起步车车接触、Suzuka 桥面失稳被保留为失败证据。",
        "mu9_title": "9.0 m/s · MuJoCo", "mu9_body": "24/24，所有 24 条赛道单圈均快于 7.25 m/s 基线。",
        "isaac9_title": "9.0 m/s · Isaac", "isaac9_body": "峰值 9.00–9.04 m/s；新增 Austin/Las Vegas 停滞，不能作为共享稳定配置。",
        "mu_race_caption": "<strong>MuJoCo Bahrain</strong><br>代表性双车样例；完整计数口径见仓库中的验证台账。",
        "isaac_race_caption": "<strong>Isaac Bahrain</strong><br>代表性双车样例；不以一个样例外推 24 条赛道。",
        "drift_title": "漂移与主动观测：有实验，不能夸大",
        "drift_caption": "<strong>回头弯漂移 A/B</strong><br>后轮增速 3.5 时最大侧滑 27.0°；设为 0 后仅 4.36°。结论是当前漂移动作依赖后轮增速。",
        "perturb_caption": "<strong>组合传感器扰动失败</strong><br>0.02 m 噪声 + 5% 丢束 + 50 ms 延迟，5 赛道 × 2 种子为 10/10 失败。此项是未关闭的鲁棒性缺口。",
        "next_title": "下一步不是重复展示，而是关闭具体缺口",
        "perception_title": "感知", "perception_body": "修复噪声下扫描匹配，结合降级感知限速与最小走廊宽度规则后重测。",
        "physics_title": "物理", "physics_body": "专项处理 Isaac Spa 起步接触与 Suzuka 桥接触；不把失败实验包装为修复。",
        "real_title": "实车", "real_body": "先标定/测量转向，再 shadow 回放、台架、低速逐级提速；若目标是漂移，需要改变动作硬件或重训行为。",
        "footer": "OSRACER · 静态站只展示已审核的精选证据。完整大媒体与机器索引请从仓库 output/racing/ 查阅。",
        "alt_openusd": "OpenUSD 和 Isaac Sim 可用性验证封面",
        "alt_mujoco": "MuJoCo 3.10.0 可用性验证封面",
        "alt_mu_race": "MuJoCo Bahrain 代表性整圈",
        "alt_isaac_race": "Isaac Bahrain 代表性整圈",
        "alt_drift": "发卡弯侧滑截图", "alt_perturb": "传感器扰动失败截图",
    },
    "en": {
        "page_title": "OSRACER · Dual-Engine Autonomous Racing Evidence",
        "tag": "EVIDENCE-LED AUTONOMOUS RACING",
        "hero_title": "OSRACER: what has been demonstrated?",
        "hero_body": "A public evidence record for dual-engine autonomous racing. Every result keeps its engine, track, speed, seeds, and failure boundary; a single video is not treated as qualification.",
        "nav_answers": "Two questions", "nav_evidence": "Evidence media", "nav_results": "Racing results", "nav_next": "Next steps",
        "answers_title": "Current answers to two questions",
        "asset_title": "1 · Are the exported assets usable?", "asset_metric": "Yes",
        "asset_body": "OpenUSD / Isaac Sim and MuJoCo both completed loading, native rendering, short physics stepping, and decodable recording.",
        "asset_limit": "Boundary: this is asset-level qualification only. Collision fallback, a fixed base, and actuator-tracking limits remain; it does not establish reliable driving.",
        "race_title": "2 · Can an Ackermann car drift, overtake, and set lap times?", "race_metric": "Partly demonstrated",
        "race_body": "Simulation includes valid two-car overtakes, 24/24 MuJoCo laps and a 9.0 m/s speed board, plus a sustained sideslip hairpin example.",
        "race_limit": "Boundary: the single-motor AWD hardware-equivalent action has no rear-wheel boost, so the present drift mechanism does not transfer directly. Combined sensor perturbations fail 10/10.",
        "audit_title": "Reproducible evidence", "audit_metric": "Audit first",
        "audit_body": "Each conclusion links back to versioned configurations, trajectory audits, native media, and stated failure boundaries.",
        "evidence_title": "Asset usability: recordings, with boundaries",
        "openusd_caption": "<strong>OpenUSD / Isaac Sim 6.0.1</strong><br>20 meshes, 10 rigid bodies, 9 physics joints, and 6 movable degrees of freedom; a 60-step PhysX run.",
        "mujoco_caption": "<strong>MuJoCo 3.10.0</strong><br>500 steps each for robot.xml and scene.xml, with no numerical warnings; the original model remains fixed-base.",
        "results_title": "Frozen v10c racing evidence",
        "mu725_title": "7.25 m/s · MuJoCo", "mu725_body": "Valid laps and valid overtakes; all 24 recordings and 24 trajectory audits passed.",
        "isaac725_title": "7.25 m/s · Isaac", "isaac725_body": "Spa launch contact and Suzuka bridge instability are retained as failure evidence.",
        "mu9_title": "9.0 m/s · MuJoCo", "mu9_body": "24/24; every one of the 24 track laps is faster than the 7.25 m/s baseline.",
        "isaac9_title": "9.0 m/s · Isaac", "isaac9_body": "Peak 9.00–9.04 m/s; new Austin/Las Vegas stalls prevent calling this a shared stable configuration.",
        "mu_race_caption": "<strong>MuJoCo Bahrain</strong><br>A representative two-car case; see the validation ledger in the repository for the complete counting rule.",
        "isaac_race_caption": "<strong>Isaac Bahrain</strong><br>A representative two-car case; one example is not extrapolated to 24 tracks.",
        "drift_title": "Drift and active observation: experiments, not overclaims",
        "drift_caption": "<strong>Hairpin drift A/B</strong><br>Maximum sideslip is 27.0° with a 3.5 rear-wheel boost and only 4.36° with it set to zero. The current drift action depends on that boost.",
        "perturb_caption": "<strong>Combined sensor-perturbation failure</strong><br>0.02 m noise + 5% beam dropout + 50 ms delay: 10/10 failures across five tracks and two seeds. This robustness gap remains open.",
        "next_title": "The next task is closing specific gaps, not replaying demos",
        "perception_title": "Perception", "perception_body": "Repair scan matching under noise, then retest with degraded-perception speed limits and a minimum-corridor-width rule.",
        "physics_title": "Physics", "physics_body": "Address Isaac Spa launch contact and Suzuka bridge contact directly; failed experiments are not presented as repairs.",
        "real_title": "Physical vehicle", "real_body": "Calibrate and measure steering first, then progress through shadow replay, bench work, and low-speed steps. Drifting requires different action hardware or retrained behavior.",
        "footer": "OSRACER · This static site presents curated, reviewed evidence only. Find complete large media and the machine index in output/racing/ in the repository.",
        "alt_openusd": "OpenUSD and Isaac Sim usability-validation cover",
        "alt_mujoco": "MuJoCo 3.10.0 usability-validation cover",
        "alt_mu_race": "Representative MuJoCo Bahrain lap",
        "alt_isaac_race": "Representative Isaac Bahrain lap",
        "alt_drift": "Hairpin sideslip screenshot", "alt_perturb": "Sensor-perturbation failure screenshot",
    },
}

HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OSRACER</title><style>
:root{--ink:#172b46;--blue:#2478be;--teal:#00998e;--orange:#d95a34;--paper:#faf9f6;--muted:#5d6975}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif;line-height:1.6}.wrap{max-width:1120px;margin:auto;padding:0 24px}.hero{padding:66px 0 52px;background:linear-gradient(135deg,#112b46,#2478be);color:#fff}.tag{letter-spacing:.13em;font-size:.78rem;font-weight:700;color:#86e2d8}.hero h1{font-size:clamp(2.2rem,5vw,4.4rem);line-height:1.08;margin:.35rem 0 1rem}.hero p{max-width:780px;font-size:1.15rem;opacity:.92}.nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}.nav a,.button{color:#fff;border:1px solid rgba(255,255,255,.55);padding:8px 14px;border-radius:999px;text-decoration:none}.language{position:absolute;top:20px;right:24px;display:flex;gap:7px}.language button{cursor:pointer;color:#fff;background:transparent;border:1px solid rgba(255,255,255,.65);border-radius:999px;padding:6px 11px;font:inherit}.language button.active{background:#fff;color:var(--ink);font-weight:700}.section{padding:52px 0;border-bottom:1px solid #dde4e8}.section h2{font-size:2rem;line-height:1.2;margin:0 0 18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:18px}.card{background:#fff;border:1px solid #d7e0e6;border-radius:14px;padding:22px}.card h3{margin:0 0 8px;font-size:1.25rem}.metric{font-size:2.15rem;font-weight:800;line-height:1.1;color:var(--blue);margin:13px 0 8px}.ok{color:var(--teal)}.warn{color:var(--orange)}.media{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:22px}.media figure{margin:0;background:#fff;border:1px solid #d7e0e6;border-radius:14px;overflow:hidden}.media img,.media video{width:100%;display:block;background:#19232d}.media figcaption{padding:14px 16px}.facts{border-left:4px solid var(--orange);padding:4px 0 4px 18px;background:#fff7f3}.foot{padding:36px 0;color:var(--muted);font-size:.9rem}@media(max-width:600px){.hero{padding-top:68px}.section{padding:38px 0}.language{right:18px}.wrap{padding:0 18px}}
</style></head><body>
<header class="hero"><div class="language" aria-label="Language selector"><button data-lang="zh" type="button">中文</button><button data-lang="en" type="button">EN</button></div><div class="wrap"><div class="tag" data-i18n="tag"></div><h1 data-i18n="hero_title"></h1><p data-i18n="hero_body"></p><nav class="nav"><a href="#answers" data-i18n="nav_answers"></a><a href="#evidence" data-i18n="nav_evidence"></a><a href="#results" data-i18n="nav_results"></a><a href="#next" data-i18n="nav_next"></a></nav></div></header>
<main><section id="answers" class="section"><div class="wrap"><h2 data-i18n="answers_title"></h2><div class="grid"><article class="card"><h3 data-i18n="asset_title"></h3><div class="metric ok" data-i18n="asset_metric"></div><p data-i18n="asset_body"></p><p class="facts" data-i18n="asset_limit"></p></article><article class="card"><h3 data-i18n="race_title"></h3><div class="metric warn" data-i18n="race_metric"></div><p data-i18n="race_body"></p><p class="facts" data-i18n="race_limit"></p></article><article class="card"><h3 data-i18n="audit_title"></h3><div class="metric" data-i18n="audit_metric"></div><p data-i18n="audit_body"></p></article></div></div></section>
<section id="evidence" class="section"><div class="wrap"><h2 data-i18n="evidence_title"></h2><div class="media"><figure><video controls preload="metadata" poster="assets/openusd-preview.png"><source src="media/openusd-usability.mp4" type="video/mp4"></video><figcaption data-i18n-html="openusd_caption"></figcaption></figure><figure><video controls preload="metadata" poster="assets/mujoco-preview.png"><source src="media/mujoco-usability.mp4" type="video/mp4"></video><figcaption data-i18n-html="mujoco_caption"></figcaption></figure></div></div></section>
<section id="results" class="section"><div class="wrap"><h2 data-i18n="results_title"></h2><div class="grid"><article class="card"><h3 data-i18n="mu725_title"></h3><div class="metric ok">24/24</div><p data-i18n="mu725_body"></p></article><article class="card"><h3 data-i18n="isaac725_title"></h3><div class="metric">22/24</div><p data-i18n="isaac725_body"></p></article><article class="card"><h3 data-i18n="mu9_title"></h3><div class="metric ok">8.97 m/s</div><p data-i18n="mu9_body"></p></article><article class="card"><h3 data-i18n="isaac9_title"></h3><div class="metric warn">20/24</div><p data-i18n="isaac9_body"></p></article></div><div class="media" style="margin-top:22px"><figure><img src="assets/mujoco-race.png" data-i18n-attr="alt:alt_mu_race"><figcaption data-i18n-html="mu_race_caption"></figcaption></figure><figure><img src="assets/isaac-race.png" data-i18n-attr="alt:alt_isaac_race"><figcaption data-i18n-html="isaac_race_caption"></figcaption></figure></div></div></section>
<section class="section"><div class="wrap"><h2 data-i18n="drift_title"></h2><div class="media"><figure><img src="assets/drift.png" data-i18n-attr="alt:alt_drift"><figcaption data-i18n-html="drift_caption"></figcaption></figure><figure><img src="assets/perturb.png" data-i18n-attr="alt:alt_perturb"><figcaption data-i18n-html="perturb_caption"></figcaption></figure></div></div></section>
<section id="next" class="section"><div class="wrap"><h2 data-i18n="next_title"></h2><div class="grid"><article class="card"><h3 data-i18n="perception_title"></h3><p data-i18n="perception_body"></p></article><article class="card"><h3 data-i18n="physics_title"></h3><p data-i18n="physics_body"></p></article><article class="card"><h3 data-i18n="real_title"></h3><p data-i18n="real_body"></p></article></div></div></section></main>
<footer class="foot"><div class="wrap" data-i18n="footer"></div></footer>
<script>const translations=__TRANSLATIONS__;function setLanguage(lang){const locale=translations[lang]||translations.zh;document.documentElement.lang=lang==="en"?"en":"zh-CN";document.title=locale.page_title;document.querySelectorAll("[data-i18n]").forEach(node=>node.textContent=locale[node.dataset.i18n]);document.querySelectorAll("[data-i18n-html]").forEach(node=>node.innerHTML=locale[node.dataset.i18nHtml]);document.querySelectorAll("[data-i18n-attr]").forEach(node=>{const [attribute,key]=node.dataset.i18nAttr.split(":");node.setAttribute(attribute,locale[key])});document.querySelectorAll("[data-lang]").forEach(button=>button.classList.toggle("active",button.dataset.lang===lang));const url=new URL(location.href);url.searchParams.set("lang",lang);history.replaceState(null,"",url);localStorage.setItem("osracer-language",lang)}const requested=new URLSearchParams(location.search).get("lang");const saved=localStorage.getItem("osracer-language");setLanguage(requested==="en"||requested==="zh"?requested:(saved==="en"||saved==="zh"?saved:(navigator.language.startsWith("zh")?"zh":"en")));document.querySelectorAll("[data-lang]").forEach(button=>button.addEventListener("click",()=>setLanguage(button.dataset.lang)));</script>
</body></html>"""


def copy(relative_out: str, relative_source: str) -> None:
    source, destination = ROOT / relative_source, OUT / relative_out
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for destination, source in ASSETS.items():
        copy(destination, source)
    content = HTML.replace("__TRANSLATIONS__", json.dumps(TRANSLATIONS, ensure_ascii=False))
    (OUT / "index.html").write_text(content, encoding="utf-8")
    (OUT / ".nojekyll").touch()
    print(f"built {OUT} with {len(ASSETS)} public artifacts and zh/en i18n")


if __name__ == "__main__":
    build()
