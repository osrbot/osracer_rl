"""Generate a local, source-backed Chinese race report; no network or simulation.

python -m racing.report --tag baseline_v5 --checkpoint output/racing/baseline_v5_policy.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
import shlex
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("isaac", "mujoco")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def fmt(value, places=2):
    value = finite(value)
    return "—" if value is None else f"{value:.{places}f}"


def href(path, output):
    return quote(os.path.relpath(Path(path), output).replace(os.sep, "/"), safe="/._-")


def resolve(path, artifact, root):
    if not path:
        return None
    path = Path(path)
    candidates = [path] if path.is_absolute() else [root/path, artifact.parent/path, artifact.parent/path.name]
    return next((p for p in candidates if p.is_file()), None)


def collect(root=ROOT, tag="baseline_v5", checkpoint=None):
    root = Path(root)
    tags = list(dict.fromkeys([tag] if isinstance(tag,str) else tag))
    if not tags:
        raise ValueError("At least one report tag is required")
    output = root/"output/racing"
    catalog = read(root/"tracks/catalog.json")
    # Presentation consumes current geometry metadata rather than circuit-name exceptions.
    for tid,item in catalog["tracks"].items():
        path = root/"tracks"/tid/"track.json"
        if path.is_file():
            meta = read(path)
            suitability = meta.get("training_suitability",{})
            item["has_elevation"] = bool(meta.get("has_elevation"))
            item["requires_3d_backend"] = bool(suitability.get("requires_3d_backend"))
            item["scored_racing_supported"] = suitability.get("scored_racing_supported",item.get("scored_racing_supported"))
            item["bridge"] = meta.get("bridge")
            bridge_doc = path.parent/"BRIDGE.md"
            item["bridge_doc"] = href(bridge_doc,output) if bridge_doc.is_file() else None
            full_path = path.parent/"full_scale/track.json"
            if full_path.is_file():
                full = read(full_path)
                item["full_scale_supported"] = full.get("training_suitability",{}).get("scored_racing_supported")
                item["full_scale_has_elevation"] = bool(full.get("has_elevation"))
    rows, warnings, source_files = [], [], []
    checkpoint_cache = {}
    track_hashes = {}
    override = Path(checkpoint).resolve() if checkpoint else None
    if override and not override.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {override}")
    manifests = []
    for selected_tag in tags:
        manifest_path = output/f"{selected_tag}_manifest.json"
        if manifest_path.is_file():
            try:
                raw_manifest = manifest_path.read_bytes()
                manifest = json.loads(raw_manifest)
                manifest["report_source_sha256"] = hashlib.sha256(raw_manifest).hexdigest()
                manifest["report_source_path"] = href(manifest_path,output)
                manifests.append(manifest)
            except (ValueError,OSError):
                warnings.append(f"{selected_tag} 的 manifest 正在写入或不可读；回合结果仍单独读取。")
    for engine in ENGINES:
        artifacts = {p for selected_tag in tags for p in (output/engine).glob(f"*_{selected_tag}.json")}
        for artifact in sorted(artifacts):
            artifact_tag = next(t for t in sorted(tags,key=len,reverse=True) if artifact.stem.endswith("_"+t))
            try:
                # Read a single snapshot: a running experiment may replace its file.
                raw = artifact.read_bytes()
                source_digest = hashlib.sha256(raw).hexdigest()
                episodes = json.loads(raw)
                if not isinstance(episodes, list):
                    warnings.append(f"跳过非回合列表：{artifact.name}")
                    continue
            except (OSError, ValueError) as exc:
                warnings.append(f"文件尚不可读：{artifact.name} ({type(exc).__name__})")
                continue
            source_files.append({"path":href(artifact,output),"sha256":source_digest,
                                 "modified":datetime.fromtimestamp(artifact.stat().st_mtime,timezone.utc).isoformat()})
            verification_path = artifact.with_name(artifact.stem+"_verification.json")
            verification, verification_problem = None, None
            if verification_path.is_file():
                try:
                    verification = read(verification_path)
                    if verification.get("artifact_sha256") != source_digest:
                        verification_problem = "审计文件已过期：结果哈希不匹配"
                        verification = None
                except (OSError, ValueError):
                    verification_problem = "审计文件不可读"
            for episode in episodes:
                if not isinstance(episode,dict) or not episode.get("track"):
                    warnings.append(f"跳过缺少赛道标识的记录：{artifact.name}")
                    continue
                seed = episode.get("seed")
                matches = [v for v in (verification or {}).get("episodes",[]) if v.get("seed")==seed]
                audit = matches[0] if len(matches)==1 else None
                if audit is None:
                    verification_state = verification_problem or ("审计回合缺失或种子重复" if verification else "未独立验证")
                    integrity = None
                else:
                    integrity = bool(audit.get("evidence_integrity_passed"))
                    verification_state = "证据验证通过" if integrity else "证据验证失败"
                audit_errors = list((audit or {}).get("errors",[]))
                tid = episode["track"]
                track_path = root/"tracks"/tid/"track.json"
                if tid not in track_hashes:
                    track_hashes[tid] = sha(track_path) if track_path.is_file() else None
                recorded_track_hash = episode.get("track_sha256")
                geometry_changed = bool(recorded_track_hash and track_hashes[tid] and recorded_track_hash != track_hashes[tid])
                if geometry_changed:
                    verification_state += "；对应旧赛道几何"
                    audit_errors.append("结果中的赛道 SHA 与当前资产不同，旧结果不构成当前布局验收")
                cp_path = override or resolve(episode.get("checkpoint_path"),artifact,root)
                cp_data = {}
                cp_actual = None
                if cp_path:
                    if str(cp_path) not in checkpoint_cache:
                        try:
                            checkpoint_cache[str(cp_path)] = (read(cp_path),sha(cp_path))
                        except (OSError,ValueError):
                            checkpoint_cache[str(cp_path)] = ({},None)
                    cp_data,cp_actual = checkpoint_cache[str(cp_path)]
                cp_saved = episode.get("checkpoint_sha256")
                algorithm = str(cp_data.get("algorithm",episode.get("algorithm","")))
                if "untrained" in algorithm.lower() or ("baseline" in algorithm.lower() and not cp_data.get("trained_in")):
                    training_kind = "未训练反馈基线"
                elif cp_data.get("adapted_from_checkpoint_sha256"):
                    training_kind = "训练参数适配候选"
                elif "cem" in algorithm.lower():
                    training_kind = "CEM 训练策略"
                elif "experimental" in algorithm.lower() or "search" in algorithm.lower():
                    training_kind = "实验性搜索候选"
                else:
                    training_kind = "训练来源未注明"
                cp_matches = cp_saved == cp_actual if cp_saved and cp_actual else None
                if cp_matches is False:
                    audit_errors.append("当前指定/可用检查点与记录 SHA-256 不同")
                source_version = episode.get("checkpoint_policy_version",cp_data.get("policy_version"))
                version = episode.get("policy_version")
                source_hash = cp_data.get("policy_source_sha256")
                used_hash = episode.get("policy_source_sha256")
                different_version = source_version is not None and version is not None and source_version != version
                different_source = bool(source_hash and used_hash and source_hash != used_hash)
                prior_only = bool(episode.get("parameter_only_migration") or different_version or different_source)
                if prior_only:
                    migration = "仅参数先验迁移：策略版本或源码已改变"
                elif cp_matches is False:
                    migration = "检查点不匹配，无法确认迁移身份"
                elif source_version is not None and version is not None and cp_matches is True and source_hash and used_hash:
                    migration = "检查点与策略源码一致"
                else:
                    migration = "迁移身份信息不完整"
                metric = episode.get("metric_audit") or {}
                independent = (audit or {}).get("recomputed") or {}
                # Summary values are never silently replaced with sparse-trace estimates.
                overtakes = episode.get("effective_overtakes",metric.get("effective_overtakes"))
                lap = finite(episode.get("lap_time_s")) if episode.get("valid_lap") else None
                valid_verified = bool(integrity and (audit or {}).get("task_success") and not geometry_changed)
                video = episode.get("video") or {}
                video_path = resolve(video.get("path"),artifact,root)
                frame_path = video_path.with_suffix(".png") if video_path else None
                trace = artifact.with_name(f"{artifact.stem}_seed{seed}_trace.json")
                row = {"engine":engine,"track":episode["track"],"seed":seed,"tag":artifact_tag,
                    "valid_lap":episode.get("valid_lap") is True,"verified_lap":valid_verified,
                    "integrity":integrity,"verification":verification_state,
                    "lap_time":lap,"peak_speed":finite(episode.get("peak_speed_m_s")),
                    "mean_speed":finite(episode.get("mean_speed_m_s")),"overtakes":finite(overtakes),
                    "drift":finite(episode.get("drift_duration_s")),
                    "drift_contiguous":finite(independent.get("longest_contiguous_drift_s")),
                    "failure":episode.get("failure") or ("未报告" if not episode.get("valid_lap") else "—"),
                    "policy_version":version,"checkpoint_version":source_version,
                    "checkpoint_sha256":cp_saved,"checkpoint_file_matches":cp_matches,
                    "parameters_sha256":episode.get("parameters_sha256"),"policy_source_sha256":used_hash,
                    "migration":migration,"prior_only":prior_only,"errors":audit_errors,
                    "training_kind":training_kind,"algorithm":algorithm,"trained_in":cp_data.get("trained_in"),
                    "geometry_changed":geometry_changed,"track_sha256":recorded_track_hash,
                    "limitations":(audit or {}).get("limitations",[]),
                    "source":href(artifact,output),"source_sha256":source_digest,
                    "trace":href(trace,output) if trace.is_file() else None,
                    "audit":href(verification_path,output) if verification_path.is_file() else None,
                    "video":href(video_path,output) if video_path else None,
                    "frame":href(frame_path,output) if frame_path and frame_path.is_file() else None,
                    "video_verified":bool(integrity and (audit or {}).get("video",{}).get("full_decode_passed")),
                    "sensor":episode.get("sensor",{})}
                rows.append(row)
    stats = {"episodes":len(rows),"reported_valid":sum(r["valid_lap"] for r in rows),
        "integrity_passed":sum(r["integrity"] is True for r in rows),
        "integrity_failed":sum(r["integrity"] is False for r in rows),
        "unverified":sum(r["integrity"] is None for r in rows),
        "verified_valid":sum(r["verified_lap"] for r in rows),
        "prior_only":sum(r["prior_only"] for r in rows),
        "geometry_changed":sum(r["geometry_changed"] for r in rows)}
    stats["untrained_baseline"] = sum(r["training_kind"]=="未训练反馈基线" for r in rows)
    stats["cem_trained"] = sum(r["training_kind"]=="CEM 训练策略" for r in rows)
    manifest_states = {}
    for manifest in manifests:
        for record in manifest.get("results",[]):
            state = str(record.get("status","unknown"))
            manifest_states[state] = manifest_states.get(state,0)+1
    coverage = {engine:{"observed":sorted({r["track"] for r in rows if r["engine"]==engine}),
        "missing":sorted(set(catalog["tracks"])-{r["track"] for r in rows if r["engine"]==engine})} for engine in ENGINES}
    identity = []
    for tid in catalog["tracks"]:
        left = [r for r in rows if r["engine"]=="isaac" and r["track"]==tid]
        right = [r for r in rows if r["engine"]=="mujoco" and r["track"]==tid]
        if not left or not right:
            continue
        checkpoints = {r["checkpoint_sha256"] for r in left+right}
        sources = {r["policy_source_sha256"] for r in left+right}
        versions = {r["policy_version"] for r in left+right}
        if None in checkpoints:
            label = "缺少检查点 SHA，不能确认相同检查点"
        elif len(checkpoints)!=1:
            label = "两引擎检查点不同"
        elif any(r["prior_only"] for r in left+right) or len(versions)!=1 or len(sources)!=1:
            label = "同检查点参数，策略版本/源码不同；仅先验迁移"
        elif None in sources or None in versions:
            label = "同检查点 SHA，但缺少策略版本/源码信息"
        else:
            label = "记录的检查点、策略版本与源码 SHA 一致；条件匹配仍需单独审计"
        identity.append({"track":tid,"status":label})
    command = "python3 -m racing.report " + " ".join("--tag "+shlex.quote(t) for t in tags)
    if checkpoint:
        command += " --checkpoint "+shlex.quote(str(checkpoint))
    return {"tag":" + ".join(tags),"tags":tags,"reproduce_command":command,
            "generated_at":datetime.now(timezone.utc).isoformat(),
            "evidence_latest_modified":max((s["modified"] for s in source_files),default=None),
            "rows":rows,"stats":stats,"coverage":coverage,"identity":identity,
            "sources":source_files,"warnings":warnings,"catalog":catalog,
            "manifests":manifests,"manifest_states":manifest_states}


CSS = """
:root{color-scheme:light;--ink:#142c37;--muted:#5c707a;--line:#dbe4e5;--accent:#08766e;--paper:#f4f7f6}
*{box-sizing:border-box}body{margin:0;background:var(--paper);font:15px/1.65 system-ui,-apple-system,'Noto Sans CJK SC',sans-serif;color:var(--ink)}
header{background:#102e38;color:white;padding:44px max(5vw,20px)}header small{letter-spacing:.15em;color:#a9d3ce}h1{font-size:clamp(26px,4vw,42px);line-height:1.2;margin:14px 0}header p{max-width:960px;color:#d0dedf}main{max-width:1520px;margin:auto;padding:28px max(3vw,18px) 60px}
h2{font-size:24px;margin:0 0 12px}h3{margin:8px 0}section{margin:0 0 30px}.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:24px}.muted,small{color:var(--muted)}a{color:var(--accent);text-underline-offset:3px}header a{color:#9ce3d7}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px}.stat{border-top:3px solid var(--accent)}.stat strong{display:block;font-size:34px}.stat span{color:var(--muted)}.notice{border-left:4px solid #d29522;background:#fff9e9;padding:14px 20px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0;align-items:center}select,input,button{font:inherit;border:1px solid #c9d6d8;border-radius:7px;padding:7px 10px;background:white;color:var(--ink)}button{cursor:pointer}input{min-width:220px}label{display:flex;gap:7px;align-items:center}.scroll{overflow:auto;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap;background:white}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}th{background:#edf3f2}th button{border:0;background:transparent;padding:0;font-weight:700}td.num{text-align:right;font-variant-numeric:tabular-nums}tbody tr:hover{background:#f1f8f6}.good{color:#08766e;font-weight:650}.bad{color:#a83926}.pending{color:#8b6722}.maps,.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:18px}.map,.movie{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:white}.map img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover;background:#182521}.map .body,.movie .body{padding:14px}.map p{margin:4px 0}.links{display:flex;gap:12px;flex-wrap:wrap;font-size:13px}video{display:block;width:100%;background:#102e38;aspect-ratio:16/9}.pill{font-size:12px;padding:2px 7px;border-radius:10px;background:#eaf1f0;display:inline-block}.fail{background:#fcebe5}details{margin:12px 0}summary{cursor:pointer;font-weight:600}code{font-size:12px;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#edf3f2;padding:14px;border-radius:8px}.empty{padding:24px;text-align:center;color:var(--muted)}footer{border-top:1px solid var(--line);padding-top:16px;font-size:13px}.sr{position:absolute;clip:rect(0,0,0,0)}@media(max-width:600px){.panel{padding:16px}header{padding:30px 20px}.cards{grid-template-columns:1fr 1fr}}
"""

JS = """
const D=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id), esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,n=2)=>typeof v==='number'?v.toFixed(n):'—';let sortKey='track',sortDir=1;
function rows(){let data=D.rows.filter(r=>(!$('engine').value||r.engine===$('engine').value)&&(!$('status').value||($('status').value==='verified'?r.verified_lap:$('status').value==='failed'?!r.valid_lap:r.integrity===null))&&(r.track+' '+r.engine+' '+r.tag).toLowerCase().includes($('query').value.toLowerCase()));
data.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];if(x==null)return 1;if(y==null)return -1;return sortDir*(typeof x==='number'?x-y:String(x).localeCompare(String(y)))});
$('row-count').textContent=`显示 ${data.length} / ${D.rows.length} 个回合`;
$('results').innerHTML=data.map(r=>`<tr><td>${esc(r.engine)}</td><td>${esc(r.tag)}</td><td>${esc(r.track)}</td><td>${esc(r.seed)}</td><td class="${r.valid_lap?'good':'bad'}">${r.valid_lap?'有效圈（记录）':'未完成有效圈'}</td><td class="${r.integrity===true?'good':r.integrity===false?'bad':'pending'}">${esc(r.verification)}</td><td class="num">${num(r.lap_time)}</td><td class="num">${num(r.peak_speed)}</td><td class="num">${num(r.mean_speed)}</td><td class="num">${num(r.overtakes,0)}</td><td class="num">${num(r.drift,3)}</td><td>${esc(r.failure)}</td><td title="${esc(r.migration)}">${esc(r.policy_version)}</td><td><a href="${esc(r.source)}">结果</a> ${r.trace?`<a href="${esc(r.trace)}">轨迹</a>`:''} ${r.audit?`<a href="${esc(r.audit)}">审计</a>`:''}</td></tr>`).join('')||'<tr><td colspan="14" class="empty">所选条件没有结果；缺失值没有替换为零。</td></tr>';
}
function maps(){let year=$('season').value;let rounds=D.catalog.seasons[year];$('track-maps').innerHTML=rounds.map(r=>{let t=D.catalog.tracks[r.track_id],id=encodeURIComponent(r.track_id);return `<article class="map"><a href="../../tracks/${id}/map.png"><img loading="lazy" src="../../tracks/${id}/map.png" alt="${esc(t.name)} 近似赛道地图"></a><div class="body"><span class="pill">${year} · R${r.round} · ${r.date}</span><h3>${esc(t.name)}</h3><p class="muted">${esc(r.track_id)} · 路径 ×${t.centerline_scale??0.05} · 路宽 1.5 m</p>${t.requires_3d_backend?'<p class="good">RC 已物理分层，须使用三维后端</p>':t.scored_racing_supported===false?'<p class="bad">元数据标记不支持计分竞速</p>':''}${t.full_scale_supported===false?'<p class="muted">原尺度版本为平面参考，未具备分层竞速条件</p>':''}<div class="links"><a href="../../tracks/${id}/track.dae">RC DAE</a><a href="../../tracks/${id}/track.obj">OBJ</a><a href="../../tracks/${id}/full_scale/track.dae">原尺度 DAE</a><a href="../../tracks/${id}/textures/asphalt.png">纹理</a><a href="../../tracks/${id}/track.json">元数据</a>${t.bridge_doc?`<a href="${esc(t.bridge_doc)}">桥梁说明</a>`:''}</div></div></article>`}).join('')}
for(const id of ['engine','status','query'])$(id).addEventListener(id==='query'?'input':'change',rows);
document.querySelectorAll('[data-sort]').forEach(b=>b.addEventListener('click',()=>{let k=b.dataset.sort;sortDir=sortKey===k?-sortDir:1;sortKey=k;rows()}));
$('season').addEventListener('change',maps);rows();maps();
"""


def render_html(data):
    e = html.escape
    stats = data["stats"]
    headline = f"已记录 {stats['episodes']} 个回合，独立确认 {stats['verified_valid']} 个有效圈"
    cards = [(stats["episodes"],"已记录回合"),(stats["reported_valid"],"运行记录称有效圈"),
             (stats["integrity_passed"],"证据验证通过"),(stats["unverified"],"尚未独立验证"),
             (stats["integrity_failed"],"证据验证失败")]
    coverage = "".join(f"<p><strong>{engine}</strong>：覆盖 {len(c['observed'])}/24 条赛道。"
        f"未记录：{e(', '.join(c['missing']) or '无')}。</p>" for engine,c in data["coverage"].items())
    if data.get("manifests"):
        manifest_links = " · ".join(f'<a href="{e(m["report_source_path"])}">{e(m.get("tag","manifest"))} manifest</a>' for m in data["manifests"])
        manifest_text = (f'<p>批量验收状态快照：{e(json.dumps(data["manifest_states"],ensure_ascii=False))}。'
            f'{manifest_links}。'
            'evaluated 表示任务执行并进入审计，不等于有效圈或全部目标通过。</p>'
            '<p>结果表独立读取所有所选标签的回合 JSON；没有 manifest 的代表赛道结果仍纳入。</p>')
    else:
        manifest_text = '<p class="muted">本标签尚无可读取的批量验收 manifest。</p>'
    identities = "".join(f"<li>{e(i['track'])}：{e(i['status'])}</li>" for i in data["identity"])
    if not identities:
        identities = "<li>本标签暂无同赛道的双引擎结果，不能据此确认跨引擎迁移。</li>"
    rows_detail = "".join(f"<details><summary>{e(r['engine'])} / {e(r['tag'])} / {e(r['track'])} / seed {e(str(r['seed']))} · {e(r['migration'])}</summary>"
        f"<p>策略类别：{e(r['training_kind'])}；训练引擎：{e(str(r['trained_in'] or '无/未注明'))}。</p><p>记录版本：{e(str(r['policy_version']))}；检查点版本：{e(str(r['checkpoint_version']))}</p>"
        f"<p>检查点 SHA：<code>{e(r['checkpoint_sha256'] or '未提供')}</code></p>"
        f"<p>策略源码 SHA：<code>{e(r['policy_source_sha256'] or '未提供')}</code></p>"
        f"<p>证据状态：{e(r['verification'])}；独立有效圈：{'是' if r['verified_lap'] else '未确认'}。</p>"
        f"<p>{e('；'.join(r['errors']) or '该记录没有报告审计错误；缺少审计时不等于通过。')}</p></details>" for r in data["rows"])
    videos = []
    for r in data["rows"]:
        if not r["video"]:
            continue
        poster = f' poster="{e(r["frame"])}"' if r["frame"] else ""
        videos.append(f'<article class="movie"><video controls preload="metadata"{poster} src="{e(r["video"])}"></video>'
            f'<div class="body"><h3>{e(r["engine"])} · {e(r["track"])} · seed {e(str(r["seed"]))}</h3>'
            f'<p>{"独立解码检查通过" if r["video_verified"] else "录像已存在，独立解码尚未确认"}</p>'
            f'<div class="links"><a href="{e(r["video"])}">视频文件</a>'
            +(f'<a href="{e(r["frame"])}">原生渲染帧</a>' if r["frame"] else '')+'</div></div></article>')
    headings = [("engine","引擎"),("tag","标签"),("track","赛道"),("seed","种子"),("valid_lap","圈状态"),("integrity","证据"),
        ("lap_time","圈时 s"),("peak_speed","峰值 m/s"),("mean_speed","均速 m/s"),("overtakes","有效超车"),
        ("drift","侧滑累计 s"),("failure","失败原因"),("policy_version","策略版本")]
    th = "".join(f'<th><button data-sort="{k}" aria-label="按{v}排序">{v} ↕</button></th>' for k,v in headings)
    sources = "".join(f'<li><a href="{e(s["path"])}">{e(s["path"])}</a> · <code>{s["sha256"]}</code></li>' for s in data["sources"])
    payload = json.dumps(data,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>OSRACER 阶段性竞速验收</title><style>{CSS}</style></head>
<body><header><small>OSRACER / NATIVE PHYSICS / EVIDENCE</small><h1>{headline}</h1><p>标签 <strong>{e(data['tag'])}</strong> · 当前为阶段性验收，尚无全部竞速目标通过的证据。有效圈、超车和持续侧滑分别衡量，不把资产覆盖或未测试项目记作成功。</p><a href="RESULTS.md">下载 Markdown 报告</a> · <a href="../../docs/ENGINEERING.md">工程说明</a></header><main>
<section class="cards">{''.join(f'<div class="panel stat"><strong>{n}</strong><span>{label}</span></div>' for n,label in cards)}</section>
<section class="notice"><strong>结果解读</strong><p>策略类别由检查点标识：{stats['untrained_baseline']} 个回合为未训练反馈基线，{stats['cem_trained']} 个回合为 CEM 训练策略。其余记录若来源未注明，不推定经过训练。传感器输入不包含全局位姿。</p><p>记录中的有效圈须经独立轨迹审计。累计侧滑时间不等于连续漂移通过；后轮增速也不等于车体漂移。</p><p>其中 {stats['prior_only']} 个回合使用不同版本/源码执行检查点参数，只能称为参数先验迁移。另有 {stats['geometry_changed']} 个回合对应已修订的旧赛道，不能作为当前布局的验收。</p></section>
<section class="panel"><h2>竞速结果</h2><p class="muted">每行一个真实回合；不同赛道长度不相同。圈时仅显示记录为有效的回合。点击列名排序，筛选不改变上方全标签统计。</p>
<div class="controls"><label>引擎<select id="engine"><option value="">全部</option><option>isaac</option><option>mujoco</option></select></label><label>状态<select id="status"><option value="">全部回合</option><option value="verified">独立确认有效圈</option><option value="failed">未完成有效圈</option><option value="unverified">尚未独立验证</option></select></label><label><span class="sr">搜索赛道</span><input id="query" placeholder="搜索赛道或引擎"></label><span id="row-count" class="muted"></span></div>
<div class="scroll"><table><thead><tr>{th}<th>证据文件</th></tr></thead><tbody id="results"></tbody></table></div><noscript>请启用 JavaScript 进行离线筛选；完整静态结果见 RESULTS.md。</noscript></section>
<section class="panel"><h2>覆盖范围与迁移身份</h2>{manifest_text}{coverage}<ul>{identities}</ul><details><summary>逐回合检查点与审计详情</summary>{rows_detail or '<p>尚无回合记录。</p>'}</details></section>
<section><h2>原生渲染录像</h2><p class="muted">仅列出本标签结果中引用且实际存在的视频；不使用其他实验视频替代。影片与图片通过本地相对路径读取。</p><div class="videos">{''.join(videos) or '<p class="empty">本标签尚无可链接的录像。</p>'}</div></section>
<section><h2>24 条赛道资产</h2><div class="controls"><label>赛季<select id="season"><option>2025</option><option>2024</option></select></label><a href="../../tracks/catalog.json">完整赛季索引</a><a href="../../docs/TRACK_SOURCES.md">来源与布局限制</a><a href="../../tracks/sources/LICENSE.md">MIT 来源许可</a></div><p class="muted">每季 24 场，共 48 条分站记录。路径快照不声称历史逐年准确。RC 默认路径 ×0.05；Baku ×0.100、Monaco ×0.125、Miami/Montreal ×0.075，以固定 1.5 m 路宽分离邻近路段。原尺度版本路宽假定 12 m。RC 分层状态依资产元数据展示；原尺度几何未随 RC 桥梁改造改变。</p><div class="maps" id="track-maps"></div></section>
<section class="panel"><h2>来源与复现</h2><p>生成时间 UTC：{e(data['generated_at'])}<br>本次读取的最新结果修改时间 UTC：{e(data['evidence_latest_modified'] or '暂无结果')}。本页面是读取时快照，不自动跟随训练更新。</p><pre>{e(data['reproduce_command'])}</pre><details><summary>读取的结果文件与 SHA-256</summary><ul>{sources or '<li>未发现匹配结果。</li>'}</ul></details><p>{e('；'.join(data['warnings']) or '本次读取没有发现文件格式警告。')}</p></section>
<footer>离线报告：HTML/CSS/JavaScript 均内置，无 CDN 或外部请求。共享页面时请保留 output/racing 与 tracks 的相对目录；仅复制 HTML 不包含地图、视频和证据文件。</footer></main><script type="application/json" id="report-data">{payload}</script><script>{JS}</script></body></html>'''


def render_markdown(data):
    s = data["stats"]
    lines = ["# OSRACER 阶段性竞速验收", "",
        f"标签 `{data['tag']}`：读取 {s['episodes']} 个回合；运行记录称有效圈 {s['reported_valid']} 个，独立确认有效圈 {s['verified_valid']} 个。",
        f"证据验证通过 {s['integrity_passed']} 个，失败 {s['integrity_failed']} 个，尚未独立验证 {s['unverified']} 个。当前没有全部竞速目标通过的证据。", "",
        f"策略类别：未训练反馈基线 {s['untrained_baseline']} 个回合；CEM 训练策略 {s['cem_trained']} 个回合。其余记录不推定经过训练。", "",
        f"其中 {s['prior_only']} 个回合属于策略版本或源码改变后的参数先验迁移，不能称为完整策略的直接迁移。另有 {s['geometry_changed']} 个回合对应旧赛道几何，不能确认当前布局验收。", "",
        "[打开离线交互报告](index.html) · [工程说明](../../docs/ENGINEERING.md) · [赛道来源](../../docs/TRACK_SOURCES.md)", "",
        "| 引擎 | 标签 | 赛道 | 种子 | 记录有效圈 | 证据状态 | 圈时 s | 峰值 m/s | 均速 m/s | 有效超车 | 侧滑累计 s | 失败原因 |", 
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in data["rows"]:
        values = [r["engine"],r["tag"],f'[{r["track"]}]({r["source"]})',str(r["seed"]),"是" if r["valid_lap"] else "否",
            r["verification"],fmt(r["lap_time"]),fmt(r["peak_speed"]),fmt(r["mean_speed"]),fmt(r["overtakes"],0),fmt(r["drift"],3),str(r["failure"])]
        lines.append("| "+" | ".join(v.replace("|","\\|").replace("\n"," ") for v in values)+" |")
    if not data["rows"]:
        lines += ["", "未找到匹配结果；缺失数据没有填为零。"]
    lines += ["", "## 覆盖与迁移身份", ""]
    if data.get("manifests"):
        links = " · ".join(f"[{m.get('tag','manifest')} manifest]({m['report_source_path']})" for m in data['manifests'])
        lines += [f"{links} 状态快照：`{json.dumps(data['manifest_states'],ensure_ascii=False)}`。`evaluated` 只表示任务已执行并进入审计，不表示驾驶全部通过。无 manifest 的所选标签回合仍独立纳入。", ""]
    for engine,c in data["coverage"].items():
        lines.append(f"- {engine}：覆盖 {len(c['observed'])}/24 条赛道；缺少记录：{', '.join(c['missing']) or '无'}。")
    lines += [f"- {i['track']}：{i['status']}。" for i in data["identity"]]
    if not data["identity"]:
        lines.append("- 暂无同标签、同赛道的双引擎对照，不能确认跨引擎迁移。")
    lines += ["", "## 证据限制", "",
        "累计侧滑时长不能替代连续漂移验收。结果表显示已保存汇总，独立审计的完整性与有效圈结论另列；没有审计文件或结果哈希不匹配时，不计为已验证。跨引擎相同检查点 SHA 也不证明策略代码或场景条件相同。", "",
        "赛道包含 2024/2025 的 24 条近似布局、RC 与原尺度 DAE/OBJ 和独立 PNG 纹理。RC 默认路径缩放 0.05；Baku 0.100、Monaco 0.125、Miami/Montreal 0.075。路宽固定 1.5 m，原尺度路宽 12 m 是假定。Suzuka RC 已增加物理桥面、坡道和高度连续投影，需三维后端；其 full_scale 仍为平面参考，不支持分层竞速。桥高与坡长均为明确训练假设，其他拓扑限制见赛道文档。", ""]
    for r in data["rows"]:
        lines += [f"- **{r['engine']}/{r['track']}/seed {r['seed']}**：{r['migration']}；版本 `{r['policy_version']}`，检查点版本 `{r['checkpoint_version']}`；检查点 SHA `{r['checkpoint_sha256'] or '未提供'}`。"]
        if r["errors"]:
            lines.append("  审计/身份问题："+"；".join(r["errors"])+"。")
    lines += ["", "## 录像与来源", ""]
    videos = [r for r in data["rows"] if r["video"]]
    lines += [f"- [{r['engine']} / {r['track']} / seed {r['seed']} 视频]({r['video']})：{'独立解码通过' if r['video_verified'] else '独立解码尚未确认'}。" for r in videos]
    if not videos:
        lines.append("本标签尚无可链接的录像。")
    lines += ["",f"生成时间 UTC：{data['generated_at']}。最新结果修改时间 UTC：{data['evidence_latest_modified'] or '暂无结果'}。", ""]
    lines += [f"- [{x['path']}]({x['path']}) · SHA-256 `{x['sha256']}`" for x in data["sources"]]
    lines += ["", "```bash",data["reproduce_command"],"```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag",action="append",help="Result tag; repeat to combine source and target runs")
    parser.add_argument("--tags",nargs="+",help="Alternative list of result tags")
    parser.add_argument("--checkpoint",help="Optional exact checkpoint whose hash should match each result")
    args = parser.parse_args()
    tags = list(dict.fromkeys((args.tag or [])+(args.tags or [])))
    if not tags or any(not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag) for tag in tags):
        parser.error("Supply --tag or --tags using only letters, digits, underscore or hyphen")
    data = collect(tag=tags,checkpoint=args.checkpoint)
    output = ROOT/"output/racing"
    output.mkdir(parents=True,exist_ok=True)
    # Atomic replacement keeps a previously generated report readable while rebuilding.
    for name,content in [("index.html",render_html(data)),("RESULTS.md",render_markdown(data))]:
        temporary = output/(name+".tmp")
        temporary.write_text(content,encoding="utf-8")
        temporary.replace(output/name)
    print(json.dumps({"tags":tags,**data["stats"],"html":str(output/"index.html"),"markdown":str(output/"RESULTS.md")},ensure_ascii=False))


if __name__ == "__main__":
    main()
