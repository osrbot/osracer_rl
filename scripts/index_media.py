#!/usr/bin/env python3
"""Index native racing video and frame material into one browsable collection.

Every clip keeps its provenance: engine, track, result tag, seed, decoded
duration, byte size and SHA-256, plus the measured lap metrics and whether it
is a passing lap or a retained failure (control group). Nothing is deleted or
rewritten; the index only reads what the runs already produced.
"""
import argparse
import hashlib
import html
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/racing'
VIDEO_RE = re.compile(r'^(?P<track>[a-z_]+)_(?P<tag>.+?)(?:_seed(?P<seed>\d+))?\.mp4$')


def digest(path, block=1 << 20):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(block), b''):
            value.update(chunk)
    return value.hexdigest()


def probe(path):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-show_entries', 'stream=width,height,nb_frames',
                              '-show_entries', 'format=duration',
                              '-of', 'json', str(path)],
                             capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
        stream = (data.get('streams') or [{}])[0]
        return {'duration_s': float(data.get('format', {}).get('duration', 0.)),
                'width': int(stream.get('width', 0)), 'height': int(stream.get('height', 0)),
                'frames': int(stream.get('nb_frames', 0) or 0), 'decoded': True}
    except Exception as exc:  # a broken clip is evidence too
        return {'duration_s': None, 'width': None, 'height': None, 'frames': None,
                'decoded': False, 'decode_error': f'{type(exc).__name__}: {exc}'}


def episodes_for(engine, track, tag):
    path = OUT / engine / f'{track}_{tag}.json'
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    return data if isinstance(data, list) else [data]


def metrics_for(engine, track, tag, seed, video):
    for episode in episodes_for(engine, track, tag):
        recorded = (episode.get('video') or {}).get('path')
        if recorded and Path(recorded).name == Path(video).name:
            return episode
        if seed is not None and episode.get('seed') == seed:
            return episode
    return {}


def shape(episode):
    return {'valid_lap': episode.get('valid_lap'), 'lap_time_s': episode.get('lap_time_s'),
            'mean_speed_m_s': episode.get('mean_speed_m_s'), 'peak_speed_m_s': episode.get('peak_speed_m_s'),
            'effective_overtakes': episode.get('effective_overtakes'),
            'max_continuous_drift_duration_s': episode.get('max_continuous_drift_duration_s'),
            'failure': episode.get('failure'), 'progress_m': episode.get('progress_m'),
            'collision_steps': episode.get('collision_steps'), 'offroad_steps': episode.get('offroad_steps')}


def build(args):
    evidence = set(args.tag or [])
    rows = []
    for path in sorted(OUT.rglob('*.mp4')):
        match = VIDEO_RE.match(path.name)
        if not match:
            continue
        parts = path.relative_to(OUT).parts
        engine = 'isaac' if 'isaac' in parts else ('mujoco' if 'mujoco' in parts else 'other')
        track, tag = match.group('track'), match.group('tag')
        seed = int(match.group('seed')) if match.group('seed') else None
        poster = path.with_suffix('.png')
        episode = metrics_for(engine, track, tag, seed, path)
        row = {'path': str(path.relative_to(ROOT)), 'engine': engine, 'track': track, 'tag': tag,
               'seed': seed, 'bytes': path.stat().st_size, 'sha256': digest(path),
               'poster': str(poster.relative_to(ROOT)) if poster.is_file() else None}
        row.update(probe(path))
        row.update(shape(episode))
        if tag in evidence:
            row['group'] = 'control' if row.get('valid_lap') is False else 'evidence'
        else:
            row['group'] = 'historical'
        rows.append(row)
        print('indexed', row['path'], row['group'], flush=True)
    return rows


def markdown(rows, path):
    lines = ['# 素材索引', '',
             '由 `scripts/index_media.py` 从已有产物生成；不移动、不重写任何录像。',
             '分组：`evidence` 当前证据、`control` 保留的失败对照组、`historical` 历史批次。', '']
    for group, title in [('evidence', '当前证据'), ('control', '失败对照组'), ('historical', '历史批次')]:
        subset = [r for r in rows if r['group'] == group]
        if not subset:
            continue
        lines += [f'## {title}（{len(subset)}）', '',
                  '| 引擎 | 赛道 | 配置 | 种子 | 有效圈 | 单圈 | 超车 | 最长漂移 | 时长 | 文件 |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in sorted(subset, key=lambda r: (r['engine'], r['tag'], r['track'], r['seed'] or 0)):
            lap = f"{r['lap_time_s']:.2f} s" if r.get('lap_time_s') else '—'
            drift = (f"{r['max_continuous_drift_duration_s']:.2f} s"
                     if r.get('max_continuous_drift_duration_s') else '—')
            duration = f"{r['duration_s']:.1f} s" if r.get('duration_s') else '—'
            lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | [视频]({}) |'.format(
                r['engine'], r['track'], r['tag'], r['seed'] if r['seed'] is not None else '—',
                '是' if r.get('valid_lap') else ('否' if r.get('valid_lap') is False else '—'),
                lap, r.get('effective_overtakes', '—'), drift, duration,
                os.path.relpath(ROOT / r['path'], start=path.parent).replace(os.sep, '/')))
        lines.append('')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def gallery(rows, path, root=ROOT):
    def card(r):
        label = '有效圈' if r.get('valid_lap') else ('对照组 ' + str(r.get('failure') or '失败'))
        lap = f"{r['lap_time_s']:.2f}s" if r.get('lap_time_s') else '—'
        poster = ''
        if r.get('poster'):
            poster = f' poster="{html.escape(str((root / r["poster"]).relative_to(OUT)))}"'
        return f'''<figure class="card {r['group']}">
  <video preload="none" controls muted playsinline{poster} src="{html.escape(str((root / r['path']).relative_to(OUT)))}"></video>
  <figcaption><b>{r['track']}</b> · {r['engine']} · seed {r['seed']}<br>
  <span class="badge">{label}</span> 单圈 {lap} · 超车 {r.get('effective_overtakes', '—')}</figcaption>
</figure>'''

    sections = []
    for group, title in [('evidence', '当前证据（有效圈）'), ('control', '失败对照组'), ('historical', '历史批次')]:
        subset = [r for r in rows if r['group'] == group]
        if not subset:
            continue
        grouped = {}
        for r in sorted(subset, key=lambda r: (r['tag'], r['engine'], r['track'], r['seed'] or 0)):
            grouped.setdefault((r['tag'], r['engine']), []).append(r)
        body = []
        for (tag, engine), items in grouped.items():
            body.append(f'<h3>{html.escape(tag)} · {engine} · {len(items)} 段</h3><div class="grid">')
            body.extend(card(r) for r in items)
            body.append('</div>')
        sections.append(f'<section><h2>{title}（{len(subset)}）</h2>{"".join(body)}</section>')
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>OSRACER 素材库</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;padding:24px;background:#12151c;color:#e6e9f0;
font:15px/1.5 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}}
h1{{font-size:22px;margin:0 0 6px}}h2{{font-size:18px;margin:32px 0 12px;border-bottom:1px solid #2a2f3a;padding-bottom:6px}}
h3{{font-size:14px;color:#9aa4b6;margin:20px 0 10px;font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
.card{{margin:0;background:#1a1f29;border:1px solid #262c38;border-radius:8px;overflow:hidden}}
.card video{{width:100%;aspect-ratio:16/9;background:#000;display:block}}
.card figcaption{{padding:10px 12px;font-size:13px;color:#c3cad8}}
.card.control{{border-color:#7a3b3b}} .card.evidence{{border-color:#2f5d45}}
.badge{{display:inline-block;padding:1px 8px;border-radius:999px;background:#2a3140;color:#cfd6e4;font-size:12px}}
.card.control .badge{{background:#5a2a2a;color:#ffd9d9}}
</style>
<h1>OSRACER 素材库</h1>
<p>共 {len(rows)} 段原生录像，按“当前证据 / 失败对照组 / 历史批次”分组。点开即播，未播放的录像不占用带宽。</p>
{"".join(sections)}
</html>
'''
    path.write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', action='append',
                        help='Result tag that counts as current evidence; repeatable')
    parser.add_argument('--json', type=Path, default=OUT/'MEDIA_INDEX.json')
    parser.add_argument('--markdown', type=Path, default=OUT/'MEDIA.md')
    parser.add_argument('--html', type=Path, default=OUT/'media.html')
    args = parser.parse_args()
    rows = build(args)
    args.json.write_text(json.dumps({'schema_version': 1, 'count': len(rows), 'media': rows},
                                    indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    markdown(rows, args.markdown)
    gallery(rows, args.html)
    groups = {g: sum(1 for r in rows if r['group'] == g) for g in ('evidence', 'control', 'historical')}
    broken = [r['path'] for r in rows if not r.get('decoded')]
    print(json.dumps({'media': len(rows), 'groups': groups, 'undecodable': broken,
                      'json': str(args.json), 'markdown': str(args.markdown), 'html': str(args.html)},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
