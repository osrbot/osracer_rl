#!/usr/bin/env python3
"""Group run logs and present policy files for browsability.

Nothing that is hashed as evidence is moved or rewritten. Loose run logs are
moved into ``output/racing/logs/<group>/`` because no artifact references them;
policy checkpoints stay where the results point at them and are presented
through symlinks plus a manifest. Re-running is safe and only picks up files
that appeared since the last pass.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/racing'

LOG_GROUPS = [
    ('isaac', re.compile(r'(^|_)isaac|_isaac_')),
    ('mujoco', re.compile(r'(^|_)mu(_|$)|mujoco')),
    ('bridge', re.compile(r'^bridge_')),
    ('training', re.compile(r'train|cem|joint|prior')),
    ('qualification', re.compile(r'qualification|batch|sweep|smoke|test_suite')),
]

# Files that documents link to stay where they are.
KEEP_IN_PLACE = {'full_test_suite_latest.log'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def group_for(name):
    for group, pattern in LOG_GROUPS:
        if pattern.search(name):
            return group
    return 'misc'


def organize_logs(dry_run=False):
    moved = []
    for path in sorted(OUT.glob('*.log')):
        if path.name in KEEP_IN_PLACE:
            continue
        target_dir = OUT / 'logs' / group_for(path.name)
        target = target_dir / path.name
        if target.exists():
            target = target_dir / f'{path.stem}.{sha256(path)[:8]}{path.suffix}'
        moved.append((path, target))
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
    return moved


def policy_role(path, spec):
    name = path.name
    version = str(spec.get('policy_version', ''))
    status = str(spec.get('qualification_status', ''))
    if name.startswith('g00c_v10c_lateral'):
        return 'frozen', '冻结交付版（共享策略）'
    if name.startswith('g00c_v10c_cruise9'):
        return 'speed', '速度变体（巡航 9.0 m/s，MuJoCo 全赛道有效）'
    if name.startswith('g00c_v10'):
        return 'experiment', '鲁棒性/恢复实验分支（未合入）'
    if name.startswith('g00c03') or name.startswith('g00c_'):
        return 'experiment', '历史候选'
    if 'baseline' in name or 'untrained' in status.lower():
        return 'baseline', '未训练基线'
    if spec.get('rebound_from'):
        return 'experiment', '参数迁移检查点'
    if version:
        return 'experiment', '实验检查点'
    return 'other', '未标注'


def collect_policies():
    patterns = ['candidates/*.json', '*.json', '*/policy.json', 'training/*/policy.json',
                'experimental/*/candidate*.json', 'experimental/*/policy.json']
    seen, rows = set(), []
    for pattern in patterns:
        for path in sorted(OUT.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            try:
                spec = json.loads(path.read_text())
            except Exception:
                continue
            if not isinstance(spec, dict) or 'parameters' not in spec:
                continue
            seen.add(path)
            role, label = policy_role(path, spec)
            parameters = spec.get('parameters') or []
            rows.append({'path': str(path.relative_to(ROOT)), 'role': role, 'label': label,
                         'policy_version': spec.get('policy_version'),
                         'algorithm': spec.get('algorithm'),
                         'actor_type': spec.get('actor_type'),
                         'safety_supervisor': bool(spec.get('safety_supervisor', False)),
                         'trained_in': spec.get('trained_in'),
                         'qualification_status': spec.get('qualification_status'),
                         'cruise_m_s': parameters[0] if parameters else None,
                         'parameters': parameters,
                         'bytes': path.stat().st_size, 'sha256': sha256(path)})
    order = {'frozen': 0, 'speed': 1, 'baseline': 2, 'experiment': 3, 'other': 4}
    rows.sort(key=lambda row: (order.get(row['role'], 9), row['path']))
    return rows


def write_policy_view(rows, dry_run=False):
    view = OUT / 'policies'
    if not dry_run:
        view.mkdir(parents=True, exist_ok=True)
    lines = ['# 策略文件索引', '',
             '符号链接指向原始检查点，原文件保持在结果引用它们的位置，不改动、不复制。',
             '**冻结交付版是 `frozen-g00c_v10c_lateral.json`。**', '',
             '| 角色 | 链接 | 版本 | 巡航 | 训练引擎 | 状态 | SHA-256（前 12 位） |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    wanted = {}
    for row in rows:
        relative = Path(row['path']).relative_to('output/racing')
        link = f"{row['role']}-{'-'.join(relative.parts)}"
        wanted[link] = ROOT / row['path']
    if not dry_run:
        for existing in view.iterdir():
            if existing.is_symlink() and existing.name not in wanted:
                existing.unlink()
    for row in rows:
        relative = Path(row['path']).relative_to('output/racing')
        link = f"{row['role']}-{'-'.join(relative.parts)}"
        source = ROOT / row['path']
        target = view / link
        if not dry_run and not target.exists():
            os.symlink(os.path.relpath(source, view), target)
        cruise = f"{row['cruise_m_s']:.2f}" if isinstance(row['cruise_m_s'], (int, float)) else '—'
        engines = ','.join(row['trained_in']) if isinstance(row['trained_in'], list) else (row['trained_in'] or '—')
        lines.append('| {} | [{}]({}) | {} | {} | {} | {} | `{}` |'.format(
            row['label'], link, link, row['policy_version'] or '—', cruise, engines,
            str(row['qualification_status'] or '—')[:40], row['sha256'][:12]))
    if not dry_run:
        (view / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        (view / 'INDEX.json').write_text(
            json.dumps({'schema_version': 1, 'count': len(rows), 'policies': rows},
                       indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_log_index(moved, dry_run=False):
    target = OUT / 'logs/README.md'
    lines = ['# 运行日志索引', '',
             '日志不被任何结果哈希引用，因此按来源归入子目录；历史上已有的 `<tag>/` 目录是资格入口逐赛道日志。', '',
             '- `isaac/`、`mujoco/`：按引擎的单次运行日志',
             '- `bridge/`：桥面接触与几何实验',
             '- `training/`：CEM/联合训练与先验生成',
             '- `qualification/`：批量资格、冒烟与工程测试',
             '- `misc/`：其余', '',
             f'本轮归位 {len(moved)} 个文件。', '',
             '| 日志 | 分组 |', '| --- | --- |']
    for source, destination in moved:
        lines.append(f'| [{destination.name}]({destination.parent.name}/{destination.name}) | {destination.parent.name} |')
    existing = (OUT / 'logs').glob('*/*.log')
    if not dry_run:
        (OUT / 'logs').mkdir(parents=True, exist_ok=True)
        (OUT / 'logs/README.md').write_text(
            '\n'.join(lines + ['', f'当前子目录中共有 {len(list(existing))} 个日志文件。']) + '\n',
            encoding='utf-8')


def write_root_readme(policies, dry_run=False):
    text = f'''# output/racing 目录导览

本目录存放全部竞速实验产物。**证据文件不移动、不改名**（结果里记录了它们的路径与哈希），
散落的日志被归入 `logs/`，策略检查点通过 `policies/` 以符号链接集中呈现。

## 先看哪里

| 想了解 | 打开 |
| --- | --- |
| 冻结交付版：配置、成绩、已知限制、复现命令 | [../../docs/RELEASE.md](../../docs/RELEASE.md) |
| 逐项验证状态与失败清单 | [../../docs/VALIDATION_STATUS.md](../../docs/VALIDATION_STATUS.md) |
| 恢复逻辑、鲁棒性与被撤回实验 | [../../docs/RECOVERY.md](../../docs/RECOVERY.md) |
| 可排序成绩报告 | [index.html](index.html) ／ [RESULTS.md](RESULTS.md) |
| 录像与图片素材库（可点播） | [media.html](media.html) ／ [MEDIA.md](MEDIA.md) ／ `MEDIA_INDEX.json` |
| 策略文件 | [policies/README.md](policies/README.md)（共 {len(policies)} 个检查点） |
| 运行日志 | [logs/README.md](logs/README.md) |

## 目录职责

- `policies/`：策略检查点的符号链接视图与清单（原文件保持原位）。
- `logs/`：运行日志，按 `isaac/`、`mujoco/`、`bridge/`、`training/`、`qualification/`、`misc/` 分组；
  其中 `<tag>/` 子目录是资格入口逐赛道日志。
- `isaac/`、`mujoco/`：原生回合结果、完整轨迹、录像与逐回合审计。
- `training/`：CEM 训练产出（策略、历史、契约）。
- `experimental/`：冻结工作区与桥面接触等实验。
- `candidates/`：训练候选与参数迁移检查点。
- 顶层 `*_summary.json`、`*_manifest.json`、`*_audits.json`：批量运行摘要与独立审计报告。

## 维护

新增运行后重新生成索引即可，不会改动证据：

```bash
.venv/bin/python -I scripts/organize_outputs.py
.venv/bin/python -I -m racing.report --tag <tags...> --checkpoint <checkpoint>
.venv/bin/python -I scripts/index_media.py --tag <tags...>
```
'''
    if not dry_run:
        (OUT / 'README.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    moved = organize_logs(args.dry_run)
    policies = collect_policies()
    write_policy_view(policies, args.dry_run)
    write_log_index(moved, args.dry_run)
    write_root_readme(policies, args.dry_run)
    print(json.dumps({'moved_logs': len(moved), 'policies': len(policies),
                      'groups': sorted({destination.parent.name for _, destination in moved}),
                      'dry_run': args.dry_run}, ensure_ascii=False))


if __name__ == '__main__':
    main()
