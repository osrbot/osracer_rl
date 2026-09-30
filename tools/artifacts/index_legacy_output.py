#!/usr/bin/env python3
"""Index the pre-run-layout output archive without moving or rewriting evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def category(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {'.mp4', '.mov', '.mkv', '.webm'}:
        return 'videos'
    if suffix in {'.png', '.jpg', '.jpeg', '.webp'}:
        return 'images'
    if suffix in {'.log', '.out', '.err'}:
        return 'logs'
    if suffix in {'.pt', '.pth', '.ckpt', '.onnx', '.safetensors'} or any(word in name for word in ('policy', 'checkpoint')):
        return 'checkpoints'
    if 'trace' in name or 'trajectory' in name:
        return 'trajectories'
    if suffix in {'.py', '.cpp', '.hpp', '.h', '.c', '.sh'}:
        return 'source-snapshots'
    if suffix in {'.json', '.jsonl', '.csv', '.tsv', '.tfevents'}:
        return 'metrics-and-metadata'
    return 'other'


def build_index(source: Path) -> dict:
    files = []
    counts = Counter()
    sizes = Counter()
    batches = defaultdict(lambda: {'files': 0, 'bytes': 0})
    for path in sorted(p for p in source.rglob('*') if p.is_file()):
        relative = path.relative_to(source)
        stat = path.stat()
        kind = category(path)
        batch = relative.parts[0] if len(relative.parts) > 1 else '_root'
        counts[kind] += 1
        sizes[kind] += stat.st_size
        batches[batch]['files'] += 1
        batches[batch]['bytes'] += stat.st_size
        files.append({'path':str(relative),'category':kind,'bytes':stat.st_size,
                      'modified_at':datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat()})
    return {'schema_version':1,'kind':'legacy-output-index','source':str(source),
            'generated_at':datetime.now(timezone.utc).isoformat(),
            'summary':{'files':len(files),'bytes':sum(item['bytes'] for item in files),
                       'categories':{key:{'files':counts[key],'bytes':sizes[key]} for key in sorted(counts)},
                       'top_level':dict(sorted(batches.items()))},'files':files}


def build_link_view(source: Path, index: dict, destination: Path) -> None:
    """Create a categorized, zero-copy view while keeping evidence paths stable."""
    for item in index['files']:
        link = destination / item['category'] / item['path']
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(source / item['path'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'output/racing')
    parser.add_argument('--destination',type=Path,default=ROOT/'runs/_legacy/index.json')
    parser.add_argument('--link-view',type=Path,
                        help='Optional zero-copy categorized view, for example runs/_legacy/by-category')
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f'legacy output directory does not exist: {args.source}')
    index = build_index(args.source.resolve())
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(index,ensure_ascii=False,indent=2))
    temporary.replace(args.destination)
    if args.link_view:
        build_link_view(args.source.resolve(),index,args.link_view.resolve())
    print(json.dumps(index['summary'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
