#!/usr/bin/env python3
"""Convert legacy JSON training histories into TensorBoard scalar events."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import statistics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'output/racing'
DEFAULT_DESTINATION = ROOT / 'runs/_legacy/tensorboard'

METRIC_KEYS = (
    'score', 'valid_lap', 'completed', 'duration_s', 'progress_m', 'lap_fraction',
    'peak_speed_m_s', 'mean_speed_m_s', 'effective_overtakes',
    'max_continuous_drift_duration_s', 'collision_steps', 'offroad_steps',
    'all_segments_clean', 'all_forward_progress', 'all_overtakes',
    'all_continuous_drift', 'minimum_continuous_drift_s', 'wall_seconds',
)


def scalar(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def candidate_metrics(row: dict) -> dict[str, float]:
    metrics = row.get('metrics') if isinstance(row.get('metrics'), dict) else row
    values = {}
    for key in METRIC_KEYS:
        value = scalar(row.get(key, metrics.get(key)))
        if value is not None:
            values[key] = value
    episodes = row.get('episodes')
    if isinstance(episodes, list):
        by_engine = defaultdict(list)
        for episode in episodes:
            if isinstance(episode, dict):
                by_engine[str(episode.get('engine', 'unknown'))].append(episode)
        for engine, engine_rows in by_engine.items():
            for key in METRIC_KEYS:
                samples = [scalar(item.get(key)) for item in engine_rows]
                samples = [value for value in samples if value is not None]
                if samples:
                    values[f'{engine}/{key}_mean'] = statistics.fmean(samples)
    return values


def export_history(source: Path, destination: Path) -> dict:
    from tensorboardX import SummaryWriter

    rows = json.loads(source.read_text())
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError('expected a non-empty list of candidate dictionaries')
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    writer = SummaryWriter(str(destination))
    generations = defaultdict(list)
    try:
        writer.add_text('source/history_file', str(source), 0)
        for step, row in enumerate(rows):
            generation = int(row.get('generation', 0))
            candidate = int(row.get('candidate', step))
            metrics = candidate_metrics(row)
            score = metrics.get('score')
            if score is not None:
                generations[generation].append((score, metrics))
            writer.add_scalar('candidate/generation', generation, step)
            writer.add_scalar('candidate/index', candidate, step)
            for key, value in metrics.items():
                writer.add_scalar(f'candidate/{key}', value, step)
            parameters = row.get('parameters', row.get('variables'))
            if isinstance(parameters, list):
                for index, value in enumerate(parameters):
                    value = scalar(value)
                    if value is not None:
                        writer.add_scalar(f'parameters/p{index:02d}', value, step)
        for generation, items in sorted(generations.items()):
            scores = [score for score, _ in items]
            writer.add_scalar('generation/best_score', max(scores), generation)
            writer.add_scalar('generation/mean_score', statistics.fmean(scores), generation)
            writer.add_scalar('generation/score_std', statistics.pstdev(scores), generation)
            successes = [metrics.get('valid_lap', metrics.get('all_segments_clean'))
                         for _, metrics in items]
            successes = [value for value in successes if value is not None]
            if successes:
                writer.add_scalar('generation/success_rate', statistics.fmean(successes), generation)
    finally:
        writer.close()
    metadata = {'source':str(source),'candidates':len(rows),
                'generations':len({int(row.get('generation',0)) for row in rows})}
    (destination/'source.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    return metadata


def run_name(source_root: Path, history: Path) -> Path:
    relative = history.parent.relative_to(source_root)
    return Path('_root') if relative == Path('.') else relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--destination',type=Path,default=DEFAULT_DESTINATION)
    parser.add_argument('--history',type=Path,action='append',
                        help='Export only this history file; repeat for multiple files')
    args = parser.parse_args()
    source = args.source.resolve()
    histories = [path.resolve() for path in args.history] if args.history else sorted(source.rglob('*history*.json'))
    if not histories:
        parser.error(f'no training history JSON found under {source}')
    exported = []
    for history in histories:
        try:
            name = run_name(source,history)
        except ValueError:
            name = Path(history.parent.name)
        metadata = export_history(history,args.destination/name)
        exported.append({'run':str(name),**metadata})
    print(json.dumps({'destination':str(args.destination.resolve()),'runs':exported},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
