#!/usr/bin/env python3
"""Audit every completed native episode of a frozen experimental workspace.

No success filtering is performed. Failed laps and missing drift are retained.
Controller phase records describe commands, not proof of drift causation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racing.verify import audit_trace
from racing.tracks import Track


def audit(workspace, tags):
    workspace = Path(workspace).resolve()
    checkpoint = workspace / 'candidate.json'
    bundle = json.loads(checkpoint.read_text())
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    actor_bytes_match = all((workspace / name).is_file()
        and hashlib.sha256((workspace / name).read_bytes()).hexdigest() == expected
        for name, expected in bundle.get('actor_source_sha256', {}).items())
    rows = []
    for engine in ['isaac', 'mujoco']:
        for tag in tags:
            for result_path in sorted((workspace / 'output/racing' / engine).glob(f'*_{tag}.json')):
                results = json.loads(result_path.read_text())
                if not isinstance(results, list):
                    continue
                for result in results:
                    trace_path = result_path.with_name(result_path.stem + f"_seed{result['seed']}_trace.json")
                    trace = json.loads(trace_path.read_text())
                    track = Track(workspace / 'tracks' / result.get('track', 'bahrain') / 'track.json')
                    checks = audit_trace(result, trace, track)
                    source_match = (actor_bytes_match
                                    and result.get('actor_source_sha256') == bundle.get('actor_source_sha256')
                                    and result.get('checkpoint_sha256') == checkpoint_hash)
                    rows.append(dict(engine=engine, tag=tag, seed=result['seed'],
                        evidence_integrity_passed=checks['evidence_integrity_passed'],
                        checkpoint_and_actor_match=source_match,
                        valid_lap=bool(result.get('valid_lap')), failure=result.get('failure'),
                        lap_seconds=result['duration_s'], overtakes=result['effective_overtakes'],
                        peak_slip_deg=result['max_rear_slip_deg'],
                        continuous_drift_seconds=result['max_continuous_drift_duration_s'],
                        drift_qualified=bool(result.get('continuous_drift_qualified')),
                        commanded_boost_frames=sum(r.get('actor_feedback', {}).get('rear_ratio', 0) > 0 for r in trace),
                        trace=str(trace_path.relative_to(workspace)),
                        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                        independent_verification=checks))
    report = {'checkpoint_sha256': checkpoint_hash, 'actor_bytes_match': actor_bytes_match,
              'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'algorithm': bundle.get('algorithm'),
              'rows': rows, 'limitation': 'All episodes retained. These are finite development/validation experiments, not CEM training.'}
    target = workspace / 'candidate_audit.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace')
    parser.add_argument('--tags', nargs='+', default=['development', 'heldout'])
    args = parser.parse_args()
    report = audit(args.workspace, args.tags)
    for row in report['rows']:
        print({k: v for k, v in row.items() if k != 'independent_verification'})
