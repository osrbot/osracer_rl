#!/usr/bin/env python3
"""Eighteen-candidate native MuJoCo closed-loop experiment, never a leaderboard."""
from argparse import ArgumentParser, Namespace
from itertools import product
import hashlib
import json
from pathlib import Path
import traceback
from probe_closed_loop_racing import ROOT, run_probe


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--track', default='bahrain')
    parser.add_argument('--tag', default='v6d_grid18')
    parser.add_argument('--seconds', type=float, default=90.)
    args = parser.parse_args()
    out = ROOT/'output/racing/experimental'/args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out/'search_closed_loop_racing.py').write_bytes(Path(__file__).read_bytes())
    candidates = list(product([2.5, 3., 3.5], [.4, .5, .6], [.1, .14]))
    rows = []
    for index, (ratio, pulse, bias) in enumerate(candidates):
        tag = f'{args.tag}/candidate_{index:02d}'
        settings = Namespace(track=args.track, seconds=args.seconds, ratio=ratio, pulse=pulse,
                             steer_bias=bias, trigger=.30, target_slip_deg=25., start_s=0., tag=tag)
        try:
            result = run_probe(settings)
            clean = bool(result['completed'] and result['valid_lap'] and not result['failure'])
            longest = result['max_continuous_drift_duration_s']
            qualified = bool(clean and longest > .1+1e-9)
            margin = bool(qualified and longest+1e-9 >= .2)
            reward = (1000.+2000.*min(longest/.2, 1.)-result['duration_s']) if qualified else 0.
            row = {'candidate': index, 'settings': vars(settings), 'clean_lap': clean,
                   'continuous_drift_qualified_strict': qualified, 'margin_0_2_s': margin,
                   'reward': reward, 'result_path': str(ROOT/'output/racing/experimental'/tag/'result.json'),
                   'duration_s': result['duration_s'], 'failure': result['failure'],
                   'longest_drift_s': longest, 'peak_moving_slip_deg': result['max_rear_slip_deg'],
                   'effective_overtakes': result['effective_overtakes']}
        except Exception:
            row = {'candidate': index, 'settings': vars(settings), 'clean_lap': False,
                   'continuous_drift_qualified_strict': False, 'margin_0_2_s': False, 'reward': 0.,
                   'failure': 'execution_error', 'error': traceback.format_exc()}
        rows.append(row)
        document = {'schema_version': 1, 'experimental': True, 'candidate_count': len(candidates),
                    'finished_count': len(rows), 'qualification': 'clean full lap AND longest contiguous moving slip > 0.1 s; prefer >= 0.2 s',
                    'slip_threshold_deg': 20., 'speed_threshold_m_s': 1.2,
                    'qualified_count': sum(r['continuous_drift_qualified_strict'] for r in rows),
                    'margin_count': sum(r['margin_0_2_s'] for r in rows), 'candidates': rows}
        (out/'search.json').write_text(json.dumps(document, indent=2)+'\n')
        print('V6_GRID', json.dumps(row), flush=True)
    ranked = sorted(rows, key=lambda r: (r['margin_0_2_s'], r['continuous_drift_qualified_strict'], r['reward']), reverse=True)
    (out/'ranked.json').write_text(json.dumps(ranked, indent=2)+'\n')


if __name__ == '__main__':
    main()
