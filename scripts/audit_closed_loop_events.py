#!/usr/bin/env python3
"""Audit native experimental drift continuity and identify controller phases."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racing.tracks import Track
from racing.verify import audit_trace


def audit(directory):
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    rows = json.loads((directory/'trace.json').read_text())
    feedback = json.loads((directory/'feedback.json').read_text())
    integrity = audit_trace(result, rows, Track(result.get('track', 'bahrain')))
    events, start = [], None
    slip, speed = [], []
    for row in rows:
        s = row['states'][0]
        cy, sy = math.cos(s['yaw']), math.sin(s['yaw'])
        cp, sp = math.cos(s.get('pitch', 0)), math.sin(s.get('pitch', 0))
        cr, sr = math.cos(s.get('roll', 0)), math.sin(s.get('roll', 0))
        vx = cy*cp*s['vx'] + sy*cp*s['vy'] - sp*s.get('vz', 0)
        vy = (cy*sp*sr-sy*cr)*s['vx'] + (sy*sp*sr+cy*cr)*s['vy'] + cp*sr*s.get('vz', 0)
        slip.append(math.atan2(vy, vx))
        speed.append(math.hypot(s['vx'], s['vy']))
    for i in range(len(rows)+1):
        drifting = i < len(rows) and speed[i] > 1.2 and abs(slip[i]) > math.radians(20)
        if drifting and start is None:
            start = i
        if not drifting and start is not None:
            duration = rows[i-1]['t']-(rows[start-1]['t'] if start else 0.)
            if duration >= .1-1e-9:
                segment = feedback[start:i]
                events.append(dict(start_s=rows[start]['t'], duration_s=duration,
                    frames=i-start, phases=sorted({x['state'] for x in segment}),
                    commanded_boost_frames=sum(x['rear_ratio'] > 0 for x in segment),
                    commanded_boost_frames_preceding_half_second=sum(x['rear_ratio'] > 0 for x in feedback[max(0,start-30):start]),
                    safety_intervention_frames=sum(bool(x.get('safety', {}).get('must_brake')) for x in segment),
                    peak_slip_deg=max(abs(math.degrees(x)) for x in slip[start:i]),
                    min_speed_m_s=min(speed[start:i])))
            start = None
    report = dict(independent_verification=integrity, events=events,
                  commanded_boost_frames=sum(x['rear_ratio'] > 0 for x in feedback),
                  strict_continuous_drift_qualified=bool(integrity['evidence_integrity_passed']
                    and result.get('valid_lap') and any(x['duration_s'] > .1+1e-9 for x in events)),
                  limitation='Phase labels describe commands, not causal attribution. Truth is used only in this evaluator.')
    (directory/'closed_loop_event_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    for directory in parser.parse_args().directories:
        report = audit(directory)
        print(directory, report['strict_continuous_drift_qualified'], report['events'])
