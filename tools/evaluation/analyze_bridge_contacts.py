#!/usr/bin/env python3
"""Summarize recorded support normals; does not grade policy performance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(path):
    data = json.loads(path.read_text())
    expected = np.array([-data['grade'], 0., 1.])
    expected /= np.linalg.norm(expected)
    entries = []
    for record in data.get('support_contacts', []):
        if 'front_wheel_link' not in record['actor0'] + record['actor1']:
            continue
        for point in record['contacts']:
            impulse = float(np.linalg.norm(point['impulse']))
            if impulse < 1e-6:
                continue
            normal = np.array(point['normal'], dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-15)
            is_flat = data.get('surface') != 'plane' and 'defaultGroundPlane' in record['collider0'] + record['collider1']
            target = np.array([0., 0., 1.]) if is_flat else expected
            if normal @ target < 0:
                normal = -normal
            angle = float(np.degrees(np.arccos(np.clip(normal @ target, -1, 1))))
            entries.append({'tick': record['control_tick'], 'surface': 'ground_plane' if is_flat else data.get('surface', 'box_ramp'),
                            'actor0': record['actor0'], 'actor1': record['actor1'],
                            'normal_error_deg': angle, 'impulse_norm_ns': impulse, **point})
    return {'artifact': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'fixture_passed': data.get('passed'), 'summary': data.get('summary'),
            'cylinder_approximation_enabled': data.get('cylinder_approximation_enabled'),
            'wheel_contact_slop_coefficients': data.get('wheel_contact_slop_coefficients'),
            'nonzero_front_contacts': len(entries), 'first_contacts': entries[:8],
            'largest_impulses': sorted(entries, key=lambda e:e['impulse_norm_ns'], reverse=True)[:8],
            'largest_normal_errors': sorted(entries, key=lambda e:e['normal_error_deg'], reverse=True)[:8]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps([analyze(p) for p in args.artifacts], indent=2)+'\n')

if __name__ == '__main__':
    main()
