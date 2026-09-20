"""Verifier checks deliberately independent of the production metric class."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from racing.verify import audit_artifact, audit_trace, parameter_sha256, sha256


class Straight:
    length, width = 100., 10.

    def at(self, s, offset=0.):
        return np.array([s % self.length, offset]), 0.

    def project(self, xy):
        return float(xy[0] % self.length), float(xy[1]), 0.


def state(x=0., y=0., vx=2., vy=0., collision=False):
    return dict(x=x, y=y, vx=vx, vy=vy, yaw=0., collision=collision)


def evidence(count=150):
    rows = []
    for i in range(count):
        t = (i+1)/60
        rows.append({'t': t, 'states': [state(x=2*t), state(x=3., vx=0.)],
                     'progress': 2*t, 'actions': [[[1., 1., 1., -1.], [0., 0.]]]*2})
    result = {'seed': 0, 'track': 'fake', 'duration_s': count/60, 'progress_m': count/30,
              'peak_speed_m_s': 2., 'mean_speed_m_s': 2., 'collision_steps': 0,
              'offroad_steps': 0, 'drift_duration_s': 0., 'max_abs_cte_m': 0.,
              'parameters': [1., 2.], 'completed': False, 'valid_lap': False,
              'failure': 'timeout', 'initial_states': [state(), state(x=3., vx=0.)]}
    return result, rows


class TestRacingVerifier(unittest.TestCase):
    def test_failed_episode_can_have_valid_evidence_and_one_persistent_pass(self):
        result, rows = evidence()
        result['effective_overtakes'] = 1
        audit = audit_trace(result, rows, Straight())
        self.assertTrue(audit['evidence_integrity_passed'], audit['errors'])
        self.assertFalse(audit['task_success'])
        self.assertEqual(audit['recorded_failure'], 'timeout')
        self.assertEqual(audit['recomputed']['effective_overtakes'], 1)

    def test_collision_disarms_pass_and_is_preserved(self):
        result, rows = evidence()
        rows[105]['states'][0]['collision'] = True
        result.update(collision_steps=1, effective_overtakes=0, failure='collision')
        audit = audit_trace(result, rows, Straight())
        self.assertTrue(audit['evidence_integrity_passed'], audit['errors'])
        self.assertFalse(audit['task_success'])
        self.assertEqual(audit['recomputed']['effective_overtakes'], 0)

    def test_false_lap_and_teleport_claims_fail(self):
        result, rows = evidence()
        result.update(completed=True, valid_lap=True, failure=None)
        audit = audit_trace(result, rows, Straight())
        self.assertFalse(audit['evidence_integrity_passed'])
        self.assertFalse(audit['task_success'])
        rows[30]['states'][0]['x'] += 10.
        audit = audit_trace(result, rows, Straight())
        self.assertGreater(len(audit['recomputed']['jump_records']), 0)

    def test_terminal_and_metric_tampering_are_detected(self):
        result, rows = evidence()
        result['peak_speed_m_s'] = 20.
        audit = audit_trace(result, rows[:-1], Straight())
        self.assertTrue(any('Terminal' in e for e in audit['errors']))
        self.assertTrue(any('peak_speed' in e for e in audit['errors']))

    def test_sparse_trace_reports_limitations_without_claiming_success(self):
        result, rows = evidence()
        sparse = rows[::4] + [rows[-1]]
        audit = audit_trace(result, sparse, Straight())
        self.assertTrue(audit['evidence_integrity_passed'], audit['errors'])
        self.assertFalse(audit['task_success'])
        self.assertFalse(audit['recomputed']['full_control_trace'])
        self.assertTrue(any('Sparse trace' in s for s in audit['limitations']))

    def test_drift_uses_body_velocity_angle_and_speed_gate(self):
        result, rows = evidence(count=60)
        for key in ('peak_speed_m_s', 'mean_speed_m_s', 'max_abs_cte_m', 'drift_duration_s'):
            result.pop(key)
        for i, row in enumerate(rows):
            # A 30-degree body slip at 2 m/s for the last half second.
            row['states'][0]['vx'] = np.sqrt(3) if i >= 30 else .1
            row['states'][0]['vy'] = 1. if i >= 30 else .1
        audit = audit_trace(result, rows, Straight())
        self.assertTrue(audit['evidence_integrity_passed'], audit['errors'])
        self.assertAlmostEqual(audit['recomputed']['drift_duration_s'], .5)
        self.assertAlmostEqual(audit['recomputed']['longest_contiguous_drift_s'], .5)
        self.assertAlmostEqual(audit['recomputed']['max_rear_slip_deg_above_1_2_m_s'], 30.)

    def test_disjoint_slip_spikes_cannot_fake_continuous_drift(self):
        result, rows = evidence(count=18)
        result.pop('peak_speed_m_s'); result.pop('mean_speed_m_s')
        for i, row in enumerate(rows):
            row['states'][0]['vy'] = 1. if i % 6 < 3 else 0.
        result.update(drift_duration_s=.15, max_continuous_drift_duration_s=.05,
                      drift_event_count=0, continuous_drift_qualified=False)
        audit = audit_trace(result, rows, Straight())
        self.assertTrue(audit['evidence_integrity_passed'], audit['errors'])
        self.assertFalse(audit['recomputed']['continuous_drift_qualified'])
        result.update(max_continuous_drift_duration_s=.15, drift_event_count=1,
                      continuous_drift_qualified=True)
        audit = audit_trace(result, rows, Straight())
        self.assertFalse(audit['evidence_integrity_passed'])
        self.assertTrue(any('drift_event_count' in error for error in audit['errors']))

    def test_checkpoint_hash_and_optional_video(self):
        result, rows = evidence()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, checkpoint = root/'race.json', root/'checkpoint.json'
            checkpoint.write_text(json.dumps({'parameters': result['parameters']}))
            result.update(checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                          parameters_sha256=parameter_sha256(result['parameters']))
            (root/'race_seed0_trace.json').write_text(json.dumps(rows))
            artifact.write_text(json.dumps([result]))
            with patch('racing.verify.Track', return_value=Straight()):
                audit = audit_artifact(artifact)
                self.assertTrue(audit['evidence_integrity_passed'], audit['episodes'][0]['errors'])
                self.assertFalse(audit['episodes'][0]['video']['present'])
                checkpoint.write_text(json.dumps({'parameters': [9.]}))
                tampered = audit_artifact(artifact)
                self.assertFalse(tampered['evidence_integrity_passed'])
                self.assertTrue(any('Checkpoint SHA' in e for e in tampered['episodes'][0]['errors']))

    def test_invalid_video_is_evidence_failure(self):
        from racing.verify import audit_video
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root/'invalid.mp4'
            video.write_bytes(b'not a video')
            audit = audit_video({'path': str(video), 'frames': 30, 'duration_s': 1.}, root/'race.json', 1.)
            self.assertTrue(any('ffprobe failed' in e for e in audit['errors']))


if __name__ == '__main__':
    unittest.main()
