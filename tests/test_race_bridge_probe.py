import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.race_bridge_probe import controls, run_case


class TrackFixture:
    length, width = 100., 1.5
    metadata = {'bridge': {'upper_s': 30., 'lower_s': 10., 'surface_height': .9,
                           'elevated_start_s': 20., 'elevated_end_s': 40.}}

    def project(self, xy, **kwargs):
        return xy[0] % self.length, xy[1], 0.

    def at(self, s):
        return np.array([s, 0.]), 0.

    def elevation_at(self, s):
        return .9 if 20. < s < 40. else 0.


def state(x):
    return dict(x=x, y=0., z=.045, yaw=0., roll=0., pitch=0., collision=False)


class TestBridgeProbe(unittest.TestCase):
    def test_controls_unwrap_progress_and_stop_the_lower_vehicle(self):
        track = TrackFixture()
        states = [state(.01), state(3.)]
        before = copy.deepcopy(states)
        actions, progress, _ = controls(track, states, [99.99, 3.], 35., park_lower_after=2.9)
        self.assertAlmostEqual(progress[0], 100.01)
        self.assertEqual(actions[0][0], [35., 35., 35., -35.])
        self.assertEqual(actions[1][0], [0., 0., 0., -0.])
        self.assertEqual(states, before)

    def test_stalled_fixture_cannot_pass_and_trace_is_explicitly_non_policy(self):
        import json

        class StalledEnv:
            track = TrackFixture()

            def reset(self, **kwargs):
                self.states = [state(29.3), state(9.3)]
                return self.states

            def step(self, actions):
                return self.states

        with tempfile.TemporaryDirectory() as directory:
            report = run_case(StalledEnv(), 'crossing', Path(directory)/'probe', max_seconds=.1)
            self.assertFalse(report['passed'])
            self.assertIn('Fixture did not finish within time limit', report['failures'])
            trace = json.loads(Path(report['trace_path']).read_text())
            self.assertFalse(trace['policy_evaluation'])
            self.assertEqual(trace['fixture_type'], 'privileged_native_bridge_geometry_probe')
            self.assertEqual(len(trace['samples']), 6)


if __name__ == '__main__':
    unittest.main()
