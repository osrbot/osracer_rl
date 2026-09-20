import unittest
import numpy as np
from racing.policy import RacingPolicy
from racing.policy_bundle import configuration,make_actor,actor_source_hashes,actor_version
from racing.sensors import LidarSensor
from tests.test_racing_contracts import StraightTrack,state


class PolicyBundleTests(unittest.TestCase):
    def test_legacy_bundle_matches_baseline(self):
        a,b=RacingPolicy(),make_actor()
        sensor=LidarSensor(StraightTrack())
        for tick in range(20):
            obs=sensor.observe(state(),([0,0,.2],np.eye(3)),tick=tick)
            for x,y in zip(a.action(obs),b.action(obs)):
                np.testing.assert_array_equal(x,y)

    def test_configuration_and_sources_are_explicit(self):
        spec={'actor_type':'closed_loop','controls':{'max_boost_seconds':.5,'steering_bias':.14},'safety_supervisor':True}
        self.assertIn('footprint-safety',actor_version(spec))
        self.assertIn('racing/safety.py',actor_source_hashes(spec))
        self.assertIn('racing/odometry.py',actor_source_hashes(spec))
        self.assertNotIn('racing/isaac_env.py',actor_source_hashes(spec))
        with self.assertRaises(ValueError):configuration({'actor_type':'unknown'})
        with self.assertRaises(ValueError):configuration({'actor_type':'reactive','controls':{'max_boost_seconds':.5}})

    def test_reactive_safety_freezes_its_motion_dependencies(self):
        sources=actor_source_hashes({'actor_type':'reactive','safety_supervisor':True})
        self.assertIn('racing/safety_motion.py',sources)
        self.assertIn('racing/odometry.py',sources)
        self.assertNotIn('racing/policy_closed_loop.py',sources)
        self.assertNotIn('racing/safety_motion.py',actor_source_hashes())

    def test_supervision_handles_initial_missing_scan(self):
        actor=make_actor(spec={'actor_type':'closed_loop','safety_supervisor':True})
        obs=LidarSensor(StraightTrack(),latency_s=.05).observe(state(),([0,0,.2],np.eye(3)),tick=0)
        wheel,steer=actor.action(obs)
        np.testing.assert_array_equal(wheel,0.)
        self.assertTrue(np.isfinite(steer).all())
        actor.reset()


if __name__=='__main__':unittest.main()
