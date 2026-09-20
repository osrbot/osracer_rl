"""A failed native child must not silently qualify artifacts from another run."""
from copy import deepcopy
from racing.qualify import results_match


def test_reuse_requires_full_inputs_and_exact_seed_order():
    identities={'checkpoint_sha256':'checkpoint','runtime_source_sha256':{'policy.py':'source'}}
    conditions={'seconds':120.,'lidar_noise':0.,'lidar_dropout':0.,'lidar_latency':0.,
                'opponent_speed':2.8,'opponent_gap':3.,'start_s':0.}
    rows=[dict(identities,seed=seed,evaluation_conditions=dict(conditions,recorded=i==0))
          for i,seed in enumerate([81,82])]
    assert results_match(rows,[81,82],identities,conditions,record=True)
    assert not results_match(rows,[82,81],identities,conditions,record=True)
    assert not results_match(rows,[81,82],identities,dict(conditions,seconds=160.),record=True)
    assert not results_match(rows,[81,82],identities,dict(conditions,lidar_latency=.03),record=True)
    for name, value in [('opponent_speed',5.),('opponent_gap',1.5),('start_s',20.)]:
        assert not results_match(rows,[81,82],identities,dict(conditions,**{name:value}),record=True)
    assert not results_match(rows,[81,82],dict(identities,checkpoint_sha256='new'),conditions,record=True)
    old=deepcopy(rows);old[0].pop('evaluation_conditions')
    assert not results_match(old,[81,82],identities,conditions,record=True)
    assert not results_match(rows,[81,82],identities,conditions,record=False)
