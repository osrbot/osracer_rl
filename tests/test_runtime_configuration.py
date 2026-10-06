from types import SimpleNamespace

import pytest

from racing.runtime.artifacts import RunLayout
from racing.runtime.configuration import (
    apply_joint_overrides,
    apply_play_overrides,
    apply_ppo_overrides,
    apply_run_overrides,
    load_training_profile,
    task_track,
)
from racing.runtime.play import training_defaults


def training_args():
    return SimpleNamespace(engine=None,track='bahrain',run_id=None,checkpoint=None,seed=73,
        generations=4,population=12,seconds=120.,episodes=1,opponent_speed=2.8,
        opponent_gap=3.,no_tensorboard=False,tensorboard_dir=None)


def test_packaged_profiles_hold_algorithm_details_out_of_the_command():
    default=load_training_profile('default')
    quick=load_training_profile('quick')
    assert (default.generations,default.population,default.episode_length_s)==(4,12,120.)
    assert (quick.generations,quick.population,quick.episode_length_s)==(1,4,30.)
    assert default.algorithm=='ppo'
    assert (default.iterations,default.steps_per_iteration)==(1000,2048)
    assert (default.evaluation_episodes,quick.evaluation_episodes)==(3,1)
    assert (default.refinement_cycles,default.refinement_epochs)==(2,8)
    assert (quick.refinement_cycles,quick.refinement_epochs)==(0,0)
    assert default.track=='bahrain'
    pace=load_training_profile('pace')
    assert (pace.reward_time_cost,pace.reward_pace_weight,
            pace.reward_failure_horizon_scale)==(.01,.25,1.)


def test_asap_style_training_overrides_map_to_runtime_fields():
    args=training_args()
    apply_run_overrides(args,['+simulator=mujoco','+task=racing/austin',
        'experiment_name=demo','train.generations=8','env.episode_length_s=90',
        'logger.tensorboard=false'],('isaac','mujoco'))
    assert args.engine=='mujoco'
    assert args.track=='austin'
    assert args.run_id=='demo'
    assert args.generations==8
    assert args.seconds==90.
    assert args.no_tensorboard is True


def test_invalid_composed_values_are_rejected_early():
    args=training_args()
    with pytest.raises(ValueError,match='unknown override'):
        apply_run_overrides(args,['reward_magic=3'],('isaac','mujoco'))
    with pytest.raises(ValueError,match='task must'):
        task_track('locomotion/bahrain')


def test_ppo_overrides_map_to_optimizer_fields():
    args=SimpleNamespace(engine=None,track='bahrain',run_id=None,seed=73,device='auto',
        iterations=1000,steps_per_iteration=2048,learning_epochs=5,mini_batches=4,
        learning_rate=.0003,checkpoint_interval=50,seconds=120.,opponent_speed=2.8,
        opponent_gap=3.,evaluation_episodes=3,refinement_cycles=2,
        refinement_epochs=8,no_tensorboard=False,tensorboard_dir=None,
        reward_time_cost=.01,reward_pace_weight=0.,reward_failure_horizon_scale=0.)
    apply_ppo_overrides(args,['+simulator=mujoco','+task=racing/austin',
        'experiment_name=demo','train.iterations=12','train.steps_per_iteration=512',
        'env.evaluation_episodes=5','train.refinement_cycles=3',
        'reward.pace_weight=.25','reward.failure_horizon_scale=1',
        'device=cpu'],('isaac','mujoco'))
    assert (args.engine,args.track,args.run_id)==('mujoco','austin','demo')
    assert (args.iterations,args.steps_per_iteration,args.device)==(12,512,'cpu')
    assert args.evaluation_episodes==5
    assert args.refinement_cycles==3
    assert (args.reward_pace_weight,args.reward_failure_horizon_scale)==(.25,1.)


def test_playback_and_joint_use_the_same_names():
    play=SimpleNamespace(engine=None,track=None,run_id=None,checkpoint=None,seed=0,
        seconds=None,opponent_speed=None,opponent_gap=None)
    apply_play_overrides(play,['experiment_name=demo','+simulator=isaac',
        '+task=racing/bahrain'],('isaac','mujoco'))
    assert (play.run_id,play.engine,play.track)==('demo','isaac','bahrain')
    joint=training_args();joint.seeds=[0,4,5]
    apply_joint_overrides(joint,['checkpoint=prior.json','experiment_name=joint',
        'eval.seeds=1,2'])
    assert joint.checkpoint=='prior.json'
    assert joint.run_id=='joint'
    assert joint.seeds==[1,2]


def test_playback_recovers_training_configuration(tmp_path):
    layout=RunLayout.open('demo',tmp_path)
    (layout.config/'training_isaac_austin.json').write_text(
        '{"engine":"isaac","track":"austin","seconds":75.0}')
    assert training_defaults(layout)=={'engine':'isaac','track':'austin','seconds':75.0}
