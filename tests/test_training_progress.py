from io import StringIO

from racing.runtime.progress import PPOProgress, TrainingProgress, print_evaluation


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def episode():
    return {
        'engine': 'mujoco', 'track': 'bahrain', 'score': 348.651,
        'valid_lap': True, 'completed': True, 'failure': None,
        'duration_s': 45.25, 'lap_time_s': 45.25, 'progress_m': 270.8,
        'lap_fraction': 1.0003, 'mean_speed_m_s': 5.95,
        'peak_speed_m_s': 8.79, 'effective_overtakes': 1,
        'max_rear_slip_deg': 9.16, 'max_continuous_drift_duration_s': 0.0,
        'collision_steps': 0, 'offroad_steps': 0,
        'initial_states': [{'large': 'private diagnostic payload'}],
    }


def test_candidate_progress_is_readable_and_omits_raw_episode_payload():
    stream=StringIO();clock=Clock()
    progress=TrainingProgress('mujoco','bahrain',4,12,stream=stream,clock=clock)
    clock.now=102.0
    progress.candidate(0,0,episode(),2715,1.5,2.0,348.651)
    output=stream.getvalue()
    assert 'Training iteration 1/48' in output
    assert '1810 steps/s (collection: 1.500s)' in output
    assert 'valid yes' in output
    assert 'Total timesteps' in output
    assert 'ETA' in output
    assert 'initial_states' not in output
    assert 'private diagnostic payload' not in output


def test_joint_progress_uses_same_layout_and_marks_cached_trials():
    stream=StringIO();clock=Clock()
    progress=TrainingProgress('isaac + mujoco','bahrain',2,3,stream=stream,clock=clock)
    item={'score':1200.,'episodes':[episode(),episode()],
          'episodes_evaluated':2,'episodes_expected':2,
          'all_segments_clean':True,'all_forward_progress':True,
          'all_overtakes':True,'all_continuous_drift':False,
          'minimum_continuous_drift_s':0.05}
    progress.joint_candidate(0,0,item,.02,.03,1200.,reused=True)
    output=stream.getvalue()
    assert 'Training iteration 1/6' in output
    assert 'Isaac Sim + MuJoCo / racing/bahrain' in output
    assert 'cached trial (audit: 0.02s)' in output
    assert 'overtake yes' in output


def test_evaluation_summary_omits_raw_payload():
    stream=StringIO()
    print_evaluation(episode(),1,1,1.0,stream)
    output=stream.getvalue()
    assert 'Evaluation episode 1/1' in output
    assert '2715 steps/s  ·  evaluation 1.000s' in output
    assert 'initial_states' not in output


def test_ppo_progress_reports_optimizer_and_reward_metrics():
    stream=StringIO();clock=Clock()
    progress=PPOProgress('demo','mujoco','bahrain',1000,stream=stream,clock=clock)
    clock.now=101.84
    metrics={
        'steps_per_second':71370.,'collection_seconds':1.762,'learning_seconds':.074,
        'value_loss':.0058,'surrogate_loss':-.0092,'entropy':1.4,'imitation_loss':.002,
        'action_noise_std':.63,'mean_reward':-5.75,'mean_episode_length':683.24,
        'reward_progress':.0311,'reward_alive':.3546,'reward_track':-.1,
        'reward_action_rate':-.0217,'reward_overtake':.01,'reward_completion':.02,
        'reward_collision':-.03,'reward_offroad':-.04,'reward_stalled':-.01,
        'termination_completed':10.,
        'termination_timeout':20.,'termination_collision':30.,'termination_offroad':20.,
        'termination_stalled':20.,'total_steps':3276800,'iteration_seconds':1.84,
    }
    progress.iteration(23,metrics)
    output=stream.getvalue()
    assert 'Learning iteration 24/1000' in output
    assert '71370 steps/s' in output
    assert 'Value function loss' in output and '0.0058' in output
    assert 'Surrogate loss' in output and '-0.0092' in output
    assert 'Total timesteps' in output and '3276800' in output
