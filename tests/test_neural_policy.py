import numpy as np
import pytest

from racing.control.neural_policy import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    ActorCritic,
    NeuralPolicy,
    action_to_actuators,
    actuators_to_action,
    encode_observation,
    export_onnx,
    load_checkpoint,
    save_checkpoint,
    torch,
)
from racing.runtime.ppo import _evaluation_summary, evaluation_rank


def observation():
    return {
        'wheel_vel': np.array([10.,10.,10.,-10.]),
        'steer_pos': np.array([.1,.1]),
        'lidar': {'ranges':np.full(361,5.),'valid':np.ones(361,bool),'age':.02},
    }


def test_neural_observation_and_actuator_contract():
    encoded=encode_observation(observation())
    assert encoded.shape==(OBSERVATION_SIZE,)
    assert encoded.dtype==np.float32
    assert OBSERVATION_SIZE==370
    np.testing.assert_array_equal(encoded[-2:],[-1.,0.])
    np.testing.assert_array_equal(
        encode_observation(observation(),[.25,-.5])[-2:],[.25,-.5])
    wheels,steering=action_to_actuators([0.,1.])
    assert wheels.shape==(4,)
    assert steering.shape==(2,)
    assert np.isfinite(wheels).all() and np.isfinite(steering).all()
    np.testing.assert_allclose(
        actuators_to_action(action_to_actuators([.25,-.4])), [.25,-.4], atol=2e-3)


def test_invalid_lidar_sector_uses_a_single_sentinel_feature():
    sample=observation()
    sample['lidar']['valid'][:3]=False
    encoded=encode_observation(sample)
    assert encoded[6] == -1.
    assert encoded[7] == -1.
    assert encoded[9] == pytest.approx(5./15.)


def test_pt_checkpoint_round_trip_and_actor_inference(tmp_path):
    torch.manual_seed(3);model=ActorCritic()
    checkpoint=tmp_path/'policy.pt'
    save_checkpoint(checkpoint,model,{'max_speed_m_s':8.,'iteration':4})
    restored,payload=load_checkpoint(checkpoint)
    assert payload['format']=='osracer.ppo.pt.v4'
    assert payload['model']['quantized_actions'] is True
    sample=torch.zeros(2,OBSERVATION_SIZE)
    torch.testing.assert_close(model.deterministic(sample),restored.deterministic(sample))
    action=restored.deterministic(torch.randn(4,OBSERVATION_SIZE))
    torch.testing.assert_close(action*32.,torch.round(action*32.))
    target=torch.tensor([[.02,-.02],[.04,-.04]])
    torch.testing.assert_close(restored.quantize_action(target),
                               torch.tensor([[.03125,-.03125],[.03125,-.03125]]))
    restored.train()
    sampled,log_probability,_=restored.sample(sample)
    evaluated,_,_=restored.evaluate(sample,sampled)
    torch.testing.assert_close(sampled*32.,torch.round(sampled*32.))
    torch.testing.assert_close(log_probability,evaluated)
    actor=NeuralPolicy(checkpoint)
    wheels,steering=actor.action(observation())
    assert wheels.shape==(4,) and steering.shape==(2,)


def test_v2_checkpoint_is_upgraded_without_changing_its_actor(tmp_path):
    torch.manual_seed(4)
    old=ActorCritic(observation_size=368,residual_actions=False,
                    quantized_actions=False)
    payload={
        'format':'osracer.ppo.pt.v2',
        'observation':{'size':368},
        'model':{'hidden_sizes':[256,128]},
        'state_dict':old.state_dict(),
        'metadata':{},
    }
    checkpoint=tmp_path/'v2.pt';torch.save(payload,checkpoint)
    restored,_=load_checkpoint(checkpoint)
    old_input=torch.randn(3,368)
    new_input=torch.cat((old_input,torch.zeros(3,2)),dim=1)
    torch.testing.assert_close(
        old.deterministic(old_input),restored.deterministic(new_input))


def test_multiseed_rank_prefers_safety_then_faster_completed_laps():
    completed=lambda steps,reward: {
        'reward':reward,'progress_m':270.8,'termination':'completed',
        'valid_lap':True,'episode_length':steps,'overtakes':1}
    safe=_evaluation_summary([completed(4000,430.),completed(4100,429.)])
    faster=_evaluation_summary([completed(3600,433.),completed(3700,432.)])
    partial=_evaluation_summary([completed(3500,434.),{
        'reward':-20.,'progress_m':100.,'termination':'collision',
        'valid_lap':False,'episode_length':1800,'overtakes':0}])
    assert evaluation_rank(faster)>evaluation_rank(safe)>evaluation_rank(partial)
    assert safe['overtakes']==1.
    assert faster['mean_progress_speed_m_s']>safe['mean_progress_speed_m_s']


def test_pt_exports_to_checked_dynamic_batch_onnx(tmp_path):
    checkpoint=tmp_path/'policy.pt';output=tmp_path/'policy.onnx'
    save_checkpoint(checkpoint,ActorCritic(),{'max_speed_m_s':10.})
    assert export_onnx(checkpoint,output)==output
    assert output.is_file() and output.stat().st_size>0
    import onnx
    graph=onnx.load(str(output))
    assert graph.graph.input[0].name=='observation'
    assert graph.graph.output[0].name=='action'
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param=='batch'
    assert graph.graph.input[0].type.tensor_type.shape.dim[-1].dim_value==OBSERVATION_SIZE
    assert graph.graph.output[0].type.tensor_type.shape.dim[0].dim_param=='batch'
    assert graph.graph.output[0].type.tensor_type.shape.dim[-1].dim_value==ACTION_SIZE
