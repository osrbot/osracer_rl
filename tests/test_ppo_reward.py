import numpy as np
import pytest

from racing.runtime import ppo


class Track:
    length=10.
    width=2.
    has_elevation=False

    def __init__(self):
        self.hints=[]

    def project(self,location,s_hint=None):
        self.hints.append(s_hint)
        return float(location[0])%self.length,float(location[1]),0.


class Sensor:
    def observe(self,state,*_,**__):
        return {'wheel_vel':np.zeros(4),'steer_pos':np.zeros(2),
                'lidar':{'ranges':np.full(361,5.),'valid':np.ones(361,bool),'age':0.}}


class Native:
    def __init__(self,collision=False):
        self.collision=collision

    @staticmethod
    def state(x):
        return {'x':x,'y':0.,'vx':4.,'vy':0.,'z':.05,'roll':0.,'pitch':0.,
                'collision':False,'lidar_pose':{}}

    def reset(self,seed,starts):
        return [self.state(starts[0][0]),self.state(starts[1][0])]

    def step(self,actions):
        first=self.state(.1);first['collision']=self.collision
        return [first,self.state(3.1)]


def make_task(monkeypatch,collision=False,pace_weight=.25,corner_risk_weight=0.):
    monkeypatch.setattr(ppo,'LidarSensor',lambda *args,**kwargs:object())
    monkeypatch.setattr(ppo,'make_sensor',lambda *args,**kwargs:Sensor())
    task=ppo.RacingPPOEnv(Native(collision),Track(),seconds=10/60,
                          time_cost=.01,pace_weight=pace_weight,
                          failure_horizon_scale=1.,
                          corner_risk_weight=corner_risk_weight)
    task.reset(0)
    task.opponent.action=lambda observation:(np.zeros(4),np.zeros(2))
    return task


def test_pace_reward_uses_ordered_progress_and_reports_whole_episode_speed(monkeypatch):
    task=make_task(monkeypatch)
    _,_,done,info=task.step([0.,0.])
    assert done is False
    assert task.track.hints[-2:]==[0.,3.]
    assert info['rewards']['pace']>0.
    assert info['mean_speed_m_s']==pytest.approx(4.)
    assert info['mean_progress_speed_m_s']==pytest.approx(6.)


def test_failure_pays_unspent_time_cost_instead_of_rewarding_early_exit(monkeypatch):
    task=make_task(monkeypatch,collision=True,pace_weight=0.)
    task.step_count=3
    _,_,done,info=task.step([0.,0.])
    assert done is True
    paid_time_cost=task.step_count*info['rewards']['alive']+info['rewards']['failure_horizon']
    assert paid_time_cost==pytest.approx(-task.max_steps*task.time_cost)
    assert info['termination']=='collision'


def test_corner_risk_penalizes_fast_sharp_turn_without_touching_straights(monkeypatch):
    task=make_task(monkeypatch,corner_risk_weight=.05)
    _,_,_,turn=task.step([0.,.75])
    assert turn['rewards']['corner_risk']<0.
    task=make_task(monkeypatch,corner_risk_weight=.05)
    _,_,_,straight=task.step([0.,.25])
    assert straight['rewards']['corner_risk']==0.


def test_quantized_refinement_interpolates_from_last_safe_state():
    base={'weight':ppo.torch.tensor([0.,2.]),'counter':ppo.torch.tensor(1)}
    candidate={'weight':ppo.torch.tensor([4.,6.]),'counter':ppo.torch.tensor(2)}
    state=ppo._interpolate_state(base,candidate,.25)
    ppo.torch.testing.assert_close(state['weight'],ppo.torch.tensor([1.,3.]))
    assert state['counter'].item()==2
