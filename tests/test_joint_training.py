import hashlib
import pytest
from racing.train_joint import aggregate,candidate,vector,audit_trial


def row(drift=.2,failure='timeout',**extra):
    result=dict(failure=failure,collision_steps=0,offroad_steps=0,duration_s=30.,progress_m=100.,
                max_continuous_drift_duration_s=drift,effective_overtakes=1,lap_fraction=.5,**extra)
    return result


def test_collision_and_missing_trials_cannot_win_over_clean_segments():
    valid=aggregate([row(),row()],2)
    failed=aggregate([row(drift=2.,failure='collision')],2)
    incomplete=aggregate([row(drift=2.)],2)
    assert valid['score']>failed['score']
    assert valid['score']>incomplete['score']
    assert not failed['all_segments_clean']


def test_worst_seed_drift_matters_more_than_an_impressive_peak():
    reliable=aggregate([row(.15),row(.15)],2)
    uneven=aggregate([row(2.),row(0.)],2)
    assert reliable['score']>uneven['score']
    assert reliable['all_continuous_drift'] and not uneven['all_continuous_drift']


def test_candidate_mapping_preserves_unsearched_parameters():
    prior={'parameters':[7.2,.85,.26,1.05,.65,.21,5.5,5.,.1,.15,3.5,.3],
           'controls':{'steering_bias':.1},'actor_type':'closed_loop'}
    result=candidate(prior,[7.5,3.,.32,.8,.2,4.2])
    assert vector(result).tolist()==[7.5,3.,.32,.8,.2,4.2]
    assert result['parameters'][1:10]==prior['parameters'][1:10]
    assert result['controls']['steering_bias']==.1
    assert prior['parameters'][0]==7.2


def test_local_donuts_and_missing_metrics_cannot_beat_real_racing():
    normal=aggregate([row(.15)]*6,6)
    donut=row(2.);donut.update(progress_m=0.,lap_fraction=0.,effective_overtakes=0)
    assert aggregate([donut]*6,6)['score']<normal['score']
    no_pass=row(2.);no_pass['effective_overtakes']=0
    assert aggregate([no_pass]*6,6)['score']<normal['score']
    assert not aggregate([{}]*6,6)['all_segments_clean']


def test_resume_recomputes_score_and_requires_unchanged_trace(tmp_path,monkeypatch):
    monkeypatch.setattr('racing.verify.audit_trace',lambda *_:{'errors':[]})
    trace=tmp_path/'trace.json';trace.write_text('[]')
    values=[7.2,3.5,.3,.6,.2,3.8]
    episodes=[]
    for engine in ['isaac','mujoco']:
        episode=row();episode.update(engine=engine,seed=0,trace_path=str(trace),
                                      trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        episodes.append(episode)
    saved={'variables':values,'episodes':episodes,'contract_sha256':'contract','score':999999.}
    assert audit_trial(saved,values,[0],None,'contract')['score']==aggregate(episodes,2)['score']
    with pytest.raises(RuntimeError,match='contract'):
        audit_trial(saved,values,[0],None,'changed')
    repeated=dict(saved,episodes=[episodes[0],episodes[0]])
    with pytest.raises(RuntimeError,match='identities'):
        audit_trial(repeated,values,[0],None,'contract')
    partial=dict(saved,episodes=[episodes[0]])
    with pytest.raises(RuntimeError,match='incomplete'):
        audit_trial(partial,values,[0],None,'contract')
    trace.write_text('[{}]')
    with pytest.raises(RuntimeError,match='trace'):
        audit_trial(saved,values,[0],None,'contract')
