from pathlib import Path
from types import SimpleNamespace

import pytest

from racing.control.neural_policy import ActorCritic,save_checkpoint
from racing.runtime import benchmark
from racing.runtime.artifacts import RunLayout


def checkpoint(path: Path,track="bahrain",engine="mujoco"):
    save_checkpoint(path,ActorCritic(),{
        "task":f"racing/{track}","simulator":engine,"iteration":3,
        "total_steps":2048,"metrics":{"evaluation":{
            "valid_lap":True,"valid_laps":2,"evaluation_episodes":2,
            "episode_length":3600.,"reward":430.,"progress_m":270.8,
            "termination":"completed","overtakes":1.,"episodes":[],
        }},
    })


def arguments(root):
    return SimpleNamespace(runs_root=str(root),dry_run=False,no_resume=False,
                           fail_fast=False,no_open=True)


def test_season_plan_uses_catalog_order_and_validates_selection():
    plan=benchmark._track_plan(2025,None)
    assert len(plan)==24
    assert plan[0]=={"round":1,"track":"melbourne"}
    assert plan[-1]=={"round":24,"track":"yas_marina"}
    assert benchmark._track_plan(2025,"bahrain,melbourne")==[
        {"round":4,"track":"bahrain"},{"round":1,"track":"melbourne"}]
    with pytest.raises(ValueError,match="duplicates"):
        benchmark._track_plan(2025,"bahrain,bahrain")
    with pytest.raises(ValueError,match="unknown"):
        benchmark._track_plan(2025,"not-a-track")


def test_completed_policy_requires_pt_onnx_and_matching_identity(tmp_path):
    layout=RunLayout.open("child",tmp_path)
    checkpoint(layout.checkpoints/"policy.pt")
    assert benchmark._checkpoint_result("child",tmp_path,"bahrain","mujoco") is None
    (layout.checkpoints/"policy.onnx").write_bytes(b"checked-by-exporter")
    result=benchmark._checkpoint_result("child",tmp_path,"bahrain","mujoco")
    assert result["valid_lap_rate"]==1.
    assert result["qualified"] is True
    assert result["lap_time_s"]==60.
    assert benchmark._checkpoint_result("child",tmp_path,"austin","mujoco") is None


def test_train_creates_independent_runs_and_resume_skips_completed(monkeypatch,tmp_path):
    parent=tmp_path/"parent.pt";checkpoint(parent)
    commands=[]
    def fake_run(command,log_path):
        commands.append(command)
        values={token.lstrip("+").split("=",1)[0]:token.split("=",1)[1]
                for token in command if "=" in token}
        runs_root=Path(command[command.index("--runs-root")+1])
        layout=RunLayout.open(values["experiment_name"],runs_root)
        checkpoint(layout.checkpoints/"policy.pt",values["task"].split("/")[-1],
                   values["simulator"])
        (layout.checkpoints/"policy.onnx").write_bytes(b"onnx")
        return 0
    monkeypatch.setattr(benchmark,"_run_child",fake_run)
    values={"benchmark_name":"suite","season":"2025","tracks":"melbourne,bahrain",
            "checkpoint":str(parent),"simulator":"mujoco","train":"quick"}
    summary=benchmark._train(arguments(tmp_path),values,["+train=quick"])
    assert summary["counts"]["completed"]==2
    assert [row["run_id"] for row in summary["tracks"]]==[
        "suite-r01-melbourne","suite-r04-bahrain"]
    assert all("checkpoint="+str(parent.resolve()) in command for command in commands)
    assert "+task=racing/melbourne" in commands[0]
    assert "+task=racing/bahrain" in commands[1]
    assert commands[0]!=commands[1]
    commands.clear()
    resumed=benchmark._train(arguments(tmp_path),values,["+train=quick"])
    assert resumed["counts"]["completed"]==2
    assert commands==[]
    assert (tmp_path/"suite"/"reports"/"benchmark_summary.csv").is_file()


def test_benchmark_profile_is_packaged():
    from racing.runtime.configuration import load_training_profile
    profile=load_training_profile("benchmark")
    assert profile.iterations==20
    assert profile.evaluation_episodes==3
    assert profile.refinement_cycles==1
    assert profile.imitation_steps==14400
