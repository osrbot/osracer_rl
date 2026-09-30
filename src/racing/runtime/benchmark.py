"""Train, resume, inspect, and play one PPO policy for every season track."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
from pathlib import Path
import re
import subprocess
import sys

from ..control.neural_policy import load_checkpoint
from ..simulators import available_simulators
from ..tracks import catalog, season_tracks
from .artifacts import RunLayout
from .configuration import composed_values,load_training_profile


SCHEMA_VERSION = 1
COMMANDS = {"train", "status", "play"}
MANAGED_KEYS = {"benchmark_name", "season", "tracks", "checkpoint", "attempts", "seed",
                "task", "experiment_name", "logger.tensorboard_dir"}
PPO_KEYS = {
    "train", "algorithm", "simulator", "device",
    "train.iterations", "train.steps_per_iteration", "train.learning_epochs",
    "train.mini_batches", "train.learning_rate", "train.checkpoint_interval",
    "train.imitation_steps", "train.imitation_epochs",
    "train.refinement_cycles", "train.refinement_epochs",
    "env.episode_length_s", "env.evaluation_episodes",
    "task.opponent_speed_m_s", "task.opponent_gap_m", "logger.tensorboard",
}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_sha256(profile: str) -> str:
    digest=hashlib.sha256()
    for path in (Path(__file__),Path(__file__).with_name("ppo.py"),
                 Path(__file__).parents[1]/"control"/"neural_policy.py"):
        digest.update(path.read_bytes())
    digest.update(resources.files("racing.config.training")
                  .joinpath(f"{profile}.toml").read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    RunLayout.write_json(path,value)


def _state_path(layout: RunLayout) -> Path:
    return layout.reports/"benchmark_state.json"


def _summary_paths(layout: RunLayout) -> tuple[Path,Path]:
    return (layout.reports/"benchmark_summary.json",
            layout.reports/"benchmark_summary.csv")


def _track_plan(season: int, selected: str | None) -> list[dict]:
    records=[{"round":int(row["round"]),"track":row["track_id"]}
             for row in season_tracks(season)]
    known=set(catalog()["tracks"])
    if not selected:return records
    requested=[value.strip() for value in selected.split(",") if value.strip()]
    if len(requested)!=len(set(requested)):
        raise ValueError("tracks contains duplicates")
    unknown=sorted(set(requested)-known)
    if unknown:raise ValueError(f"unknown track(s): {', '.join(unknown)}")
    by_track={row["track"]:row for row in records}
    outside=sorted(set(requested)-set(by_track))
    if outside:
        raise ValueError(f"track(s) are not in season {season}: {', '.join(outside)}")
    return [by_track[track] for track in requested]


def _checkpoint_result(run_id: str, runs_root: str | Path | None,
                       track: str, engine: str) -> dict | None:
    """Return a verified terminal result; partial checkpoints are not resumable."""
    layout=RunLayout.path(run_id,runs_root)
    checkpoint=layout.checkpoints/"policy.pt"
    onnx=layout.checkpoints/"policy.onnx"
    if not (checkpoint.is_file() and onnx.is_file()):return None
    try:
        _,payload=load_checkpoint(checkpoint,"cpu")
    except (OSError,RuntimeError,TypeError,ValueError):
        return None
    metadata=payload.get("metadata",{})
    if metadata.get("task")!=f"racing/{track}" or metadata.get("simulator")!=engine:
        return None
    evaluation=metadata.get("metrics",{}).get("evaluation",{})
    episodes=evaluation.get("episodes",[])
    count=int(evaluation.get("evaluation_episodes",len(episodes) or 1))
    valid=int(evaluation.get("valid_laps",int(bool(evaluation.get("valid_lap")))))
    return {
        "checkpoint":str(checkpoint),"checkpoint_sha256":_sha256(checkpoint),
        "onnx":str(onnx),"onnx_sha256":_sha256(onnx),
        "iteration":int(metadata.get("iteration",0)),
        "total_steps":int(metadata.get("total_steps",0)),
        "evaluation_episodes":count,"valid_laps":valid,
        "valid_lap_rate":valid/max(1,count),
        "qualified":bool(count and valid==count),
        "reward":evaluation.get("reward"),
        "progress_m":evaluation.get("progress_m"),
        "episode_length":evaluation.get("episode_length"),
        "lap_time_s":((float(evaluation["episode_length"])/60.)
                      if evaluation.get("valid_lap") and evaluation.get("episode_length") is not None
                      else None),
        "termination":evaluation.get("termination"),
        "overtakes":evaluation.get("overtakes"),
    }


def _write_summary(layout: RunLayout,state: dict) -> dict:
    rows=[]
    for item in state["tracks"]:
        row={key:item.get(key) for key in
             ("round","track","run_id","status","attempt","started_at","completed_at","error")}
        row.update(item.get("result") or {})
        rows.append(row)
    counts={status:sum(row["status"]==status for row in rows)
            for status in ("pending","running","completed","unqualified","failed")}
    counts["qualified"]=sum(bool(row.get("qualified")) for row in rows)
    summary={
        "schema_version":SCHEMA_VERSION,
        "benchmark_name":state["benchmark_name"],"season":state["season"],
        "simulator":state["simulator"],"training_profile":state["training_profile"],
        "base_seed":state["base_seed"],"max_attempts":state["max_attempts"],
        "protocol_sha256":state["protocol_sha256"],
        "source_checkpoint_sha256":state["source_checkpoint_sha256"],
        "created_at":state["created_at"],"updated_at":_now(),
        "counts":{"total":len(rows),**counts},"tracks":rows,
    }
    json_path,csv_path=_summary_paths(layout);_atomic_json(json_path,summary)
    fields=["round","track","run_id","status","attempt","qualified","valid_laps",
            "evaluation_episodes","valid_lap_rate","lap_time_s","reward",
            "progress_m","episode_length","overtakes","termination","total_steps",
            "checkpoint","checkpoint_sha256","onnx","onnx_sha256","error"]
    temporary=csv_path.with_suffix(".csv.tmp")
    with temporary.open("w",newline="",encoding="utf-8") as target:
        writer=csv.DictWriter(target,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    temporary.replace(csv_path)
    return summary


def _print_summary(summary: dict) -> None:
    counts=summary["counts"]
    print(f"Benchmark {summary['benchmark_name']} · {summary['simulator'].upper()} · "
          f"season {summary['season']} · completed {counts['completed']}/{counts['total']} · "
          f"unqualified {counts['unqualified']} · failed {counts['failed']}")
    print("rnd  track             status     valid       lap       overtakes  run")
    for row in summary["tracks"]:
        valid=(f"{row.get('valid_laps',0)}/{row.get('evaluation_episodes',0)}"
               if row.get("evaluation_episodes") is not None else "--")
        lap=f"{row['lap_time_s']:.2f}s" if row.get("lap_time_s") is not None else "--"
        over=(f"{row['overtakes']:.2f}" if isinstance(row.get("overtakes"),(float,int)) else "--")
        print(f"{row['round']:>3}  {row['track']:<16} {row['status']:<11} "
              f"{valid:<11} {lap:<9} {over:<10} {row.get('run_id') or '--'}")


def _run_child(command: list[str],log_path: Path) -> int:
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line,end="",flush=True);log.write(line);log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
            raise


def _load_state(layout: RunLayout) -> dict:
    path=_state_path(layout)
    if not path.is_file():raise FileNotFoundError(f"benchmark not found: {layout.run_id}")
    return json.loads(path.read_text())


def _train(args,values: dict[str,str],forwarded: list[str]) -> dict:
    name=values.get("benchmark_name")
    if not name or not _NAME.fullmatch(name):
        raise ValueError("benchmark_name must use letters, digits, '.', '_' or '-'")
    checkpoint=Path(values.get("checkpoint","")).expanduser().resolve()
    if not checkpoint.is_file():raise ValueError("checkpoint must point to a PPO .pt file")
    # Validate the common parent before creating any campaign artifacts.
    load_checkpoint(checkpoint,"cpu")
    season=int(values.get("season","2025"))
    plan=_track_plan(season,values.get("tracks"))
    engine=values.get("simulator","mujoco")
    if engine not in available_simulators():
        raise ValueError(f"unknown simulator: {engine}")
    profile=values.get("train","benchmark")
    profile_config=load_training_profile(profile)
    base_seed=int(values.get("seed",profile_config.seed))
    max_attempts=int(values.get("attempts","3"))
    if max_attempts<1:raise ValueError("attempts must be positive")
    source_sha=_sha256(checkpoint)
    protocol_sha=_protocol_sha256(profile)
    layout=RunLayout.path(name,args.runs_root)
    if args.dry_run:
        print(json.dumps({"benchmark_name":name,"season":season,"simulator":engine,
            "training_profile":profile,"tracks":plan,"source_checkpoint":str(checkpoint),
            "source_checkpoint_sha256":source_sha,"protocol_sha256":protocol_sha,
            "base_seed":base_seed,"max_attempts":max_attempts,
            "forwarded_overrides":forwarded},ensure_ascii=False,indent=2))
        return {"dry_run":True,"tracks":plan}
    layout=RunLayout.open(name,args.runs_root)
    state_path=_state_path(layout)
    if state_path.is_file():
        if args.no_resume:raise ValueError(f"benchmark {name!r} already exists")
        state=_load_state(layout)
        frozen=(state["season"],state["simulator"],state["training_profile"],
                state["source_checkpoint_sha256"],state.get("protocol_sha256"),
                state.get("base_seed"),state.get("max_attempts"),state["forwarded_overrides"])
        requested=(season,engine,profile,source_sha,protocol_sha,base_seed,max_attempts,forwarded)
        if frozen!=requested:
            raise ValueError("resume arguments differ from the frozen benchmark protocol")
        requested_tracks=[row["track"] for row in plan]
        frozen_tracks=[row["track"] for row in state["tracks"]]
        if requested_tracks!=frozen_tracks:
            raise ValueError("resume track selection differs from the frozen benchmark protocol")
    else:
        state={"schema_version":SCHEMA_VERSION,"benchmark_name":name,"season":season,
            "simulator":engine,"training_profile":profile,
            "source_checkpoint":str(checkpoint),"source_checkpoint_sha256":source_sha,
            "protocol_sha256":protocol_sha,
            "base_seed":base_seed,"max_attempts":max_attempts,
            "forwarded_overrides":forwarded,"created_at":_now(),"updated_at":_now(),
            "tracks":[{"round":row["round"],"track":row["track"],"status":"pending",
                       "attempt":0,"run_id":None,"result":None,"error":None}
                      for row in plan]}
        _atomic_json(state_path,state);_write_summary(layout,state)
    for _ in range(max_attempts):
        for selection_index,item in enumerate(state["tracks"],1):
            if item["status"]=="completed":
                result=_checkpoint_result(item["run_id"],args.runs_root,item["track"],engine)
                if result is not None and result["qualified"]:
                    item["result"]=result;continue
                item["status"]=("unqualified" if result is not None else "failed")
                item["error"]=("policy did not pass every evaluation seed" if result is not None
                               else "completed artifacts failed validation")
            if int(item.get("attempt",0))>=max_attempts:continue
            attempt=int(item.get("attempt",0))+1
            base=f"{name}-r{item['round']:02d}-{item['track']}"
            run_id=base if attempt==1 else f"{base}-a{attempt:02d}"
            while RunLayout.path(run_id,args.runs_root).root.exists():
                attempt+=1;run_id=f"{base}-a{attempt:02d}"
            if attempt>max_attempts:continue
            item.update(status="running",attempt=attempt,run_id=run_id,
                        started_at=_now(),completed_at=None,error=None,result=None)
            state["updated_at"]=_now();_atomic_json(state_path,state);_write_summary(layout,state)
            tensorboard=layout.tensorboard/f"r{item['round']:02d}-{item['track']}"/f"a{attempt:02d}"
            child=[sys.executable,"-m","racing.runtime.train",
                   f"+simulator={engine}",f"+task=racing/{item['track']}",
                   f"experiment_name={run_id}",f"checkpoint={checkpoint}",
                   f"seed={base_seed+attempt-1}",
                   f"logger.tensorboard_dir={tensorboard}",*forwarded,
                   "--runs-root",str(layout.root.parent)]
            print(f"\nTraining track {selection_index}/{len(state['tracks'])} · "
                  f"round {item['round']} · attempt {attempt}/{max_attempts}: "
                  f"{item['track']} ({run_id})",flush=True)
            return_code=_run_child(child,layout.logs/f"{run_id}.log")
            result=_checkpoint_result(run_id,args.runs_root,item["track"],engine)
            if return_code==0 and result is not None and result["qualified"]:
                item.update(status="completed",completed_at=_now(),result=result,error=None)
            elif return_code==0 and result is not None:
                item.update(status="unqualified",completed_at=_now(),result=result,
                            error="policy did not pass every evaluation seed")
            else:
                item.update(status="failed",completed_at=_now(),result=result,
                    error=(f"training exited with status {return_code}" if return_code else
                           "training artifacts failed validation"))
            state["updated_at"]=_now();_atomic_json(state_path,state)
            summary=_write_summary(layout,state);_print_summary(summary)
            if item["status"]!="completed" and args.fail_fast:raise SystemExit(1)
    summary=_write_summary(layout,state);_print_summary(summary)
    json_path,csv_path=_summary_paths(layout)
    print("OSRACER_BENCHMARK_ARTIFACTS",json.dumps({
        "benchmark_name":name,"benchmark_dir":str(layout.root),
        "summary":str(json_path),"csv":str(csv_path),
        "tensorboard":str(layout.tensorboard)},ensure_ascii=False),flush=True)
    if any(row["status"]!="completed" for row in state["tracks"]):raise SystemExit(1)
    return summary


def _status(args,values: dict[str,str]) -> dict:
    name=values.get("benchmark_name")
    if not name:raise ValueError("status requires benchmark_name=<name>")
    layout=RunLayout.existing(name,args.runs_root);state=_load_state(layout)
    summary=_write_summary(layout,state);_print_summary(summary);return summary


def _play(args,values: dict[str,str]) -> None:
    name=values.get("benchmark_name");track=values.get("tracks") or values.get("track")
    if not name or not track:
        raise ValueError("play requires benchmark_name=<name> track=<track>")
    layout=RunLayout.existing(name,args.runs_root);state=_load_state(layout)
    matches=[row for row in state["tracks"] if row["track"]==track]
    if not matches or matches[0]["status"]!="completed":
        raise ValueError(f"track {track!r} has no completed policy in benchmark {name!r}")
    from .play import main as play_main
    child=[f"experiment_name={matches[0]['run_id']}"]
    if args.no_open:child.append("--no-open")
    if args.runs_root:child.extend(("--runs-root",args.runs_root))
    play_main(child)


def main(argv=None):
    raw=list(sys.argv[1:] if argv is None else argv)
    command=raw.pop(0) if raw and raw[0] in COMMANDS else "train"
    parser=argparse.ArgumentParser(description=__doc__,epilog=(
        "Example: osracer-benchmark train benchmark_name=ppo-2025 season=2025 "
        "checkpoint=runs/ppo-cem-pace-v2/checkpoints/policy.pt +simulator=mujoco"))
    parser.add_argument("overrides",nargs="*",metavar="KEY=VALUE")
    parser.add_argument("--runs-root",help=argparse.SUPPRESS)
    parser.add_argument("--no-resume",action="store_true")
    parser.add_argument("--fail-fast",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--no-open",action="store_true")
    args=parser.parse_args(raw)
    try:
        values=composed_values(args.overrides)
        unknown=sorted(set(values)-MANAGED_KEYS-PPO_KEYS-{"track"})
        if unknown:raise ValueError(f"unknown override(s): {', '.join(unknown)}")
        if command=="train":
            forbidden=sorted(set(values)&{"task","experiment_name","logger.tensorboard_dir","track"})
            if forbidden:raise ValueError(f"benchmark manages override(s): {', '.join(forbidden)}")
            forwarded=[token for token in args.overrides
                       if token.lstrip("+").split("=",1)[0] not in MANAGED_KEYS|{"simulator"}]
            return _train(args,values,forwarded)
        if command=="status":return _status(args,values)
        return _play(args,values)
    except (FileNotFoundError,KeyError,TypeError,ValueError) as exc:
        parser.error(str(exc))


if __name__=="__main__":main()
