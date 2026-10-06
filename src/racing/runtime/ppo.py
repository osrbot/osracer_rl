"""Sensor-only PPO training, PyTorch checkpointing, and ONNX export."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from ..control.neural_policy import (
    ActorCritic, NeuralPolicy, encode_observation, action_to_actuators,
    actuators_to_action,
    export_onnx, require_torch, save_checkpoint,
    load_checkpoint,
)
from ..control.policy import RW, RacingPolicy
from ..paths import source_files
from ..perception.sensors import LidarSensor
from ..simulators import available_simulators, create_simulator
from ..tracks import Track
from ..vehicle.real import make_sensor
from .artifacts import RunLayout
from .configuration import (apply_ppo_overrides, load_training_profile,
                            selected_training_profile, task_track)
from .progress import PPOProgress
from .run import DT, save
from .tracking import MetricLogger


torch = require_torch()


class RacingPPOEnv:
    """One native two-car episode with a sensor-only actor and privileged reward."""

    def __init__(self, env, track, seconds=120., opponent_speed=2.8,
                 opponent_gap=3., observation_profile="sim", max_speed=10.,
                 time_cost=.01, pace_weight=0., failure_horizon_scale=0.):
        self.env, self.track = env, track
        self.max_steps = round(float(seconds) / DT)
        self.opponent_speed = float(opponent_speed)
        self.opponent_gap = float(opponent_gap)
        self.observation_profile = observation_profile
        self.max_speed = float(max_speed)
        self.time_cost = float(time_cost)
        self.pace_weight = float(pace_weight)
        self.failure_horizon_scale = float(failure_horizon_scale)
        self.episode_index = 0

    def _location(self, state):
        if "z" in state and getattr(self.track,"has_elevation",False):
            return np.array([state["x"], state["y"], state["z"]])
        return np.array([state["x"], state["y"]])

    def reset(self, seed=None, start_s=0., lateral=0.):
        seed = self.episode_index if seed is None else int(seed)
        self.episode_index += 1
        start_s=float(start_s)%self.track.length
        lateral=float(lateral)
        self.states = self.env.reset(seed, starts=[
            (start_s, lateral),
            ((start_s+self.opponent_gap)%self.track.length, 0.),
        ])
        self.sensors = [make_sensor(LidarSensor(self.track, seed=seed+i), self.observation_profile)
                        for i in range(2)]
        self.opponent = RacingPolicy();self.opponent.p[0] = self.opponent_speed
        self.teacher = RacingPolicy()
        self.step_count = 0
        self.progress = 0.;self.opponent_progress = self.opponent_gap
        self.last_s = self.track.project(self._location(self.states[0]))[0]
        self.last_opponent_s = self.track.project(self._location(self.states[1]))[0]
        self.last_action = np.array([-1., 0.], np.float32)
        self.last_motion_step = 0
        self.speed_integral = 0.
        self.behind = True;self.ahead_steps = 0;self.overtakes = 0
        return self._observe(0)

    def teacher_action(self):
        # DAgger may execute the student's command. Synchronize the teacher's
        # rate limiter to the measured vehicle state before asking it for a
        # corrective label, rather than pretending its previous command ran.
        observation=self.observations[0]
        wheel=np.asarray(observation["wheel_vel"],float)
        steering=np.asarray(observation["steer_pos"],float)
        self.teacher.last_speed=float(np.mean(wheel[:2]*np.cos(steering))*RW)
        self.teacher.last_steer=float(np.mean(steering))
        return actuators_to_action(
            self.teacher.action(observation), self.max_speed)

    def _observe(self, tick):
        observations=[]
        for index, sensor in enumerate(self.sensors):
            observations.append(sensor.observe(
                self.states[index], self.states[index]["lidar_pose"],
                opponents=[self.states[1-index]], tick=tick))
        self.observations = observations
        return encode_observation(observations[0],self.last_action)

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1., 1.)
        primary = action_to_actuators(action, self.max_speed)
        opponent = self.opponent.action(self.observations[1])
        for sensor, command in zip(self.sensors, [primary, opponent]):
            if hasattr(sensor, "record_action"):
                sensor.record_action(command)
        self.states = self.env.step([primary, opponent]);self.step_count += 1
        state = self.states[0]
        speed = float(np.hypot(state["vx"], state["vy"]))
        s, cte, _ = self.track.project(self._location(state),s_hint=self.last_s)
        delta = (s-self.last_s+self.track.length/2) % self.track.length-self.track.length/2
        if abs(delta) > max(.6, 2*speed*DT):delta = 0.
        self.last_s = s;self.progress += delta
        other_s = self.track.project(
            self._location(self.states[1]),s_hint=self.last_opponent_s)[0]
        other_delta = ((other_s-self.last_opponent_s+self.track.length/2)
                       % self.track.length-self.track.length/2)
        self.last_opponent_s = other_s;self.opponent_progress += other_delta
        self.speed_integral += speed*DT
        collision = bool(state.get("collision", False))
        offroad = abs(cte) > self.track.width/2-.125
        unstable = abs(state.get("roll", 0.)) > .5 or abs(state.get("pitch", 0.)) > .5 or state.get("z", .05) < .015
        if speed > .25:self.last_motion_step = self.step_count
        stalled = (self.step_count-self.last_motion_step)*DT > 4.
        completed = self.progress >= self.track.length
        timeout = self.step_count >= self.max_steps
        relative = self.progress-self.opponent_progress
        overtake_reward = 0.
        if relative < -.5:self.behind=True;self.ahead_steps=0
        elif relative > .5 and self.behind and not (collision or offroad):
            self.ahead_steps += 1
            if self.ahead_steps*DT >= .3:
                self.overtakes += 1;self.behind=False;self.ahead_steps=0;overtake_reward=2.
        elif collision or offroad:self.ahead_steps=0
        safe_completion = completed and not (collision or offroad or unstable)
        failed = collision or offroad or unstable or stalled
        normalized_cte = cte/max(self.track.width/2, .01)
        centered = max(0., 1.-normalized_cte**2)
        rewards = {
            "progress": float(delta),
            # Progress integrates to nearly the same return at any lap time;
            # a small per-step cost makes faster safe completion preferable.
            "alive": -self.time_cost,
            # Integral(v ds) distinguishes two valid laps with the same
            # distance: covering each metre faster earns more while edge
            # running is suppressed by the privileged centre-line gate.
            "pace": self.pace_weight*max(0.,float(delta))*min(speed/self.max_speed,1.)*centered,
            "track": -.02*float(normalized_cte**2),
            "action_rate": -.002*float(np.square(action-self.last_action).sum()),
            "overtake": overtake_reward,
            "completion": 200.*float(safe_completion),
            "collision": -100.*float(collision or unstable),
            "offroad": -100.*float(offroad),
            "stalled": -50.*float(stalled),
            # A failed episode pays the rest of its time budget immediately,
            # so crashing early cannot avoid the per-step pace cost.
            "failure_horizon": (-self.time_cost*self.failure_horizon_scale
                                *max(0,self.max_steps-self.step_count)*float(failed)),
        }
        self.last_action = action.copy()
        reason = ("collision" if collision or unstable else "offroad" if offroad else
                  "completed" if safe_completion else "stalled" if stalled else
                  "timeout" if timeout else None)
        done = reason is not None
        observation = (self._observe(self.step_count) if not done else
                       np.zeros_like(encode_observation(self.observations[0],self.last_action)))
        return observation, float(sum(rewards.values())), done, {
            "rewards": rewards, "termination": reason, "progress_m": self.progress,
            "episode_length": self.step_count, "completed": completed,
            "valid_lap": safe_completion,
            "overtakes": self.overtakes,
            "speed_m_s": speed,
            "mean_speed_m_s": self.speed_integral/max(self.step_count*DT,DT),
            "mean_progress_speed_m_s": self.progress/max(self.step_count*DT,DT),
            "lap_time_s": self.step_count*DT if safe_completion else None,
        }


def _device(value):
    if value == "auto":return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return value


def _checkpoint_metadata(args, iteration, total_steps, metrics):
    sources={name:hashlib.sha256(path.read_bytes()).hexdigest()
             for name,path in source_files().items()}
    return {
        "experiment_name": args.layout.run_id,
        "simulator": args.engine,
        "task": f"racing/{args.track}",
        "iteration": int(iteration),
        "total_steps": int(total_steps),
        "seed": int(args.seed),
        "max_speed_m_s": 10.,
        "reward": {"time_cost":float(args.reward_time_cost),
                   "pace_weight":float(args.reward_pace_weight),
                   "failure_horizon_scale":float(args.reward_failure_horizon_scale)},
        "control_hz": 60,
        "scan_hz": 15,
        "metrics": metrics,
        "source_sha256": sources,
    }


def pretrain_actor(task, model, optimizer, args, device):
    """Warm-start with DAgger using the proven sensor-only structured policy."""
    if not args.imitation_steps or not args.imitation_epochs:
        return {"loss": None, "steps": 0, "epochs": 0}
    rounds=min(4,args.imitation_steps)
    first=max(2048,min(4096,args.imitation_steps//2))
    # Bahrain needs roughly 3,200 control ticks per lap. Each later DAgger
    # rollout must therefore be long enough to reach the current policy's
    # failure point instead of repeatedly collecting only the opening sector.
    tail=max(4096,(args.imitation_steps-first)//max(1,rounds-1))
    counts=[first]+[tail]*(rounds-1) if rounds>1 else [args.imitation_steps]
    observations=[];targets=[];losses=[];round_evaluations=[]
    perturbation_rng=np.random.default_rng(args.seed+104729)
    evaluation_seeds=tuple(range(args.evaluation_episodes))
    initial_evaluation=evaluate_actor_suite(task,model,device,evaluation_seeds)
    best_rank=evaluation_rank(initial_evaluation)
    best_state={name:value.detach().cpu().clone()
                for name,value in model.state_dict().items()}
    for round_index,count in enumerate(counts):
        learning_rate=min(args.learning_rate,1e-5) if (args.checkpoint or round_index) else args.learning_rate
        for group in optimizer.param_groups:group["lr"]=learning_rate
        # A full Bahrain lap is about 3,700 control steps. Collect that much
        # closed-loop experience for every checkpoint-selection seed so a
        # good seed cannot hide missing recovery data from a failed seed.
        for collection_seed in evaluation_seeds:
            episode_offset=0
            rollout_seed=collection_seed+round_index*1000
            observation=task.reset(rollout_seed)
            for _ in range(count):
                target=task.teacher_action()
                observations.append(observation);targets.append(target)
                if round_index==rounds-1:
                    with torch.inference_mode():
                        executed=model.deterministic(
                            torch.from_numpy(observation).to(device).unsqueeze(0))[0].cpu().numpy()
                elif round_index:
                    # DART-style recovery data: the perturbation scale matches
                    # the observed one-step imitation error while the expert
                    # remains in control long enough to demonstrate correction.
                    noise=np.array([.015,.025],np.float32)*round_index
                    executed=np.clip(target+perturbation_rng.normal(0.,noise),-1.,1.)
                else:
                    executed=target
                observation,_,done,_=task.step(executed)
                if done:
                    episode_offset+=1
                    observation=task.reset(rollout_seed+episode_offset)
        observation_tensor=torch.from_numpy(np.asarray(observations,np.float32)).to(device)
        target_tensor=torch.from_numpy(np.asarray(targets,np.float32)).to(device)
        target_tensor=model.quantize_action(target_tensor)
        batch_size=min(512,len(observations));model.train()
        valid_lap_found=False
        for epoch_index in range(args.imitation_epochs):
            order=torch.randperm(len(observations),device=device)
            for start in range(0,len(observations),batch_size):
                indices=order[start:start+batch_size]
                error=model.deterministic(observation_tensor[indices])-target_tensor[indices]
                target=target_tensor[indices]
                sample_weight=1.+4.*target[:,1].abs()+2.*(target[:,0]<-.25).float()
                loss=(sample_weight*(error[:,0].square()+12.*error[:,1].square())).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_gradient_norm)
                optimizer.step();losses.append(float(loss.item()))
            if ((epoch_index+1)%5==0 or epoch_index+1==args.imitation_epochs):
                model.eval()
                probe=evaluate_actor_suite(task,model,device,evaluation_seeds)
                probe_rank=evaluation_rank(probe)
                if probe_rank>best_rank:
                    best_rank=probe_rank
                    best_state={name:value.detach().cpu().clone()
                                for name,value in model.state_dict().items()}
                if probe["valid_lap"]:
                    valid_lap_found=True
                    break
                model.train()
        model.eval()
        evaluation=evaluate_actor_suite(task,model,device,evaluation_seeds)
        round_evaluations.append(evaluation)
        rank=evaluation_rank(evaluation)
        if rank>best_rank:
            best_rank=rank
            best_state={name:value.detach().cpu().clone()
                        for name,value in model.state_dict().items()}
        else:
            # Keep the next DAgger rollout on the strongest policy seen so
            # far. A difficult recovery batch must not move later collection
            # back toward an already rejected controller.
            model.load_state_dict(best_state)
            model.to(device);model.eval()
            optimizer.state.clear()
        print(f"Imitation round {round_index+1}/{rounds}  data {len(observations)}  ·  "
              f"loss {losses[-1]:.6f}  ·  progress {evaluation['progress_m']:.2f}m  ·  "
              f"{evaluation['termination']}",flush=True)
        if evaluation["valid_lap"] or valid_lap_found:
            break
    model.load_state_dict(best_state);model.to(device);model.eval()
    final_evaluation=evaluate_actor_suite(task,model,device,evaluation_seeds)
    result={"loss":float(losses[-1]),"steps":len(observations),
            "epochs":args.imitation_epochs,"rounds":len(round_evaluations),
            "evaluation":final_evaluation}
    return result


def transfer_actor(task,model,args,device):
    """Adapt a proven checkpoint with evenly distributed expert sectors.

    A checkpoint trained on one circuit often reaches the same unseen corner
    before failing, so start-only DAgger repeatedly labels the opening sector.
    This transfer pass covers the whole target circuit, anchors the update to
    the parent actor, and retains an epoch only after closed-loop evaluation.
    """
    seeds=tuple(range(args.evaluation_episodes))
    baseline=evaluate_actor_suite(task,model,device,seeds)
    if baseline["valid_lap"] or not args.imitation_steps or not args.imitation_epochs:
        return {"mode":"sector_transfer","improved":False,"steps":0,"epochs":0,
                "baseline":baseline,"evaluation":baseline,"history":[]}
    best_rank=evaluation_rank(baseline);best_evaluation=baseline
    best_state={name:value.detach().cpu().clone()
                for name,value in model.state_dict().items()}
    observations=[];targets=[]
    segment_count=min(12,max(4,args.imitation_steps//512))
    steps_per_segment=max(1,int(np.ceil(args.imitation_steps/segment_count)))
    for segment in range(segment_count):
        observation=task.reset(
            args.seed+10000+segment,
            start_s=task.track.length*segment/segment_count)
        for _ in range(steps_per_segment):
            target=task.teacher_action()
            observations.append(observation);targets.append(target)
            observation,_,done,_=task.step(target)
            if done:break
    observation_tensor=torch.from_numpy(
        np.asarray(observations,np.float32)).to(device)
    target_tensor=model.quantize_action(torch.from_numpy(
        np.asarray(targets,np.float32)).to(device))
    model.eval()
    anchors=[]
    with torch.inference_mode():
        for start in range(0,len(observation_tensor),1024):
            anchors.append(model.deterministic(observation_tensor[start:start+1024]))
    anchor_tensor=torch.cat(anchors)
    gradient_flags={name:parameter.requires_grad
                    for name,parameter in model.named_parameters()}
    for parameter in model.critic.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=min(args.learning_rate,5e-6))
    generator=torch.Generator(device=device).manual_seed(args.seed+15485863)
    history=[]
    for epoch in range(args.imitation_epochs):
        model.train();losses=[]
        order=torch.randperm(len(observation_tensor),generator=generator,device=device)
        for start in range(0,len(order),512):
            indices=order[start:start+512]
            action=model.deterministic(observation_tensor[indices])
            teacher_error=(action-target_tensor[indices]).square()
            anchor_error=(action-anchor_tensor[indices]).square()
            loss=(teacher_error[:,0]+12.*teacher_error[:,1]
                  +4.*(anchor_error[:,0]+12.*anchor_error[:,1])).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_gradient_norm)
            optimizer.step();losses.append(float(loss.item()))
        model.eval();primary=evaluate_actor(task,model,device,seeds[0])
        if primary["valid_lap"]:
            evaluation=_evaluation_summary([primary]+[
                evaluate_actor(task,model,device,seed) for seed in seeds[1:]])
        else:evaluation=_evaluation_summary([primary])
        row={"epoch":epoch+1,"loss":float(np.mean(losses)),
             "evaluation":evaluation};history.append(row)
        print(f"Transfer epoch {epoch+1}/{args.imitation_epochs}  ·  "
              f"loss {row['loss']:.6f}  ·  valid "
              f"{evaluation['valid_laps']}/{len(seeds)}  ·  "
              f"progress {evaluation['progress_m']:.2f}m",flush=True)
        if len(evaluation["episodes"])==len(seeds):
            rank=evaluation_rank(evaluation)
            if rank>best_rank:
                best_rank=rank;best_evaluation=evaluation
                best_state={name:value.detach().cpu().clone()
                            for name,value in model.state_dict().items()}
            if evaluation["valid_lap"]:break
    model.load_state_dict(best_state);model.to(device);model.eval()
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(gradient_flags[name])
    return {"mode":"sector_transfer",
            "improved":evaluation_rank(best_evaluation)>evaluation_rank(baseline),
            "steps":len(observations),"epochs":len(history),
            "baseline":baseline,"evaluation":best_evaluation,"history":history}


def evaluate_actor(task, model, device, seed):
    observation=task.reset(seed);total_reward=0.;info={}
    while True:
        with torch.inference_mode():
            action=model.deterministic(
                torch.from_numpy(observation).to(device).unsqueeze(0))[0].cpu().numpy()
        observation,reward,done,info=task.step(action);total_reward+=reward
        if done:
            return {"reward":total_reward,"progress_m":info["progress_m"],
                    "termination":info["termination"],"valid_lap":info["valid_lap"],
                    "episode_length":info["episode_length"],
                    "overtakes":info["overtakes"],
                    "mean_speed_m_s":info["mean_speed_m_s"],
                    "mean_progress_speed_m_s":info["mean_progress_speed_m_s"],
                    "lap_time_s":info["lap_time_s"]}


def _evaluation_summary(episodes):
    worst=min(episodes,key=lambda row:(int(row["valid_lap"]),row["progress_m"]))
    return {
        "reward":float(np.mean([row["reward"] for row in episodes])),
        "progress_m":float(worst["progress_m"]),
        "termination":worst["termination"],
        "valid_lap":all(row["valid_lap"] for row in episodes),
        "valid_laps":sum(bool(row["valid_lap"]) for row in episodes),
        "evaluation_episodes":len(episodes),
        "episode_length":float(np.mean([row["episode_length"] for row in episodes])),
        "overtakes":float(np.mean([row["overtakes"] for row in episodes])),
        "mean_speed_m_s":float(np.mean([row.get("mean_speed_m_s",0.) for row in episodes])),
        "mean_progress_speed_m_s":float(np.mean([
            row.get("mean_progress_speed_m_s",row["progress_m"]/(row["episode_length"]*DT))
            for row in episodes])),
        "mean_lap_time_s": (float(np.mean([row["lap_time_s"] for row in episodes]))
                            if all(row.get("lap_time_s") is not None for row in episodes)
                            else None),
        "collision_episode_rate":float(np.mean([
            row["termination"]=="collision" for row in episodes])),
        "offroad_episode_rate":float(np.mean([
            row["termination"]=="offroad" for row in episodes])),
        "episodes":episodes,
    }


def evaluate_actor_suite(task,model,device,seeds):
    return _evaluation_summary([
        evaluate_actor(task,model,device,int(seed)) for seed in seeds])


def evaluation_rank(evaluation):
    valid_laps=int(evaluation.get("valid_laps",evaluation["valid_lap"]))
    episode_count=int(evaluation.get("evaluation_episodes",1))
    if valid_laps==episode_count:
        # Once every seed is safe, select explicitly for whole-episode speed.
        # Reward remains a tie-breaker for centre-line and action smoothness.
        return (valid_laps,float(evaluation.get("mean_progress_speed_m_s",0.)),
                float(evaluation.get("mean_speed_m_s",0.)),
                float(evaluation["reward"]),float(evaluation["progress_m"]))
    return (valid_laps,float(evaluation["progress_m"]),float(evaluation["reward"]))


def _interpolate_state(base, candidate, scale):
    """Interpolate floating checkpoint tensors for a quantized trust region."""
    scale=float(scale)
    return {name:(value+scale*(candidate[name]-value)
                  if value.is_floating_point() else candidate[name].clone())
            for name,value in base.items()}


def _action_change_fraction(model,observations,anchors):
    changed=0
    with torch.inference_mode():
        for start in range(0,len(observations),1024):
            action=model.deterministic(observations[start:start+1024])
            changed += int((action!=anchors[start:start+1024]).any(dim=1).sum().item())
    return changed/max(1,len(observations))


def refine_actor(task,model,args,device,tracker=None):
    """Safely distil the expert on successful on-policy trajectories.

    Quantized policies change discontinuously when one output crosses a bin.
    Every epoch is therefore accepted by closed-loop multi-seed evaluation,
    while an anchor loss keeps the update near the last proven controller.
    """
    seeds=tuple(range(args.evaluation_episodes))
    if not args.refinement_cycles or not args.refinement_epochs:
        return {"enabled":False,"improved":False,"cycles":0,"epochs":0,
                "baseline":None,"evaluation":None,"history":[]}
    baseline=evaluate_actor_suite(task,model,device,seeds)
    best_rank=evaluation_rank(baseline);best_evaluation=baseline
    best_state={name:value.detach().cpu().clone()
                for name,value in model.state_dict().items()}
    gradient_flags={name:parameter.requires_grad
                    for name,parameter in model.named_parameters()}
    for parameter in model.critic.parameters():parameter.requires_grad_(False)
    history=[]
    for cycle in range(args.refinement_cycles):
        model.load_state_dict(best_state);model.to(device);model.eval()
        observations=[];targets=[];anchors=[]
        for seed in seeds:
            observation=task.reset(seed)
            while True:
                target=task.teacher_action()
                with torch.inference_mode():
                    action=model.deterministic(
                        torch.from_numpy(observation).to(device).unsqueeze(0))[0].cpu().numpy()
                observations.append(observation);targets.append(target);anchors.append(action)
                observation,_,done,_=task.step(action)
                if done:break
        observation_tensor=torch.from_numpy(
            np.asarray(observations,np.float32)).to(device)
        target_tensor=model.quantize_action(torch.from_numpy(
            np.asarray(targets,np.float32)).to(device))
        anchor_tensor=torch.from_numpy(np.asarray(anchors,np.float32)).to(device)
        learning_rate=min(2e-6,args.learning_rate)/(cycle+1)
        anchor_weight=4.*(cycle+1)
        optimizer=torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate)
        generator=torch.Generator(device=device).manual_seed(args.seed+857+cycle*1000)
        for epoch in range(args.refinement_epochs):
            model.train();losses=[]
            order=torch.randperm(len(observation_tensor),generator=generator,device=device)
            for start in range(0,len(order),512):
                indices=order[start:start+512]
                action=model.deterministic(observation_tensor[indices])
                teacher_error=(action-target_tensor[indices]).square()
                anchor_error=(action-anchor_tensor[indices]).square()
                loss=(teacher_error[:,0]+12.*teacher_error[:,1]
                      +anchor_weight*(anchor_error[:,0]+12.*anchor_error[:,1])).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_gradient_norm)
                optimizer.step();losses.append(float(loss.item()))
            candidate_state={name:value.detach().cpu().clone()
                             for name,value in model.state_dict().items()}
            model.eval()
            primary=evaluate_actor(task,model,device,seeds[0])
            if primary["valid_lap"]:
                episodes=[primary]+[
                    evaluate_actor(task,model,device,seed) for seed in seeds[1:]]
                evaluation=_evaluation_summary(episodes)
            else:
                evaluation=_evaluation_summary([primary])
            selected_scale=1.;change_fraction=_action_change_fraction(
                model,observation_tensor,anchor_tensor)
            rank=(evaluation_rank(evaluation)
                  if len(evaluation["episodes"])==len(seeds) else None)
            accepted=rank is not None and rank>best_rank
            # A full optimizer epoch can cross many discrete action bins at
            # once. If it is rejected, interpolate from the last safe policy
            # and evaluate one candidate whose changed-action footprint is at
            # most one percent of the accepted trajectories.
            if not accepted:
                for scale in (.5,.25,.125,.0625):
                    model.load_state_dict(_interpolate_state(best_state,candidate_state,scale))
                    model.to(device);model.eval()
                    fraction=_action_change_fraction(model,observation_tensor,anchor_tensor)
                    if 0.<fraction<=.01:
                        selected_scale=scale;change_fraction=fraction
                        primary=evaluate_actor(task,model,device,seeds[0])
                        if primary["valid_lap"]:
                            evaluation=_evaluation_summary([primary]+[
                                evaluate_actor(task,model,device,seed) for seed in seeds[1:]])
                        else:evaluation=_evaluation_summary([primary])
                        rank=(evaluation_rank(evaluation)
                              if len(evaluation["episodes"])==len(seeds) else None)
                        accepted=rank is not None and rank>best_rank
                        break
            row={"cycle":cycle+1,"epoch":epoch+1,
                 "loss":float(np.mean(losses)),"step_scale":selected_scale,
                 "action_change_fraction":change_fraction,"accepted":accepted,
                 "evaluation":evaluation}
            history.append(row)
            if tracker is not None:
                step=args.iterations+cycle*args.refinement_epochs+epoch
                tracker.log(step,"refinement",{
                    "cycle":cycle+1,"epoch":epoch+1,"loss":row["loss"],
                    "valid_laps":evaluation["valid_laps"],
                    "evaluation_episodes":evaluation["evaluation_episodes"],
                    "reward":evaluation["reward"],
                    "episode_length":evaluation["episode_length"],
                    "overtakes":evaluation["overtakes"]})
            print(f"Refinement cycle {cycle+1}/{args.refinement_cycles}  ·  "
                  f"epoch {epoch+1}/{args.refinement_epochs}  ·  loss {row['loss']:.6f}  ·  "
                  f"valid {evaluation['valid_laps']}/{len(seeds)}  ·  "
                  f"scale {selected_scale:.4f}  ·  accepted {'yes' if accepted else 'no'}  ·  "
                  f"steps {evaluation['episode_length']:.0f}",flush=True)
            if accepted:
                best_rank=rank;best_evaluation=evaluation
                best_state={name:value.detach().cpu().clone()
                            for name,value in model.state_dict().items()}
            else:
                model.load_state_dict(best_state);model.to(device);model.eval()
                optimizer.state.clear()
        model.load_state_dict(best_state);model.to(device);model.eval()
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(gradient_flags[name])
    return {"enabled":True,"improved":evaluation_rank(best_evaluation)>evaluation_rank(baseline),
            "cycles":args.refinement_cycles,"epochs":args.refinement_epochs,
            "baseline":baseline,"evaluation":best_evaluation,"history":history}


def train(args):
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device(_device(args.device))
    # Construct the optimizer before Isaac starts. Isaac's legacy ML extension
    # prepends an older torch package fragment which otherwise shadows the
    # matching torch._dynamo used by optimizer initialization.
    model=(load_checkpoint(args.checkpoint,"cpu")[0]
           if args.checkpoint else ActorCritic())
    if args.checkpoint:
        with torch.no_grad():model.log_std.fill_(-4.)
    imitation_optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    track=Track(args.track)
    native=create_simulator(args.engine,track)
    task=RacingPPOEnv(native,track,args.seconds,args.opponent_speed,args.opponent_gap,
                      time_cost=args.reward_time_cost,
                      pace_weight=args.reward_pace_weight,
                      failure_horizon_scale=args.reward_failure_horizon_scale)
    model.to(device)
    checkpoint=args.layout.checkpoints/'policy.pt'
    imitation=(transfer_actor(task,model,args,device) if args.checkpoint else
               pretrain_actor(task,model,imitation_optimizer,args,device))
    warmstart=evaluate_actor_suite(
        task,model,device,range(args.evaluation_episodes))
    print(f"Warm-start check  progress {warmstart['progress_m']:.2f}m  ·  "
          f"mean speed {warmstart['mean_speed_m_s']:.2f}m/s  ·  "
          f"termination {warmstart['termination']}  ·  "
          f"valid lap {'yes' if warmstart['valid_lap'] else 'no'}",flush=True)
    save_checkpoint(checkpoint,model,_checkpoint_metadata(
        args,0,0,{"imitation":imitation,"evaluation":warmstart}))
    best_rank=evaluation_rank(warmstart)
    history=[];total_steps=0;training_episode=0
    observation=task.reset(training_episode%args.evaluation_episodes)
    episode_return=0.;episode_length=0
    progress=PPOProgress(args.layout.run_id,args.engine,args.track,args.iterations)
    with MetricLogger(args.layout.metrics,args.tensorboard_dir,not args.no_tensorboard) as tracker:
        args.tensorboard_active=tracker.writer is not None
        args.tensorboard_path=tracker.tensorboard_dir
        try:
            for iteration in range(args.iterations):
                iteration_started=time.monotonic();collection_started=time.monotonic()
                observations=[];actions=[];teacher_actions=[];rewards=[];dones=[];log_probs=[];values=[]
                completed_returns=[];completed_lengths=[];terminations=defaultdict(int)
                reward_sums=defaultdict(float);speed_sum=0.
                for _ in range(args.steps_per_iteration):
                    teacher_action=task.teacher_action()
                    tensor=torch.from_numpy(observation).to(device).unsqueeze(0)
                    with torch.no_grad():action,log_prob,value=model.sample(tensor)
                    action_np=action[0].cpu().numpy()
                    next_observation,reward,done,info=task.step(action_np)
                    observations.append(observation);actions.append(action_np)
                    teacher_actions.append(teacher_action);rewards.append(reward)
                    dones.append(done);log_probs.append(float(log_prob.item()));values.append(float(value.item()))
                    episode_return+=reward;episode_length+=1;total_steps+=1
                    for key,val in info["rewards"].items():reward_sums[key]+=float(val)
                    speed_sum += float(info["speed_m_s"])
                    if done:
                        completed_returns.append(episode_return);completed_lengths.append(episode_length)
                        terminations[info["termination"]]+=1
                        training_episode+=1
                        observation=task.reset(training_episode%args.evaluation_episodes)
                        episode_return=0.;episode_length=0
                    else:observation=next_observation
                collection_seconds=time.monotonic()-collection_started
                with torch.no_grad():
                    last_value=float(model(torch.from_numpy(observation).to(device).unsqueeze(0))[1].item())
                advantages=np.zeros(args.steps_per_iteration,np.float32);last_gae=0.
                for step in reversed(range(args.steps_per_iteration)):
                    next_value=last_value if step==args.steps_per_iteration-1 else values[step+1]
                    mask=1.-float(dones[step])
                    delta=rewards[step]+args.gamma*next_value*mask-values[step]
                    last_gae=delta+args.gamma*args.gae_lambda*mask*last_gae
                    advantages[step]=last_gae
                returns=advantages+np.asarray(values,np.float32)
                obs_tensor=torch.from_numpy(np.asarray(observations,np.float32)).to(device)
                action_tensor=torch.from_numpy(np.asarray(actions,np.float32)).to(device)
                teacher_tensor=torch.from_numpy(np.asarray(teacher_actions,np.float32)).to(device)
                teacher_tensor=model.quantize_action(teacher_tensor)
                old_log_tensor=torch.tensor(log_probs,dtype=torch.float32,device=device)
                return_tensor=torch.from_numpy(returns).to(device)
                advantage_tensor=torch.from_numpy(advantages).to(device)
                advantage_tensor=(advantage_tensor-advantage_tensor.mean())/(advantage_tensor.std()+1e-8)
                learning_started=time.monotonic();losses=defaultdict(list)
                batch_size=args.steps_per_iteration
                mini_size=max(1,batch_size//args.mini_batches)
                for _ in range(args.learning_epochs):
                    permutation=torch.randperm(batch_size,device=device)
                    for start in range(0,batch_size,mini_size):
                        indices=permutation[start:start+mini_size]
                        new_log,entropy,predicted=model.evaluate(obs_tensor[indices],action_tensor[indices])
                        # Quantized actions can move between adjacent bins
                        # while the exploration std is small. Bound the
                        # likelihood ratio before exponentiation so one bin
                        # transition cannot inject inf/NaN into the actor.
                        log_ratio=(new_log-old_log_tensor[indices]).clamp(-2.,2.)
                        ratio=log_ratio.exp()
                        unclipped=ratio*advantage_tensor[indices]
                        clipped=ratio.clamp(1-args.clip_ratio,1+args.clip_ratio)*advantage_tensor[indices]
                        policy_loss=-torch.minimum(unclipped,clipped).mean()
                        value_loss=.5*(predicted-return_tensor[indices]).square().mean()
                        entropy_mean=entropy.mean()
                        imitation_error=(model.deterministic(obs_tensor[indices])
                                         -teacher_tensor[indices])
                        imitation_target=teacher_tensor[indices]
                        imitation_sample_weight=(1.+4.*imitation_target[:,1].abs()
                                                 +2.*(imitation_target[:,0]<-.25).float())
                        imitation_loss=(imitation_sample_weight*(
                            imitation_error[:,0].square()
                            +12.*imitation_error[:,1].square())).mean()
                        imitation_weight=max(.2,1.-iteration/max(1,args.iterations))
                        loss=(policy_loss+args.value_coefficient*value_loss
                              -args.entropy_coefficient*entropy_mean
                              +imitation_weight*imitation_loss)
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_gradient_norm)
                        optimizer.step()
                        losses["surrogate"].append(float(policy_loss.item()))
                        losses["value"].append(float(value_loss.item()))
                        losses["entropy"].append(float(entropy_mean.item()))
                        losses["imitation"].append(float(imitation_loss.item()))
                learning_seconds=time.monotonic()-learning_started
                iteration_seconds=time.monotonic()-iteration_started
                episode_count=len(completed_returns)
                denom=float(args.steps_per_iteration)
                metrics={
                    "steps_per_second":args.steps_per_iteration/max(iteration_seconds,1e-9),
                    "collection_seconds":collection_seconds,"learning_seconds":learning_seconds,
                    "iteration_seconds":iteration_seconds,
                    "value_loss":float(np.mean(losses["value"])),
                    "surrogate_loss":float(np.mean(losses["surrogate"])),
                    "entropy":float(np.mean(losses["entropy"])),
                    "imitation_loss":float(np.mean(losses["imitation"])),
                    "action_noise_std":float(model.log_std.exp().mean().detach().cpu()),
                    "mean_reward":float(np.mean(completed_returns)) if episode_count else episode_return,
                    "mean_episode_length":float(np.mean(completed_lengths)) if episode_count else episode_length,
                    "mean_speed_m_s":speed_sum/denom,
                    "mean_progress_speed_m_s":reward_sums["progress"]/denom/DT,
                    "reward_progress":reward_sums["progress"]/denom,
                    "reward_alive":reward_sums["alive"]/denom,
                    "reward_pace":reward_sums["pace"]/denom,
                    "reward_track":reward_sums["track"]/denom,
                    "reward_action_rate":reward_sums["action_rate"]/denom,
                    "reward_overtake":reward_sums["overtake"]/denom,
                    "reward_completion":reward_sums["completion"]/denom,
                    "reward_collision":reward_sums["collision"]/denom,
                    "reward_offroad":reward_sums["offroad"]/denom,
                    "reward_stalled":reward_sums["stalled"]/denom,
                    "reward_failure_horizon":reward_sums["failure_horizon"]/denom,
                    "termination_timeout":100*terminations["timeout"]/max(1,episode_count),
                    "termination_collision":100*terminations["collision"]/max(1,episode_count),
                    "termination_offroad":100*terminations["offroad"]/max(1,episode_count),
                    "termination_completed":100*terminations["completed"]/max(1,episode_count),
                    "termination_stalled":100*terminations["stalled"]/max(1,episode_count),
                    "episodes":episode_count,"total_steps":total_steps,
                }
                tracker.log(iteration,"ppo",{
                    "loss":{"value_function":metrics["value_loss"],"surrogate":metrics["surrogate_loss"],
                            "entropy":metrics["entropy"],"imitation":metrics["imitation_loss"]},
                    "policy":{"action_noise_std":metrics["action_noise_std"]},
                    "episode":{"mean_reward":metrics["mean_reward"],
                               "mean_length":metrics["mean_episode_length"],
                               "mean_speed_m_s":metrics["mean_speed_m_s"],
                               "mean_progress_speed_m_s":metrics["mean_progress_speed_m_s"]},
                    "reward":{"progress":metrics["reward_progress"],"alive":metrics["reward_alive"],
                              "pace":metrics["reward_pace"],
                              "track":metrics["reward_track"],"action_rate":metrics["reward_action_rate"],
                              "overtake":metrics["reward_overtake"],
                              "completion":metrics["reward_completion"],
                              "collision":metrics["reward_collision"],
                              "offroad":metrics["reward_offroad"],
                              "stalled":metrics["reward_stalled"],
                              "failure_horizon":metrics["reward_failure_horizon"]},
                    "termination":{"timeout":metrics["termination_timeout"],
                                   "collision":metrics["termination_collision"],
                                   "offroad":metrics["termination_offroad"],
                                   "completed":metrics["termination_completed"],
                                   "stalled":metrics["termination_stalled"]},
                    "performance":{"steps_per_second":metrics["steps_per_second"],
                                   "total_steps":total_steps},
                })
                history.append({"iteration":iteration+1,**metrics});save(args.layout.metrics/'training/history.json',history)
                progress.iteration(iteration,metrics)
                if (iteration+1)%args.checkpoint_interval==0 or iteration+1==args.iterations:
                    evaluation=evaluate_actor_suite(
                        task,model,device,range(args.evaluation_episodes))
                    rank=evaluation_rank(evaluation)
                    tracker.log(iteration,"evaluation",evaluation)
                    if rank>best_rank:
                        best_rank=rank
                        save_checkpoint(checkpoint,model,_checkpoint_metadata(
                            args,iteration+1,total_steps,metrics|{"evaluation":evaluation}))
                    training_episode+=1
                    observation=task.reset(training_episode%args.evaluation_episodes)
                    episode_return=0.;episode_length=0
            # PPO may end on a transient policy while the checkpoint stores
            # an earlier, stronger controller. Refine that accepted model and
            # keep it only when the same multi-seed rank improves.
            model,_=load_checkpoint(checkpoint,device)
            refinement=refine_actor(task,model,args,device,tracker)
            if refinement["improved"]:
                save_checkpoint(checkpoint,model,_checkpoint_metadata(
                    args,args.iterations,total_steps,{"refinement":refinement,
                    "evaluation":refinement["evaluation"]}))
        except BaseException:
            native.close()
            raise
    try:
        onnx_path=export_onnx(checkpoint,args.layout.checkpoints/'policy.onnx')
    except BaseException:
        native.close()
        raise
    return (checkpoint,onnx_path,args.tensorboard_path if args.tensorboard_active else None,
            native)


def _parser(profile):
    parser=argparse.ArgumentParser(description=__doc__,epilog=(
        "Example: osracer-train +simulator=mujoco +task=racing/bahrain experiment_name=ppo-demo"))
    parser.add_argument("overrides",nargs="*",metavar="KEY=VALUE")
    parser.add_argument("--simulator","--engine",dest="engine",choices=available_simulators())
    parser.add_argument("--task","--track",dest="track",type=task_track,default=profile.track)
    parser.add_argument("--experiment-name","--run-id",dest="run_id")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--no-tensorboard",action="store_true",default=not profile.tensorboard)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--runs-root",help=argparse.SUPPRESS)
    parser.add_argument("--iterations",type=int,default=profile.iterations,help=argparse.SUPPRESS)
    parser.add_argument("--steps-per-iteration",type=int,default=profile.steps_per_iteration,help=argparse.SUPPRESS)
    parser.add_argument("--learning-epochs",type=int,default=profile.learning_epochs,help=argparse.SUPPRESS)
    parser.add_argument("--mini-batches",type=int,default=profile.mini_batches,help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate",type=float,default=profile.learning_rate,help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-interval",type=int,default=profile.checkpoint_interval,help=argparse.SUPPRESS)
    parser.add_argument("--imitation-steps",type=int,default=profile.imitation_steps,help=argparse.SUPPRESS)
    parser.add_argument("--imitation-epochs",type=int,default=profile.imitation_epochs,help=argparse.SUPPRESS)
    parser.add_argument("--refinement-cycles",type=int,default=profile.refinement_cycles,help=argparse.SUPPRESS)
    parser.add_argument("--refinement-epochs",type=int,default=profile.refinement_epochs,help=argparse.SUPPRESS)
    parser.add_argument("--seconds",type=float,default=profile.episode_length_s,help=argparse.SUPPRESS)
    parser.add_argument("--evaluation-episodes",type=int,default=profile.evaluation_episodes,
                        help=argparse.SUPPRESS)
    parser.add_argument("--seed",type=int,default=profile.seed,help=argparse.SUPPRESS)
    parser.add_argument("--opponent-speed",type=float,default=profile.opponent_speed_m_s,help=argparse.SUPPRESS)
    parser.add_argument("--opponent-gap",type=float,default=profile.opponent_gap_m,help=argparse.SUPPRESS)
    parser.set_defaults(reward_time_cost=profile.reward_time_cost,
        reward_pace_weight=profile.reward_pace_weight,
        reward_failure_horizon_scale=profile.reward_failure_horizon_scale)
    parser.set_defaults(gamma=profile.gamma,gae_lambda=profile.gae_lambda,
        clip_ratio=profile.clip_ratio,entropy_coefficient=profile.entropy_coefficient,
        value_coefficient=profile.value_coefficient,max_gradient_norm=profile.max_gradient_norm)
    return parser


def main(argv=None):
    raw=list(sys.argv[1:] if argv is None else argv)
    tokens=[token for token in raw if "=" in token and not token.startswith("--")]
    try:profile=load_training_profile(selected_training_profile(tokens))
    except (KeyError,TypeError,ValueError) as exc:raise SystemExit(f"configuration error: {exc}") from exc
    parser=_parser(profile);args=parser.parse_args(raw)
    try:apply_ppo_overrides(args,args.overrides,available_simulators())
    except (TypeError,ValueError) as exc:parser.error(str(exc))
    if not args.engine:parser.error("select a simulator with +simulator=mujoco")
    positive=[args.iterations,args.steps_per_iteration,args.learning_epochs,args.mini_batches,
              args.learning_rate,args.checkpoint_interval,args.seconds,
              args.evaluation_episodes]
    if any(value<=0 for value in positive):parser.error("PPO training values must be positive")
    if (args.imitation_steps<0 or args.imitation_epochs<0
            or args.refinement_cycles<0 or args.refinement_epochs<0
            or args.reward_time_cost<0 or args.reward_pace_weight<0
            or args.reward_failure_horizon_scale<0):
        parser.error("imitation, refinement, and reward values cannot be negative")
    run_id=args.run_id or f"{datetime.now():%Y%m%d_%H%M%S}-{args.engine}-{args.track}-ppo"
    try:args.layout=RunLayout.open(run_id,args.runs_root)
    except ValueError as exc:parser.error(str(exc))
    if (args.layout.checkpoints/'policy.pt').exists() or (args.layout.metrics/'training/history.json').exists():
        parser.error(f"run {args.layout.run_id!r} already contains training; choose a new experiment_name")
    save(args.layout.config/f"training_{args.engine}_{args.track}.json",
         vars(args)|{"layout":str(args.layout.root),"algorithm":"ppo"})
    checkpoint,onnx_path,tensorboard,native=train(args)
    artifacts={"run_id":args.layout.run_id,"run_dir":str(args.layout.root),
        "checkpoint":str(checkpoint),"onnx":str(onnx_path),
        "metrics_jsonl":str(args.layout.metrics/'metrics.jsonl'),
        "tensorboard":str(tensorboard) if tensorboard else None}
    try:
        print("OSRACER_ARTIFACTS",json.dumps(artifacts,ensure_ascii=False),flush=True)
    finally:
        native.close()
    return artifacts


def export_main(argv=None):
    parser=argparse.ArgumentParser(description="Export an OSRACER PPO .pt checkpoint to ONNX")
    parser.add_argument("checkpoint",nargs="?")
    parser.add_argument("--experiment-name","--run-id",dest="run_id")
    parser.add_argument("--runs-root")
    parser.add_argument("--output")
    args=parser.parse_args(argv)
    if args.run_id:
        layout=RunLayout.existing(args.run_id,args.runs_root)
        checkpoint=layout.checkpoints/'policy.pt'
    elif args.checkpoint:checkpoint=Path(args.checkpoint)
    else:parser.error("provide a checkpoint path or --experiment-name")
    if not checkpoint.is_file():parser.error(f"checkpoint not found: {checkpoint}")
    output=export_onnx(checkpoint,args.output)
    print(f"ONNX_MODEL {output}",flush=True)


if __name__=="__main__":main()
