"""Tensor-only PPO actor and portable checkpoint/ONNX helpers."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import numpy as np

from .policy import CENTER_LIMIT, RW, TRACK, WB


LIDAR_FEATURES = 361
OBSERVATION_SIZE = 4 + 2 + LIDAR_FEATURES + 1 + 2
ACTION_SIZE = 2
ACTION_DELTA_LIMITS = (.07, .16)
ACTION_QUANTIZATION = 32.
POLICY_VERSION = "ppo-lidar-conv-quantized-action-v6"


def require_torch():
    """Import torch, reusing Isaac Sim's Python packages when available locally."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as original:
        isaac = Path(os.environ.get(
            "OSRACER_ISAAC_DIR",
            str(Path.home() / "rlgpu_ws/isaac-sim-standalone-6.0.1-linux-x86_64"),
        ))
        candidate = isaac / "kit/python/lib/python3.12/site-packages"
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            try:
                return importlib.import_module("torch")
            except ModuleNotFoundError:
                pass
        raise RuntimeError(
            "PPO needs PyTorch. Install pip install -e '.[ppo]' or set "
            "OSRACER_ISAAC_DIR to an Isaac Sim installation that provides torch."
        ) from original


torch = require_torch()
nn = torch.nn


def encode_observation(observation: dict, previous_action=None) -> np.ndarray:
    wheel = np.asarray(observation["wheel_vel"], np.float32).reshape(4)
    steer = np.asarray(observation["steer_pos"], np.float32).reshape(2)
    packet = observation["lidar"]
    ranges = np.asarray(packet["ranges"], np.float32).reshape(361)
    valid = np.asarray(packet.get("valid", packet.get("validmask")), bool).reshape(361)
    valid &= np.isfinite(ranges)
    lidar = np.where(valid, np.clip(ranges / 15., 0., 1.), -1.).astype(np.float32)
    age = float(packet["age"])
    age = 1.0 if not np.isfinite(age) else np.clip(age / .14, 0., 1.)
    encoded = np.concatenate((
        np.clip(wheel * RW / 12., -1., 1.),
        np.clip(steer / .45, -1., 1.),
        lidar,
        np.array([age], np.float32),
        np.asarray(previous_action if previous_action is not None else [-1., 0.],
                   np.float32).reshape(ACTION_SIZE),
    )).astype(np.float32, copy=False)
    if encoded.shape != (OBSERVATION_SIZE,) or not np.isfinite(encoded).all():
        raise ValueError("PPO observation must be one finite fixed-size sensor vector")
    return encoded


def action_to_actuators(action, max_speed_m_s: float = 10.):
    action = np.clip(np.asarray(action, float).reshape(ACTION_SIZE), -1., 1.)
    speed = float((action[0] + 1.) * .5 * max_speed_m_s)
    steer = float(action[1] * CENTER_LIMIT)
    curvature = np.tan(steer) / WB
    left, right = 1 - TRACK * curvature / 2, 1 + TRACK * curvature / 2
    steering = np.clip(np.arctan2(WB * curvature, [left, right]), -.45, .45)
    wheels = speed * np.array([
        np.hypot(left, WB * curvature), np.hypot(right, WB * curvature), left, right,
    ]) / RW * np.array([1., 1., 1., -1.])
    return wheels, steering


def actuators_to_action(command, max_speed_m_s: float = 10.) -> np.ndarray:
    """Project an Ackermann teacher command into the neural action space."""
    wheels, steering = command
    wheels = np.asarray(wheels, float).reshape(4)
    steering = np.asarray(steering, float).reshape(2)
    curvatures = []
    for angle, side in zip(steering, (-1., 1.)):
        tangent = np.tan(angle)
        denominator = WB - side * TRACK * tangent / 2
        if abs(denominator) > 1e-8:
            curvatures.append(tangent / denominator)
    curvature = float(np.mean(curvatures)) if curvatures else 0.
    central = float(np.arctan(WB * curvature))
    left, right = 1 - TRACK * curvature / 2, 1 + TRACK * curvature / 2
    speeds = [wheels[2] * RW / left, -wheels[3] * RW / right]
    speed = float(np.clip(np.mean(speeds), 0., max_speed_m_s))
    return np.array([
        2 * speed / max_speed_m_s - 1,
        np.clip(central / CENTER_LIMIT, -1., 1.),
    ], np.float32)


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int = OBSERVATION_SIZE,
                 action_size: int = ACTION_SIZE, hidden_sizes=(256, 128),
                 residual_actions: bool = False,
                 quantized_actions: bool = True, pace_guard: bool = False,
                 corner_speed_threshold_m_s: float = 3.5,
                 corner_steer_threshold: float = .5,
                 corner_slowdown_bins: float = 1.):
        super().__init__()
        first, second = map(int, hidden_sizes)
        self.observation_size = int(observation_size)
        self.residual_actions = bool(residual_actions)
        self.quantized_actions = bool(quantized_actions)
        self.pace_guard = bool(pace_guard)
        self.corner_speed_threshold_m_s = float(corner_speed_threshold_m_s)
        self.corner_steer_threshold = float(corner_steer_threshold)
        self.corner_slowdown_bins = float(corner_slowdown_bins)
        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=1, padding=3), nn.ELU(),
            nn.Conv1d(8, 8, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.Flatten(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(8 * 181 + observation_size - LIDAR_FEATURES, first), nn.ELU(),
            nn.Linear(first, second), nn.ELU(),
        )
        self.actor = nn.Linear(second, action_size)
        self.critic = nn.Linear(second, 1)
        self.log_std = nn.Parameter(torch.full((action_size,), -4.0))

    def forward(self, observation):
        lidar = self.lidar_encoder(observation[:, 6:6 + LIDAR_FEATURES].unsqueeze(1))
        proprioception = torch.cat((observation[:, :6],
                                    observation[:, 6 + LIDAR_FEATURES:]),dim=-1)
        features = self.encoder(torch.cat((proprioception, lidar), dim=-1))
        return self.actor(features), self.critic(features.detach()).squeeze(-1)

    def deterministic(self, observation):
        mean, _ = self(observation)
        action=self._bounded_action(observation,mean)
        if self.quantized_actions:
            quantized=self.quantize_action(action)
            action=(action+(quantized-action).detach()
                    if self.training else quantized)
        return self.apply_pace_guard(observation,action)

    def apply_pace_guard(self,observation,action):
        """Remove one speed bin only during sensor-observed fast, sharp turns.

        The wheel and steering terms come from the portable observation vector,
        so the same rule is present in PyTorch, ONNX, and the runtime policy.
        """
        if not self.pace_guard or self.corner_slowdown_bins <= 0:return action
        wheel_m_s=observation[:,:2]*12.
        steering_rad=observation[:,4:6]*.45
        measured_speed=(wheel_m_s*torch.cos(steering_rad)).mean(dim=-1)
        intervene=((action[:,1].abs()>=self.corner_steer_threshold)
                   & (measured_speed>=self.corner_speed_threshold_m_s))
        speed=(action[:,0]-intervene.to(action.dtype)
               *self.corner_slowdown_bins/ACTION_QUANTIZATION).clamp(-1.,1.)
        return torch.stack((speed,action[:,1]),dim=-1)

    def quantize_action(self,action):
        if not self.quantized_actions:return action
        return torch.round(action*ACTION_QUANTIZATION)/ACTION_QUANTIZATION

    def _bounded_action(self,observation,raw):
        unit=torch.tanh(raw)
        if not self.residual_actions:return unit
        limits=raw.new_tensor(ACTION_DELTA_LIMITS)
        return (observation[:,-ACTION_SIZE:]+limits*unit).clamp(-1.,1.)

    def sample(self, observation):
        mean, value = self(observation)
        distribution = torch.distributions.Normal(mean, self.log_std.exp())
        raw = distribution.rsample()
        action=self._bounded_action(observation,raw)
        if self.quantized_actions:
            quantized=self.quantize_action(action)
            action=action+(quantized-action).detach()
        action=self.apply_pace_guard(observation,action)
        if self.residual_actions:
            scale=raw.new_tensor(ACTION_DELTA_LIMITS)
            bounded=((action-observation[:,-ACTION_SIZE:])/scale).clamp(-1+1e-6,1-1e-6)
        else:
            scale=raw.new_ones(ACTION_SIZE)
            bounded=action.clamp(-1+1e-6,1-1e-6)
        executed_raw=torch.atanh(bounded)
        log_prob = (distribution.log_prob(executed_raw)
                    -torch.log(scale*(1-bounded.square())+1e-6)).sum(-1)
        return action, log_prob, value

    def evaluate(self, observation, action):
        mean, value = self(observation)
        distribution = torch.distributions.Normal(mean, self.log_std.exp())
        if self.residual_actions:
            limits=action.new_tensor(ACTION_DELTA_LIMITS)
            bounded=((action-observation[:,-ACTION_SIZE:])/limits).clamp(-1+1e-6,1-1e-6)
            scale=limits
        else:
            bounded=action.clamp(-1+1e-6,1-1e-6)
            scale=action.new_ones(ACTION_SIZE)
        raw = torch.atanh(bounded)
        log_prob = (distribution.log_prob(raw)
                    -torch.log(scale*(1-bounded.square())+1e-6)).sum(-1)
        return log_prob, distribution.entropy().sum(-1), value


class ExportedActor(nn.Module):
    def __init__(self, model: ActorCritic):
        super().__init__()
        self.model = model

    def forward(self, observation):
        return self.model.deterministic(observation)


def checkpoint_payload(model: ActorCritic, metadata: dict) -> dict:
    return {
        "schema_version": 1,
        "format": "osracer.ppo.pt.v4",
        "algorithm": "ppo",
        "policy_version": POLICY_VERSION,
        "observation": {
            "size": OBSERVATION_SIZE,
            "fields": ["wheel_vel[4]", "steer_pos[2]",
                       "lidar_range[361] (-1 means invalid)", "lidar_age[1]",
                       "previous_action[2]"],
        },
        "action": {"size": ACTION_SIZE, "fields": ["speed", "steering"],
                   "range": [-1., 1.]},
        "model": {"architecture": "lidar-conv-quantized-action-v3",
                  "hidden_sizes": [256, 128],
                  "residual_actions": model.residual_actions,
                  "quantized_actions": model.quantized_actions,
                  "pace_guard": model.pace_guard,
                  "corner_speed_threshold_m_s": model.corner_speed_threshold_m_s,
                  "corner_steer_threshold": model.corner_steer_threshold,
                  "corner_slowdown_bins": model.corner_slowdown_bins},
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }


def save_checkpoint(path, model: ActorCritic, metadata: dict) -> None:
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(model, metadata), temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    checkpoint_format=payload.get("format")
    if checkpoint_format not in {"osracer.ppo.pt.v2","osracer.ppo.pt.v3",
                                  "osracer.ppo.pt.v4"}:
        raise ValueError(f"unsupported PPO checkpoint format: {payload.get('format')!r}")
    checkpoint_size=int(payload.get("observation",{}).get("size",0))
    if checkpoint_size not in {368,OBSERVATION_SIZE}:
        raise ValueError("checkpoint observation contract does not match this runtime")
    hidden = tuple(payload.get("model", {}).get("hidden_sizes", [256, 128]))
    model_spec=payload.get("model",{})
    model = ActorCritic(
        hidden_sizes=hidden,
        residual_actions=bool(model_spec.get("residual_actions",False)),
        quantized_actions=bool(model_spec.get(
            "quantized_actions",checkpoint_format!="osracer.ppo.pt.v2")),
        pace_guard=bool(model_spec.get("pace_guard",False)),
        corner_speed_threshold_m_s=float(model_spec.get(
            "corner_speed_threshold_m_s",3.5)),
        corner_steer_threshold=float(model_spec.get("corner_steer_threshold",.5)),
        corner_slowdown_bins=float(model_spec.get("corner_slowdown_bins",1.)),
    ).to(device)
    state=dict(payload["state_dict"])
    if checkpoint_size==368:
        # v2 placed [wheel, steer, age] before convolution features. v3 adds
        # previous_action immediately after age; initialize its two new
        # columns to zero so an existing actor remains numerically identical.
        old=state["encoder.0.weight"]
        expanded=torch.zeros((old.shape[0],old.shape[1]+2),dtype=old.dtype,
                             device=old.device)
        expanded[:,:7]=old[:,:7]
        expanded[:,9:]=old[:,7:]
        state["encoder.0.weight"]=expanded
    model.load_state_dict(state)
    model.eval()
    return model, payload


class NeuralPolicy:
    version = POLICY_VERSION

    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.model, self.spec = load_checkpoint(checkpoint, self.device)
        self.max_speed = float(self.spec.get("metadata", {}).get("max_speed_m_s", 10.))
        self.reset()

    def reset(self):
        self.phase = "ppo"
        self.last_action = np.array([-1.,0.],np.float32)

    def action(self, observation):
        vector = torch.from_numpy(encode_observation(
            observation,self.last_action)).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            action = self.model.deterministic(vector)[0].cpu().numpy()
        self.last_action=action.copy()
        return action_to_actuators(action, self.max_speed)


def export_onnx(checkpoint, output=None, device="cpu") -> Path:
    checkpoint = Path(checkpoint)
    output = Path(output or checkpoint.with_suffix(".onnx"))
    model, payload = load_checkpoint(checkpoint, device)
    actor = ExportedActor(model).to(device).eval()
    example = torch.zeros(1, OBSERVATION_SIZE, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor, example, output, input_names=["observation"], output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    import onnx
    graph = onnx.load(str(output));onnx.checker.check_model(graph)
    from onnx.reference import ReferenceEvaluator
    reference = ReferenceEvaluator(graph)
    sample = np.random.default_rng(7).normal(size=(3, OBSERVATION_SIZE)).astype(np.float32)
    with torch.inference_mode():
        expected = actor(torch.from_numpy(sample).to(device)).cpu().numpy()
    actual = reference.run(None, {"observation": sample})[0]
    error = float(np.max(np.abs(expected - actual)))
    if error > 1e-5:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX parity check failed: max abs error {error}")
    return output
