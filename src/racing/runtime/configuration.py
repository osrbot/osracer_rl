"""Small composable configuration layer inspired by ASAP/HumanoidVerse Hydra CLI.

The public command accepts values such as ``+simulator=mujoco`` and
``+task=racing/bahrain`` while the runtime keeps regular typed attributes.
Hydra is intentionally not required for this small training stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import tomllib


CONFIG_PACKAGE = "racing.config.training"


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    description: str
    algorithm: str
    iterations: int
    steps_per_iteration: int
    learning_epochs: int
    mini_batches: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_ratio: float
    entropy_coefficient: float
    value_coefficient: float
    max_gradient_norm: float
    checkpoint_interval: int
    imitation_steps: int
    imitation_epochs: int
    refinement_cycles: int
    refinement_epochs: int
    generations: int
    population: int
    episode_length_s: float
    evaluation_episodes: int
    task: str
    opponent_speed_m_s: float
    opponent_gap_m: float
    reward_time_cost: float
    reward_pace_weight: float
    reward_failure_horizon_scale: float
    reward_corner_risk_weight: float
    pace_guard: bool
    corner_speed_threshold_m_s: float
    corner_steer_threshold: float
    corner_slowdown_bins: float
    seed: int
    tensorboard: bool

    @property
    def track(self) -> str:
        return task_track(self.task)


def task_track(value: str) -> str:
    value = str(value).strip().strip("/")
    if not value:
        raise ValueError("task cannot be empty")
    parts = value.split("/")
    if len(parts) > 2 or (len(parts) == 2 and parts[0] != "racing"):
        raise ValueError("task must be a track name or racing/<track>")
    return parts[-1]


def available_training_profiles() -> tuple[str, ...]:
    root = resources.files(CONFIG_PACKAGE)
    return tuple(sorted(path.name.removesuffix(".toml") for path in root.iterdir()
                        if path.name.endswith(".toml")))


def load_training_profile(name: str = "default") -> TrainingProfile:
    if not name or Path(name).name != name:
        raise ValueError(f"unknown training profile: {name!r}")
    source = resources.files(CONFIG_PACKAGE).joinpath(f"{name}.toml")
    if not source.is_file():
        choices = ", ".join(available_training_profiles())
        raise ValueError(f"unknown training profile {name!r}; choose one of: {choices}")
    data = tomllib.loads(source.read_text(encoding="utf-8"))
    algorithm = data["algorithm"]
    environment = data["environment"]
    task = data["task"]
    reward = data.get("reward", {})
    policy = data.get("policy", {})
    reproducibility = data["reproducibility"]
    logging = data["logging"]
    profile = TrainingProfile(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        algorithm=str(algorithm.get("name", "ppo")),
        iterations=int(algorithm["iterations"]),
        steps_per_iteration=int(algorithm["steps_per_iteration"]),
        learning_epochs=int(algorithm["learning_epochs"]),
        mini_batches=int(algorithm["mini_batches"]),
        learning_rate=float(algorithm["learning_rate"]),
        gamma=float(algorithm["gamma"]),
        gae_lambda=float(algorithm["gae_lambda"]),
        clip_ratio=float(algorithm["clip_ratio"]),
        entropy_coefficient=float(algorithm["entropy_coefficient"]),
        value_coefficient=float(algorithm["value_coefficient"]),
        max_gradient_norm=float(algorithm["max_gradient_norm"]),
        checkpoint_interval=int(algorithm["checkpoint_interval"]),
        imitation_steps=int(algorithm.get("imitation_steps", 0)),
        imitation_epochs=int(algorithm.get("imitation_epochs", 0)),
        refinement_cycles=int(algorithm.get("refinement_cycles", 0)),
        refinement_epochs=int(algorithm.get("refinement_epochs", 0)),
        generations=int(algorithm["generations"]),
        population=int(algorithm["population"]),
        episode_length_s=float(environment["episode_length_s"]),
        evaluation_episodes=int(environment["evaluation_episodes"]),
        task=str(task["name"]),
        opponent_speed_m_s=float(task["opponent_speed_m_s"]),
        opponent_gap_m=float(task["opponent_gap_m"]),
        reward_time_cost=float(reward.get("time_cost", .01)),
        reward_pace_weight=float(reward.get("pace_weight", 0.)),
        reward_failure_horizon_scale=float(reward.get("failure_horizon_scale", 0.)),
        reward_corner_risk_weight=float(reward.get("corner_risk_weight", 0.)),
        pace_guard=bool(policy.get("pace_guard",False)),
        corner_speed_threshold_m_s=float(policy.get("corner_speed_threshold_m_s",3.5)),
        corner_steer_threshold=float(policy.get("corner_steer_threshold",.5)),
        corner_slowdown_bins=float(policy.get("corner_slowdown_bins",1.)),
        seed=int(reproducibility["seed"]),
        tensorboard=bool(logging["tensorboard"]),
    )
    if (profile.algorithm not in {"ppo", "cem"} or profile.iterations < 1
            or profile.steps_per_iteration < 1 or profile.learning_epochs < 1
            or profile.mini_batches < 1 or profile.generations < 1
            or profile.population < 1 or profile.episode_length_s <= 0
            or profile.evaluation_episodes < 1
            or profile.reward_time_cost < 0 or profile.reward_pace_weight < 0
            or profile.reward_failure_horizon_scale < 0
            or profile.reward_corner_risk_weight < 0
            or profile.corner_speed_threshold_m_s < 0
            or not 0 <= profile.corner_steer_threshold <= 1
            or profile.corner_slowdown_bins < 0
            or profile.imitation_steps < 0 or profile.imitation_epochs < 0
            or profile.refinement_cycles < 0 or profile.refinement_epochs < 0):
        raise ValueError(f"invalid training profile: {source}")
    task_track(profile.task)
    return profile


def composed_values(tokens: list[str]) -> dict[str, str]:
    """Parse Hydra-style CLI assignments without interpreting arbitrary Python."""
    values: dict[str, str] = {}
    for token in tokens:
        normalized = token[1:] if token.startswith("+") else token
        if "=" not in normalized:
            raise ValueError(f"override must use key=value syntax: {token!r}")
        key, value = normalized.split("=", 1)
        key = key.strip();value = value.strip()
        if not key or not value:
            raise ValueError(f"override must use a non-empty key and value: {token!r}")
        if key in values:
            raise ValueError(f"override was provided more than once: {key}")
        values[key] = value
    return values


def selected_training_profile(tokens: list[str], default: str = "default") -> str:
    return composed_values(tokens).get("train", default)


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, received {value!r}")


def apply_run_overrides(args, tokens: list[str], simulators: tuple[str, ...]) -> None:
    values = composed_values(tokens)
    allowed = {
        "train", "algorithm", "simulator", "task", "experiment_name", "checkpoint",
        "seed", "train.generations", "train.population", "env.episode_length_s",
        "eval.episodes", "task.opponent_speed_m_s", "task.opponent_gap_m",
        "logger.tensorboard", "logger.tensorboard_dir",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown override(s): {', '.join(unknown)}")
    if values.get("algorithm", "cem") != "cem":
        raise ValueError("only algorithm=cem is currently available")
    if "simulator" in values:
        if values["simulator"] not in simulators:
            raise ValueError(f"unknown simulator {values['simulator']!r}; choose one of: {', '.join(simulators)}")
        args.engine = values["simulator"]
    if "task" in values: args.track = task_track(values["task"])
    if "experiment_name" in values: args.run_id = values["experiment_name"]
    if "checkpoint" in values: args.checkpoint = values["checkpoint"]
    if "seed" in values: args.seed = int(values["seed"])
    if "train.generations" in values: args.generations = int(values["train.generations"])
    if "train.population" in values: args.population = int(values["train.population"])
    if "env.episode_length_s" in values: args.seconds = float(values["env.episode_length_s"])
    if "eval.episodes" in values: args.episodes = int(values["eval.episodes"])
    if "task.opponent_speed_m_s" in values: args.opponent_speed = float(values["task.opponent_speed_m_s"])
    if "task.opponent_gap_m" in values: args.opponent_gap = float(values["task.opponent_gap_m"])
    if "logger.tensorboard" in values: args.no_tensorboard = not _boolean(values["logger.tensorboard"])
    if "logger.tensorboard_dir" in values: args.tensorboard_dir = values["logger.tensorboard_dir"]


def apply_play_overrides(args, tokens: list[str], simulators: tuple[str, ...]) -> None:
    values = composed_values(tokens)
    allowed = {"simulator", "task", "experiment_name", "checkpoint", "seed",
               "env.episode_length_s", "task.opponent_speed_m_s", "task.opponent_gap_m"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown override(s): {', '.join(unknown)}")
    if "simulator" in values:
        if values["simulator"] not in simulators:
            raise ValueError(f"unknown simulator {values['simulator']!r}; choose one of: {', '.join(simulators)}")
        args.engine = values["simulator"]
    if "task" in values: args.track = task_track(values["task"])
    if "experiment_name" in values: args.run_id = values["experiment_name"]
    if "checkpoint" in values: args.checkpoint = values["checkpoint"]
    if "seed" in values: args.seed = int(values["seed"])
    if "env.episode_length_s" in values: args.seconds = float(values["env.episode_length_s"])
    if "task.opponent_speed_m_s" in values: args.opponent_speed = float(values["task.opponent_speed_m_s"])
    if "task.opponent_gap_m" in values: args.opponent_gap = float(values["task.opponent_gap_m"])


def apply_joint_overrides(args, tokens: list[str]) -> None:
    values = composed_values(tokens)
    allowed = {"train", "algorithm", "task", "experiment_name", "checkpoint", "seed",
               "train.generations", "train.population", "env.episode_length_s",
               "eval.seeds", "logger.tensorboard", "logger.tensorboard_dir"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown override(s): {', '.join(unknown)}")
    if values.get("algorithm", "cem") != "cem":
        raise ValueError("only algorithm=cem is currently available")
    if "task" in values: args.track = task_track(values["task"])
    if "experiment_name" in values: args.run_id = values["experiment_name"]
    if "checkpoint" in values: args.checkpoint = values["checkpoint"]
    if "seed" in values: args.seed = int(values["seed"])
    if "train.generations" in values: args.generations = int(values["train.generations"])
    if "train.population" in values: args.population = int(values["train.population"])
    if "env.episode_length_s" in values: args.seconds = float(values["env.episode_length_s"])
    if "eval.seeds" in values:
        args.seeds = [int(seed.strip()) for seed in values["eval.seeds"].split(",") if seed.strip()]
    if "logger.tensorboard" in values: args.no_tensorboard = not _boolean(values["logger.tensorboard"])
    if "logger.tensorboard_dir" in values: args.tensorboard_dir = values["logger.tensorboard_dir"]


def apply_ppo_overrides(args, tokens: list[str], simulators: tuple[str, ...]) -> None:
    values = composed_values(tokens)
    allowed = {
        "train", "algorithm", "simulator", "task", "experiment_name", "checkpoint", "seed", "device",
        "train.iterations", "train.steps_per_iteration", "train.learning_epochs",
        "train.mini_batches", "train.learning_rate", "train.checkpoint_interval",
        "train.imitation_steps", "train.imitation_epochs",
        "train.refinement_cycles", "train.refinement_epochs",
        "env.episode_length_s", "env.evaluation_episodes",
        "task.opponent_speed_m_s", "task.opponent_gap_m",
        "reward.time_cost", "reward.pace_weight", "reward.failure_horizon_scale",
        "reward.corner_risk_weight", "policy.pace_guard",
        "policy.corner_speed_threshold_m_s", "policy.corner_steer_threshold",
        "policy.corner_slowdown_bins",
        "logger.tensorboard", "logger.tensorboard_dir",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown override(s): {', '.join(unknown)}")
    if values.get("algorithm", "ppo") != "ppo":
        raise ValueError("the PPO entry point requires algorithm=ppo")
    if "simulator" in values:
        if values["simulator"] not in simulators:
            raise ValueError(f"unknown simulator {values['simulator']!r}; choose one of: {', '.join(simulators)}")
        args.engine = values["simulator"]
    if "task" in values: args.track = task_track(values["task"])
    if "experiment_name" in values: args.run_id = values["experiment_name"]
    if "checkpoint" in values: args.checkpoint = values["checkpoint"]
    if "seed" in values: args.seed = int(values["seed"])
    if "device" in values: args.device = values["device"]
    if "train.iterations" in values: args.iterations = int(values["train.iterations"])
    if "train.steps_per_iteration" in values: args.steps_per_iteration = int(values["train.steps_per_iteration"])
    if "train.learning_epochs" in values: args.learning_epochs = int(values["train.learning_epochs"])
    if "train.mini_batches" in values: args.mini_batches = int(values["train.mini_batches"])
    if "train.learning_rate" in values: args.learning_rate = float(values["train.learning_rate"])
    if "train.checkpoint_interval" in values: args.checkpoint_interval = int(values["train.checkpoint_interval"])
    if "train.imitation_steps" in values: args.imitation_steps = int(values["train.imitation_steps"])
    if "train.imitation_epochs" in values: args.imitation_epochs = int(values["train.imitation_epochs"])
    if "train.refinement_cycles" in values: args.refinement_cycles = int(values["train.refinement_cycles"])
    if "train.refinement_epochs" in values: args.refinement_epochs = int(values["train.refinement_epochs"])
    if "env.episode_length_s" in values: args.seconds = float(values["env.episode_length_s"])
    if "env.evaluation_episodes" in values:
        args.evaluation_episodes = int(values["env.evaluation_episodes"])
    if "task.opponent_speed_m_s" in values: args.opponent_speed = float(values["task.opponent_speed_m_s"])
    if "task.opponent_gap_m" in values: args.opponent_gap = float(values["task.opponent_gap_m"])
    if "reward.time_cost" in values: args.reward_time_cost = float(values["reward.time_cost"])
    if "reward.pace_weight" in values: args.reward_pace_weight = float(values["reward.pace_weight"])
    if "reward.failure_horizon_scale" in values:
        args.reward_failure_horizon_scale = float(values["reward.failure_horizon_scale"])
    if "reward.corner_risk_weight" in values:
        args.reward_corner_risk_weight = float(values["reward.corner_risk_weight"])
    if "policy.pace_guard" in values: args.pace_guard = _boolean(values["policy.pace_guard"])
    if "policy.corner_speed_threshold_m_s" in values:
        args.corner_speed_threshold_m_s = float(values["policy.corner_speed_threshold_m_s"])
    if "policy.corner_steer_threshold" in values:
        args.corner_steer_threshold = float(values["policy.corner_steer_threshold"])
    if "policy.corner_slowdown_bins" in values:
        args.corner_slowdown_bins = float(values["policy.corner_slowdown_bins"])
    if "logger.tensorboard" in values: args.no_tensorboard = not _boolean(values["logger.tensorboard"])
    if "logger.tensorboard_dir" in values: args.tensorboard_dir = values["logger.tensorboard_dir"]
