"""Human-readable console progress for native racing jobs.

Complete episode records belong in run artifacts and TensorBoard.  This module
keeps stdout small enough for a person to follow while a long job is running.
"""
from __future__ import annotations

import math
import sys
import time
from typing import Callable, TextIO


WIDTH = 88


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes):02d}m {seconds:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}h {int(minutes):02d}m {seconds:04.1f}s"


def _flag(value: object) -> str:
    return "yes" if bool(value) else "no"


def _number(value: object, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "--"
    return f"{number:.{digits}f}{suffix}"


def _line(left: str, middle: str = "") -> str:
    inner = WIDTH - 2
    prefix = f"─ {left} " if left else ""
    suffix = f" {middle} ─" if middle else ""
    fill = max(0, inner - len(prefix) - len(suffix))
    return f"├{prefix}{'─' * fill}{suffix}┤"


def _block(title: str, sections: list[tuple[str, list[tuple[str, str]]]],
           stream: TextIO) -> None:
    inner = WIDTH - 2
    heading = f"─ {title} "
    print(f"\n╭{heading}{'─' * max(0, inner - len(heading))}╮", file=stream)
    for index, (name, rows) in enumerate(sections):
        if index:
            print(_line(name), file=stream)
        for label, value in rows:
            content = f"  {label:<22} {value}"
            print(f"│{content:<{inner}}│", file=stream)
    print(f"╰{'─' * inner}╯", file=stream, flush=True)


def _engine(name: str) -> str:
    names = {"mujoco": "MuJoCo", "isaac": "Isaac Sim",
             "isaac + mujoco": "Isaac Sim + MuJoCo"}
    return names.get(name.lower(), name)


class TrainingProgress:
    """Render consistent CEM progress for either native engine or a joint run."""

    def __init__(self, engine: str, track: str, generations: int, population: int,
                 run_id: str | None = None,
                 stream: TextIO | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.engine = _engine(engine)
        self.track = track
        self.run_id = run_id
        self.generations = generations
        self.population = population
        self.total_iterations = generations * population
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock
        self.started = clock()
        self.total_steps = 0

    def _common(self, generation: int, candidate: int, simulated_steps: int,
                collection_s: float, iteration_s: float, reused: bool = False):
        completed = generation * self.population + candidate + 1
        self.total_steps += max(0, int(simulated_steps))
        elapsed = max(0.0, self.clock() - self.started)
        eta = elapsed / completed * max(0, self.total_iterations - completed)
        if reused:
            computation = f"cached trial (audit: {_duration(collection_s)})"
        else:
            rate = simulated_steps / collection_s if collection_s > 0 else 0.0
            computation = f"{rate:.0f} steps/s (collection: {collection_s:.3f}s)"
        percentage = 100.0 * completed / self.total_iterations
        context = []
        if self.run_id:
            context.append(("Experiment", self.run_id))
        context.extend([
            ("Simulator / task", f"{self.engine} / racing/{self.track}"),
            ("Search", f"generation {generation + 1}/{self.generations}  ·  "
             f"candidate {candidate + 1}/{self.population}  ·  {percentage:.1f}%"),
            ("Throughput", computation),
        ])
        footer = [
            ("Total timesteps", f"{self.total_steps}"),
            ("Time", f"iteration {_duration(iteration_s)}  ·  total {_duration(elapsed)}  ·  "
             f"ETA {_duration(eta)}"),
        ]
        return completed, context, footer

    def candidate(self, generation: int, candidate: int, result: dict,
                  simulated_steps: int, collection_s: float, iteration_s: float,
                  best_score: float) -> None:
        completed, context, footer = self._common(
            generation, candidate, simulated_steps, collection_s, iteration_s)
        lap_fraction = 100.0 * float(result.get("lap_fraction", 0.0))
        failure = result.get("failure") or "none"
        performance = [
            ("Score", f"{_number(result.get('score'))}  ·  best {_number(best_score)}"),
            ("Episode", f"valid {_flag(result.get('valid_lap'))}  ·  "
             f"completed {_flag(result.get('completed'))}  ·  failure {failure}"),
            ("Lap time", _number(result.get("lap_time_s"), 3, "s")),
            ("Progress", f"{_number(result.get('progress_m'), 2, 'm')} ({lap_fraction:.1f}%)"),
            ("Speed", f"mean {_number(result.get('mean_speed_m_s'), 2)}  ·  "
             f"peak {_number(result.get('peak_speed_m_s'), 2)} m/s"),
            ("Overtakes", str(int(result.get("effective_overtakes", 0)))),
        ]
        safety = [
            ("Drift", f"max slip {_number(result.get('max_rear_slip_deg'), 2)}°  ·  "
             f"longest {_number(result.get('max_continuous_drift_duration_s'), 3, 's')}"),
            ("Safety", f"collision {int(result.get('collision_steps', 0))}  ·  "
             f"offroad {int(result.get('offroad_steps', 0))}"),
        ]
        _block(f"Training iteration {completed}/{self.total_iterations}",
               [("", context), ("Performance", performance), ("Safety", safety), ("Timing", footer)],
               self.stream)

    def joint_candidate(self, generation: int, candidate: int, item: dict,
                        collection_s: float, iteration_s: float, best_score: float,
                        reused: bool = False) -> None:
        episodes = item.get("episodes", [])
        simulated_steps = sum(round(float(row.get("duration_s", 0.0)) * 60) for row in episodes)
        completed, context, footer = self._common(
            generation, candidate, simulated_steps, collection_s, iteration_s, reused)
        mean_progress = (sum(float(row.get("progress_m", 0.0)) for row in episodes) / len(episodes)
                         if episodes else 0.0)
        performance = [
            ("Score", f"{_number(item.get('score'))}  ·  best {_number(best_score)}"),
            ("Episodes", f"{item.get('episodes_evaluated', len(episodes))}/"
             f"{item.get('episodes_expected', len(episodes))}"),
            ("Qualification", f"clean {_flag(item.get('all_segments_clean'))}  ·  "
             f"forward {_flag(item.get('all_forward_progress'))}  ·  "
             f"overtake {_flag(item.get('all_overtakes'))}"),
            ("Mean progress", _number(mean_progress, 2, "m")),
            ("Drift", f"qualified {_flag(item.get('all_continuous_drift'))}  ·  minimum "
             f"{_number(item.get('minimum_continuous_drift_s'), 3, 's')}"),
        ]
        _block(f"Training iteration {completed}/{self.total_iterations}",
               [("", context), ("Performance", performance), ("Timing", footer)], self.stream)

    def generation(self, generation: int, scores, success_rate: float,
                   success_label: str = "Valid lap rate") -> None:
        values = [float(value) for value in scores]
        elapsed = max(0.0, self.clock() - self.started)
        rows = [
            ("Score mean / std",
             f"{sum(values) / len(values):.3f} / "
             f"{(sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** .5:.3f}"),
            ("Generation best", f"{max(values):.3f}"),
            (success_label, f"{100.0 * success_rate:.1f}%"),
            ("Total timesteps", f"{self.total_steps}"),
            ("Total time", _duration(elapsed)),
        ]
        _block(f"Generation {generation + 1}/{self.generations} complete",
               [("Summary", rows)], self.stream)


class PPOProgress:
    """IsaacLab-style PPO update report using metrics produced by the optimizer."""

    def __init__(self, run_id: str, engine: str, track: str, iterations: int,
                 stream: TextIO | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.run_id, self.engine, self.track = run_id, _engine(engine), track
        self.iterations = int(iterations)
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock
        self.started = clock()

    def iteration(self, index: int, metrics: dict) -> None:
        completed = index + 1
        elapsed = max(0., self.clock() - self.started)
        eta = elapsed / completed * max(0, self.iterations - completed)
        context = [
            ("Experiment", self.run_id),
            ("Simulator / task", f"{self.engine} / racing/{self.track}"),
            ("Computation", f"{metrics['steps_per_second']:.0f} steps/s  ·  "
             f"collection {metrics['collection_seconds']:.3f}s  ·  "
             f"learning {metrics['learning_seconds']:.3f}s"),
        ]
        optimization = [
            ("Value function loss", f"{metrics['value_loss']:.4f}"),
            ("Surrogate loss", f"{metrics['surrogate_loss']:.4f}"),
            ("Entropy", f"{metrics['entropy']:.4f}"),
            ("Imitation loss", f"{metrics['imitation_loss']:.4f}"),
            ("Mean action noise std", f"{metrics['action_noise_std']:.3f}"),
        ]
        episodes = [
            ("Mean total reward", f"{metrics['mean_reward']:.3f}"),
            ("Mean episode length", f"{metrics['mean_episode_length']:.1f}"),
            ("Mean speed / progress", f"{metrics['mean_speed_m_s']:.2f} / "
             f"{metrics['mean_progress_speed_m_s']:.2f} m/s"),
            ("Reward / progress", f"{metrics['reward_progress']:.4f}"),
            ("Reward / pace", f"{metrics['reward_pace']:.4f}"),
            ("Reward / alive", f"{metrics['reward_alive']:.4f}"),
            ("Reward / track", f"{metrics['reward_track']:.4f}"),
            ("Reward / action rate", f"{metrics['reward_action_rate']:.4f}"),
            ("Reward / pass + finish", f"pass {metrics['reward_overtake']:.4f}  ·  "
             f"finish {metrics['reward_completion']:.4f}"),
            ("Reward / fail", f"crash {metrics['reward_collision']:.4f}  ·  "
             f"offroad {metrics['reward_offroad']:.4f}"),
            ("Reward / stalled", f"{metrics['reward_stalled']:.4f}"),
            ("Reward / failure horizon", f"{metrics['reward_failure_horizon']:.4f}"),
            ("Termination / normal", f"done {metrics['termination_completed']:.1f}%  ·  "
             f"timeout {metrics['termination_timeout']:.1f}%"),
            ("Termination / failure", f"collision {metrics['termination_collision']:.1f}%  ·  "
             f"offroad {metrics['termination_offroad']:.1f}%  ·  "
             f"stalled {metrics['termination_stalled']:.1f}%"),
        ]
        timing = [
            ("Total timesteps", str(int(metrics["total_steps"]))),
            ("Iteration time", _duration(metrics["iteration_seconds"])),
            ("Total time", _duration(elapsed)),
            ("ETA", _duration(eta)),
        ]
        _block(f"Learning iteration {completed}/{self.iterations}",
               [("", context), ("Optimization", optimization),
                ("Episode", episodes), ("Timing", timing)], self.stream)


def print_evaluation(result: dict, episode: int, total: int, wall_seconds: float,
                     stream: TextIO | None = None) -> None:
    """Print a compact final evaluation without dumping privileged state."""
    stream = stream if stream is not None else sys.stdout
    steps = round(float(result.get("duration_s", 0.0)) * 60)
    rate = steps / wall_seconds if wall_seconds > 0 else 0.0
    context = [
        ("Simulator / task", f"{_engine(str(result.get('engine', 'unknown')))} / "
         f"racing/{result.get('track', 'unknown')}"),
        ("Throughput", f"{rate:.0f} steps/s  ·  evaluation {wall_seconds:.3f}s"),
    ]
    performance = [
        ("Episode", f"valid {_flag(result.get('valid_lap'))}  ·  "
         f"completed {_flag(result.get('completed'))}  ·  failure {result.get('failure') or 'none'}"),
        ("Score", _number(result.get("score"))),
        ("Lap time", _number(result.get("lap_time_s"), 3, "s")),
        ("Progress", f"{_number(result.get('progress_m'), 2, 'm')} "
         f"({100.0 * float(result.get('lap_fraction', 0.0)):.1f}%)"),
        ("Speed", f"mean {_number(result.get('mean_speed_m_s'), 2)}  ·  "
         f"peak {_number(result.get('peak_speed_m_s'), 2)} m/s"),
        ("Overtakes", str(int(result.get("effective_overtakes", 0)))),
    ]
    safety = [
        ("Drift", f"max slip {_number(result.get('max_rear_slip_deg'), 2)}°  ·  "
         f"longest {_number(result.get('max_continuous_drift_duration_s'), 3, 's')}"),
        ("Safety", f"collision {int(result.get('collision_steps', 0))}  ·  "
         f"offroad {int(result.get('offroad_steps', 0))}"),
    ]
    _block(f"Evaluation episode {episode}/{total}",
           [("", context), ("Performance", performance), ("Safety", safety)], stream)
