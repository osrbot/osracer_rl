"""Configuration-driven simulator discovery and construction."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, resources
from pathlib import Path
import tomllib
from typing import Any

from .base import SimulatorBackend


CONFIG_PACKAGE = "racing.config.simulators"


@dataclass(frozen=True)
class SimulatorSpec:
    name: str
    target: str
    description: str
    defaults: dict[str, Any]

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source: str) -> "SimulatorSpec":
        name = str(data.get("name", "")).strip()
        target = str(data.get("target", "")).strip()
        if not name or not target or ":" not in target:
            raise ValueError(f"Invalid simulator configuration: {source}")
        defaults = data.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ValueError(f"Simulator defaults must be a table: {source}")
        return cls(
            name=name,
            target=target,
            description=str(data.get("description", "")).strip(),
            defaults=dict(defaults),
        )


def available_simulators() -> tuple[str, ...]:
    root = resources.files(CONFIG_PACKAGE)
    return tuple(sorted(path.name.removesuffix(".toml") for path in root.iterdir()
                        if path.name.endswith(".toml")))


def get_simulator_spec(name: str) -> SimulatorSpec:
    if not name or Path(name).name != name:
        raise ValueError(f"Unknown simulator backend: {name!r}")
    config = resources.files(CONFIG_PACKAGE).joinpath(f"{name}.toml")
    if not config.is_file():
        choices = ", ".join(available_simulators())
        raise ValueError(f"Unknown simulator backend {name!r}; choose one of: {choices}")
    return SimulatorSpec.from_mapping(tomllib.loads(config.read_text(encoding="utf-8")), str(config))


def load_simulator_class(name: str) -> type[SimulatorBackend]:
    spec = get_simulator_spec(name)
    module_name, class_name = spec.target.split(":", 1)
    implementation = getattr(import_module(module_name), class_name)
    if not isinstance(implementation, type) or not issubclass(implementation, SimulatorBackend):
        raise TypeError(f"Configured target {spec.target!r} does not implement SimulatorBackend")
    return implementation


def create_simulator(name: str, track: Any, **overrides: Any) -> SimulatorBackend:
    spec = get_simulator_spec(name)
    options = {**spec.defaults, **overrides}
    return load_simulator_class(name)(track, **options)
