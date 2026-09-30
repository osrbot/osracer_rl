"""Common contract shared by every native racing simulator backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


SimulatorAction = tuple[Sequence[float], Sequence[float]]
SimulatorState = dict[str, Any]


class SimulatorBackend(ABC):
    """Minimal lifecycle and timing contract required by the racing runtime."""

    backend_name: str
    control_hz: int = 60
    physics_hz: int
    track: Any
    num_cars: int

    @classmethod
    def validate_physics_hz(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"physics_hz must be an integer multiple of the {cls.control_hz} Hz control rate"
            )
        if value < cls.control_hz or value % cls.control_hz:
            raise ValueError(
                f"physics_hz must be an integer multiple of the {cls.control_hz} Hz control rate"
            )
        return value

    @property
    def physics_substeps(self) -> int:
        return self.physics_hz // self.control_hz

    @abstractmethod
    def reset(
        self, seed: int = 0, starts: Sequence[tuple[float, float]] | None = None
    ) -> list[SimulatorState]:
        """Reset all vehicles and return one state mapping per vehicle."""

    @abstractmethod
    def step(self, actions: Sequence[SimulatorAction]) -> list[SimulatorState]:
        """Apply one control frame and return the resulting vehicle states."""

    @abstractmethod
    def states(self) -> list[SimulatorState]:
        """Read the current vehicle states without advancing physics."""

    @abstractmethod
    def render(self) -> Any:
        """Return the current native RGB frame."""

    @abstractmethod
    def close(self) -> None:
        """Release native simulator resources."""

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "version": str(getattr(self, "version", "unknown")),
            "control_hz": self.control_hz,
            "physics_hz": self.physics_hz,
            "physics_substeps": self.physics_substeps,
        }

    def __enter__(self) -> "SimulatorBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
