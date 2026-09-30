"""The simulator selection layer must stay backend-neutral and package-safe."""

from racing.simulators import (
    SimulatorBackend,
    available_simulators,
    create_simulator,
    get_simulator_spec,
)
from racing.simulators import registry


class DummyBackend(SimulatorBackend):
    backend_name = "dummy"

    def __init__(self, track, **options):
        self.track = track
        self.options = options
        self.num_cars = options["num_cars"]
        self.physics_hz = options["physics_hz"]

    def reset(self, seed=0, starts=None):
        return []

    def step(self, actions):
        return []

    def states(self):
        return []

    def render(self):
        return None

    def close(self):
        return None


def test_packaged_simulator_configs_are_discoverable():
    assert available_simulators() == ("isaac", "mujoco")
    for name in available_simulators():
        spec = get_simulator_spec(name)
        assert spec.name == name
        assert spec.target.startswith(f"racing.simulators.{name}.")
        assert spec.defaults["physics_hz"] == 480
        assert spec.defaults["num_cars"] == 2


def test_factory_merges_config_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.setattr(registry, "load_simulator_class", lambda _name: DummyBackend)
    backend = create_simulator("mujoco", "track", render=True, physics_hz=960)
    assert isinstance(backend, SimulatorBackend)
    assert backend.track == "track"
    assert backend.options == {"num_cars": 2, "render": True, "physics_hz": 960}
    assert backend.metadata()["physics_substeps"] == 16


def test_unknown_simulator_is_rejected_before_import():
    try:
        get_simulator_spec("../isaac")
    except ValueError as exc:
        assert "Unknown simulator backend" in str(exc)
    else:
        raise AssertionError("unsafe simulator name was accepted")
