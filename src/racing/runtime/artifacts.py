"""Run-scoped artifact layout for training, evaluation, and recordings."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path

from ..paths import RUNS_ROOT


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @classmethod
    def path(cls, run_id: str, runs_root: str | Path | None = None) -> "RunLayout":
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run id must contain only letters, digits, '.', '_' and '-'")
        return cls(Path(runs_root or RUNS_ROOT).expanduser().resolve() / run_id)

    @classmethod
    def open(cls, run_id: str, runs_root: str | Path | None = None) -> "RunLayout":
        layout = cls.path(run_id,runs_root)
        for path in (
            layout.config,
            layout.checkpoints,
            layout.metrics,
            layout.tensorboard,
            layout.trajectories,
            layout.videos,
            layout.logs,
            layout.reports,
            layout.source,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if not layout.manifest.exists():
            layout.write_json(
                layout.manifest,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        return layout

    @classmethod
    def existing(cls, run_id: str, runs_root: str | Path | None = None) -> "RunLayout":
        layout=cls.path(run_id,runs_root)
        if not layout.root.is_dir():
            raise FileNotFoundError(f'run not found: {layout.root}')
        return layout

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def tensorboard(self) -> Path:
        return self.metrics / "tensorboard"

    @property
    def trajectories(self) -> Path:
        return self.root / "trajectories"

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def source(self) -> Path:
        return self.root / "source"

    def trajectory_dir(self, engine: str, track: str) -> Path:
        path = self.trajectories / engine / track
        path.mkdir(parents=True, exist_ok=True)
        return path

    def video_dir(self, engine: str, track: str) -> Path:
        path = self.videos / engine / track
        path.mkdir(parents=True, exist_ok=True)
        return path

    def evaluation_path(self, engine: str, track: str) -> Path:
        path = self.metrics / "evaluation" / engine
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{track}.json"

    @staticmethod
    def write_json(path: str | Path, value) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(path)
