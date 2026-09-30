"""JSONL metrics with optional TensorBoard event output."""
from __future__ import annotations

from contextlib import suppress
import json
import math
from pathlib import Path
import warnings


def _scalars(prefix, values):
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool):
            yield name, float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            yield name, float(value)
        elif isinstance(value, dict):
            yield from _scalars(name, value)


class MetricLogger:
    def __init__(self, metrics_dir: str | Path, tensorboard_dir: str | Path | None = None,
                 enabled: bool = True, purge_step: int | None = None):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = (self.metrics_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self.tensorboard_dir = Path(tensorboard_dir or self.metrics_dir / "tensorboard").expanduser().resolve()
        self.writer = None
        if enabled:
            try:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(str(self.tensorboard_dir),purge_step=purge_step)
            except ModuleNotFoundError:
                warnings.warn("TensorBoard logging needs the training extra: pip install -e '.[training]'", stacklevel=2)

    def log(self, step: int, prefix: str, values: dict) -> None:
        row = {"step": int(step), "scope": prefix, "values": values}
        self.jsonl.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.jsonl.flush()
        if self.writer:
            for name, value in _scalars(prefix, values):
                self.writer.add_scalar(name, value, step)
            self.writer.flush()

    def close(self) -> None:
        with suppress(Exception):
            if self.writer:
                self.writer.close()
        self.jsonl.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
