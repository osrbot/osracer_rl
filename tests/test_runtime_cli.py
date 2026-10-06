"""Exercise the configured console entry point with the real CPU simulator."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY_SCRIPT = """
import importlib
from pathlib import Path
import sys
import tomllib

project = tomllib.loads(Path('pyproject.toml').read_text())
module, name = project['project']['scripts']['osracer'].split(':')
sys.exit(getattr(importlib.import_module(module), name)())
"""


@pytest.mark.parametrize('engine,expected_status', [('mujoco', 0), ('invalid', 2)])
def test_console_episode_exit_status(tmp_path, engine, expected_status):
    process = subprocess.run(
        [sys.executable, '-c', ENTRY_SCRIPT, '--simulator', engine,
         '--task', 'bahrain', '--seconds', '0.05', '--episodes', '1',
         '--tag', 'cli-smoke', '--runs-root', str(tmp_path)],
        cwd=ROOT, env={**os.environ, 'MUJOCO_GL': 'disable',
                       'PYTHONPATH': str(ROOT / 'src')},
        capture_output=True, text=True, timeout=60,
    )
    assert process.returncode == expected_status, process.stderr
    if expected_status == 0:
        line = next(line for line in process.stdout.splitlines()
                    if line.startswith('OSRACER_ARTIFACTS '))
        artifacts = json.loads(line.split(' ', 1)[1])
        assert Path(artifacts['run_dir']).is_dir()
        assert list(Path(artifacts['run_dir']).glob('trajectories/mujoco/bahrain/*.json'))
        assert "{'run_id':" not in process.stderr
    else:
        assert 'invalid choice' in process.stderr
