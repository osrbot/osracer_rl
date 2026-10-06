#!/usr/bin/env bash
# Project-local setup only. Isaac, drivers and system packages are never installed.
set -euo pipefail
racing_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
racing_python="${RACING_PYTHON:-python3}"
racing_run_tests=false
racing_mujoco_only=false
for racing_arg in "$@"; do
  case "$racing_arg" in
    --test) racing_run_tests=true ;;
    --mujoco-only) racing_mujoco_only=true ;;
    -h|--help)
      echo "Usage: bash tools/environment/setup_racing.sh [--test] [--mujoco-only]"
      echo "Creates .venv and installs -e '.[build,training,ppo]'. RACING_PYTHON selects Python >=3.11."
      echo "OSRACER_ISAAC_DIR selects an existing Isaac Sim 6.0.1 installation."
      exit 0 ;;
    *) echo "Unknown option: $racing_arg" >&2; exit 2 ;;
  esac
done
cd -- "$racing_root"
"$racing_python" -I -c 'import sys; assert sys.version_info >= (3, 11), "Python >=3.11 is required"'
if [[ ! -f .venv/pyvenv.cfg ]]; then
  if [[ -e .venv ]]; then
    echo ".venv exists but is not a Python virtual environment; move it before setup." >&2
    exit 1
  fi
  "$racing_python" -I -m venv .venv
fi
racing_venv_python="$racing_root/.venv/bin/python"
if [[ ! -x "$racing_venv_python" ]]; then
  echo ".venv/bin/python is missing; repair the existing project environment." >&2
  exit 1
fi
"$racing_venv_python" -I -m pip --disable-pip-version-check install -e '.[build,training,ppo]'
export MUJOCO_GL="${MUJOCO_GL:-egl}"
racing_check_args=(--output "$racing_root/runs/_environment/reports/environment_check.json")
if "$racing_mujoco_only"; then racing_check_args+=(--mujoco-only); fi
"$racing_venv_python" -I tools/environment/check_racing_environment.py "${racing_check_args[@]}"
if "$racing_run_tests"; then
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$racing_venv_python" -I -m pytest -q tests/test_mujoco_racing.py tests/test_racing_contracts.py
fi
echo "Ready: source .venv/bin/activate"
echo "TensorBoard: .venv/bin/tensorboard --logdir runs --port 6006"
echo "Isaac continues to use: bash tools/runtime/run_isaac.sh tools/runtime/run_racing.py ..."
