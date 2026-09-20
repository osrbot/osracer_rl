#!/usr/bin/env python3
"""Inspect local racing prerequisites without importing Isaac or changing setup."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC = Path.home()/'rlgpu_ws/isaac-sim-standalone-6.0.1-linux-x86_64'


def inspect_environment(mujoco_only=False):
    checks = []

    def add(name, passed, detail, required=True, **extra):
        item = {'name': name, 'passed': bool(passed), 'required': required, 'detail': detail}
        item.update(extra)
        checks.append(item)

    add('python', sys.version_info >= (3, 11), sys.version.split()[0],
        executable=sys.executable, prefix=sys.prefix, virtual_environment=sys.prefix != sys.base_prefix)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    for distribution, module_name, minimum, exact in (
        ('numpy', 'numpy', (1, 26), None), ('scipy', 'scipy', (1, 11), None),
        ('Pillow', 'PIL', (10,), None), ('mujoco', 'mujoco', None, '3.10.0'),
        ('shapely', 'shapely', (2, 1), None), ('trimesh', 'trimesh', (4,), None),
        ('pycollada', 'collada', (0, 9), None), ('pytest', 'pytest', (8,), None),
    ):
        try:
            module = importlib.import_module(module_name)
            version = importlib.metadata.version(distribution)
            numbers = tuple(int(part) for part in version.split('.')[:len(minimum or ())])
            meets = version == exact if exact else numbers >= minimum
            origin = str(Path(module.__file__).resolve())
            inside = Path(origin).is_relative_to(Path(sys.prefix).resolve())
            add(distribution, meets, version, module_path=origin, installed_under_interpreter_prefix=inside)
        except Exception as exc:
            add(distribution, False, f'{type(exc).__name__}: {exc}')
    try:
        add('editable_project', importlib.util.find_spec('racing') is not None,
            importlib.metadata.version('osracer-racing'))
    except Exception as exc:
        add('editable_project', False, str(exc))
    for command in ('ffmpeg', 'ffprobe'):
        executable = shutil.which(command)
        if executable is None:
            add(command, False, f'{command} is not on PATH; install it with the OS package manager')
            continue
        try:
            run = subprocess.run([executable, '-version'], capture_output=True, text=True, timeout=15)
            add(command, run.returncode == 0, run.stdout.splitlines()[0] if run.stdout else run.stderr[:300], path=executable)
        except Exception as exc:
            add(command, False, str(exc), path=executable)
    gpu = shutil.which('nvidia-smi')
    if gpu:
        try:
            run = subprocess.run([gpu, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=15)
            add('nvidia_gpu', run.returncode == 0 and bool(run.stdout.strip()),
                run.stdout.strip() or run.stderr.strip(), required=not mujoco_only)
        except Exception as exc:
            add('nvidia_gpu', False, str(exc), required=not mujoco_only)
    else:
        add('nvidia_gpu', False, 'nvidia-smi is not on PATH', required=not mujoco_only)
    isaac = Path(os.environ.get('OSRACER_ISAAC_DIR', DEFAULT_ISAAC)).expanduser().resolve()
    version_path = isaac/'VERSION'
    version = version_path.read_text().strip() if version_path.is_file() else 'missing VERSION file'
    launcher = isaac/'python.sh'
    add('isaac_6_0_1', launcher.is_file() and version.startswith('6.0.1'), version,
        required=not mujoco_only, path=str(isaac), launcher=str(launcher),
        note='Isaac uses its bundled Python through scripts/run_isaac.sh, not the project virtual environment.')
    nccl = isaac/'extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib/libnccl.so.2'
    add('isaac_nccl_library', nccl.is_file(), str(nccl), required=not mujoco_only)
    icd = Path('/etc/vulkan/icd.d/nvidia_icd.json')
    add('isaac_vulkan_icd', icd.is_file(), str(icd), required=not mujoco_only)
    model = ROOT/'OSRACER/MuJoCo/osracer_description/robot.xml'
    try:
        xml = ET.parse(model)
        assets = [model.parent/mesh.attrib['file'] for mesh in xml.findall('./asset/mesh')]
        missing = [str(path) for path in assets if not path.is_file() or path.stat().st_size == 0]
        add('mujoco_source_assets', bool(assets) and not missing,
            f'{len(assets)} referenced CAD meshes; {len(missing)} missing', model=str(model), missing=missing)
    except Exception as exc:
        add('mujoco_source_assets', False, str(exc), model=str(model))
    usd = ROOT/'OSRACER/USD/osracer_description/robot.usd'
    visual = list((usd.parent/'meshes/visual').glob('*.STL'))
    collision = list((usd.parent/'meshes/collision').glob('*.STL'))
    usd_ok = usd.is_file() and usd.stat().st_size > 0 and len(visual) >= 10 and len(collision) >= 10
    add('isaac_source_assets', usd_ok, f'{len(visual)} visual and {len(collision)} collision STL meshes',
        required=not mujoco_only, model=str(usd))
    add('track_catalog', (ROOT/'tracks/catalog.json').is_file(), str(ROOT/'tracks/catalog.json'))
    return {'schema_version': 1, 'project_root': str(ROOT),
            'requested_backend': 'mujoco' if mujoco_only else 'mujoco_and_isaac',
            'passed': all(item['passed'] for item in checks if item['required']), 'checks': checks,
            'scope': 'Dependency, executable, driver and asset checks only; native contact/render tests are separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON only')
    parser.add_argument('--output', type=Path, help='Also save the JSON report to this path')
    parser.add_argument('--mujoco-only', action='store_true', help='Report Isaac/GPU checks without requiring them to pass')
    args = parser.parse_args()
    report = inspect_environment(mujoco_only=args.mujoco_only)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded+'\n')
    if args.json:
        print(encoded)
    else:
        print(f"OSRACER environment: {'PASS' if report['passed'] else 'FAIL'}")
        print(f'Python: {sys.executable}')
        for item in report['checks']:
            mark = 'OK' if item['passed'] else ('FAIL' if item['required'] else 'OPTIONAL')
            print(f"[{mark}] {item['name']}: {item['detail']}")
        print('Isaac launcher: bash scripts/run_isaac.sh scripts/run_racing.py ...')
        if not report['passed']:
            print("Project packages: .venv/bin/python -m pip install -e '.[build]'")
            print('Existing Isaac location: export OSRACER_ISAAC_DIR=/path/to/isaac-sim-6.0.1')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
