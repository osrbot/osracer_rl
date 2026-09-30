"""Stable project paths shared by package modules and command-line tools."""
from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SOURCE_ROOT.parent
ASSETS_ROOT = REPO_ROOT / "assets"
RUNS_ROOT = Path(os.environ.get("OSRACER_RUNS_DIR", REPO_ROOT / "runs")).expanduser().resolve()
LEGACY_OUTPUT_ROOT = REPO_ROOT / "output" / "racing"


def source_files():
    """Return package source files keyed by their stable installed names."""
    return {
        str(path.relative_to(SOURCE_ROOT)): path
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    }
