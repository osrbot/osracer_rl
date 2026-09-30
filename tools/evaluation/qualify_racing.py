#!/usr/bin/env python3
"""Use the adjacent project source even with an isolated/shared Python venv."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]

from racing.evaluation.qualify import main

if __name__ == '__main__':
    main()
