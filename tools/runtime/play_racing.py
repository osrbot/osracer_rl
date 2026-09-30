#!/usr/bin/env python3
"""Project-source wrapper for racing.runtime.play, including Isaac Python."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]

from racing.runtime.play import main

if __name__ == '__main__':
    main()
