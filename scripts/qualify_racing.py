#!/usr/bin/env python3
"""Use the adjacent project source even with an isolated/shared Python venv."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing.qualify import main

if __name__ == '__main__':
    main()
