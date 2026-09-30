#!/usr/bin/env python3
"""Entry point usable with system Python or the bundled Isaac launcher."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from racing.runtime.run import main
if __name__=='__main__':main()
