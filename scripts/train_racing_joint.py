#!/usr/bin/env python3
"""Isaac launcher entry point for native joint-engine CEM training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from racing.train_joint import main
if __name__=='__main__':main()
