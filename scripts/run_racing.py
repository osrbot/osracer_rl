#!/usr/bin/env python3
"""Entry point usable with system Python or the bundled Isaac launcher."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from racing.run import main
if __name__=='__main__':main()
