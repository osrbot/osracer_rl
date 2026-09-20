"""Native two-car PhysX integration smoke, including laser frame readback."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from racing.tracks import Track
from racing.isaac_env import RaceIsaacEnv

env=RaceIsaacEnv(Track('bahrain'),render=True)
try:
    states=env.reset()
    print('RACING_RESET',states,flush=True)
    for i in range(120):
        states=env.step([(np.array([70,70,70,-70]),np.zeros(2)),(np.array([50,50,50,-50]),np.zeros(2))])
    print('RACING_DRIVE',states,flush=True)
    from PIL import Image
    Image.fromarray(env.render()).save('output/racing/isaac/smoke.png')
except Exception:
    import traceback
    traceback.print_exc()
finally:env.close()
