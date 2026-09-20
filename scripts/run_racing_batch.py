#!/usr/bin/env python3
"""Run independent Isaac tracks in one Kit process, preserving normal artifacts.

Use scripts/run_isaac.sh. A new physics stage, World and vehicles are built for
each track; only application startup and renderer compilation are shared.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks',nargs='+',required=True)
    parser.add_argument('--record-tracks',nargs='*',default=['bahrain'])
    parser.add_argument('--batch-log',default='output/racing/isaac_batch.json')
    parser.add_argument('--stop-file',type=Path,
                        help='Finish the current track, then stop if this file exists; completed artifacts remain available')
    args,forward=parser.parse_known_args()
    if '--engine' in forward or '--track' in forward or '--record' in forward:
        parser.error('Engine is isaac; use --tracks and --record-tracks for track and video selection')
    from racing import isaac_env,run
    native_class=isaac_env.RaceIsaacEnv
    app=native_class.create_app()
    class SharedApplicationEnv(native_class):
        def __init__(self,*positional,**kwargs):
            super().__init__(*positional,simulation_app=app,**kwargs)
    isaac_env.RaceIsaacEnv=SharedApplicationEnv
    rows=[]
    try:
        for track in args.tracks:
            if args.stop_file is not None and args.stop_file.exists():
                print('RACING_BATCH_PAUSED',json.dumps({'next_track':track,'stop_file':str(args.stop_file)}),flush=True)
                break
            sys.argv=['run_racing.py','--engine','isaac','--track',track,*forward]
            if track in args.record_tracks:sys.argv+=['--record']
            try:
                run.main()
                rows.append({'track':track,'execution':'finished'})
            except Exception:
                rows.append({'track':track,'execution':'failed','error':traceback.format_exc()})
                # A partially constructed world may not have reached env.close.
                from isaacsim.core.api import World
                World.clear_instance()
            run.save(args.batch_log,rows)
            print('RACING_BATCH',json.dumps(rows[-1]),flush=True)
    finally:
        isaac_env.RaceIsaacEnv=native_class
        app.close()


if __name__=='__main__':main()
